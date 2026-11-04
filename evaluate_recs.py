"""Judge recommenders against what teams actually did next season.

There is no ground truth for what a team *should* have done, so this scores
the two claims a recommendation makes against things that are observable.
Every list is built from season-t information only and judged on t+1:

  realism   Would the team make this move? For each actual acquisition (a
            veteran on team T in t+1 who was not on T in t), where did he
            rank in T's season-t list, among the same candidate pool for
            every system? Reported as the mean percentile (0 = top, 0.5 =
            random), with hit rates in the top 10 / 50, and a team-season
            bootstrap for the difference between systems. Teams are not
            optimal, so this measures plausibility, not quality -- the
            ground LightFM's roster co-occurrence should be strongest on.

            Three views of the same arrivals:
              headline   every rankable veteran arrival, unweighted. Fringe
                         signings add noise that pulls every system toward
                         0.5 alike; they do not bias the comparison.
              weighted   by the player's season-t floor-time share, so
                         established players count more. Fixed, nothing fit.
              rotation   only arrivals who took >= ROTATION_SHARE of the new
                         team's floor time in t+1. This conditions on the
                         outcome -- players who earned minutes skew good,
                         which favours quality-based systems -- so it is a
                         secondary line, never the headline.
            Results are also split by season-t possessions, because below
            1,500 an impact estimate is mostly its box-score prior.

  quality   Among those same arrivals, did the ones a system ranked higher
            deliver more? Realised gain = share of the new team's floor time
            in t+1 x net impact in t+1; scored as the Spearman rho of the
            within-list percentile with it, and the mean realised gain of
            arrivals ranked in the top 50 minus the rest. Realism alone
            rewards ranking every fringe signing high; this is its check.
            Only arrivals are seen, so it cannot credit a player nobody took.

  value     Does a higher score mean a better outcome? Two levels:
              player  the acquired player's net impact in t+1, against his
                      season-t BPM, impact, and impact + archetype value.
                      Tests whether quality transfers across a move.
              team    the team's net-rating change t -> t+1, regressed on
                      net_rtg_t (regression to the mean) and the value of
                      who left; the question is how much R^2 each way of
                      scoring the arrivals adds on top. This is where the
                      team-specific part (who gets replaced) has to earn its
                      place over a generic quality sum. For impact_recs
                      systems the sum is of the value part of each arrival's
                      score (`gain`, the legal ones), not the score itself,
                      which also carries REALISM_WEIGHT * log P(acquired).

Candidate pool for team-season (T, t): players with a season-t impact on at
least --min-poss offensive possessions (default EVAL_MIN_POSS), not on T in t;
impact_recs scores the same pool. Rookies (debut in t+1, per Player Career
Info) arrive through the draft, which no system here recommends, so they are
left out of the denominator and reported separately. Other arrivals outside
the pool (too few possessions in t, back from injury or abroad) are counted,
not scored. A system that gives a pool member no score (LightFM missing the
item, the salary screen ruling him out) ranks him last.

Known leak, small and favouring LightFM: its item features for p_t resemble
those for p_{t+1}, which it was trained on. (Archetype effects used to be
pooled over every season; impact_recs now reads them as of season t.)

Usage:
    python evaluate_recs.py                  # full report
    python evaluate_recs.py --min-poss 1500  # sensitivity: impact_recs' own pool
    python evaluate_recs.py --gap-grid 0.5 1 2 3 5   # tune impact_recs' --max-gap
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import impact_recs as ir

MIN_ACQ_SHARE = 0.05    # of the new team's floor time in t+1: he actually played
ROTATION_SHARE = 0.15   # the "rotation in t+1" view
EVAL_MIN_POSS = 500     # pool threshold; impact_recs' MIN_CAND_POSS is 1500
N_BOOT = 500
TOP = (10, 50)

_as_of_t = ir.as_of_t


def lightfm_scores(path):
    """Every score from the retired LightFM recommender, for every team-season,
    or None without the dump: an optional baseline row, not a dependency. The
    engine that produced it (rec_sys.py, backend/inference.py) is in
    local_archive/ and the dump can no longer be regenerated.
    """
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path)
    parts = s["item_id"].str.rsplit("_", n=1, expand=True)
    return s.assign(player_id=parts[0], season=parts[1].astype(int))


memberships = ir.memberships


def moves(member, impact, debut, min_poss=EVAL_MIN_POSS):
    """Arrivals and departures, both keyed to the season the decision is made in (t).

    `debut` maps player_id -> first NBA season. Arrivals get `rookie`,
    `prior_share` (summed over his season-t teams), `off_poss` in t and
    `expansion` and `rankable` (a veteran in the season-t pool).
    """
    key = ["season", "TeamId", "player_id"]
    now = member.rename(columns={"share": "share_t"})
    nxt = _as_of_t(member.assign(season=member["season"] - 1).rename(columns={"share": "share_next"}))
    last = member["season"].max()

    arr = nxt[(nxt["share_next"] >= MIN_ACQ_SHARE) & (nxt["season"] >= member["season"].min())]
    arr = arr.merge(now[key], on=key, how="left", indicator=True)
    arr = arr[arr["_merge"] == "left_only"].drop(columns="_merge")
    arr["rookie"] = arr["player_id"].map(debut).eq(arr["season"] + 1)
    prior = now.groupby(["player_id", "season"])["share_t"].sum().rename("prior_share")
    arr = arr.join(prior, on=["player_id", "season"]).fillna({"prior_share": 0.0})
    arr = arr.merge(impact[["player_id", "season", "off_poss"]], on=["player_id", "season"], how="left")
    # An expansion team (2005 Bobcats) has no season-t list to rank against.
    existing = set(zip(member["season"], member["TeamId"]))
    arr["expansion"] = [k not in existing for k in zip(arr["season"], arr["TeamId"])]
    arr["rankable"] = (~arr["rookie"] & ~arr["expansion"]
                       & (arr["off_poss"].fillna(0) >= min_poss))

    dep = now[(now["share_t"] >= MIN_ACQ_SHARE) & (now["season"] < last)]
    dep = dep.merge(nxt[key], on=key, how="left", indicator=True)
    dep = dep[dep["_merge"] == "left_only"].drop(columns="_merge")
    return arr.reset_index(drop=True), dep


def label_mechanism(arr, transactions, teams):
    """How each arrival happened, from BBRef's transaction log: 'trade',
    'signing' (incl. 10-day / rest of season), 'claim', or 'not listed'.

    Evaluation-only: the log omits most re-signings, so it cannot say who was
    available, only how the players who moved got there.
    """
    from lineup_features import bbref_team
    code = teams.assign(season=teams["season"] - 1,
                        code=[bbref_team(a, s) for a, s in
                              zip(teams["TeamAbbreviation"], teams["season"])])
    tx = pd.read_csv(transactions, parse_dates=["date"])
    tx = tx[tx["kind"].isin(["trade", "signing", "claim"])].assign(season=tx["page_season"] - 1)
    first = tx.sort_values("date").drop_duplicates(["season", "player_id", "team_to"])
    a = arr.merge(code[["season", "TeamId", "code"]], how="left")
    a = a.merge(first[["season", "player_id", "team_to", "kind"]].rename(columns={"kind": "via"}),
                left_on=["season", "player_id", "code"], right_on=["season", "player_id", "team_to"],
                how="left")
    return a.drop(columns=["code", "team_to"]).fillna({"via": "not listed"})


def variant(salary=True, gap=None, fa=False, scoring=ir.SCORING, projected=True, soft=True,
            realism=None, creator=ir.CREATOR_TERM, surplus_penalty=None, aggregate=None,
            fit_weight=None, partner_match=None, move_cap=None):
    """One impact_recs configuration to score. soft: rank moves the screens rule
    out below the legal ones, by score, instead of tied last. realism: weight
    on log P(acquired) (None = impact_recs.REALISM_WEIGHT). surplus_penalty:
    lambda on max(0, surplus gap) (None = impact_recs.SURPLUS_PENALTY)."""
    return dict(salary=salary, gap=gap, fa=fa, scoring=scoring, projected=projected, soft=soft,
                realism=realism, creator=creator, surplus_penalty=surplus_penalty,
                aggregate=aggregate, fit_weight=fit_weight, partner_match=partner_match,
                move_cap=move_cap)


def impact_rankings(args, variants):
    """Full rankings from impact_recs; `variants` maps name -> variant(...)."""
    panel, impact, main, players, teams, salaries, books = ir.load(args)
    impact = ir.with_projection(impact, args.adv, panel)
    draft = pd.read_csv(args.draft, usecols=["season", "lg", "round", "player_id"])
    flags = ir.pending_fa(salaries, draft)
    rooms = ir.team_room(teams, books, flags)
    keep = ["season", "TeamId", "user_id"]
    out = {name: [] for name in variants}
    debut = pd.read_csv(args.career, usecols=["player_id", "from"]) \
        .drop_duplicates("player_id").set_index("player_id")["from"]
    member = ir.memberships(panel)
    creators, b2 = ir.creator_inputs()
    ptab_sets = {}
    for projected in {v["projected"] for v in variants.values()}:
        imp = impact if projected else impact.drop(columns=["proj_impact", "proj_stay"], errors="ignore")
        ptabs = ir.fa_prices({s: ir.player_table(s, imp, players, main, salaries, flags, creators)
                              for s in sorted(panel["season"].unique())}, salaries)
        ptab_sets[projected] = ir.availability(ptabs, member, debut, teams)
    import acquisition
    p_acq = acquisition.p_acq_table(args, ptab_sets[True] if True in ptab_sets else None,
                                    variant=getattr(args, "acq_variant", "base"))
    for season in sorted(panel["season"].unique()):
        df = panel[panel["season"] == season]
        r = rooms[rooms["season"] == season].set_index("TeamId")
        for name, v in variants.items():
            recs = ir.recommend_season(season, df, ptab_sets[v["projected"]][season], teams[keep],
                                       r["room"] if v["salary"] else None,
                                       args.filler, k=None, min_poss=args.min_poss,
                                       max_gap=v["gap"] if v["salary"] else None,
                                       fa_budget=r["fa_budget"] if v["salary"] and v["fa"] else None,
                                       scoring=v["scoring"], soft=v["soft"],
                                       p_acq=p_acq.xs(season, level="season"),
                                       realism=v["realism"],
                                       surplus_penalty=v.get("surplus_penalty"),
                                       aggregate=v.get("aggregate"),
                                       fit_weight=v.get("fit_weight"),
                                       partner_match=v.get("partner_match"),
                                       move_cap=v.get("move_cap"),
                                       creator_b2=b2.get(season) if v["creator"] else None)
            out[name].append(recs[["season", "TeamId", "player_id", "score", "gain"]])
    out = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    out["acquisition model alone"] = p_acq.rename("score").reset_index()
    return out, panel, impact, players, teams


def percentiles(pool, scores):
    """Rank pool members by `scores` within each team-season; unscored rank last."""
    s = pool.merge(scores, on=["season", "TeamId", "player_id"], how="left")
    s["score"] = s["score"].replace(-np.inf, np.nan).fillna(-np.inf)
    g = s.groupby(["season", "TeamId"])["score"]
    s["rank"] = g.rank(ascending=False, method="average")
    s["pct"] = (s["rank"] - 0.5) / g.transform("size")
    return s[["season", "TeamId", "player_id", "rank", "pct"]]


def _boot_diff(d, w, codes, n_clusters, rng):
    """95% CI of a weighted mean difference, resampling whole team-seasons."""
    num = np.bincount(codes, d * w, n_clusters)
    den = np.bincount(codes, w, n_clusters)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_clusters, n_clusters)
        boot.append(num[idx].sum() / den[idx].sum())
    return np.percentile(boot, [2.5, 97.5])


def realism(arr, pool, systems, rng, ref="impact + salary"):
    vets = arr[~arr["rookie"] & ~arr["expansion"]]
    hits = arr[arr["rankable"]].reset_index(drop=True)
    print(f"Arrivals in t+1 with >= {MIN_ACQ_SHARE:.0%} of the new team's floor time: {len(arr)}")
    print(f"  rookies (draft; out of scope): {arr['rookie'].sum()} ({arr['rookie'].mean():.0%})")
    print(f"  veterans joining an expansion team (no season-t team): {(arr['expansion'] & ~arr['rookie']).sum()}")
    print(f"  veterans: {len(vets)}, of which rankable: {len(hits)} "
          f"({len(hits) / len(vets):.0%} of veterans, {len(hits) / len(arr):.0%} of all arrivals)")
    print(f"Pool size per team-season: {pool.groupby(['season', 'TeamId']).size().mean():.0f}")

    pcts = {name: hits[["season", "TeamId", "player_id"]].merge(
        percentiles(pool, scores), how="left") for name, scores in systems.items()}
    one = np.ones(len(hits))
    big = (hits["off_poss"] >= ir.MIN_CAND_POSS).to_numpy(float)
    views = {
        "headline: all rankable veterans, unweighted": one,
        "weighted by season-t floor-time share": hits["prior_share"].to_numpy(),
        "rotation in t+1 (conditioned on outcome)":
            (hits["share_next"] >= ROTATION_SHARE).to_numpy(float),
        f"split: >= {ir.MIN_CAND_POSS} possessions in t": big,
        f"split: < {ir.MIN_CAND_POSS} possessions in t": 1.0 - big,
    }
    cluster = hits["season"].astype(str) + "_" + hits["TeamId"].astype(str)
    codes, uniq = pd.factorize(cluster)
    for view, w in views.items():
        if w.sum() == 0:
            continue
        rows = {}
        for name, p in pcts.items():
            row = {"mean percentile": np.average(p["pct"], weights=w),
                   **{f"in top {k}": np.average(p["rank"] <= k, weights=w) for k in TOP}}
            if name != ref:
                d = p["pct"].to_numpy() - pcts[ref]["pct"].to_numpy()
                lo, hi = _boot_diff(d, w, codes, len(uniq), rng)
                row[f"vs {ref} [95% CI]"] = f"{np.average(d, weights=w):+.3f} [{lo:+.3f}, {hi:+.3f}]"
            rows[name] = row
        print(f"\n-- {view}  (n = {int((w > 0).sum())})")
        print(pd.DataFrame(rows).T.round(3).fillna("").to_string())
    return pcts


def _spearman(x, y):
    return np.corrcoef(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())[0, 1]


def realised_gain(arr, impact):
    """Rankable arrivals with what they delivered in t+1: share_next * net impact
    there, points per 100 of team floor time vs. a league-average player."""
    after = impact[["player_id", "season", "net_impact"]].assign(
        season=impact["season"] - 1).rename(columns={"net_impact": "impact_next"})
    hits = arr[arr["rankable"]].reset_index(drop=True)
    hits = hits.merge(after, on=["player_id", "season"], how="left")
    hits["realised"] = hits["share_next"] * hits["impact_next"]
    return hits


def quality(hits, pcts, rng, ref="impact + salary", quiet=False):
    """Among actual arrivals, does ranking higher go with delivering more?

    Realism asks whether a system ranks the players teams took near the top;
    this asks whether, among those same players, the ones it ranked higher
    were the ones who helped. Scored on each system's within-list percentile,
    so systems with incomparable score scales (LightFM) are judged alike.
    Selection caveat: only players who moved are seen, so this cannot credit
    a system for a player nobody acquired.

      rho        Spearman of -percentile with realised gain
      top-50 lift  mean realised gain of arrivals ranked in the top 50 minus
                 the rest, pts/100 of team floor time
    """
    ok = hits["realised"].notna().to_numpy()
    cluster = (hits["season"].astype(str) + "_" + hits["TeamId"].astype(str))[ok]
    codes, uniq = pd.factorize(cluster)
    y = hits["realised"].to_numpy()[ok]
    groups = [np.flatnonzero(codes == c) for c in range(len(uniq))]
    boot_idx = [np.concatenate([groups[i] for i in rng.integers(0, len(uniq), len(uniq))])
                for _ in range(N_BOOT // 2)]
    x_ref = -pcts[ref]["pct"].to_numpy()[ok] if ref in pcts else None
    rows = {}
    for name, p in pcts.items():
        x = -p["pct"].to_numpy()[ok]
        top = (p["rank"].to_numpy()[ok] <= 50)
        row = {"rho": _spearman(x, y),
               "top-50 lift": y[top].mean() - y[~top].mean() if top.any() else np.nan}
        if x_ref is not None and name != ref:
            d = [_spearman(x[i], y[i]) - _spearman(x_ref[i], y[i]) for i in boot_idx]
            lo, hi = np.percentile(d, [2.5, 97.5])
            row[f"rho vs {ref} [95% CI]"] = f"{row['rho'] - _spearman(x_ref, y):+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows[name] = row
    if not quiet:
        print("\n== Quality: among actual arrivals, does a higher rank mean more delivered?\n")
        print(f"{ok.sum()} rankable arrivals with a t+1 impact; realised gain = share in t+1 "
              f"x net impact in t+1 (mean {y.mean():+.2f}, sd {y.std():.2f})")
        print(pd.DataFrame(rows).T.round(3).fillna("").to_string())
    return {k: v["rho"] for k, v in rows.items()}


def _wcorr(x, y, w):
    x, y = x - np.average(x, weights=w), y - np.average(y, weights=w)
    return np.average(x * y, weights=w) / np.sqrt(np.average(x**2, weights=w) * np.average(y**2, weights=w))


def _r2(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def value(arr, dep, systems, impact, players, teams, main, quiet=False):
    """Prints the player- and team-level tests; returns {scorer: team-level R^2 gain}."""
    imp = impact.merge(players[["player_id", "season", "bpm", "cluster_label"]], how="left")
    eff = ir.archetype_effect(main, imp["season"], imp["cluster_label"])
    imp["value"] = imp["net_impact"] + eff
    imp["projected value"] = imp["proj_impact"] + eff
    cols = ["player_id", "season", "bpm", "net_impact", "value", "projected value"]
    a = arr[arr["rankable"]].merge(imp[cols], on=["player_id", "season"], how="left")
    for name, s in systems.items():
        if name not in ("LightFM (today)", "best available (BPM)", "random", "acquisition model alone"):
            # Sum the value part (gain) only: the score also carries
            # REALISM_WEIGHT * log P(acquired), a likelihood, not points.
            sys_cols = s[["season", "TeamId", "player_id", "score"]
                         + (["gain"] if "gain" in s else [])].rename(columns={"score": f"{name} legal"})
            a = a.merge(sys_cols.rename(columns={"gain": name}) if "gain" in s
                        else sys_cols.assign(**{name: sys_cols[f"{name} legal"]}), how="left")
            a[name] = a[name].where(a[f"{name} legal"] > -ir.SOFT_PENALTY / 2)  # ruled out -> NaN
            a = a.drop(columns=f"{name} legal")

    # Player level: does season-t quality carry over to the new team?
    after = impact[["player_id", "season", "net_impact", "off_poss"]].assign(
        season=impact["season"] - 1).rename(columns={"net_impact": "impact_next", "off_poss": "w"})
    p = a.merge(after, on=["player_id", "season"], how="inner").dropna(subset=["bpm"])
    if not quiet:
        print(f"\nPlayer level: {len(p)} movers with an impact in both seasons; "
              f"weighted correlation with net impact in t+1")
        for c in ["bpm", "net_impact", "value", "projected value"]:
            r = _wcorr(p[c].to_numpy(), p["impact_next"].to_numpy(), p["w"].to_numpy())
            print(f"  {c:<12} {r:.3f}")

    # Team level: net-rating change explained by who arrived, given who left.
    net = teams[["season", "TeamId", "net_rtg"]]
    t = net.merge(_as_of_t(net.assign(season=net["season"] - 1)).rename(columns={"net_rtg": "net_next"}))
    t["dnet"] = t["net_next"] - t["net_rtg"]
    d = dep.merge(imp[cols], on=["player_id", "season"], how="left")
    d["lost"] = d["share_t"] * d["value"].fillna(0.0)
    t = t.merge(d.groupby(["season", "TeamId"])["lost"].sum().reset_index(), how="left")
    scorers = {
        "arrivals only (count)": a.assign(x=1.0),
        "sum of BPM": a.assign(x=a["bpm"].fillna(0.0)),
        "sum of value": a.assign(x=a["value"].fillna(0.0)),
        "sum of projected value": a.assign(x=a["projected value"].fillna(0.0)),
    } | {f"sum of {name} value part": a.assign(x=a[name].fillna(0.0))
         for name in systems if name in a}
    t = t.fillna({"lost": 0.0})
    base = _r2(t["dnet"].to_numpy(), t[["net_rtg", "lost"]].to_numpy())
    if not quiet:
        print(f"\nTeam level: {len(t)} team-seasons; net-rating change t -> t+1")
        print(f"  base (net_rtg_t + value lost)      R^2 {base:.3f}")
    gains = {}
    for name, sc in scorers.items():
        x = t.merge(sc.groupby(["season", "TeamId"])["x"].sum().reset_index(), how="left")["x"]
        r2 = _r2(t["dnet"].to_numpy(), np.column_stack([t[["net_rtg", "lost"]], x.fillna(0.0)]))
        gains[name] = r2 - base
        if not quiet:
            print(f"  + {name:<40} R^2 {r2:.3f}  ({r2 - base:+.3f})")
    if not quiet:
        for name in systems:
            if name in a and a[name].isna().any():
                print(f"  ('{name}' rules out {a[name].isna().mean():.0%} of arrivals; counted as 0)")
    return gains


def gap_grid(arr, dep, pool, systems, impact, players, teams, main_eff, rng):
    """One row per variant: realism in each view, quality, and the team-level value gain."""
    hits = realised_gain(arr, impact)
    views = {"headline": np.ones(len(hits)), "weighted": hits["prior_share"].to_numpy(),
             "rotation t+1": (hits["share_next"] >= ROTATION_SHARE).to_numpy(float),
             "trades": (hits["via"] == "trade").to_numpy(float),
             "signings": (hits["via"] == "signing").to_numpy(float)}
    gains = value(arr, dep, systems, impact, players, teams, main_eff, quiet=True)
    pcts = {name: hits[["season", "TeamId", "player_id"]].merge(percentiles(pool, scores), how="left")
            for name, scores in systems.items()}
    rho = quality(hits, pcts, rng, ref=None, quiet=True)
    rows = {}
    for name, scores in systems.items():
        p = pcts[name]
        row = {f"pct, {v}": np.average(p["pct"], weights=w) for v, w in views.items()}
        row["in top 50"] = (p["rank"] <= 50).mean()
        row["quality rho"] = rho[name]
        sc = hits[["season", "TeamId", "player_id"]].merge(scores, how="left")["score"]
        row["ruled out"] = (~(sc > -ir.SOFT_PENALTY / 2)).mean()
        row["team R^2 gain"] = gains.get(f"sum of {name} value part", np.nan)
        rows[name] = row
    print("Mean percentile of actual acquisitions (lower = more realistic) and value test:")
    print(pd.DataFrame(rows).T.round(3).to_string())


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lightfm-full", default=os.path.join("data", "lightfm_scores_full.csv"))
    ap.add_argument("--filler", type=int, default=ir.FILLER)
    ap.add_argument("--max-gap", type=float, default=None,
                    help="also score impact + salary with this value cap")
    ap.add_argument("--gap-grid", type=float, nargs="+",
                    help="compact table over these value-cap tolerances instead of the full report")
    ap.add_argument("--scoring-grid", nargs="+", choices=ir.SCORINGS,
                    help="compact table over these scorings (trades only); with --gap-grid, "
                         "each scoring is also run at those value-cap tolerances")
    ap.add_argument("--realism-grid", type=float, nargs="+",
                    help="with --scoring-grid: also score these weights on log P(acquired)")
    ap.add_argument("--min-poss", type=int, default=EVAL_MIN_POSS,
                    help="season-t offensive possessions to enter the candidate pool")
    ap.add_argument("--career", default=os.path.join("data", "Player Career Info.csv"))
    ap.add_argument("--adv", default=os.path.join("data", "Advanced.csv"))
    ap.add_argument("--panel", default=os.path.join("data", "lineup_panel.csv"))
    ap.add_argument("--impact", default=os.path.join("data", "player_impact.csv"))
    ap.add_argument("--fit", default=os.path.join("data", "archetype_main.csv"))
    ap.add_argument("--players", default=os.path.join("data", "player_seasons.csv"))
    ap.add_argument("--teams", default=os.path.join("data", "team_style.csv"))
    ap.add_argument("--salaries", default=os.path.join("data", "salaries.csv"))
    ap.add_argument("--payroll", default=os.path.join("data", "team_payroll.csv"))
    ap.add_argument("--rosters", default=os.path.join("data", "team_salaries.csv"))
    ap.add_argument("--transactions", default=os.path.join("data", "transactions.csv"))
    ap.add_argument("--draft", default=os.path.join("data", "Draft Pick History.csv"))
    ap.add_argument("--seasons", type=int, nargs=2, metavar=("START", "END"),
                    help="score only decisions made in seasons START-END")
    ap.add_argument("--acq-variant", default="base",
                    help="acquisition.VARIANTS entry used for impact_recs' realism term")
    return ap


def main():
    args = parser().parse_args()
    rng = np.random.default_rng(0)

    variants = {"impact": variant(salary=False), "impact + salary": variant(gap=ir.MAX_GAP)}
    gaps = [] if args.scoring_grid else (args.gap_grid or [])
    for gap in ([args.max_gap] if args.max_gap is not None else []) + gaps:
        variants[f"impact + salary, gap <= {gap:g}"] = variant(gap=gap)
        variants[f"impact + salary, gap <= {gap:g}, FA"] = variant(gap=gap, fa=True)
    for mode in args.scoring_grid or []:
        variants[f"{mode}, raw"] = variant(gap=ir.MAX_GAP, scoring=mode, projected=False)
        variants[f"{mode}"] = variant(gap=ir.MAX_GAP, scoring=mode)
        for gap in args.gap_grid or []:
            variants[f"{mode}, gap <= {gap:g}"] = variant(gap=gap, scoring=mode)
        variants[f"{mode}, no cap"] = variant(scoring=mode)
        variants[f"{mode}, no salary"] = variant(salary=False, scoring=mode)
        variants[f"{mode}, hard"] = variant(gap=ir.MAX_GAP, scoring=mode, soft=False)
        variants[f"{mode}, no creator term"] = variant(gap=ir.MAX_GAP, scoring=mode, creator=False)
        variants[f"{mode}, no creator, realism 0"] = variant(gap=ir.MAX_GAP, scoring=mode,
                                                            creator=False, realism=0.0)
        variants[f"{mode}, realism 0"] = variant(gap=ir.MAX_GAP, scoring=mode, realism=0.0)
        for w in args.realism_grid or []:
            variants[f"{mode}, realism {w:g}"] = variant(gap=ir.MAX_GAP, scoring=mode, realism=w)
    systems, panel, impact, players, _ = impact_rankings(args, variants)
    teams = pd.read_csv(args.teams, usecols=["season", "TeamId", "user_id", "net_rtg"])
    main_eff = ir.main_effects(pd.read_csv(args.fit)) if ir.ARCHETYPES else pd.Series(dtype=float)
    pool = systems["impact"][["season", "TeamId", "player_id"]]

    lf = lightfm_scores(args.lightfm_full)
    if lf is not None:
        lf = lf.merge(teams[["user_id", "TeamId"]], on="user_id", how="inner")
        systems["LightFM (today)"] = lf[["season", "TeamId", "player_id", "score"]]
    else:
        print(f"(no {args.lightfm_full}: LightFM baseline skipped)\n")
    bpm = players[["player_id", "season", "bpm"]].rename(columns={"bpm": "score"})
    systems["best available (BPM)"] = pool.merge(bpm, how="left")
    systems["random"] = pool.assign(score=rng.random(len(pool)))

    debut = pd.read_csv(args.career, usecols=["player_id", "from"]) \
        .drop_duplicates("player_id").set_index("player_id")["from"]
    arr, dep = moves(memberships(panel), impact, debut, args.min_poss)
    if args.seasons:
        lo, hi = args.seasons
        arr, dep = (x[(x["season"] >= lo) & (x["season"] <= hi)] for x in (arr, dep))
        pool = pool[(pool["season"] >= lo) & (pool["season"] <= hi)]
        teams = teams[(teams["season"] >= lo) & (teams["season"] <= hi)]
        print(f"(scoring decisions in seasons {lo}-{hi} only)\n")
    tcodes = pd.read_csv(args.teams, usecols=["season", "TeamId", "TeamAbbreviation"])
    arr = label_mechanism(arr, args.transactions, tcodes)
    if args.gap_grid or args.scoring_grid:
        gap_grid(arr, dep, pool, systems, impact, players, teams, main_eff, rng)
        return 0
    print("== Realism: where did actual acquisitions rank?\n")
    pcts = realism(arr, pool, systems, rng)
    quality(realised_gain(arr, impact), pcts, rng)
    print("\n== Value: does a higher score mean a better outcome?")
    value(arr, dep, systems, impact, players, teams, main_eff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
