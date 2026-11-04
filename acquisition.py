"""P(team T acquires player p for next season): the realism half of a recommendation.

impact_recs.py ranks on quality, and quality alone cannot be realistic: teams
mostly acquire affordable role players, so any ranking by value, even plain
BPM, puts real arrivals below random (evaluate_recs.py). LightFM was the most
realistic system only because it learns what teams typically take. This
learns the same thing from features instead of roster co-occurrence, so it
can be combined with value:

    score = value(c) + REALISM_WEIGHT * log P(T acquires c)

One row per (team-season, candidate) in the evaluation pool (>= 500
offensive possessions in t, not on T in t). Label: c is a veteran arrival on
T in t+1 (>= 5% of its floor time), exactly evaluate_recs.moves(). Every
feature is season-t information:

  candidate  projected value, raw impact, BPM, age, experience, cap share,
             floor-time share, possessions, traded mid-season, his team's
             net rating (sellers), position
  team       net rating, cap room and payroll as shares of the cap
  pair       salary fit (his cap share vs. 125% of T's largest rotation
             contract plus room), upgrade over T's weakest rotation player,
             value over T's minutes-weighted value at his position, and
             whether he has played for T before

Gradient-boosted trees, refit per season t on seasons < t only (an
arrival in s is known once s+1 is played, so season t trains on s <= t-1).
Seasons with fewer than MIN_HISTORY seasons of labels get no model (P = the
pooled base rate), which only affects the first few.

Scored the way impact_recs uses it: inside each team-season's list (held out
season by season, 2002-2024, 2,767 arrivals among 257,447 pairs). Pooled AUC
also credits telling movers from stayers league-wide, which experience and
age do without ranking any one team's candidates; it is reported, not used.

                       mean pct   within AUC   top 10   top 50   log loss
  all features         0.370      0.632        5.3%     22.8%    0.0583
  without candidate    0.405      0.597        4.1%     20.2%    0.0588
  without team         0.375      0.627        4.8%     22.2%    0.0584
  without pair         0.368      0.634        5.3%     22.6%    0.0583
  without contract     0.371      0.631        4.5%     22.2%    0.0584
  base rate only                                                 0.0594

Candidate features carry it (experience and age most); the team block adds a
little; pair features nothing. Contract status (contract_status.py: years
into the contract from the salary chain, rookie scale, minimum deals, at the
CBA's max length) is neutral within the noise: by arrival type the mean
percentile moves +0.002 [-0.002, +0.006] for signings, +0.001 [-0.003,
+0.005] for trades, and the full recommender's realism / quality do not move
(0.476 -> 0.475, 0.280 -> 0.280). Calibration holds by era (predicted /
actual move rate 0.87-1.06) and by decile, but log loss is only 2% under the
base rate: the probabilities are close to flat.

Team intent (contender vs rebuilding) does not help either. It is real as a
description -- top-third teams by net rating acquire players 1.2 years older
(28.3 vs 27.1), better (value -0.38 vs -0.67) and pricier (9.8% vs 7.9% of
the cap) than bottom-third teams -- but team_net is already a feature, and
adding roster age, net-rating trend, best player, share of minutes to <= 24
year-olds and candidate x team interactions (age gap, contender x age, value
vs the team's best) left the within-list ranking unchanged (AUC 0.6319 ->
0.6313, top 10 5.3% -> 5.4%), so they were taken out. The lists are older
than real arrivals for every tier, most for bad teams (top-10 28.5 vs 27.1;
age <= 24: 16% vs 28%), but weighting youth for them does not pay off on
the score side: adding expected age growth to value moves the Spearman with
what arrivals delivered over two seasons by -0.001 [-0.013, +0.011] for
bad teams, +0.001 [-0.011, +0.014] for good ones.

Model form does not help either (--model-compare, decisions 2002-2025):
                       mean pct   within AUC   top 10   vs current [95% CI]
  current classifier   0.370      0.632        5.3%
  LambdaMART ranker    0.379      0.623        5.6%     +0.009 [+0.002, +0.015]
  two-stage            0.368      0.634        4.6%     -0.002 [-0.008, +0.005]
The ranker trains on the within-list choice directly (XGBoost rank:ndcg,
grouped by team-season, Platt-calibrated); its best of three objectives
reached 0.372. The two-stage model -- P(moves anywhere) x P(this team |
moves) -- explains why nothing team-side has worked: stage 1 ranks movers at
AUC 0.667, but stage 2, fit on movers alone so team features are not drowned
by "does he move at all", puts the real destination at mean rank 14.2 of 29
teams (random 14.9; top 3 11.6% vs 10.5%). With these features, realism is
almost entirely who moves, not where.

Destination features (destination.py: trade partners, front-office changes
and ties, roster churn) do not help either. Among players who moved, each
lifts a team's chance of being the destination by at most ~1.2x (traded
with his team 2+ times in 3 years 1.21x, its new GM or coach came from his
team 1.20x, he played there before 1.12x, it changed GM 1.11x), and in stage
2 they move the real destination up +0.17 places of 29 [-0.16, +0.50]; in
the pooled model they are neutral (mean pct 0.3697 -> 0.3710). They are kept
out of FEATURES; the "two-stage + destination" variant reproduces this.

Free-agency labels (fetch_wikipedia_fa.py; complete for offseasons after
2012-2025) do not help. On decisions in 2015-2024 (1,296 arrivals), mean
percentile vs the default trained on all history, team-season bootstrap:
  trained on 2012+ only                +0.005 [-0.004, +0.013]
  2012+ only, + learned P(FA)          +0.006 [-0.002, +0.014]
  all history, + learned P(FA)         -0.001 [-0.005, +0.002]
  2012+ only, + listed (leaky bound)   -0.023 [-0.035, -0.011]
P(free agent) itself is learnable from season-t information (held-out AUC
0.80), but the candidate and contract features already carry that signal;
and even knowing the list exactly is worth ~2 points, because only 52% of
arrivals were free agents while 32% of all candidates are each offseason.
In the full recommender over 2015-2024: realism 0.458 / 0.457 / 0.436,
quality rho 0.272 / 0.269 / 0.226 for default / + P(FA) / leaky bound.
So the default stays "base" (VARIANTS keeps the others for --wiki-compare).

Usage:
    python acquisition.py --model-compare            # classifier vs ranker vs two-stage
    python acquisition.py --wiki-compare 2015 2024   # the free-agency comparison
    python acquisition.py          # within-list ranking, calibration, feature-group ablation
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import impact_recs as ir
from destination import DESTINATION_FEATURES
from fetch_salaries import CAP

POOL_MIN_POSS = 500     # evaluate_recs.EVAL_MIN_POSS
MIN_ACQ_SHARE = 0.05    # evaluate_recs.MIN_ACQ_SHARE
MIN_HISTORY = 3         # seasons of labels before the model is fit
NO_HISTORY_RATE = 0.01  # P for seasons with no labelled history at all (~ the base rate)
CALIBRATE_MOVES = False # recalibrate P(he moves at all) per season (calibrate_moves(): null)
WIKIPEDIA_FA = os.path.join("data", "wikipedia_fa.csv")
WIKI_FIRST = 2012       # first offseason (after season t) Wikipedia lists completely
FA_MIN_HISTORY = 2      # labelled seasons before P(free agent) is fit
WIKI_FEATURES = ["p_fa"]                 # fair: learned from earlier seasons' labels
ORACLE_FEATURES = ["fa_listed", "fa_rfa"]  # leaky upper bound: the list itself
CONTRACT_FEATURES = ["contract_year", "censored", "at_max_len", "on_rookie", "rookie_final",
                     "on_minimum", "partial", "cba"]
FEATURES = [
    "value", "net_impact", "bpm", "age", "experience", "cap_pct", "c_share", "off_poss",
    "traded", "from_net", "pos_index",
    "team_net", "room_pct", "payroll_pct",
    "salary_fit", "upgrade", "need", "former",
] + CONTRACT_FEATURES


def arrivals(member, debut):
    """(season t, TeamId as of t, player_id) of veterans who joined T for t+1."""
    key = ["season", "TeamId", "player_id"]
    nxt = ir.as_of_t(member.assign(season=member["season"] - 1))
    nxt = nxt[(nxt["share"] >= MIN_ACQ_SHARE) & (nxt["season"] >= member["season"].min())]
    now = member[key].assign(on=True)
    a = nxt[key].merge(now, on=key, how="left")
    a = a[a["on"].isna()][key]
    a = a[a["player_id"].map(debut).ne(a["season"] + 1)]
    return a.drop_duplicates().assign(acquired=1)


def season_pairs(season, df, ptab, member, room, payroll, net):
    """Feature rows for every (team, candidate) in season t's pool."""
    share, _, _ = ir.rotation(df)
    on_team = share.groupby("TeamId")["player_id"].apply(set)
    by_player = member[member["season"] == season]
    c = ptab[ptab["off_poss"] >= POOL_MIN_POSS].copy()
    c["c_share"] = share.groupby("player_id")["share"].sum().clip(upper=1.0).reindex(c.index).fillna(0.0)
    c["traded"] = (by_player.groupby("player_id")["TeamId"].nunique().reindex(c.index).fillna(1) > 1) * 1.0
    main_team = by_player.sort_values("share").drop_duplicates("player_id", keep="last") \
        .set_index("player_id")["TeamId"]
    c["from_net"] = net.reindex(main_team.reindex(c.index).to_numpy()).to_numpy()
    earlier = member[member["season"] < season].groupby("player_id")["TeamId"].apply(set)
    cap = CAP[season]

    rows = []
    for team_id, rot in share[share["share"] >= ir.MIN_SHARE].groupby("TeamId"):
        r = ptab.reindex(rot["player_id"])
        r_val = r["value"].fillna(0.0).to_numpy()
        r_pos = r["pos_index"].to_numpy()
        r_share = rot["share"].to_numpy()
        win = np.abs(c["pos_index"].to_numpy()[:, None] - r_pos[None, :]) <= ir.POS_WINDOW
        w = win * r_share[None, :]
        tot = w.sum(1)
        v_pos = np.where(tot > 0, (w @ r_val) / np.where(tot > 0, tot, 1.0),
                         np.average(r_val, weights=r_share))
        top_sal = np.nanmax(r["salary"].to_numpy()) if r["salary"].notna().any() else 0.0
        reach = (ir.MATCH_PCT * top_sal + ir.match_pad(season) + room.get(team_id, 0.0)) / cap
        f = pd.DataFrame({
            "season": season, "TeamId": team_id, "player_id": c.index,
            **{col: c[col].to_numpy() for col in [
                "value", "net_impact", "bpm", "age", "experience", "cap_pct", "c_share",
                "off_poss", "traded", "from_net", "pos_index"]},
            "team_net": net.get(team_id, np.nan),
            "room_pct": room.get(team_id, 0.0) / cap,
            "payroll_pct": payroll.get(team_id, np.nan) / cap,
            "salary_fit": c["cap_pct"].to_numpy() - reach,
            "upgrade": c["value"].to_numpy() - r_val.min(),
            "need": c["value"].to_numpy() - v_pos,
            "former": [team_id in earlier.get(p, ()) for p in c.index],
        })
        rows.append(f[~f["player_id"].isin(on_team.get(team_id, set()))])
    out = pd.concat(rows, ignore_index=True)
    out["former"] = out["former"].astype(float)
    return out


def build_pairs(paths, ptabs=None):
    """Feature rows and labels for every season. `ptabs` from impact_recs
    (player tables with projected value); built here if not given."""
    panel, impact, main, players, teams, salaries, books = ir.load(paths)
    if ptabs is None:
        impact = ir.with_projection(impact, paths.adv, panel)
        ptabs = {s: ir.player_table(s, impact, players, main, salaries)
                 for s in sorted(panel["season"].unique())}
    rooms = ir.team_room(teams, books)
    payroll, _ = books
    from lineup_features import bbref_team
    pay = teams.assign(team=[bbref_team(a, s) for a, s in zip(teams["TeamAbbreviation"], teams["season"])]) \
        .merge(payroll, on=["season", "team"], how="left")
    debut = pd.read_csv(paths.career, usecols=["player_id", "from"]) \
        .drop_duplicates("player_id").set_index("player_id")["from"]
    member = ir.memberships(panel)
    parts = []
    for season, ptab in ptabs.items():
        t = ptab.assign(experience=season - debut.reindex(ptab.index).to_numpy())
        room = rooms[rooms["season"] == season].set_index("TeamId")["room"]
        pr = pay[pay["season"] == season].set_index("TeamId")["payroll"]
        net = teams[teams["season"] == season].set_index("TeamId")["net_rtg"]
        parts.append(season_pairs(season, panel[panel["season"] == season], t, member, room, pr, net))
    pairs = pd.concat(parts, ignore_index=True)
    import contract_status as cs
    status = cs.contract_status(*cs.load_inputs(paths.salaries, paths.career, paths.draft))
    pairs = pairs.merge(status[["player_id", "season"] + CONTRACT_FEATURES],
                        on=["player_id", "season"], how="left")
    for c in CONTRACT_FEATURES:
        pairs[c] = pairs[c].astype(float)      # no salary row -> NaN, which the trees handle
    import destination
    tx = pd.read_csv(getattr(paths, "transactions", os.path.join("data", "transactions.csv")))
    pairs = destination.add_destination(pairs, member, teams, tx, status)
    labels = arrivals(member, debut)
    pairs = pairs.merge(labels, on=["season", "TeamId", "player_id"], how="left")
    pairs["acquired"] = pairs["acquired"].fillna(0).astype(int)
    last = panel["season"].max()
    pairs.loc[pairs["season"] == last, "acquired"] = -1        # t+1 not played yet
    return pairs


def fit_predict(pairs, features=None, min_train_season=None):
    """Add p_acq: per season t, a model fit on seasons < t (labelled ones),
    and >= min_train_season if given. Also p_base, the training base rate,
    the no-information baseline."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    features = FEATURES if features is None else features
    pairs = pairs.copy()
    pairs["p_acq"] = np.nan
    pairs["p_base"] = np.nan
    labelled = pairs["acquired"] >= 0
    if min_train_season is not None:
        labelled &= pairs["season"] >= min_train_season
    seasons = sorted(pairs["season"].unique())
    for t in seasons:
        train = labelled & (pairs["season"] < t)
        here = pairs["season"] == t
        base = pairs.loc[train, "acquired"].mean() if train.any() else np.nan
        pairs.loc[here, "p_base"] = base
        if pairs.loc[train, "season"].nunique() < MIN_HISTORY:
            pairs.loc[here, "p_acq"] = base
            continue
        model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                               min_samples_leaf=100, l2_regularization=1.0,
                                               random_state=0)
        model.fit(pairs.loc[train, features], pairs.loc[train, "acquired"])
        pairs.loc[here, "p_acq"] = model.predict_proba(pairs.loc[here, features])[:, 1]
    # No history at all (first season): a fixed rate, so log P is defined. A
    # constant within a team-season leaves its ranking unchanged.
    for col in ("p_acq", "p_base"):
        pairs[col] = pairs[col].fillna(NO_HISTORY_RATE).clip(lower=1e-6)
    return pairs


def add_free_agency(pairs, path=WIKIPEDIA_FA):
    """Wikipedia free-agent labels (fetch_wikipedia_fa.py) and a fair P(FA).

    fa_listed / fa_rfa: he is in the offseason-after-t free agency table. Not
    known at decision time in this form -- the table lists free agents who
    signed (plus, recently, a few renounced and unsigned) -- so it is an upper
    bound, never a production feature. p_fa: P(free agent after t) from
    season-t information only (contract status, age, salary, role), fit per
    season on earlier seasons' labels from WIKI_FIRST on.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    fa = pd.read_csv(path)
    key = pd.MultiIndex.from_frame(pairs[["player_id", "season"]])
    lab = fa.set_index(["player_id", "season"])
    pairs = pairs.copy()
    covered = pairs["season"] >= WIKI_FIRST
    pairs["fa_listed"] = np.where(covered, key.isin(lab.index), np.nan)
    pairs["fa_rfa"] = np.where(covered, lab["rfa"].reindex(key).fillna(False).astype(float), np.nan)

    cols = ["age", "experience", "cap_pct", "c_share", "value", "net_impact", "off_poss",
            "traded", "from_net"] + CONTRACT_FEATURES
    players = pairs.drop_duplicates(["player_id", "season"])[["player_id", "season", "fa_listed"] + cols]
    players["p_fa"] = np.nan
    last = pairs["season"].max()
    for t in sorted(players["season"].unique()):
        train = (players["season"] >= WIKI_FIRST) & (players["season"] < t) & (players["season"] < last)
        if players.loc[train, "season"].nunique() < FA_MIN_HISTORY:
            continue
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                           min_samples_leaf=50, random_state=0)
        m.fit(players.loc[train, cols], players.loc[train, "fa_listed"].astype(int))
        here = players["season"] == t
        players.loc[here, "p_fa"] = m.predict_proba(players.loc[here, cols])[:, 1]
    return pairs.merge(players[["player_id", "season", "p_fa"]], on=["player_id", "season"], how="left")


VARIANTS = {
    # name: (features, min_train_season, needs free-agency columns[, model])
    "base": (FEATURES, None, False),
    "period only": (FEATURES, WIKI_FIRST, False),
    "period + P(FA)": (FEATURES + WIKI_FEATURES, WIKI_FIRST, True),
    "period + listed (leaky bound)": (FEATURES + ORACLE_FEATURES, WIKI_FIRST, True),
    "base + P(FA)": (FEATURES + WIKI_FEATURES, None, True),
    "ranker": (FEATURES, None, False, "ranker"),
    "two-stage": (FEATURES, None, False, "two-stage"),
    "two-stage + destination": (FEATURES + DESTINATION_FEATURES, None, False, "two-stage"),
}


def calibrate_moves(p, min_history=MIN_HISTORY):
    """Rescale each player's P(T acquires him) so that P(he moves at all) --
    the sum over teams -- matches how often players like him really moved.

    Tested and off (CALIBRATE_MOVES). It was motivated by P(moves) looking
    twice too high for stars (0.14 vs 0.07 on 1999-2017), but that came from
    the check, not the model: it included 1999-2001, when every pair gets
    NO_HISTORY_RATE, and scored "moved" as rubric.player_status does, which
    misses mid-season trades from a team he also played for. Against the
    model's own arrival label, 2002-2017, P(moves) by BPM tier is already
    close: bench 0.31 vs 0.30, rotation 0.34 vs 0.33, starters 0.26 vs 0.25,
    fringe all-stars 0.18 vs 0.17, stars 0.12 vs 0.085 (n = 130). This
    calibration overcorrects (starters 0.21, stars 0.06), raises log loss
    0.5685 -> 0.5763, and ranks real arrivals lower (within-list percentile
    0.390 -> 0.398).

    Per season t, a logistic regression of "moved" (any arrival label) on
    logit P(moves), the player's BPM percentile in the season pool, its
    square, and their product, fit on seasons < t (labels known by then;
    min_history seasons needed, else left as is). Each pair's p is
    multiplied by calibrated / raw P(moves), so P(T | he moves) -- where he
    goes -- is unchanged.
    """
    from sklearn.linear_model import LogisticRegression
    key = ["season", "player_id"]
    g = p.groupby(key)
    d = pd.concat([g["p_acq"].sum().rename("pm"), g["acquired"].max().rename("y"),
                   g["bpm"].first()], axis=1).reset_index()
    d["q"] = d.groupby("season")["bpm"].rank(pct=True).fillna(0.5)
    pm = d["pm"].clip(1e-4, 1 - 1e-4)
    lp = np.log(pm / (1 - pm))
    X = np.column_stack([lp, d["q"], d["q"] ** 2, lp * d["q"]])
    cal = d["pm"].to_numpy(float).copy()
    season = d["season"].to_numpy()
    for t in np.unique(season):
        train = (season < t) & (d["y"].to_numpy() >= 0)
        if len(np.unique(season[train])) < min_history:
            continue
        m = LogisticRegression(C=1e6, max_iter=2000).fit(X[train], d["y"].to_numpy()[train])
        cal[season == t] = m.predict_proba(X[season == t])[:, 1]
    factor = pd.Series(cal / d["pm"].to_numpy(float), index=pd.MultiIndex.from_frame(d[key]))
    out = p.copy()
    out["p_acq"] = (out["p_acq"] * factor.reindex(pd.MultiIndex.from_frame(out[key])).to_numpy()).clip(
        1e-6, 0.99)
    return out


def p_acq_table(paths, ptabs=None, variant="base", calibrate=None):
    """(season, TeamId, player_id) -> p_acq, for impact_recs and evaluate_recs.
    calibrate (default CALIBRATE_MOVES): apply calibrate_moves()."""
    features, since, needs_fa, *model = VARIANTS[variant]
    pairs = build_pairs(paths, ptabs)
    if needs_fa:
        pairs = add_free_agency(pairs)
    fn = MODELS[model[0]] if model else fit_predict
    p = fn(pairs, features, since)
    if CALIBRATE_MOVES if calibrate is None else calibrate:
        p = calibrate_moves(p)
    return p.set_index(["season", "TeamId", "player_id"])["p_acq"]


def wiki_compare(pairs, start, end, rng_seed=0):
    """Every VARIANTS entry scored on the same seasons [start, end], with a
    team-season bootstrap of the mean-percentile difference vs 'base'."""
    pairs = add_free_agency(pairs)
    window = (pairs["season"] >= start) & (pairs["season"] <= end) & (pairs["acquired"] >= 0)
    scored_v, rows = {}, {}
    for name, (features, since, _, *_m) in list(VARIANTS.items())[:5]:
        p = fit_predict(pairs, features, since)[window]
        scored_v[name] = p
        rows[name] = summary(p)
    print(f"Seasons {start}-{end}: {int(scored_v['base']['acquired'].sum())} arrivals, "
          f"{int(window.sum())} pairs\n")
    print(pd.DataFrame(rows).T[["mean pct", "AUC", "top 10", "top 50", "log loss",
                               "base log loss"]].round(4).to_string())

    def pct(p):
        g = p.groupby(["season", "TeamId"])["p_acq"]
        r = g.rank(ascending=False, method="average")
        return ((r - 0.5) / g.transform("size"))[p["acquired"] == 1]
    ref = pct(scored_v["base"])
    team = (scored_v["base"].loc[ref.index, "season"].astype(str) + "_"
            + scored_v["base"].loc[ref.index, "TeamId"].astype(str)).to_numpy()
    codes, uniq = pd.factorize(team)
    groups = [np.flatnonzero(codes == c) for c in range(len(uniq))]
    rng = np.random.default_rng(rng_seed)
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
             for _ in range(1000)]
    print("\nMean percentile vs base (negative = arrivals ranked higher), 95% CI:")
    for name, p in scored_v.items():
        if name == "base":
            continue
        d = (ref - pct(p).reindex(ref.index)).to_numpy()
        lo, hi = np.percentile([d[b].mean() for b in boots], [2.5, 97.5])
        print(f"  {name:<32} {-d.mean():+.4f}  [{-hi:+.4f}, {-lo:+.4f}]")

    p = scored_v["period + P(FA)"]
    ok = p["p_fa"].notna()
    pl = p[ok].drop_duplicates(["player_id", "season"])
    from sklearn.metrics import roc_auc_score
    print(f"\nP(free agent) held out, {pl['season'].min()}-{pl['season'].max()}: "
          f"AUC {roc_auc_score(pl['fa_listed'], pl['p_fa']):.3f}, "
          f"base rate {pl['fa_listed'].mean():.3f}, mean p_fa {pl['p_fa'].mean():.3f}")
    arrivals = p[p["acquired"] == 1]
    print(f"Arrivals that were listed free agents: {arrivals['fa_listed'].mean():.1%}; "
          f"non-arrivals: {p.loc[p['acquired'] == 0, 'fa_listed'].mean():.1%}")


def within_list(p, col="p_acq"):
    """How the model ranks real arrivals inside each team-season's list, which
    is how impact_recs uses it. Pooled AUC would also credit separating
    players who move anywhere from those who stay, without ranking any one
    team's candidates well.

      mean pct   arrivals' mean percentile within their team's list (0 = top)
      AUC        per team-season AUC, averaged over arrivals (= 1 - mean pct
                 up to ties)
      top 10/50  share of arrivals ranked in the top 10 / 50
    """
    g = p.groupby(["season", "TeamId"])[col]
    rank = g.rank(ascending=False, method="average")
    n = g.transform("size")
    pos = p["acquired"] == 1
    pct = ((rank - 0.5) / n)[pos]
    n_pos = pos.groupby([p["season"], p["TeamId"]]).transform("sum")[pos]
    # Exact grouped AUC from the arrivals' ranks among non-arrivals.
    up = g.rank(ascending=True, method="average")[pos]
    auc = (up - (n_pos + 1) / 2) / (n[pos] - n_pos)
    return {"mean pct": pct.mean(), "AUC": auc.mean(),
            "top 10": (rank[pos] <= 10).mean(), "top 50": (rank[pos] <= 50).mean()}


def log_loss(y, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))


def era_of_season(t):
    return "1999-04" if t < 2005 else "2005-10" if t < 2011 else "2011-16" if t < 2017 else "2017-"


def scored(pairs, features=None):
    """Held-out predictions on the seasons that have a model."""
    p = fit_predict(pairs, features)
    ok = (p["acquired"] >= 0) & (p["season"] >= p["season"].min() + MIN_HISTORY)
    return p[ok]


def summary(p):
    from sklearn.metrics import roc_auc_score
    y = p["acquired"].to_numpy()
    return {**within_list(p),
            "log loss": log_loss(y, p["p_acq"].to_numpy()),
            "base log loss": log_loss(y, p["p_base"].to_numpy()),
            "pooled AUC": roc_auc_score(y, p["p_acq"])}


GROUPS = {
    "candidate": ["value", "net_impact", "bpm", "age", "experience", "cap_pct", "c_share",
                  "off_poss", "traded", "from_net", "pos_index"],
    "team": ["team_net", "room_pct", "payroll_pct"],
    "pair": ["salary_fit", "upgrade", "need", "former"],
    "contract": CONTRACT_FEATURES,
    "destination": DESTINATION_FEATURES,
}




def _calibrate(train_score, train_y, score):
    """Platt scaling: P = sigmoid(a * score + b), fit on training predictions."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(np.asarray(train_score).reshape(-1, 1), train_y)
    return lr.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]


def fit_predict_ranker(pairs, features=None, min_train_season=None):
    """LambdaMART (XGBoost rank:ndcg) grouped by team-season: trained on the
    within-list choice impact_recs makes, then Platt-calibrated on its own
    training predictions so log P keeps the scale REALISM_WEIGHT was tuned on.
    Same expanding window as fit_predict."""
    from xgboost import XGBRanker
    features = FEATURES if features is None else features
    pairs = pairs.sort_values(["season", "TeamId"]).reset_index(drop=True)
    pairs["p_acq"] = np.nan
    pairs["p_base"] = np.nan
    labelled = pairs["acquired"] >= 0
    if min_train_season is not None:
        labelled &= pairs["season"] >= min_train_season
    # Dense group numbers: XGBoost keeps qid as 32-bit, so season * 1e10 + TeamId overflows.
    group = pd.Series(pd.factorize(list(zip(pairs["season"], pairs["TeamId"])))[0], index=pairs.index)
    for t in sorted(pairs["season"].unique()):
        train = labelled & (pairs["season"] < t)
        here = pairs["season"] == t
        base = pairs.loc[train, "acquired"].mean() if train.any() else np.nan
        pairs.loc[here, "p_base"] = base
        if pairs.loc[train, "season"].nunique() < MIN_HISTORY:
            pairs.loc[here, "p_acq"] = base
            continue
        X, y, q = pairs.loc[train, features], pairs.loc[train, "acquired"], group[train]
        m = XGBRanker(objective="rank:ndcg", lambdarank_pair_method="topk",
                      lambdarank_num_pair_per_sample=16, n_estimators=300, learning_rate=0.05,
                      max_depth=4, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=1.0, tree_method="hist", random_state=0)
        m.fit(X, y, qid=q)
        pairs.loc[here, "p_acq"] = _calibrate(m.predict(X), y, m.predict(pairs.loc[here, features]))
    for col in ("p_acq", "p_base"):
        pairs[col] = pairs[col].fillna(NO_HISTORY_RATE).clip(lower=1e-6)
    return pairs


def fit_predict_two_stage(pairs, features=None, min_train_season=None, stage2="hgb"):
    """P(c joins T) = P(c moves) x P(T | c moves).

    Stage 1, player level: does he join any new team for t+1 (candidate and
    contract features; team features would be noise here). Stage 2, movers
    only: which of the other teams, from every feature; scores normalised
    to sum to 1 over each mover's teams. Fitting stage 2 on movers alone
    gives team-side features a test the pooled model cannot: there, the
    question "does he move at all" dominates every split. Columns p_move and
    p_dest are kept for diagnostics.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    features = FEATURES if features is None else features
    player_feats = [c for c in features if c in GROUPS["candidate"] + GROUPS["contract"]]
    pairs = pairs.copy()
    labelled = pairs["acquired"] >= 0
    if min_train_season is not None:
        labelled &= pairs["season"] >= min_train_season
    key = ["player_id", "season"]
    players = pairs.groupby(key, as_index=False).agg(
        moved=("acquired", "max"), **{c: (c, "first") for c in player_feats})
    players["labelled"] = labelled.groupby([pairs["player_id"], pairs["season"]]).all() \
        .reindex(pd.MultiIndex.from_frame(players[key])).to_numpy()
    mover = pairs.join(players.set_index(key)["moved"].rename("is_mover"), on=key)["is_mover"] == 1
    pairs["p_move"] = np.nan
    pairs["p_dest"] = np.nan
    pairs["p_base"] = np.nan

    def hgb(leaf):
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                              min_samples_leaf=leaf, l2_regularization=1.0,
                                              random_state=0)
    for t in sorted(pairs["season"].unique()):
        here = pairs["season"] == t
        tr1 = players["labelled"] & (players["season"] < t)
        tr2 = labelled & mover & (pairs["season"] < t)
        base = pairs.loc[labelled & (pairs["season"] < t), "acquired"].mean()
        pairs.loc[here, "p_base"] = base
        if players.loc[tr1, "season"].nunique() < MIN_HISTORY:
            continue
        m1 = hgb(50).fit(players.loc[tr1, player_feats], players.loc[tr1, "moved"])
        ph = players[players["season"] == t]
        pm = pd.Series(m1.predict_proba(ph[player_feats])[:, 1],
                       index=pd.MultiIndex.from_frame(ph[key]))
        pairs.loc[here, "p_move"] = pm.reindex(pd.MultiIndex.from_frame(pairs.loc[here, key])).to_numpy()
        m2 = hgb(100).fit(pairs.loc[tr2, features], pairs.loc[tr2, "acquired"])
        raw = pd.Series(m2.predict_proba(pairs.loc[here, features])[:, 1], index=pairs.index[here])
        pairs.loc[here, "p_dest"] = raw / raw.groupby([pairs.loc[here, "player_id"]]).transform("sum")
    pairs["p_acq"] = pairs["p_move"] * pairs["p_dest"]
    for col in ("p_acq", "p_base"):
        pairs[col] = pairs[col].fillna(NO_HISTORY_RATE).clip(lower=1e-6)
    return pairs


MODELS = {"hgb": fit_predict, "ranker": fit_predict_ranker, "two-stage": fit_predict_two_stage}


def model_compare(pairs, start=None, end=None, rng_seed=0):
    """The current classifier, the LambdaMART ranker and the two-stage model on
    the same seasons, with a team-season bootstrap of the mean-percentile
    difference vs the current model, plus the two stages' own diagnostics."""
    from sklearn.metrics import roc_auc_score
    lo = pairs["season"].min() + MIN_HISTORY if start is None else start
    hi = pairs["season"].max() if end is None else end
    window = (pairs["season"] >= lo) & (pairs["season"] <= hi) & (pairs["acquired"] >= 0)
    out = {}
    for name, fn in MODELS.items():
        p = fn(pairs)
        out[name] = p.set_index(["season", "TeamId", "player_id"]).loc[
            pd.MultiIndex.from_frame(pairs.loc[window, ["season", "TeamId", "player_id"]])].reset_index()
    print(f"Seasons {lo}-{hi}: {int(out['hgb']['acquired'].sum())} arrivals, {int(window.sum())} pairs\n")
    print(pd.DataFrame({k: summary(v) for k, v in out.items()}).T[
        ["mean pct", "AUC", "top 10", "top 50", "log loss", "base log loss"]].round(4).to_string())
    _bootstrap_vs(out, "hgb", rng_seed)

    ts = out["two-stage"]
    pl = ts.drop_duplicates(["player_id", "season"])
    moved = ts.groupby(["player_id", "season"])["acquired"].max()
    pl = pl.join(moved.rename("moved"), on=["player_id", "season"])
    print(f"\nStage 1, P(moves anywhere), player level: AUC {roc_auc_score(pl['moved'], pl['p_move']):.3f}"
          f" (base rate {pl['moved'].mean():.3f})")
    mv = ts[ts.join(moved.rename("m"), on=["player_id", "season"])["m"] == 1]
    g = mv.groupby(["player_id", "season"])["p_dest"]
    r = g.rank(ascending=False, method="average")[mv["acquired"] == 1]
    n = g.transform("size")[mv["acquired"] == 1]
    print(f"Stage 2, P(team | moves), among movers' {n.mean():.0f} candidate teams: "
          f"true team's mean rank {r.mean():.1f} (random {((n + 1) / 2).mean():.1f}), "
          f"top-3 {(r <= 3).mean():.1%} (random {(3 / n).mean():.1%})")
    return out


def _bootstrap_vs(scored_v, ref_name, rng_seed=0):
    def pct(p):
        g = p.groupby(["season", "TeamId"])["p_acq"]
        r = g.rank(ascending=False, method="average")
        return pd.Series(((r - 0.5) / g.transform("size")).to_numpy(),
                         index=pd.MultiIndex.from_frame(p[["season", "TeamId", "player_id"]]))[
            (p["acquired"] == 1).to_numpy()]
    ref = pct(scored_v[ref_name])
    team = (ref.index.get_level_values(0).astype(str) + "_" + ref.index.get_level_values(1).astype(str))
    codes, uniq = pd.factorize(team)
    groups = [np.flatnonzero(codes == c) for c in range(len(uniq))]
    rng = np.random.default_rng(rng_seed)
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
             for _ in range(1000)]
    print(f"\nMean percentile vs {ref_name} (negative = arrivals ranked higher), 95% CI:")
    for name, p in scored_v.items():
        if name == ref_name:
            continue
        d = (ref - pct(p).reindex(ref.index)).to_numpy()
        lo, hi = np.percentile([d[b].mean() for b in boots], [2.5, 97.5])
        print(f"  {name:<32} {-d.mean():+.4f}  [{-hi:+.4f}, {-lo:+.4f}]")


def report(pairs, ablate=True):
    p = scored(pairs)
    print(f"{int(p['acquired'].sum())} arrivals among {len(p)} pairs, seasons "
          f"{p['season'].min()}-{p['season'].max()} (each scored by a model fit on earlier ones)\n")
    rows = {"all features": summary(p)}
    if ablate:
        for name, cols in GROUPS.items():
            rows[f"without {name}"] = summary(scored(pairs, [c for c in FEATURES if c not in cols]))
        extra = [c for c in FEATURES if not any(c in g for g in GROUPS.values())]
        if extra:
            rows["without new (" + ", ".join(extra) + ")"] = summary(
                scored(pairs, [c for c in FEATURES if c not in extra]))
    print("Within team-season lists (what the recommender uses), then calibration:")
    print(pd.DataFrame(rows).T[["mean pct", "AUC", "top 10", "top 50", "log loss",
                               "base log loss", "pooled AUC"]].round(4).to_string())

    print("\nCalibration by era (held-out): mean predicted vs actual move rate")
    era = p["season"].map(era_of_season)
    cal = pd.DataFrame({"predicted": p.groupby(era)["p_acq"].mean(),
                        "actual": p.groupby(era)["acquired"].mean(),
                        "pairs": p.groupby(era).size()})
    cal["ratio"] = cal["predicted"] / cal["actual"]
    print(cal.round(4).to_string())
    dec = pd.qcut(p["p_acq"].rank(method="first"), 10, labels=False)
    print("\nBy predicted decile: mean predicted vs actual")
    print(pd.DataFrame({"predicted": p.groupby(dec)["p_acq"].mean(),
                        "actual": p.groupby(dec)["acquired"].mean()}).round(4).T.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=os.path.join("data", "lineup_panel.csv"))
    ap.add_argument("--impact", default=os.path.join("data", "player_impact.csv"))
    ap.add_argument("--fit", default=os.path.join("data", "archetype_main.csv"))
    ap.add_argument("--players", default=os.path.join("data", "player_seasons.csv"))
    ap.add_argument("--teams", default=os.path.join("data", "team_style.csv"))
    ap.add_argument("--salaries", default=os.path.join("data", "salaries.csv"))
    ap.add_argument("--payroll", default=os.path.join("data", "team_payroll.csv"))
    ap.add_argument("--rosters", default=os.path.join("data", "team_salaries.csv"))
    ap.add_argument("--career", default=os.path.join("data", "Player Career Info.csv"))
    ap.add_argument("--draft", default=os.path.join("data", "Draft Pick History.csv"))
    ap.add_argument("--adv", default=os.path.join("data", "Advanced.csv"))
    ap.add_argument("--wiki-compare", type=int, nargs=2, metavar=("START", "END"),
                    help="score every VARIANTS entry on seasons START-END (Wikipedia labels)")
    ap.add_argument("--model-compare", action="store_true",
                    help="current classifier vs LambdaMART ranker vs two-stage, same seasons")
    args = ap.parse_args()
    if args.model_compare:
        model_compare(build_pairs(args))
        return 0
    if args.wiki_compare:
        wiki_compare(build_pairs(args), *args.wiki_compare)
        return 0
    report(build_pairs(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
