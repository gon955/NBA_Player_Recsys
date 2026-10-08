"""Impact-based recommendations: who would make this team-season better?

Replaces LightFM's roster co-occurrence score. For a team-season and a
candidate from the same season (the app's existing framing):

  value(p)    net impact (impact_model.py) + the net main effect of p's
              archetype (fit_model.py) -- e.g. the rim-protector correction.
  slot        the candidate takes the minutes of one rotation player (>= 15%
              of the team's floor time) whose position is within one spot.
  gain        share(r) * (value(c) - value(r))       points per 100, team-level
  fit         change in archetype-pair terms over the lineups r actually
              played in, with c's archetype swapped in. Steps 3-4 found pair
              effects at the edge of detection, so this is reported and added,
              but it is a tie-breaker by construction (|fit| << |gain|).
  score       max over eligible r of gain + fit; `replaces` names that r.

Scoring alternatives (--scoring; evaluate_recs.py --scoring-grid), tried because
the team-level value test credited gain-over-r with almost nothing:
  value      value(c) alone      minutes   share_t(c) * value(c)
  displace   share_t(c) * (value(c) - minutes-weighted value at his position)
Scored on c alone, a team could "swap" its own star for an equal one (Embiid
for Giannis is salary- and surplus-legal), which made `value` look best (team
R^2 +0.062 vs +0.012). Requiring an upgrade (value(c) > value(r)) removes that,
and then all three tie random on realism (~0.50) with R^2 gains of 0.000-0.006
-- while ruling out 80% of actual arrivals, which by this value measure are
mostly not upgrades over anyone at their position. `replace` stays the default.

Salary (fetch_salaries.py). Unrestricted, every team's list converges on the
season's best players, whatever they cost. With salary on, a swap must also be
one the team could make:
  outgoing    salary(r) + the team's FILLER largest non-rotation contracts
  incoming    salary(c) <= max(MATCH_PCT * outgoing + pad,  outgoing + room)
              The first term is the over-the-cap trade rule (125% + $100k,
              $250k from the 2023 CBA on); the second is a team under the cap
              absorbing salary into its room = max(0, cap - payroll).
  Candidates with no known salary are dropped; a rotation player with none
  sends out $0.

Value cap (--max-gap). Salary matching alone lets Cleveland "trade" Ricky
Davis for Dirk Nowitzki, and makes every star on a rookie deal reachable by
everyone. Teams trade on surplus -- value beyond what the contract normally
buys -- so a swap is also ruled out when
  surplus(c) - surplus(r) > max_gap        points per 100
  surplus(p)  value(p) - (a_t + b_t * salary(p) / cap_t), the line fit each
              season by possession-weighted least squares over players with
              >= MARKET_MIN_POSS (season-t data only; a monotone fit chased
              the single highest salary, zeroing Garnett's '07 surplus).
Garnett for Al Jefferson ('07) is a gap of ~0.7; Dirk for Ricky Davis ('03)
is ~6. Real trades close gaps with picks and prospects this cannot see, so
the tolerance is tuned on evaluate_recs.py's realism test, not set by hand:
`--gap-grid 0.5 1 2 3 5` moved the headline percentile from 0.524 (no cap)
to 0.490 / 0.492 / 0.503 / 0.510 / 0.517; 1.0 keeps nearly all of 0.5's gain
while ruling out 21% of actual arrivals instead of 26%.
It binds trades only: pending free agents are exempt (below).

Free agency (opt-in, --free-agency; off by default -- see the caveat below). A pending free agent is not traded
for; he signs. So he skips salary matching and the value cap, and instead
  fa_cost     his market price for t+1 (fa_prices): next-season cap share
              regressed on value and current cap share, fit on earlier
              seasons' pending free agents who signed, clipped to
              [FA_MIN_PCT, FA_MAX_PCT]. Running the surplus line backwards
              instead priced +2..+4 players at 32% of the cap against an
              actual 16% (mean error 0.062 of cap vs 0.037).
  budget      max(cap_{t+1} - salary still committed, MLE_{t+1}): offseason
              room once the team's own pending free agents come off its books,
              or the mid-level exception every team has regardless.
  reachable   fa_cost <= budget.
Contract status itself is a proxy (pending_fa): no accessible source lists
contract end dates or re-signings for 1999-2025 (BBRef's transaction log omits
most re-signings; Pro Sports Transactions, RealGM and Spotrac refuse scripted
requests). A contract is taken to end after t when the player's t+1 salary is
outside FA_BAND x his t salary, or he has none -- a new deal, whether he
re-signs or moves, breaks the fixed-raise schedule. That reads t+1 data, but
the same way for players who stay and players who leave, so it says "his
contract was up", not "he moved". Against BBRef's transaction labels it flags
83% of free-agent signings and 41% of traded players (AUC 0.80).

An *extension* breaks the schedule too, and extensions go mostly to stars
(Durant '10 and Cunningham '25 on rookie deals; Harden '16, Mitchell '25 as
veterans). Unchecked, 99% of recommendations arrived "via free agency" and the
stars came back. Two corrections:
  rookie scale  first-round picks are never flagged through draft year +
                ROOKIE_YEARS: on the rookie deal, then restricted free agents
                whose teams match offers.
  p_avail       (availability) P(a flagged player actually changes teams),
                gradient-boosted trees on age, value, cap share, experience,
                floor-time share, whether he was traded mid-season, and his
                team's net rating, fit on earlier seasons' flagged players
                only. Held-out AUC 0.63 (a logistic fit: 0.56 -- stars stay far
                more often than a linear term in value allows: 14% of flagged
                +4s moved vs 38% overall). Weak, but it is the signal there is. A
                free agent's score is gain * p_avail: the expected gain of
                pursuing him, so a star likely to re-sign or extend ranks
                below a lesser player likely to leave. This deliberately ignores the CBA's finer tiers (150-200%
  matching for small salaries, apron limits) and sign-and-trades; it is a
  "could this work on salary at all" screen, not a trade machine.

Usage:
    python impact_recs.py              # write data/recs_impact.csv (+ _open)
    python impact_recs.py --compare    # vs. LightFM (data/recs_lightfm.csv)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import impact_model as im
from fetch_salaries import CAP, MLE

SLOTS = im.SLOTS
K = 10
MIN_SHARE = 0.15        # rotation players a candidate can replace
MIN_CAND_POSS = 1500    # offensive possessions; below this impact is mostly prior
POS_WINDOW = 1.0        # |pos_index difference| allowed, 1 = PG ... 5 = C
MATCH_PCT = 1.25        # incoming salary per $ outgoing, over the cap
FILLER = 0              # non-rotation contracts a team may add to the outgoing side
MAX_GAP = 1.0           # surplus(c) - surplus(r) allowed, pts/100; None = no value cap
SCORINGS = ("replace", "value", "minutes", "displace")
SCORING = "replace"
MARKET_MIN_POSS = 500   # players the salary -> value line is fit on
FA_BAND = (0.97, 1.13)  # t+1 / t salary on one contract; outside it, a new deal
FA_MIN_PCT, FA_MAX_PCT = 0.01, 0.35     # a free agent's price, as share of the cap
ROOKIE_YEARS = 4        # first-round rookie scale seasons; RFA right after
AVAIL_MIN_HISTORY = 2   # seasons of flagged players needed before p_avail is fit

# Franchise continuity where the NBA's team id does not provide it: Charlotte's
# 1999-2002 Hornets carry 1610612766 (the history went back to Charlotte),
# while the same roster plays 2003 in New Orleans as 1610612740. Keyed by the
# t+1 season. Seattle -> OKC, New Jersey -> Brooklyn and Vancouver -> Memphis
# keep their ids.
PREDECESSOR = {(2003, 1610612740): 1610612766}


def as_of_t(df):
    """Rows keyed (t, TeamId of t+1) -> the same franchise's season-t TeamId."""
    key = list(zip(df["season"] + 1, df["TeamId"]))
    return df.assign(TeamId=[PREDECESSOR.get(k, k[1]) for k in key])


def match_pad(season):
    return 250_000 if season >= 2024 else 100_000


def load(paths):
    panel = pd.read_csv(paths.panel, usecols=SLOTS + ["season", "TeamId", "OffPoss", "DefPoss"])
    impact = pd.read_csv(paths.impact)
    coef = pd.read_csv(paths.fit).fillna({"b": ""})
    players = pd.read_csv(paths.clusters, usecols=[
        "player_id", "season", "player", "cluster_label", "pos_index", "age", "bpm"])
    teams = pd.read_csv(paths.teams, usecols=["season", "TeamId", "user_id", "TeamAbbreviation",
                                              "net_rtg"])
    salaries = pd.read_csv(paths.salaries, usecols=["player_id", "season", "salary"])
    payroll = pd.read_csv(paths.payroll)
    rosters = pd.read_csv(paths.rosters)
    return panel, impact, coef, players, teams, salaries, (payroll, rosters)


def pending_fa(salaries, draft=None):
    """(player_id, season) -> True if his contract looks to end after that season.

    `draft` (Draft Pick History.csv) exempts first-round picks through draft
    year + ROOKIE_YEARS.
    """
    s = salaries.set_index(["player_id", "season"])["salary"]
    nxt = salaries.assign(season=salaries["season"] - 1).set_index(["player_id", "season"])["salary"]
    ratio = nxt.reindex(s.index) / s.where(s > 0)
    flag = ~ratio.between(*FA_BAND)          # NaN (no t+1 salary) -> True
    if draft is not None:
        first = draft[(draft["lg"] == "NBA") & (draft["round"] == 1)]
        until = first.drop_duplicates("player_id").set_index("player_id")["season"] + ROOKIE_YEARS
        until = until.reindex(s.index.get_level_values("player_id")).to_numpy()
        flag &= ~(s.index.get_level_values("season").to_numpy() <= until)   # NaN -> False
    return flag[s.index.get_level_values("season") < max(CAP)]


def memberships(panel):
    """(season, TeamId, player_id, share) for everyone who appears in a lineup."""
    return pd.concat([rotation(df)[0].assign(season=season)
                      for season, df in panel.groupby("season")], ignore_index=True)


def availability(ptabs, member, debut, teams):
    """Add p_avail to every table in {season: player_table} with flagged players.

    Label for season s: a flagged player (>= MARKET_MIN_POSS) appears in s+1
    and on none of his season-s teams. Retiring counts as not available.
    Fit on seasons < t only; until AVAIL_MIN_HISTORY seasons exist, the pooled
    rate so far (0.5 with none).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    on = member[member["share"] > 0].groupby(["player_id", "season"])["TeamId"].apply(set)
    nxt = as_of_t(member[member["share"] > 0].assign(season=member["season"] - 1)) \
        .groupby(["player_id", "season"])["TeamId"].apply(set)
    main_team = member.sort_values("share").drop_duplicates(["player_id", "season"], keep="last") \
        .set_index(["player_id", "season"])["TeamId"]
    share = member.groupby(["player_id", "season"])["share"].sum()
    n_teams = member[member["share"] > 0].groupby(["player_id", "season"])["TeamId"].nunique()
    net = teams.set_index(["season", "TeamId"])["net_rtg"]
    cols = ["age", "value", "cap_pct", "experience", "team_net", "share", "traded"]

    def features(t, season):
        idx = pd.MultiIndex.from_arrays([t.index, [season] * len(t)])
        team = main_team.reindex(idx).to_numpy()
        f = pd.DataFrame({
            "age": t["age"].to_numpy(), "value": t["value"].to_numpy(),
            "cap_pct": t["cap_pct"].to_numpy(),
            "experience": season - debut.reindex(t.index).to_numpy(),
            "team_net": net.reindex(pd.MultiIndex.from_arrays([[season] * len(t), team])).to_numpy(),
            "share": share.reindex(idx).to_numpy(),
            "traded": (n_teams.reindex(idx).fillna(1).to_numpy() > 1).astype(float),
        }, index=t.index)
        return f.fillna(f.median())

    history = []
    for season in sorted(ptabs):
        t = ptabs[season]
        f = features(t, season)
        if len(history) >= AVAIL_MIN_HISTORY:
            h = pd.concat(history)
            model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                   min_samples_leaf=40, random_state=0)
            model.fit(h[cols], h["moved"])
            t["p_avail"] = model.predict_proba(f[cols])[:, 1]
        else:
            t["p_avail"] = pd.concat(history)["moved"].mean() if history else 0.5
        flagged = t["pending_fa"] & (t["off_poss"] >= MARKET_MIN_POSS)
        idx = pd.MultiIndex.from_arrays([t.index[flagged], [season] * int(flagged.sum())])
        before, after = on.reindex(idx), nxt.reindex(idx)
        moved = [isinstance(a, set) and not (a & b) for a, b in zip(after, before)]
        if season + 1 in ptabs or season + 1 < max(CAP):
            history.append(f[flagged].assign(moved=np.array(moved, dtype=float)))
    return ptabs


def team_room(teams, books, flags=None):
    """Per (season, TeamId): in-season cap room, and (given `flags`) the offseason
    free-agent budget. Joins pbpstats' abbreviations to BBRef's."""
    from lineup_features import bbref_team
    payroll, rosters = books
    t = teams.assign(team=[bbref_team(a, s) for a, s in
                           zip(teams["TeamAbbreviation"], teams["season"])])
    t = t.merge(payroll, on=["season", "team"], how="left", validate="1:1")
    if t["payroll"].isna().any():
        raise SystemExit(f"no payroll for {t.loc[t['payroll'].isna(), ['season', 'team']].values[:5]}")
    t["room"] = (t["cap"] - t["payroll"]).clip(lower=0)
    if flags is not None:
        r = rosters.join(flags.rename("fa"), on=["player_id", "season"])
        kept = r[~r["fa"].fillna(True).astype(bool)].groupby(["season", "team"])["salary"].sum()
        t = t.join(kept.rename("committed"), on=["season", "team"]).fillna({"committed": 0.0})
        nxt = t["season"] + 1
        t["fa_budget"] = np.maximum(nxt.map(CAP) - t["committed"], nxt.map(MLE))
    return t[["season", "TeamId", "room"] + (["fa_budget"] if flags is not None else [])]


def effects(coef):
    """Archetype main effects (Series) and symmetric pair matrix (DataFrame)."""
    main = coef[coef["kind"] == "main"].set_index("a")["net_effect"]
    pairs = coef[coef["kind"] == "pair"]
    names = sorted(set(pairs["a"]) | set(pairs["b"]))
    F = pd.DataFrame(0.0, index=names, columns=names)
    for a, b, v in pairs[["a", "b", "net_effect"]].itertuples(index=False):
        F.loc[a, b] = F.loc[b, a] = v
    return main, F


def player_table(season, impact, players, main, salaries, flags=None):
    est = impact[impact["season"] == season].set_index("player_id")
    meta = players[players["season"] == season].set_index("player_id")
    sal = salaries[salaries["season"] == season].set_index("player_id")["salary"]
    t = est[["net_impact", "o_impact", "d_impact", "off_poss"]].join(
        meta[["player", "cluster_label", "pos_index", "age", "bpm"]], how="left").join(sal)
    t["cluster_label"] = t["cluster_label"].fillna("unknown")
    t["value"] = t["net_impact"] + t["cluster_label"].map(main).fillna(0.0)
    t["cap_pct"] = t["salary"] / CAP[season]
    fit = t[(t["off_poss"] >= MARKET_MIN_POSS) & t["cap_pct"].notna()]
    b, a = np.polyfit(fit["cap_pct"], fit["value"], 1, w=np.sqrt(fit["off_poss"]))
    t["surplus"] = t["value"] - (a + b * t["cap_pct"])
    if flags is not None:
        t["pending_fa"] = flags.xs(season, level="season").reindex(t.index).fillna(False).astype(bool)
    return t


def fa_prices(ptabs, salaries):
    """Add fa_cost ($, for season t+1) to every table in {season: player_table}.

    Next-season cap share ~ value + current cap share, fit by least squares on
    the pending free agents of seasons before t who signed a t+1 contract
    (expanding window, so season t never sees its own outcomes). The first
    season has no history and is priced at its current cap share.
    """
    nxt = salaries.assign(season=salaries["season"] - 1).set_index(["player_id", "season"])["salary"]
    train = []
    for season in sorted(ptabs):
        t = ptabs[season]
        X = np.column_stack([np.ones(len(t)), t["value"], t["cap_pct"].fillna(0.0)])
        if train:
            h = pd.concat(train)
            beta = np.linalg.lstsq(h[["one", "value", "cap_pct"]], h["y"], rcond=None)[0]
            pct = X @ beta
        else:
            pct = t["cap_pct"].fillna(FA_MIN_PCT).to_numpy()
        t["fa_cost"] = np.clip(pct, FA_MIN_PCT, FA_MAX_PCT) * CAP[season + 1]
        done = t[t["pending_fa"] & (t["off_poss"] >= MARKET_MIN_POSS) & t["cap_pct"].notna()]
        y = nxt.reindex(pd.MultiIndex.from_arrays([done.index, [season] * len(done)])).to_numpy()
        train.append(pd.DataFrame({"one": 1.0, "value": done["value"].to_numpy(),
                                   "cap_pct": done["cap_pct"].to_numpy(),
                                   "y": y / CAP[season + 1]}).dropna())
    return ptabs


def rotation(df):
    """On-floor share per (TeamId, player), and per-player teammate archetype mix."""
    w = (df["OffPoss"] + df["DefPoss"]).to_numpy(float)
    long = pd.DataFrame({
        "TeamId": np.repeat(df["TeamId"].to_numpy(), 5),
        "player_id": df[SLOTS].to_numpy().ravel(),
        "w": np.repeat(w, 5),
        "lineup": np.repeat(np.arange(len(df)), 5),
    })
    team_w = df.assign(w=w).groupby("TeamId")["w"].sum()
    share = long.groupby(["TeamId", "player_id"])["w"].sum() / 5
    share = share / team_w.reindex(share.index.get_level_values(0)).to_numpy() * 5
    return share.rename("share").reset_index(), long, team_w


def teammate_mix(long, team_w, arche):
    """M[(team, r)][a]: possession-weighted count of archetype a beside r, / team poss."""
    long = long.assign(arch=long["player_id"].map(arche).fillna("unknown"))
    per_lineup = long.groupby(["lineup", "arch"]).size().unstack(fill_value=0)
    me = pd.get_dummies(long["arch"]).reindex(columns=per_lineup.columns, fill_value=0)
    beside = per_lineup.loc[long["lineup"]].to_numpy() - me.to_numpy()
    beside = pd.DataFrame(beside * long["w"].to_numpy()[:, None], columns=per_lineup.columns)
    beside[["TeamId", "player_id"]] = long[["TeamId", "player_id"]].to_numpy()
    M = beside.groupby(["TeamId", "player_id"]).sum()
    return M.div(team_w.reindex(M.index.get_level_values(0)).to_numpy(), axis=0)


def recommend_season(season, df, ptab, F, teams, room=None, filler=FILLER, k=K,
                     min_poss=MIN_CAND_POSS, max_gap=MAX_GAP, fa_budget=None, scoring=SCORING):
    """Top-k per team. `room` (cap room by TeamId) turns the salary screen on;
    `fa_budget` (by TeamId) lets ptab's pending free agents in on free-agent terms.
    `scoring` is one of SCORINGS; the salary and value screens decide who is
    eligible either way.

    k=None ranks every candidate instead, keeping those with no legal swap
    at score -inf (evaluate_recs.py needs where an actual acquisition fell).
    """
    share, long, team_w = rotation(df)
    M = teammate_mix(long, team_w, ptab["cluster_label"])
    M = M.reindex(columns=F.columns, fill_value=0.0)
    on_team = share.groupby("TeamId")["player_id"].apply(set)

    cands = ptab[ptab["off_poss"] >= min_poss]
    if room is not None:
        cands = cands[cands["salary"].notna()]
        sal = ptab["salary"]
        bench = share[share["share"] < MIN_SHARE].assign(salary=lambda d: d["player_id"].map(sal))
        filler_sal = bench.groupby("TeamId")["salary"].apply(
            lambda s: s.nlargest(filler).sum() if filler else 0.0)
    c_ids = cands.index.to_numpy()
    c_val = cands["value"].to_numpy()
    c_pos = cands["pos_index"].to_numpy()
    c_F = F.reindex(cands["cluster_label"]).fillna(0.0).to_numpy()    # (n_c, n_arch)
    # Floor-time share he carried in season t, across teams if traded.
    c_share = share.groupby("player_id")["share"].sum().clip(upper=1.0) \
        .reindex(c_ids).fillna(0.0).to_numpy()

    out = []
    for team_id, rot in share[share["share"] >= MIN_SHARE].groupby("TeamId"):
        rot = rot.set_index("player_id")
        r_ids = rot.index.to_numpy()
        r = ptab.reindex(r_ids)
        r_val = r["value"].fillna(0.0).to_numpy()
        r_pos = r["pos_index"].to_numpy()
        r_F = F.reindex(r["cluster_label"].fillna("unknown")).fillna(0.0).to_numpy()
        Mr = M.loc[[(team_id, p) for p in r_ids]].to_numpy()              # (n_r, n_arch)

        # fit[c, r] = sum_a M[r, a] * (F[c, a] - F[r, a])
        fit = c_F @ Mr.T - (Mr * r_F).sum(1)[None, :]
        pos_ok = np.abs(c_pos[:, None] - r_pos[None, :]) <= POS_WINDOW
        pos_ok |= np.isnan(c_pos)[:, None] | np.isnan(r_pos)[None, :]
        r_share = rot["share"].to_numpy()
        if scoring != "replace":
            # Scored on c alone, so the swap must be an upgrade, or a team
            # could "trade" its own star for an equal one (Embiid for Giannis).
            pos_ok = pos_ok & (c_val[:, None] > r_val[None, :])
        if scoring == "replace":
            gain = r_share[None, :] * (c_val[:, None] - r_val[None, :])
        else:
            if scoring == "value":
                cs = c_val
            elif scoring == "minutes":
                cs = c_share * c_val
            else:   # displace: vs the minutes-weighted value at his position
                w = pos_ok * r_share[None, :]
                disp = (w @ r_val) / np.where(w.sum(1) > 0, w.sum(1), np.nan)
                disp = np.where(np.isnan(disp), np.average(r_val, weights=r_share), disp)
                cs = c_share * (c_val - disp)
            gain = np.repeat(cs[:, None], len(r_ids), axis=1)
        ok = pos_ok
        fa_total = None
        if room is not None:
            outgoing = r["salary"].fillna(0.0).to_numpy() + filler_sal.get(team_id, 0.0)
            limit = np.maximum(MATCH_PCT * outgoing + match_pad(season),
                               outgoing + room.get(team_id, 0.0))
            ok = ok & (cands["salary"].to_numpy()[:, None] <= limit[None, :])
            if max_gap is not None:
                # NaN surplus (r with no salary) compares False: ruled out.
                gap = cands["surplus"].to_numpy()[:, None] - r["surplus"].to_numpy()[None, :]
                ok = ok & (gap <= max_gap)
            if fa_budget is not None:
                signable = (cands["pending_fa"].to_numpy()
                            & (cands["fa_cost"].to_numpy() <= fa_budget.get(team_id, 0.0)))
                p = cands["p_avail"].to_numpy() if "p_avail" in cands else np.ones(len(cands))
                fa_total = np.where(pos_ok & signable[:, None], p[:, None] * (gain + fit), -np.inf)
        total = np.where(ok, gain + fit, -np.inf)
        via_fa = np.zeros(total.shape, dtype=bool)
        if fa_total is not None:
            via_fa = fa_total > total
            total = np.maximum(total, fa_total)
        best = total.argmax(1)
        rows = np.arange(len(c_ids))

        rec = pd.DataFrame({
            "player_id": c_ids, "score": total[rows, best],
            "gain": gain[rows, best], "fit": fit[rows, best],
            "replaces": r_ids[best],
            "salary": cands["salary"].to_numpy() if "salary" in cands else np.nan,
            "replaces_salary": r["salary"].to_numpy()[best] if "salary" in r else np.nan,
            "via": np.where(via_fa[np.arange(len(c_ids)), best], "free agency", "trade")
            if fa_total is not None else "",
        })
        rec = rec[~rec["player_id"].isin(on_team.get(team_id, set()))]
        if k is None:
            top = rec.sort_values("score", ascending=False, kind="stable")
        else:
            top = rec[np.isfinite(rec["score"])].nlargest(k, "score")
        top = top.assign(TeamId=team_id, season=season)
        top["rank"] = np.arange(1, len(top) + 1)
        out.append(top)
    recs = pd.concat(out, ignore_index=True)
    recs = recs.merge(teams, on=["season", "TeamId"], how="left")
    recs["item_id"] = recs["player_id"] + "_" + recs["season"].astype(str)
    return recs


def build(paths, salary=True, filler=FILLER, max_gap=MAX_GAP, free_agency=False,
          scoring=SCORING):
    panel, impact, coef, players, teams, salaries, books = load(paths)
    main, F = effects(coef)
    draft = pd.read_csv(paths.draft, usecols=["season", "lg", "round", "player_id"])
    flags = pending_fa(salaries, draft) if salary and free_agency else None
    rooms = team_room(teams, books, flags) if salary else None
    ptabs = {s: player_table(s, impact, players, main, salaries, flags)
             for s in sorted(panel["season"].unique())}
    if flags is not None:
        fa_prices(ptabs, salaries)
        debut = pd.read_csv(paths.career, usecols=["player_id", "from"]) \
            .drop_duplicates("player_id").set_index("player_id")["from"]
        availability(ptabs, memberships(panel), debut, teams)
    out = []
    for season, ptab in ptabs.items():
        r = rooms[rooms["season"] == season].set_index("TeamId") if salary else None
        out.append(recommend_season(season, panel[panel["season"] == season], ptab, F,
                                    teams[["season", "TeamId", "user_id"]],
                                    r["room"] if salary else None, filler,
                                    max_gap=max_gap if salary else None,
                                    fa_budget=r["fa_budget"] if flags is not None else None,
                                    scoring=scoring))
    return pd.concat(out, ignore_index=True)[
        ["user_id", "rank", "item_id", "score", "gain", "fit", "replaces",
         "salary", "replaces_salary", "via"]]


def _jaccard_within_season(recs, rng, n_pairs=2000):
    """Mean overlap between two different teams' lists in the same season."""
    lists = recs.groupby("user_id")["item_id"].apply(set)
    season = lists.index.str.rsplit("_", n=1).str[1]
    vals = []
    for _, group in lists.groupby(season):
        g = group.to_list()
        for _ in range(n_pairs // 27):
            i, j = rng.choice(len(g), 2, replace=False)
            vals.append(len(g[i] & g[j]) / len(g[i] | g[j]))
    return float(np.mean(vals))


def compare(paths, examples):
    _, impact, _, players, _, salaries, _ = load(paths)
    salaries = salaries[salaries["season"] < max(CAP)]
    runs = {"LightFM (today)": pd.read_csv(paths.lightfm),
            "impact": pd.read_csv(paths.out_open),
            "impact + salary": pd.read_csv(paths.out)}
    cap = salaries.assign(cap=salaries["season"].map(CAP))
    meta = impact.merge(players, on=["player_id", "season"], how="left") \
        .merge(cap, on=["player_id", "season"], how="left")
    meta["cap_pct"] = meta["salary"] / meta["cap"]
    meta = meta.set_index(meta["player_id"] + "_" + meta["season"].astype(str))
    rng = np.random.default_rng(0)

    def describe(recs):
        m = meta.reindex(recs["item_id"])
        per_season = recs.assign(season=recs["user_id"].str.rsplit("_", n=1).str[1])
        top10_share = per_season.groupby("season")["item_id"].apply(
            lambda s: s.value_counts().head(10).sum() / len(s)).mean()
        return {
            "net impact (pts/100)": m["net_impact"].mean(),
            "BPM": m["bpm"].mean(),
            "age": m["age"].mean(),
            "salary, % of cap": 100 * m["cap_pct"].mean(),
            "share earning > 25% of cap": (m["cap_pct"] > 0.25).mean(),
            "share with < 1500 poss": (m["off_poss"].fillna(0) < MIN_CAND_POSS).mean(),
            "distinct players / season": per_season.groupby("season")["item_id"].nunique().mean(),
            "share of recs to top-10 players": top10_share,
            "overlap between 2 teams (Jaccard)": _jaccard_within_season(recs, rng),
        }

    print(pd.DataFrame({k: describe(v) for k, v in runs.items()}).round(3).to_string())

    sets = {k: v.groupby("user_id")["item_id"].apply(set) for k, v in runs.items()}
    for a, b in [("LightFM (today)", "impact + salary"), ("impact", "impact + salary")]:
        both = sets[a].to_frame("a").join(sets[b].to_frame("b"), how="inner")
        shared = both.apply(lambda r: len(r["a"] & r["b"]), axis=1)
        print(f"\n{a} vs {b}: {shared.mean():.2f} of 10 shared on average; "
              f"{(shared == 0).mean():.0%} of team-seasons share none")
    lists = runs["impact + salary"].groupby("user_id").size()
    print(f"Team-seasons with fewer than {K} affordable candidates: {(lists < K).sum()}")

    arch = pd.DataFrame({
        k: meta.reindex(v["item_id"])["cluster_label"].value_counts(normalize=True)
        for k, v in runs.items()
    } | {"league (>=1500 poss)": meta[meta["off_poss"] >= MIN_CAND_POSS]["cluster_label"]
         .value_counts(normalize=True)}).fillna(0).sort_values("impact + salary", ascending=False)
    print("\nArchetype share of recommendations:")
    print(arch.round(3).to_string())

    name = meta["player"].fillna(pd.Series(meta.index, index=meta.index)).to_dict()

    def money(x):
        return "?" if pd.isna(x) else f"${x / 1e6:.1f}M"

    for uid in examples:
        season = uid.rsplit("_", 1)[1]
        cols = [v[v["user_id"] == uid].sort_values("rank") for v in runs.values()]
        if cols[2].empty:
            continue
        print(f"\n{uid}")
        print(f"  {'LightFM (today)':<24}| {'impact':<24}| impact + salary: replaces, +pts/100")
        for (_, a), (_, b), (_, c) in zip(*(x.iterrows() for x in cols)):
            rep = name.get(f"{c['replaces']}_{season}", c["replaces"])
            print(f"  {str(name.get(a['item_id'], a['item_id']))[:23]:<24}| "
                  f"{str(name.get(b['item_id'], b['item_id']))[:23]:<24}| "
                  f"{name.get(c['item_id'], c['item_id'])} {money(c['salary'])} -> "
                  f"{rep} {money(c['replaces_salary'])}, {c['score']:+.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--filler", type=int, default=FILLER,
                    help="non-rotation contracts a team may add to the outgoing salary")
    ap.add_argument("--max-gap", type=float, default=MAX_GAP,
                    help="value cap: surplus(candidate) - surplus(replaced) allowed, pts/100")
    ap.add_argument("--lightfm", default=os.path.join("data", "recs_lightfm.csv"),
                    help="today's top-10s, dumped from backend/inference.py")
    ap.add_argument("--examples", nargs="*", default=[
        "Golden State Warriors_2016", "Denver Nuggets_2024",
        "Houston Rockets_2019", "Washington Wizards_2025", "Detroit Pistons_2004"])
    ap.add_argument("--panel", default=os.path.join("data", "lineup_panel.csv"))
    ap.add_argument("--impact", default=os.path.join("data", "player_impact.csv"))
    ap.add_argument("--fit", default=os.path.join("data", "archetype_fit.csv"))
    ap.add_argument("--clusters", default="master_clustered.csv")
    ap.add_argument("--teams", default=os.path.join("data", "team_style.csv"))
    ap.add_argument("--salaries", default=os.path.join("data", "salaries.csv"))
    ap.add_argument("--payroll", default=os.path.join("data", "team_payroll.csv"))
    ap.add_argument("--rosters", default=os.path.join("data", "team_salaries.csv"))
    ap.add_argument("--draft", default=os.path.join("data", "Draft Pick History.csv"))
    ap.add_argument("--career", default=os.path.join("data", "Player Career Info.csv"))
    ap.add_argument("--scoring", choices=SCORINGS, default=SCORING,
                    help="how a legal candidate is scored (see module docstring)")
    ap.add_argument("--free-agency", action="store_true",
                    help="let flagged pending free agents in on free-agent terms (experimental)")
    ap.add_argument("--out", default=os.path.join("data", "recs_impact.csv"))
    ap.add_argument("--out-open", default=os.path.join("data", "recs_impact_open.csv"),
                    help="the same ranking without the salary screen")
    args = ap.parse_args()

    if args.compare:
        compare(args, args.examples)
        return 0
    for salary, path in [(True, args.out), (False, args.out_open)]:
        recs = build(args, salary=salary, filler=args.filler, max_gap=args.max_gap,
                     free_agency=args.free_agency, scoring=args.scoring)
        recs.round(4).to_csv(path, index=False)
        print(f"Wrote {len(recs)} recommendations for {recs['user_id'].nunique()} "
              f"team-seasons to '{path}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
