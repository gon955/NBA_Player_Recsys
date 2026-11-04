"""Grade the recommender only on questions real data can answer.

There is no ground truth for what a team *should* have done, so real moves
are not used to say a recommendation was right (evaluate_recs.py's realism
and quality do that, and are kept for reference only). What data can answer:

  1 availability   Was the player we list actually available? Of each
                   system's top-K, the share who moved to another team in
                   t+1, stayed, or dropped out of the league / rotation;
                   against a random pool player; and `joined`, the share who
                   moved to that very team (destinations are close to
                   unpredictable, so this is small for every system). And how star-heavy the
                   lists are -- top STAR_PCT of the season's pool by BPM, a
                   box-score measure the recommender does not produce --
                   against the share of actual veteran arrivals who are
                   stars. Moving is evidence of availability, not of a
                   good move.
  3 value accuracy When we rate a player highly, is he? Among listed
                   players who moved anywhere (each player-season once,
                   however many teams list him), what they delivered in
                   t+1: minutes share at the new team(s), BPM, and net
                   impact. For systems that claim a value (impact_recs:
                   projected net impact; BPM: BPM), its rank correlation
                   with each outcome, and for impact_recs the bias,
                   predicted - realised net impact: selecting the top of a
                   noisy estimate overrates (the winner's curse). Moves
                   only decide whose outcome is visible; outcomes, not
                   moves, are what is judged. BPM t+1 is the independent
                   outcome; net impact comes from the same lineup model as
                   impact_recs' value and flatters it.
  4 specificity    Do teams get different lists? Mean Jaccard overlap of two
                   teams' top-K in the same season, distinct players per
                   list slot, and the share of slots taken by the season's
                   10 most-listed players. Descriptive: it has no right
                   answer (random lists score best), so read it against a
                   target, not as better/worse. Needs no outcome, so it is
                   also shown for the serving season.
  5 holdout        Decisions in seasons <= SPLIT are for tuning and the
                   default; `--holdout` reports seasons after it, and every
                   holdout run is appended to HOLDOUT_LOG with the settings
                   it graded, so how often it has been looked at is visible.
                   Settings chosen after looking at it are tuned on it.

Usage:
    python rubric.py              # tuning seasons (decisions 1999-SPLIT)
    python rubric.py --holdout    # SPLIT+1 onward; logged
    python rubric.py --k 25
"""

import argparse
import datetime
import itertools
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import evaluate_recs as ev
import impact_recs as ir

TOP_K = 10
SPLIT = 2017            # decisions in seasons <= SPLIT tune; later ones are the holdout
STAR_PCT = 0.95         # top 5% of the season's pool by BPM
HOLDOUT_LOG = os.path.join("data", "holdout_log.csv")
KEY = ["player_id", "season"]


def top_lists(scores, k=TOP_K):
    """Top-k per team-season from a full ranking (season, TeamId, player_id,
    score). Moves impact_recs' screens rule out (soft-penalised) are not
    listed, matching what the app serves."""
    s = scores[np.isfinite(scores["score"]) & (scores["score"] > -ir.SOFT_PENALTY / 2)]
    s = s.sort_values(["season", "TeamId", "score"], ascending=[True, True, False], kind="stable")
    s = s.groupby(["season", "TeamId"]).head(k).copy()
    s["rank"] = s.groupby(["season", "TeamId"]).cumcount() + 1
    return s[["season", "TeamId", "player_id", "rank"]].reset_index(drop=True)


def player_status(member, min_share=ev.MIN_ACQ_SHARE):
    """impact_recs.move_status(): moved / stayed / gone in t+1."""
    return ir.move_status(member, min_share)


def star_flags(pool, players, pct=STAR_PCT):
    """(player_id, season) -> top `pct` of that season's pool by BPM."""
    b = pool[KEY].drop_duplicates().merge(players[KEY + ["bpm"]], on=KEY, how="left")
    b["q"] = b.groupby("season")["bpm"].rank(pct=True)
    return (b.set_index(KEY)["q"] >= pct).rename("star")


def availability(lists, status, stars, arrivals=None):
    """Share of listed slots by t+1 status, and the star share. With
    `arrivals` (season, TeamId, player_id of real veteran arrivals), also
    `joined`: the share of slots whose player joined *that* team."""
    x = lists.join(status["status"], on=KEY, how="inner")
    out = {s: float((x["status"] == s).mean()) for s in ("moved", "stayed", "gone")}
    if arrivals is not None:
        hit = x.merge(arrivals[["season", "TeamId", "player_id"]].drop_duplicates().assign(j=1),
                      how="left")["j"].notna()
        out["joined"] = float(hit.mean())
    out["stars"] = float(stars.reindex(pd.MultiIndex.from_frame(x[KEY])).fillna(False).mean())
    out["slots"] = len(x)
    return out


def outcomes(impact, players, status):
    """t+1 outcomes keyed to t: BPM, net impact and share at the new team(s)."""
    nxt = players[KEY + ["bpm"]].assign(season=players["season"] - 1).set_index(KEY)["bpm"]
    imp = impact[KEY + ["net_impact"]].assign(season=impact["season"] - 1).set_index(KEY)["net_impact"]
    out = status[["status", "share_new"]].copy()
    out["bpm_next"] = nxt.reindex(out.index)
    out["impact_next"] = imp.reindex(out.index)
    return out


def value_accuracy(lists, outcome, predicted=None, bias=True):
    """Listed players who moved anywhere, once each: what they delivered.
    `predicted` ((player_id, season) -> the system's value claim) adds its
    rank correlation with each outcome, and with `bias` the mean predicted
    - realised net impact (only meaningful on the net-impact scale)."""
    ids = lists[KEY].drop_duplicates()
    x = outcome.reindex(pd.MultiIndex.from_frame(ids))
    x = x[x["status"] == "moved"]
    out = {"movers": len(x), "share_next": x["share_new"].mean(),
           "bpm_next": x["bpm_next"].mean(), "impact_next": x["impact_next"].mean()}
    if predicted is not None:
        p = predicted.reindex(x.index)
        for col in ("impact_next", "bpm_next"):
            ok = p.notna() & x[col].notna()
            out[f"rho {col}"] = (p[ok].rank().corr(x.loc[ok, col].rank()) if ok.sum() > 2 else np.nan)
        out["bias"] = (p - x["impact_next"]).mean() if bias else np.nan
    return out


def specificity(lists):
    """Per season, then averaged: mean Jaccard of two teams' lists, distinct
    players per slot, and the share of slots held by the 10 most-listed."""
    rows = []
    for _, g in lists.groupby("season"):
        sets = [set(v) for _, v in g.groupby("TeamId")["player_id"]]
        pairs = list(itertools.combinations(sets, 2))
        jac = np.mean([len(a & b) / len(a | b) for a, b in pairs]) if pairs else np.nan
        counts = g["player_id"].value_counts()
        rows.append({"jaccard": jac, "distinct": len(counts) / len(g),
                     "top10 share": counts.head(10).sum() / len(g)})
    return pd.DataFrame(rows).mean().to_dict()


def config():
    """The settings a holdout run grades, for HOLDOUT_LOG."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = "unknown", True
    return {"git": head + ("+dirty" if dirty else ""), "realism_weight": ir.REALISM_WEIGHT,
            "max_gap": ir.MAX_GAP, "surplus_penalty": ir.SURPLUS_PENALTY, "scoring": ir.SCORING,
            "creator": ir.CREATOR_TERM, "proj_age": ir.PROJ_AGE}


def log_holdout(path, settings, now=None):
    """Append one holdout view; return how many came before it."""
    prior = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    row = pd.DataFrame([{"when": (now or datetime.datetime.now()).isoformat(timespec="seconds"),
                         **settings}])
    pd.concat([prior, row], ignore_index=True).to_csv(path, index=False)
    return len(prior)


def systems_for(args):
    """Full rankings for every system graded, plus what the metrics need."""
    variants = {"current": ev.variant(gap=ir.MAX_GAP),
                "no salary screen": ev.variant(salary=False)}
    systems, panel, impact, players, _ = ev.impact_rankings(args, variants)
    teams = pd.read_csv(args.teams, usecols=["season", "TeamId", "user_id"])
    pool = systems["current"][["season", "TeamId", "player_id"]]
    lf = ev.lightfm_scores(args.lightfm_full)
    if lf is not None:
        systems["LightFM"] = lf.merge(teams[["user_id", "TeamId"]], on="user_id")[
            ["season", "TeamId", "player_id", "score"]]
    bpm = players[KEY + ["bpm"]].rename(columns={"bpm": "score"})
    systems["best available (BPM)"] = pool.merge(bpm, how="left")
    systems["random"] = pool.assign(score=np.random.default_rng(0).random(len(pool)))
    return systems, panel, impact, players, pool


def report(systems, panel, impact, players, pool, debut, seasons, k=TOP_K):
    member = ir.memberships(panel)
    status = player_status(member)
    stars = star_flags(pool, players)
    outcome = outcomes(impact, players, status)
    lo, hi = seasons
    within = lambda d: d[(d["season"] >= lo) & (d["season"] <= hi)]   # noqa: E731
    lists = {name: within(top_lists(s, k)) for name, s in systems.items()}
    pool_in = within(pool)

    print(f"Decisions in seasons {lo}-{hi}, top {k} per team-season.\n")
    print("== 1. Availability: were the listed players actually available?")
    arr, _ = ev.moves(member, impact, debut)
    arr = within(arr[arr["rankable"]])
    rows = {name: availability(top, status, stars, arr) for name, top in lists.items()}
    rows["(random pool player)"] = availability(pool_in, status, stars, arr)
    ref = float(stars.reindex(pd.MultiIndex.from_frame(arr[KEY])).fillna(False).mean())
    t = pd.DataFrame(rows).T
    t["stars vs arrivals"] = t["stars"] / ref
    pct = ["moved", "joined", "stayed", "gone", "stars"]
    print(t[pct + ["stars vs arrivals"]].to_string(
        formatters={c: "{:.1%}".format for c in pct} | {"stars vs arrivals": "{:.1f}x".format}))
    print(f"(joined: the player joined that very team; stars: top {1 - STAR_PCT:.0%} of the "
          f"season's pool by BPM;\n {ref:.1%} of actual veteran arrivals are stars, n = {len(arr)})\n")

    print("== 3. Value accuracy: did listed players who moved deliver?")
    proj = impact.set_index(KEY)["proj_impact"]
    claims = {"current": proj, "no salary screen": proj,
              "best available (BPM)": players.set_index(KEY)["bpm"]}
    rows = {name: value_accuracy(top, outcome, claims.get(name),
                                 bias=name in ("current", "no salary screen"))
            for name, top in lists.items()}
    # Same players for both claims: the like-for-like test of the valuation.
    rows["(every mover) projected impact"] = value_accuracy(pool_in, outcome, proj)
    rows["(every mover) BPM"] = value_accuracy(pool_in, outcome, players.set_index(KEY)["bpm"],
                                               bias=False)
    t = pd.DataFrame(rows).T
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    print("(share_next: of the new team's floor time; bias = projected - realised net impact, "
          "pts/100.\n Lists' rows are each list's own movers, so compare their rho only with "
          "the two\n every-mover rows, which score both claims on the same players.)\n")

    print("== 4. Specificity: do teams get different lists? (descriptive)")
    rows = {name: specificity(top) for name, top in lists.items()}
    last = pool["season"].max()
    for name in ("current", "LightFM"):
        if name in systems:
            served = top_lists(systems[name], k)
            rows[f"{name}, {last} (serving)"] = specificity(served[served["season"] == last])
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.3f}"))
    print("(jaccard: overlap of two teams' lists; distinct: unique players per slot; "
          "top10 share: slots held by the 10 most-listed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", action="store_true",
                    help=f"grade decisions after {SPLIT} (logged to {HOLDOUT_LOG})")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--log", default=HOLDOUT_LOG)
    opts = ap.parse_args()
    args = ev.parser().parse_args([])
    systems, panel, impact, players, pool = systems_for(args)
    debut = pd.read_csv(args.career, usecols=["player_id", "from"]) \
        .drop_duplicates("player_id").set_index("player_id")["from"]
    last_labelled = int(pool["season"].max()) - 1          # t+1 must have been played
    if opts.holdout:
        n = log_holdout(opts.log, config())
        print(f"HOLDOUT view #{n + 1} (logged to {opts.log}). Settings changed after "
              f"reading this are tuned on it.\n")
        seasons = (SPLIT + 1, last_labelled)
    else:
        seasons = (int(pool["season"].min()), SPLIT)
    report(systems, panel, impact, players, pool, debut, seasons, opts.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
