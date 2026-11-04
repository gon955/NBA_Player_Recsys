"""Export the impact recommender's lists for the backend to serve.

The backend does no inference: it looks up a precomputed list per team-season.
This runs impact_recs.build() with its defaults (salary screen, value cap,
upgrade gate, acquisition model) at a deeper K than data/recs_impact.csv, so
the app's "number of recommendations" box has room, and joins on the names
and context the explanation card shows. Everything is season-t information,
the same as the evaluated lists (evaluate_recs.py).

Columns:
  user_id, rank, player_id, player, age, bpm   the candidate in season t
  score     value + REALISM_WEIGHT * log P (the ranking)
  value     projected net impact in t+1 if he moves, pts/100 (impact_recs.project_mover())
  position_value   the team's rotation at his position, minutes-weighted, pts/100
  replaces_id, replaces, replaces_value   the rotation player sent out whose
            departure costs least among the legal swaps
  with_ids, with, with_salary   who else goes out to match salary, when his
            salary alone does not (impact_recs.package_members); ';'-joined
  salary, replaces_salary, cap            dollars, season t

Veto rules switched on in audit.RULES (active=True, after passing
`python audit.py rules`) and the review's flags (audit.JUDGMENTS, from
review_app.py) are applied here: the lists are built
VETO_SLACK deeper, removed entries dropped, and the rest re-ranked.

Usage:
    python export_recs.py                    # writes backend/data/recs.csv
    python export_recs.py --k 25 --out other.csv
"""

import argparse
import os
import sys

import pandas as pd

import impact_recs as ir
from fetch_salaries import CAP

K_SERVE = 50     # the frontend's input allows up to 50
VETO_SLACK = 30  # extra depth built when veto rules may remove entries


def enrich(recs, players, cap=CAP):
    """recs: impact_recs.build() output. players: player_seasons.csv."""
    out = recs.copy()
    out["season"] = out["user_id"].str.rsplit("_", n=1).str[1].astype(int)
    out["player_id"] = out["item_id"].str.rsplit("_", n=1).str[0]
    ps = players.drop_duplicates(["player_id", "season"]).set_index(["player_id", "season"])

    def lookup(ids, col):
        idx = pd.MultiIndex.from_arrays([ids, out["season"]])
        return ps[col].reindex(idx).to_numpy()

    out["player"] = lookup(out["player_id"], "player")
    out["player"] = out["player"].fillna(out["player_id"])
    out["age"] = lookup(out["player_id"], "age")
    out["bpm"] = lookup(out["player_id"], "bpm")
    out["replaces_id"] = out["replaces"]
    out["replaces"] = pd.Series(lookup(out["replaces_id"], "player")).fillna(out["replaces_id"]).to_numpy()
    sal = pd.read_csv(os.path.join("data", "salaries.csv"), usecols=["player_id", "season", "salary"]) \
        .drop_duplicates(["player_id", "season"]).set_index(["player_id", "season"])["salary"]
    ids = out["with"].fillna("").astype(str).str.split(";").apply(lambda v: [x for x in v if x])
    names = ps["player"]
    out["with_ids"] = ids.str.join(";")
    out["with"] = [";".join(names.get((p, s), p) for p in v) for v, s in zip(ids, out["season"])]
    out["with_salary"] = [sum(sal.get((p, s), 0.0) for p in v) if v else 0.0
                          for v, s in zip(ids, out["season"])]
    out["value"] = out["gain"]
    out["position_value"] = out["gain"] - out["need"]
    out["cap"] = out["season"].map(cap)
    cols = ["user_id", "rank", "player_id", "player", "age", "bpm", "score", "value", "position_value",
            "replaces_id", "replaces", "replaces_value", "with_ids", "with", "with_salary",
            "salary", "replaces_salary", "cap"]
    return out[cols]


def apply_vetoes(recs, k, rules=None, judgments=None):
    """Drop what active veto rules and the review's flags remove, re-rank,
    keep the top k. Returns (recs, removed by rules, removed by flags)."""
    import audit
    import evaluate_recs as ev
    rules = audit.active_rules() if rules is None else rules
    judgments = audit.load_judgments() if judgments is None else judgments
    flags = judgments[judgments["verdict"] == "flag"] if len(judgments) else judgments
    if not rules and not len(flags):
        return recs[recs["rank"] <= k], 0, 0
    teams = pd.read_csv(audit.TEAMS, usecols=["season", "TeamId", "user_id"])
    rows = recs.assign(player_id=recs["item_id"].str.rsplit("_", n=1).str[0]) \
        .merge(teams, on="user_id", how="left")
    by_rule = pd.Series(False, index=rows.index)
    if rules:
        pairs, _, _ = audit.load(ev.parser().parse_args([]))
        by_rule = audit.veto_mask(rows, pairs, rules)
    by_flag = audit.flagged_mask(rows, flags) & ~by_rule
    drop = (by_rule | by_flag).to_numpy()
    kept = recs[~drop].copy()
    kept["rank"] = kept.groupby("user_id").cumcount() + 1
    return kept[kept["rank"] <= k], int(by_rule.sum()), int(by_flag.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=K_SERVE)
    ap.add_argument("--out", default=os.path.join("backend", "data", "recs.csv"))
    args = ap.parse_args()

    paths = ir.parser().parse_args([])      # impact_recs' default inputs
    import audit
    rules, judgments = audit.active_rules(), audit.load_judgments()
    n_flags = int((judgments["verdict"] == "flag").sum()) if len(judgments) else 0
    recs = ir.build(paths, k=args.k + (VETO_SLACK if rules or n_flags else 0))
    recs, by_rule, by_flag = apply_vetoes(recs, args.k, rules, judgments)
    if rules:
        print(f"Veto rules {', '.join(r.name for r in rules)} removed {by_rule} entries.")
    if n_flags:
        print(f"Review flags removed {by_flag} entries ({n_flags} flags on file).")
    out = enrich(recs, pd.read_csv(paths.players))
    out.round({"score": 3, "value": 3, "position_value": 3, "replaces_value": 3, "bpm": 1}).to_csv(
        args.out, index=False)
    print(f"Wrote {len(out)} recommendations for {out['user_id'].nunique()} "
          f"team-seasons to '{args.out}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
