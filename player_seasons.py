"""One row per player-season since 1999: the player table the recommender reads.

Replaces master_clustered.csv as the source of name, age, position and BPM
for impact_recs.py, acquisition.py, evaluate_recs.py, fit_model.py and
team_style.py. master_clustered.csv is player_cluster.py's output for the
LightFM pipeline: it keeps only player-seasons with >= 20 games, so it misses
17% of the player-seasons with a lineup impact (1.3% of the >= 500-possession
evaluation pool), and it ties the new pipeline to the old one.

This is the full BBRef universe instead (Advanced.csv, the TOT row for traded
players), with:
  pos_index   centre of mass of minutes by position from Player Play By
              Play.csv, 1 = pure PG ... 5 = pure C (player_cluster.py's
              definition, kept so position windows mean the same thing)
  bpm/obpm/dbpm, age, g, mp, usg_percent
  cluster_label  the style archetype where player_cluster.py assigned one
              (games >= 20), else "unknown" -- only if --clusters is given

Usage:
    python player_seasons.py                    # data/player_seasons.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from helper import season_totals

FIRST_SEASON = 1999
POS_SHARE_COLS = ["pg_percent", "sg_percent", "sf_percent", "pf_percent", "c_percent"]


def build(adv_path, pbp_path, clusters_path=None):
    adv = pd.read_csv(adv_path, usecols=["season", "player_id", "player", "team", "age", "g", "mp",
                                         "bpm", "obpm", "dbpm", "usg_percent"])
    adv = season_totals(adv[(adv["season"] >= FIRST_SEASON) & (adv["mp"] > 0)])
    pbp = pd.read_csv(pbp_path, usecols=["season", "player_id", "team", "mp"] + POS_SHARE_COLS)
    pbp = season_totals(pbp[(pbp["season"] >= FIRST_SEASON) & (pbp["mp"] > 0)])
    shares = pbp[POS_SHARE_COLS].fillna(0.0)
    pbp["pos_index"] = ((shares.to_numpy() * np.arange(1, 6)).sum(axis=1)
                        / shares.sum(axis=1).replace(0, np.nan))
    out = adv.drop(columns="team").merge(pbp[["season", "player_id", "pos_index"]],
                                         on=["season", "player_id"], how="left")
    out["season"] = out["season"].astype(int)
    if clusters_path:
        cl = pd.read_csv(clusters_path, usecols=["player_id", "season", "cluster_label"]) \
            .drop_duplicates(["player_id", "season"])
        out = out.merge(cl, on=["player_id", "season"], how="left")
        out["cluster_label"] = out["cluster_label"].fillna("unknown")
    else:
        out["cluster_label"] = "unknown"
    if out.duplicated(["player_id", "season"]).any():
        raise ValueError("duplicate player-seasons after season_totals")
    return out.sort_values(["season", "player_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adv", default=os.path.join("data", "Advanced.csv"))
    ap.add_argument("--pbp", default=os.path.join("data", "Player Play By Play.csv"))
    ap.add_argument("--clusters", default=None,
                    help="player_cluster.py output to take archetype labels from (optional)")
    ap.add_argument("--out", default=os.path.join("data", "player_seasons.csv"))
    args = ap.parse_args()
    t = build(args.adv, args.pbp, args.clusters)
    t.to_csv(args.out, index=False)
    print(f"Wrote {len(t)} player-seasons to '{args.out}' "
          f"(pos_index missing {t['pos_index'].isna().mean():.1%}, age missing {t['age'].isna().mean():.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
