"""Team-season style and archetype mix, built from the lineup data.

Replaces the recommender's `tcluster` token. The audit found team clusters were
a standings table -- offensive quality entered twice (pts_per_100 and o_rtg are
the same number), and the labels were indistinguishable from a random 5-way
split of team-seasons. Here a team is described by HOW it plays, with quality
kept out of the vector and carried separately:

  style      pace, where shots come from, free-throw rate, assisted share,
             offensive rebounding, turnovers; steals, blocks and defensive
             rebounds per possession. z-scored within season, so "high 3PA"
             means high for that year, not high because it's 2019.
  mix        possession-weighted count of each archetype on the floor (sums
             to 5): what the team actually plays, not who is on the roster.
  quality    o_rtg / d_rtg / net_rtg, plus turnover and defensive-rebound
             rates, which track quality more than style. Context only.

pbpstats has no opponent shot diet, so defensive style is limited to the
defence's own events.

Validation (--report): the 11 style z-scores predict net rating at CV R^2
0.31 (old tcluster: 0.60); style persists year to year at r 0.48-0.65; the
extremes are the right teams (HOU '18-'19 threes, Sloan-era UTA assists,
MEM '22 offensive boards, OKC '12 blocks).

Fit test (--evaluate-fit): adding archetype x team-style terms to the
archetype main effects, within-season held-out lineups, weighted R^2 O / D:

                      main only        + archetype x style
  real                .0364 / .0167    .0361 / .0165
  team style shuffled .0364 / .0167    .0362 / .0164
  archetypes shuffled .0358 / .0158    .0355 / .0154

Real team styles do no better than shuffled ones, at lam 1e5 and 1e6. Like
archetype pairs (fit_model.py), "this archetype suits this kind of team" is
not detectable in season-aggregated lineup data. The style vector is still
the right *description* of a team for display and explanation; it just isn't
a scoring input.

Usage:
    python team_style.py            # write data/team_style.csv
    python team_style.py --report   # validation: not quality, persistent, sane
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import fit_model as fm
import impact_model as im
from helper import canonical_team, era_of
from lineup_features import bbref_team

KEYS = ["season", "TeamId", "EntityId"]
COUNT_COLS = [
    "SecondsPlayed", "OffPoss", "DefPoss", "Points", "OpponentPoints",
    "FG2A", "FG2M", "FG3A", "FG3M", "FTA",
    "AtRimFGA", "ShortMidRangeFGA", "LongMidRangeFGA", "Corner3FGA", "Arc3FGA",
    "Assists", "OffRebounds", "DefRebounds", "Turnovers", "Steals", "Blocks",
]

STYLE = [
    "pace", "rim_freq", "short_mid_freq", "long_mid_freq", "corner3_freq", "arc3_freq",
    "ft_rate", "ast_rate", "oreb_rate", "stl_rate", "blk_rate",
]
# Computed and written, but kept OUT of the style vector: both are mostly
# quality. Defensive rebounds per possession is high when opponents miss a lot
# (r = +0.59 with net rating); turnover rate is ball security (r = -0.37). With
# them in, the style vector predicts net rating at CV R^2 0.73; without, 0.31
# (the old tcluster one-hot: 0.60). What remains is the honest overlap -- good
# teams do take more corner threes and block more shots.
QUALITY = ["o_rtg", "d_rtg", "net_rtg", "tov_rate", "dreb_rate"]
# Subset used for archetype x style terms: one column per idea (the five shot
# frequencies sum to ~1, and tov_rate is closer to quality than to style).
STYLE_FOR_FIT = ["pace", "rim_freq", "mid_freq", "three_freq", "ft_rate",
                 "ast_rate", "oreb_rate", "stl_rate", "blk_rate"]


def load_lineup_stats(path, panel):
    """Raw counts for every panel lineup, in panel row order."""
    raw = pd.read_csv(path, usecols=KEYS + ["TeamAbbreviation"] + COUNT_COLS, low_memory=False)
    raw[COUNT_COLS] = raw[COUNT_COLS].fillna(0.0)
    raw["season"] = raw["season"].astype(int)
    out = panel[KEYS].merge(raw, on=KEYS, how="left", validate="one_to_one")
    if out["OffPoss"].isna().any():
        raise ValueError(f"{out['OffPoss'].isna().sum()} panel lineups missing from {path}")
    return out


def team_style(stats):
    """Aggregate lineup counts to one style row per (season, TeamId)."""
    g = stats.groupby(["season", "TeamId"], sort=True)
    t = g[COUNT_COLS].sum()
    t["TeamAbbreviation"] = g["TeamAbbreviation"].first()
    fga = t["FG2A"] + t["FG3A"]
    misses = fga - t["FG2M"] - t["FG3M"]
    s = pd.DataFrame(index=t.index)
    # Possessions per 48 minutes; SecondsPlayed is the unit's floor time.
    s["pace"] = 2880 * (t["OffPoss"] + t["DefPoss"]) / 2 / t["SecondsPlayed"]
    for col, src in [("rim_freq", "AtRimFGA"), ("short_mid_freq", "ShortMidRangeFGA"),
                     ("long_mid_freq", "LongMidRangeFGA"), ("corner3_freq", "Corner3FGA"),
                     ("arc3_freq", "Arc3FGA")]:
        s[col] = t[src] / fga
    s["ft_rate"] = t["FTA"] / fga
    s["ast_rate"] = t["Assists"] / (t["FG2M"] + t["FG3M"])
    s["oreb_rate"] = t["OffRebounds"] / misses
    s["tov_rate"] = t["Turnovers"] / t["OffPoss"]
    s["stl_rate"] = t["Steals"] / t["DefPoss"]
    s["blk_rate"] = t["Blocks"] / t["DefPoss"]
    s["dreb_rate"] = t["DefRebounds"] / t["DefPoss"]
    s["mid_freq"] = s["short_mid_freq"] + s["long_mid_freq"]
    s["three_freq"] = s["corner3_freq"] + s["arc3_freq"]
    s["o_rtg"] = 100 * t["Points"] / t["OffPoss"]
    s["d_rtg"] = 100 * t["OpponentPoints"] / t["DefPoss"]
    s["net_rtg"] = s["o_rtg"] - s["d_rtg"]
    s["poss"] = t["OffPoss"] + t["DefPoss"]
    s["TeamAbbreviation"] = t["TeamAbbreviation"]
    return s.reset_index()


def zscore_within_season(df, cols):
    z = df.groupby("season")[cols].transform(lambda c: (c - c.mean()) / c.std(ddof=0))
    return z.add_prefix("z_")


def archetype_mix(panel, labels):
    """Possession-weighted archetype counts on the floor per team-season."""
    names = fm.slot_labels(panel, labels, panel["season"].to_numpy())
    w = (panel["OffPoss"] + panel["DefPoss"]).to_numpy(float)
    vocab = fm.Vocab(labels["cluster_label"], pairs=False)
    X = vocab.design(names).multiply(w[:, None]).tocsr()
    keys = panel[["season", "TeamId"]].copy()
    mix = pd.DataFrame(X.toarray(), columns=[f"mix_{n}" for n in vocab.names])
    mix = pd.concat([keys.reset_index(drop=True), mix], axis=1)
    mix["_w"] = w
    agg = mix.groupby(["season", "TeamId"]).sum()
    return agg.drop(columns="_w").div(agg["_w"], axis=0).reset_index()


def build(panel, stats, labels):
    style = team_style(stats)
    style = pd.concat([style, zscore_within_season(style, STYLE + ["mid_freq", "three_freq"])],
                      axis=1)
    style["team_full"] = [canonical_team(bbref_team(a, s), s)
                          for a, s in zip(style["TeamAbbreviation"], style["season"])]
    style["era"] = style["season"].map(era_of)
    style["user_id"] = style["team_full"] + "_" + style["season"].astype(str)
    return style.merge(archetype_mix(panel, labels), on=["season", "TeamId"], how="left")


# ---------------------------------------------------------------- validation

def _cv_r2(X, y, folds=5, seed=0):
    """Out-of-fold R^2 of a linear regression (least squares + intercept)."""
    fold = np.random.default_rng(seed).integers(0, folds, len(y))
    pred = np.empty(len(y))
    A = np.column_stack([X, np.ones(len(y))])
    for k in range(folds):
        tr = fold != k
        beta = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        pred[~tr] = A[~tr] @ beta
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def report(teams):
    z = [f"z_{c}" for c in STYLE]
    net_z = teams.groupby("season")["net_rtg"].transform(lambda c: c - c.mean())
    print(f"Team-seasons: {len(teams)}")

    print("\n1. Style is not quality")
    print(f"   net rating explained by the {len(z)} style z-scores (5-fold CV R^2): "
          f"{_cv_r2(teams[z].to_numpy(), net_z.to_numpy()):.2f}")
    corr = teams[z].corrwith(net_z).sort_values()
    print("   corr with net rating: " + ", ".join(f"{c[2:]} {v:+.2f}" for c, v in corr.items()))

    print("\n2. Style persists (same franchise, consecutive seasons)")
    nxt = teams.assign(season=teams["season"] - 1)
    pair = teams.merge(nxt, on=["season", "TeamId"], suffixes=("", "_next"))
    auto = {c: pair[f"z_{c}"].corr(pair[f"z_{c}_next"]) for c in STYLE}
    auto["net_rtg (quality)"] = pair["net_rtg"].corr(pair["net_rtg_next"])
    print("   year-to-year r: " + ", ".join(f"{k} {v:.2f}" for k, v in auto.items()))

    print("\n3. Extremes")
    show = [("three_freq", True), ("mid_freq", True), ("ast_rate", True), ("pace", True),
            ("pace", False), ("oreb_rate", True), ("blk_rate", True)]
    for col, high in show:
        t = teams.nlargest(3, f"z_{col}") if high else teams.nsmallest(3, f"z_{col}")
        print(f"   {'most' if high else 'least'} {col}: " + ", ".join(
            f"{r.TeamAbbreviation} '{str(r.season)[2:]} ({getattr(r, 'z_' + col):+.1f})"
            for r in t.itertuples()))

    mix_cols = [c for c in teams.columns if c.startswith("mix_") and c != "mix_unknown"]
    print("\n4. Archetype mix, most of each on the floor:")
    for c in mix_cols:
        r = teams.loc[teams[c].idxmax()]
        print(f"   {c[4:]:<20} {r.TeamAbbreviation} '{str(r.season)[2:]} {r[c]:.2f} per lineup")


# ------------------------------------------------- archetype x style fit test

def style_design(slot_names, team_z, vocab):
    """Main effects plus archetype-count x team-style terms, one row per lineup."""
    from scipy import sparse
    mains = vocab.design(slot_names).tocsr()          # (n, n_archetypes)
    blocks = [mains] + [mains.multiply(team_z[:, [j]]).tocsr() for j in range(team_z.shape[1])]
    return sparse.hstack(blocks).tocsr()


def evaluate_style_fit(panel, stats, labels, box, lam, folds=5, seed=0, shuffle=None):
    """Within-season CV: does a player's archetype interact with team style?

    Team style for each fold is rebuilt from the training lineups only, so a
    held-out lineup never contributes to the style it is scored against.
    shuffle="style" permutes team style vectors among teams within a season;
    shuffle="labels" permutes archetypes -- the two controls.
    """
    rng = np.random.default_rng(seed)
    if shuffle == "labels":
        labels = fm.shuffle_labels(labels, seed)
    vocab = fm.Vocab(labels["cluster_label"], pairs=False)
    fold = np.random.default_rng(seed).integers(0, folds, len(panel))
    seasons = sorted(panel["season"].unique())
    n_cols = vocab.n_cols * (1 + len(STYLE_FOR_FIT))
    res = {"main": ([], []), "style": ([], [])}

    for k in range(folds):
        acc = {m: (fm._Accumulator(n), fm._Accumulator(n))
               for m, n in (("main", vocab.n_cols), ("style", n_cols))}
        held = []
        for s in seasons:
            in_s = (panel["season"] == s).to_numpy()
            tr_mask, te_mask = in_s & (fold != k), in_s & (fold == k)
            train, test = panel[tr_mask], panel[te_mask]
            st = team_style(stats[tr_mask])
            z = zscore_within_season(st, STYLE_FOR_FIT).fillna(0.0)
            z.index = st["TeamId"].to_numpy()
            if shuffle == "style":
                z = pd.DataFrame(rng.permutation(z.to_numpy()), index=z.index, columns=z.columns)
            prior = box.xs(s, level="season")
            est = im.fit_season(train, im.DEFAULT_LAM, prior["o"], prior["d"]).assign(season=s)
            for part, is_train in ((train, True), (test, False)):
                y_o, w_o, y_d, w_d = im._targets(part)
                o, d = fm.impact_sum(part, est, np.full(len(part), s))
                names = fm.slot_labels(part, labels, np.full(len(part), s))
                tz = z.reindex(part["TeamId"].to_numpy()).fillna(0.0).to_numpy()
                X = {"main": vocab.design(names), "style": style_design(names, tz, vocab)}
                if is_train:
                    for m in X:
                        acc[m][0].add(X[m], y_o - o, w_o)
                        acc[m][1].add(X[m], y_d - d, w_d)
                else:
                    held.append((X, y_o, o, w_o, y_d, d, w_d))
        for m in res:
            bo, bd = acc[m][0].solve(lam), acc[m][1].solve(lam)
            for X, y_o, o, w_o, y_d, d, w_d in held:
                res[m][0].append((y_o, o + X[m] @ bo, w_o))
                res[m][1].append((y_d, d + X[m] @ bd, w_d))

    def r2(side):
        return im._weighted_r2(*map(np.concatenate, zip(*side)))
    return {m: (r2(o), r2(d)) for m, (o, d) in res.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=os.path.join("data", "lineup_panel.csv"))
    ap.add_argument("--lineups", default=os.path.join("data", "pbpstats_lineups.csv"))
    ap.add_argument("--clusters", default="master_clustered.csv")
    ap.add_argument("--adv", default=os.path.join("data", "Advanced.csv"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--evaluate-fit", action="store_true",
                    help="within-season CV of archetype x team-style terms vs. controls")
    ap.add_argument("--lam", type=float, default=1e5)
    ap.add_argument("--out", default=os.path.join("data", "team_style.csv"))
    args = ap.parse_args()

    panel = pd.read_csv(args.panel, usecols=KEYS + im.SLOTS + [
        "OffPoss", "DefPoss", "Points", "OpponentPoints"])
    panel["season"] = panel["season"].astype(int)
    labels = pd.read_csv(args.clusters, usecols=["player_id", "season", "cluster_label"])
    stats = load_lineup_stats(args.lineups, panel)

    if args.evaluate_fit:
        box = im.box_prior(args.adv)
        for name, shuffle in [("real", None), ("style shuffled", "style"),
                              ("labels shuffled", "labels")]:
            r = evaluate_style_fit(panel, stats, labels, box, args.lam, shuffle=shuffle)
            print(f"{name:<16} main only O {r['main'][0]:.4f} D {r['main'][1]:.4f} | "
                  f"+ archetype x style O {r['style'][0]:.4f} D {r['style'][1]:.4f}", flush=True)
        return 0

    teams = build(panel, stats, labels)
    if args.report:
        report(teams)
        return 0
    teams.round(4).to_csv(args.out, index=False)
    print(f"Wrote {len(teams)} team-seasons to '{args.out}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
