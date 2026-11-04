"""Does a skill matter more to a lineup that lacks it?

The premise of the recommender is that a team short on something -- three-
point shooting, say -- gains more from a player who has it than a team that is
already full of it. fit_model.py and team_style.py tested "fit" with 105 and
135 parameters and found nothing; with season-total lineup data, a small effect
spread that thin has no power. This module tests the premise directly, with
two parameters per skill.

For each skill, n = how many of a lineup's five players have it, judged on the
PREVIOUS season's box score (so a lineup that happened to shoot well doesn't
label its own players as shooters). Per season, jointly with player effects:

    rating = sum(player effects) + b1 * n + b2 * n^2 + noise

Player effects are ridge toward the minutes-shrunk BPM prior, exactly as in
impact_model.py; b1, b2 are unpenalised. n is the sum of its players' flags,
so b1 is collinear with the skilled players' own effects -- a skill's average
value is already in each player's impact. The curvature b2 is what is left:
b2 < 0 means diminishing returns, i.e. the k-th shooter is worth more to a
lineup with few shooters than to one with many. That is the team-specific
signal a weakness-based recommender needs. Relative to the average lineup, an
extra skilled player is worth 2 * b2 * (k - n_bar) more on a lineup that has k.

Seasons are pooled exactly (the 2x2 Schur complements of the per-season
normal equations add up). Evidence, all reported per skill:
  - permutation null: flags shuffled among rotation players within each label
    season, same prevalence; p is two-sided on b2
  - consistency: seasons whose own b2 has the pooled sign
  - next season: does adding b1*n + b2*n^2 (fit on seasons <= t, residuals of
    data/player_impact.csv) improve season t+1's weighted R^2? Same yardstick
    as impact_model.py and fit_model.py
  - known-only: the main threshold refitted on lineups whose five players all
    have a previous season (rookies otherwise count as not skilled)

Skills (season-relative, among players with 500+ minutes in the label season):
  shooter   3PA per 100 in the top 40%, 3P% at or above league, 50+ 3PA -> offence
  rim       block % in the top 15%                                      -> defence
  dreb      defensive rebound % in the top 20%                          -> defence
  creator   usage % in the top 20%                                      -> offence

Each b2 is fitted twice: as above, and with unpenalised team-season
intercepts so it comes only from differences between a team's own lineups.
The team version is the one to trust. Without it, a team whose system lifts
every lineup (shooter-heavy late-2010s offences, say) shows up as a lineup-mix
effect that the ridge-shrunk player effects can't absorb, and the permutation
null doesn't preserve team composition, so it can't catch this.

Results, 1999-2025, 200 permutations (b2 in pts/100 per lineup, team FE;
"no FE" for contrast; looser/stricter cuts agree in sign):

            b2 team FE   p      same sign   known-only   b2 no FE
  shooter   +0.02       .77     17/27       +0.06        +0.34 (team confound)
  creator   -0.34       <.005   22/27       -0.24        +0.07
  rim       +0.32       .005    17/27       +0.25        +0.31
  dreb      +0.18       .02     16/27       +0.18        +0.06

Read:
  - Shooting: no curvature within teams at any cut. The null sd (~0.06) rules
    out a shooter being worth more than ~0.25 pts/100 extra to a lineup with
    none vs. the average lineup. "Lacks shooting -> add a shooter" is a style
    statement here, not a measured net-rating gain.
  - Creators: clear diminishing returns, stable across cuts, seasons and the
    rookie-free subset. One more high-usage player is worth about +0.8 on a
    lineup with none and -0.6 / -1.2 on a lineup that already has 2 / 3,
    relative to the average lineup. This is the one weakness signal to use.
  - Rim protection and defensive rebounding: increasing returns -- two-big
    lineups defend better than their parts -- the opposite of filling a gap.
    Driven by the ~8-18% of possessions with two such players and the same
    sign in only 16-17 of 27 seasons; without opponent lineups this may be
    who those units face. Not used for ranking.
  - Next season: the count terms (no FE, on player_impact residuals) move
    weighted R^2 by +.0002 at most. Effects this size are below what that
    yardstick can show; the within-team permutation test is the evidence.

Usage:
    python weakness_test.py                 # all skills, main thresholds
    python weakness_test.py --sensitivity   # also looser and stricter cuts
    python weakness_test.py --write-curvature   # creator b2 per season, for impact_recs
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

import fit_model as fm
import impact_model as im
from helper import season_totals

ROTATION_MP = 500
MIN_3PA = 50

# name: (side, column, main percentile, sensitivity percentiles)
SKILLS = {
    "shooter": ("off", "x3pa_per_100_poss", 0.6, (0.5, 0.7)),
    "rim": ("def", "blk_percent", 0.85, (0.8, 0.9)),
    "dreb": ("def", "drb_percent", 0.8, (0.75, 0.85)),
    "creator": ("off", "usg_percent", 0.8, (0.75, 0.85)),
}


def load_box(data_dir):
    """One row per player-season (TOT row for traded players) with skill inputs."""
    def read(name, cols):
        return season_totals(pd.read_csv(os.path.join(data_dir, name), usecols=cols))

    keys = ["season", "player_id", "team"]
    p100 = read("Per 100 Poss.csv", keys + ["mp", "x3pa_per_100_poss"])
    tot = read("Player Totals.csv", keys + ["x3p", "x3pa"])
    adv = read("Advanced.csv", keys + ["blk_percent", "drb_percent", "usg_percent"])
    box = (p100.merge(tot.drop(columns="team"), on=["season", "player_id"])
           .merge(adv.drop(columns="team"), on=["season", "player_id"]))
    box["season"] = box["season"].astype(int)
    return box


def skill_flags(box, skill, pct):
    """Boolean flag per player-season, plus which rows could hold the flag.

    Thresholds are percentiles among that season's rotation players, so a
    1999 shooter is judged against 1999, not against 2024's volume.
    """
    _, col, _, _ = SKILLS[skill]
    rot = box["mp"] >= ROTATION_MP
    thr = box[rot].groupby("season")[col].quantile(pct)
    flag = rot & (box[col] >= box["season"].map(thr))
    if skill == "shooter":
        lg = box.groupby("season").apply(lambda g: g["x3p"].sum() / g["x3pa"].sum())
        pct3 = box["x3p"] / box["x3pa"].where(box["x3pa"] > 0)
        flag &= (box["x3pa"] >= MIN_3PA) & (pct3 >= box["season"].map(lg))
    return pd.DataFrame({"player_id": box["player_id"], "season": box["season"],
                         "flag": flag.to_numpy(), "eligible": rot.to_numpy()})


def load_panel(path):
    cols = im.SLOTS + ["season", "TeamId", "OffPoss", "DefPoss", "Points", "OpponentPoints"]
    panel = pd.read_csv(path, usecols=cols)
    panel["season"] = panel["season"].astype(int)
    return panel


class Labels:
    """Maps each lineup slot to a row of a flag table, once, so a permutation
    only re-indexes an array instead of re-joining 1.6M slots."""

    def __init__(self, panel, flags):
        key = pd.MultiIndex.from_frame(flags[["player_id", "season"]])
        ids = panel[im.SLOTS].to_numpy().ravel()
        label_season = np.repeat(panel["season"].to_numpy() - 1, 5)
        self.row = key.get_indexer(pd.MultiIndex.from_arrays([ids, label_season]))
        self.row = self.row.reshape(-1, 5)
        self.known = (self.row >= 0).all(1)
        self.flags = flags

    def counts(self, flag):
        # Row -1 (no previous season, e.g. a rookie) reads the appended False.
        return np.append(flag, False)[self.row].sum(1)

    def shuffled(self, rng):
        """Flags permuted among eligible rows within each label season."""
        out = self.flags["flag"].to_numpy().copy()
        for _, idx in self.flags[self.flags["eligible"]].groupby("season").groups.items():
            pos = self.flags.index.get_indexer(idx)
            out[pos] = rng.permutation(out[pos])
        return out


class SeasonFit:
    """One season's player ridge, factored once, so (b1, b2) for any count
    vector costs two triangular solves (Frisch-Waugh with a ridge block)."""

    def __init__(self, df, side, prior, lam):
        players = pd.unique(df[im.SLOTS].to_numpy().ravel())
        X = im.design(df, players)
        y_off, w_off, y_def, w_def = im._targets(df)
        y, w = (y_off, w_off) if side == "off" else (y_def, w_def)
        p = prior.reindex(players).fillna(0.0).to_numpy()
        self.w = w
        self.team = pd.factorize(df["TeamId"])[0]
        self.r = y - X @ p                       # residual from the prior
        self.XtW = X.T.multiply(w[None, :]).tocsr()
        self.chol = cho_factor((self.XtW @ X).toarray() + lam * np.eye(len(players)))
        self.XtWr = self.XtW @ self.r

    def normal_eqs(self, n, team_fe=False):
        """Schur complement S and right side s for Z = [n, n^2].

        With team_fe, unpenalised team intercepts are added and then
        eliminated too, so (b1, b2) come only from differences between a
        team's own lineups -- a team whose system flatters every lineup it
        plays can't masquerade as a lineup-composition effect.
        """
        Z = np.column_stack([n, n ** 2]).astype(float)
        if team_fe:
            Z = np.column_stack([Z, np.eye(self.team.max() + 1)[self.team]])
        WZ = Z * self.w[:, None]
        XtWZ = self.XtW @ Z
        AinvXtWZ = cho_solve(self.chol, XtWZ)
        S = Z.T @ WZ - XtWZ.T @ AinvXtWZ
        s = WZ.T @ self.r - AinvXtWZ.T @ self.XtWr
        if team_fe:
            k = np.linalg.pinv(S[2:, 2:])
            s = s[:2] - S[:2, 2:] @ k @ s[2:]
            S = S[:2, :2] - S[:2, 2:] @ k @ S[2:, :2]
        return S, s


def build_fits(panel, side, box_prior, lam, mask=None):
    fits = {}
    for season, df in panel.groupby("season"):
        if mask is not None:
            df = df[mask[df.index]]
        prior = box_prior.xs(season, level="season")["o" if side == "off" else "d"]
        fits[season] = (df.index.to_numpy(), SeasonFit(df, side, prior, lam))
    return fits


def pooled(fits, n, team_fe=False):
    """Pooled (b1, b2) and the per-season b2s, for lineup counts n."""
    S_sum, s_sum, per = np.zeros((2, 2)), np.zeros(2), {}
    for season, (rows, fit) in fits.items():
        S, s = fit.normal_eqs(n[rows], team_fe)
        S_sum += S
        s_sum += s
        if np.linalg.cond(S) < 1e10:
            per[season] = np.linalg.solve(S, s)[1]
    return np.linalg.solve(S_sum, s_sum), per


def curvature_by_season(skill, panel, box, box_prior, lam, pct=None):
    """(b1, b2) as of each season t, pooled over seasons <= t with team effects.

    What impact_recs.py reads: a season-t recommendation may use season t's
    own lineups (they have been played) but nothing later. Counts are judged
    on the previous season's flags, exactly as in run_skill.
    """
    side, _, main_pct, _ = SKILLS[skill]
    flags = skill_flags(box, skill, main_pct if pct is None else pct)
    n = Labels(panel, flags).counts(flags["flag"].to_numpy())
    S_sum, s_sum, out = np.zeros((2, 2)), np.zeros(2), []
    for season, (rows, fit) in sorted(build_fits(panel, side, box_prior, lam).items()):
        S, s = fit.normal_eqs(n[rows], team_fe=True)
        S_sum += S
        s_sum += s
        b = np.linalg.solve(S_sum, s_sum) if np.linalg.cond(S_sum) < 1e10 else (np.nan, np.nan)
        out.append({"season": season, "skill": skill, "b1": b[0], "b2": b[1]})
    return pd.DataFrame(out)


def next_season_r2(panel, impact, labels, flag, side):
    """Weighted R^2 for season t+1 from season-t impacts, with and without the
    count terms fit on residuals of seasons <= t (expanding window)."""
    n = labels.counts(flag)
    seasons = sorted(panel["season"].unique())
    acc = fm._Accumulator(2)
    base, with_terms = [], []
    pick = 0 if side == "off" else 2

    def block(df, impact_season):
        targets = im._targets(df)
        y, w = targets[pick], targets[pick + 1]
        sums = fm.impact_sum(df, impact, np.full(len(df), impact_season))
        Z = np.column_stack([n[df.index], n[df.index] ** 2]).astype(float)
        return y, w, sums[0 if side == "off" else 1], Z

    for t, nxt in zip(seasons[:-1], seasons[1:]):
        y, w, o, Z = block(panel[panel["season"] == t], t)
        acc.add(sparse.csr_matrix(Z), y - o, w)
        y, w, o, Z = block(panel[panel["season"] == nxt], t)
        base.append((y, o, w))
        with_terms.append((y, o + Z @ acc.solve(1e-6), w))

    def r2(side_):
        return im._weighted_r2(*map(np.concatenate, zip(*side_)))
    return r2(base), r2(with_terms)


def run_skill(skill, pct, panel, box, box_prior, lam, perms, seed, impact=None):
    side = SKILLS[skill][0]
    flags = skill_flags(box, skill, pct)
    labels = Labels(panel, flags)
    n = labels.counts(flags["flag"].to_numpy())
    w = (panel["OffPoss"] if side == "off" else panel["DefPoss"]).to_numpy(float)

    fits = build_fits(panel, side, box_prior, lam)
    known_fits = build_fits(panel, side, box_prior, lam, mask=labels.known)
    n_bar = np.average(n, weights=w)
    out = {
        "skill": skill, "side": side, "pct": pct,
        "share": flags.loc[flags["eligible"], "flag"].mean(),
        "n_bar": n_bar,
        "dist": np.bincount(n, weights=w, minlength=6)[:5] / w.sum(),
        "known_share": w[labels.known].sum() / w.sum(),
    }
    for fe in (False, True):
        (_, b2), per = pooled(fits, n, fe)
        rng = np.random.default_rng(seed)
        null = np.array([pooled(fits, labels.counts(labels.shuffled(rng)), fe)[0][1]
                         for _ in range(perms)])
        out["fe" if fe else "plain"] = {
            "b2": b2,
            "null_sd": null.std(),
            "p": (np.abs(null) >= abs(b2)).mean() if perms else np.nan,
            "agree": sum(np.sign(v) == np.sign(b2) for v in per.values()),
            "seasons": len(per),
            "b2_known": pooled(known_fits, n, fe)[0][1],
            "extra": {k: 2 * b2 * (k - n_bar) for k in range(4)},
        }
    if impact is not None:
        out["r2"] = next_season_r2(panel, impact, labels, flags["flag"].to_numpy(), side)
    return out


def report(results):
    print("Curvature b2 (pts/100 per lineup; negative = diminishing returns)")
    print(f"{'skill':<8} {'side':<4} {'cut':>4} {'share':>6} {'n_bar':>5} {'model':<9} "
          f"{'b2':>7} {'null sd':>7} {'p':>5} {'same sign':>9} {'known-only':>10}")
    for r in results:
        for model in ("plain", "fe"):
            m = r[model]
            print(f"{r['skill']:<8} {r['side']:<4} {r['pct']:>4.2f} {r['share']:>6.1%} "
                  f"{r['n_bar']:>5.2f} {'team FE' if model == 'fe' else 'no FE':<9} "
                  f"{m['b2']:>+7.3f} {m['null_sd']:>7.3f} {m['p']:>5.3f} "
                  f"{m['agree']:>4}/{m['seasons']:<4} {m['b2_known']:>+10.3f}")
    print("\nLineup mix (share of possessions with 0/1/2/3/4 skilled players)")
    for r in results:
        print(f"  {r['skill']:<8} {r['pct']:.2f}  " + "  ".join(f"{v:5.1%}" for v in r["dist"]))
    print("\nWorth of one more skilled player vs. on the average lineup (pts/100)")
    print(f"  {'skill':<8} {'cut':>4} " + " ".join(f"{'has ' + str(k):>7}" for k in range(4)))
    for r in results:
        print(f"  {r['skill']:<8} {r['pct']:>4.2f} "
              + " ".join(f"{r['fe']['extra'][k]:>+7.2f}" for k in range(4)) + "   (team FE)")
    rows = [r for r in results if "r2" in r]
    if rows:
        print("\nNext season, weighted R^2 (impacts only -> + count terms)")
        for r in rows:
            base, new = r["r2"]
            print(f"  {r['skill']:<8} {r['side']:<4} {base:.5f} -> {new:.5f}  ({new - base:+.5f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=os.path.join("data", "lineup_panel.csv"))
    ap.add_argument("--impact", default=os.path.join("data", "player_impact.csv"))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--skills", nargs="+", choices=list(SKILLS), default=list(SKILLS))
    ap.add_argument("--lam", type=float, default=im.DEFAULT_LAM)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sensitivity", action="store_true",
                    help="also run each skill at a looser and a stricter cut")
    ap.add_argument("--write-curvature", metavar="CSV", nargs="?",
                    const=os.path.join("data", "creator_curvature.csv"),
                    help="write the creator b2 as of each season (what impact_recs.py "
                         "reads) instead of the report")
    args = ap.parse_args()

    panel = load_panel(args.panel)
    box = load_box(args.data_dir)
    box_prior = im.box_prior(os.path.join(args.data_dir, "Advanced.csv"))
    if args.write_curvature:
        curv = curvature_by_season("creator", panel, box, box_prior, args.lam)
        curv.round(4).to_csv(args.write_curvature, index=False)
        print(f"Wrote {len(curv)} seasons of creator curvature to '{args.write_curvature}'")
        return 0
    impact = pd.read_csv(args.impact)

    results = []
    for skill in args.skills:
        _, _, main_pct, alt = SKILLS[skill]
        for pct in [main_pct] + (list(alt) if args.sensitivity else []):
            results.append(run_skill(skill, pct, panel, box, box_prior, args.lam,
                                     args.perms, args.seed,
                                     impact if pct == main_pct else None))
            print(f"  done: {skill} @ {pct}", file=sys.stderr, flush=True)
    report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
