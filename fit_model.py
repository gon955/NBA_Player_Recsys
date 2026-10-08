"""Archetype fit: do some combinations beat the sum of their players?

Stage 1 (impact_model.py) explains a lineup's rating as the sum of its five
players' impacts. This stage fits what is left over:

    residual = lineup rating - sum(player impacts)
             = sum over the lineup's archetypes      m[a]      (main effects)
             + sum over its 10 archetype pairs       f[a, b]   (fit effects)
             + noise

m[a] picks up an archetype the impact model systematically over- or under-
rates (e.g. DBPM's lean toward centres); f[a, b] is the part that depends on
who a player shares the floor with -- the thing a fit recommender needs.
Both are possession-weighted ridge, offence and defence fitted separately, and
pooled across eras by archetype *name* (the style clusters reuse one
vocabulary: "Rim-Running Bigs" is a cluster in every era).

The test is the same as stage 1's: fit on seasons <= t, predict season t+1's
lineups from season-t impacts and archetypes, score by weighted R^2. Two
controls keep it honest:

  impacts only     stage 1 alone; fit terms must beat this
  shuffled labels  archetypes permuted among player-seasons within a season.
                   Any gain that survives shuffling is regularisation or
                   pooling, not basketball.

Results (weighted R^2, O / D; lam 1e5 within season, 1e6 next season):

                          next season        within season
  impacts only            .0204 / .0081      .0347 / .0156
  main only, real         .0219 / .0083      .0364 / .0167
  main + pairs, real      .0221 / .0082      .0366 / .0167
  main only, shuffled     .0218 / .0081      .0358 / .0158
  main + pairs, shuffled  .0218 / .0079      .0356 / .0156

Read: most of the next-season gain survives shuffling (it is an intercept and
the "unknown" bucket correcting stage 1's average bias). What shuffling can't
reproduce is mostly the archetype MAIN effects -- clearest within season, and
almost all of the defensive gain. Pair effects add ~.0002 on offence and
nothing on defence: archetype-pair fit is at the edge of what these data can
detect, and its coefficients (sd ~0.08, max ~0.23 pts/100 per pair) are small
next to player impacts (sd ~2.2). Treat them as a tie-breaker, not a ranking.

Usage:
    python fit_model.py --evaluate     # compare models, pick lam
    python fit_model.py                # fit on all seasons, write coefficients
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse

import impact_model as im

SLOTS = im.SLOTS
UNKNOWN = "unknown"
PAIRS = list(combinations(range(5), 2))

LAMBDAS = [1e5, 1e6, 1e7]
DEFAULT_LAM = 1e6


def load_inputs(panel_path, impact_path, clusters_path):
    panel = im.load_panel(panel_path)
    impact = pd.read_csv(impact_path)
    labels = pd.read_csv(clusters_path, usecols=["player_id", "season", "cluster_label"])
    labels["season"] = labels["season"].astype(int)
    return panel, impact, labels


def shuffle_labels(labels, seed):
    """Permute archetypes among player-seasons within each season."""
    rng = np.random.default_rng(seed)
    out = labels.copy()
    out["cluster_label"] = out.groupby("season")["cluster_label"].transform(
        lambda s: rng.permutation(s.to_numpy()))
    return out


def slot_labels(df, labels, label_season):
    """(n, 5) archetype names for each slot, as of `label_season` per row.

    Players without a row (under 20 games, or not in the league that season)
    are UNKNOWN rather than dropped, so every lineup keeps its possessions.
    """
    lookup = labels.set_index(["player_id", "season"])["cluster_label"]
    ids = df[SLOTS].to_numpy().ravel()
    seasons = np.repeat(np.asarray(label_season), 5)
    idx = pd.MultiIndex.from_arrays([ids, seasons])
    return lookup.reindex(idx).fillna(UNKNOWN).to_numpy().reshape(-1, 5)


class Vocab:
    """Column layout: one main effect per archetype, one per unordered pair."""

    def __init__(self, names, pairs=True):
        self.names = sorted(set(names) | {UNKNOWN})
        self.code = {n: i for i, n in enumerate(self.names)}
        k = len(self.names)
        self.pairs = [(a, b) for a in range(k) for b in range(a, k)] if pairs else []
        self.pair_col = {p: k + i for i, p in enumerate(self.pairs)}
        self.n_cols = k + len(self.pairs)

    def columns(self):
        main = [("main", n, "") for n in self.names]
        pair = [("pair", self.names[a], self.names[b]) for a, b in self.pairs]
        return pd.MultiIndex.from_tuples(main + pair, names=["kind", "a", "b"])

    def design(self, slot_names):
        """Sparse counts: 5 main-effect hits and 10 pair hits per lineup."""
        n = len(slot_names)
        codes = np.vectorize(lambda s: self.code.get(s, self.code[UNKNOWN]))(slot_names)
        rows, cols = [], []
        for k in range(5):
            rows.append(np.arange(n))
            cols.append(codes[:, k])
        k_names = len(self.names)
        for i, j in (PAIRS if self.pairs else []):
            a = np.minimum(codes[:, i], codes[:, j])
            b = np.maximum(codes[:, i], codes[:, j])
            # Index of (a, b) in the upper-triangular pair list.
            tri = a * k_names - a * (a - 1) // 2 + (b - a)
            rows.append(np.arange(n))
            cols.append(k_names + tri)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, self.n_cols))


def impact_sum(df, impact, impact_season):
    """Sum of the five players' impacts as of `impact_season` (missing -> 0)."""
    est = impact.set_index(["player_id", "season"])
    ids = df[SLOTS].to_numpy().ravel()
    idx = pd.MultiIndex.from_arrays([ids, np.repeat(np.asarray(impact_season), 5)])
    o = est["o_impact"].reindex(idx).fillna(0.0).to_numpy().reshape(-1, 5).sum(1)
    d = est["d_impact"].reindex(idx).fillna(0.0).to_numpy().reshape(-1, 5).sum(1)
    return o, d


def season_block(panel, impact, labels, vocab, season):
    """Design, residual targets and weights for one season, same-season inputs."""
    df = panel[panel["season"] == season]
    y_off, w_off, y_def, w_def = im._targets(df)
    o, d = impact_sum(df, impact, np.full(len(df), season))
    X = vocab.design(slot_labels(df, labels, np.full(len(df), season)))
    return X, y_off - o, w_off, y_def - d, w_def


def ridge(X, y, w, lam):
    Xw = X.multiply(w[:, None]).tocsr()
    A = (X.T @ Xw).toarray() + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, Xw.T @ y)


class _Accumulator:
    """Running X'WX and X'Wy, so an expanding window costs one pass."""

    def __init__(self, n):
        self.A = np.zeros((n, n))
        self.b = np.zeros(n)

    def add(self, X, y, w):
        Xw = X.multiply(w[:, None]).tocsr()
        self.A += (X.T @ Xw).toarray()
        self.b += Xw.T @ y

    def solve(self, lam):
        return np.linalg.solve(self.A + lam * np.eye(len(self.b)), self.b)


def evaluate_model(panel, impact, labels, vocab, lams, use_fit=True):
    """Next-season weighted R^2 (O, D) per lam, expanding training window."""
    seasons = sorted(panel["season"].unique())
    acc_o, acc_d = _Accumulator(vocab.n_cols), _Accumulator(vocab.n_cols)
    out = {lam: ([], []) for lam in lams}
    for t, nxt in zip(seasons[:-1], seasons[1:]):
        X, r_o, w_o, r_d, w_d = season_block(panel, impact, labels, vocab, t)
        acc_o.add(X, r_o, w_o)
        acc_d.add(X, r_d, w_d)

        df = panel[panel["season"] == nxt]
        y_off, w_off, y_def, w_def = im._targets(df)
        base_o, base_d = impact_sum(df, impact, np.full(len(df), t))
        X_next = vocab.design(slot_labels(df, labels, np.full(len(df), t)))
        for lam in lams:
            fo = X_next @ acc_o.solve(lam) if use_fit else 0.0
            fd = X_next @ acc_d.solve(lam) if use_fit else 0.0
            out[lam][0].append((y_off, base_o + fo, w_off))
            out[lam][1].append((y_def, base_d + fd, w_def))
    return {lam: tuple(im._weighted_r2(*map(np.concatenate, zip(*side))) for side in sides)
            for lam, sides in out.items()}


def evaluate_within_season(panel, labels, box, vocab, lams, folds=5, seed=0):
    """Held-out lineups within each season, impacts refitted without them.

    Less harsh than next-season prediction: impacts and archetypes are current,
    so a fit effect isn't drowned by players changing between seasons. Returns
    ({lam: (O, D)}, (O, D) for impacts alone).
    """
    fold = np.random.default_rng(seed).integers(0, folds, len(panel))
    seasons = sorted(panel["season"].unique())
    res = {lam: ([], []) for lam in lams}
    base = ([], [])
    for k in range(folds):
        acc_o, acc_d = _Accumulator(vocab.n_cols), _Accumulator(vocab.n_cols)
        held = []
        for s in seasons:
            in_season = (panel["season"] == s).to_numpy()
            train, test = panel[in_season & (fold != k)], panel[in_season & (fold == k)]
            prior = box.xs(s, level="season")
            est = im.fit_season(train, im.DEFAULT_LAM, prior["o"], prior["d"]).assign(season=s)
            for part, is_train in ((train, True), (test, False)):
                y_o, w_o, y_d, w_d = im._targets(part)
                o, d = impact_sum(part, est, np.full(len(part), s))
                X = vocab.design(slot_labels(part, labels, np.full(len(part), s)))
                if is_train:
                    acc_o.add(X, y_o - o, w_o)
                    acc_d.add(X, y_d - d, w_d)
                else:
                    held.append((X, y_o, o, w_o, y_d, d, w_d))
        for _X, y_o, o, w_o, y_d, d, w_d in held:
            base[0].append((y_o, o, w_o))
            base[1].append((y_d, d, w_d))
        for lam in lams:
            bo, bd = acc_o.solve(lam), acc_d.solve(lam)
            for X, y_o, o, w_o, y_d, d, w_d in held:
                res[lam][0].append((y_o, o + X @ bo, w_o))
                res[lam][1].append((y_d, d + X @ bd, w_d))

    def r2(side):
        return im._weighted_r2(*map(np.concatenate, zip(*side)))
    return {lam: (r2(o), r2(d)) for lam, (o, d) in res.items()}, (r2(base[0]), r2(base[1]))


def evaluate(panel, impact, labels, box):
    names = labels["cluster_label"]
    shuffled = shuffle_labels(labels, 0)
    variants = [
        ("main only, real", labels, Vocab(names, pairs=False)),
        ("main + pairs, real", labels, Vocab(names)),
        ("main only, shuffled", shuffled, Vocab(names, pairs=False)),
        ("main + pairs, shuffled", shuffled, Vocab(names)),
    ]
    base = evaluate_model(panel, impact, labels, Vocab(names), [1.0], use_fit=False)[1.0]
    print("Next season (fit on seasons <= t, predict t+1)")
    print(f"  {'impacts only':<24} {'':>6} O {base[0]:.4f}  D {base[1]:.4f}")
    for name, lab, vocab in variants:
        for lam, (o, d) in evaluate_model(panel, impact, lab, vocab, LAMBDAS).items():
            print(f"  {name:<24} {lam:6.0e} O {o:.4f}  D {d:.4f}", flush=True)
    print("Within season (5-fold held-out lineups)")
    for i, (name, lab, vocab) in enumerate(variants):
        res, base = evaluate_within_season(panel, lab, box, vocab, [1e5, DEFAULT_LAM])
        if i == 0:
            print(f"  {'impacts only':<24} {'':>6} O {base[0]:.4f}  D {base[1]:.4f}")
        for lam, (o, d) in res.items():
            print(f"  {name:<24} {lam:6.0e} O {o:.4f}  D {d:.4f}", flush=True)


def fit_all(panel, impact, labels, lam):
    """Coefficients on every season, as a tidy table."""
    vocab = Vocab(labels["cluster_label"])
    acc_o, acc_d = _Accumulator(vocab.n_cols), _Accumulator(vocab.n_cols)
    exposure = np.zeros(vocab.n_cols)
    for season in sorted(panel["season"].unique()):
        X, r_o, w_o, r_d, w_d = season_block(panel, impact, labels, vocab, season)
        acc_o.add(X, r_o, w_o)
        acc_d.add(X, r_d, w_d)
        exposure += X.T @ (w_o + w_d) / 2
    o, d = acc_o.solve(lam), acc_d.solve(lam)
    coef = pd.DataFrame({"o_effect": o, "d_effect": d, "net_effect": o + d,
                         "possessions": exposure}, index=vocab.columns())
    return coef.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=os.path.join("data", "lineup_panel.csv"))
    ap.add_argument("--impact", default=os.path.join("data", "player_impact.csv"))
    ap.add_argument("--clusters", default="master_clustered.csv",
                    help="player_cluster.py output: one style archetype per player-season")
    ap.add_argument("--adv", default=os.path.join("data", "Advanced.csv"),
                    help="box-score prior for the impact refits in --evaluate")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAM)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--out", default=os.path.join("data", "archetype_fit.csv"))
    args = ap.parse_args()

    panel, impact, labels = load_inputs(args.panel, args.impact, args.clusters)
    if args.evaluate:
        evaluate(panel, impact, labels, im.box_prior(args.adv))
        return 0

    coef = fit_all(panel, impact, labels, args.lam)
    coef.round(4).to_csv(args.out, index=False)
    print(f"Wrote {len(coef)} coefficients to '{args.out}' (lam={args.lam:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
