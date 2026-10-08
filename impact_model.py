"""Player impact per season from lineup data: possession-weighted ridge (RAPM-style).

Each row of data/lineup_panel.csv is one 5-man unit's season with a team. Its
offensive rating relative to the league is modelled as the sum of its five
players' offensive effects, and the same on defence:

    100 * Points / OffPoss          - league ORtg  =  sum(o_i)  + noise
    100 * OpponentPoints / DefPoss  - league ORtg  = -sum(d_i)  + noise

so o_i and d_i are points per 100 possessions vs. league average, positive is
good on both ends, and o_i + d_i is the player's net impact. Rows are weighted
by possessions (the noise variance of a rate scales with 1/possessions) and the
coefficients are shrunk toward a prior with strength `lam`, in possessions.

What this is NOT: real RAPM regresses on stints with the *opposing* five on the
floor too. pbpstats only gives season totals per unit, so opponent quality is
not adjusted for -- a bench unit that mostly faced other benches looks better
than it is. The prior and the out-of-sample check below are how that is kept
honest, not a fix for it.

Priors compared (`--evaluate`):
  zero  plain ridge toward league average
  box   toward OBPM/DBPM, shrunk toward 0 for low-minute players
  prev  toward the player's previous-season estimate, else the box prior

Each is scored on how well season t's estimates predict season t+1's lineup
ratings (weighted R^2 vs. predicting league average), which is the use the
recommender will put them to. Results, 1999-2025:

  BPM alone, no lineups      O 0.0197  D 0.0075
  zero prior, best lam 4000  O 0.0139  D 0.0056   worse than BPM
  prev prior, lam 8000       O 0.0185  D 0.0084
  box prior,  lam 16000      O 0.0204  D 0.0081   default; beats BPM in 24/26
                                                  seasons on O, 20/26 on D

R^2 that low is mostly the data: with ~1 point^2 of variance per possession,
roughly three quarters of the weighted variance in lineup ratings is possession
noise, so ~0.26 is a rough ceiling even for perfect knowledge of each lineup.
For regular players the estimates correlate 0.99 (O) / 0.96 (D) with the box
prior -- the lineups refine BPM, they don't overturn it.

Usage:
    python impact_model.py --evaluate    # choose lam per prior, compare priors
    python impact_model.py               # fit every season, write the CSV
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

from helper import season_totals

SLOTS = [f"p{k}_id" for k in range(1, 6)]

# BPM from a few hundred minutes is mostly noise; pull it toward 0 by minutes
# so a 9-game call-up doesn't arrive with a +8 prior.
PRIOR_MP_HALF = 500

LAMBDAS = [1000, 2000, 4000, 8000, 16000, 32000]
PRIORS = ["zero", "box", "prev"]
DEFAULT_PRIOR = "box"
DEFAULT_LAM = 16000


def load_panel(path):
    cols = SLOTS + ["season", "OffPoss", "DefPoss", "Points", "OpponentPoints"]
    panel = pd.read_csv(path, usecols=cols)
    panel["season"] = panel["season"].astype(int)
    return panel


def box_prior(adv_path):
    """Minutes-shrunk OBPM/DBPM per player-season (TOT row for traded players)."""
    adv = pd.read_csv(adv_path, usecols=["season", "player_id", "team", "mp", "obpm", "dbpm"])
    adv = season_totals(adv[(adv["season"] >= 1999) & (adv["mp"] > 0)])
    shrink = adv["mp"] / (adv["mp"] + PRIOR_MP_HALF)
    return pd.DataFrame({
        "player_id": adv["player_id"],
        "season": adv["season"].astype(int),
        "o": adv["obpm"].fillna(0.0) * shrink,
        "d": adv["dbpm"].fillna(0.0) * shrink,
    }).set_index(["player_id", "season"])


def _targets(df):
    """Per-lineup offence/defence targets vs. this sample's league rating."""
    lg = 100 * (df["Points"].sum() + df["OpponentPoints"].sum()) / (
        df["OffPoss"].sum() + df["DefPoss"].sum())
    off = df["OffPoss"] > 0
    dfn = df["DefPoss"] > 0
    y_off = np.where(off, 100 * df["Points"] / df["OffPoss"].where(off, 1) - lg, 0.0)
    # Sign-flipped so a positive defensive effect means fewer points allowed.
    y_def = np.where(dfn, lg - 100 * df["OpponentPoints"] / df["DefPoss"].where(dfn, 1), 0.0)
    return y_off, df["OffPoss"].to_numpy(float), y_def, df["DefPoss"].to_numpy(float)


def design(df, players):
    """Sparse lineup x player indicator matrix, five ones per row."""
    col = pd.Index(players)
    rows = np.repeat(np.arange(len(df)), 5)
    cols = col.get_indexer(df[SLOTS].to_numpy().ravel())
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(col)))


def ridge(X, y, w, lam, prior):
    """argmin sum w (y - Xb)^2 + lam |b - prior|^2, solved in closed form.

    A season has ~500 players, so the normal equations are a small dense
    solve. No intercept: y is already centred on the league rating.
    """
    Xw = X.multiply(w[:, None]).tocsr()
    A = (X.T @ Xw).toarray() + lam * np.eye(X.shape[1])
    b = Xw.T @ (y - X @ prior)
    return prior + np.linalg.solve(A, b)


def fit_season(df, lam, prior_o, prior_d):
    """Fit one season. prior_* are Series indexed by player_id (missing -> 0)."""
    players = pd.unique(df[SLOTS].to_numpy().ravel())
    X = design(df, players)
    y_off, w_off, y_def, w_def = _targets(df)
    p_o = prior_o.reindex(players).fillna(0.0).to_numpy()
    p_d = prior_d.reindex(players).fillna(0.0).to_numpy()
    o = ridge(X, y_off, w_off, lam, p_o)
    d = ridge(X, y_def, w_def, lam, p_d)

    exposure = X.T @ np.column_stack([w_off, w_def])
    return pd.DataFrame({
        "player_id": players,
        "o_impact": o,
        "d_impact": d,
        "net_impact": o + d,
        "off_poss": exposure[:, 0],
        "def_poss": exposure[:, 1],
    })


def _prior_for(kind, season, box, prev):
    if kind == "zero":
        return pd.Series(dtype=float), pd.Series(dtype=float)
    try:
        b = box.xs(season, level="season")
    except KeyError:
        b = pd.DataFrame(columns=["o", "d"])
    if kind == "box" or prev is None:
        return b["o"], b["d"]
    p = prev.set_index("player_id")
    o = p["o_impact"].combine_first(b["o"])
    d = p["d_impact"].combine_first(b["d"])
    return o, d


def fit_all(panel, box, lam, prior_kind):
    """Fit every season in order; `prev` chains each season into the next."""
    out, prev = [], None
    for season in sorted(panel["season"].unique()):
        prior_o, prior_d = _prior_for(prior_kind, season, box, prev)
        est = fit_season(panel[panel["season"] == season], lam, prior_o, prior_d)
        est["season"] = season
        out.append(est)
        prev = est
    return pd.concat(out, ignore_index=True)


def _weighted_r2(y, yhat, w):
    """1 - SSE / SSE of predicting league average (0). Can be negative."""
    return 1 - np.sum(w * (y - yhat) ** 2) / np.sum(w * y ** 2)


def score_next_season(panel, est):
    """Predict season t+1 lineups from season-t estimates (unseen players = 0)."""
    off, dfn = [], []
    for season in sorted(panel["season"].unique())[1:]:
        prev = est[est["season"] == season - 1].set_index("player_id")
        if prev.empty:
            continue
        df = panel[panel["season"] == season]
        y_off, w_off, y_def, w_def = _targets(df)
        ids = df[SLOTS].to_numpy()
        o = prev["o_impact"].reindex(ids.ravel()).fillna(0.0).to_numpy().reshape(ids.shape).sum(1)
        d = prev["d_impact"].reindex(ids.ravel()).fillna(0.0).to_numpy().reshape(ids.shape).sum(1)
        off.append((y_off, o, w_off))
        dfn.append((y_def, d, w_def))
    return tuple(_weighted_r2(*map(np.concatenate, zip(*side))) for side in (off, dfn))


def box_as_estimates(panel, box):
    """The box prior on its own, shaped like fit_all's output, as a baseline."""
    b = box.reset_index().rename(columns={"o": "o_impact", "d": "d_impact"})
    return b[b["season"].isin(panel["season"].unique())]


def evaluate(panel, box):
    print(f"{'prior':>6} {'lam':>6} {'O R2':>7} {'D R2':>7}   (next-season lineup ratings)")
    o, d = score_next_season(panel, box_as_estimates(panel, box))
    print(f"{'BPM':>6} {'-':>6} {o:7.4f} {d:7.4f}   box score alone, no lineups")
    best = {}
    for kind in PRIORS:
        for lam in LAMBDAS:
            o, d = score_next_season(panel, fit_all(panel, box, lam, kind))
            print(f"{kind:>6} {lam:>6} {o:7.4f} {d:7.4f}", flush=True)
            if o + d > best.get(kind, (None, -np.inf))[1]:
                best[kind] = (lam, o + d)
    print("best lam per prior:", {k: v[0] for k, v in best.items()})
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=os.path.join("data", "lineup_panel.csv"))
    ap.add_argument("--adv", default=os.path.join("data", "Advanced.csv"))
    ap.add_argument("--prior", choices=PRIORS, default=DEFAULT_PRIOR)
    ap.add_argument("--lam", type=float, default=None,
                    help="ridge strength in possessions (default: chosen by --evaluate)")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--out", default=os.path.join("data", "player_impact.csv"))
    args = ap.parse_args()

    panel = load_panel(args.panel)
    box = box_prior(args.adv)

    if args.evaluate:
        evaluate(panel, box)
        return 0

    lam = args.lam if args.lam is not None else DEFAULT_LAM
    est = fit_all(panel, box, lam, args.prior)
    est = est[["player_id", "season", "o_impact", "d_impact", "net_impact",
               "off_poss", "def_poss"]].round(3)
    est.to_csv(args.out, index=False)
    print(f"Wrote {len(est)} player-seasons to '{args.out}' (prior={args.prior}, lam={lam:g})")
    return 0



if __name__ == "__main__":
    sys.exit(main())
