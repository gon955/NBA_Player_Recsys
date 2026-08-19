"""
similarity.py — "who plays like X" via stat-profile nearest neighbors.
 
WHY THIS EXISTS (and why it isn't in rec_sys.py):
The LightFM embeddings encode ROSTER CO-OCCURRENCE (team-season <-> player-season
interactions), so their nearest neighbors are teammates, not stylistic comps.
That signal is correct for "which players fit this team" — it's just the wrong
signal for "who plays like Nash". Playing style lives in the per-100 rate stats,
so style similarity is a separate computation over those columns.
 
Note this module does NOT load models_all.joblib — it's pure pandas/numpy over
the stat CSV, so it's cheap to import.
 
ERA HANDLING: stats are z-scored WITHIN each era, so every player is expressed
as "how far from his era's peers" rather than in raw units. That makes
cross-era comparison meaningful (a 40% 3P shooter in 2003 and one in 2023 sit
at different z-scores), which is exactly the era-aware framing of the project.
"""
 
import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "players.csv"),      # runtime / container (primary)
    os.path.join(BASE_DIR, "players.csv"),
    os.path.join(PROJECT_ROOT, "players.csv"),          # build-time pipeline output
    os.path.join(BASE_DIR, "data", "master_clustered.csv"),
    os.path.join(PROJECT_ROOT, "master_clustered.csv"),
]
CSV_PATH = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])


MIN_MP = 500

# Style features only: per-100 rates + efficiency + usage/role rates.
# Deliberately EXCLUDED: mp/g/gs (availability), ws/ows/dws/ws_48/vorp/bpm/per
# (accumulated value & quality, not style), trb (redundant with orb+drb).
# One measure per concept — no more five correlated efficiency columns.
STYLE_COLS = [
    # playmaking — the defining axis for this archetype
    "ast_percent", "ast_per_100_poss", "tov_percent",
    # efficiency
    "ts_percent", "x3p_percent", "ft_percent",
    # volume / usage (Nash = low volume, so these matter as *low* z-scores)
    "usg_percent", "fga_per_100_poss", "pts_per_100_poss",
    # shot selection
    "x3p_ar", "f_tr", "x3pa_per_100_poss",
    # role/positional context — kept light so it separates guards from wings
    "orb_percent", "drb_percent", "stl_percent", "blk_percent",
]

WEIGHTS = {
    "ast_percent": 2.0, "ast_per_100_poss": 2.0, "tov_percent": 1.5,
    "ts_percent": 1.5, "x3p_percent": 1.0, "ft_percent": 1.0,
    "usg_percent": 1.5, "fga_per_100_poss": 1.5, "pts_per_100_poss": 1.0,
    "x3p_ar": 1.0, "f_tr": 1.0, "x3pa_per_100_poss": 1.0,
    "orb_percent": 0.75, "drb_percent": 0.75,
    "stl_percent": 0.5, "blk_percent": 0.75,
}
 
 
class _Model:
    """Immutable, fully-derived state: the stat matrix plus the lookup indices.

    Everything is built once by `get_model()` and never mutated, so it's safe to
    share across concurrent requests.
    """

    def __init__(self, df, X_n, cols, typical_nn):
        self.df = df
        self.X_n = X_n
        self.cols = cols
        self.typical_nn = typical_nn
        # name(lower) -> row indices, for resolving "steve nash" out of a question
        self.name_index = {}
        for i, nm in enumerate(df["player"].astype(str).str.lower()):
            self.name_index.setdefault(nm, []).append(i)
        # longest match wins, so "steve nash" beats "steve" in a query scan
        self.all_names = sorted(self.name_index.keys(), key=len, reverse=True)


_MODEL = None


def _typical_nn(X_n):
    """Median nearest-neighbor distance over a random sample — the yardstick for
    "is this comp real?". X_n rows are unit vectors, so euclidean distance is
    sqrt(2 - 2·cos); we get every cosine from ONE (N x sample) matmul instead of
    materializing the full (sample x N x F) difference tensor (that was ~0.5 GB
    and OOM-killed small containers at load time)."""
    rng = np.random.default_rng(0)
    samp = rng.choice(len(X_n), size=min(500, len(X_n)), replace=False)
    dots = X_n @ X_n[samp].T                          # (N, S), ~N*S*8 bytes
    d = np.sqrt(np.clip(2.0 - 2.0 * dots, 0.0, None))
    d[samp, np.arange(len(samp))] = np.inf            # drop each sample's self-pair
    return float(np.median(d.min(axis=0)))


def _build_model():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"similarity: no data file found. Looked for players.csv / "
            f"master_clustered.csv in {BASE_DIR} and {PROJECT_ROOT}. Bundle the "
            f"CSV into the deployment artifact."
        )
    df = pd.read_csv(CSV_PATH, low_memory=False)


    required = ("player", "player_id", "season", "era")
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"similarity: {os.path.basename(CSV_PATH)} is missing required "
            f"columns {missing_required} — cannot build the style model."
        )

    # Drop multi-team aggregate rows if present, then one row per player-season.
    team_col = "team_per100" if "team_per100" in df.columns else None
    if team_col:
        df = df[~df[team_col].isin(["2TM", "3TM", "4TM", "5TM"])]
    df = df.dropna(subset=["player_id", "season"])
    df = df.drop_duplicates(subset=["player_id", "season"], keep="last")

    df["season"] = df["season"].astype(int)
    if "item_id" not in df.columns:
        df["item_id"] = df["player_id"].astype(str) + "_" + df["season"].astype(str)
    if "display_name" not in df.columns:
        df["display_name"] = df["player"].astype(str) + " (" + df["season"].astype(str) + ")"

    cols = [c for c in STYLE_COLS if c in df.columns]
    missing = set(STYLE_COLS) - set(cols)
    if missing:
        logger.info("similarity: missing columns skipped: %s", sorted(missing))

    # Z-score WITHIN era so each value means "vs. this player's era peers".
    # NaN percentages (e.g. 3P% with zero attempts) become 0 = era-average,
    # which is neutral rather than extreme.
    z = df.groupby("era")[cols].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0)
    )
    z = z.fillna(0.0)

    # weight groups per the target archetype definition
    w = np.array([WEIGHTS.get(c, 1.0) for c in cols], dtype=np.float64)
    X = z.to_numpy(dtype=np.float64) * w
    X_n = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)

    return _Model(df.reset_index(drop=True), X_n, cols, _typical_nn(X_n))


def get_model():
    """Lazily build (and cache) the model on first use. Kept OUT of import so
    importing this module has no side effects — the CSV read and matrix build
    happen on first query, not at container/cold-start init."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _build_model()
    return _MODEL


def resolve_player(text: str):
    """Find the player named anywhere in `text`. 'prime steve nash' -> 'steve nash'."""
    m = get_model()
    t = str(text).lower()
    if t in m.name_index:
        return t
    for name in m.all_names:
        if name in t:
            return name
    return None


def _peak_row(df, rows):
    """Pick the player's 'prime' season: most minutes played."""
    if "mp" in df.columns:
        return max(rows, key=lambda i: (df.at[i, "mp"] if pd.notna(df.at[i, "mp"]) else -1))
    return rows[0]

 
def similar_players(name_or_query: str, era: str = None, k: int = 5, season: int = None):
    """
    k most statistically similar player-seasons to the named player's prime.
 
    era:    restrict CANDIDATES to this era (e.g. "2016-present" for
            "modern players like prime Nash"). None = search all eras.
    season: pin the query to a specific season instead of the peak-MP one.
 
    Returns [{"item_id", "label", "player", "season", "era", "score"}], best
    first, one row per player, query player excluded.
    """
    m = get_model()
    name = resolve_player(name_or_query)
    if name is None:
        return []

    df, X_n = m.df, m.X_n
    rows = m.name_index[name]
    if season is not None:
        rows = [i for i in rows if int(df.at[i, "season"]) == int(season)] or rows
    qi = _peak_row(df, rows)
    q = X_n[qi]

    d = np.linalg.norm(X_n - q, axis=1)
    sims = 1.0 / (1.0 + d)

    mask = np.ones(len(df), dtype=bool)
    mask &= df["player_id"].to_numpy() != df.at[qi, "player_id"]   # exclude self
    if "mp" in df.columns:
        mask &= df["mp"].fillna(0).to_numpy() >= MIN_MP           # kill small-sample noise
    if era is not None:
        mask &= df["era"].astype(str).to_numpy() == era

    cand = np.flatnonzero(mask)
    if cand.size == 0:
        return []
    cand = cand[np.argsort(-sims[cand])]

    best_d = d[cand[0]]
    weak = bool(best_d > 1.5 * m.typical_nn)

    out, seen = [], set()
    for i in cand:
        pid = df.at[i, "player_id"]
        if pid in seen:               # one season per player
            continue
        seen.add(pid)
        out.append({
            "item_id": str(df.at[i, "item_id"]),
            "label": str(df.at[i, "display_name"]),
            "player": str(df.at[i, "player"]),
            "season": int(df.at[i, "season"]),
            "era": str(df.at[i, "era"]),
            "score": float(sims[i]),
            "weak_comp": weak,
        })
        if len(out) >= k:
            break
    return out
 
 

