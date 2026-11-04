"""Serves the impact recommender's precomputed lists.

There is no model at runtime: export_recs.py (repo root) runs impact_recs.py
and writes data/recs.csv, one ranked list per team-season, already filtered
to moves the team could legally make. This module only looks them up.
Regenerate after any pipeline change with `python export_recs.py`.
"""

import math
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECS_PATH = os.path.join(BASE_DIR, "data", "recs.csv")

recs = pd.read_csv(RECS_PATH).rename(columns={"with": "with_"}).sort_values(["user_id", "rank"])
_by_user = {uid: g for uid, g in recs.groupby("user_id", sort=False)}
MAX_K = int(recs.groupby("user_id").size().min())


def team_seasons() -> dict[str, list[int]]:
    """team -> sorted seasons with a list."""
    parts = recs["user_id"].drop_duplicates().str.rsplit("_", n=1, expand=True)
    parts[1] = parts[1].astype(int)
    return {team: sorted(g[1].tolist()) for team, g in parts.groupby(0)}


def _num(x):
    """NaN -> None, so the JSON stays valid."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def recommend(user_id: str, k: int = 10) -> list[dict]:
    """Top-k for a team-season ('Denver Nuggets_2024'). KeyError if unknown."""
    if user_id not in _by_user:
        raise KeyError(f"No recommendations for '{user_id}'.")
    rows = _by_user[user_id].head(max(1, min(k, MAX_K)))
    out = []
    for r in rows.itertuples(index=False):
        value, rep_value = _num(r.value), _num(r.replaces_value)
        out.append({
            "player": r.player,
            "player_id": r.player_id,
            "score": float(r.score),
            "explanation": {
                "age": _num(r.age),
                "bpm": _num(r.bpm),
                "value": value,
                "position_value": _num(r.position_value),
                "replaces": r.replaces,
                "replaces_id": r.replaces_id,
                "replaces_value": rep_value,
                # Others sent out to match salary, when his alone does not.
                "with": [w for w in str(r.with_).split(";") if w and w != "nan"],
                "with_salary": _num(r.with_salary),
                # Per 100 possessions, ignoring minutes, and against a player who may
                # only be salary going out. The frontend no longer shows it; it stays
                # until no deployed frontend reads it.
                "upgrade": None if value is None or rep_value is None else value - rep_value,
                "salary": _num(r.salary),
                "replaces_salary": _num(r.replaces_salary),
                "cap": _num(r.cap),
                "minutes": _num(r.minutes),
            },
        })
    return out
