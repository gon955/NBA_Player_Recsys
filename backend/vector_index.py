"""
vector_index.py — the numpy replacement for Chroma at query time.

WHY THIS EXISTS:
Two reasons, one hard and one soft.

HARD: chromadb.PersistentClient opens SQLite read-write, which fails outright on
a Lambda container image ("attempt to write a readonly database") because
everything outside /tmp is read-only. Copying 69 MB of chroma_db into /tmp on
every cold start would work, but it buys nothing over the alternative.

SOFT: the whole corpus is 8,640 vectors x 384 dims. That is small enough that a
brute-force scan beats an HNSW lookup once you count index load time — a
filtered scan measures ~3 ms. Dropping chromadb also removes a ~0.56 s import
and its dependency tree from the image.

Search reproduces Chroma's default metric (squared L2) and returns byte-identical
top-k to the collections it replaced. Files come from export_index.py.

Loading is lazy — importing this module touches no disk, so a cold start that
only serves /health or /recommendations never pays for the index.
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "data", "index")


class _Index:
    """Embeddings + row metadata for one collection. Built once, never mutated,
    so it is safe to share across concurrent requests."""

    def __init__(self, emb, meta):
        self.emb = emb                                  # (N, D) float32
        self.docs = meta["document"].tolist()
        self.era = meta["era"].astype(str).to_numpy()
        # ||e||^2, precomputed: the ranking term of squared L2 that does not
        # depend on the query. Lets query() skip materializing an (N, D)
        # difference array — that temp is 12 MB per call on the player index.
        self.sq_norms = np.einsum("ij,ij->i", emb, emb)
        self._era_rows = {}

    def _rows_for_era(self, era):
        if era not in self._era_rows:
            self._era_rows[era] = np.flatnonzero(self.era == era)
        return self._era_rows[era]

    def query(self, embedding, n_results=5, era=None):
        """Return the n_results closest documents, nearest first.

        Mirrors Chroma's `query(query_embeddings=..., n_results=..., where={"era": era})`.
        """
        q = np.asarray(embedding, dtype=np.float32)

        if era is None:
            rows = None
            emb, sq = self.emb, self.sq_norms
        else:
            rows = self._rows_for_era(era)
            if rows.size == 0:
                return []
            emb, sq = self.emb[rows], self.sq_norms[rows]

        # argmin over ||e - q||^2 == argmin over ||e||^2 - 2*e.q; the +||q||^2
        # term is constant across candidates, so it cannot change the ranking.
        scores = sq - 2.0 * (emb @ q)

        k = min(n_results, scores.shape[0])
        # argpartition is O(N) vs argsort's O(N log N); we only sort the k head.
        top = np.argpartition(scores, k - 1)[:k]
        top = top[np.argsort(scores[top], kind="stable")]
        if rows is not None:
            top = rows[top]
        return [self.docs[i] for i in top]


_INDEXES = {}


def get_index(name):
    """Lazily load (and cache) one collection. `name` is 'nba_players' or 'nba_teams'."""
    if name not in _INDEXES:
        emb_path = os.path.join(INDEX_DIR, f"{name}_emb.npy")
        meta_path = os.path.join(INDEX_DIR, f"{name}_meta.csv.gz")
        if not os.path.exists(emb_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"vector_index: missing index files for '{name}' in {INDEX_DIR}. "
                f"Run `python backend/export_index.py` against chroma_db and commit "
                f"the output, or bundle data/index/ into the deployment artifact."
            )
        emb = np.load(emb_path)
        meta = pd.read_csv(meta_path)
        if len(meta) != len(emb):
            raise RuntimeError(
                f"vector_index: '{name}' has {len(meta)} metadata rows but "
                f"{len(emb)} embeddings — re-run export_index.py."
            )
        _INDEXES[name] = _Index(emb, meta)
    return _INDEXES[name]
