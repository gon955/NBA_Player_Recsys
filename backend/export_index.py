"""
export_index.py — BUILD-TIME ONLY. Dumps the Chroma collections to flat files.

WHY: Chroma's PersistentClient opens its SQLite store read-write, so it raises
"attempt to write a readonly database" on a Lambda container image (everything
outside /tmp is read-only). The index is only ~8.6k vectors, which is far below
the point where HNSW beats a brute-force scan, so at runtime we drop chromadb
entirely and search the vectors with numpy (see vector_index.py).

Run this whenever embed_chunks.py regenerates chroma_db, and commit the output:

    python backend/export_index.py

chromadb is a dependency of THIS script only — it is not installed at runtime.
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
OUT_DIR = os.path.join(BASE_DIR, "data", "index")

# collection name -> the metadata column naming its subject
COLLECTIONS = {"nba_players": "player", "nba_teams": "team"}


def export():
    import chromadb  # build-time dep; deliberately not a runtime import

    os.makedirs(OUT_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    for name, subject in COLLECTIONS.items():
        col = client.get_collection(name)
        got = col.get(include=["embeddings", "documents", "metadatas"])

        emb = np.asarray(got["embeddings"], dtype=np.float32)

        df = pd.DataFrame(got["metadatas"])
        df.insert(0, "id", got["ids"])
        df["document"] = got["documents"]
        # Column order is load-bearing only for readability; vector_index reads by name.
        cols = ["id", subject, "season", "era", "cluster_label", "document"]
        df = df[[c for c in cols if c in df.columns]]

        if len(df) != len(emb):
            raise RuntimeError(f"{name}: {len(df)} rows but {len(emb)} embeddings")

        emb_path = os.path.join(OUT_DIR, f"{name}_emb.npy")
        meta_path = os.path.join(OUT_DIR, f"{name}_meta.csv.gz")
        np.save(emb_path, emb)
        # gzip: ~0.5 MB vs ~4.4 MB raw, and unlike pickle it survives a pandas
        # version skew between the build env and the runtime image.
        df.to_csv(meta_path, index=False, compression="gzip")

        print(
            f"{name}: {emb.shape[0]} x {emb.shape[1]} -> "
            f"{os.path.basename(emb_path)} ({os.path.getsize(emb_path)/1e6:.1f} MB), "
            f"{os.path.basename(meta_path)} ({os.path.getsize(meta_path)/1e6:.1f} MB)"
        )


if __name__ == "__main__":
    export()
