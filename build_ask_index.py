"""Embed the Ask tab's index with Amazon Titan (Bedrock).

The Ask tab retrieves stat chunks (one per player-season and team-season) by
vector search over backend/data/index/. They used to be embedded locally with
fastembed (BAAI/bge-small-en-v1.5, ONNX), which cost the Lambda image ~210 MB
(onnxruntime, sympy, tokenizers, the weights) to embed one question per /ask
call. Questions and chunks are now embedded by Bedrock's Titan Text Embeddings
v2, the same service /ask already calls for the answer, so the image carries
no model at all.

This re-embeds the chunk text already in the index metadata: the scripts that
first wrote it (chunk_data.py, embed_chunks.py) read the retired LightFM
pipeline's outputs and are in local_archive/. It also strips the fragments
that named LightFM features ("-- LightFM item feature pcluster=...").

Writes, per collection, {name}_emb.npy (float32, unit-normalised) and
{name}_meta.csv.gz, plus index_info.json naming the model and dimensions,
which rag.py and vector_index.py check at load. About 10,300 calls, ~1.5M
tokens (a few cents); needs bedrock:InvokeModel on the model in REGION.

Usage:
    python build_ask_index.py                 # rewrites backend/data/index/
    python build_ask_index.py --workers 4
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

INDEX_DIR = os.path.join("backend", "data", "index")
COLLECTIONS = ("nba_players", "nba_teams")
MODEL = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 512          # Titan v2 offers 256 / 512 / 1024
REGION = "us-west-2"
LIGHTFM_FRAGMENT = re.compile(r"\s+—\s+LightFM (?:item|user) feature [pt]cluster=[^.]*")
# Finished embeddings, one JSON line per text, so an interrupted run resumes
# instead of re-embedding everything. Gitignored; safe to delete.
CACHE = os.path.join("data", "ask_embed_cache.jsonl")
# Bedrock errors worth retrying: botocore retries throttling itself, but not
# ModelErrorException ("unexpected error ... try your request again").
TRANSIENT = ("ModelErrorException", "ThrottlingException", "ServiceUnavailableException",
             "InternalServerException", "ModelNotReadyException")


def clean(doc):
    return LIGHTFM_FRAGMENT.sub("", str(doc))


def _key(text):
    return hashlib.sha1(f"{MODEL}|{DIMENSIONS}|{text}".encode()).hexdigest()


def load_cache(path=CACHE):
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                out[rec["k"]] = rec["e"]
    return out


def embed_all(texts, workers, cache_path=CACHE, attempts=8):
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    client = boto3.client("bedrock-runtime", region_name=REGION,
                          config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                                        max_pool_connections=workers))
    cache = load_cache(cache_path)
    lock = threading.Lock()
    todo = sorted({t for t in texts if _key(t) not in cache})
    print(f"  {len(texts) - sum(_key(t) not in cache for t in texts)} cached, {len(todo)} to embed")

    def one(text):
        body = json.dumps({"inputText": text, "dimensions": DIMENSIONS, "normalize": True})
        for attempt in range(attempts):
            try:
                resp = client.invoke_model(modelId=MODEL, body=body)
                emb = json.loads(resp["body"].read())["embedding"]
                break
            except ClientError as e:
                if e.response["Error"]["Code"] not in TRANSIENT or attempt == attempts - 1:
                    raise
                time.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))
        with lock:
            cache[_key(text)] = emb
            with open(cache_path, "a") as f:
                f.write(json.dumps({"k": _key(text), "e": emb}) + "\n")
        return emb

    with ThreadPoolExecutor(workers) as pool:
        for done, _ in enumerate(pool.map(one, todo), 1):
            if done % 1000 == 0:
                print(f"  {done} / {len(todo)}")
    return np.asarray([cache[_key(t)] for t in texts], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--index-dir", default=INDEX_DIR)
    args = ap.parse_args()
    for name in COLLECTIONS:
        meta_path = os.path.join(args.index_dir, f"{name}_meta.csv.gz")
        meta = pd.read_csv(meta_path)
        meta["document"] = meta["document"].map(clean)
        emb = embed_all(meta["document"].tolist(), args.workers)
        np.save(os.path.join(args.index_dir, f"{name}_emb.npy"), emb)
        meta.to_csv(meta_path, index=False, compression="gzip")
        print(f"{name}: {emb.shape[0]} chunks embedded, {emb.shape[1]} dims")
    with open(os.path.join(args.index_dir, "index_info.json"), "w") as f:
        json.dump({"model": MODEL, "dimensions": DIMENSIONS, "normalize": True, "region": REGION}, f,
                  indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
