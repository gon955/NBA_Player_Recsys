# NBA Recommendation System

An era-aware NBA roster recommender with a grounded question-answering layer. Four parts:

- **Offline modeling pipeline** (`player_cluster.py`, `team_cluster.py`, `rec_sys.py`) — clusters player-seasons and team-seasons into archetypes *within era*, builds the interaction matrix, and trains one LightFM model per era.
- **Retrieval pipeline** (`chunk_data.py`, `embed_chunks.py`, `backend/export_index.py`) — renders every player-season and team-season as a natural-language stat chunk, embeds it, and exports a flat vector index the runtime can search without a database.
- **FastAPI backend** (`backend/`) — recommendations with feature-level explanations, cluster summaries, static assets, and a `/ask` endpoint that answers questions over the retrieved chunks via AWS Bedrock.
- **Next.js frontend** (`nba-recs-frontend/`) — three tabs: Recommend, Clusters, and Ask.

The backend ships as a Lambda container image behind a Function URL. The same code runs under `uvicorn` locally — `lambda_handler.py` wraps the FastAPI app in Mangum, so there is no branching inside the routes.

Eras are fixed at three: `1999-2007`, `2008-2015`, `2016-present` (see `era_of()` in `helper.py`).


## Getting Started

### 1. Environment

**Python 3.9 is required.** `lightfm==1.17` does not build on 3.11+, which constrains the whole stack — it is also why the Dockerfile uses `python:3.9-slim` rather than an AWS Lambda base image (the official python3.9 base image is past EOL). Lambda accepts any image implementing the Runtime API, which `awslambdaric` provides.

```bash
conda env create -f environment.yml && conda activate recsys   # offline pipeline
pip install -r backend/requirements.txt                        # backend runtime
pip install -r backend/requirements-build.txt                  # + chromadb, for index builds only
```

`backend/requirements.txt` is runtime-only on purpose. `chromadb` lives in `requirements-build.txt` because it is needed to *build* the index and never to serve it — see [Vector search](#vector-search) below.

Node.js ≥ 18 for the Next.js app (the frontend image uses `node:20-alpine`).

AWS credentials with `bedrock:InvokeModel` in `us-west-2` are needed for `/ask`. In Lambda this is the execution role; locally, standard credentials from the environment. There is no API key to set — the migration off Groq replaced key auth with IAM.

### 2. Data preparation & modeling

Raw Basketball-Reference CSVs live in `data/` (already populated).

```bash
python player_cluster.py     # player archetypes, PCA plots, cluster counts
python team_cluster.py       # team archetypes, same outputs
python rec_sys.py            # trains per-era LightFM models -> models_all.joblib
```

These write intermediate CSVs (`master_clustered.csv`, `master_stats.csv`, `per_100_combined.csv`, …) to the repo root. **Those are gitignored regenerable outputs, not missing files** — if the repo looks like it is missing `master_clustered.csv`, run the clustering scripts. The copies under `backend/` (`models_all.joblib`, `interactions.csv`, `data/players.csv`) are tracked intentionally because they ship inside the container.

### 3. Build the retrieval index

```bash
python embed_chunks.py            # chunks + fastembed -> backend/chroma_db (69 MB)
python backend/export_index.py    # chroma_db -> backend/data/index/*.npy + *_meta.csv.gz
```

Commit the contents of `backend/data/index/` — that directory is what the deployed image reads. `backend/chroma_db` is **not** tracked and never ships: it is excluded from the build context via `.dockerignore` and gitignored, so a fresh clone must run `embed_chunks.py` before `export_index.py` can do anything.

Re-run both steps whenever the clustering or stat pipelines change, or the chunks go stale relative to the models.

### 4. Backend (FastAPI)

```bash
cd backend
uvicorn app:app --reload
```

| Endpoint | Purpose |
| --- | --- |
| `GET /` | liveness banner |
| `GET /health` | health check |
| `GET /teams` | team → available seasons |
| `POST /recommendations` | `{ user_id, era, k }` → ranked players with explanations |
| `GET /cluster-summary/player` · `/team` | cluster counts per era |
| `POST /ask` | `{ query, era?, n_results? }` → grounded answer |
| `/static/...` | headshots and cluster visuals |

Environment variables:

- `ALLOWED_ORIGINS` — comma-separated CORS origins, defaults to `http://localhost:3000`.
- `FASTEMBED_CACHE_DIR` — where the embedding weights live, defaults to `/opt/fastembed` (set by the Dockerfile).

### 5. Frontend (Next.js)

```bash
cd nba-recs-frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_BASE` in `.env.local` if the backend is not on `http://127.0.0.1:8000`. Note it is read at *build* time in the Docker image (`ARG NEXT_PUBLIC_API_BASE`), so a containerized frontend must be rebuilt to point at a different backend.

### 6. Both, via Docker

```bash
docker compose up --build
```

Compose builds the same Lambda image used in production and overrides its entrypoint (`entrypoint: []`) so `uvicorn` serves plain HTTP on `:8000` instead of `awslambdaric`. AWS credentials are passed through from the environment for the Bedrock call.

### 7. Deploy to Lambda

Build, push to ECR, and point the `nba-recsys` function at the new image. The function runs as a container package on x86_64 with a 60 s timeout and 2048 MB — the memory matters more than it looks, since Lambda scales CPU with it and both the LightFM scoring and the ONNX embedding pass are CPU-bound.


## Key features

- **Explainable recommendations.** Each suggestion carries its top user features (team traits), top item features (player traits), user/item biases, and matching cluster visuals.
- **Season filtering.** Recommendations are restricted to the selected season's player pool.
- **Era-relative clustering.** Archetypes are computed within era, so a player is labeled against his own era's peers rather than against 1999 and 2024 simultaneously.
- **Grounded Q&A.** `/ask` retrieves the nearest player and team chunks (optionally era-filtered) and prompts `amazon.nova-lite-v1:0` under strict rules: use only the provided data, invent no numbers, and refuse when the data is insufficient. Refusal is treated as a correct answer, not a failure.
- **Stat-similarity comps.** "Who plays like X" questions get a third retrieval signal injected — see below.


## Design notes

### Why stat similarity is separate from the recommender

The LightFM embeddings encode **roster co-occurrence** (team-season ↔ player-season interactions), so their nearest neighbors are teammates, not stylistic comps. That is the right signal for "which players fit this team" and the wrong one for "who plays like Nash." Playing style lives in the per-100 rate stats, so `backend/similarity.py` computes it independently — pure pandas/numpy over the stat CSVs, no `models_all.joblib` load.

Stats are z-scored **within era** before comparison, so every player is expressed as distance from his era's peers rather than in raw units. A 40% three-point shooter in 2003 and one in 2023 land at different z-scores, which is what makes cross-era comparison meaningful.

`rag.py` fires this path only when the query matches a similarity cue ("plays like", "similar to", "in the mold of", …), then formats the results as a labeled `STAT-SIMILARITY COMPS` block the LLM can cite. When even the closest match is statistically distant, the block says so and instructs the model to hedge.

### Vector search

`backend/vector_index.py` replaces Chroma at query time, for one hard reason and one soft one.

**Hard:** `chromadb.PersistentClient` opens SQLite read-write, which fails outright on a Lambda container image (`attempt to write a readonly database` — everything outside `/tmp` is read-only). Copying 69 MB into `/tmp` on every cold start would work but buys nothing.

**Soft:** the corpus is 8,640 vectors × 384 dims. A brute-force scan beats an HNSW lookup once index load time is counted — a filtered scan measures ~3 ms. Dropping `chromadb` also removes a ~0.56 s import and its whole dependency tree from the image.

Search reproduces Chroma's default metric (squared L2) and returns byte-identical top-k to the collections it replaced. It skips the `||q||²` term (constant across candidates, so it cannot change the ranking), precomputes `||e||²` to avoid materializing a 12 MB difference array per call, and uses `argpartition` to get O(N) selection with only the k-head sorted.

### Cold-start behavior

Everything expensive loads lazily: the vector index, the ONNX embedder, the similarity model, and the LightFM models. A cold start that only serves `/health` or `/recommendations` never pays to build an embedding session it does not use.

The embedding weights (`BAAI/bge-small-en-v1.5`) are **baked into the image** at build time. Without that, `fastembed` caches into `tempfile.gettempdir()` — `/tmp`, which is wiped between cold starts — so every cold start would re-pull ~65 MB from HuggingFace, adding ~3 s and failing outright in a VPC with no NAT gateway. `rag.py` reads the baked cache with `local_files_only=True`, falling back to the normal download path in dev when no cache is present.

### Image layout

Two build stages, so `gcc`/`libc6-dev` — needed only to compile lightfm — never reach the final image. A smaller image means less to lazy-fetch from ECR on a cold start.

The runtime stage still installs `libgomp1`: lightfm's extension is compiled against OpenMP in the builder, so without it `import lightfm` dies with `libgomp.so.1: cannot open shared object file`.


## Project structure

```
backend/
  app.py              FastAPI routes + CORS + static mount
  lambda_handler.py   Mangum wrapper (Lambda entrypoint)
  inference.py        recommendation + explanation logic
  rag.py              retrieval, prompt construction, Bedrock call
  similarity.py       era-z-scored stat comps
  vector_index.py     numpy brute-force search (runtime)
  export_index.py     chroma_db -> data/index/ (build-time only)
  helper.py           model (de)serialization
  data/index/         exported embeddings + metadata (shipped)
  static/cluster/     PCA plots, cluster counts
  Dockerfile          two-stage Lambda container image
  requirements.txt        runtime deps
  requirements-build.txt  + chromadb, offline only

data/                 raw Basketball-Reference CSVs
player_cluster.py     player archetype clustering & visuals
team_cluster.py       team archetype clustering & visuals
rec_sys.py            per-era LightFM training -> models_all.joblib
chunk_data.py         stat rows -> natural-language chunks
embed_chunks.py       chunks -> fastembed -> chroma_db
helper.py             era mapping, team canonicalization, model I/O
docker-compose.yml    backend + frontend for local dev
nba-recs-frontend/    Next.js 15 / React 19 / Tailwind 4 UI
```


## Notes & tips

- The recommender relies on Basketball-Reference player IDs (e.g. `wrighde01`); headshot URLs are constructed to match.
- When adding cluster labels, make sure `backend/static/cluster/` contains matching `player_<slug>.png` / `team_<slug>.png`, or the frontend cards render empty.
- `player_cluster.py`, `team_cluster.py`, and `rec_sys.py` each take a couple of minutes — worth running under screen/tmux.
- `backend/compare.py` is a one-off benchmark used to pick the Bedrock model wired into `rag.py`. It is gitignored and not part of the deployed app.
- If `vector_index` raises `FileNotFoundError`, the index was never exported — run `python backend/export_index.py`. If it raises a row-count mismatch, `chroma_db` and `data/index/` have drifted; re-export.
