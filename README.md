# NBA Recommendation System

For a team-season (say *Chicago 2025*), recommends players who would make the team better next season, among moves it could realistically make. Also answers natural-language questions about the 1999–2025 data. Four parts:

- **Offline recommender pipeline** (root `*.py`) — a lineup-based player impact model, a projection of next-season impact, a model of how likely each team is to acquire each player, and the salary screen. It writes a ranked, explained list per team-season.
- **Review loop** (`review_app.py`, `audit.py`) — a local page for a person to flag recommendations that should never be made. Flags only ever remove entries, and veto *rules* must first pass an audit against real moves.
- **FastAPI backend** (`backend/`) — serves the precomputed lists, plus a `/ask` endpoint that answers questions over retrieved stat chunks via AWS Bedrock.
- **Next.js frontend** (`nba-recs-frontend/`) — two tabs: Recommendations and Ask.

The backend ships as a Lambda container image behind a Function URL. The same code runs under `uvicorn` locally — `lambda_handler.py` wraps the FastAPI app in Mangum, so there is no branching inside the routes.

The project's history (the retired LightFM recommender, the clustering pipeline) is kept out of the repo, in a gitignored `local_archive/`. `HANDOFF.md` is the detailed record of how the current system was built and evaluated.


## Getting Started

### 1. Environment

**Python 3.9.** The backend still pins `pandas==1.2.4`, which has no wheels past 3.9; that is also why the Dockerfile uses `python:3.9-slim` rather than an AWS Lambda base image (the official python3.9 base image is past EOL). Lambda accepts any image implementing the Runtime API, which `awslambdaric` provides.

```bash
conda env create -f environment.yml && conda activate recsys   # offline pipeline (+ tests, lint)
pip install -r backend/requirements.txt                        # backend runtime only
pip install -r backend/requirements-dev.txt                    # + pytest/ruff
```

`backend/requirements.txt` is runtime-only on purpose: the Dockerfile installs it alone, so nothing in `requirements-dev.txt` reaches the Lambda image.

Node.js ≥ 18 for the Next.js app (the frontend image uses `node:20-alpine`).

AWS credentials with `bedrock:InvokeModel` in `us-west-2` are needed for `/ask`, on two models: `amazon.nova-lite-v1:0` (answers) and `amazon.titan-embed-text-v2:0` (embedding the question). In Lambda this is the execution role; locally, standard credentials from the environment.

### 2. The recommender pipeline

Raw Basketball-Reference CSVs live in `data/` (already populated), along with the salary, payroll and transaction tables the fetchers render. To regenerate everything after an upstream change:

```bash
python player_seasons.py                 # player table: name, age, position, BPM
python weakness_test.py --write-curvature
python contract_status.py
python fetch_wikipedia_fa.py --consolidate-only   # no network if the cache exists
python impact_recs.py                    # data/recs_impact*.csv (top 10s)
python export_recs.py                    # backend/data/recs.csv: what the app serves
```

`export_recs.py` applies active veto rules and the review's flags, and writes the top 50 per team-season with the context each explanation needs. The backend does no inference; restart it to pick up a new `recs.csv`.

The lineup panel (`data/lineup_panel.csv`, ~137 MB) and the pbpstats bulk files behind it are gitignored; `fetch_pbpstats.py` and `lineup_features.py` rebuild them.

### 3. Grading

```bash
python rubric.py              # tuning seasons (decisions 1999-2017)
python rubric.py --holdout    # 2018 onward; every run is logged to data/holdout_log.csv
python audit.py screen        # which real moves the salary screen rules out
python evaluate_recs.py       # the older realism / quality report, for reference
```

There is no ground truth for what a team *should* have done, so `rubric.py` grades only what real data can answer:
- **availability:** did the players we list actually move?
- **value accuracy:** did the ones who moved deliver what we projected?
- **specificity:** do teams get different lists?

Each season is predicted from earlier seasons only. Settings are tuned on the tuning seasons, and the holdout is meant to be read once.

### 4. Review

```bash
python review_app.py          # http://127.0.0.1:8765
```

A local page over the live lists. Each decision — ok, or flagged with reasons and a note — goes into `data/review/judgments.csv`, and is kept for good:
- **Repeat passes show only new entries.** The team queue counts them.
- **"Apply flags" rebuilds the lists from the page.** Flagged picks drop out and their replacements appear as new.

Flags are unaudited and narrow: each removes one (season, team, player) entry. Patterns become veto rules in `audit.RULES`, which must pass `python audit.py rules` (a check against real moves) before they are switched on. `python audit.py summarize` shows what flagged picks have in common.

### 5. Backend (FastAPI)

```bash
cd backend
uvicorn app:app --reload
```

| Endpoint | Purpose |
| --- | --- |
| `GET /` | liveness banner |
| `GET /health` | health check |
| `GET /teams` | team → available seasons |
| `POST /recommendations` | `{ user_id, k }` → ranked players with explanations (who goes out, the projected upgrade, salaries) |
| `POST /ask` | `{ query, era?, n_results? }` → grounded answer |

Environment variables:

- `ALLOWED_ORIGINS` — comma-separated exact CORS origins, defaults to `http://localhost:3000`.
- `ALLOWED_ORIGIN_REGEX` — optional pattern matched against the request origin, for origins that cannot be enumerated (Cloudflare Pages preview deployments). Unset by default, which keeps exact matching only. Starlette `fullmatch()`es it, so it must cover the whole origin.

### 6. Frontend (Next.js)

```bash
cd nba-recs-frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_BASE` in `.env.local` if the backend is not on `http://127.0.0.1:8000`. It is read at *build* time — baked into the bundle by `next build` — so any deployed frontend must be rebuilt to point at a different backend, whether it is the Docker image (`ARG NEXT_PUBLIC_API_BASE`) or Cloudflare Pages.

`NEXT_PUBLIC_SITE_URL` is the canonical origin used to build the link-preview tags, and defaults to `https://nba-recs.pages.dev`. Set it when deploying to a custom domain — see [Link previews](#link-previews).

`next.config.ts` emits a **static export** (`out/`) by default. Every page here is a client component and all data is fetched from the backend in the browser, so there is no server to run. The compose image needs a Node server, so its Dockerfile sets `NEXT_OUTPUT=standalone` to get the previous `output: "standalone"` behaviour back.

### 7. Both, via Docker

```bash
docker compose up --build
```

Compose builds the same Lambda image used in production and overrides its entrypoint (`entrypoint: []`) so `uvicorn` serves plain HTTP on `:8000` instead of `awslambdaric`. AWS credentials are passed through from the environment for the Bedrock call.

### 8. Deploy the backend to Lambda

Build, push to ECR, and point the `nba-recsys` function at the new image. The function runs as a container package on x86_64 with a 60 s timeout and 2048 MB. That memory was sized for a local ONNX embedding pass (Lambda scales CPU with memory), which is gone: what remains is a CSV lookup, a numpy scan and two Bedrock calls. A lower setting is likely enough and cheaper; measure before changing it.

The execution role needs `bedrock:InvokeModel` on `amazon.titan-embed-text-v2:0` as well as `amazon.nova-lite-v1:0`, with model access enabled for both in `us-west-2`. If the role's policy names model ARNs, add Titan before deploying, or `/ask` will fail with AccessDenied.

### 9. Deploy the frontend to Cloudflare Pages

The frontend is a static bundle, so it needs no server and no AWS resources. Co-locating it with the backend would buy nothing: the browser calls the Lambda Function URL directly, so there is no server-to-server hop that being in the same region would shorten.

Cloudflare Pages settings:

| Setting | Value |
| --- | --- |
| Root directory | `nba-recs-frontend` |
| Build command | `npm run build` |
| Output directory | `out` |
| Environment variable | `NEXT_PUBLIC_API_BASE` = the Lambda Function URL |
| Environment variable | `NEXT_PUBLIC_SITE_URL` = the deployed origin (only if not `https://nba-recs.pages.dev`) |

Pick framework preset **None**, not "Next.js" — that preset builds for the Workers runtime via `@cloudflare/next-on-pages`, which is for apps needing SSR and would fight the static export. `.nvmrc` pins Node 20 because Next 15 requires `>=18.18` and the default build image has shipped older.

Things that will bite otherwise:

- `NEXT_PUBLIC_API_BASE` is baked in at build time, so changing the backend URL means a rebuild, not just a settings change.
- Add the Pages origin to `ALLOWED_ORIGINS` on the Lambda, or the browser blocks every call on CORS. Both variables are read at import, so changing them takes effect on the next cold start — no image rebuild or ECR push.
- Preview deployments each get their own origin (`<hash>.<project>.pages.dev`), which an exact-match list cannot cover, and `allow_credentials=True` rules out a `"*"` wildcard because browsers reject that pair. Set `ALLOWED_ORIGIN_REGEX` to cover them:

  ```
  ALLOWED_ORIGIN_REGEX=https://([a-z0-9-]+\.)?nba-recs\.pages\.dev
  ```

- Do not also configure CORS on the Lambda Function URL itself. It has its own CORS block, separate from the FastAPI middleware, and having both emit `Access-Control-Allow-Origin` produces a duplicate header that browsers reject.

### Link previews

`nba-recs-frontend/src/app/layout.tsx` emits Open Graph and Twitter card tags, and `nba-recs-frontend/public/og.png` is the 1200x630 image they point at. Its source is `nba-recs-frontend/og-image.html`; the render command is in that file.

This matters because link unfurlers are stricter than they look. LinkedIn, Slack and iMessage refuse to render a preview at all when `og:title`, `og:description` or `og:image` is missing — the crawl succeeds, the page returns 200, and the preview is still rejected. A `<title>` and `<meta name="description">` alone, which is what Next's default `metadata` block produced here, are not enough.

Two constraints shape how the tags are built:

- **The URL has to be absolute and hardcoded.** Crawlers do not resolve relative `og:image` paths, and a static export has no request to infer its own host from, so the origin is baked in at build time from `NEXT_PUBLIC_SITE_URL` (via `metadataBase`) rather than discovered at runtime.
- **The image has to be a plain, unauthenticated HTTPS fetch.** Serving it from `public/` puts it on the Cloudflare CDN, which satisfies that; a Lambda-served image would make every crawl a cold-startable invocation.

After deploying, re-scrape through [LinkedIn's Post Inspector](https://www.linkedin.com/post-inspector/) — every crawler caches the first response it saw, so a fixed page keeps unfurling with the old error until the cache is refreshed.

### 10. Tests and linting

```bash
pytest                                # tests/ — the API; config lives in pyproject.toml
pytest tests_pipeline --no-cov        # the offline pipeline and review tooling
ruff check .                          # lint
```

Run the two suites separately: `tests/` puts `backend/` first on `sys.path` (the container's import graph), while the pipeline needs the root `helper.py`.

`tests/` covers the API at the route level: `tests/conftest.py` builds a `TestClient` over the real `app`, the same import graph the container gets from `WORKDIR /app; COPY backend/ .`. Because `data/recs.csv` and `data/index/` are tracked, the suite runs off a plain checkout. It needs no AWS credentials — `rag.py` constructs its Bedrock client and embedder lazily, so nothing under `tests/` reaches `/ask`.

`.github/workflows/ci.yml` runs two jobs. The `ruff` job installs just the binary and finishes in seconds. The `pytest` job builds the Dockerfile's `test` stage and runs `tests/` **inside the image**, so CI exercises exactly the runtime's dependency set. The `test` stage reuses the `deps` layer the runtime is built from. It triggers on pull requests targeting `main` (where GitHub tests the merge result) and on pushes to `main`.

To reproduce a CI run locally:

```bash
docker build --target test -t nba-recsys-test -f backend/Dockerfile .
docker run --rm nba-recsys-test pytest
```


## How a recommendation is scored

For each candidate and team, among swaps the team could legally make (salary matching with a package of up to one extra lesser contract, a value cap, and an upgrade gate):

    score = projected value if he moves
          + REALISM_WEIGHT × log P(this team acquires him)   [movability capped]
          + FIT_WEIGHT × how weak the team is at his position

- **Projected value** (`impact_recs.project_mover()`): next-season net impact from a possession-weighted ridge over 5-man lineups. It uses the previous season, the box score and age, and is *conditioned on changing teams*, because movers underperform their estimates.
- **Realism** (`acquisition.py`): gradient-boosted P(team acquires player), fit on earlier seasons' real arrivals. "How likely he is to move at all" is capped at the season's 80th percentile, so the most movable journeymen do not top every list.
- **Fit:** the one team-specific signal with observable support. Players joining a team that is weak at their position get more minutes.

Each explanation names who goes out (and who else, to match salary), the projected upgrade, and both salaries. Every constant above is tuned in `HANDOFF.md`, with the evidence in the module docstrings.


## Design notes

### Stat similarity for the Ask tab

"Who plays like X" questions get stat-profile nearest neighbours from `backend/similarity.py` — pure pandas/numpy over `backend/data/players.csv`, separate from the recommender, which asks a different question (who would make a team better).

Stats are z-scored **within era** before comparison, so every player is expressed as distance from his era's peers rather than in raw units. A 40% three-point shooter in 2003 and one in 2023 land at different z-scores, which is what makes cross-era comparison meaningful.

`rag.py` fires this path only when the query matches a similarity cue ("plays like", "similar to", "in the mold of", …), then formats the results as a labeled `STAT-SIMILARITY COMPS` block the LLM can cite. When even the closest match is statistically distant, the block says so and instructs the model to hedge.

### Vector search

`backend/vector_index.py` searches the shipped index (`backend/data/index/`) with numpy rather than Chroma, for one hard reason and one soft one.

**Hard:** `chromadb.PersistentClient` opens SQLite read-write, which fails outright on a Lambda container image (`attempt to write a readonly database` — everything outside `/tmp` is read-only).

**Soft:** the corpus is ~10,300 vectors × 512 dims. A brute-force scan beats an HNSW lookup once index load time is counted — a filtered scan measures ~3 ms.

Search reproduces Chroma's default metric (squared L2) and returns byte-identical top-k to the collections it replaced. It skips the `||q||²` term (constant across candidates, so it cannot change the ranking), precomputes `||e||²` to avoid materializing a 12 MB difference array per call, and uses `argpartition` to get O(N) selection with only the k-head sorted.

The vectors are Amazon Titan Text Embeddings v2 (512 dims, unit length, so squared L2 ranks as cosine does). `python build_ask_index.py` writes them, and records the model and dimensions in `backend/data/index/index_info.json`. `rag.py` embeds each question with that same model, and `vector_index.py` refuses an index whose dimensions disagree. The build re-embeds the chunk text stored in the index metadata. The scripts that first wrote that text read the retired LightFM pipeline and are in `local_archive/`, so the chunks themselves are frozen; refreshing them from current data would need a new chunk builder.

### Cold-start behavior

Everything expensive loads lazily: the vector index and the similarity model. Recommendations are a CSV lookup, and there is no embedding model to load at all — each `/ask` makes one Titan call to embed the question (tens of milliseconds) before the answer call. That replaced a local ONNX model (`fastembed`, `BAAI/bge-small-en-v1.5`), whose weights had to be baked into the image and whose session cost seconds on a cold start.

### Image layout

Stages are `deps` -> `prune`/`test` -> runtime. Every dependency installs from a wheel, so there is no compiler or system library in any stage. `prune` strips pip and setuptools from the virtualenv before the runtime copies it (deleting them in a later runtime layer would only hide them), and `test` adds pytest on top of the same dependency layer.

Retiring LightFM (and with it `gcc`, `libgomp1`, `joblib` and `scikit-learn`) took the runtime image from 761 MB to 551 MB. Moving the Ask tab's embeddings to Bedrock (dropping `fastembed`, `onnxruntime`, `sympy`, `tokenizers`, Pillow and the 65 MB of weights) and pruning pip took it further, to the size given in `HANDOFF.md`. That is less to lazy-fetch from ECR on a cold start.


## Project structure

```
backend/
  app.py              FastAPI routes + CORS
  lambda_handler.py   Mangum wrapper (Lambda entrypoint)
  recommender.py      serves data/recs.csv
  rag.py              retrieval, prompt construction, Bedrock call
  similarity.py       era-z-scored stat comps
  vector_index.py     numpy brute-force search
  data/recs.csv       precomputed lists (export_recs.py)
  data/index/         Ask embeddings (Titan) + metadata + index_info.json (shipped)
  data/players.csv    Ask player table
  Dockerfile          Lambda container image: deps -> builder/test -> runtime

impact_model.py       lineup ridge: player impact per season
lineup_features.py    5-man lineup panel from pbpstats
impact_recs.py        projection, salary screen, scoring -> data/recs_impact.csv
acquisition.py        P(team acquires player)
export_recs.py        -> backend/data/recs.csv (vetoes and review flags applied)
rubric.py             grading on availability, value accuracy, specificity; holdout log
audit.py              veto-rule audit, screen audit, review judgments
review_app.py         local review page (review_ui.html)
build_ask_index.py    embeds the Ask index with Titan (Bedrock)
evaluate_recs.py      older realism / quality report
player_seasons.py, contract_status.py, destination.py, weakness_test.py,
team_style.py, fit_model.py, fetch_*.py    inputs and analyses (see HANDOFF.md)
data/                 Basketball-Reference CSVs, salaries, payroll, transactions
nba-recs-frontend/    Next.js 15 / React 19 / Tailwind 4 UI
```


## Notes & tips

- Player IDs are Basketball-Reference IDs (e.g. `wrighde01`); headshot URLs are constructed to match.
- `backend/compare.py` is a one-off benchmark used to pick the Bedrock model wired into `rag.py`. It is gitignored and not part of the deployed app.
- If `vector_index` raises `FileNotFoundError`, `backend/data/index/` is missing from the checkout.
