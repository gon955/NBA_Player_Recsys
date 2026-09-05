# nba-recs-frontend

Next.js 15 (App Router) + React 19 + Tailwind 4 UI for the NBA Player Recommender.
It is a thin client over the FastAPI backend in `../backend` — all data fetching
happens in the browser, there are no server components or server actions.

## Layout

```
src/
  app/
    layout.tsx      root layout, metadata, Geist fonts
    page.tsx        state + composition only; no markup beyond the shell
    globals.css     Tailwind import and the dark palette tokens
  components/
    ViewTabs            Recommend / Clusters / Ask tab switcher
    RecommendForm       team -> season -> k form
    RecommendResults    ranked player cards
    ExplanationCard     per-recommendation LightFM feature breakdown
    PlayerAvatar        headshot with initials fallback
    ClusterImage        cluster PNG served off the backend's /static mount
    ClusterOverview     per-era cluster population bars + PCA links
    AskPanel            RAG question box with an era filter
  lib/
    api.ts          API_BASE, typed fetch wrappers, response types
    era.ts          the three model eras and the season -> era mapping
    features.ts     raw LightFM feature strings -> plain English
```

## Running locally

```bash
npm install
npm run dev
```

Open http://localhost:3000. The backend must be running — see the root README.

## Configuration

`NEXT_PUBLIC_API_BASE` selects the backend origin and defaults to
`http://127.0.0.1:8000`. Set it in `.env.local` for local dev.

Note it is read at **build** time. The Docker image takes it as
`ARG NEXT_PUBLIC_API_BASE`, so a containerized frontend must be rebuilt — not
just restarted with a new env var — to point at a different backend.

The backend's CORS allowlist (`ALLOWED_ORIGINS`) must include whatever origin
this app is served from; it defaults to `http://localhost:3000`.

`NEXT_PUBLIC_SITE_URL` is the canonical origin this app is served from, and
defaults to `https://nba-recs.pages.dev`. It only feeds the link-preview
metadata in `layout.tsx` — nothing at runtime reads it — but it has to be
absolute and correct, because a static export has no request to infer the host
from and crawlers reject relative `og:image` paths. Set it when deploying to a
custom domain or when you want a preview deployment to unfurl as itself.

## Link previews

`layout.tsx` emits Open Graph and Twitter card tags, and `public/og.png` is the
1200x630 preview image they point at. LinkedIn, Slack and iMessage refuse to
build a preview at all when `og:title`, `og:description` or `og:image` is
missing, even when the crawl itself succeeded — a bare `<title>` and
`<meta name="description">` are not enough for them.

Two things to know when changing them:

- Every crawler caches aggressively. After a deploy, re-scrape the URL through
  [LinkedIn's Post Inspector](https://www.linkedin.com/post-inspector/) before
  concluding the tags are wrong; LinkedIn will otherwise keep serving whatever
  it saw first for days.
- `og:image` must resolve over plain HTTPS with no redirect and no auth.
  Cloudflare Pages serves `public/og.png` straight from the CDN, which
  satisfies that; pointing it at the Lambda's `/static/` mount would not, since
  every fetch would be a cold-startable invocation.

## Backend contract

| Call | Endpoint |
|---|---|
| `getTeams` | `GET /teams` → `{ "<team name>": [seasons] }` |
| `getRecommendations` | `POST /recommendations` `{era, user_id, k}` |
| `getClusterSummaries` | `GET /cluster-summary/{player,team}` |
| `askQuestion` | `POST /ask` `{query, era, n_results}` |

`user_id` is `"<team name>_<season>"` (e.g. `Boston Celtics_2018`), and `era` is
derived from the season by `getEraFromYear`.

`POST /recommendations` reports domain errors (unknown era, team not in the
model) as HTTP **200** with an `{error}` body, so `api.ts` checks the payload
rather than the status code.
