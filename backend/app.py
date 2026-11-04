import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import ask as rag_ask
from recommender import recommend, team_seasons

app = FastAPI(title="NBA Recommender API")

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Cloudflare Pages gives every preview deployment its own origin
# (<hash>.<project>.pages.dev), so an exact-match list can only ever cover the
# production URL. Widening it to "*" is not an option either: browsers reject
# a wildcard origin combined with allow_credentials. So previews need a pattern.
# Starlette checks allow_origins first, then fullmatch()es this regex, and
# leaving it unset keeps the previous exact-match-only behaviour.
_origin_regex = os.environ.get("ALLOWED_ORIGIN_REGEX") or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/teams")
def get_team_seasons():
    """Return a mapping of team -> seasons that have recommendations."""
    return team_seasons()


class RecommendRequest(BaseModel):
    user_id: str
    # Unused since the impact recommender replaced LightFM's per-era models;
    # still accepted (and echoed) so older clients keep working.
    era: Optional[str] = None
    k: int = 10


def player_photo_url(raw_item_id: Optional[str]):
    if not raw_item_id:
        return None
    player_id = raw_item_id.split("_", 1)[0]
    if not player_id:
        return None
    return f"https://www.basketball-reference.com/req/202106291/images/players/{player_id}.jpg"

@app.post("/recommendations")
def get_recommendations(req: RecommendRequest):
    try:
        recs = recommend(req.user_id, k=req.k)
    except KeyError as e:
        # Reported as a 200 with an {error} body, which the frontend expects.
        return {"error": str(e.args[0]) if e.args else str(e)}
    return {
        "user": req.user_id,
        "era": req.era,
        "recommendations": [
            {
                "player": rec["player"],
                "score": rec["score"],
                "photo_url": player_photo_url(rec["player_id"]),
                "explanation": rec["explanation"],
            }
            for rec in recs
        ],
    }


@app.get("/")
def root():
    return {"message": "NBA Player Recommender API running"}


class AskRequest(BaseModel):
    query: str
    era: Optional[str] = None
    n_results: int = 10

@app.post("/ask")
def ask(request: AskRequest):
    try:
        answer = rag_ask(request.query, request.era, request.n_results)
        return {"query": request.query,"era": request.era, "answer": answer}
    except Exception as e:
        print(f"[ERROR] RAG ask failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
