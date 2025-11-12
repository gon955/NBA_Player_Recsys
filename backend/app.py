import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from fastapi import FastAPI, Request, HTTPException
from helper import load_models
from inference import recommend_for_user, get_roster_for_team, models, interactions
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
app = FastAPI(title="NBA Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/teams")
def get_team_seasons():
    """Return a mapping of team -> available seasons based on the interactions data."""
    df = interactions.copy()
    df["team"] = df["user_id"].str.extract(r"^(.*)_")
    df["season"] = df["user_id"].str.extract(r"_(\d+)$").astype(int)

    team_map = (
        df.groupby("team")["season"]
        .unique()
        .apply(lambda x: sorted(x.tolist()))
        .to_dict()
    )

    return team_map
class RecommendRequest(BaseModel):
    user_id:str
    era:str
    k: int = 10
    

def serialize_feature_list(entries):
    return [
        {"feature": name, "weight": weight}
        for name, weight in entries
    ]

def serialize_explanation(expl: dict | None, base_url: str):
    if not expl:
        return None
    if "error" in expl:
        return {"error": expl["error"]}
    top_user = serialize_feature_list(expl.get("top_user_features", []))
    top_item = serialize_feature_list(expl.get("top_item_features", []))
    return {
        "score_total": expl.get("score_total"),
        "user_bias": expl.get("user_bias"),
        "item_bias": expl.get("item_bias"),
        "top_user_features": top_user,
        "top_item_features": top_item,
        "team_cluster_image": cluster_image_for(top_user, "team", base_url),
        "player_cluster_image": cluster_image_for(top_item, "player", base_url),
    }

def cluster_image_for(feature_list, kind: str, base_url: str):
    if not feature_list:
        return None
    prefix = "tcluster=" if kind == "team" else "pcluster="
    for entry in feature_list:
        feat = entry.get("feature", "")
        if feat.startswith(prefix):
            label = feat.split("=", 1)[1]
            slug = label.lower().replace(" ", "_")
            return f"{base_url}/static/cluster/{kind}_{slug}.png"
    return None

@app.get("/cluster-summary/{kind}")
def cluster_summary(kind: str):
    if kind not in {"player", "team"}:
        raise HTTPException(status_code=400, detail="kind must be 'player' or 'team'")
    records = read_cluster_counts(kind)
    return {"kind": kind, "clusters": records}

def player_photo_url(raw_item_id: str | None):
    if not raw_item_id:
        return None
    player_id = raw_item_id.split("_", 1)[0]
    if not player_id:
        return None
    return f"https://www.basketball-reference.com/req/202106291/images/players/{player_id}.jpg"

@app.post("/recommendations")
def get_recommendations(req: RecommendRequest, request: Request):
    if req.era not in models:
        return {"error": f"Era '{req.era}' not found in trained models."}
    

    pack = models[req.era]
    disp = getattr(pack, "display_map", None)
    
    print(f"[DEBUG] Model for era '{req.era}' loaded successfully.")
    print(f"[DEBUG] Has display_map: {disp is not None}")
    print(f"[DEBUG] # of items in display_map: {len(disp) if disp else 0}")
    print(f"[DEBUG] # of items in pack.items: {len(pack.items)}")
    print(f"[DEBUG] # of user features: {pack.ufeat.shape}")
    print(f"[DEBUG] # of item features: {pack.ifeat.shape}")
    
    
    exclude_items = get_roster_for_team(req.user_id, req.era, interactions)
    exclude_items = [str(x) for x in exclude_items]  # ensure string format
    print(f"[INFO] Excluding {len(exclude_items)} players from roster.")
    try:
        recs = recommend_for_user(
            req.era,
            req.user_id,
            k=req.k,
            exclude_items=exclude_items,
            disp=disp,
            explain=True,
        )
        print(f"[INFO] Got {len(recs)} recommendations.")
        if len(recs) > 0:
            print(f"[DEBUG] Sample recommendations: {recs[:3]}")
    except KeyError as e:
        print(f"[ERROR] Recommendation failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"[ERROR] Unexpected error during recommendation: {e}")
        return {"error": str(e)}
    base_url = str(request.base_url).rstrip("/")
    return {
        "user": req.user_id,
        "era": req.era,
        "recommendations": [
            {
                "player": rec.get("player"),
                "score": float(rec.get("score", 0.0)),
                "photo_url": player_photo_url(rec.get("raw_item_id")),
                "explanation": serialize_explanation(rec.get("explanation"), base_url),
            }
            for rec in recs
        ]
    }


@app.get("/")



def root():
    return {"message": "NBA Player Recommender API running"}
def read_cluster_counts(kind: str):
    filename = "player_cluster_counts.csv" if kind == "player" else "team_cluster_counts.csv"
    path = os.path.join(static_dir, "cluster", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{kind.capitalize()} cluster counts not found")
    import pandas as pd
    df = pd.read_csv(path)
    return df.to_dict(orient="records")
