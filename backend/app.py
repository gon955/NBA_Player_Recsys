import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from fastapi import FastAPI 
from helper import load_models
from inference import recommend_for_user, get_roster_for_team, models, interactions
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="NBA Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

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
    

@app.post("/recommendations")
def get_recommendations(req: RecommendRequest):
    # Ensure the era exists
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
            disp=disp
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
    return {
        "user": req.user_id,
        "era": req.era,
        "recommendations": [
            {"player": label, "score": float(score)}
            for label, score in recs
        ]
    }


@app.get("/")



def root():
    return {"message": "NBA Player Recommender API running"}