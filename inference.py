import numpy as np
import scipy.sparse as sp
from helper import load_models
from types import SimpleNamespace
import pandas as pd
import os
import scipy.sparse as sp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models_all.joblib")
INTERACTIONS_PATH = os.path.join(BASE_DIR, "interactions.csv")

models: dict[str, SimpleNamespace] = load_models(MODEL_PATH)
interactions = pd.read_csv(INTERACTIONS_PATH)

def explain_score(pack, user_id: str, item_id: str, top: int = 3):
    """
    Compute a simple explanation of why a player (item) is recommended
    for a team (user). Returns top contributing features and biases.
    """

    model = pack.model
    user_map, user_feat_map, item_map, item_feat_map = pack.ds.mapping()

    # --- Defensive checks ---
    if user_id not in user_map or item_id not in item_map:
        raise KeyError(f"User or item not found in model mapping: {user_id}, {item_id}")

    # --- Get internal LightFM indices ---
    uidx = user_map[user_id]
    iid = item_map[item_id]
    u_feat_idx = user_feat_map[user_id]
    i_feat_idx = item_feat_map[item_id]

    # --- Ensure matrices are CSR so we can safely index them ---
    ufeat_csr = pack.ufeat.tocsr() if sp.issparse(pack.ufeat) else sp.csr_matrix(pack.ufeat)
    ifeat_csr = pack.ifeat.tocsr() if sp.issparse(pack.ifeat) else sp.csr_matrix(pack.ifeat)

    # --- Extract feature indices for this user and item (exclude identity feature) ---
    ufeat_ids = [fid for fid in ufeat_csr[uidx].indices if fid != u_feat_idx]
    ifeat_ids = [fid for fid in ifeat_csr[iid].indices if fid != i_feat_idx]

    # --- Embedding components ---
    u_feat_matrix = model.user_embeddings
    i_feat_matrix = model.item_embeddings

    u_vec = u_feat_matrix[u_feat_idx] + (u_feat_matrix[ufeat_ids].sum(axis=0) if ufeat_ids else 0)
    i_vec = i_feat_matrix[i_feat_idx] + (i_feat_matrix[ifeat_ids].sum(axis=0) if ifeat_ids else 0)

    # --- Biases and score ---
    u_bias = float(model.user_biases[u_feat_idx])
    i_bias = float(model.item_biases[i_feat_idx])
    dot_total = float(np.dot(u_vec, i_vec))
    total_score = dot_total + u_bias + i_bias

    # --- Invert feature maps for readability ---
    inv_user_feat = {v: k for k, v in user_feat_map.items()}
    inv_item_feat = {v: k for k, v in item_feat_map.items()}

    # --- Compute contributions ---
    user_contribs = [
        (inv_user_feat.get(fid, f"feat_{fid}"), float(np.dot(i_vec, u_feat_matrix[fid])))
        for fid in ufeat_ids
    ]
    item_contribs = [
        (inv_item_feat.get(fid, f"feat_{fid}"), float(np.dot(u_vec, i_feat_matrix[fid])))
        for fid in ifeat_ids
    ]

    user_contribs.sort(key=lambda x: -x[1])
    item_contribs.sort(key=lambda x: -x[1])

    return {
        "score_total": total_score,
        "user_bias": u_bias,
        "item_bias": i_bias,
        "top_user_features": user_contribs[:top],
        "top_item_features": item_contribs[:top],
    }
def get_roster_for_team(user_id: str, era: str, interactions_df) -> list[str]:
    roster = (
        interactions_df.loc[
            (interactions_df["user_id"] == user_id) & (interactions_df["era"] == era),
            "item_id"
        ]
        .dropna()
        .unique()
        .tolist()
    )
    return roster

def rev_item_map(pack:SimpleNamespace):
    _, _, item_map,_ = pack.ds.mapping()
    return {v: k for k, v in item_map.items()}

def get_maps(pack):
    user_map, item_map, *_ = pack.ds.mapping()
    return user_map, item_map

def recommend_for_user(era: str, user_id: str, k: int = 10, exclude_items=None,disp = None, explain: bool = False):
    pack = models[era]
    exclude_items = set(exclude_items or [])

    user_map, user_feat_map, item_map, item_feat_map = pack.ds.mapping()

    candidates = [i for i in item_map.keys() if i not in exclude_items]

    uidx = user_map[user_id]
    iidx = np.array([item_map[i] for i in candidates if i in item_map], dtype=np.int64)

    scores = pack.model.predict(
        user_ids=uidx,
        item_ids=iidx,
        user_features=pack.ufeat,
        item_features=pack.ifeat,
        num_threads=4,
    )

    k = min(k, len(scores))
    sel = np.argpartition(-scores, kth=k-1)[:k]
    sel = sel[np.argsort(-scores[sel])]

    rev_item = rev_item_map(pack)
    disp = disp or getattr(pack, "display_map", None)

    out = []
    for j in sel:
        raw_item_id = rev_item[iidx[j]]
        label = disp.get(raw_item_id, raw_item_id) if disp else raw_item_id
        
        rec = {
            "player": label,
            "score": float(scores[j])
        }

        if explain:
            try:
                rec["explanation"] = explain_score(pack, user_id, raw_item_id)
            except Exception as e:
                rec["explanation"] = {"error": str(e)}
                
        
        out.append(rec)

    return out
    
if __name__ == "__main__":
    era = "2016-present"
    user = "Atlanta Hawks_2019"
    pack = models[era]
    disp = pack.display_map
    exclude_items = get_roster_for_team(user, era, interactions)

    recs = recommend_for_user(era, user, k=5, exclude_items=exclude_items, disp=disp, explain=True)
    for r in recs:
        print(f"{r['player']}: {r['score']:.4f}")
        if "explanation" in r:
            print("  Top user features:", r["explanation"]["top_user_features"])
            print("  Top item features:", r["explanation"]["top_item_features"])
    
