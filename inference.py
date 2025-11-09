import numpy as np
from helper import load_models
from types import SimpleNamespace
import pandas as pd

models: dict[str, SimpleNamespace] = load_models("models_all.joblib")
interactions = pd.read_csv("interactions.csv")

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

def recommend_for_user(era: str, user_id: str, k: int = 10, exclude_items=None,disp = None):
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

        if raw_item_id == label:
            print(f"[WARN] No display name for {raw_item_id}")
        else:
            print(f"[OK] {raw_item_id} → {label}")

    return out
    
if __name__ == "__main__":
    era  = "2016-present"
    user = "Atlanta Hawks_2019"   # must exist in this era
    pack = models[era]
    disp = pack.display_map
    exclude_items = get_roster_for_team(user,era,interactions)
    print(f"[INFO] Excluding {len(exclude_items)} players already on {user} roster.")
    recs = recommend_for_user(era,user, k = 10,exclude_items=exclude_items)
