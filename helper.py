import pandas as pd
import joblib
import numpy as np
from types import SimpleNamespace

TOTAL_MARKERS = {"2TM","3TM","4TM","5TM","6TM","TOT"}

def season_totals(df):
    keys = ['player_id', 'season']
    df = df.copy()
    
    has_tot = df.groupby(keys)['team'].transform(lambda s: s.isin(TOTAL_MARKERS).any())
    
    keep_totals = df[has_tot & df['team'].isin(TOTAL_MARKERS)]
    keep_others = df[~has_tot & ~df['team'].isin(TOTAL_MARKERS)]
    
    out = pd.concat([keep_totals, keep_others], ignore_index=True)
    out = out.sort_values(keys).drop_duplicates(keys, keep="first").reset_index(drop=True)
    return out

def assign_era(season: int) -> str:
    if 2016 <= season:
        return "2016-present"          # Pace & Space
    if 2008 <= season <= 2015:
        return "2008-2015"             # Early Spacing Adaptation
    if 1999 <= season <= 2007:
        return "1999-2007"             # Post Hand-Check / Zone
    return "1990-1998"                  # Physical Defense


def save_models(models,path="models.joblib"):
    serial = {}
    for era, pack in models.items():
        serial[era] = {
            "model":pack.model,
            "ds":pack.ds,
            "ufeat":pack.ufeat,
            "ifeat":pack.ifeat,
            "items":np.array(pack.items),
            "display_map": getattr(pack,"display_map",None),
        }
    joblib.dump(serial,path)
    
def load_models(path="models.joblib"):
    serial = joblib.load(path)
    models = {}
    for era, d in serial.items():
        models[era] = SimpleNamespace(
            model=d["model"],
            ds=d["ds"],
            ufeat=d["ufeat"],
            ifeat=d["ifeat"],
            items=d["items"],
            display_map=d.get("display_map", {}),
        )
    return models