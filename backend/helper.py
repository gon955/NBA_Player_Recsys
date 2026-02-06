import pandas as pd
import joblib
import numpy as np
from types import SimpleNamespace

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
