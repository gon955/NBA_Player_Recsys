import numpy as np
import pandas as pd

players = pd.read_csv("master_clustered.csv")
teams = pd.read_csv("master_team_clustered.csv")

players["item_id"] = players["player_id"].astype(str) + "_" + players["season"].astype(str)
teams["user_id"]   = teams["team_id"].astype(str)    + "_" + teams["season"].astype(str)

players["player_key"] = players["player_id"].astype(str)
teams["team_key"]     = teams["team_id"].astype(str)


# Interactions

interactions = (
    players.loc[players["mp"] > 0 , ["team_per100","season","item_id","mp"]]
    .assign(
        user_id = lambda d: d["team_per100"].astype(str) + "_" + d["season"].astype(str),
        weight = lambda d: np.log1p(d["mp"])
    )[["user_id","item_id","weight"]]
)