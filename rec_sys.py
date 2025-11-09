import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k
import time
from helper import save_models

def era_of(season):
    if 1999 <= season <= 2007: return "1999-2007"
    if 2008 <= season <= 2015: return "2008-2015"
    if 2016 <= season:         return "2016-present"
    return None

TEAM_CANON = {
    # simple, unambiguous
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "BRK": "Brooklyn Nets", "NJN": "New Jersey Nets",
    "NYK": "New York Knicks", "PHI": "Philadelphia 76ers", "TOR": "Toronto Raptors", "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers", "DET": "Detroit Pistons", "IND": "Indiana Pacers", "MIL": "Milwaukee Bucks",
    "MIA": "Miami Heat", "ORL": "Orlando Magic", "WAS": "Washington Wizards", "WSH": "Washington Wizards",
    "DAL": "Dallas Mavericks", "HOU": "Houston Rockets", "SAS": "San Antonio Spurs", "LAL": "Los Angeles Lakers",
    "LAC": "Los Angeles Clippers", "PHX": "Phoenix Suns", "PHO": "Phoenix Suns", "SAC": "Sacramento Kings",
    "GSW": "Golden State Warriors", "POR": "Portland Trail Blazers", "UTA": "Utah Jazz", "DEN": "Denver Nuggets",
    "MIN": "Minnesota Timberwolves", "OKC": "Oklahoma City Thunder", "MEM": "Memphis Grizzlies",
    "NOP": "New Orleans Pelicans", "NOH": "New Orleans Hornets", "NOK": "New Orleans/Oklahoma City Hornets","SEA": "Seattle SuperSonics",
    "VAN": "Vancouver Grizzlies", "WSB": "Washington Bullets","CHH": "Charlotte Hornets",
    # multi-team markers handled separately
    "2TM": None, "3TM": None, "4TM": None, "5TM": None
}
def canonical_team(abbr: str, season: int):
    if pd.isna(abbr):
        return np.nan
    a = str(abbr).strip()
    # Season-dependent edge cases
    if a == "CHA":
        # 2005–2014 Bobcats, 2014–present Hornets
        return "Charlotte Bobcats" if 2005 <= season <= 2014 else "Charlotte Hornets"
    if a in {"NOK", "NOH"}:
        if season in (2006, 2007):
            return "New Orleans/Oklahoma City Hornets"
        elif 2002 <= season <= 2013:
            return "New Orleans Hornets"
        # Pelicans begin 2013-14 (season=2014 in your schema)
        # 'NOP' handled via TEAM_CANON as "New Orleans Pelicans"
        return "New Orleans Hornets"
    return TEAM_CANON.get(a, a)  # fall back to itself if unknown

def norm_str(x):
    return np.nan if pd.isna(x) else str(x).strip()


players = pd.read_csv("master_clustered.csv")
teams   = pd.read_csv("master_team_clustered.csv")
adv_raw = pd.read_csv("data/Advanced.csv")

adv_raw = adv_raw[adv_raw["g"] > 20]

players["team_full"] = [
    canonical_team(abbr, int(season)) for abbr, season in zip(players["team_per100"], players["season"])
]
players = players[~players["team_full"].isin([None, "2TM", "3TM", "4TM", "5TM"])]

teams["team"] = teams["team"].map(norm_str)
teams["team_full"] = [
    canonical_team(abbr if abbr in TEAM_CANON else abbr, int(season))
    for abbr, season in zip(teams["team"], teams["season"])
]
teams["user_id"] = teams["team_full"] + "_" + teams["season"].astype(str)

adv_raw["team"]   = adv_raw["team"].map(norm_str)
adv_raw["season"] = adv_raw["season"].astype(int)

stints = (
    adv_raw.loc[(adv_raw["mp"] > 0) & (adv_raw["season"] >= 1999),
                ["team","season","player_id","mp"]]
           .groupby(["team","season","player_id"], as_index=False)["mp"].sum()
)
stints = stints[~stints["team"].isin(["2TM","3TM","4TM","5TM"])]
stints["team_full"] = [
    canonical_team(abbr, int(season)) for abbr, season in zip(stints["team"], stints["season"])
]
stints = stints[stints["team_full"].notna()]  # defensively drop any leftovers

stints["user_id"] = stints["team_full"] + "_" + stints["season"].astype(str)
stints["item_id"] = stints["player_id"].astype(str) + "_" + stints["season"].astype(str)



stints["season"] = stints["season"].astype(int)
stints["era"] = stints["season"].map(era_of)


players["era"] = players["season"].astype(int).map(era_of)
teams["era"]   = teams["season"].astype(int).map(era_of)
stints["era"]  = stints["season"].astype(int).map(era_of)

players = players.drop(columns=["Unnamed: 0"], errors="ignore")


players["season"] = players["season"].astype("Int32")
teams["season"]   = teams["season"].astype("Int32")
stints["season"]  = stints["season"].astype("Int32")

for df, cols in [
    (players, ["player_id","pos","lg","lg_adv","player_adv","team_adv","pos_adv","era","cluster_label","team_full"]),
    (teams,   ["team","lg","era","cluster_label","team_full","user_id"]),
    (stints,  ["team","team_full","user_id","item_id","era"]),
]:
    for c in cols:
        if c in df:
            df[c] = df[c].astype("category")

if "gs" in players:
    players["gs"] = players["gs"].astype("Int32")


players["mp"] = players["mp"].astype("float64")

if "item_id" not in players:
    players["item_id"] = players["player_id"].astype(str) + "_" + players["season"].astype(str)
    players["item_id"] = players["item_id"].astype("category")


interactions = (
    stints[["user_id","item_id","mp","era"]]
      .assign(weight=lambda d: np.log1p(d["mp"]))
      [["user_id","item_id","weight","era"]]
      .dropna(subset=["user_id","item_id"])
      .drop_duplicates(subset=["user_id","item_id"])
)
interactions.to_csv("interactions.csv",index = False)


item_feats = (
    players[["item_id","era","cluster_label","pos","age"]]
      .assign(
        age_bin=lambda d: pd.cut(d["age"], bins=[0,22,26,30,40], labels=["U22","23-26","27-30","31+"]),
        era_feat=lambda d: "era=" + d["era"].astype(str),
        pcl_feat=lambda d: "pcluster=" + d["cluster_label"].astype(str),
        pos_feat=lambda d: "pos=" + d["pos"].astype(str),
        age_feat=lambda d: "age=" + d["age_bin"].astype(str),
      )
      .dropna(subset=["item_id","era"])
      [["item_id","era_feat","pcl_feat","pos_feat","age_feat"]]
)

players["item_id"] = players["player_id"].astype(str) + "_" + players["season"].astype(str)
players["display_name"] = (
    players["player"].astype(str)
    + " (" + players["season"].astype(str) + ")"
)

display_map = dict(
    zip(players["item_id"].astype(str), players["display_name"].astype(str))
)

user_feats = (
    teams[["user_id","era","cluster_label","lg","pace","o_rtg","d_rtg"]]
      .assign(
        era_feat=lambda d: "era=" + d["era"].astype(str),
        tcl_feat=lambda d: "tcluster=" + d["cluster_label"].astype(str),
        pace_bin=lambda d: pd.qcut(d["pace"], 4, labels=["slow","med-","med+","fast"]),
        o_bin=lambda d: pd.qcut(d["o_rtg"], 4, labels=["o_low","o_mid-","o_mid+","o_high"]),
        d_bin=lambda d: pd.qcut(d["d_rtg"], 4, labels=["d_best","d_good","d_ok","d_poor"]),
      )
      .assign(
        pace_feat=lambda d: "pace=" + d["pace_bin"].astype(str),
        o_feat=lambda d: "ortg=" + d["o_bin"].astype(str),
        d_feat=lambda d: "drtg=" + d["d_bin"].astype(str),
      )
      [["user_id","era_feat","tcl_feat","pace_feat","o_feat","d_feat"]]
)


# ------------- 3) Train per era -------------
def build_lightfm_for_era(era,epochs, no_components, loss):
    I = interactions[interactions["era"] == era][["user_id","item_id","weight"]]
    U = user_feats[user_feats["era_feat"] == f"era={era}"]
    V = item_feats[item_feats["era_feat"] == f"era={era}"]

    # Gather vocabularies
    users = I["user_id"].unique()
    items = I["item_id"].unique()
    
    U = U[U["user_id"].isin(users)]
    V = V[V["item_id"].isin(items)]
    
    
    print(f"[{era}] counts: I={len(I)} users={len(users)} items={len(items)} Ufeat={len(U)} Vfeat={len(V)}", flush=True)
    if len(I) == 0 or len(users) == 0 or len(items) == 0:
        print(f"[{era}] SKIP (no data)", flush=True)
        return None  # or return a sentinel
    user_feature_tokens = (
        U[["tcl_feat","pace_feat","o_feat","d_feat","era_feat"]].stack().unique()
    )
    item_feature_tokens = (
        V[["pcl_feat","pos_feat","age_feat","era_feat"]].stack().unique()
    )

    # Dataset
    users = users.astype(str)
    items = items.astype(str)
    
    print("SAMPLE item_ids:", pd.Series(items).drop_duplicates().astype(str).head(10).tolist())
    ds = Dataset()
    ds.fit(users=users, items=items,
           user_features=user_feature_tokens,
           item_features=item_feature_tokens)
    print(f"[{era}] ds.fit done", flush=True)

    ufeat = ds.build_user_features(
        (row.user_id, [row.tcl_feat, row.pace_feat, row.o_feat, row.d_feat, row.era_feat])
        for row in U.itertuples(index=False)
    )
    ifeat = ds.build_item_features(
        (row.item_id, [row.pcl_feat, row.pos_feat, row.age_feat, row.era_feat])
        for row in V.itertuples(index=False)
    )
    print(f"[{era}] built feature matrices: U={ufeat.shape} V={ifeat.shape}", flush=True)

    inter_mat, weight_mat = ds.build_interactions(
        (u, i, w) for u, i, w in I.itertuples(index=False)
    )
    print(f"[{era}] built feature matrices: U={ufeat.shape} V={ifeat.shape}", flush=True)

    # Model – tune per era
    model = LightFM(
        no_components=no_components,  
        loss=loss,        
        item_alpha=1e-6,    
        user_alpha=1e-6,
        random_state=1243,
    )
    print(f"[{era}] starting fit (epochs={epochs})…", flush=True)
    model.fit(
        inter_mat,
        sample_weight=weight_mat,
        user_features=ufeat,
        item_features=ifeat,
        epochs=epochs,          
        num_threads=4,
    )
    
    
    auc  = auc_score(model, inter_mat, user_features=ufeat, item_features=ifeat, num_threads=1).mean()
    p10  = precision_at_k(model, inter_mat, k=10, user_features=ufeat, item_features=ifeat, num_threads=1).mean()
    print(f"[{era}] train AUC={auc:.3f}  P@10={p10:.3f}", flush=True)
    
    pmap = (
    players[players["era"] == era]
    .assign(item_id=lambda d: d["player_id"].astype(str) + "_" + d["season"].astype(str))
    .set_index("item_id")["player"]
    .dropna()
    .to_dict()
)

    smap = (
        stints[stints["era"] == era]
        .assign(item_id=lambda d: d["player_id"].astype(str) + "_" + d["season"].astype(str))
        .merge(players[["player_id", "player"]], on="player_id", how="left")
        .assign(player=lambda d: d["player"].fillna("Unknown Player"))
        .set_index("item_id")["player"]
        .to_dict()
    )

    display_map = {**smap, **pmap}

    
    
    return model, ds, ufeat, ifeat, items, display_map
    
from types import SimpleNamespace

models = {}
for era in ["1999-2007","2008-2015","2016-present"]:
    model, ds, ufeat, ifeat, items, display_map = result = build_lightfm_for_era(era,30,64,"warp")
    models[era] = SimpleNamespace(
        model=model,ds=ds,ufeat=ufeat,ifeat=ifeat,items=items,display_map=display_map
    )
    
save_models(models,display_map, path="models_all.joblib")
    

