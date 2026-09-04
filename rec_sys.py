from types import SimpleNamespace

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k

from helper import TEAM_CANON, canonical_team, era_of, save_models


def display_map_for_era(era: str) -> dict[str, str]:
    era_stints = stints.loc[stints["era"] == era, ["player_id","season"]].copy()
    era_stints["item_id"] = era_stints["player_id"].astype(str) + "_" + era_stints["season"].astype(str)
    era_stints["player_name"] = era_stints["player_id"].map(name_map)
    era_stints["label"] = era_stints["player_name"].fillna("Unknown Player") \
                          + " (" + era_stints["season"].astype(str) + ")"
    era_stints = era_stints.drop_duplicates("item_id")
    return dict(zip(era_stints["item_id"].astype(str), era_stints["label"].astype(str)))



def norm_str(x):
    return np.nan if pd.isna(x) else str(x).strip()


players = pd.read_csv("master_clustered.csv")

teams = pd.read_csv("master_team_clustered.csv")

adv_raw = pd.read_csv("data/Advanced.csv")

# Positional role, measured instead of labelled.
#
# `pos` is one string per player-season ("SG", or "PG-SG" when the source can't
# decide), so it became one binary token and threw away everything about degree:
# a pure point guard and a combo guard splitting his minutes 60/40 got the same
# feature, and every hyphenated label spawned its own sparse category that shares
# no weight with either of its halves. `data/Player Play By Play.csv` carries
# pg_percent…c_percent — the share of minutes actually played at each spot — for
# every season this model uses (1999+; 5 rows league-wide are missing them).
# The shares are renormalised to sum to 1, so each item still contributes the
# same total positional mass as the single token did; only its distribution
# across the five spots is new.
POS_SHARE_COLS = ["pg_percent", "sg_percent", "sf_percent", "pf_percent", "c_percent"]
POS_TOKENS = ["pos=pg", "pos=sg", "pos=sf", "pos=pf", "pos=c"]
POS_UNKNOWN = "pos=unknown"

pbp_raw = pd.read_csv("data/Player Play By Play.csv")
# The *TM rows are season totals that duplicate the per-team rows; drop them and
# blend the splits by minutes instead, which reproduces the same season profile
# while working identically for players who never moved.
pbp_raw = pbp_raw[~pbp_raw["team"].isin(["2TM", "3TM", "4TM", "5TM"])]
pbp_raw = pbp_raw.dropna(subset=POS_SHARE_COLS, how="all")
pbp_raw["mp"] = pbp_raw["mp"].fillna(0.0)


def _blend_shares(g):
    w = g["mp"].to_numpy(dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(g), dtype=float)
    s = np.average(g[POS_SHARE_COLS].fillna(0.0).to_numpy(dtype=float), axis=0, weights=w)
    total = s.sum()
    return pd.Series(s / total if total > 0 else s, index=POS_TOKENS)


_pos_shares = (
    pbp_raw.groupby(["player_id", "season"])[POS_SHARE_COLS + ["mp"]]
           .apply(_blend_shares)
           .reset_index()
)
_pos_shares["item_id"] = (
    _pos_shares["player_id"].astype(str) + "_" + _pos_shares["season"].astype(int).astype(str)
)
_pos_shares = _pos_shares[_pos_shares[POS_TOKENS].sum(axis=1) > 0]
pos_share_map = _pos_shares.set_index("item_id")[POS_TOKENS].to_dict("index")

_POS_STRING_TOKEN = {"PG": "pos=pg", "SG": "pos=sg", "SF": "pos=sf", "PF": "pos=pf", "C": "pos=c"}


def shares_from_pos_string(pos_feat):
    """Fallback for item-seasons with no play-by-play row: spread the label evenly."""
    raw = str(pos_feat)
    raw = raw[len("pos="):] if raw.startswith("pos=") else raw
    parts = [_POS_STRING_TOKEN[p] for p in raw.upper().split("-") if p in _POS_STRING_TOKEN]
    if not parts:
        return {POS_UNKNOWN: 1.0}
    return {t: 1.0 / len(parts) for t in parts}


name_map = (
    adv_raw.loc[:, ["player_id", "player"]]
        .dropna()
        .drop_duplicates("player_id", keep="last") 
        .set_index("player_id")["player"]
        .to_dict()
)



adv_raw = adv_raw[adv_raw["g"] > 20]
adv_raw = adv_raw[~adv_raw["team"].isin(["2TM", "3TM", "4TM", "5TM"])]

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
stints["team_full"] = [
    canonical_team(abbr, int(season)) for abbr, season in zip(stints["team"], stints["season"])
]
stints = stints[stints["team_full"].notna()] 

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

stints.to_csv("stints.csv", index=False)
teams.to_csv("teams.csv", index=False)

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
      .assign(weight=lambda d: np.sqrt(d["mp"]))
      .dropna(subset=["user_id","item_id"])
      .drop_duplicates(subset=["user_id","item_id"])
)

# LightFM applies sample_weight as a per-example multiplier on the gradient,
# on top of learning_rate=0.05. Raw sqrt(minutes) averages ~35 (min 7, max 59),
# which made the effective step size ~1.7 and drove the model to memorise the
# training rosters: train AUC 0.90 against test AUC 0.53, and held-out true
# teammates ranked *below* a random shuffle. Rescaling to mean 1 within each
# era keeps the relative minutes signal and restores the intended learning
# rate — held-out Recall@10 goes 0.0010 -> 0.0254 (random is 0.0185).
interactions["weight"] = (
    interactions.groupby("era", observed=True)["weight"].transform(lambda s: s / s.mean())
)
interactions = interactions[["user_id","item_id","weight","era"]]
interactions.to_csv("interactions.csv",index = False)


age_bins = [0, 22, 26, 30, 35, 40, 60]
age_labels = ["U22", "23-26", "27-30", "31-34", "35-39", "40+"]

item_feats = (
    players[["item_id","era","cluster_label","pos","age","player_id","season"]]
      .assign(
        age_bin=lambda d: pd.cut(d["age"], bins=age_bins, labels=age_labels, right=False, include_lowest=True),
        pcl_feat=lambda d: "pcluster=" + d["cluster_label"].astype(str),
        pos_feat=lambda d: "pos=" + d["pos"].astype(str),
        age_feat=lambda d: "age=" + d["age_bin"].astype(str),
      )
      .dropna(subset=["item_id","era"])
      [["item_id","era","player_id","season","pcl_feat","pos_feat","age_feat"]]
)

all_items = interactions[["item_id","era"]].drop_duplicates()
all_items[["player_id","season"]] = all_items["item_id"].str.split("_", n=1, expand=True)
all_items["season"] = all_items["season"].astype(int)

adv_meta = (
    adv_raw[["player_id","season","pos","age"]]
      .dropna(subset=["player_id","season"])
      .drop_duplicates(subset=["player_id","season"], keep="last")
      .assign(
        item_id=lambda d: d["player_id"].astype(str) + "_" + d["season"].astype(str),
        age_bin=lambda d: pd.cut(d["age"], bins=age_bins, labels=age_labels, right=False, include_lowest=True),
        pos_feat=lambda d: "pos=" + d["pos"].astype(str),
        age_feat=lambda d: "age=" + d["age_bin"].astype(str),
      )
      [["item_id","pos_feat","age_feat"]]
)

item_feats = all_items.merge(
    item_feats.rename(columns={"player_id":"player_id_meta","season":"season_meta"}),
    on="item_id",
    how="left",
)
if "era_y" in item_feats:
    item_feats = item_feats.drop(columns=["era_y"])
if "era_x" in item_feats:
    item_feats = item_feats.rename(columns={"era_x": "era"})
item_feats = item_feats.merge(adv_meta, on="item_id", how="left", suffixes=("","_adv"))

cluster_lookup = (
    players.sort_values("season")
           .groupby("player_id")["cluster_label"]
           .agg(lambda s: s.iloc[-1])
           .to_dict()
)

def fill_cluster(row):
    if pd.notna(row["pcl_feat"]):
        return row["pcl_feat"]
    fallback = cluster_lookup.get(row["player_id"])
    if isinstance(fallback, str):
        return "pcluster=" + fallback
    return "pcluster=unknown"

item_feats["pcl_feat"] = item_feats.apply(fill_cluster, axis=1)
item_feats["pos_feat"] = item_feats["pos_feat"].fillna(item_feats["pos_feat_adv"])
item_feats["age_feat"] = item_feats["age_feat"].fillna(item_feats["age_feat_adv"])
item_feats["pos_feat"] = item_feats["pos_feat"].fillna("pos=unknown")
item_feats["age_feat"] = item_feats["age_feat"].fillna("age=unknown")

# `pos_feat` survives only as the fallback source for items the play-by-play file
# doesn't reach; the served feature is the share dict.
item_feats["pos_shares"] = [
    pos_share_map.get(i) or shares_from_pos_string(p)
    for i, p in zip(item_feats["item_id"].astype(str), item_feats["pos_feat"])
]
_covered = sum(1 for i in item_feats["item_id"].astype(str) if i in pos_share_map)
print(f"positional shares: {_covered}/{len(item_feats)} items from play-by-play, "
      f"{len(item_feats) - _covered} from the pos string")

item_feats = item_feats[["item_id","era","pcl_feat","pos_feat","pos_shares","age_feat"]]

players["item_id"] = players["player_id"].astype(str) + "_" + players["season"].astype(str)
players["display_name"] = (
    players["player"].astype(str)
    + " (" + players["season"].astype(str) + ")"
)

# Carry the same shares into players.csv so the RAG chunks describe the measured
# positional mix rather than repeating the `pos` label.
players = players.merge(
    _pos_shares[["item_id"] + POS_TOKENS].rename(
        columns={t: t.replace("pos=", "share_") for t in POS_TOKENS}
    ),
    on="item_id",
    how="left",
)

players.to_csv("players.csv", index=False)

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
def build_lightfm_for_era(era, epochs, no_components, loss, holdout=3, eval_seed=0):
    inter = interactions[interactions["era"] == era][["user_id","item_id","weight"]]
    U = user_feats[user_feats["era_feat"] == f"era={era}"]
    V = item_feats[item_feats["era"] == era]

    users = inter["user_id"].unique()
    items = inter["item_id"].unique()
    
    U = U[U["user_id"].isin(users)]
    V = V[V["item_id"].isin(items)]
    
    
    print(f"[{era}] counts: inter={len(inter)} users={len(users)} items={len(items)} Ufeat={len(U)} Vfeat={len(V)}", flush=True)
    if len(inter) == 0 or len(users) == 0 or len(items) == 0:
        print(f"[{era}] SKIP (no data)", flush=True)
        return None  
    user_feature_tokens = (
        U[["tcl_feat","pace_feat","o_feat","d_feat","era_feat"]].stack().unique()
    )
    item_feature_tokens = (
        list(pd.unique(V[["pcl_feat","age_feat"]].stack())) + POS_TOKENS + [POS_UNKNOWN]
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
        (row.item_id, {row.pcl_feat: 1.0, row.age_feat: 1.0, **row.pos_shares})
        for row in V.itertuples(index=False)
    )
    print(f"[{era}] built feature matrices: U={ufeat.shape} V={ifeat.shape}", flush=True)

    inter_mat, weight_mat = ds.build_interactions(
        (u, i, w) for u, i, w in inter.itertuples(index=False)
    )
    print(f"[{era}] built feature matrices: U={ufeat.shape} V={ifeat.shape}", flush=True)

    def _new_model():
        return LightFM(
            no_components=no_components,
            loss=loss,
            item_alpha=1e-6,
            user_alpha=1e-6,
            random_state=1243,
        )

    # ---- honest evaluation ----------------------------------------------
    # Scoring the model on the same matrix it was fitted to reports memorisation,
    # not skill: that AUC read 0.90 for a model whose held-out AUC is 0.52. So
    # hold out `holdout` interactions per team-season, fit a throwaway model on
    # the rest, and report both numbers side by side.
    cols = ["user_id", "item_id", "weight"]
    rng = np.random.default_rng(eval_seed)
    split = inter.reset_index(drop=True).copy()
    split["_r"] = rng.random(len(split))
    split["_rank"] = split.groupby("user_id")["_r"].rank(method="first")
    sizes = split.groupby("user_id")["item_id"].transform("size")
    held = (split["_rank"] <= holdout) & (sizes >= 2 * holdout)

    train_mat, train_w = ds.build_interactions(
        (u, i, w) for u, i, w in split.loc[~held, cols].itertuples(index=False)
    )
    test_mat, _ = ds.build_interactions(
        (u, i, w) for u, i, w in split.loc[held, cols].itertuples(index=False)
    )
    print(f"[{era}] evaluating on {int(held.sum())} held-out interactions…", flush=True)
    probe = _new_model()
    probe.fit(train_mat, sample_weight=train_w, user_features=ufeat,
              item_features=ifeat, epochs=epochs, num_threads=4)
    ev = dict(user_features=ufeat, item_features=ifeat, num_threads=1)
    train_auc = auc_score(probe, train_mat, **ev).mean()
    test_auc = auc_score(probe, test_mat, train_interactions=train_mat, **ev).mean()
    test_p10 = precision_at_k(probe, test_mat, k=10, train_interactions=train_mat, **ev).mean()
    print(f"[{era}] train AUC={train_auc:.3f} | HELD-OUT AUC={test_auc:.3f} "
          f"P@10={test_p10:.4f}", flush=True)

    # ---- served model: refit on every interaction ------------------------
    print(f"[{era}] starting fit (epochs={epochs})…", flush=True)
    model = _new_model()
    model.fit(
        inter_mat,
        sample_weight=weight_mat,
        user_features=ufeat,
        item_features=ifeat,
        epochs=epochs,
        num_threads=4,
    )
    

    display_map = display_map_for_era(era)

    
    
    return model, ds, ufeat, ifeat, items, display_map
    

models = {}
for era in ["1999-2007","2008-2015","2016-present"]:
    model, ds, ufeat, ifeat, items, display_map = result = build_lightfm_for_era(era,30,128,"bpr")
    models[era] = SimpleNamespace(
        model=model,ds=ds,ufeat=ufeat,ifeat=ifeat,items=items,display_map=display_map
    )
    
save_models(models, path="models_all.joblib")
    

