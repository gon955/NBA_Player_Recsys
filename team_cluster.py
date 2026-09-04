import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cluster_naming import label_clusters
from helper import assign_era

DATA_DIR = "data"
team_per_100_stats = pd.read_csv(os.path.join(DATA_DIR, "Team Stats Per 100 Poss.csv"))
team_adv_stats = pd.read_csv(os.path.join(DATA_DIR, "Team Summaries.csv"))

team_adv_stats = team_adv_stats[team_adv_stats['season'] >= 1999]
team_per_100_stats = team_per_100_stats[team_per_100_stats['season'] >= 1999]

team_adv_stats.drop(columns = ['arena','attend','attend_g','abbreviation','w','l','pw','pl','playoffs','age'], inplace=True)
team_per_100_stats.drop(columns = ['lg','abbreviation','playoffs'], inplace=True)

master_team = pd.merge(
    team_per_100_stats,
    team_adv_stats,
    on=["season","team"],
    how="inner", 
    suffixes=("_per100", "_adv")
)


master_team["era"] = master_team["season"].apply(assign_era)

#master_team.to_csv("master_team_stats.csv", index=False)


team_features = [
    # NOTE: exact duplicates removed — each is recoverable from the kept set
    # with R^2 >= 0.998, so including it double-counted that axis:
    #   pts_per_100_poss  -> o_rtg          (1.0000, same quantity)
    #   n_rtg             -> o_rtg - d_rtg  (1.0000, exact by definition)
    #   x3pa_per_100_poss -> x3p_ar         (0.9996)
    #   e_fg_percent      -> ts_percent     (0.9997)
    #   tov_per_100_poss  -> tov_percent    (0.9988)
    #   fta_per_100_poss  -> ft_fga         (0.9983)

    "fga_per_100_poss",

    "fg_percent", "x3p_percent", "ts_percent", "ft_percent",

    "ast_per_100_poss", "tov_percent",

    "orb_percent", "drb_percent",

    "stl_per_100_poss", "blk_per_100_poss",

    "opp_e_fg_percent", "opp_tov_percent", "opp_ft_fga",

    "pace", "x3p_ar", "ft_fga",

    "o_rtg", "d_rtg", "srs"
]

# Archetype names now live in cluster_reference.json and are matched to cluster
# CONTENT by cluster_naming.label_clusters().

X = master_team[team_features]

X = X.fillna(X.mean())

id_cols = master_team[['season','team']].reset_index(drop=True)

preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

X_processed = preprocessor.fit_transform(X)

K = 5

era_models = {}

cluster_labels_all = np.full(len(master_team), fill_value=-1, dtype=int)
cluster_names_all = np.empty(len(master_team), dtype=object)

for era, df_era in master_team.groupby("era", sort=False):
    idx = df_era.index
    X_era = df_era[team_features].copy()

    # simple impute + standardize *within this era*
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    X_era_processed = preprocessor.fit_transform(X_era)

    # n_init pinned — sklearn's default changed from 10 to "auto" (=1) in 1.4.
    km = KMeans(n_clusters=K, random_state=89, n_init=10)
    labels_era = km.fit_predict(X_era_processed)

    names_era, mapping, quality = label_clusters(df_era, labels_era, team_features, "team", era)

    # write labels back into the full array
    cluster_labels_all[idx] = labels_era
    cluster_names_all[idx] = names_era
    era_models[era] = (preprocessor, km)

    print(f"[{era}] cluster index -> archetype: {mapping}")

master_team_clustered = master_team.copy()
master_team_clustered["cluster"] = cluster_labels_all
master_team_clustered["cluster_label"] = cluster_names_all

CLUSTER_CARD_DIR = os.path.join("backend", "static", "cluster")
os.makedirs(CLUSTER_CARD_DIR, exist_ok=True)

counts = (
    master_team_clustered
    .groupby(["era", "cluster_label"])
    .size()
    .reset_index(name="count")
    .sort_values(["era", "cluster_label"])
)

print(counts)
counts_path = os.path.join(CLUSTER_CARD_DIR, "team_cluster_counts.csv")
counts.to_csv(counts_path, index=False)

#master_clustered['cluster_label'] = master_clustered['cluster'].map(cluster_labels)

master_team_clustered.to_csv("master_team_clustered.csv",index = False)


TEAM_CARD_FEATURES = [
    "o_rtg",
    "d_rtg",
    "n_rtg",
    "pace",
    "x3p_ar",
    "ts_percent",
]

def slugify(label: str) -> str:
    return label.lower().replace(" ", "_").replace("/", "-")

def save_team_cluster_card(label: str, stats: pd.Series, reps: pd.DataFrame):
    slug = slugify(label)
    path = os.path.join(CLUSTER_CARD_DIR, f"team_{slug}.png")
    if os.path.exists(path):
        return
    data = stats.reindex(TEAM_CARD_FEATURES).dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(data.index, data.values, color="#fbbf24")
    ax.set_title(label)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", color="#e2e8f0", alpha=0.4)
    top_names = ", ".join((reps["team"] + " " + reps["season"].astype(str)).head(3).tolist())
    fig.text(0.02, 0.01, f"Examples: {top_names}", fontsize=8, color="#475569")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

for era, df_era in master_team_clustered.groupby("era", sort=False):
    prep, km = era_models[era]
    Xz = prep.transform(df_era[team_features])

    pca = PCA(n_components=2, random_state=0)
    Xp = pca.fit_transform(Xz)

    df_plot = pd.DataFrame({
        "PC1": Xp[:,0],
        "PC2": Xp[:,1],
        "cluster": df_era["cluster"].values,
        "cluster_label": df_era["cluster_label"].values
    })

    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(
        data=df_plot, x="PC1", y="PC2",
        hue="cluster_label", palette="tab10", alpha=0.7, ax=ax
    )
    ax.set_title(f"PCA projection — {era}")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(CLUSTER_CARD_DIR, f"team_pca_{slugify(era)}.png"), dpi=150)
    plt.close(fig)
    
cluster_profiles = (
    master_team_clustered
    .groupby(["era","cluster"])[team_features]
    .mean()
    .round(2)
)
print(cluster_profiles.T)

for (era, cid), reps in master_team_clustered.groupby(["era","cluster"]):
    reps = reps.sort_values("n_rtg", ascending=False).head(15)
    label = reps["cluster_label"].iloc[0]
    print(f"\nEra {era} — Cluster {cid} ({label}) : Representative Teams")
    cols = [
        "team","season",
        "pts_per_100_poss","o_rtg","d_rtg","n_rtg",
        "ast_per_100_poss","tov_percent",
        "x3p_ar","ft_fga","pace",
        "orb_percent","drb_percent",
        "opp_e_fg_percent","opp_tov_percent","opp_ft_fga"
    ]
    print(reps[cols])
    stats = cluster_profiles.loc[(era, cid)]
    save_team_cluster_card(label, stats, reps)
