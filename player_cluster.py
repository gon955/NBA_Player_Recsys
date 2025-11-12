import pandas as pd
import numpy as np
from helper import season_totals
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from helper import assign_era


def map_cluster_label(row):
    return cluster_labels_map.get(row["era"], {}).get(int(row["cluster"]), f"c{row['cluster']}")

per100 = pd.read_csv("data/Per 100 Poss.csv")

adv = pd.read_csv("data/Advanced.csv")

per100 = per100[(per100["season"] >= 1999) & (per100["mp"] > 0)]
adv  = adv[(adv["season"] >= 1999) & (adv["mp"] > 0)]

per100_comb = season_totals(per100)
adv_comb    = season_totals(adv)

per100_comb.rename(
    columns={
        "player_per100": "player",
        "lg_per100": "lg",
        "pos_per100": "pos",
        "age_per100": "age",
        "g_per100": "g",
        "gs_per100": "gs",
    },
    inplace=True,
)

per100_comb.to_csv("per_100_combined.csv")

per100_clean = pd.read_csv("per_100_combined.csv")
adv_clean = pd.read_csv("adv_stats_combined.csv")


master = pd.merge(
    per100_clean,
    adv_clean,
    on=["season","player_id"],
    how="inner",  # only keep players that exist in both
    suffixes=("_per100", "_adv")
)


#master.rename(columns={"player_per36":"player","lg_per36":"lg","pos_per36":"pos", "age_per36":"age","g_per36":"g","gs_per36":"gs"}, inplace=True)


master.rename(columns=
              {"age_per100": "age",
               "player_per100": "player",
               "lg_per100" : "lg",
               "g_per100": "g",
               "gs_per100": "gs",
               "mp_per100": "mp",
               "pos_per100": "pos"
             }
              ,inplace=True)

master = master[master["g"] >= 20]
master["era"] = master["season"].apply(assign_era)

master.to_csv("master_stats.csv", index=False)

features = [
    # per-100 volume (mirrors team features)
    "pts_per_100_poss", "fga_per_100_poss", "x3pa_per_100_poss", "fta_per_100_poss",

    # playmaking & handling
    "ast_per_100_poss", "tov_per_100_poss",

    # rebounding split
    "orb_per_100_poss", "drb_per_100_poss",

    # events / fouls
    "stl_per_100_poss", "blk_per_100_poss", "pf_per_100_poss",

    # efficiency
    "fg_percent", "x3p_percent", "ft_percent", "e_fg_percent", "ts_percent",

    # usage/share style
    "usg_percent", "orb_percent", "drb_percent", "ast_percent", "tov_percent",

    # impact-style advanced
    "per", "ws_48", "obpm", "dbpm", "bpm", "vorp",
]


id_cols = ["season","player_id","player","lg","age","pos"]
IDs = master[id_cols]

cluster_labels_map = {
    "1999-2007": {
        0: "Playmakers",
        1: "High-Scoring Forwards",
        2: "Floor Spacers",
        3: "MVP",
        4: "Defensive Centres",
        5: "High Volume Scorers",
        6: "Bench Big Man",
        7: "Stretch 4 / Shooting Wing",
    },
    "2008-2015": {
        0: "Big Man Defensive Stoppers",
        1: "Bench Guards",
        2: "MVP",
        3: "Floor Spacers",
        4: "Bench Centres",
        5: "Offensive-Minded Guards",
        6: "Bench Power Forwards",
        7: "High Scoring Big Man",
    },
    "2016-present": {
        0: "Defense First Guards",
        1: "Inefficient Scorers",
        2: "Offense First Guards/Wings",
        3: "Scoring Big Men",
        4: "MVP",
        5: "Defense First Big Men",
        6: "Bench Big Men",
        7: "Defense Minded Bench Player",
    }
}

K = 8
era_models = {}
master = master.reset_index(drop=True)
cluster_labels_all = np.full(len(master), fill_value=-1, dtype=int)

for era, df_era in master.groupby("era", sort=False):
    idx = df_era.index
    X_era = df_era[features].copy()

    preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    Xz = preprocessor.fit_transform(X_era)

    km = KMeans(n_clusters=K, random_state=35,)
    labels_era = km.fit_predict(Xz)

    cluster_labels_all[idx] = labels_era
    era_models[era] = (preprocessor, km)
    
    

master_clustered = master.copy()
master_clustered["cluster"] = cluster_labels_all
master_clustered["cluster_label"] = master_clustered.apply(map_cluster_label, axis=1)

# HAND DEFINED LABELS FOR INTERPRETATION

# cluster_labels = {
#     0: "Rim Protecting Big Man",
#     1: "Floor General",
#     2: "Spark Plug Big",
#     3: "High Volume Off Ball Shooter",
#     4: "High Usage Shot Creator",
#     5: "MVP",
#     6: "Inefficient Volume Shooter",
#     7: "Star Scoring Big Man",
#     8: "Inside Scoring Big Man",
#     9: "Defensive Specialist Wing",
# }

#master_clustered = master.copy()
#master_clustered['cluster'] = labels

fitted_preprocessor = preprocessor
fitted_clusterer = km
feature_names_used = features


# print(f"Clustering done with K={K}. Cluster counts:")
#print(master_clustered['cluster'].value_counts().sort_index())

counts = (
    master_clustered
    .groupby(["era", "cluster"])
    .size()
    .reset_index(name="count")
    .sort_values(["era", "cluster"])
)

#print(counts)

# master_clustered['cluster_label'] = master_clustered['cluster'].map(cluster_labels)

master_clustered.to_csv("master_clustered.csv", index=False)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

cluster_profiles = (
    master_clustered
    .groupby(["era","cluster"])[features]
    .mean()
    .round(2)
)
print(cluster_profiles.T)


for (era, cid), reps in master_clustered.groupby(["era","cluster"]):
    reps = reps.sort_values("per", ascending=False).head(15)
    label = cluster_labels_map.get(era, {}).get(int(cid), f"c{cid}")
    print(f"\nEra {era} — Cluster {cid} ({label}) : Representative Players")
    cols = [
    "player","season",
    "pts_per_100_poss", "fga_per_100_poss", "x3pa_per_100_poss", "fta_per_100_poss",
    "ast_per_100_poss", "tov_per_100_poss",
    "orb_per_100_poss", "drb_per_100_poss",
    "stl_per_100_poss", "blk_per_100_poss", "pf_per_100_poss",
    "fg_percent", "x3p_percent", "ft_percent", "e_fg_percent", "ts_percent",
    "usg_percent", "orb_percent", "drb_percent", "ast_percent", "tov_percent",
    "per", "ws_48", "obpm", "dbpm", "bpm", "vorp",
    ]
    print(reps[cols])
    
for era, df_era in master_clustered.groupby("era", sort=False):
    prep, km = era_models[era]
    Xz = prep.transform(df_era[features])

    pca = PCA(n_components=2, random_state=0)
    Xp = pca.fit_transform(Xz)

    df_plot = pd.DataFrame({
        "PC1": Xp[:,0],
        "PC2": Xp[:,1],
        "cluster": df_era["cluster"].values,
        "cluster_label": df_era["cluster_label"].values
    })

    plt.figure(figsize=(7,5))
    sns.scatterplot(
        data=df_plot, x="PC1", y="PC2",
        hue="cluster_label", palette="tab10", alpha=0.7
    )
    plt.title(f"PCA projection — {era}")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# plt.figure(figsize=(8,5))
# sns.countplot(
#     x= "cluster_label",data=master_clustered,
#     palette="viridis"
# )
# plt.title("Cluster Sizes (Players per Archetype)")
# plt.xlabel("Player Archetype")
# plt.ylabel("Count")
# plt.xticks(rotation=45, ha='right')
# plt.show()

# centers_z = fitted_clusterer.cluster_centers_
# centroids = pd.DataFrame(centers_z, columns=features)
# centroids.index = [f"{i} - {cluster_labels[i]}" for i in range(len(centroids))]

# plt.figure(figsize=(12,6))
# sns.heatmap(centroids, cmap="coolwarm", center=0, annot=False)
# plt.title("Cluster Centroids (z-score space)")
# plt.show()


# # Project to 2D with PCA
# pca = PCA(n_components=2, random_state=0)
# X_pca = pca.fit_transform(X_processed)

# df_plot = pd.DataFrame({
#     "PC1": X_pca[:,0],
#     "PC2": X_pca[:,1],
#     "cluster": master_clustered["cluster_label"]
# })

# plt.figure(figsize=(8,6))
# sns.scatterplot(
#     data=df_plot, x="PC1", y="PC2",
#     hue="cluster", palette="tab10", alpha=0.7
# )
# plt.title("Player Archetypes (PCA projection)")
# plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.show()


# tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
# X_tsne = tsne.fit_transform(X_processed)

# df_tsne = pd.DataFrame({
#     "TSNE1": X_tsne[:,0],
#     "TSNE2": X_tsne[:,1],
#     "cluster": master_clustered["cluster"].map(cluster_labels)
# })

# plt.figure(figsize=(8,12))
# sns.scatterplot(
#     data=df_tsne, x="TSNE1", y="TSNE2",
#     hue="cluster", palette="tab10", alpha=0.7
# )
# plt.title("Player Archetypes (t-SNE projection)")
# plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.show()


# cluster_profiles = master_clustered.groupby("cluster_label")[feature_names_used].mean()
# cluster_profiles = cluster_profiles.round(2)
# print(cluster_profiles.T)

# centers_z = fitted_clusterer.cluster_centers_
# centroids = pd.DataFrame(centers_z, columns=feature_names_used)
# print(centroids.T.round(2))

# for cid in sorted(master_clustered["cluster"].unique()):
#     reps = master_clustered[master_clustered["cluster"] == cid]
#     reps = reps.sort_values("per", ascending=False).head(15)
#     label = cluster_labels[cid]
#     print(f"\nCluster {cid} - {label}: Representative Players")
#     print(reps[["player","season","pos","pts_per_36_min","ast_per_36_min","trb_per_36_min","usg_percent","ts_percent"]])
