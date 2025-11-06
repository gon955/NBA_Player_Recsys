import pandas as pd
from helper import season_totals
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import numpy as np
from helper import assign_era


def map_cluster_label(row):
    return cluster_labels_map.get(row["era"], {}).get(int(row["cluster"]), f"c{row['cluster']}")


team_per_100_stats = pd.read_csv("data\Team Stats Per 100 Poss.csv")
team_adv_stats = pd.read_csv("data\Team Summaries.csv")

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

    "pts_per_100_poss", "fga_per_100_poss", "x3pa_per_100_poss", "fta_per_100_poss",
   
    "fg_percent", "x3p_percent", "ts_percent", "e_fg_percent", "ft_percent",
    
    "ast_per_100_poss", "tov_per_100_poss", "tov_percent",
   
    "orb_percent", "drb_percent",  # or orb_per_100_poss, drb_per_100_poss

    "stl_per_100_poss", "blk_per_100_poss",

    "opp_e_fg_percent", "opp_tov_percent", "opp_ft_fga",

    "pace", "x3p_ar", "ft_fga",
    
    "o_rtg", "d_rtg", "n_rtg", "srs"
]

cluster_labels_map = {
    "1999-2007": {
        0: "Balanced",
        1: "Defensive Grind",
        2: "Inefficient Offense",
        3: "High-Pace Offense",
        4: "Rebuilder"
    },
    "2008-2015": {
        0: "Balanced",
        1: "Modern Offense",
        2: "Iso Scorers",
        3: "Fast & Flawed D",
        4: "Contender"
    },
    "2016-present": {
        0: "Playoff Team",
        1: "Pace & Space",
        2: "Rebuilder",
        3: "Fast & Flawed D",
        4: "Defensive Powerhouse"
    }
}

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

for era, df_era in master_team.groupby("era", sort=False):
    idx = df_era.index
    X_era = df_era[team_features].copy()

    # simple impute + standardize *within this era*
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    X_era_processed = preprocessor.fit_transform(X_era)

    km = KMeans(n_clusters=K, random_state=89)
    labels_era = km.fit_predict(X_era_processed)

    # write labels back into the full array
    cluster_labels_all[idx] = labels_era
    era_models[era] = (preprocessor, km)

master_team_clustered = master_team.copy()
master_team_clustered["cluster"] = cluster_labels_all
master_team_clustered["cluster_label"] = master_team_clustered.apply(map_cluster_label, axis=1)

counts = (
    master_team_clustered
    .groupby(["era", "cluster_label"])
    .size()
    .reset_index(name="count")
    .sort_values(["era", "cluster_label"])
)

print(counts)

#master_clustered['cluster_label'] = master_clustered['cluster'].map(cluster_labels)

master_team_clustered.to_csv("master_team_clustered.csv",index = False)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


from sklearn.decomposition import PCA

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

    plt.figure(figsize=(7,5))
    sns.scatterplot(
        data=df_plot, x="PC1", y="PC2",
        hue="cluster_label", palette="tab10", alpha=0.7
    )
    plt.title(f"PCA projection — {era}")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    
cluster_profiles = (
    master_team_clustered
    .groupby(["era","cluster"])[team_features]
    .mean()
    .round(2)
)
print(cluster_profiles.T)

for (era, cid), reps in master_team_clustered.groupby(["era","cluster"]):
    reps = reps.sort_values("n_rtg", ascending=False).head(15)
    label = cluster_labels_map.get(era, {}).get(int(cid), f"c{cid}")
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