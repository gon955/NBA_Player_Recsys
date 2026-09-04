"""Content-addressed cluster labels.

WHY THIS EXISTS
---------------
Both clustering scripts used to map a KMeans *index* to an archetype name:

    cluster_labels_map = {"2016-present": {0: "Defense First Guards", ...}}

The index is not a stable identity. Re-running with a different ``random_state``
preserved the meaning of only 1-4 clusters out of 8, and because the scripts
never pinned ``n_init`` they also inherited scikit-learn's default -- which was
10 before 1.4 and 1 ("auto") from 1.4 onward. Under the other default the
"MVP" name landed on the era's *worst* cluster (BPM z-score -1.34) with no
error raised.

So labels are matched to cluster CONTENT instead. Each archetype is stored as a
reference profile of raw stat means; a freshly fitted cluster gets the name of
the reference profile it is closest to, under a globally-optimal one-to-one
assignment. Seeds, n_init and sklearn versions can then change freely -- the
name follows the basketball, not the array index.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_reference.json")


def centroid_table(df, labels, features):
    """Raw-unit mean of every feature, per cluster. Index = cluster id."""
    return df.groupby(pd.Series(labels, index=df.index))[features].mean()


def _scaled(frame, scale):
    return frame.divide(scale.reindex(frame.columns).replace(0, 1.0), axis=1)


def match_labels(centroids, reference, features):
    """Assign each fitted cluster the name of its nearest reference profile.

    Distances are computed in units of the reference population's per-feature
    spread, so a feature measured in possessions cannot dominate one measured
    as a percentage. ``linear_sum_assignment`` gives a one-to-one matching, so
    two clusters can never claim the same name.
    """
    ref = pd.DataFrame(reference["profiles"]).T[features].astype(float)
    scale = pd.Series(reference["scale"])[features].astype(float)
    a = _scaled(centroids[features].astype(float), scale).to_numpy()
    b = _scaled(ref, scale).to_numpy()
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    names = list(ref.index)
    out = {int(centroids.index[r]): names[c] for r, c in zip(rows, cols)}
    quality = {int(centroids.index[r]): float(cost[r, c]) for r, c in zip(rows, cols)}
    return out, quality


def load_reference(kind, era):
    if not os.path.exists(REFERENCE_PATH):
        return None
    with open(REFERENCE_PATH) as fh:
        ref = json.load(fh)
    return ref.get(kind, {}).get(era)


def save_reference(kind, era, centroids, names, features, population):
    """Persist the current fit as the reference profiles for `kind`/`era`."""
    ref = {}
    if os.path.exists(REFERENCE_PATH):
        with open(REFERENCE_PATH) as fh:
            ref = json.load(fh)
    ref.setdefault(kind, {})[era] = {
        "features": list(features),
        "scale": {f: float(population[f].std(ddof=0)) or 1.0 for f in features},
        "profiles": {
            names[int(cid)]: {f: float(centroids.loc[cid, f]) for f in features}
            for cid in centroids.index
        },
    }
    with open(REFERENCE_PATH, "w") as fh:
        json.dump(ref, fh, indent=2, sort_keys=True)


def label_clusters(df, labels, features, kind, era, fallback_names=None):
    """Return a name per row. Falls back to ``c{index}`` when no reference exists."""
    centroids = centroid_table(df, labels, features)
    reference = load_reference(kind, era)
    if reference is None:
        mapping = {int(c): f"c{int(c)}" for c in centroids.index}
        if fallback_names:
            mapping = {int(c): fallback_names.get(int(c), f"c{int(c)}") for c in centroids.index}
        return [mapping[int(lab)] for lab in labels], mapping, None
    missing = [f for f in features if f not in reference["features"]]
    if missing:
        raise ValueError(
            f"cluster_reference.json for {kind}/{era} was built on a different "
            f"feature set; missing {missing}. Rebuild it with "
            f"`python {os.path.basename(__file__)} --rebuild`."
        )
    mapping, quality = match_labels(centroids, reference, features)
    return [mapping[int(lab)] for lab in labels], mapping, quality
