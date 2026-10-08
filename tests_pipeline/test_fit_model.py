import itertools

import numpy as np
import pandas as pd
import pytest

import fit_model as fm


def test_design_counts_mains_and_pairs():
    vocab = fm.Vocab(["Big", "Guard", "Wing"])
    slots = np.array([["Big", "Big", "Guard", "Wing", "unknown"]])
    X = pd.Series(vocab.design(slots).toarray()[0], index=vocab.columns())

    assert X["main"].sum() == 5 and X["pair"].sum() == 10
    assert X[("main", "Big", "")] == 2
    assert X[("pair", "Big", "Big")] == 1        # C(2, 2)
    assert X[("pair", "Big", "Guard")] == 2      # 2 bigs x 1 guard
    assert X[("pair", "Guard", "Wing")] == 1
    assert X[("pair", "Guard", "Guard")] == 0


def test_design_without_pairs_has_only_mains():
    vocab = fm.Vocab(["Big", "Guard"], pairs=False)
    X = vocab.design(np.array([["Big"] * 5]))
    assert X.shape[1] == 3                     # Big, Guard, unknown
    assert X.toarray()[0].tolist() == [5, 0, 0]


def test_unrecognised_label_falls_back_to_unknown():
    vocab = fm.Vocab(["Big"])
    X = pd.Series(vocab.design(np.array([["Martian"] * 5])).toarray()[0], index=vocab.columns())
    assert X[("main", "unknown", "")] == 5


def test_shuffle_keeps_each_seasons_label_mix():
    labels = pd.DataFrame({
        "player_id": list("abcdefgh"),
        "season": [2020] * 4 + [2021] * 4,
        "cluster_label": ["X", "X", "Y", "Z", "Y", "Y", "Y", "Z"],
    })
    shuffled = fm.shuffle_labels(labels, seed=3)
    for season in (2020, 2021):
        before = labels[labels.season == season].cluster_label.value_counts()
        after = shuffled[shuffled.season == season].cluster_label.value_counts()
        assert before.sort_index().equals(after.sort_index())


def test_ridge_recovers_a_planted_pair_effect():
    """Two 'Big's together cost 3 points; nothing else matters."""
    names = ["Big", "Guard", "Wing"]
    vocab = fm.Vocab(names)
    rng = np.random.default_rng(0)
    slots = np.array([rng.choice(names, 5) for _ in range(4000)])
    n_big_pairs = np.array([sum(1 for a, b in itertools.combinations(r, 2) if a == b == "Big")
                            for r in slots])
    y = -3.0 * n_big_pairs
    w = np.full(len(y), 100.0)

    coef = pd.Series(fm.ridge(vocab.design(slots), y, w, lam=1.0), index=vocab.columns())
    X = vocab.design(slots)
    assert X @ coef.to_numpy() == pytest.approx(y, abs=1e-3)
    # Identified up to the mains/pairs overlap, but the Big-Big pair must stand out.
    pairs = coef["pair"]
    assert pairs[("Big", "Big")] == pairs.min()
