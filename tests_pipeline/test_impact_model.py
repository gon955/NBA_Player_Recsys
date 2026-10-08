import itertools

import numpy as np
import pandas as pd
import pytest

import impact_model as im


def _synthetic_panel(true_o, true_d, season=2020, poss=2000, seed=0):
    """Every 5-man combination of the players, outcomes from known effects."""
    rng = np.random.default_rng(seed)
    rows = []
    for combo in itertools.combinations(true_o, 5):
        o = sum(true_o[p] for p in combo)
        d = sum(true_d[p] for p in combo)
        rows.append({
            **{f"p{k + 1}_id": p for k, p in enumerate(combo)},
            "season": season,
            "OffPoss": poss, "DefPoss": poss,
            # League rating 110; tiny noise so the fit is near-exact.
            "Points": poss * (110 + o + rng.normal(0, 0.01)) / 100,
            "OpponentPoints": poss * (110 - d + rng.normal(0, 0.01)) / 100,
        })
    return pd.DataFrame(rows)


def test_fit_season_recovers_known_effects():
    players = [f"p{i}" for i in range(8)]
    true_o = dict(zip(players, [4, 2, 0, 0, 0, -1, -2, -3]))
    true_d = dict(zip(players, [-1, 0, 3, 0, 1, 0, -2, -1]))
    panel = _synthetic_panel(true_o, true_d)

    est = im.fit_season(panel, lam=1, prior_o=pd.Series(dtype=float),
                        prior_d=pd.Series(dtype=float)).set_index("player_id")

    # Effects are identified only up to a shared constant per side (every row
    # has five players), and the targets are centred, so compare demeaned.
    for col, truth in (("o_impact", true_o), ("d_impact", true_d)):
        got = est[col] - est[col].mean()
        want = pd.Series(truth) - np.mean(list(truth.values()))
        assert got.reindex(want.index).to_numpy() == pytest.approx(want.to_numpy(), abs=0.05)
    assert (est["net_impact"] == est["o_impact"] + est["d_impact"]).all()


def test_ridge_returns_the_prior_when_shrinkage_dominates():
    panel = _synthetic_panel(dict.fromkeys("abcdef", 0.0), dict.fromkeys("abcdef", 0.0))
    prior = pd.Series({"a": 5.0, "b": -3.0})

    est = im.fit_season(panel, lam=1e12, prior_o=prior, prior_d=prior).set_index("player_id")

    assert est.loc["a", "o_impact"] == pytest.approx(5.0)
    assert est.loc["b", "d_impact"] == pytest.approx(-3.0)
    assert est.loc["c", "o_impact"] == pytest.approx(0.0, abs=1e-6)  # no prior -> 0


def test_targets_skip_zero_possession_sides():
    df = pd.DataFrame({"Points": [0, 11], "OpponentPoints": [5, 0],
                       "OffPoss": [0, 10], "DefPoss": [5, 0]})
    y_off, w_off, y_def, w_def = im._targets(df)

    assert w_off[0] == 0 and w_def[1] == 0       # weight 0, so no influence
    assert np.isfinite(y_off).all() and np.isfinite(y_def).all()


def test_score_next_season_perfect_prediction_has_r2_near_one():
    players = [f"p{i}" for i in range(7)]
    true_o = dict(zip(players, [3, 1, 0, 0, -1, -1, -2]))
    true_d = dict(zip(players, [0, 2, -1, 0, 0, 1, -2]))
    panel = pd.concat([_synthetic_panel(true_o, true_d, season=s, seed=s) for s in (2020, 2021)])
    est = pd.DataFrame({"player_id": players, "season": 2020,
                        "o_impact": [true_o[p] for p in players],
                        "d_impact": [true_d[p] for p in players]})
    # Truth is centred on the league; shift the estimates the same way.
    est["o_impact"] -= np.mean(list(true_o.values()))
    est["d_impact"] -= np.mean(list(true_d.values()))

    o_r2, d_r2 = im.score_next_season(panel, est)
    assert o_r2 > 0.99 and d_r2 > 0.99
