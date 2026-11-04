import numpy as np
import pandas as pd
import pytest

import acquisition as aq


def test_arrivals_are_veterans_who_joined_for_next_season():
    member = pd.DataFrame({
        "season":    [2020, 2020, 2021, 2021, 2021, 2021],
        "TeamId":    [1,    2,    1,    1,    1,    1],
        "player_id": ["a",  "b",  "a",  "b",  "r",  "x"],
        "share":     [0.5,  0.5,  0.5,  0.3,  0.2,  0.01],
    })
    debut = pd.Series({"a": 2010, "b": 2012, "r": 2021, "x": 2015})
    a = aq.arrivals(member, debut)

    # b moved 2 -> 1; a stayed; r is a rookie; x barely played.
    assert list(zip(a["season"], a["TeamId"], a["player_id"])) == [(2020, 1, "b")]


def test_fit_predict_uses_only_earlier_seasons():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(2000, 2008):
        f = pd.DataFrame(rng.normal(size=(400, len(aq.FEATURES))), columns=aq.FEATURES)
        f["acquired"] = (rng.random(400) < 0.05).astype(int)
        rows.append(f.assign(season=s, TeamId=1, player_id=[f"p{i}" for i in range(400)]))
    pairs = pd.concat(rows, ignore_index=True)
    changed = pairs.copy()
    changed.loc[changed["season"] == 2007, "acquired"] = 1 - changed.loc[changed["season"] == 2007, "acquired"]
    a, b = aq.fit_predict(pairs), aq.fit_predict(changed)

    # Relabelling 2007 changes nothing up to and including 2007's own predictions.
    upto = a["season"] <= 2007
    np.testing.assert_allclose(a.loc[upto, "p_acq"], b.loc[upto, "p_acq"])
    assert (a["p_acq"] > 0).all()


def _synthetic_pairs(seasons=range(2000, 2008), teams=4, players=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        f = pd.DataFrame(rng.normal(size=(players, len(aq.FEATURES))), columns=aq.FEATURES)
        f["player_id"] = [f"p{i}" for i in range(players)]
        for t in range(teams):
            g = f.assign(season=s, TeamId=t)
            # each team takes ~2 players; a player joins at most one team
            g["acquired"] = ((np.arange(players) % teams == t) & (rng.random(players) < 0.1)).astype(int)
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def test_two_stage_destinations_sum_to_one_per_mover():
    p = aq.fit_predict_two_stage(_synthetic_pairs())
    late = p[p["season"] == 2007]
    sums = late.groupby("player_id")["p_dest"].sum()
    np.testing.assert_allclose(sums, 1.0)
    np.testing.assert_allclose(late["p_acq"], (late["p_move"] * late["p_dest"]).clip(lower=1e-6))


def test_ranker_and_two_stage_use_only_earlier_seasons():
    pairs = _synthetic_pairs()
    changed = pairs.copy()
    last = changed["season"] == 2007
    changed.loc[last, "acquired"] = 1 - changed.loc[last, "acquired"]
    for fn in (aq.fit_predict_ranker, aq.fit_predict_two_stage):
        a = fn(pairs).sort_values(["season", "TeamId", "player_id"]).reset_index(drop=True)
        b = fn(changed).sort_values(["season", "TeamId", "player_id"]).reset_index(drop=True)
        upto = a["season"] <= 2007
        np.testing.assert_allclose(a.loc[upto, "p_acq"], b.loc[upto, "p_acq"])


def test_calibrate_moves_rescales_movability_but_not_destination():
    rng = np.random.default_rng(0)
    rows = []
    for season in range(2010, 2016):
        for p in range(40):
            bpm = rng.normal()
            moved_to = rng.integers(0, 4) if rng.random() < 0.3 else -1
            for team in range(4):
                rows.append({"season": season, "TeamId": team, "player_id": f"p{p}", "bpm": bpm,
                             "p_acq": 0.15 + 0.05 * team,          # raw P(moves) = 1.2: far too high
                             "acquired": int(team == moved_to)})
    p = pd.DataFrame(rows)
    out = aq.calibrate_moves(p, min_history=3)
    late = out[out["season"] >= 2013].groupby(["season", "player_id"])
    # P(moves) pulled toward the real ~30%; seasons without history untouched.
    assert late["p_acq"].sum().mean() == pytest.approx(0.3, abs=0.1)
    early = out[out["season"] < 2013]
    assert np.allclose(early["p_acq"], p[p["season"] < 2013]["p_acq"])
    # Where he goes, given he moves, is unchanged.
    share = out[out["season"] >= 2013].groupby(["season", "player_id"])["p_acq"].transform(lambda s: s / s.sum())
    assert np.allclose(share.groupby(out["TeamId"]).mean(), [0.15, 0.20, 0.25, 0.30] / np.float64(0.9))
