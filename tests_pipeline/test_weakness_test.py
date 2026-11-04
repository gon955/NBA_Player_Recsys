import itertools

import numpy as np
import pandas as pd
import pytest

import weakness_test as wt


def _panel(teams, effect, season=2020, poss=2000, seed=0):
    """Every 5-man combination within each team; offence from `effect(team, combo)`."""
    rng = np.random.default_rng(seed)
    rows = []
    for team, players in teams.items():
        for combo in itertools.combinations(players, 5):
            rows.append({
                **{f"p{k + 1}_id": p for k, p in enumerate(combo)},
                "season": season, "TeamId": team,
                "OffPoss": poss, "DefPoss": poss,
                "Points": poss * (110 + effect(team, combo) + rng.normal(0, 0.01)) / 100,
                "OpponentPoints": poss * 1.10,
            })
    return pd.DataFrame(rows)


def _counts(panel, shooters):
    flags = pd.DataFrame({"player_id": sorted(set(panel[wt.im.SLOTS].to_numpy().ravel()))})
    flags["season"] = panel["season"].iloc[0] - 1
    flags["flag"] = flags["player_id"].isin(shooters)
    flags["eligible"] = True
    return wt.Labels(panel, flags).counts(flags["flag"].to_numpy())


def _b2(panel, n, lam, team_fe):
    fit = wt.SeasonFit(panel, "off", pd.Series(dtype=float), lam)
    return wt.pooled({2020: (panel.index.to_numpy(), fit)}, n, team_fe)[0][1]


def test_recovers_planted_curvature_alongside_player_effects():
    players = [f"p{i}" for i in range(10)]
    shooters = set(players[:4])
    skill = dict(zip(players, [2, 1, 0, -1, 3, 0, 0, -2, 1, 0]))
    panel = _panel({1: players}, lambda _, c: (
        sum(skill[p] for p in c) - 0.4 * sum(p in shooters for p in c) ** 2))
    n = _counts(panel, shooters)

    assert _b2(panel, n, lam=1e-3, team_fe=False) == pytest.approx(-0.4, abs=0.01)
    assert _b2(panel, n, lam=1e-3, team_fe=True) == pytest.approx(-0.4, abs=0.01)


def test_team_effects_absorb_what_shrunk_player_effects_miss():
    # Heavy shrinkage pins players at 0, so a team-wide boost has nowhere to go
    # but the count terms -- unless team intercepts take it.
    teams = {1: [f"a{i}" for i in range(8)], 2: [f"b{i}" for i in range(8)]}
    shooters = {"a0", "a1", "a2", "a3", "a4", "b0"}
    boost = {1: 4.0, 2: -4.0}
    panel = _panel(teams, lambda t, c: boost[t] + 0.3 * sum(p in shooters for p in c) ** 2)
    n = _counts(panel, shooters)

    assert _b2(panel, n, lam=1e12, team_fe=True) == pytest.approx(0.3, abs=0.01)
    assert _b2(panel, n, lam=1e12, team_fe=False) != pytest.approx(0.3, abs=0.05)


def test_labels_use_the_previous_season_and_rookies_count_as_unskilled():
    panel = pd.DataFrame({**{f"p{k}_id": [f"x{k}"] for k in range(1, 6)}, "season": [2020]})
    flags = pd.DataFrame({
        "player_id": ["x1", "x2", "x3", "x1", "x4"],
        "season": [2019, 2019, 2019, 2020, 2020],
        "flag": [True, True, False, False, True],
        "eligible": True,
    })
    labels = wt.Labels(panel, flags)

    # x1, x2 flagged in 2019; x4's flag is from 2020 (same season), ignored.
    assert labels.counts(flags["flag"].to_numpy()).tolist() == [2]
    assert labels.known.tolist() == [False]        # x4, x5 have no 2019 row


def test_shuffle_keeps_prevalence_and_leaves_ineligible_rows_alone():
    flags = pd.DataFrame({
        "player_id": list("abcdefgh"),
        "season": [2019] * 5 + [2020] * 3,
        "flag": [True, True, False, False, False, True, False, False],
        "eligible": [True, True, True, True, False, True, True, True],
    })
    panel = pd.DataFrame({**{f"p{k}_id": ["a"] for k in range(1, 6)}, "season": [2020]})
    labels = wt.Labels(panel, flags)
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = labels.shuffled(rng)
        assert out[:4].sum() == 2 and out[5:].sum() == 1
        assert not out[4]


def test_skill_flags_are_season_relative_and_need_minutes():
    box = pd.DataFrame({
        "season": [2000] * 4 + [2020] * 4,
        "player_id": list("abcdefgh"),
        "mp": [1000, 1000, 1000, 100] * 2,
        "x3pa_per_100_poss": [1, 2, 3, 9, 7, 8.5, 9, 20],
        "x3p": [10, 20, 40, 30, 70, 80, 70, 90],
        "x3pa": [30, 60, 100, 100, 200, 200, 200, 200],
        "blk_percent": 0.0, "drb_percent": 0.0, "usg_percent": 0.0,
    })
    flags = wt.skill_flags(box, "shooter", 0.3).set_index("player_id")["flag"]

    # Volume cut is the 30th percentile of that season's rotation players.
    # 2000 (cut 1.6, league 34.5%): b clears volume but shoots 33%; c makes it;
    # d shoots most but has too few minutes. 2020 (cut 7.9, league 38.75%):
    # f makes it at 40%; g has volume but shoots 35%.
    assert flags[list("abcd")].tolist() == [False, False, True, False]
    assert flags[list("efgh")].tolist() == [False, True, False, False]
