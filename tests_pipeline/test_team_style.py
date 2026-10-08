import numpy as np
import pandas as pd
import pytest

import fit_model as fm
import team_style as ts


def _stats(**overrides):
    row = dict.fromkeys(ts.COUNT_COLS, 0.0)
    row.update({
        "season": 2020, "TeamId": 1, "EntityId": "1-2-3-4-5", "TeamAbbreviation": "BOS",
        "SecondsPlayed": 2880.0, "OffPoss": 100.0, "DefPoss": 100.0,
        "Points": 110.0, "OpponentPoints": 100.0,
        "FG2A": 60.0, "FG2M": 30.0, "FG3A": 40.0, "FG3M": 14.0, "FTA": 20.0,
        "AtRimFGA": 30.0, "ShortMidRangeFGA": 20.0, "LongMidRangeFGA": 10.0,
        "Corner3FGA": 10.0, "Arc3FGA": 30.0,
        "Assists": 22.0, "OffRebounds": 14.0, "DefRebounds": 35.0,
        "Turnovers": 13.0, "Steals": 8.0, "Blocks": 5.0,
    })
    row.update(overrides)
    return row


def test_team_style_rates_from_counts():
    # Two lineups of one team, split evenly: rates must match the single total.
    half = {k: v / 2 for k, v in _stats().items() if isinstance(v, float)}
    df = pd.DataFrame([_stats(**half, EntityId="a"), _stats(**half, EntityId="b")])
    s = ts.team_style(df).iloc[0]

    assert s["pace"] == pytest.approx(100.0)          # 100 poss in 48 minutes
    assert s["rim_freq"] == pytest.approx(0.30)
    assert s["three_freq"] == pytest.approx(0.40)
    assert s["mid_freq"] == pytest.approx(0.30)
    assert s["ft_rate"] == pytest.approx(0.20)
    assert s["ast_rate"] == pytest.approx(22 / 44)
    assert s["oreb_rate"] == pytest.approx(14 / 56)   # 100 FGA - 44 makes
    assert s["net_rtg"] == pytest.approx(10.0)


def test_quality_columns_stay_out_of_style():
    assert not set(ts.QUALITY) & set(ts.STYLE)
    assert not set(ts.QUALITY) & set(ts.STYLE_FOR_FIT)


def test_zscore_is_within_season():
    df = pd.DataFrame({"season": [1, 1, 2, 2], "pace": [90.0, 92.0, 100.0, 104.0]})
    z = ts.zscore_within_season(df, ["pace"])["z_pace"].tolist()
    assert z == pytest.approx([-1, 1, -1, 1])


def test_archetype_mix_sums_to_five_per_team():
    panel = pd.DataFrame({
        "season": [2020, 2020],
        "TeamId": [1, 1],
        **{f"p{k}_id": [f"a{k}", f"b{k}"] for k in range(1, 6)},
        "OffPoss": [300.0, 100.0], "DefPoss": [300.0, 100.0],
    })
    labels = pd.DataFrame({
        "player_id": [f"a{k}" for k in range(1, 6)] + ["b1"],
        "season": 2020,
        "cluster_label": ["Big"] * 5 + ["Guard"],
    })
    mix = ts.archetype_mix(panel, labels).iloc[0]

    assert mix.filter(like="mix_").sum() == pytest.approx(5.0)
    # 3/4 of possessions are the all-Big unit; the other has 0 Bigs.
    assert mix["mix_Big"] == pytest.approx(3.75)
    assert mix["mix_Guard"] == pytest.approx(0.25)
    assert mix["mix_unknown"] == pytest.approx(1.0)


def test_style_design_shape_and_interactions():
    vocab = fm.Vocab(["Big", "Guard"], pairs=False)
    names = np.array([["Big", "Big", "Guard", "Guard", "Guard"]])
    z = np.array([[2.0, -1.0]])
    X = ts.style_design(names, z, vocab).toarray()[0]

    k = vocab.n_cols
    assert len(X) == k * 3
    assert X[:k].tolist() == [2, 3, 0]          # mains: Big, Guard, unknown
    assert X[k:2 * k].tolist() == [4, 6, 0]     # x style 1 (z = 2)
    assert X[2 * k:].tolist() == [-2, -3, 0]    # x style 2 (z = -1)
