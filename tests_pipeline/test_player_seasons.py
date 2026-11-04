import pandas as pd
import pytest

import player_seasons as ps


def test_full_universe_tot_rows_and_position_index(tmp_path):
    adv = pd.DataFrame({
        "season": [2010, 2010, 2010, 2010],
        "player_id": ["a", "b", "b", "b"],
        "player": ["A", "B", "B", "B"],
        "team": ["BOS", "2TM", "NYK", "MIA"],
        "age": [25, 30, 30, 30], "g": [5, 60, 30, 30], "mp": [40, 1500, 700, 800],
        "bpm": [1.0, 2.0, 1.5, 2.5], "obpm": [0.5, 1.0, 0.7, 1.2], "dbpm": [0.5, 1.0, 0.8, 1.3],
        "usg_percent": [20, 22, 21, 23],
    })
    pbp = pd.DataFrame({
        "season": [2010, 2010], "player_id": ["a", "b"], "team": ["BOS", "2TM"], "mp": [40, 1500],
        "pg_percent": [100, 0], "sg_percent": [0, 0], "sf_percent": [0, 50],
        "pf_percent": [0, 50], "c_percent": [0, 0],
    })
    adv.to_csv(tmp_path / "adv.csv", index=False)
    pbp.to_csv(tmp_path / "pbp.csv", index=False)
    t = ps.build(tmp_path / "adv.csv", tmp_path / "pbp.csv").set_index("player_id")

    # A 5-game player is kept (no games >= 20 filter); a traded player is his TOT row.
    assert list(t.index) == ["a", "b"]
    assert t.loc["b", "bpm"] == 2.0
    assert t.loc["a", "pos_index"] == pytest.approx(1.0)
    assert t.loc["b", "pos_index"] == pytest.approx(3.5)
    assert (t["cluster_label"] == "unknown").all()
