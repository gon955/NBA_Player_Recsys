import numpy as np
import pandas as pd
import pytest

import rubric


def _member():
    # 2020: a on 1, b on 1, c on 2 (and 3 after a trade), d on 2.
    # 2021: a stays on 1, b moves to 2, c plays for 3 (one of his 2020 teams),
    #       d barely plays. 2021 is the last season, so it has no status.
    return pd.DataFrame({
        "season":    [2020, 2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021],
        "TeamId":    [1,    1,    2,    3,    2,    1,    2,    3,    2],
        "player_id": ["a",  "b",  "c",  "c",  "d",  "a",  "b",  "c",  "d"],
        "share":     [0.5,  0.4,  0.3,  0.2,  0.3,  0.5,  0.35, 0.3,  0.01],
    })


def test_player_status_moved_stayed_gone():
    st = rubric.player_status(_member())
    assert st.loc[("a", 2020), "status"] == "stayed"
    assert st.loc[("b", 2020), "status"] == "moved"
    assert st.loc[("b", 2020), "share_new"] == pytest.approx(0.35)
    assert st.loc[("c", 2020), "status"] == "stayed"      # back with a season-t team
    assert st.loc[("d", 2020), "status"] == "gone"        # under 5% everywhere
    assert np.isnan(st.loc[("a", 2020), "share_new"])
    assert 2021 not in st.index.get_level_values("season")


def test_top_lists_drop_ruled_out_and_rank():
    scores = pd.DataFrame({
        "season": 2020, "TeamId": [1, 1, 1, 1],
        "player_id": ["x", "y", "z", "w"],
        "score": [3.0, 5.0, -2e3, -np.inf],   # z soft-penalised, w ruled out
    })
    top = rubric.top_lists(scores, k=10)
    assert top["player_id"].tolist() == ["y", "x"]
    assert top["rank"].tolist() == [1, 2]


def test_availability_and_star_share():
    status = rubric.player_status(_member())
    lists = pd.DataFrame({"season": 2020, "TeamId": [9, 9, 9, 9],
                          "player_id": ["a", "b", "c", "d"], "rank": [1, 2, 3, 4]})
    stars = pd.Series([True, False, False, False],
                      index=pd.MultiIndex.from_tuples([("a", 2020), ("b", 2020), ("c", 2020),
                                                       ("d", 2020)], names=rubric.KEY))
    out = rubric.availability(lists, status, stars)
    assert out["moved"] == 0.25 and out["stayed"] == 0.5 and out["gone"] == 0.25
    assert out["stars"] == 0.25


def test_value_accuracy_counts_each_mover_once():
    idx = pd.MultiIndex.from_tuples([("p", 2020), ("q", 2020), ("r", 2020), ("s", 2020)],
                                    names=rubric.KEY)
    outcome = pd.DataFrame({"status": ["moved", "moved", "moved", "stayed"],
                            "share_new": [0.5, 0.3, 0.1, np.nan],
                            "bpm_next": [3.0, 1.0, -1.0, 9.0],
                            "impact_next": [2.0, 0.5, -1.0, 9.0]}, index=idx)
    # p is listed by two teams; s stayed, so is not a mover.
    lists = pd.DataFrame({"season": 2020, "TeamId": [1, 2, 1, 1, 1],
                          "player_id": ["p", "p", "q", "r", "s"], "rank": [1, 1, 2, 3, 4]})
    pred = pd.Series([3.0, 1.0, 0.0, 0.0], index=idx)
    out = rubric.value_accuracy(lists, outcome, pred)
    assert out["movers"] == 3
    assert out["bpm_next"] == pytest.approx(1.0)
    assert out["rho impact_next"] == pytest.approx(1.0)
    assert out["bias"] == pytest.approx(((3 - 2) + (1 - 0.5) + (0 + 1)) / 3)
    assert np.isnan(rubric.value_accuracy(lists, outcome, pred, bias=False)["bias"])


def test_specificity_identical_vs_disjoint_lists():
    same = pd.DataFrame({"season": 2020, "TeamId": [1, 1, 2, 2],
                         "player_id": ["a", "b", "a", "b"], "rank": [1, 2, 1, 2]})
    apart = same.assign(player_id=["a", "b", "c", "d"])
    s, a = rubric.specificity(same), rubric.specificity(apart)
    assert s["jaccard"] == 1.0 and s["distinct"] == 0.5
    assert a["jaccard"] == 0.0 and a["distinct"] == 1.0


def test_holdout_log_appends_and_counts(tmp_path):
    path = tmp_path / "log.csv"
    assert rubric.log_holdout(path, {"realism_weight": 6.0}) == 0
    assert rubric.log_holdout(path, {"realism_weight": 4.0}) == 1
    log = pd.read_csv(path)
    assert log["realism_weight"].tolist() == [6.0, 4.0]
