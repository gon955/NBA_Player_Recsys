import numpy as np
import pandas as pd
import pytest

import evaluate_recs as ev


def test_moves_finds_arrivals_and_departures():
    member = pd.DataFrame({
        "season":    [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021],
        "TeamId":    [1,    1,    2,    3,    1,    1,    1,    1],
        "player_id": ["a",  "b",  "c",  "c",  "a",  "c",  "d",  "r"],
        "share":     [0.5,  0.5,  0.3,  0.2,  0.5,  0.3,  0.01, 0.2],
    })
    impact = pd.DataFrame({"player_id": ["a", "b", "c"], "season": 2020,
                           "off_poss": [3000.0, 3000.0, 800.0]})
    debut = pd.Series({"a": 2010, "b": 2010, "c": 2015, "d": 2012, "r": 2021})
    arr, dep = ev.moves(member, impact, debut, min_poss=500)
    arr = arr.set_index("player_id")

    # c (traded between 2 and 3 in 2020) and rookie r joined team 1 for 2021;
    # d played too little to count.
    assert sorted(arr.index) == ["c", "r"]
    assert arr.loc["c", "prior_share"] == pytest.approx(0.5)
    assert arr.loc["c", "rankable"] and not arr.loc["c", "rookie"]
    assert arr.loc["r", "rookie"] and not arr.loc["r", "rankable"]
    # 800 possessions is below a 1500 pool.
    assert not ev.moves(member, impact, debut, min_poss=1500)[0].set_index("player_id").loc["c", "rankable"]
    # b left team 1; c left teams 2 and 3. Nobody "departs" from the last season.
    assert sorted(dep["player_id"]) == ["b", "c", "c"]


def test_percentiles_rank_within_team_and_put_unscored_last():
    pool = pd.DataFrame({"season": 2020, "TeamId": [1, 1, 1, 1],
                         "player_id": ["w", "x", "y", "z"]})
    scores = pd.DataFrame({"season": 2020, "TeamId": [1, 1, 1],
                           "player_id": ["w", "x", "y"], "score": [3.0, 1.0, -np.inf]})
    p = ev.percentiles(pool, scores).set_index("player_id")

    assert p.loc["w", "rank"] == 1
    assert p.loc["w", "pct"] == pytest.approx(0.125)
    # y (ruled out) and z (unscored) tie for last.
    assert p.loc["y", "rank"] == p.loc["z", "rank"] == 3.5


def test_moves_follow_hornets_relocation_and_skip_expansion():
    chh, noh = 1610612766, 1610612740
    member = pd.DataFrame({
        "season":    [2002, 2003, 2003, 2004, 2005],
        "TeamId":    [chh,  noh,  noh,  noh,  chh],
        "player_id": ["bd", "bd", "new", "x",  "x"],
        "share":     [0.5,  0.5,  0.3,  0.5,  0.5],
    })
    impact = pd.DataFrame({"player_id": ["bd", "new", "x"], "season": [2002, 2002, 2004],
                           "off_poss": 3000.0})
    debut = pd.Series({"bd": 1999, "new": 1999, "x": 1999})
    arr, dep = ev.moves(member, impact, debut)
    arr = arr.set_index("player_id")

    # Baron Davis moving with the franchise is not an arrival; "new" is, and is
    # keyed to the Charlotte id the 2002 decision was made under.
    assert "bd" not in arr.index and arr.loc["new", "TeamId"] == chh
    assert not (dep["player_id"].eq("bd") & dep["season"].eq(2002)).any()
    # x joining the 2005 expansion Bobcats has no season-t team to rank against.
    x = arr.loc["x"].set_index("TeamId").loc[chh]
    assert x["expansion"] and not x["rankable"]
