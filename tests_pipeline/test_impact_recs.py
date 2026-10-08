import numpy as np
import pandas as pd
import pytest

import impact_recs as ir


def _setup():
    # Team 1 plays one unit all season: a1 (weak PG) + four average forwards.
    # Team 2 owns the candidates: s1 (great PG) and c1 (great C).
    panel = pd.DataFrame({
        "season": [2020, 2020],
        "TeamId": [1, 2],
        "p1_id": ["a1", "s1"], "p2_id": ["a2", "c1"], "p3_id": ["a3", "b3"],
        "p4_id": ["a4", "b4"], "p5_id": ["a5", "b5"],
        "OffPoss": [1000.0, 1000.0], "DefPoss": [1000.0, 1000.0],
    })
    ptab = pd.DataFrame({
        "player_id": ["a1", "a2", "a3", "a4", "a5", "s1", "c1", "b3", "b4", "b5"],
        "value": [-3.0, 0, 0, 0, 0, 5.0, 6.0, 0, 0, 0],
        "pos_index": [1.0, 4.2, 4.2, 4.2, 4.2, 1.2, 5.0, 3, 3, 3],
        "cluster_label": "unknown",
        "off_poss": 3000.0,
    }).set_index("player_id")
    teams = pd.DataFrame({"season": [2020, 2020], "TeamId": [1, 2], "user_id": ["T1_2020", "T2_2020"]})
    F = pd.DataFrame(0.0, index=["unknown"], columns=["unknown"])
    return panel, ptab, teams, F


def test_gain_is_share_times_value_difference():
    panel, ptab, teams, F = _setup()
    recs = ir.recommend_season(2020, panel, ptab, F, teams)
    t1 = recs[recs["user_id"] == "T1_2020"].set_index("player_id")

    # s1 replaces a1, who played every possession: 1.0 * (5 - -3) = 8.
    assert t1.loc["s1", "replaces"] == "a1"
    assert t1.loc["s1", "score"] == pytest.approx(8.0)


def test_position_window_blocks_out_of_position_swaps():
    panel, ptab, teams, F = _setup()
    recs = ir.recommend_season(2020, panel, ptab, F, teams)
    t1 = recs[recs["user_id"] == "T1_2020"].set_index("player_id")

    # c1 (pos 5.0) may not replace a1 (pos 1.0); its best legal swap is a
    # forward at 3.5, worth 1.0 * (6 - 0).
    assert t1.loc["c1", "replaces"] != "a1"
    assert t1.loc["c1", "score"] == pytest.approx(6.0)
    assert list(t1.sort_values("rank").index[:2]) == ["s1", "c1"]


def test_own_players_are_never_recommended():
    panel, ptab, teams, F = _setup()
    recs = ir.recommend_season(2020, panel, ptab, F, teams)
    for uid, own in [("T1_2020", {"a1", "a2", "a3", "a4", "a5"}),
                     ("T2_2020", {"s1", "c1", "b3", "b4", "b5"})]:
        assert not set(recs.loc[recs["user_id"] == uid, "player_id"]) & own
    assert np.isfinite(recs["score"]).all()


def _with_salaries(ptab, **sal):
    base = dict.fromkeys(ptab.index, 1_000_000.0)
    return ptab.assign(salary=pd.Series(base | sal))


def test_salary_screen_applies_matching_rule():
    panel, ptab, teams, F = _setup()
    # a1 earns $4M: over the cap, T1 can take back 1.25 * 4M + 100k = $5.1M.
    ptab = _with_salaries(ptab, a1=4_000_000.0, s1=5_100_000.0, c1=30_000_000.0)
    room = pd.Series({1: 0.0, 2: 0.0})
    recs = ir.recommend_season(2020, panel, ptab, F, teams, room=room, max_gap=None)
    t1 = recs[recs["user_id"] == "T1_2020"].set_index("player_id")

    assert t1.loc["s1", "replaces"] == "a1"
    assert "c1" not in t1.index          # $30M fits nobody's outgoing salary


def test_salary_screen_cap_room_absorbs_salary():
    panel, ptab, teams, F = _setup()
    ptab = _with_salaries(ptab, a1=4_000_000.0, s1=5_100_001.0)
    no_room = ir.recommend_season(2020, panel, ptab, F, teams,
                                  room=pd.Series({1: 0.0, 2: 0.0}), max_gap=None)
    t1 = no_room[no_room["user_id"] == "T1_2020"].set_index("player_id")
    assert "s1" not in t1.index or t1.loc["s1", "replaces"] != "a1"

    # With $2M of room, outgoing 4M + room 2M = $6M covers $5.1M.
    roomy = ir.recommend_season(2020, panel, ptab, F, teams,
                                room=pd.Series({1: 2e6, 2: 0.0}), max_gap=None)
    t1 = roomy[roomy["user_id"] == "T1_2020"].set_index("player_id")
    assert t1.loc["s1", "replaces"] == "a1"


def test_salary_screen_filler_and_unknown_salaries():
    panel, ptab, teams, F = _setup()
    # A bench player (b9) on T1 with a $20M contract and a sliver of minutes.
    bench = panel.iloc[[0]].assign(p5_id="b9", OffPoss=1.0, DefPoss=1.0)
    panel = pd.concat([panel, bench], ignore_index=True)
    b9 = pd.DataFrame({"value": [0.0], "pos_index": [3.0], "cluster_label": ["unknown"],
                       "off_poss": [1.0]}, index=pd.Index(["b9"], name="player_id"))
    ptab = _with_salaries(pd.concat([ptab, b9]), a1=4_000_000.0, b9=20_000_000.0,
                          c1=np.nan, s1=25_000_000.0)
    room = pd.Series({1: 0.0, 2: 0.0})

    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731
    assert "s1" not in t1(ir.recommend_season(2020, panel, ptab, F, teams, room=room, max_gap=None)).index
    # Adding b9's $20M: 1.25 * 24M + 100k covers s1's $25M.
    with_filler = t1(ir.recommend_season(2020, panel, ptab, F, teams, room=room, filler=1,
                                         max_gap=None))
    assert with_filler.loc["s1", "replaces"] == "a1"
    assert "c1" not in with_filler.index   # unknown salary: never recommended


def test_value_cap_blocks_lopsided_trades():
    panel, ptab, teams, F = _setup()
    ptab = _with_salaries(ptab, **dict.fromkeys(["a1", "a2", "a3", "a4", "a5", "s1", "c1"], 4e6))
    # Same salaries, so only surplus separates them: s1 is a star on a cheap
    # deal (+6 over a1), c1 a modest step up (+1 over the forward it replaces).
    ptab = ptab.assign(surplus=pd.Series({"a1": -2.0, "s1": 4.0, "c1": 1.0}).reindex(ptab.index)
                       .fillna(0.0))
    room = pd.Series({1: 0.0, 2: 0.0})
    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731

    open_ = t1(ir.recommend_season(2020, panel, ptab, F, teams, room=room, max_gap=None))
    assert open_.loc["s1", "replaces"] == "a1"

    capped = t1(ir.recommend_season(2020, panel, ptab, F, teams, room=room, max_gap=2.0))
    assert "s1" not in capped.index            # gap 6 vs a1, 4 vs any forward
    assert capped.loc["c1", "replaces"] != "a1"


def test_pending_free_agent_signs_with_budget_not_trade_rules():
    panel, ptab, teams, F = _setup()
    # s1: a star whose salary no outgoing contract on T1 can match ($30M vs
    # $4M), and a +6 surplus gap: unreachable by trade either way.
    ptab = _with_salaries(ptab, a1=4e6, s1=30e6).assign(
        surplus=lambda d: np.where(d.index == "s1", 6.0, 0.0),
        pending_fa=lambda d: d.index == "s1",
        fa_cost=20e6)
    room = pd.Series({1: 0.0, 2: 0.0})
    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731

    assert "s1" not in t1(ir.recommend_season(2020, panel, ptab, F, teams, room=room)).index
    poor = ir.recommend_season(2020, panel, ptab, F, teams, room=room,
                               fa_budget=pd.Series({1: 10e6, 2: 0.0}))
    assert "s1" not in t1(poor).index      # price 20M over a 10M budget
    rich = t1(ir.recommend_season(2020, panel, ptab, F, teams, room=room,
                                  fa_budget=pd.Series({1: 25e6, 2: 0.0})))
    assert rich.loc["s1", "via"] == "free agency" and rich.loc["s1", "replaces"] == "a1"


def test_pending_fa_flags_contract_breaks():
    sal = pd.DataFrame({
        "player_id": ["p"] * 4 + ["q"] * 2,
        "season":    [2020, 2021, 2022, 2023, 2020, 2021],
        "salary":    [10e6, 10.8e6, 25e6, 26e6, 5e6, 4.4e6],
    })
    f = ir.pending_fa(sal)
    # p: one contract 2020-21 (+8%), a new deal for 2022; 2023 has no t+1 data.
    assert not f[("p", 2020)] and f[("p", 2021)] and not f[("p", 2022)]
    assert f[("p", 2023)]
    assert f[("q", 2020)]                  # a 12% cut is a new deal


def test_fa_prices_use_only_earlier_seasons():
    from fetch_salaries import CAP

    def tab(values, pcts):
        return pd.DataFrame({"value": values, "cap_pct": pcts, "pending_fa": True,
                             "off_poss": 3000.0}, index=pd.Index(["x", "y", "z"], name="player_id"))
    ptabs = {2010: tab([0.0, 1.0, 2.0], [0.05, 0.10, 0.15]),
             2011: tab([0.0, 1.0, 2.0], [0.05, 0.10, 0.15])}
    # 2010's free agents re-signed at exactly 2x their old share; 2011's at 1x.
    sal = pd.DataFrame({"player_id": list("xyz") * 2, "season": [2011] * 3 + [2012] * 3,
                        "salary": [s * CAP[y] for y, k in [(2011, 2), (2012, 1)]
                                   for s in (0.05 * k, 0.10 * k, 0.15 * k)]})
    ir.fa_prices(ptabs, sal)
    # 2010 has no history: priced at its current share. 2011 learns 2010's 2x,
    # not its own 1x.
    assert ptabs[2010]["fa_cost"].tolist() == pytest.approx([0.05 * CAP[2011], 0.10 * CAP[2011],
                                                              0.15 * CAP[2011]])
    assert ptabs[2011].loc["z", "fa_cost"] == pytest.approx(0.30 * CAP[2012])


def test_scoring_modes():
    panel, ptab, teams, F = _setup()
    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731
    # s1 (5.0) and c1 (6.0) each carry a 1.0 share on team 2.
    val = t1(ir.recommend_season(2020, panel, ptab, F, teams, scoring="value"))
    assert val.loc["c1", "score"] == pytest.approx(6.0)
    mins = t1(ir.recommend_season(2020, panel, ptab, F, teams, scoring="minutes"))
    assert mins.loc["s1", "score"] == pytest.approx(5.0)
    # displace: s1 (pos 1.2) sees only a1 (pos 1.0, value -3) within one spot:
    # 1.0 * (5 - -3) = 8. c1 (pos 5.0) sees the four 4.2 forwards at 0: 6.
    disp = t1(ir.recommend_season(2020, panel, ptab, F, teams, scoring="displace"))
    assert disp.loc["s1", "score"] == pytest.approx(8.0)
    assert disp.loc["c1", "score"] == pytest.approx(6.0)
