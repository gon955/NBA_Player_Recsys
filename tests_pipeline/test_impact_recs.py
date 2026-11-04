import numpy as np
import pandas as pd
import pytest

import impact_recs as ir


@pytest.fixture(autouse=True)
def _no_fit_term(monkeypatch):
    """The scoring tests below pin exact scores from the value formulas; the
    positional fit term (FIT_WEIGHT) is tested on its own, by passing
    fit_weight explicitly."""
    monkeypatch.setattr(ir, "FIT_WEIGHT", 0.0)


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
    return panel, ptab, teams


def test_gain_is_share_times_value_difference():
    panel, ptab, teams = _setup()
    recs = ir.recommend_season(2020, panel, ptab, teams, scoring="replace")
    t1 = recs[recs["user_id"] == "T1_2020"].set_index("player_id")

    # s1 replaces a1, who played every possession: 1.0 * (5 - -3) = 8.
    assert t1.loc["s1", "replaces"] == "a1"
    assert t1.loc["s1", "score"] == pytest.approx(8.0)


def test_position_window_blocks_out_of_position_swaps():
    panel, ptab, teams = _setup()
    recs = ir.recommend_season(2020, panel, ptab, teams, scoring="replace")
    t1 = recs[recs["user_id"] == "T1_2020"].set_index("player_id")

    # c1 (pos 5.0) may not replace a1 (pos 1.0); its best legal swap is a
    # forward at 3.5, worth 1.0 * (6 - 0).
    assert t1.loc["c1", "replaces"] != "a1"
    assert t1.loc["c1", "score"] == pytest.approx(6.0)
    assert list(t1.sort_values("rank").index[:2]) == ["s1", "c1"]


def test_own_players_are_never_recommended():
    panel, ptab, teams = _setup()
    recs = ir.recommend_season(2020, panel, ptab, teams)
    for uid, own in [("T1_2020", {"a1", "a2", "a3", "a4", "a5"}),
                     ("T2_2020", {"s1", "c1", "b3", "b4", "b5"})]:
        assert not set(recs.loc[recs["user_id"] == uid, "player_id"]) & own
    assert np.isfinite(recs["score"]).all()


def _with_salaries(ptab, **sal):
    base = dict.fromkeys(ptab.index, 1_000_000.0)
    return ptab.assign(salary=pd.Series(base | sal))


def test_salary_screen_applies_matching_rule():
    panel, ptab, teams = _setup()
    # a1 earns $4M: over the cap, T1 can take back 1.25 * 4M + 100k = $5.1M.
    ptab = _with_salaries(ptab, a1=4_000_000.0, s1=5_100_000.0, c1=30_000_000.0)
    room = pd.Series({1: 0.0, 2: 0.0})
    recs = ir.recommend_season(2020, panel, ptab, teams, room=room, max_gap=None, aggregate=0)
    t1 = recs[recs["user_id"] == "T1_2020"].set_index("player_id")

    assert t1.loc["s1", "replaces"] == "a1"
    assert t1.loc["s1", "with"] == ""
    assert "c1" not in t1.index          # $30M fits nobody's outgoing salary

    # With packages, the most salary-balanced legal deal is named: a2 ($1M)
    # plus a1 ($4M) sends $5M for $5.1M, closer than a1 alone.
    packaged = ir.recommend_season(2020, panel, ptab, teams, room=room, max_gap=None, aggregate=1)
    p1 = packaged[packaged["user_id"] == "T1_2020"].set_index("player_id")
    assert {p1.loc["s1", "replaces"], p1.loc["s1", "with"]} == {"a1", "a2"}


def test_salary_screen_cap_room_absorbs_salary():
    panel, ptab, teams = _setup()
    ptab = _with_salaries(ptab, a1=4_000_000.0, s1=5_100_001.0)
    # r's salary alone (aggregate=0): the matching rule itself.
    no_room = ir.recommend_season(2020, panel, ptab, teams,
                                  room=pd.Series({1: 0.0, 2: 0.0}), max_gap=None, aggregate=0)
    t1 = no_room[no_room["user_id"] == "T1_2020"].set_index("player_id")
    assert "s1" not in t1.index or t1.loc["s1", "replaces"] != "a1"

    # With $2M of room, outgoing 4M + room 2M = $6M covers $5.1M.
    roomy = ir.recommend_season(2020, panel, ptab, teams,
                                room=pd.Series({1: 2e6, 2: 0.0}), max_gap=None, aggregate=0)
    t1 = roomy[roomy["user_id"] == "T1_2020"].set_index("player_id")
    assert t1.loc["s1", "replaces"] == "a1"


def test_salary_screen_filler_and_unknown_salaries():
    panel, ptab, teams = _setup()
    # A bench player (b9) on T1 with a $20M contract and a sliver of minutes.
    bench = panel.iloc[[0]].assign(p5_id="b9", OffPoss=1.0, DefPoss=1.0)
    panel = pd.concat([panel, bench], ignore_index=True)
    b9 = pd.DataFrame({"value": [0.0], "pos_index": [3.0], "cluster_label": ["unknown"],
                       "off_poss": [1.0]}, index=pd.Index(["b9"], name="player_id"))
    ptab = _with_salaries(pd.concat([ptab, b9]), a1=4_000_000.0, b9=20_000_000.0,
                          c1=np.nan, s1=25_000_000.0)
    room = pd.Series({1: 0.0, 2: 0.0})

    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731
    alone = t1(ir.recommend_season(2020, panel, ptab, teams, room=room, max_gap=None, aggregate=0))
    assert "s1" not in alone.index
    # Adding b9's $20M: 1.25 * 24M + 100k covers s1's $25M.
    with_filler = t1(ir.recommend_season(2020, panel, ptab, teams, room=room, filler=1,
                                         max_gap=None, aggregate=0))
    assert with_filler.loc["s1", "replaces"] == "a1"
    assert "c1" not in with_filler.index   # unknown salary: never recommended
    # Aggregating one lesser contract finds the same package without FILLER:
    # b9 is out of the rotation, so he can go out with a1.
    packaged = t1(ir.recommend_season(2020, panel, ptab, teams, room=room, max_gap=None, aggregate=1))
    assert packaged.loc["s1", "replaces"] == "a1"
    assert "c1" not in packaged.index


def test_value_cap_blocks_lopsided_trades():
    panel, ptab, teams = _setup()
    ptab = _with_salaries(ptab, **dict.fromkeys(["a1", "a2", "a3", "a4", "a5", "s1", "c1"], 4e6))
    # Same salaries, so only surplus separates them: s1 is a star on a cheap
    # deal (+6 over a1), c1 a modest step up (+1 over the forward it replaces).
    ptab = ptab.assign(surplus=pd.Series({"a1": -2.0, "s1": 4.0, "c1": 1.0}).reindex(ptab.index)
                       .fillna(0.0))
    room = pd.Series({1: 0.0, 2: 0.0})
    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731

    open_ = t1(ir.recommend_season(2020, panel, ptab, teams, room=room, max_gap=None))
    assert open_.loc["s1", "replaces"] == "a1"

    capped = t1(ir.recommend_season(2020, panel, ptab, teams, room=room, max_gap=2.0))
    assert "s1" not in capped.index            # gap 6 vs a1, 4 vs any forward
    assert capped.loc["c1", "replaces"] != "a1"


def test_pending_free_agent_signs_with_budget_not_trade_rules():
    panel, ptab, teams = _setup()
    # s1: a star whose salary no outgoing contract on T1 can match ($30M vs
    # $4M), and a +6 surplus gap: unreachable by trade either way.
    ptab = _with_salaries(ptab, a1=4e6, s1=30e6).assign(
        surplus=lambda d: np.where(d.index == "s1", 6.0, 0.0),
        pending_fa=lambda d: d.index == "s1",
        fa_cost=20e6)
    room = pd.Series({1: 0.0, 2: 0.0})
    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731

    assert "s1" not in t1(ir.recommend_season(2020, panel, ptab, teams, room=room)).index
    poor = ir.recommend_season(2020, panel, ptab, teams, room=room,
                               fa_budget=pd.Series({1: 10e6, 2: 0.0}))
    assert "s1" not in t1(poor).index      # price 20M over a 10M budget
    rich = t1(ir.recommend_season(2020, panel, ptab, teams, room=room,
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
    panel, ptab, teams = _setup()
    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731
    # s1 (5.0) and c1 (6.0) each carry a 1.0 share on team 2.
    val = t1(ir.recommend_season(2020, panel, ptab, teams, scoring="value"))
    assert val.loc["c1", "score"] == pytest.approx(6.0)
    mins = t1(ir.recommend_season(2020, panel, ptab, teams, scoring="minutes"))
    assert mins.loc["s1", "score"] == pytest.approx(5.0)
    # displace: s1 (pos 1.2) sees only a1 (pos 1.0, value -3) within one spot:
    # 1.0 * (5 - -3) = 8. c1 (pos 5.0) sees the four 4.2 forwards at 0: 6.
    disp = t1(ir.recommend_season(2020, panel, ptab, teams, scoring="displace"))
    assert disp.loc["s1", "score"] == pytest.approx(8.0)
    assert disp.loc["c1", "score"] == pytest.approx(6.0)


def _two_point_guards():
    # Team 1 splits its PG minutes between a1 (-3) and a6 (+3) in two equal units.
    panel, ptab, teams = _setup()
    second = panel.iloc[[0]].assign(p1_id="a6")
    panel = pd.concat([panel.iloc[[0]], second, panel.iloc[[1]]], ignore_index=True)
    ptab.loc["a6"] = {"value": 3.0, "pos_index": 1.1, "cluster_label": "unknown", "off_poss": 3000.0}
    return panel, ptab, teams


def test_expected_scores_against_minutes_weighted_value_at_position():
    panel, ptab, teams = _two_point_guards()
    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731
    exp = t1(ir.recommend_season(2020, panel, ptab, teams, scoring="expected"))
    rep = t1(ir.recommend_season(2020, panel, ptab, teams, scoring="replace"))

    # s1 carried all of team 2's floor time; the PG minutes he would take are
    # worth 0.5 * -3 + 0.5 * 3 = 0, so expected = 1.0 * (5 - 0).
    assert exp.loc["s1", "score"] == pytest.approx(5.0)
    # replace instead credits swapping out the weakest PG alone: 0.5 * (5 - -3).
    assert rep.loc["s1", "score"] == pytest.approx(4.0)
    # The outgoing player named is the legal one whose departure costs least.
    assert exp.loc["s1", "replaces"] == "a1"


def test_quality_is_the_default_scoring():
    panel, ptab, teams = _two_point_guards()
    default = ir.recommend_season(2020, panel, ptab, teams)
    q = ir.recommend_season(2020, panel, ptab, teams, scoring="quality")
    pd.testing.assert_frame_equal(default, q)


def test_quality_scores_value_and_lets_any_position_go_out():
    panel, ptab, teams = _setup()
    # Team 1's only rotation guard is a1; the forwards cost least to send out.
    ptab.loc["a1", "value"] = 2.0
    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731
    q = t1(ir.recommend_season(2020, panel, ptab, teams, scoring="quality"))
    ruled = t1(ir.recommend_season(2020, panel, ptab, teams, scoring="quality", pos_rule=True))

    assert q.loc["s1", "score"] == pytest.approx(5.0)
    assert q.loc["c1", "score"] == pytest.approx(6.0)
    # Without the position rule, the PG s1 can come in for a forward (value 0,
    # share 1) rather than the better a1 (value 2).
    assert q.loc["s1", "replaces"] in {"a2", "a3", "a4", "a5"}
    assert ruled.loc["s1", "replaces"] == "a1"


def _impact_history(n_seasons=8, n_players=60, seed=0):
    rng = np.random.default_rng(seed)
    talent = rng.normal(0, 2, n_players)
    rows = []
    for s in range(2000, 2000 + n_seasons):
        noise = rng.normal(0, 2, n_players)
        rows.append(pd.DataFrame({"player_id": [f"p{i}" for i in range(n_players)], "season": s,
                                  "net_impact": talent + noise, "off_poss": 2000.0}))
    return pd.concat(rows, ignore_index=True)


def test_projection_shrinks_noisy_estimates_and_keeps_raw_without_history():
    impact = _impact_history()
    p = ir.project(impact)
    first = p[p["season"] < 2000 + ir.PROJ_MIN_HISTORY]
    late = p[p["season"] == 2007]

    # Not enough earlier pairs: the raw estimate is kept.
    np.testing.assert_allclose(first["proj_impact"], first["net_impact"])
    # Half of each season's variance is noise, so the projection is pulled
    # toward the mean and spreads less than the raw estimates.
    assert late["proj_impact"].std() < 0.9 * late["net_impact"].std()
    assert np.corrcoef(late["proj_impact"], late["net_impact"])[0, 1] > 0.6


def test_projection_never_reads_the_future():
    impact = _impact_history()
    changed = impact.copy()
    changed.loc[changed["season"] == 2007, "net_impact"] *= -5
    a = ir.project(impact).set_index(["player_id", "season"])["proj_impact"]
    b = ir.project(changed).set_index(["player_id", "season"])["proj_impact"]

    # Scrambling 2007 leaves every earlier projection untouched.
    pd.testing.assert_series_equal(a.xs(2006, level="season"), b.xs(2006, level="season"))


def test_soft_ranks_ruled_out_moves_below_legal_ones_but_in_order():
    panel, ptab, teams = _setup()
    ptab["salary"] = 1e6
    ptab.loc[["s1", "c1"], "salary"] = [50e6, 60e6]      # unaffordable for team 1
    ptab["surplus"] = 0.0
    room = pd.Series({1: 0.0, 2: 0.0})
    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731
    hard = t1(ir.recommend_season(2020, panel, ptab, teams, room=room, k=None, min_poss=0))
    soft = t1(ir.recommend_season(2020, panel, ptab, teams, room=room, k=None, min_poss=0, soft=True))
    listed = t1(ir.recommend_season(2020, panel, ptab, teams, room=room, min_poss=0, soft=True))

    assert np.isneginf(hard.loc[["s1", "c1"], "score"]).all()
    # Soft: still below every legal candidate, but c1 (6) ahead of s1 (5).
    legal = soft.drop(index=["s1", "c1"])["score"]
    assert soft.loc["c1", "score"] > soft.loc["s1", "score"]
    assert soft.loc[["s1", "c1"], "score"].max() < legal.min()
    # The top-k list never shows a ruled-out move.
    assert not {"s1", "c1"} & set(listed.index)


def test_creator_term_favours_teams_short_on_creators():
    panel, ptab, teams = _setup()
    # Team 1 has no creators; team 2's lineup has two (s1 and c1).
    ptab["creator"] = ptab.index.isin(["s1", "c1", "b3"])
    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731
    t2 = lambda r: r[r["user_id"] == "T2_2020"].set_index("player_id")  # noqa: E731
    recs = ir.recommend_season(2020, panel, ptab, teams, scoring="quality", creator_b2=-0.5)
    off = ir.recommend_season(2020, panel, ptab, teams, scoring="quality")

    # League mean n_bar = (0 + 3) / 2 = 1.5 creators per lineup. s1 to team 1
    # (k = 0): 2 * -0.5 * (0 - 1.5) = +1.5 on top of value 5.
    assert t1(recs).loc["s1", "creator_adj"] == pytest.approx(1.5)
    assert t1(recs).loc["s1", "score"] == pytest.approx(t1(off).loc["s1", "score"] + 1.5)
    # Non-creators are untouched; team 2 (k = 3) would value a1 no differently.
    assert t1(recs).loc["b4", "creator_adj"] == 0.0
    assert (t2(recs)["creator_adj"] == 0.0).all()


def _roster():
    # r0 (rotation, value 1, $10M), r1 (rotation, value 3, $20M),
    # b (bench, value 5, $8M), m (rotation, value 0, $5M).
    return (np.array([10.0, 20.0, 8.0, 5.0]), np.array([1.0, 3.0, 5.0, 0.0]),
            np.array([True, True, False, True]))


def test_trade_packages_picks_the_cheapest_legal_package():
    ros_sal, ros_val, ros_rot = _roster()
    r_sal, r_idx = np.array([10.0, 20.0]), np.array([0, 1])
    cv = np.array([2.0, 2.0])
    c_sal = np.array([12.0, 22.0])
    ok, pick, combos, _ = ir.trade_packages(cv, c_sal, r_sal, r_idx, ros_sal, ros_val, ros_rot, 1,
                                         room=0.0, pad=0.1, partner=False)
    # $12M fits r0's $10M alone (1.25 * 10 + 0.1); $22M needs r0 plus the
    # cheapest eligible contract that gets there: m ($5M -> 15 * 1.25 = 18.85,
    # too little) fails, b ($8M -> 18 * 1.25 + 0.1 = 22.6) works. r1 is worth
    # more than the candidate, so he cannot be added.
    assert ok[0, 0] and combos[pick[0, 0]] == ()
    assert ok[1, 0] and combos[pick[1, 0]] == (2,)
    # Through r1 ($20M) alone, $22M fits too.
    assert ok[1, 1] and combos[pick[1, 1]] == ()
    # Without aggregation, r0 alone cannot cover $22M.
    ok0, _, _, _ = ir.trade_packages(cv, c_sal, r_sal, r_idx, ros_sal, ros_val, ros_rot, 0,
                                  room=0.0, pad=0.1, partner=False)
    assert not ok0[1, 0] and ok0[1, 1]


def test_trade_packages_partner_cannot_absorb_far_more_salary():
    ros_sal, ros_val, ros_rot = _roster()
    cv = np.array([2.0])
    # A $3M player for r1's $20M: legal for us, not for an over-cap partner.
    one_sided, _, _, _ = ir.trade_packages(cv, np.array([3.0]), np.array([20.0]), np.array([1]),
                                        ros_sal, ros_val, ros_rot, 1, 0.0, 0.1, partner=False)
    both, _, _, _ = ir.trade_packages(cv, np.array([3.0]), np.array([20.0]), np.array([1]),
                                   ros_sal, ros_val, ros_rot, 1, 0.0, 0.1, partner=True)
    assert one_sided[0, 0] and not both[0, 0]


def _mover_inputs(n_players=60, seasons=range(2010, 2017), seed=0):
    rng = np.random.default_rng(seed)
    rows, inp = [], []
    for p in range(n_players):
        skill = rng.normal()
        for s in seasons:
            rows.append({"player_id": f"p{p}", "season": s, "net_impact": skill + rng.normal(0, 0.5),
                         "off_poss": 2000.0})
            inp.append({"player_id": f"p{p}", "season": s, "age": 22 + (s - 2010) + p % 8,
                        "moved": float(rng.random() < 0.3),
                        **{c: skill + rng.normal(0, 1) for c in ir.PROJ_BOX}})
    return pd.DataFrame(rows), pd.DataFrame(inp).set_index(["player_id", "season"])


def test_project_mover_never_sees_the_future():
    impact, inputs = _mover_inputs()
    base = ir.project_mover(impact, inputs).set_index(["player_id", "season"])
    # Scramble everything about the last season, and every "moved" label that
    # could only be known from it; projections for earlier seasons must not move.
    last = impact["season"].max()
    impact2 = impact.copy()
    impact2.loc[impact2["season"] == last, "net_impact"] = 99.0
    inputs2 = inputs.copy()
    inputs2.loc[inputs2.index.get_level_values("season") == last - 1, "moved"] = 1.0
    after = ir.project_mover(impact2, inputs2).set_index(["player_id", "season"])
    early = base.index.get_level_values("season") < last - 1
    assert np.allclose(base.loc[early, "proj_impact"], after.loc[early, "proj_impact"])
    assert np.allclose(base.loc[early, "proj_stay"], after.loc[early, "proj_stay"])


def test_fit_weight_reorders_positions_not_players():
    panel, ptab, teams = _setup()
    # Two equal candidates; T1's rotation is weak at s1's position only.
    ptab = ptab.copy()
    ptab.loc["c1", ["value", "pos_index"]] = [ptab.loc["s1", "value"], 1.0]
    ptab.loc["s1", "pos_index"] = 5.0
    for pid, pos, val in [("a1", 5.0, -3.0), ("a2", 1.0, 2.0), ("a3", 1.0, 2.0), ("a4", 1.0, 2.0),
                          ("a5", 1.0, 2.0)]:
        ptab.loc[pid, ["pos_index", "value"]] = [pos, val]
    t1 = lambda r: r[r["user_id"] == "T1_2020"].set_index("player_id")  # noqa: E731
    flat = t1(ir.recommend_season(2020, panel, ptab, teams, fit_weight=0.0))
    fit = t1(ir.recommend_season(2020, panel, ptab, teams, fit_weight=2.0))
    assert flat.loc["s1", "score"] == pytest.approx(flat.loc["c1", "score"])
    assert fit.loc["s1", "score"] > fit.loc["c1", "score"]
    assert fit.loc["s1", "need"] > fit.loc["c1", "need"]
    # The value part is untouched.
    assert fit.loc["s1", "gain"] == pytest.approx(flat.loc["s1", "gain"])
