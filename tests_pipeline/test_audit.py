import numpy as np
import pandas as pd
import pytest

import audit


def _pairs():
    """Season 2020: 3 teams x 4 candidates. Candidate s is a star; team 1 is
    the worst. Real arrivals: s never joins team 1; x joins team 1."""
    rows = []
    for team, net in [(1, -8.0), (2, 0.0), (3, 6.0)]:
        for pid, star, age in [("s", True, 30), ("x", False, 33), ("y", False, 25), ("z", False, 26)]:
            rows.append({"season": 2020, "TeamId": team, "player_id": pid, "star": star,
                         "age": age, "team_net": net, "team_net_rank": {1: 1, 2: 2, 3: 3}[team],
                         "team_age_rank": 1, "on_rookie": 0.0, "cap_pct": 0.1,
                         "acquired": int((pid, team) in {("x", 1), ("s", 3), ("y", 2)})})
    return pd.DataFrame(rows)


def test_veto_mask_matches_rule_and_keeps_uncovered_rows():
    rule = audit.Rule("t", "star to the worst team", lambda p: p["star"] & (p["team_net_rank"] <= 1))
    rows = pd.DataFrame({"season": 2020, "TeamId": [1, 1, 3, 7],
                         "player_id": ["s", "x", "s", "s"]})
    assert audit.veto_mask(rows, _pairs(), [rule]).tolist() == [True, False, False, False]
    assert not audit.veto_mask(rows, _pairs(), []).any()


def test_audit_rule_counts_blocked_moves():
    nxt = pd.DataFrame({"season": [2020], "TeamId": [1], "player_id": ["x"],
                        "share_next": [0.4], "bpm_next": [1.0]})
    names = pd.DataFrame({"player_id": ["s", "x", "y", "z"], "season": 2020,
                          "player": ["S", "X", "Y", "Z"]})
    served = pd.DataFrame({"season": [2020, 2020], "TeamId": [1, 2],
                           "player_id": ["x", "x"], "rank": [1, 1]})
    clean = audit.Rule("star_worst", "", lambda p: p["star"] & (p["team_net_rank"] <= 1))
    row, blocked, removed = audit.audit_rule(clean, _pairs(), nxt, names, served, (2020, 2020))
    assert row["vetoed pairs"] == 1 and row["blocked"] == 0 and row["rate ratio"] == 0.0
    assert row["verdict"] == "INCONCLUSIVE"        # one pair: far too few to tell
    assert len(removed) == 0

    wrong = audit.Rule("old_worst", "", lambda p: (p["age"] >= 32) & (p["team_net_rank"] <= 1))
    row, blocked, removed = audit.audit_rule(wrong, _pairs(), nxt, names, served, (2020, 2020))
    assert row["blocked"] == 1 and blocked["player"].tolist() == ["X"]
    assert row["served removed"] == 1 and removed["TeamId"].tolist() == [1]


def test_context_ranks_teams_and_flags_stars():
    pairs = pd.DataFrame({"season": 2020, "TeamId": [1, 1, 2, 2],
                          "player_id": ["a", "b", "a", "b"], "bpm": [9.0, 1.0, 9.0, 1.0],
                          "team_net": [-5.0, -5.0, 5.0, 5.0], "acquired": 0})
    member = pd.DataFrame({"season": 2020, "TeamId": [1, 1, 2],
                           "player_id": ["m", "n", "o"], "share": [0.5, 0.5, 1.0]})
    players = pd.DataFrame({"player_id": ["m", "n", "o"], "season": 2020, "age": [20.0, 24.0, 31.0]})
    p = audit.context(pairs, member, players).set_index(["TeamId", "player_id"])
    assert p.loc[(1, "a"), "star"] and not p.loc[(1, "b"), "star"]
    assert p.loc[(1, "a"), "team_net_rank"] == 1 and p.loc[(2, "a"), "team_net_rank"] == 2
    assert np.isclose(p.loc[(1, "a"), "team_age"], 22.0)
    assert p.loc[(1, "a"), "team_age_rank"] == 1


def test_every_registered_rule_runs_on_the_pair_columns():
    p = _pairs()
    for rule in audit.RULES:
        out = rule.veto(p)
        assert len(out) == len(p), rule.name
    assert not any(r.active for r in audit.RULES), "rules ship inactive until audited"


def _shown():
    return pd.DataFrame({"season": 2025, "TeamId": [1, 1, 2], "player_id": ["a", "b", "a"],
                         "team": ["Bulls", "Bulls", "Hawks"], "player": ["A", "B", "A"],
                         "age": [30, 25, 30], "bpm": [5.0, 1.0, 5.0], "salary_pct_cap": [30.0, 5.0, 30.0],
                         "rookie_deal": "", "replaces": "x", "also_sends": ""})


def test_judge_upserts_and_keeps_a_snapshot():
    j = audit.judge(audit.load_judgments("/nonexistent.csv"),
                    [{"season": 2025, "TeamId": 1, "player_id": "a", "reasons": ["other", "star_wont_move"],
                      "notes": "franchise"}], _shown(), now="t1")
    assert len(j) == 1
    row = j.iloc[0]
    assert (row["verdict"], row["reason"], row["notes"], row["player"]) == \
        ("flag", "star_wont_move;other", "franchise", "A")
    # Unflag, then mark another ok; the first keeps its note and snapshot.
    j = audit.judge(j, [{"season": 2025, "TeamId": 1, "player_id": "a", "reasons": []},
                        {"season": 2025, "TeamId": 1, "player_id": "b", "verdict": "ok"}], _shown(), now="t2")
    j = j.set_index("player_id")
    assert j.loc["a", "verdict"] == "ok" and j.loc["a", "reason"] == "" and j.loc["a", "notes"] == "franchise"
    assert j.loc["b", "verdict"] == "ok" and j.loc["b", "updated"] == "t2"
    with pytest.raises(ValueError):
        audit.judge(j.reset_index(), [{"season": 2025, "TeamId": 1, "player_id": "a", "reasons": ["made_up"]}],
                    _shown())
    with pytest.raises(ValueError):           # never shown, never judged
        audit.judge(j.reset_index(), [{"season": 2025, "TeamId": 7, "player_id": "q", "verdict": "ok"}],
                    _shown())


def test_judgments_round_trip_and_flag_mask(tmp_path):
    path = tmp_path / "judgments.csv"
    j = audit.judge(audit.load_judgments(path),
                    [{"season": 2025, "TeamId": 2, "player_id": "a", "reasons": ["team_direction"]},
                     {"season": 2025, "TeamId": 1, "player_id": "b", "verdict": "ok"}], _shown())
    audit.save_judgments(j, path)
    back = audit.load_judgments(path)
    assert len(back) == 2 and set(back["verdict"]) == {"ok", "flag"}
    rows = pd.DataFrame({"season": [2025, 2025, 2025, 2026], "TeamId": [2, 1, 1, 2],
                         "player_id": ["a", "a", "b", "a"]})
    # Only the exact flagged (season, team, player): not other teams, not ok
    # judgments, not next season.
    assert audit.flagged_mask(rows, back).tolist() == [True, False, False, False]


def test_import_sheet_maps_flags_and_reviewed_rows(tmp_path):
    sheet = tmp_path / "review_2025.csv"
    s = _shown().assign(flag=["y", "", ""], reason=["star_wont_move", "", ""], notes="",
                        reviewed=["y", "y", ""], rank=[1, 2, 1])
    s.to_csv(sheet, index=False)
    j = audit.import_sheet(sheet, audit.load_judgments(tmp_path / "none.csv")).set_index(["TeamId", "player_id"])
    # Flagged -> flag, reviewed -> ok, untouched rows are not judged.
    assert j.loc[(1, "a"), "verdict"] == "flag" and j.loc[(1, "b"), "verdict"] == "ok"
    assert (2, "a") not in j.index
    # Importing again replaces, not duplicates.
    again = audit.import_sheet(sheet, j.reset_index())
    assert len(again) == 2
