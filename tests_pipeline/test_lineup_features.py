import math

import pandas as pd
import pytest

import lineup_features as lf


@pytest.mark.parametrize("raw, expected", [
    ("Nikola Jokić", "nikola jokic"),
    ("A.J. Green", "aj green"),
    ("Jrue Holiday", "jrue holiday"),         # "jr" only stripped at the end
    ("Gary Payton II", "gary payton"),
    ("Robert Horry Jr. III", "robert horry"),
    ("Ömer Aşık", "omer asik"),               # ş has no NFKD decomposition
    (None, ""),
])
def test_normalize_name(raw, expected):
    assert lf.normalize_name(raw) == expected


@pytest.mark.parametrize("abbr, season, expected", [
    ("BKN", 2020, "BRK"),
    ("PHX", 2001, "PHO"),
    ("CHA", 2010, "CHA"),   # Bobcats keep CHA
    ("CHA", 2016, "CHO"),   # Hornets are CHO on BBRef
    ("LAL", 2016, "LAL"),
])
def test_bbref_team(abbr, season, expected):
    assert lf.bbref_team(abbr, season) == expected


def _write_crosswalk_inputs(tmp_path):
    pbp = pd.DataFrame([
        # Two Chris Johnsons in 2013, told apart only by team.
        (203187, "Chris Johnson", 2013, "MEM"),
        (202419, "Chris Johnson", 2013, "MIN"),
        # A name override can't target him: another Steven Smith exists.
        (120, "Steven Smith", 2001, "POR"),
        (200848, "Steven Smith", 2007, "PHI"),
        (1, "Nikola Jokić", 2023, "DEN"),
    ], columns=["EntityId", "Name", "season", "TeamAbbreviation"])
    adv = pd.DataFrame([
        ("johnsch04", "Chris Johnson", 2013, "MEM"),
        ("johnsch03", "Chris Johnson", 2013, "MIN"),
        ("smithst01", "Steve Smith", 2001, "POR"),
        ("smithst03", "Steven Smith", 2007, "PHI"),
        ("jokicni01", "Nikola Jokić", 2023, "DEN"),
    ], columns=["player_id", "player", "season", "team"])
    pbp.to_csv(tmp_path / "pbp.csv", index=False)
    adv.to_csv(tmp_path / "adv.csv", index=False)


def test_build_crosswalk_resolves_collisions_and_id_overrides(tmp_path):
    _write_crosswalk_inputs(tmp_path)
    out = lf.build_crosswalk(tmp_path / "pbp.csv", tmp_path / "adv.csv", tmp_path)

    got = dict(zip(out["nba_player_id"], out["bbref_id"]))
    assert got == {
        203187: "johnsch04",
        202419: "johnsch03",
        120: "smithst01",
        200848: "smithst03",
        1: "jokicni01",
    }
    assert not (tmp_path / "unmatched_report.csv").exists()


def _lineups():
    return pd.DataFrame({
        "season": [2023, 2023],
        "TeamId": [1610612743, 1610612743],
        "TeamAbbreviation": ["DEN", "DEN"],
        "EntityId": ["1-2-3-4-5", "1-2-3-4-99"],   # 99 is not in the crosswalk
        "Name": ["a", "b"],
        "OffPoss": [100, 10], "DefPoss": [0, 10],
        "Points": [120, 9], "OpponentPoints": [0, 11],
        "Minutes": [40, 5], "GamesPlayed": [10, 2], "PlusMinus": [120, -2],
    })


def _crosswalk():
    return pd.DataFrame({
        "nba_player_id": [1, 2, 3, 4, 5],
        "bbref_id": ["a01", "b01", "c01", "d01", "e01"],
        "player_name": list("abcde"),
        "season": [2023] * 5,
    })


def test_build_panel_drops_partial_lineups_and_rates():
    panel = lf.build_panel(_lineups(), _crosswalk())

    assert len(panel) == 1
    row = panel.iloc[0]
    assert [row[f"p{k}_id"] for k in range(1, 6)] == ["a01", "b01", "c01", "d01", "e01"]
    assert row["team_full"] == "Denver Nuggets"
    assert row["era"] == "2016-present"
    assert row["off_rtg"] == pytest.approx(120.0)
    # Never defended: NaN, not a 0 that reads as a perfect defense.
    assert math.isnan(row["def_rtg"]) and math.isnan(row["net_rtg"])


def test_build_panel_rejects_a_player_listed_twice():
    cw = _crosswalk()
    cw.loc[cw["nba_player_id"] == 5, "bbref_id"] = "a01"
    with pytest.raises(ValueError, match="twice"):
        lf.build_panel(_lineups(), cw)


def test_attach_covariates_marks_missing_players_unknown():
    panel = lf.build_panel(_lineups(), _crosswalk())
    players = pd.DataFrame({
        "player_id": ["a01", "b01"], "season": [2023, 2023],
        "cluster_label": ["MVP", "Floor Spacers"], "pos": ["C", "PG"], "bpm": [13.0, 2.0],
    })
    out = lf.attach_covariates(panel, players)

    row = out.iloc[0]
    assert row["p1_cluster"] == "MVP" and row["p1_pos"] == "C"
    assert row["p3_cluster"] == "unknown" and math.isnan(row["p3_bpm"])
    assert row["n_known"] == 2


def test_explode_gives_five_rows_per_lineup():
    panel = lf.build_panel(_lineups(), _crosswalk())
    long = lf.explode(panel)

    assert len(long) == 5
    assert sorted(long["player_id"]) == ["a01", "b01", "c01", "d01", "e01"]
    assert (long["OffPoss"] == 100).all()
