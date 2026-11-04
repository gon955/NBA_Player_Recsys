import fetch_wikipedia_fa as fw

PAGE = """==Player movement==
===Free agency===
{{sticky header}}{| class="wikitable sortable"
! Player
! Date signed
! New team
! Former team
! Ref
|-
| {{sortname|Luke|Kornet}}
|rowspan=2 align="center"| July 2
|rowspan=2 colspan=2 align="center"| [[Boston Celtics]]
| <ref>x</ref>
|-
| {{sortname|Patrick|Beverley}} <small>([[Restricted free agent (NBA)|RFA]])</small>
| <ref>y</ref>
|-
| {{sortname|Louis|Williams|Lou Williams}}
| July 9
| [[Atlanta Hawks]]
| [[Houston Rockets]]
| <ref name="z"/>
|-
| style="background:#cfc" | {{sortname|Ed|Davis}}
| July 10
| colspan=2 | [[Utah Jazz]]
|
|}
====10-day contracts====
{| class="wikitable"
! Player !! Date signed !! New team !! Former team
|-
| [[Someone Else]] || January 5 || [[Utah Jazz]] || [[Utah Jazz]]
|}
==Draft==
"""


def test_rowspan_colspan_and_tags(monkeypatch):
    monkeypatch.setattr(fw, "load_wikitext", lambda season: PAGE)
    rows = fw.free_agency_rows(2025).set_index("player_name")

    # Kornet and Beverley share a rowspan+colspan re-signing cell.
    assert list(rows.index) == ["Luke Kornet", "Patrick Beverley", "Louis Williams", "Ed Davis"]
    assert rows.loc["Patrick Beverley", "same_cell"] and rows.loc["Patrick Beverley", "rfa"]
    assert rows.loc["Patrick Beverley", "new_team"] == "Boston Celtics"
    assert rows.loc["Patrick Beverley", "month"] == "July"
    # A move: different new and former teams.
    lw = rows.loc["Louis Williams"]
    assert (lw["new_team"], lw["former_team"], lw["same_cell"]) == ("Atlanta Hawks", "Houston Rockets", False)
    # The article name carries BBRef's spelling.
    assert lw["article_name"] == "Lou Williams"
    # Shaded #cfc: renounced, unsigned.
    assert rows.loc["Ed Davis", "unsigned"] and not rows.loc["Luke Kornet", "unsigned"]
    # The 10-day sub-table is skipped.
    assert "Someone Else" not in rows.index


def test_initials_collapse_for_bbref_matching():
    from lineup_features import normalize_name
    assert normalize_name(fw._initials("J. R. Smith")) == normalize_name("J.R. Smith")
    assert fw._initials("Jaylen Brown") == "Jaylen Brown"
