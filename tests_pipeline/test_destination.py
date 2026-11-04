import pandas as pd

import destination as dn
import fetch_wikipedia_fa as fw

PAGE = """==Front office movements==
===Head coach changes===
;Off-season
{| class="wikitable"
|-
! Date || Team || Outgoing head coach || Reason || Incoming head coach || Last coaching position
|-
| June 1 || [[Orlando Magic]] || [[Stan Van Gundy]] || Fired || [[Jacque Vaughn]] || [[San Antonio Spurs]] assistant coach
|}
;Season
{| class="wikitable"
|-
! Date || Team || Outgoing head coach || Reason || Incoming head coach || Last coaching position
|-
| January 5 || [[Utah Jazz]] || [[Jerry Sloan]] || Resigned || [[Tyrone Corbin]] || Jazz assistant
|}
===General manager changes===
{| class="wikitable"
|-
!colspan="6"|Off-season
|-
! Date || Team || Former GM || New GM || Previous position || Ref.
|-
| May 11 || [[Golden State Warriors]] || [[Chris Mullin]] || [[Larry Riley]] || Assistant coach of the Warriors ||
|}
==Player movement==
"""


def test_front_office_tables_parse_period_team_and_previous_team(monkeypatch):
    monkeypatch.setattr(fw, "load_wikitext", lambda season: PAGE)
    by_name = pd.Series({"Orlando Magic": 1, "San Antonio Spurs": 2, "Utah Jazz": 3,
                         "Golden State Warriors": 4})
    fo = dn.front_office_changes([2013], by_name).set_index("TeamId")

    assert fo.loc[1, "kind"] == "coach" and fo.loc[1, "period"] == "off"
    assert fo.loc[1, "prev_TeamId"] == 2                 # came from the Spurs
    assert fo.loc[3, "period"] == "in" and pd.isna(fo.loc[3, "prev_TeamId"])
    assert fo.loc[4, "kind"] == "gm" and fo.loc[4, "period"] == "off"


def test_trade_counts_one_per_trade_not_per_player():
    by_code = pd.Series({(2014, "BOS"): 10, (2014, "BRK"): 20, (2014, "MIA"): 30})
    tx = pd.DataFrame({
        "page_season": [2015] * 4, "date": ["2014-07-12"] * 3 + ["2014-08-01"],
        "kind": "trade", "team_from": ["BOS", "BOS", "BRK", "MIA"],
        "team_to": ["BRK", "BRK", "BOS", "BOS"],
    })
    tr = dn.trade_counts(tx, by_code)
    # Three players in one BOS-BRK deal count once; the MIA deal is separate.
    assert len(tr) == 2
    assert set(zip(tr["lo"], tr["hi"])) == {(10, 20), (10, 30)}
