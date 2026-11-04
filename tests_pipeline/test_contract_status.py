import pandas as pd

import contract_status as cs
from fetch_salaries import CAP


def _status(rows, debut=None, draft=None):
    sal = pd.DataFrame(rows, columns=["player_id", "season", "salary"])
    debut = pd.Series(debut or {p: 1995 for p in sal["player_id"]})
    return cs.contract_status(sal, debut, pd.Series(draft or {}, dtype=float)) \
        .set_index(["player_id", "season"])


def test_raise_band_follows_the_cba_the_contract_started_under():
    # 2006 contract (10.5% allowed): +10% continues. 2013 contract (7.5%): +10% is a new deal.
    st = _status([("a", 2006, 10e6), ("a", 2007, 11e6),
                  ("b", 2013, 10e6), ("b", 2014, 11e6)])
    assert st.loc[("a", 2007), "contract_year"] == 2
    assert st.loc[("b", 2014), "contract_year"] == 1


def test_declining_contracts_are_not_split():
    st = _status([("a", 2019, 10e6), ("a", 2020, 9.3e6), ("a", 2021, 8.6e6)])
    assert list(st["contract_year"]) == [1, 2, 3]


def test_legacy_1995_cba_raises_continue_until_2005():
    # Garnett's deal: +20% of year one every season from 1999.
    rows = [("kg", 1999 + i, 14e6 + 2.8e6 * i) for i in range(6)]
    st = _status(rows)
    assert list(st["contract_year"]) == [1, 2, 3, 4, 5, 6]
    assert st["censored"].all()


def test_rookie_scale_option_jump_is_not_a_break_and_ends_on_time():
    rows = [("r", 2010, 2.0e6), ("r", 2011, 2.1e6), ("r", 2012, 2.2e6), ("r", 2013, 2.8e6),
            ("r", 2014, 2.9e6)]
    st = _status(rows, debut={"r": 2010}, draft={"r": 2009})
    # +26% into year 4 continues; year 5 is a new contract even at +4%.
    assert list(st["contract_year"]) == [1, 2, 3, 4, 1]
    assert st.loc[("r", 2013), "rookie_final"] and not st.loc[("r", 2014), "on_rookie"]


def test_pre_1999_draftees_have_three_year_scales():
    st = _status([("o", 1999, 1e6), ("o", 2000, 1.05e6)], debut={"o": 1997}, draft={"o": 1996})
    assert st.loc[("o", 1999), "rookie_final"]
    assert st.loc[("o", 2000), "contract_year"] == 1


def test_partial_and_minimum_seasons_break_the_chain():
    tiny = 0.004 * CAP[2015]                       # a 10-day / prorated figure
    vet_min = 0.0235 * CAP[2015]
    st = _status([("p", 2015, tiny), ("p", 2016, tiny * 1.02),
                  ("m", 2015, vet_min), ("m", 2016, 0.0235 * CAP[2016])],
                 debut={"p": 2014, "m": 2000})
    assert st.loc[("p", 2016), "partial"] and st.loc[("p", 2016), "contract_year"] == 1
    assert st.loc[("m", 2016), "on_minimum"] and st.loc[("m", 2016), "contract_year"] == 1


def test_contract_cannot_outrun_its_max_length():
    rows = [("x", 2012 + i, 10e6 * (1 + 0.05 * i)) for i in range(7)]
    st = _status(rows + [("early", 1999, 1e6)])    # data starts 1999, as the real file does
    # 2011-CBA contracts run at most 5 seasons.
    assert list(st.loc["x", "contract_year"]) == [1, 2, 3, 4, 5, 1, 2]
    assert not st.loc[("x", 2012), "censored"]
    assert st.loc[("x", 2016), "at_max_len"]
