"""Where each player is in his contract at the end of season t, from salaries up to t.

acquisition.py needs to know who can move. impact_recs.pending_fa answers that
by looking at the player's t+1 salary, which leaks the answer. This reads only
seasons <= t: a contract's salaries follow a fixed raise schedule, so a break
in a player's salary chain marks a new contract, and counting seasons since
the last break gives his contract year.

The schedule changed with each CBA (raises may rise or fall after year 1 by up
to X% of the first-year salary; Bird / non-Bird):

  signed under      seasons start   max raise    max length
  1995 CBA          <= 1998         ~20%         7+
  1999 CBA          1999-2005       12.5 / 10%   7 / 6
  2005 CBA          2006-2011       10.5 / 8%    6 / 5
  2011 CBA          2012-2017       7.5 / 4.5%   5 / 4
  2017 / 2023 CBA   2018-           8 / 5%       5 / 4

salaries.csv agrees: median within-contract raises run 9.1% (1999-04), 7.7%,
6.5%, 5.0% (2017-), and Garnett's 1995-CBA deal steps +20% of its first year
every season through 2004. A flat 0.97-1.13 band, the old rule, splits those
legacy deals every year, hides new contracts with small raises after 2011, and
splits the declining contracts that became common (2.7% of transitions in
1999-04, 6.3% since 2017).

So season s continues the contract of s-1 when, in order:
  rookie scale   a first-round pick inside his scale (draft year + 1..L):
                 L = 3 for drafts through 1998 (1995 CBA), 4 from 1999
                 (salaries confirm: +74% at year 4 for 1996-98 picks, the
                 +26% 4th-year option for later ones). The option jump is not
                 a break; the season after year L always is.
  partial        either season below PARTIAL_SHARE of the cap (10-day,
                 rest-of-season, two-way: prorated or half-scale) -> break
  minimum        both seasons at the minimum for his years of service -> break
                 (minimum deals are 1-2 years and most are 1; a multi-year
                 minimum's raises follow the scale, not the contract rules)
  band           ratio within 1 +/- (max raise of the CBA the contract started
                 under + BAND_TOL), and the contract not past that CBA's max
                 length. A chain first seen in 1999 may be a 1995-CBA deal,
                 so it gets the 20% band through 2005.

What is written per (player_id, season):
  contract_year   seasons into the current contract, counting t (1 = new in t)
  censored        the chain starts at the data's first season (1999), so the
                  contract may be older than contract_year says
  max_len         the max length for the CBA it was signed under
  at_max_len      contract_year >= max_len: it cannot run another season
  on_rookie       inside the rookie scale; rookie_final: in its last year
                  (restricted free agent or extended after t)
  on_minimum      at (~) the minimum for his years of service
  partial         a partial-season / non-standard salary
  cba             0..3, the CBA in force for season t (1999, 2005, 2011, 2017+)

Known limits: incentives move a year's salary by up to 20% and cannot be
separated out; veteran extensions after 2017 (120%, then 140% of the last
salary) show as breaks without a free agency; player options end a contract
early without any sign in the salaries. Tune the bands against real labels
(transactions, Wikipedia free agency lists) before trusting them exactly.

Usage:
    python contract_status.py      # write data/contract_status.csv + checks
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from fetch_salaries import CAP

BAND_TOL = 0.01
PARTIAL_SHARE = 0.007     # of the cap; the rookie minimum has been 0.75-0.82%
MIN_TOL = 1.10            # salary <= this x the minimum's cap share -> on_minimum
LEGACY_RAISE = 0.20       # 1995-CBA contracts still running from 1999
LEGACY_UNTIL = 2005       # last season a chain first seen in 1999 gets the legacy band
# Minimum salary by years of service as a share of the cap (2024-25 scale /
# $140.588M cap; 2009-10's 0.79% / 2.26% at 0 / 10+ years has the same shape).
MIN_SHARE = [0.0082, 0.0132, 0.0148, 0.0154, 0.0159, 0.0173, 0.0186, 0.0199, 0.0213, 0.0214,
             0.0235]


def cba_of_start(season):
    """(max Bird raise, max length) for a contract whose first season is `season`."""
    if season <= 2005:
        return 0.125, 7
    if season <= 2011:
        return 0.105, 6
    if season <= 2017:
        return 0.075, 5
    return 0.08, 5


def cba_index(season):
    return 0 if season <= 2005 else 1 if season <= 2011 else 2 if season <= 2017 else 3


def rookie_years(draft_season):
    return 3 if draft_season <= 1998 else 4


def contract_status(salaries, debut, draft):
    """One row per (player_id, season) with a salary. `debut`: player_id ->
    first NBA season; `draft`: player_id -> draft year of first-round picks."""
    s = salaries[salaries["salary"] > 0][["player_id", "season", "salary"]] \
        .drop_duplicates(["player_id", "season"]).sort_values(["player_id", "season"])
    cap = s["season"].map(CAP)
    share = s["salary"] / cap
    yos = (s["season"] - s["player_id"].map(debut)).fillna(0).clip(0, 10).astype(int)
    s = s.assign(share=share,
                 partial=share < PARTIAL_SHARE,
                 on_minimum=(share <= MIN_TOL * np.take(MIN_SHARE, yos)) & (share >= PARTIAL_SHARE))
    dy = s["player_id"].map(draft)
    L = dy.map(lambda y: rookie_years(y) if pd.notna(y) else np.nan)
    ry = s["season"] - dy
    s["on_rookie"] = (ry >= 1) & (ry <= L)
    s["rookie_year"] = ry.where(s["on_rookie"])
    s["rookie_final"] = s["on_rookie"] & (ry == L)
    first_season = s["season"].min()

    out = []
    for _, g in s.groupby("player_id", sort=False):
        se, sa = g["season"].to_numpy(), g["salary"].to_numpy()
        part, mn = g["partial"].to_numpy(), g["on_minimum"].to_numpy()
        rk, rfin = g["rookie_year"].to_numpy(), g["rookie_final"].to_numpy()
        cy = np.ones(len(g), dtype=int)
        start = se.copy()
        for i in range(1, len(g)):
            same = se[i] == se[i - 1] + 1 and not (part[i] or part[i - 1])
            if same and not np.isnan(rk[i]) and rk[i] >= 2:
                pass                                      # inside the rookie scale
            elif same and rfin[i - 1]:
                same = False                              # scale over: new deal or QO
            elif same and mn[i] and mn[i - 1]:
                same = False
            elif same:
                raise_, max_len = cba_of_start(start[i - 1])
                if start[i - 1] == first_season and se[i] <= LEGACY_UNTIL:
                    raise_, max_len = LEGACY_RAISE, 7
                ratio = sa[i] / sa[i - 1]
                same = (1 - raise_ - BAND_TOL <= ratio <= 1 + raise_ + BAND_TOL) \
                    and cy[i - 1] < max_len
            if same:
                cy[i] = cy[i - 1] + 1
                start[i] = start[i - 1]
        out.append(pd.DataFrame({"player_id": g["player_id"].to_numpy(), "season": se,
                                 "contract_year": cy, "contract_start": start}))
    st = s.merge(pd.concat(out, ignore_index=True), on=["player_id", "season"])
    st["censored"] = st["contract_start"] == first_season
    st["max_len"] = [7 if (c == first_season) else cba_of_start(c)[1] for c in st["contract_start"]]
    st["at_max_len"] = st["contract_year"] >= st["max_len"]
    st["cba"] = st["season"].map(cba_index)
    cols = ["player_id", "season", "contract_year", "contract_start", "censored", "max_len",
            "at_max_len", "on_rookie", "rookie_year", "rookie_final", "on_minimum", "partial", "cba"]
    return st[cols].reset_index(drop=True)


def load_inputs(salaries_path, career_path, draft_path):
    salaries = pd.read_csv(salaries_path)
    debut = pd.read_csv(career_path, usecols=["player_id", "from"]) \
        .drop_duplicates("player_id").set_index("player_id")["from"]
    d = pd.read_csv(draft_path, usecols=["season", "lg", "round", "player_id"])
    draft = d[(d["lg"] == "NBA") & (d["round"] == 1)].drop_duplicates("player_id") \
        .set_index("player_id")["season"]
    return salaries, debut, draft


def check_against_signings(status, transactions_path):
    """Recall of contract breaks on known offseason signings: a player BBRef
    lists as signing in the offseason after t must start a contract in t+1."""
    tx = pd.read_csv(transactions_path)
    sig = tx[(tx["kind"] == "signing") & tx["offseason"].astype(bool)
             & ~tx["detail"].fillna("").str.contains("10-day|rest of", case=False)]
    sig = sig[["player_id", "page_season"]].drop_duplicates().rename(columns={"page_season": "season"})
    m = sig.merge(status, on=["player_id", "season"], how="inner")
    m = m[~m["partial"]]
    return (m["contract_year"] == 1).mean(), len(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salaries", default=os.path.join("data", "salaries.csv"))
    ap.add_argument("--career", default=os.path.join("data", "Player Career Info.csv"))
    ap.add_argument("--draft", default=os.path.join("data", "Draft Pick History.csv"))
    ap.add_argument("--transactions", default=os.path.join("data", "transactions.csv"))
    ap.add_argument("--out", default=os.path.join("data", "contract_status.csv"))
    args = ap.parse_args()
    status = contract_status(*load_inputs(args.salaries, args.career, args.draft))
    status.to_csv(args.out, index=False)
    print(f"Wrote {len(status)} player-seasons to '{args.out}'")
    rec, n = check_against_signings(status, args.transactions)
    print(f"Known offseason signings (BBRef) that start a new contract here: {rec:.1%} of {n}")
    print("\nContract year distribution by CBA (non-partial, non-rookie):")
    x = status[~status["partial"] & ~status["on_rookie"]]
    print(pd.crosstab(x["cba"], x["contract_year"].clip(upper=6), normalize="index").round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
