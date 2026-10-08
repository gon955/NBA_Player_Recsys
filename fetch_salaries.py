"""Player salaries per season, keyed by Basketball-Reference id.

Why: impact_recs.py ranks every player in the league for every team, so its
lists converge on the same few stars ("the '25 Wizards should add SGA"). Salary
is what makes a candidate attainable or not: a team can only take in what it
can match in outgoing salary or absorb into cap room.

Sources, in priority order:

  bbref    Each team-season page on basketball-reference.com carries a salary
           table keyed by the same player ids this repo uses, so no name
           matching is needed. One page per team-season, ~800 in all.
           Sports-Reference allows 20 requests a minute; MIN_INTERVAL keeps
           this under that.
  github   gabriel1200/site_Data's nba_salaries_master.csv (1991-2025, also
           bbref-keyed), pinned to a commit. Fills player-seasons a team page
           does not list, mostly players who were waived or traded away before
           the page's roster was frozen. Its own gap is players whose names
           carry accents (Stojakovic, Turkoglu, Schroder are absent entirely),
           which is why it is only the fallback.

A player traded mid-season appears once per team at the same contract figure,
so `salary` is the max across his rows; `payroll` sums the team page alone, so
a traded player counts toward the team whose page lists him. team_salaries.csv
keeps the page rows themselves (season, team, player_id, salary): the roster a
team carries into the offseason, for impact_recs' free-agent cap room.

Raw pages are cached gzipped under data/salaries_raw/; re-running skips what is
on disk, so an interrupted pull resumes, and --consolidate-only never touches
the network.

Usage:
    python fetch_salaries.py                     # fetch + write CSVs
    python fetch_salaries.py --consolidate-only  # rebuild CSVs from cache
"""

import argparse
import gzip
import os
import re
import sys
import time

import pandas as pd
import requests

RAW_DIR = os.path.join("data", "salaries_raw")
OUT_PLAYERS = os.path.join("data", "salaries.csv")
OUT_TEAMS = os.path.join("data", "team_payroll.csv")
OUT_ROSTERS = os.path.join("data", "team_salaries.csv")
TEAM_SUMMARIES = os.path.join("data", "Team Summaries.csv")
SEASONS = range(1999, 2027)
# 2026 (2025-26) is fetched only so the last modelled season, 2025, can be
# flagged for contract status (impact_recs.pending_fa compares t with t+1).
# Team Summaries.csv stops at 2025; nobody relocated, so 2025's codes are used.
CONTRACT_ONLY = 2026

BBREF = "https://www.basketball-reference.com/teams/{abbr}/{season}.html"
GITHUB = ("https://raw.githubusercontent.com/gabriel1200/site_Data/"
          "9654466d1788b6a0714161d3a392fd2c260d96bd/random/nba_salaries_master.csv")
MIN_INTERVAL = 3.5
TIMEOUT = 30
USER_AGENT = "NBA_Player_Recsys/1.0 (research; contact via github)"

# Salary cap by season (season = the year it ends, as everywhere in this repo).
# 1999 is the lockout season's $30.0M.
CAP = {
    1999: 30_000_000, 2000: 34_000_000, 2001: 35_500_000, 2002: 42_500_000,
    2003: 40_271_000, 2004: 43_840_000, 2005: 43_870_000, 2006: 49_500_000,
    2007: 53_135_000, 2008: 55_630_000, 2009: 58_680_000, 2010: 57_700_000,
    2011: 58_044_000, 2012: 58_044_000, 2013: 58_044_000, 2014: 58_679_000,
    2015: 63_065_000, 2016: 70_000_000, 2017: 94_143_000, 2018: 99_093_000,
    2019: 101_869_000, 2020: 109_140_000, 2021: 109_140_000, 2022: 112_414_000,
    2023: 123_655_000, 2024: 136_021_000, 2025: 140_588_000, 2026: 154_647_000,
}

# Non-taxpayer mid-level exception: what a team over the cap can still offer a
# free agent. Introduced by the 1999 CBA; the lockout season had $1.75M.
MLE = {
    1999: 1_750_000, 2000: 2_000_000, 2001: 2_250_000, 2002: 4_538_000,
    2003: 4_546_000, 2004: 4_917_000, 2005: 4_903_000, 2006: 5_000_000,
    2007: 5_215_000, 2008: 5_356_000, 2009: 5_585_000, 2010: 5_854_000,
    2011: 5_765_000, 2012: 5_000_000, 2013: 5_000_000, 2014: 5_150_000,
    2015: 5_305_000, 2016: 5_464_000, 2017: 5_628_000, 2018: 8_406_000,
    2019: 8_641_000, 2020: 9_258_000, 2021: 9_258_000, 2022: 9_536_000,
    2023: 10_490_000, 2024: 12_405_000, 2025: 12_822_000, 2026: 14_104_000,
}

_TABLE = re.compile(r'id="salaries2".*?</table>', re.S)
_ROW = re.compile(r'data-append-csv="([^"]+)".*?data-stat="salary"[^>]*>([^<]*)<', re.S)


def team_seasons(path=TEAM_SUMMARIES):
    t = pd.read_csv(path, usecols=["season", "abbreviation"])
    t = t[t["season"].isin(SEASONS) & t["abbreviation"].notna()]
    if CONTRACT_ONLY not in set(t["season"]):
        t = pd.concat([t, t[t["season"] == CONTRACT_ONLY - 1].assign(season=CONTRACT_ONLY)])
    return list(t.itertuples(index=False, name=None))


def raw_path(season, abbr):
    return os.path.join(RAW_DIR, f"{season}_{abbr}.html.gz")


def fetch_pages(pairs):
    os.makedirs(RAW_DIR, exist_ok=True)
    todo = [(s, a) for s, a in pairs if not os.path.exists(raw_path(s, a))]
    print(f"{len(pairs) - len(todo)} pages cached, {len(todo)} to fetch")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    last = 0.0
    for i, (season, abbr) in enumerate(todo, 1):
        time.sleep(max(0.0, last + MIN_INTERVAL - time.monotonic()))
        last = time.monotonic()
        r = session.get(BBREF.format(abbr=abbr, season=season), timeout=TIMEOUT)
        if r.status_code == 429:
            # Sports-Reference answers rate-limit breaches with a long jail;
            # retrying would only extend it.
            raise SystemExit(f"429 at {season} {abbr}; stopping. Re-run later to resume.")
        r.raise_for_status()
        with gzip.open(raw_path(season, abbr), "wt", encoding="utf-8") as f:
            f.write(r.text)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}")


def parse_page(html):
    """[(player_id, salary)] from a team page's salary table; [] if it has none."""
    m = _TABLE.search(html)
    if not m:
        return []
    out = []
    for pid, sal in _ROW.findall(m.group(0)):
        sal = sal.strip().replace("$", "").replace(",", "")
        if sal:
            out.append((pid, int(sal)))
    return out


def load_bbref(pairs):
    rows = []
    for season, abbr in pairs:
        with gzip.open(raw_path(season, abbr), "rt", encoding="utf-8") as f:
            for pid, sal in parse_page(f.read()):
                rows.append((pid, season, abbr, sal))
    return pd.DataFrame(rows, columns=["player_id", "season", "team", "salary"])


def load_github():
    path = os.path.join(RAW_DIR, "github_nba_salaries_master.csv")
    if not os.path.exists(path):
        os.makedirs(RAW_DIR, exist_ok=True)
        r = requests.get(GITHUB, timeout=TIMEOUT)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    g = pd.read_csv(path, usecols=["bref_id", "year", "salary"])
    g = g.dropna().rename(columns={"bref_id": "player_id", "year": "season"})
    return g[g["season"].isin(SEASONS)].astype({"season": int, "salary": int})


def consolidate(pairs):
    bb = load_bbref(pairs)
    empty = sorted(set(pairs) - set(zip(bb["season"], bb["team"])))
    if empty:
        raise SystemExit(f"{len(empty)} team pages have no salary table, e.g. {empty[:5]}")

    payroll = bb.groupby(["season", "team"])["salary"].sum().rename("payroll").reset_index()
    payroll["cap"] = payroll["season"].map(CAP)

    gh = load_github()
    main = bb.groupby(["player_id", "season"])["salary"].max().reset_index().assign(source="bbref")
    fill = gh.groupby(["player_id", "season"])["salary"].max().reset_index().assign(source="github")
    fill = fill.merge(main[["player_id", "season"]], how="left", indicator=True)
    fill = fill[fill["_merge"] == "left_only"].drop(columns="_merge")
    players = pd.concat([main, fill], ignore_index=True)
    players["cap_pct"] = players["salary"] / players["season"].map(CAP)
    if players.duplicated(["player_id", "season"]).any():
        raise SystemExit("duplicate player-seasons after merge")
    return players.sort_values(["season", "player_id"]), payroll, bb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consolidate-only", action="store_true")
    args = ap.parse_args()

    pairs = team_seasons()
    if not args.consolidate_only:
        fetch_pages(pairs)
    players, payroll, rosters = consolidate(pairs)
    players.to_csv(OUT_PLAYERS, index=False)
    payroll.to_csv(OUT_TEAMS, index=False)
    rosters.to_csv(OUT_ROSTERS, index=False)
    print(f"Wrote {len(players)} player-seasons to '{OUT_PLAYERS}' "
          f"({(players['source'] == 'github').sum()} from the fallback) and "
          f"{len(payroll)} team payrolls to '{OUT_TEAMS}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
