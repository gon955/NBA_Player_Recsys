"""NBA transactions per season from basketball-reference.com, one row per player moved.

Why: evaluate_recs.py found that no ranking predicts real acquisitions, and that
the salary / value screens rule out a fifth of them -- mostly, it seemed, free
agents, whom trade rules do not bind. This labels each move as a trade, a
signing or a waiver claim so that can be checked, and so the contract-status
proxy in impact_recs.py can be validated against known signings.

What it cannot be: an availability input. BBRef lists free agents who sign
with a *new* team but omits re-signings (Duncan, Kidd and Jermaine O'Neal all
re-signed in July 2003; none appear). A "free agent" flag built from it would
contain mostly players we already know moved, which leaks the answer into the
realism test. Pro Sports Transactions, RealGM and Spotrac, which do record
re-signings, refuse scripted requests.

Page NBA_{season}_transactions covers July 1 of season-1 to June 30 of
season, so the offseason after season t is on page t+1.

Each <p> on the page is one transaction. Parsing:
  trade    "[In an N-team trade,] the A traded P.. to the B[ for Q..]" clauses,
           split on ';'. P go A -> B, Q go B -> A. Players linked inside
           parentheses are draft-pick footnotes ("X was later selected") and
           are dropped.
  signing  "The A signed P ..." -> P to A. `detail` keeps the contract
           wording (free agent / 10-day / rest of season / multi-year).
  claim    "The A claimed P on waivers from the B".
  waive / release / other kinds are kept with the team they leave.

Usage:
    python fetch_transactions.py                     # fetch + write CSV
    python fetch_transactions.py --consolidate-only
"""

import argparse
import gzip
import os
import re
import sys
import time

import pandas as pd
import requests

RAW_DIR = os.path.join("data", "transactions_raw")
OUT = os.path.join("data", "transactions.csv")
SEASONS = range(2000, 2026)       # offseasons after 1999 ... 2024
URL = "https://www.basketball-reference.com/leagues/NBA_{season}_transactions.html"
MIN_INTERVAL = 3.5
TIMEOUT = 30
USER_AGENT = "NBA_Player_Recsys/1.0 (research; contact via github)"

_LIST = re.compile(r"<ul class='page_index'>(.*?)</ul>", re.S)
_DAY = re.compile(r"<li><span>([^<]+)</span>(.*?)</li>", re.S)
_P = re.compile(r"<p>(.*?)</p>", re.S)
_PLAYER = re.compile(r'href="/players/./([a-z0-9]+)\.html"')
_FROM = re.compile(r'data-attr-from="([A-Z]+)"')
_TO = re.compile(r'data-attr-to="([A-Z]+)"')
_PAREN = re.compile(r"\([^()]*\)")


def raw_path(season):
    return os.path.join(RAW_DIR, f"{season}.html.gz")


def fetch(seasons):
    os.makedirs(RAW_DIR, exist_ok=True)
    todo = [s for s in seasons if not os.path.exists(raw_path(s))]
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for i, season in enumerate(todo):
        if i:
            time.sleep(MIN_INTERVAL)
        r = session.get(URL.format(season=season), timeout=TIMEOUT)
        if r.status_code == 429:
            raise SystemExit(f"429 at {season}; stopping. Re-run later to resume.")
        r.raise_for_status()
        with gzip.open(raw_path(season), "wt", encoding="utf-8") as f:
            f.write(r.text)
    print(f"{len(seasons) - len(todo)} pages cached, fetched {len(todo)}")


def _players(html):
    return _PLAYER.findall(_PAREN.sub("", html))


def parse_transaction(p):
    """[(kind, player_id, team_from, team_to, detail)] for one <p>."""
    text = re.sub(r"<[^>]+>", "", p)
    if " traded " in text:
        out = []
        for clause in re.split(r";(?: and)?", p):
            frm, to = _FROM.findall(clause), _TO.findall(clause)
            if not frm or not to:
                continue
            give, _, get = clause.partition(" for ")
            out += [("trade", pid, frm[0], to[0], "") for pid in _players(give)]
            out += [("trade", pid, to[0], frm[0], "") for pid in _players(get)]
        return out
    to, frm = _TO.findall(p), _FROM.findall(p)
    if " signed " in text:
        detail = ("10-day" if "10-day" in text else "rest of season" if "rest of the season" in text
                  else "free agent" if "free agent" in text else "extension" if "extension" in text
                  else "other")
        return [("signing", pid, None, to[0] if to else None, detail) for pid in _players(p)]
    if " claimed " in text:
        return [("claim", pid, frm[0] if frm else None, to[0] if to else None, "")
                for pid in _players(p)]
    kind = next((k for k in ("waived", "released", "drafted", "retired") if f" {k} " in f" {text} "),
                "other")
    return [(kind, pid, frm[0] if frm else None, to[0] if to else None, "") for pid in _players(p)]


def parse_page(html, season):
    m = _LIST.search(html)
    if not m:
        raise SystemExit(f"no transaction list on the {season} page")
    rows = []
    for day, body in _DAY.findall(m.group(1)):
        # A few entries know only the month ("October ?, 2018"); the month is
        # all the offseason flag uses.
        date = pd.to_datetime(day.strip().replace(" ?,", " 1,"))
        for p in _P.findall(body):
            for kind, pid, frm, to, detail in parse_transaction(p):
                rows.append((season, date, kind, pid, frm, to, detail))
    return rows


def consolidate(seasons):
    rows = []
    for season in seasons:
        with gzip.open(raw_path(season), "rt", encoding="utf-8") as f:
            rows += parse_page(f.read(), season)
    df = pd.DataFrame(rows, columns=["page_season", "date", "kind", "player_id",
                                     "team_from", "team_to", "detail"])
    # The offseason a move belongs to: before November, the decision was made
    # on season page_season - 1's information.
    df["offseason"] = df["date"].dt.month.isin([7, 8, 9, 10])
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consolidate-only", action="store_true")
    args = ap.parse_args()
    if not args.consolidate_only:
        fetch(list(SEASONS))
    df = consolidate(SEASONS)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} player moves to '{OUT}'")
    print(df.groupby(["kind", "detail"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
