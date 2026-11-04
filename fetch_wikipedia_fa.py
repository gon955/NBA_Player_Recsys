"""Offseason free-agent signings from Wikipedia's per-season NBA transaction lists.

Why: acquisition.py needs to know who could move. Contract status inferred
from salaries (contract_status.py) catches 88% of known signings but adds
nothing measurable, and BBRef's transaction log omits re-signings, so it can
only say how players who moved got there. Wikipedia's "List of YYYY-YY NBA
season transactions" pages have a free agency table that lists re-signings
too (former team = new team) and tags restricted free agents.

Fetching: the MediaWiki API, one page per request, raw wikitext (no HTML
parser needed), cached gzipped under data/wikipedia_raw/. This is a shared
public service that rate-limits (it answered 429 to a burst of ~20 queries
from here), so:
  - one request at a time, MIN_INTERVAL seconds between request starts
  - Retry-After honoured on 429/503; otherwise exponential backoff
  - CIRCUIT_BREAKER consecutive failures stop the whole run
  - pages already on disk are never fetched again
  - a User-Agent naming the project, as Wikimedia's policy asks

Parsing (consolidate): every table under the free-agency heading (sub-tables
for 10-day, two-way and undrafted signings skipped), with rowspan/colspan
expanded. A team cell spanning New and Former team is a re-signing. A player
cell tagged RFA is a restricted free agent; rows shaded #cfc on recent pages
are unsigned free agents whose rights were renounced. Players are matched to
BBRef ids by normalised name among that season's players (Advanced.csv),
teams to NBA TeamIds by full name (team_style.csv). Only offseason signings
are kept (June-October, plus November-December after the 2011 lockout and in
2020's COVID-delayed offseason).

Output data/wikipedia_fa.csv, one row per (season t, player): he was a free
agent in the offseason after t. moved = signed with a different NBA team.

Usage:
    python fetch_wikipedia_fa.py                    # 2004-05 .. 2025-26
    python fetch_wikipedia_fa.py --seasons 2011 2012
    python fetch_wikipedia_fa.py --consolidate-only
"""

import argparse
import gzip
import os
import re
import sys
import time

import pandas as pd
import requests

API = "https://en.wikipedia.org/w/api.php"
RAW_DIR = os.path.join("data", "wikipedia_raw")
USER_AGENT = "NBA_Player_Recsys/1.0 (https://github.com/gon955/NBA_Player_Recsys; research)"
MIN_INTERVAL = 6.0
MAX_RETRIES = 3
CIRCUIT_BREAKER = 3
SEASONS = range(2005, 2027)     # page season (2005 = 2004-05); its offseason follows season - 1

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
SKIP_TABLES = re.compile(r"10-day|two-way|undrafted|exhibit|regular season|hardship", re.I)
REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S)


def title(season):
    return f"List of {season - 1}–{season % 100:02d} NBA season transactions"


def raw_path(season):
    return os.path.join(RAW_DIR, f"{season}.wikitext.gz")


class CircuitOpen(RuntimeError):
    pass


class Client:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})
        self.last = 0.0
        self.failures = 0

    def get(self, params):
        for attempt in range(MAX_RETRIES + 1):
            wait = self.last + MIN_INTERVAL - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            err = None
            try:
                r = self.s.get(API, params=params, timeout=60)
            except requests.RequestException as e:
                r, err = None, str(e)
            if r is not None and r.status_code == 200:
                self.failures = 0
                return r.json()
            self.failures += 1
            if self.failures >= CIRCUIT_BREAKER:
                raise CircuitOpen(f"{self.failures} consecutive failures; stopping")
            retry_after = r.headers.get("Retry-After") if r is not None else None
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 30 * 2 ** attempt
            print(f"  {r.status_code if r is not None else err}; waiting {delay:.0f}s", flush=True)
            time.sleep(delay)
        raise RuntimeError(f"gave up on {params.get('page')}")


def fetch(seasons):
    os.makedirs(RAW_DIR, exist_ok=True)
    client = Client()
    for season in seasons:
        path = raw_path(season)
        if os.path.exists(path):
            continue
        d = client.get({"action": "parse", "page": title(season), "prop": "wikitext",
                        "format": "json", "formatversion": "2", "redirects": "1"})
        text = d["parse"]["wikitext"]
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
        print(f"{season}: {len(text):,} chars", flush=True)


def load_wikitext(season):
    with gzip.open(raw_path(season), "rt", encoding="utf-8") as f:
        return f.read()


def _split_top(text, sep):
    """Split on `sep` outside [[ ]] and {{ }}."""
    out, depth, i, start = [], 0, 0, 0
    while i < len(text):
        two = text[i:i + 2]
        if two in ("[[", "{{"):
            depth += 1
            i += 2
            continue
        if two in ("]]", "}}"):
            depth = max(0, depth - 1)
            i += 2
            continue
        if depth == 0 and text.startswith(sep, i):
            out.append(text[start:i])
            i += len(sep)
            start = i
            continue
        i += 1
    out.append(text[start:])
    return out


def _cell(raw):
    """(attrs, content) for one cell's text."""
    parts = _split_top(raw, "|")
    if len(parts) > 1 and re.search(r"(rowspan|colspan|style|align|class|bgcolor)\s*=", parts[0]):
        return parts[0], "|".join(parts[1:])
    return "", raw


def _span(attrs, name):
    m = re.search(name + r'\s*=\s*"?(\d+)', attrs)
    return int(m.group(1)) if m else 1


def parse_tables(section):
    """Each wikitable in `section` as (offset, rows); a row is a list of
    (content, is_header, attrs) with rowspan / colspan expanded."""
    tables = []
    for m in re.finditer(r"\{\|(.*?)^\|\}", section, re.M | re.S):
        rows, carry = [], {}                   # carry: col -> [remaining, content, header, attrs]
        # Chunk 0 holds the table's attributes and, on some pages, the header row.
        for chunk in re.split(r"^\|-", m.group(1), flags=re.M):
            lines = chunk.split("\n")
            row_attrs = lines[0]
            cells = []
            for line in lines[1:]:
                line = line.strip()
                if not line or line.startswith("|}") or line.startswith("|+"):
                    continue
                if line[0] in "|!":
                    head = line[0] == "!"
                    # Header lines separate cells with "!!" or, on many pages, "||".
                    parts = [x for h in _split_top(line[1:], "!!") for x in _split_top(h, "||")] \
                        if head else _split_top(line[1:], "||")
                    for c in parts:
                        cells.append((c, head))
                elif cells:                      # continuation of the previous cell
                    cells[-1] = (cells[-1][0] + "\n" + line, cells[-1][1])
            row, col, queue = [], 0, list(cells)
            while queue or col in carry:
                if col in carry:
                    left, content, head, attrs = carry[col]
                    row.append((content, head, attrs))
                    if left <= 1:
                        del carry[col]
                    else:
                        carry[col][0] -= 1
                    col += 1
                    continue
                raw, head = queue.pop(0)
                attrs, content = _cell(raw)
                rs, cs = _span(attrs, "rowspan"), _span(attrs, "colspan")
                full = row_attrs + " " + attrs
                for k in range(cs):
                    row.append((content, head, full))
                    if rs > 1:
                        carry[col + k] = [rs - 1, content, head, attrs]
                col += cs
            if row:
                rows.append(row)
        tables.append((m.start(), rows))
    return tables


def _text(c):
    """(display name, article name) of a player cell. {{sortname|First|Last|Article}}
    shows First Last but links Article, which carries the common name BBRef uses
    ("Louis Williams" -> "Lou Williams", "Patrick Mills" -> "Patty Mills")."""
    c = REF.sub("", c)
    m = re.search(r"\{\{sortname\|([^|}]*)\|([^|}]*)(\|[^}]*)?\}\}", c)
    if m:
        shown = f"{m.group(1).strip()} {m.group(2).strip()}"
        extra = [x.strip() for x in (m.group(3) or "").split("|") if x.strip()]
        article = next((x for x in extra if "=" not in x), shown)
        return shown, re.sub(r"\s*\(.*?\)\s*$", "", article)
    m = re.search(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", c)
    if m:
        return (m.group(2) or m.group(1)).strip(), re.sub(r"\s*\(.*?\)\s*$", "", m.group(1).strip())
    plain = re.sub(r"\{\{[^}]*\}\}|<[^>]+>|'+", "", c).strip()
    return plain, plain


def _initials(name):
    """'J. R. Smith' -> 'J.R. Smith', so normalize_name gives BBRef's 'jr smith'."""
    return re.sub(r"\b([A-Z])\.\s+(?=[A-Z]\.)", r"\1.", name)


def _team_link(c):
    """Link target of the first wikilink (the article, e.g. 'Houston Rockets')."""
    m = re.search(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", REF.sub("", c))
    return m.group(1).strip() if m else ""


def _months(c):
    found = [mo for mo in MONTHS if mo in c]
    return found[0] if found else None


def free_agency_rows(season):
    """Raw (player, rfa, unsigned, month, new, former) rows for page `season`."""
    text = load_wikitext(season)
    heads = [(m.start(), len(m.group(1)), m.group(2).strip())
             for m in re.finditer(r"^(=+)\s*([^=\n]+?)\s*=+\s*$", text, re.M)]
    out = []
    for i, (pos, level, name) in enumerate(heads):
        if not re.search(r"free agen", name, re.I):
            continue
        end = next((p for p, lv, _ in heads[i + 1:] if lv <= level), len(text))
        sub = [(p - pos, n) for p, lv, n in heads[i + 1:] if p < end]
        for tpos, rows in parse_tables(text[pos:end]):
            under = [n for p, n in sub if p < tpos]
            if under and SKIP_TABLES.search(under[-1]):
                continue
            hdr = next((r for r in rows if any(re.search(r"\bPlayer\b", c) for c, _, _ in r)), None)
            if hdr is None:
                continue
            names = [re.sub(r"<[^>]+>|'+", "", c).strip().lower() for c, _, _ in hdr]
            try:
                ip = next(k for k, n in enumerate(names) if "player" in n)
                inew = next(k for k, n in enumerate(names) if "new team" in n)
                iold = next(k for k, n in enumerate(names) if "former team" in n)
            except StopIteration:
                continue
            idate = next((k for k, n in enumerate(names) if "date" in n or n == "signed"), None)
            month, past_header = None, False
            for r in rows:
                if r is hdr:
                    past_header = True
                    continue
                if not past_header:
                    continue
                if all(h for _, h, _ in r) or len(r) <= max(ip, inew, iold):
                    month = _months(r[0][0]) or month      # "! colspan=4|July"
                    if any("regular season" in c.lower() for c, _, _ in r):
                        month = "REGULAR"
                    continue
                pc = r[ip][0]
                row_month = month
                if idate is not None and idate < len(r):
                    row_month = _months(r[idate][0]) or month
                shown, article = _text(pc)
                out.append({
                    "page_season": season, "player_name": shown, "article_name": article,
                    "rfa": bool(re.search(r"\bRFA\b", pc)),
                    "unsigned": "#cfc" in r[ip][2].lower(),
                    "month": row_month,
                    "new_team": _team_link(r[inew][0]), "former_team": _team_link(r[iold][0]),
                    "same_cell": r[inew][0] == r[iold][0] and r[inew][2] == r[iold][2],
                })
    return pd.DataFrame(out)


def consolidate(seasons, adv_path, teams_path, out_path):
    from lineup_features import normalize_name
    rows = pd.concat([free_agency_rows(s) for s in seasons if os.path.exists(raw_path(s))],
                     ignore_index=True)
    rows["season"] = rows["page_season"] - 1
    # Late offseasons: the 2011 lockout (page 2012) and COVID (page 2021) ran to December.
    late = rows["page_season"].isin([2012, 2021])
    offseason = rows["month"].isin(["June", "July", "August", "September", "October"]) \
        | rows["month"].isna() | (late & rows["month"].isin(["November", "December"]))
    rows["offseason"] = offseason

    adv = pd.read_csv(adv_path, usecols=["player_id", "player", "season"]).drop_duplicates()
    adv["norm"] = adv["player"].map(normalize_name)
    by = adv.groupby(["season", "norm"])["player_id"].unique()
    norms = [{normalize_name(_initials(a)), normalize_name(_initials(b))} - {""}
             for a, b in zip(rows["player_name"], rows["article_name"])]
    ids = []
    for t, cands in zip(rows["season"], norms):
        hit = None
        for s in (t, t + 1, t - 1):      # played in t; else a returnee who plays in t+1
            found = {p for n in cands for p in by.get((s, n), [])}
            if len(found) == 1:
                hit = found.pop()
                break
            if len(found) > 1:
                break                    # two players share the name: leave unmatched
        ids.append(hit)
    rows["player_id"] = ids

    teams = pd.read_csv(teams_path, usecols=["season", "TeamId", "user_id"])
    teams["name"] = teams["user_id"].str.rsplit("_", n=1).str[0]
    by_name = teams.drop_duplicates("name").set_index("name")["TeamId"]
    rows["new_TeamId"] = rows["new_team"].map(by_name)
    rows["former_TeamId"] = rows["former_team"].map(by_name)
    rows["resigned"] = rows["same_cell"] | (rows["new_TeamId"].notna()
                                            & (rows["new_TeamId"] == rows["former_TeamId"]))
    rows["moved"] = rows["new_TeamId"].notna() & ~rows["resigned"] & ~rows["unsigned"]

    per = rows.groupby("page_season").agg(
        rows=("player_name", "size"), offseason=("offseason", "mean"),
        matched=("player_id", lambda x: x.notna().mean()),
        nba_team=("new_TeamId", lambda x: x.notna().mean()),
        resigned=("resigned", "mean"), rfa=("rfa", "sum"), unsigned=("unsigned", "sum"))
    print(per.round(3).to_string())
    out = rows[rows["offseason"]].dropna(subset=["player_id"]) \
        .sort_values(["season", "player_id", "moved"]).drop_duplicates(["season", "player_id"], keep="last")
    cols = ["season", "player_id", "player_name", "rfa", "unsigned", "resigned", "moved",
            "new_TeamId", "former_TeamId", "new_team", "former_team", "month"]
    out[cols].to_csv(out_path, index=False)
    print(f"Wrote {len(out)} free-agent player-seasons to '{out_path}'")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", default=list(SEASONS))
    ap.add_argument("--consolidate-only", action="store_true")
    ap.add_argument("--adv", default=os.path.join("data", "Advanced.csv"))
    ap.add_argument("--teams", default=os.path.join("data", "team_style.csv"))
    ap.add_argument("--out", default=os.path.join("data", "wikipedia_fa.csv"))
    args = ap.parse_args()
    if not args.consolidate_only:
        try:
            fetch(args.seasons)
        except CircuitOpen as e:
            print(f"STOPPED: {e}", file=sys.stderr)
            return 1
    consolidate(args.seasons, args.adv, args.teams, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
