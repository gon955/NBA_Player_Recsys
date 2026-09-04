"""Pull possession-level lineup and player totals from the pbpstats API.

Why not nba_api / stats.nba.com: that host accepts the TLS connection and then
never answers on any /stats/ path from here, with or without the full browser
header set, over IPv4 or IPv6 (Akamai bot filtering — curl exits 124 on a 40s
timeout while stats.nba.com/ itself returns a 301). api.pbpstats.com serves the
same possession-level aggregates, derived from the same play-by-play, and
answers in a few seconds.

What this is for: season aggregates give one row per player-season, so a player
and his context are perfectly collinear and no player x team interaction is
identifiable. Lineup rows give dozens of contexts per player per season — the
same player measured beside different teammates — which is what makes fit
estimable at all.

Two shapes are fetched:

  Lineup  per (season, team). The league-wide endpoint silently truncates to the
          top 500 rows by minutes, which is about a sixth of the real total
          (Denver alone fielded 326 distinct 5-man units in 2023-24), so the
          per-team query is the only way to get full coverage.
  Player  per (season, team), for the same reason. The league-wide call was
          tried first and is wrong: it returns exactly 500 rows every season
          while 529-602 distinct players actually appear in lineups, so 29-102
          players per season were silently absent. Those missing ids are
          unresolvable in the crosswalk, which capped it at 83.7% coverage even
          though name matching itself reaches 99.9%.

Raw responses are cached gzipped under data/pbpstats_raw/. Re-running skips what
is already on disk, so an interrupted pull resumes where it stopped and the
consolidation step can be re-run without touching the network.

Usage:
    python fetch_pbpstats.py                      # 1999-2025, lineups + players
    python fetch_pbpstats.py --seasons 2016-2025  # a subset
    python fetch_pbpstats.py --consolidate-only   # rebuild CSVs from cache
"""

import argparse
import gzip
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

API = "https://api.pbpstats.com/get-totals/nba"
RAW_DIR = os.path.join("data", "pbpstats_raw")
OUT_LINEUPS = os.path.join("data", "pbpstats_lineups.csv")
OUT_PLAYERS = os.path.join("data", "pbpstats_players.csv")

# The NBA's team ids are a contiguous block, and a franchise keeps its id across
# relocations and renames (Seattle -> OKC, New Jersey -> Brooklyn, Vancouver ->
# Memphis, Charlotte Bobcats -> Hornets), so this range covers every team that
# existed in any season we ask for. Teams that did not exist yet return no rows.
TEAM_IDS = [str(1610612737 + i) for i in range(30)]

MAX_WORKERS = 5
MAX_RETRIES = 4
TIMEOUT = 90

# Minimum seconds between request *starts*, enforced globally across workers.
# Concurrency alone is not a rate limit: 5 workers finishing at once fire 5 new
# requests in the same millisecond. This spaces them.
MIN_INTERVAL = 0.35

# Stop the whole pull if this many requests fail back to back. A dead or angry
# API should cost us a handful of requests, not grind through all 837 and look
# to the far end like someone hammering an endpoint that keeps saying no.
CIRCUIT_BREAKER = 12

# pbpstats is one person's free API. Identify the client so it can be blocked
# or throttled deliberately rather than mistaken for an anonymous scraper.
USER_AGENT = "NBA_Player_Recsys/1.0 (research; contact via github)"

_print_lock = threading.Lock()
_rate_lock = threading.Lock()
_last_request = [0.0]
_consecutive_failures = [0]


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def _throttle():
    """Global spacing between request starts, shared by every worker."""
    with _rate_lock:
        wait = _last_request[0] + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()


class CircuitOpen(Exception):
    """Raised once the API has failed enough times to stop trying."""


def _make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    # One pooled connection per worker: fewer TLS handshakes for us, fewer
    # sockets for them.
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS
    )
    s.mount("https://", adapter)
    return s


_thread_local = threading.local()


def session():
    if not hasattr(_thread_local, "s"):
        _thread_local.s = _make_session()
    return _thread_local.s


def season_str(year):
    """Repo seasons are the ending year (2024); pbpstats wants '2023-24'."""
    return f"{year - 1}-{year % 100:02d}"


def cache_path(season, kind, team_id=None):
    name = f"{season}_{kind}" + (f"_{team_id}" if team_id else "") + ".json.gz"
    return os.path.join(RAW_DIR, name)


def fetch_one(season, kind, team_id=None):
    """Fetch and cache one response. Returns (path, 'cached'|'fetched'|'failed')."""
    path = cache_path(season, kind, team_id)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, "cached"

    params = {
        "Season": season_str(season),
        "SeasonType": "Regular Season",
        "Type": kind,
    }
    if team_id:
        params["TeamId"] = team_id

    if _consecutive_failures[0] >= CIRCUIT_BREAKER:
        raise CircuitOpen(f"{CIRCUIT_BREAKER} consecutive failures")

    for attempt in range(MAX_RETRIES):
        try:
            _throttle()
            r = session().get(API, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                payload = r.json()
                rows = payload.get("multi_row_table_data") or []
                # The league-wide endpoint truncates at 500 rows by minutes. A
                # per-team response landing on exactly 500 would mean the cap is
                # biting here too and we are silently losing lineups.
                if team_id and len(rows) == 500:
                    log(f"  WARNING {season} {kind} team {team_id}: exactly 500 rows, "
                        f"may be truncated")
                tmp = path + ".tmp"
                with gzip.open(tmp, "wt", encoding="utf-8") as f:
                    json.dump(payload, f)
                os.replace(tmp, path)
                _consecutive_failures[0] = 0
                return path, "fetched"

            # 429/5xx are worth another try; anything else is our bug, not theirs.
            if r.status_code not in (429, 500, 502, 503, 504):
                log(f"  {season} {kind} {team_id or ''}: HTTP {r.status_code}, giving up")
                _consecutive_failures[0] += 1
                return path, "failed"

            # Honour an explicit Retry-After over our own guess at a backoff.
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    log(f"  429/5xx with Retry-After={retry_after}s, obeying")
                    time.sleep(min(float(retry_after), 120))
                    continue
                except ValueError:
                    pass
        except (requests.RequestException, ValueError) as e:
            if attempt == MAX_RETRIES - 1:
                log(f"  {season} {kind} {team_id or ''}: {type(e).__name__}, giving up")
                _consecutive_failures[0] += 1
                return path, "failed"
        # Exponential backoff with jitter, so a stumble doesn't turn into a stampede.
        time.sleep((2 ** attempt) + random.random())

    # Retries exhausted on 429/5xx. This used to return silently, which meant a
    # 373-failure run printed nothing at all and only showed up in the summary
    # counter — the precise blindness the circuit breaker exists to prevent.
    log(f"  {season} {kind} {team_id or ''}: retries exhausted (429/5xx), giving up")
    _consecutive_failures[0] += 1
    return path, "failed"


def fetch_all(seasons, do_lineups, do_players):
    os.makedirs(RAW_DIR, exist_ok=True)
    jobs = []
    for yr in seasons:
        if do_players:
            jobs.extend((yr, "Player", tid) for tid in TEAM_IDS)
        if do_lineups:
            jobs.extend((yr, "Lineup", tid) for tid in TEAM_IDS)

    # Cached work costs nothing, so report how much is actually going over the
    # wire before starting.
    todo = [j for j in jobs if not os.path.exists(cache_path(j[0], j[1], j[2]))]
    log(f"  {len(jobs)} jobs, {len(jobs) - len(todo)} already cached, "
        f"{len(todo)} to fetch (~{len(todo) * MIN_INTERVAL / 60:.0f} min floor "
        f"at {MIN_INTERVAL}s spacing)")

    counts = {"cached": 0, "fetched": 0, "failed": 0}
    done = 0
    tripped = False
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, *j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                _, status = fut.result()
            except CircuitOpen:
                tripped = True
                continue
            counts[status] += 1
            done += 1
            if done % 50 == 0 or done == len(jobs):
                log(f"  {done}/{len(jobs)}  cached={counts['cached']} "
                    f"fetched={counts['fetched']} failed={counts['failed']}")
    if tripped:
        log(f"CIRCUIT BREAKER: stopped after {CIRCUIT_BREAKER} consecutive failures. "
            f"Cached work is kept — fix the cause and re-run to resume.")
    return counts


def load_rows(season, kind, team_id=None):
    path = cache_path(season, kind, team_id)
    if not os.path.exists(path):
        return []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        log(f"  unreadable cache: {path}")
        return []
    rows = payload.get("multi_row_table_data") or []
    for r in rows:
        r["season"] = season
    return rows


def consolidate(seasons, do_lineups, do_players):
    # Field sets drift across eras (218 columns in 2000-01, 231 in 2023-24), so
    # let pandas take the union and leave the gaps as NaN rather than trying to
    # force one schema onto twenty-five years.
    if do_lineups:
        rows = []
        for yr in seasons:
            for tid in TEAM_IDS:
                rows.extend(load_rows(yr, "Lineup", tid))
        if rows:
            df = pd.DataFrame(rows)
            # A lineup appears once per team per season; EntityId is the five
            # player ids joined by '-'. Guard against a repeated fetch anyway.
            df = df.drop_duplicates(subset=["season", "TeamId", "EntityId"])
            df.to_csv(OUT_LINEUPS, index=False)
            log(f"lineups: {len(df):,} rows x {df.shape[1]} cols -> {OUT_LINEUPS}")
        else:
            log("lineups: no cached rows")

    if do_players:
        rows = []
        for yr in seasons:
            # Union both sources rather than choosing between them. A partially
            # fetched season has per-team files for only some teams, and picking
            # those over the complete league-wide file throws away more than it
            # adds. Deduping on (season, TeamId, EntityId) makes the overlap
            # harmless, so this is monotone: coverage only ever improves.
            rows.extend(r for tid in TEAM_IDS for r in load_rows(yr, "Player", tid))
            rows.extend(load_rows(yr, "Player", None))
        if rows:
            df = pd.DataFrame(rows).drop_duplicates(subset=["season", "TeamId", "EntityId"])
            df.to_csv(OUT_PLAYERS, index=False)
            log(f"players: {len(df):,} rows x {df.shape[1]} cols -> {OUT_PLAYERS}")
        else:
            log("players: no cached rows")


def verify(seasons):
    """Cross-check coverage against the repo's own team-seasons and possessions.

    Two things can go wrong quietly, and row counts catch neither.

    A team can be missing entirely. An empty response is ambiguous on its own —
    Charlotte legitimately has no 2003 rows because the franchise didn't exist,
    while a dropped request also looks like zero rows — so team counts are
    compared against data/Player Play By Play.csv.

    A team can be *present but truncated*. The endpoint caps at 500 rows even
    per-team, so a busy roster comes back clipped. Row counts can't see this;
    possessions can. Summing OffPoss across a team's lineups must reconstruct
    its whole season, and master_team_stats.csv has the games and pace to say
    what that should be. Measured on Atlanta, the capped seasons still recover
    ~99% of possessions because the truncation falls on 0-1 minute lineups —
    but that is a fact to re-check per season, not to assume.
    """
    pbp = pd.read_csv(os.path.join("data", "Player Play By Play.csv"),
                      usecols=["season", "team"])
    pbp = pbp[~pbp["team"].isin(["2TM", "3TM", "4TM", "5TM"])]
    expected_teams = pbp.groupby("season")["team"].nunique().to_dict()

    ts = pd.read_csv("master_team_stats.csv", usecols=["season", "team", "g", "pace"])
    ts["expected_poss"] = ts["g"] * ts["pace"]
    expected_poss = ts.groupby("season")["expected_poss"].sum().to_dict()

    log("\nseason  teams  exp_teams  lineups   off_poss   expected   recovered  status")
    problems = 0
    for yr in seasons:
        exp_t = expected_teams.get(yr)
        exp_p = expected_poss.get(yr)
        got_teams = got_rows = 0
        got_poss = 0.0
        uncached = 0
        capped = []
        for tid in TEAM_IDS:
            if not os.path.exists(cache_path(yr, "Lineup", tid)):
                uncached += 1
                continue
            rows = load_rows(yr, "Lineup", tid)
            if not rows:
                continue
            got_teams += 1
            got_rows += len(rows)
            got_poss += sum(r.get("OffPoss") or 0 for r in rows)
            if len(rows) == 500:
                capped.append(tid)

        pct = (100 * got_poss / exp_p) if exp_p else float("nan")
        if exp_t is None:
            status = "no repo baseline"
        elif uncached and got_teams < exp_t:
            status = f"MISSING {uncached} uncached"
            problems += 1
        elif got_teams < exp_t:
            status = f"SHORT {exp_t - got_teams} teams"
            problems += 1
        elif pct == pct and pct < 97.0:
            status = f"LOW possessions ({len(capped)} teams at cap)"
            problems += 1
        else:
            status = "ok" + (f" ({len(capped)} at cap)" if capped else "")
        log(f"{yr:6d}  {got_teams:5d}  {str(exp_t):>9}  {got_rows:7d}  "
            f"{got_poss:9.0f}  {exp_p or 0:9.0f}  {pct:8.1f}%  {status}")

    log(f"\n{'all seasons complete' if not problems else f'{problems} season(s) need a re-run'}")
    return problems


def parse_seasons(spec):
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="1999-2025",
                    help="ending years, e.g. '2016-2025' or '2019,2023'")
    ap.add_argument("--only", choices=["lineups", "players"], default=None)
    ap.add_argument("--consolidate-only", action="store_true",
                    help="rebuild the CSVs from cache without any requests")
    ap.add_argument("--verify", action="store_true",
                    help="report cache coverage against the repo's team-seasons and exit")
    args = ap.parse_args()

    if args.verify:
        return 1 if verify(parse_seasons(args.seasons)) else 0

    seasons = parse_seasons(args.seasons)
    do_lineups = args.only in (None, "lineups")
    do_players = args.only in (None, "players")

    log(f"seasons {seasons[0]}-{seasons[-1]} ({len(seasons)}) "
        f"lineups={do_lineups} players={do_players}")

    if not args.consolidate_only:
        counts = fetch_all(seasons, do_lineups, do_players)
        log(f"fetch done: {counts}")
        if counts["failed"]:
            log("some requests failed; re-run to retry just those")

    consolidate(seasons, do_lineups, do_players)


if __name__ == "__main__":
    sys.exit(main())
