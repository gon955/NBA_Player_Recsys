"""Destination-side features: why a player would land on THIS team.

acquisition.py's two-stage model showed that, given a player moves, the
features it had barely predict where (the real destination ranks 14.2 of 29
teams; random 14.9). Those features describe a team's quality, cap and
roster; none describe its relationships or its churn. This adds three kinds,
all known by the end of season t:

  trade partners   (transactions.csv) distinct trades between the candidate's
                   team and T over the last 3 seasons and all history, and
                   each side's trades per season -- some front offices deal
                   with each other, and some deal a lot
  front office     (Wikipedia's per-season coach / GM change tables, cached by
                   fetch_wikipedia_fa.py) T changed GM / head coach during
                   season t or in the offseason after it; and fo_tie: T's new
                   GM or coach last worked for the candidate's team. Offseason
                   hires mostly land April-June, before free agency opens.
                   Pages 2006-07 and 2007-08 have no such tables (NaN).
  roster churn     (contract_status.py) the share of T's floor time held by
                   players at the end of their deals (rookie scale's last year,
                   minimum and partial-season deals, a CBA's maximum length),
                   and how many players T used

Season keys: a decision in season t is made in the offseason after it.
transactions.csv's page_season P runs July P-1 to June P, so trades on pages
<= t are known; Wikipedia page t+1's "off-season" rows are the summer after t.
"""

import re

import numpy as np
import pandas as pd

import fetch_wikipedia_fa as fw

DESTINATION_FEATURES = ["partner_trades_3y", "partner_trades_all", "team_trade_rate",
                        "from_trade_rate", "new_gm", "new_coach", "fo_tie",
                        "team_expiring", "team_roster_n"]
TRADE_WINDOW = 3
FO_SECTION = re.compile(r"(head )?coach(ing)? changes?|general manager changes?|coaching changes", re.I)
SEASON_MARK = re.compile(r"(off-?season|in-?season|^\W*season\W*$)", re.I)


def team_ids(teams):
    """Wikipedia article / full team name -> TeamId, and (season, BBRef code) -> TeamId."""
    from lineup_features import bbref_team
    t = teams.copy()
    t["name"] = t["user_id"].str.rsplit("_", n=1).str[0]
    by_name = t.drop_duplicates("name").set_index("name")["TeamId"]
    t["code"] = [bbref_team(a, s) for a, s in zip(t["TeamAbbreviation"], t["season"])]
    by_code = t.set_index(["season", "code"])["TeamId"]
    return by_name, by_code


def front_office_changes(seasons, by_name):
    """One row per coach / GM change: page_season, kind, period, TeamId, prev_TeamId."""
    rows = []
    for page in seasons:
        try:
            text = fw.load_wikitext(page)
        except FileNotFoundError:
            continue
        heads = [(m.start(), len(m.group(1)), m.group(2).strip())
                 for m in re.finditer(r"^(=+)\s*([^=\n]+?)\s*=+\s*$", text, re.M)]
        for i, (pos, level, name) in enumerate(heads):
            if not FO_SECTION.search(name):
                continue
            kind = "gm" if "manager" in name.lower() else "coach"
            end = next((p for p, lv, _ in heads[i + 1:] if lv <= level), len(text))
            section = text[pos:end]
            marks = [(m.start(), m.group(0)) for m in re.finditer(r"^;\s*([^\n]+)$", section, re.M)]
            for tpos, table in fw.parse_tables(section):
                before = [t for p, t in marks if p < tpos]
                period = "off" if before and "off" in before[-1].lower() else "in" if before else None
                hdr = next((r for r in table if sum("Team" in c for c, _, _ in r) >= 1
                            and all(h for _, h, _ in r)), None)
                if hdr is None:
                    continue
                names = [re.sub(r"<[^>]+>|'+", "", c).strip().lower() for c, _, _ in hdr]
                try:
                    i_team = next(k for k, n in enumerate(names) if n == "team")
                except StopIteration:
                    continue
                i_prev = next((k for k, n in enumerate(names)
                               if "last" in n or "previous" in n), None)
                past = False
                for r in table:
                    if r is hdr:
                        past = True
                        continue
                    if all(h for _, h, _ in r):          # "! colspan=6|In-season"
                        txt = " ".join(c for c, _, _ in r).lower()
                        if "off" in txt and "season" in txt:
                            period = "off"
                        elif "season" in txt:
                            period = "in"
                        continue
                    if not past or len(r) <= i_team:
                        continue
                    team = by_name.get(fw._team_link(r[i_team][0]))
                    if team is None:
                        continue
                    prev = None
                    if i_prev is not None and i_prev < len(r):
                        for target in re.findall(r"\[\[([^\]|]+)", fw.REF.sub("", r[i_prev][0])):
                            if target.strip() in by_name.index:
                                prev = by_name[target.strip()]
                                break
                    rows.append({"page_season": page, "kind": kind, "period": period or "in",
                                 "TeamId": team, "prev_TeamId": prev})
    return pd.DataFrame(rows, columns=["page_season", "kind", "period", "TeamId", "prev_TeamId"])


def trade_counts(transactions, by_code):
    """Distinct trades per (page_season, team A, team B), TeamIds, A < B."""
    tx = transactions[transactions["kind"] == "trade"].dropna(subset=["team_from", "team_to"])
    a = [by_code.get((s, c)) for s, c in zip(tx["page_season"] - 1, tx["team_from"])]
    b = [by_code.get((s, c)) for s, c in zip(tx["page_season"] - 1, tx["team_to"])]
    tx = tx.assign(a=a, b=b).dropna(subset=["a", "b"])
    tx["lo"] = tx[["a", "b"]].min(axis=1).astype(np.int64)
    tx["hi"] = tx[["a", "b"]].max(axis=1).astype(np.int64)
    return tx.drop_duplicates(["date", "lo", "hi"])[["page_season", "lo", "hi"]]


def add_destination(pairs, member, teams, transactions, status):
    """Add DESTINATION_FEATURES to acquisition.build_pairs' rows."""
    by_name, by_code = team_ids(teams)
    out = pairs.copy()
    # Candidate's main team in season t.
    main = member.sort_values("share").drop_duplicates(["player_id", "season"], keep="last") \
        .set_index(["player_id", "season"])["TeamId"]
    out["from_team"] = main.reindex(pd.MultiIndex.from_frame(out[["player_id", "season"]])).to_numpy()

    # Trade partners.
    tr = trade_counts(transactions, by_code)
    seasons = sorted(out["season"].unique())
    pair_3y, pair_all, rate = {}, {}, {}
    for t in seasons:
        known = tr[tr["page_season"] <= t]
        recent = known[known["page_season"] > t - TRADE_WINDOW]
        for (lo, hi), n in recent.groupby(["lo", "hi"]).size().items():
            pair_3y[(t, lo, hi)] = n
        for (lo, hi), n in known.groupby(["lo", "hi"]).size().items():
            pair_all[(t, lo, hi)] = n
        per_team = pd.concat([recent["lo"], recent["hi"]]).value_counts() / TRADE_WINDOW
        for team, n in per_team.items():
            rate[(t, team)] = n
    ft = out["from_team"].fillna(-1).astype(np.int64).to_numpy()
    tt = out["TeamId"].astype(np.int64).to_numpy()
    lo, hi = np.minimum(ft, tt), np.maximum(ft, tt)
    s = out["season"].to_numpy()
    out["partner_trades_3y"] = [pair_3y.get((a, b, c), 0) for a, b, c in zip(s, lo, hi)]
    out["partner_trades_all"] = [pair_all.get((a, b, c), 0) for a, b, c in zip(s, lo, hi)]
    out["team_trade_rate"] = [rate.get((a, b), 0.0) for a, b in zip(s, tt)]
    out["from_trade_rate"] = [rate.get((a, b), 0.0) for a, b in zip(s, ft)]
    first_trade_page = tr["page_season"].min()
    early = out["season"] < first_trade_page
    out.loc[early, ["partner_trades_3y", "partner_trades_all", "team_trade_rate",
                    "from_trade_rate"]] = np.nan

    # Front office: changes during season t (page t) or in the summer after it
    # (page t+1, off-season rows).
    fo = front_office_changes(range(min(seasons), max(seasons) + 2), by_name)
    covered = set(fo["page_season"])
    window = pd.concat([fo.assign(season=fo["page_season"]),
                        fo[fo["period"] == "off"].assign(season=fo["page_season"] - 1)])
    gm = set(zip(window.loc[window["kind"] == "gm", "season"], window.loc[window["kind"] == "gm", "TeamId"]))
    coach = set(zip(window.loc[window["kind"] == "coach", "season"],
                    window.loc[window["kind"] == "coach", "TeamId"]))
    ties = set(zip(window["season"], window["TeamId"], window["prev_TeamId"].dropna()))
    has_fo = np.array([(t in covered) and (t + 1 in covered) for t in s])
    out["new_gm"] = np.where(has_fo, [(a, b) in gm for a, b in zip(s, tt)], np.nan)
    out["new_coach"] = np.where(has_fo, [(a, b) in coach for a, b in zip(s, tt)], np.nan)
    out["fo_tie"] = np.where(has_fo, [(a, b, c) in ties for a, b, c in zip(s, tt, ft)], np.nan)

    # Roster churn.
    st = status.set_index(["player_id", "season"])
    m = member.join(st[["rookie_final", "on_minimum", "partial", "at_max_len"]],
                    on=["player_id", "season"])
    flags = m[["rookie_final", "on_minimum", "partial", "at_max_len"]].fillna(False).astype(bool)
    m["expiring"] = flags.any(axis=1).astype(float)
    team = m[m["share"] > 0].groupby(["season", "TeamId"]).apply(
        lambda g: pd.Series({"team_expiring": np.average(g["expiring"], weights=g["share"]),
                             "team_roster_n": float(len(g))}))
    out = out.join(team, on=["season", "TeamId"])
    return out.drop(columns="from_team")
