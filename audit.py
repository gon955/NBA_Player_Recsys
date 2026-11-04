"""Human review of recommendations, and the audit every veto rule must pass.

Real moves cannot say a recommendation was good, so judgment comes from a
person reviewing lists. To keep that judgment's bias out of the ranking it
only ever *removes* recommendations, through written rules, and every rule
is first audited against history: real moves cannot certify a
recommendation, but they can falsify a veto. If a rule would have blocked
moves teams actually made, it is wrong.

The loop:
  1 `python review_app.py`         review the served lists in a local page,
                                   blind to the model's score; every decision
                                   goes into JUDGMENTS (verdict ok / flag,
                                   REASONS below) and is kept for good, so a
                                   later pass shows only entries not yet judged.
                                   "Apply flags" rebuilds the lists in place.
  2 `python audit.py summarize`    what the flagged recommendations share
  3 write a Rule in RULES          a stated condition on the pair table
  4 `python audit.py rules`        audit it on the tuning seasons;
                                   `--holdout` once, logged (rubric.py's split)
  5 set active=True                export_recs.py then drops what it vetoes
(Spreadsheet instead: `audit.py sheet` exports a CSV, `audit.py import` reads
it back into JUDGMENTS.)

The audit, per rule:
  vetoed      pairs (team, candidate) it removes, of the season pool
  blocked     real veteran arrivals it would have prevented, listed by name,
              biggest t+1 contributions first -- read them for landmark moves
  rate ratio  arrival rate inside the vetoed pairs / outside. A rule that
              only removes impossible moves has a ratio near 0; one keyed on
              a mere tendency (stars go to bad teams 3x less often) does not
  verdict     PASS: ratio <= MAX_RATE_RATIO and blocked <= MAX_BLOCKED_SHARE
              of arrivals; INCONCLUSIVE: the vetoed pairs are so few that
              even a wrong rule would be expected to block under 3 moves
  served      how many of the app's current top-10 entries it removes

Flags are the other way a review acts: each flagged judgment removes exactly
that (season, team, player) entry from that season's served lists
(export_recs.py), and nothing else. They are for facts the data cannot see
(a franchise player, an extension just signed, an injury) and are not
audited -- there is no history to test a one-off against -- so they are kept
narrow: one season, only removals, each with its reasons and note.

`python audit.py screen` holds impact_recs' salary screen and value cap to
the same standard: which real arrivals it rules out, by how they happened.

Pair-table columns a rule can use (acquisition.build_pairs, plus context()):
  candidate   value, net_impact, bpm, star (top 5% of the pool by BPM), age,
              experience, cap_pct, on_rookie, rookie_final, contract_year,
              on_minimum, traded, pos_index
  team        team_net, team_net_rank (1 = worst), team_age (minutes-weighted),
              team_age_rank (1 = youngest), room_pct, payroll_pct, n_teams
"""

import argparse
import datetime
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import evaluate_recs as ev
import impact_recs as ir
import rubric

REASONS = {
    "star_wont_move": "this player would not be made available",
    "bad_contract": "a contract no one would take, or that no one would give up",
    "team_direction": "wrong for where the team is (rebuilding / contending)",
    "age_injury": "age or health makes the move implausible",
    "other": "explain in notes",
}
FLAG_YES = {"y", "yes", "x", "1", "true"}
MAX_RATE_RATIO = 0.10      # moves inside the veto at <= 10% of the normal rate
MAX_BLOCKED_SHARE = 0.005  # and <= 0.5% of real arrivals blocked
MIN_EXPECTED = 3.0         # below this many expected moves, the audit cannot tell
SERVED = os.path.join("backend", "data", "recs.csv")
REVIEW_DIR = os.path.join("data", "review")
JUDGMENTS = os.path.join(REVIEW_DIR, "judgments.csv")
TEAMS = os.path.join("data", "team_style.csv")


@dataclass
class Rule:
    name: str
    description: str
    veto: Callable[[pd.DataFrame], pd.Series]   # pair table -> True where removed
    active: bool = False


def _rebuilding(p):
    """Bottom 8 by net rating and among the 8 youngest rotations: bad and young."""
    return (p["team_net_rank"] <= 8) & (p["team_age_rank"] <= 8)


# Candidates, all inactive until they pass and you switch them on.
RULES = [
    Rule("vet_star_to_bottom8",
         "a top-5% player aged 27+ to a bottom-8 team by net rating",
         lambda p: p["star"] & (p["age"] >= 27) & (p["team_net_rank"] <= 8)),
    Rule("vet_star_to_rebuilder",
         "a top-5% player aged 27+ to a bottom-8 team with one of the 8 youngest rotations",
         lambda p: p["star"] & (p["age"] >= 27) & _rebuilding(p)),
    Rule("aging_vet_to_rebuilder",
         "a player aged 32+ to a bottom-8 team with one of the 8 youngest rotations",
         lambda p: (p["age"] >= 32) & _rebuilding(p)),
    Rule("star_on_rookie_deal",
         "a top-5% player still on his rookie-scale contract, to anyone",
         lambda p: p["star"] & (p["on_rookie"] == 1)),
    # From the 2025 review: 14 team_direction flags, nearly all players on
    # 25-35% of the cap, mostly to young rotations (good or bad), plus 33+
    # veterans to young teams. All four FAIL on 1999-2017: big contracts go to
    # young / bad teams at 0.6-0.7x the normal rate (Chris Paul to the 2011
    # Clippers, Garnett, LeBron, Love), 33+ veterans on real money to young
    # teams at 0.99x, and 30+ players to rebuilders at 1.06x. Applied to the
    # 2025 lists anyway they would remove 1-3 of the 19 flags and leave the
    # duplication (O'Neale on 23 lists) all but unchanged.
    Rule("big_contract_to_young_team",
         "a player on >= 25% of the cap to one of the 10 youngest rotations",
         lambda p: (p["cap_pct"] >= 0.25) & (p["team_age_rank"] <= 10)),
    Rule("big_contract_to_rebuilder",
         "a player on >= 25% of the cap to a bottom-10 team with one of the 10 youngest rotations",
         lambda p: (p["cap_pct"] >= 0.25) & (p["team_net_rank"] <= 10) & (p["team_age_rank"] <= 10)),
    Rule("old_big_contract_to_young_team",
         "a player aged 33+ on >= 15% of the cap to one of the 10 youngest rotations",
         lambda p: (p["age"] >= 33) & (p["cap_pct"] >= 0.15) & (p["team_age_rank"] <= 10)),
    Rule("vet_30_to_rebuilder",
         "a player aged 30+ to a bottom-10 team with one of the 10 youngest rotations",
         lambda p: (p["age"] >= 30) & (p["team_net_rank"] <= 10) & (p["team_age_rank"] <= 10)),
]


def active_rules():
    return [r for r in RULES if r.active]


def context(pairs, member, players):
    """Add star, team ranks and team age to acquisition's pair table."""
    p = pairs.copy()
    cand = p[["season", "player_id", "bpm"]].drop_duplicates(["season", "player_id"])
    cand["star"] = cand.groupby("season")["bpm"].rank(pct=True) >= rubric.STAR_PCT
    p = p.merge(cand[["season", "player_id", "star"]], how="left")
    teams = p[["season", "TeamId", "team_net"]].drop_duplicates(["season", "TeamId"])
    ages = member.merge(players[["player_id", "season", "age"]], how="left").dropna(subset=["age"])
    ages = (ages.assign(w=ages["age"] * ages["share"]).groupby(["season", "TeamId"])
            .apply(lambda g: g["w"].sum() / g["share"].sum()).rename("team_age").reset_index())
    teams = teams.merge(ages, how="left")
    g = teams.groupby("season")
    teams["team_net_rank"] = g["team_net"].rank(method="first")
    teams["team_age_rank"] = g["team_age"].rank(method="first")
    teams["n_teams"] = g["TeamId"].transform("size")
    return p.merge(teams.drop(columns="team_net"), on=["season", "TeamId"], how="left")


def load(args):
    """Pair table with context, memberships and players."""
    import acquisition
    panel = pd.read_csv(args.panel, usecols=ir.SLOTS + ["season", "TeamId", "OffPoss", "DefPoss"])
    member = ir.memberships(panel)
    players = pd.read_csv(args.players, usecols=["player_id", "season", "player", "age", "bpm"])
    pairs = context(acquisition.build_pairs(args), member, players)
    return pairs, member, players


def served_lists(path=SERVED, teams_path=TEAMS, k=10):
    """The app's top-k as (season, TeamId, player_id, rank)."""
    r = pd.read_csv(path)
    r = r[r["rank"] <= k]
    teams = pd.read_csv(teams_path, usecols=["season", "TeamId", "user_id"])
    return r.merge(teams, on="user_id")[["season", "TeamId", "player_id", "rank"]]


def veto_mask(rows, pairs, rules):
    """True for rows (season, TeamId, player_id) any rule removes. Rows the
    pair table does not cover are kept."""
    if not rules:
        return pd.Series(False, index=rows.index)
    key = ["season", "TeamId", "player_id"]
    p = pairs.set_index(key)
    hit = np.zeros(len(p), dtype=bool)
    for rule in rules:
        hit |= rule.veto(p.reset_index()).fillna(False).to_numpy(bool)
    vetoed = pd.Series(hit, index=p.index)
    return pd.Series(vetoed.reindex(pd.MultiIndex.from_frame(rows[key])).fillna(False).to_numpy(bool),
                     index=rows.index)


def outcome_next(member, players):
    """(season t, TeamId, player_id) -> his share of that team in t+1, and BPM t+1."""
    nxt = member.assign(season=member["season"] - 1).rename(columns={"share": "share_next"})
    bpm = players[["player_id", "season", "bpm"]].assign(season=players["season"] - 1) \
        .rename(columns={"bpm": "bpm_next"})
    return nxt.merge(bpm, how="left")


def audit_rule(rule, pairs, nxt, names, served, seasons):
    """One rule over seasons (lo, hi): the numbers in the module docstring."""
    lo, hi = seasons
    p = pairs[(pairs["season"] >= lo) & (pairs["season"] <= hi) & (pairs["acquired"] >= 0)]
    v = rule.veto(p).fillna(False).to_numpy(bool)
    y = p["acquired"].to_numpy() == 1
    inside = y[v].mean() if v.any() else np.nan
    outside = y[~v].mean()
    expected = v.sum() * outside
    blocked = p[v & y].merge(nxt, how="left").merge(names, how="left")
    blocked["delivered"] = blocked["share_next"] * blocked["bpm_next"]
    ratio = inside / outside if v.any() else np.nan
    share = (v & y).sum() / y.sum()
    if expected < MIN_EXPECTED:
        verdict = "INCONCLUSIVE"
    elif ratio <= MAX_RATE_RATIO and share <= MAX_BLOCKED_SHARE:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    s_mask = veto_mask(served, pairs, [rule])
    last = served["season"].max()
    return {
        "rule": rule.name, "vetoed pairs": int(v.sum()), "of pool": v.mean(),
        "blocked": int((v & y).sum()), "of arrivals": share,
        "rate ratio": ratio, "expected if wrong": expected, "verdict": verdict,
        "served removed": int(s_mask.sum()),
        f"served removed, {last}": int(s_mask[served["season"] == last].sum()),
    }, blocked.sort_values("delivered", ascending=False), served[s_mask]


def cmd_rules(opts, args):
    pairs, member, players = load(args)
    last_labelled = int(pairs.loc[pairs["acquired"] >= 0, "season"].max())
    if opts.holdout:
        n = rubric.log_holdout(opts.log, {**rubric.config(), "audit": ",".join(
            r.name for r in RULES if not opts.rule or r.name in opts.rule)})
        print(f"HOLDOUT view #{n + 1} (logged). Rules changed after reading this are "
              f"tuned on it.\n")
        seasons = (rubric.SPLIT + 1, last_labelled)
    else:
        seasons = (int(pairs["season"].min()), rubric.SPLIT)
    nxt = outcome_next(member, players)
    names = players[["player_id", "season", "player"]]
    teams = pd.read_csv(args.teams, usecols=["season", "TeamId", "user_id"])
    team_name = teams.assign(team=teams["user_id"].str.rsplit("_", n=1).str[0])[
        ["season", "TeamId", "team"]]
    served = served_lists()
    rows = []
    print(f"Decisions in seasons {seasons[0]}-{seasons[1]}.\n")
    for rule in RULES:
        if opts.rule and rule.name not in opts.rule:
            continue
        row, blocked, removed = audit_rule(rule, pairs, nxt, names, served, seasons)
        rows.append(row)
        print(f"-- {rule.name}{' (ACTIVE)' if rule.active else ''}: {rule.description}")
        print(f"   {row['verdict']}: blocks {row['blocked']} real moves "
              f"({row['of arrivals']:.2%} of arrivals); moves happen inside the veto at "
              f"{row['rate ratio']:.2f}x the normal rate; removes {row['served removed']} "
              f"served top-10 entries")
        if len(blocked):
            b = blocked.merge(team_name, how="left").head(opts.show)
            for r in b.itertuples():
                print(f"     {r.season}->{r.season + 1}  {r.player:<24} age {r.age:.0f} -> "
                      f"{r.team:<24} share {r.share_next:.0%}, BPM {r.bpm_next:+.1f}")
            if len(blocked) > opts.show:
                print(f"     ... {len(blocked) - opts.show} more")
        print()
    print(pd.DataFrame(rows).set_index("rule").to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nPASS needs a rate ratio <= {MAX_RATE_RATIO:g} and <= {MAX_BLOCKED_SHARE:.1%} of "
          f"arrivals blocked; read the blocked moves for landmark ones either way.")
    return 0


def cmd_screen(opts, args):
    """Which real arrivals the screens rule out, adding one layer at a time."""
    layers = {"upgrade gate": ev.variant(salary=False),
              "+ salary matching": ev.variant(gap=None),
              f"+ value cap {ir.MAX_GAP:g} (current)": ev.variant(gap=ir.MAX_GAP)}
    systems, panel, impact, players, _ = ev.impact_rankings(args, layers)
    debut = pd.read_csv(args.career, usecols=["player_id", "from"]) \
        .drop_duplicates("player_id").set_index("player_id")["from"]
    arr, _ = ev.moves(ir.memberships(panel), impact, debut, args.min_poss)
    tcodes = pd.read_csv(args.teams, usecols=["season", "TeamId", "TeamAbbreviation"])
    arr = ev.label_mechanism(arr[arr["rankable"]], args.transactions, tcodes)
    salaries = pd.read_csv(args.salaries, usecols=["player_id", "season", "salary"])
    draft = pd.read_csv(args.draft, usecols=["season", "lg", "round", "player_id"])
    ending = ir.pending_fa(salaries, draft).rename("contract ending")
    arr = arr.join(ending, on=["player_id", "season"])
    arr["contract ending"] = arr["contract ending"].map({True: "yes", False: "no"}).fillna("no salary")
    arr["all"] = "all"
    for name in layers:
        sc = systems[name][["season", "TeamId", "player_id", "score"]].rename(columns={"score": name})
        arr = arr.merge(sc, how="left")
        arr[name] = ~(arr[name] > -ir.SOFT_PENALTY / 2)
    print(f"Rankable veteran arrivals: {len(arr)}. Share each screen rules out, layers cumulative:\n")
    for col in ("all", "via", "contract ending"):
        t = arr.groupby(col)[list(layers)].mean()
        t.insert(0, "arrivals", arr.groupby(col).size())
        print(t.to_string(formatters={c: "{:.0%}".format for c in layers}), "\n")
    print("(upgrade gate: he must be worth more than some rotation player who goes out;\n"
          " contract ending: his salary does not continue into t+1 in the expected band,\n"
          " read from t+1 salaries -- hindsight, evaluation only)")
    return 0


def review_rows(args, seasons=None, k=10, loaded=None):
    """The served top-k for review, one row per (season, team, player), with
    the context a reviewer sees and never the model's score or value."""
    pairs, member, players = loaded or load(args)
    served = served_lists(k=k)
    seasons = seasons or [int(served["season"].max())]
    rows = served[served["season"].isin(seasons)].merge(pairs, on=["season", "TeamId", "player_id"],
                                                        how="left")
    full = pd.read_csv(SERVED)[["user_id", "player_id", "replaces", "with"]]
    teams = pd.read_csv(args.teams, usecols=["season", "TeamId", "user_id", "net_rtg"])
    rows = rows.merge(teams, on=["season", "TeamId"], how="left").merge(full, how="left")
    rows = rows.merge(players[["player_id", "season", "player"]], how="left")
    rows["team"] = rows["user_id"].str.rsplit("_", n=1).str[0]
    rows["team_rank"] = (rows["n_teams"] - rows["team_net_rank"] + 1).astype("Int64").astype(str) \
        + "/" + rows["n_teams"].astype("Int64").astype(str)
    return pd.DataFrame({
        "team": rows["team"], "season": rows["season"],
        "team_net_rtg": rows["net_rtg"].round(1), "team_rank (1 = best)": rows["team_rank"],
        "team_age": rows["team_age"].round(1), "team_payroll_pct_cap": (100 * rows["payroll_pct"]).round(0),
        "rank": rows["rank"], "player": rows["player"], "age": rows["age"],
        "bpm": rows["bpm"].round(1), "salary_pct_cap": (100 * rows["cap_pct"]).round(1),
        "rookie_deal": rows["on_rookie"].map({1.0: "yes", 0.0: ""}).fillna(""),
        "replaces": rows["replaces"], "also_sends": rows["with"].fillna("").str.replace(";", ", "),
        "player_id": rows["player_id"], "TeamId": rows["TeamId"],
    }).sort_values(["season", "team", "rank"]).reset_index(drop=True)


def cmd_sheet(opts, args):
    """Export served lists as a CSV, for reviewing in a spreadsheet instead of
    review_app.py; bring it back with `audit.py import`."""
    out = review_rows(args, opts.season, opts.k)
    if opts.teams and opts.teams < out["team"].nunique():
        rng = np.random.default_rng(opts.seed)
        pick = rng.choice(sorted(out["team"].unique()), opts.teams, replace=False)
        out = out[out["team"].isin(pick)]
    out = out.assign(flag="", reason="", notes="", reviewed="")
    os.makedirs(REVIEW_DIR, exist_ok=True)
    seasons = sorted(out["season"].unique())
    path = opts.out or os.path.join(REVIEW_DIR, f"review_{'-'.join(map(str, seasons))}.csv")
    if os.path.exists(path) and not opts.force:
        print(f"'{path}' exists (a review in progress?); pass --force or --out.")
        return 1
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} recommendations to '{path}'. Put y in `flag` and reason codes\n"
          f"(';'-separated) for ones that should never be made, then `python audit.py import {path}`:")
    for code, text in REASONS.items():
        print(f"  {code:<15} {text}")
    return 0


JUDGMENT_KEY = ["season", "TeamId", "player_id"]
# What was shown when the judgment was made, kept so a flagged entry -- gone
# from the served lists once applied -- can still be shown and unflagged.
SNAPSHOT = ["team", "player", "age", "bpm", "salary_pct_cap", "rookie_deal", "replaces", "also_sends"]
JUDGMENT_COLS = JUDGMENT_KEY + ["verdict", "reason", "notes"] + SNAPSHOT + ["updated"]


def load_judgments(path=JUDGMENTS):
    """Every review decision: verdict 'ok' (seen, fine) or 'flag' (never
    make this recommendation), one row per (season, TeamId, player_id)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=JUDGMENT_COLS)
    j = pd.read_csv(path, dtype={"player_id": str, "verdict": str, "reason": str, "notes": str,
                                 "replaces": str, "also_sends": str, "rookie_deal": str})
    return j.fillna({"reason": "", "notes": "", "replaces": "", "also_sends": "", "rookie_deal": ""})


def save_judgments(j, path=JUDGMENTS):
    """Atomic write: a crash mid-save must not lose the review."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    j[JUDGMENT_COLS].sort_values(JUDGMENT_KEY).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _key(r):
    return int(r["season"]), int(r["TeamId"]), str(r["player_id"])


def judge(judgments, updates, shown, now=None):
    """Upsert decisions. Each update has the key, and any of `reasons`
    (codes; non-empty = flag, empty = ok), `notes`, `verdict`. `shown` (the
    review rows) supplies the snapshot for entries judged the first time.
    Raises ValueError on an unknown reason code or verdict, or a key that is
    neither shown nor judged before."""
    now = now or datetime.datetime.now().isoformat(timespec="seconds")
    table = {_key(r): dict(r) for r in judgments.to_dict("records")} if len(judgments) else {}
    info = {_key(r): r for r in shown.to_dict("records")} if len(shown) else {}
    for u in updates:
        key = _key(u)
        if key not in table:
            if key not in info:
                raise ValueError(f"{key} is neither in the lists nor judged before")
            table[key] = {"season": key[0], "TeamId": key[1], "player_id": key[2],
                          "verdict": "ok", "reason": "", "notes": "",
                          **{c: info[key].get(c, "") for c in SNAPSHOT}}
        row = table[key]
        if "reasons" in u:
            codes = [c for c in u["reasons"] if c]
            bad = [c for c in codes if c not in REASONS]
            if bad:
                raise ValueError(f"unknown reason {bad}")
            row["reason"] = ";".join(c for c in REASONS if c in codes)
            row["verdict"] = "flag" if codes else "ok"
        if "verdict" in u:
            if u["verdict"] not in ("ok", "flag"):
                raise ValueError(f"unknown verdict {u['verdict']}")
            row["verdict"] = u["verdict"]
            if u["verdict"] == "ok" and "reasons" not in u:
                row["reason"] = ""
        if "notes" in u:
            row["notes"] = str(u["notes"])
        row["updated"] = now
    return pd.DataFrame(list(table.values()), columns=JUDGMENT_COLS)


def flagged_mask(rows, judgments):
    """True for rows (season, TeamId, player_id) the review flagged."""
    f = judgments[judgments["verdict"] == "flag"] if len(judgments) else judgments
    if not len(f):
        return pd.Series(False, index=rows.index)
    o = f[JUDGMENT_KEY].astype({"season": int, "TeamId": int}).drop_duplicates().assign(hit=True)
    m = rows[JUDGMENT_KEY].astype({"season": int, "TeamId": int}).merge(o, on=JUDGMENT_KEY, how="left")["hit"]
    return pd.Series(m.notna().to_numpy(), index=rows.index)


def import_sheet(sheet, judgments):
    """A review sheet (audit.py sheet, or the old review_app.py format) into
    judgments: flagged rows -> 'flag', rows marked reviewed -> 'ok'. Existing
    judgments for the same entries are replaced."""
    s = pd.read_csv(sheet, dtype={"flag": str, "reason": str, "notes": str, "reviewed": str,
                                  "player_id": str, "rookie_deal": str})
    flag = s["flag"].fillna("").str.strip().str.lower().isin(FLAG_YES)
    seen = s.get("reviewed", pd.Series("", index=s.index)).fillna("").str.strip().str.lower().isin(FLAG_YES)
    s = s[flag | seen].assign(verdict=np.where(flag[flag | seen], "flag", "ok"))
    s["reason"] = np.where(s["verdict"] == "flag", s["reason"].fillna(""), "")
    s["notes"] = s["notes"].fillna("")
    s["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    for c in SNAPSHOT:
        if c not in s:
            s[c] = ""
    new = s[JUDGMENT_COLS].astype({"season": int, "TeamId": int})
    if not len(judgments):
        return new.reset_index(drop=True)
    old = judgments.astype({"season": int, "TeamId": int})
    old = old.merge(new[JUDGMENT_KEY].assign(_dup=True), on=JUDGMENT_KEY, how="left")
    old = old[old["_dup"].isna()].drop(columns="_dup")
    return pd.concat([old, new], ignore_index=True)[JUDGMENT_COLS]


def cmd_import(opts, args):
    j = import_sheet(opts.file, load_judgments(opts.out))
    save_judgments(j, opts.out)
    print(f"'{opts.out}' now holds {len(j)} judgments ({(j['verdict'] == 'flag').sum()} flags). "
          f"Flags apply on the next `python export_recs.py`.")
    return 0


def cmd_summarize(opts, args):
    """What flagged recommendations have in common (the judgments, or a sheet)."""
    if opts.file:
        s = pd.read_csv(opts.file, dtype={"flag": str, "reason": str, "notes": str})
    else:
        j = load_judgments()
        s = j.assign(flag=np.where(j["verdict"] == "flag", "y", ""), rank="")
    s["flagged"] = s["flag"].fillna("").str.strip().str.lower().isin(FLAG_YES)
    s["reason"] = s["reason"].fillna("").str.strip()
    # review_app.py can store several codes, ';'-separated.
    s["codes"] = s["reason"].str.split(";").apply(lambda c: [x.strip() for x in c if x.strip()])
    valid = s["codes"].apply(lambda c: bool(c) and all(x in REASONS for x in c))
    bad = s[s["flagged"] & ~valid]
    if len(bad):
        print(f"{len(bad)} flagged rows have no valid reason (use one of {', '.join(REASONS)}):")
        print(bad[["team", "season", "player", "reason"]].to_string(index=False), "\n")
    unflagged_reason = s[~s["flagged"] & (s["reason"] != "")]
    if len(unflagged_reason):
        print(f"{len(unflagged_reason)} rows have a reason but no flag; counted as unflagged.\n")
    pairs, _, _ = load(args)
    s = s.merge(pairs[["season", "TeamId", "player_id", "star", "team_net_rank", "team_age_rank"]],
                on=["season", "TeamId", "player_id"], how="left")
    print(f"{s['flagged'].sum()} of {len(s)} recommendations flagged.\n")
    cols = {"age": "mean", "bpm": "mean", "star": "mean", "salary_pct_cap": "mean",
            "team_net_rank": "mean", "team_age_rank": "mean"}
    s["star"] = s["star"].astype(float)
    s["rookie"] = (s["rookie_deal"] == "yes").astype(float)
    cols["rookie"] = "mean"
    # A row flagged for two reasons counts once under each.
    by_code = s[s["flagged"]].assign(group=s["codes"].apply(lambda c: c or ["(no reason)"]))
    by_code = by_code.explode("group")
    groups = pd.concat([by_code, s[~s["flagged"]].assign(group="(not flagged)")], ignore_index=True)
    t = groups.groupby("group").agg(n=("player", "size"), **{c: (c, f) for c, f in cols.items()})
    print("Who gets flagged (team_net_rank / team_age_rank: 1 = worst / youngest):")
    print(t.to_string(float_format=lambda v: f"{v:.2f}"), "\n")
    print(f"Distinct players flagged: {s.loc[s['flagged'], 'player_id'].nunique()}\n")
    for reason, g in by_code.groupby("group"):
        print(f"-- {reason or '(no reason)'} ({len(g)})")
        for r in g.itertuples():
            note = f"  -- {r.notes}" if isinstance(r.notes, str) and r.notes.strip() else ""
            rank = f"#{r.rank:<3} " if str(r.rank).strip() else ""
            print(f"   {r.season} {r.team:<24} {rank}{r.player:<24} age {float(r.age):.0f}{note}")
    print("\nTo act on a cluster: add a Rule to audit.RULES, then `python audit.py rules --rule NAME`.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rules", help="audit veto rules against real moves")
    r.add_argument("--rule", nargs="+", help="only these rules")
    r.add_argument("--holdout", action="store_true", help="audit on the holdout seasons (logged)")
    r.add_argument("--show", type=int, default=10, help="blocked moves listed per rule")
    r.add_argument("--log", default=rubric.HOLDOUT_LOG)
    sub.add_parser("screen", help="which real arrivals the salary screen rules out")
    sh = sub.add_parser("sheet", help="export served lists as a CSV (spreadsheet review)")
    sh.add_argument("--season", type=int, nargs="+", help="default: the serving season")
    sh.add_argument("--teams", type=int, help="sample this many team-seasons (default all)")
    sh.add_argument("--k", type=int, default=10)
    sh.add_argument("--seed", type=int, default=0)
    sh.add_argument("--out")
    sh.add_argument("--force", action="store_true", help="overwrite an existing sheet")
    sm = sub.add_parser("summarize", help="what flagged recommendations have in common")
    sm.add_argument("file", nargs="?", help="a review sheet (default: the judgments)")
    im = sub.add_parser("import", help="add a filled-in review sheet to the judgments")
    im.add_argument("file")
    im.add_argument("--out", default=JUDGMENTS)
    opts = ap.parse_args()
    args = ev.parser().parse_args([])
    return {"rules": cmd_rules, "screen": cmd_screen, "sheet": cmd_sheet,
            "summarize": cmd_summarize, "import": cmd_import}[opts.cmd](opts, args)


if __name__ == "__main__":
    sys.exit(main())
