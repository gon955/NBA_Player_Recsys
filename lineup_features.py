"""Crosswalk pbpstats integer NBA player ids to Basketball-Reference string ids.

pbpstats keys on NBA ids (203999); this repo keys on BBRef ids (jokicni01). The
two share no key, so the join has to go through normalised name + season.

Source matters more than the matching does. `players.csv` is the *filtered*
modelling subset (9,530 rows) — building against it caps resolution at ~71%
because it has already dropped players the clustering pipeline excluded, and
`EntityId` contains every player who appeared in any lineup. `data/Advanced.csv`
is the full universe (13,107 player-seasons from 1999) and reaches ~99%.
"""

import argparse
import os
import re
import sys
import unicodedata

import pandas as pd

from helper import TOTAL_MARKERS, canonical_team, era_of

### Cross reference players using BB_ref_id (jokicni01) to integer NBA IDS (203999)

COVERAGE_FLOOR = 0.98

# NBA id -> BBRef id, for players a name override can't target safely because
# the same pbpstats spelling belongs to someone else in another season: the
# 1999-2005 "Steven Smith" is BBRef's Steve Smith, the 2007 one is BBRef's
# Steven Smith. Applied after the name join, so it only fills gaps.
ID_OVERRIDES = {
    120: "smithst01",      # Steven Smith -> Steve Smith
    1631466: "willije02",  # Nate Williams, listed as Jeenathan until 2023
}

# A handful of 2025 lineups have PlusMinus 2 or 4 points above Points -
# OpponentPoints in pbpstats itself. The ratings use Points/OpponentPoints, so
# tolerate a trickle of that, but fail loudly if it stops being a trickle.
MAX_DRIFT_SHARE = 0.001

# Keyed on the *pbpstats* spelling, which is the side that carries the variants.
# The BBRef file is the reference spelling, so nothing here is applied to it.
NAME_OVERRIDES = {
    "Clar. Weatherspoon": "Clarence Weatherspoon",
    "Dan Schayes": "Danny Schayes",
    "Ike Austin": "Isaac Austin",
    "Charles R. Jones": "Charles Jones",
    # Nicknames and short forms BBRef spells out (or vice versa).
    "Enes Kanter": "Enes Freedom",          # legal name change, 2021
    "Flip Murray": "Ronald Murray",
    "Slava Medvedenko": "Stanislav Medvedenko",
    "Michael Sweetney": "Mike Sweetney",
    "Nicolas Claxton": "Nic Claxton",
    "Matt Hurt": "Matthew Hurt",
    "Mitchell Creek": "Mitch Creek",
    "Pooh Jeter": "Eugene Jeter",
    "Ike Fontaine": "Isaac Fontaine",
    "Ibrahim Kutluay": "Ibo Kutluay",
    "Norman Richardson": "Norm Richardson",
    "Jeffery Taylor": "Jeff Taylor",
    "Kiwane Garris": "Kiwane Lemorris Garris",
    "Walter Lemon Jr.": "Walt Lemon Jr.",
    "Vitor Faverani": "Vitor Luiz Faverani",
    "Juan Hernangomez": "Juancho Hernangomez",
    "Charles Brown Jr.": "Charlie Brown Jr.",
    "Ronald Holland II": "Ron Holland",
    "Kenyon Martin Jr.": "KJ Martin",
    "Vincent Edwards": "Vince Edwards",
    "Vincent Hunter": "Vince Hunter",
    # pbpstats mangles a few transliterated names into duplicated tokens.
    "Wang Zhi-zhi": "Wang Zhizhi",
    "Ha Ha": "Ha Seung-Jin",
    "Sun Sun": "Sun Yue",
}

# NFKD decomposes accented Latin letters into base + combining mark, so
# encode("ascii","ignore") leaves the base behind. These characters have no
# decomposition, so the same call silently DELETES them: "Aşık" -> "Ask" and
# "Pleiß" -> "Plei". Map them explicitly before folding.
_CHAR_FOLD = str.maketrans({
    "ı": "i", "ß": "ss", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D",
    "ł": "l", "Ł": "L", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "ð": "d", "þ": "th", "ŧ": "t", "ħ": "h",
})

# Anchored so a suffix is only stripped at the end of the name. An unanchored
# .replace("jr","") turns "Jrue Holiday" into "ue holiday" and "Salah Mejri"
# into "salah mei".
_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    # Fold accents before anything else: pbpstats and BBRef disagree on Kukoč,
    # Stojaković and López, which alone is worth ~5 points of coverage.
    name = name.translate(_CHAR_FOLD)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    # Periods and apostrophes are *deleted*, not spaced out: "A.J. Green" has to
    # collapse to "aj green" to meet pbpstats' "AJ Green". Spacing them instead
    # yields "a j green" and silently loses every initialised first name.
    name = re.sub(r"[.'’`]", "", name)
    name = re.sub(r"[^a-z ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    # Repeat: "Robert Horry Jr. III" would carry two suffixes.
    while _SUFFIX_RE.search(name):
        name = _SUFFIX_RE.sub("", name)
    return name.strip()


# pbpstats uses the NBA's abbreviations, BBRef its own. Only these three differ
# from 1999 on; CHA is the Bobcats to both until 2014, then BBRef's Hornets are CHO.
_PBP_TO_BBREF_TEAM = {"BKN": "BRK", "PHX": "PHO"}


def bbref_team(abbr, season):
    if abbr == "CHA" and season >= 2015:
        return "CHO"
    return _PBP_TO_BBREF_TEAM.get(abbr, abbr)


def _resolve_by_team(crosswalk, collisions, pbp_df, bbref_df):
    """Second pass for same-name, same-season players: add team to the key.

    Each pbpstats row is one (player, team) stint and BBRef keeps a row per team
    as well, so a stint names exactly one BBRef id. A player whose stints point
    at different ids is left unmatched rather than guessed.
    """
    key = ["norm_name", "season"]

    pbp = pbp_df.copy()
    pbp["season"] = pbp["season"].astype(int)
    pbp["norm_name"] = pbp["Name"].replace(NAME_OVERRIDES).apply(normalize_name)
    pbp = pbp[pd.MultiIndex.from_frame(pbp[key]).isin(collisions.index)]
    pbp["team"] = [bbref_team(a, s) for a, s in zip(pbp["TeamAbbreviation"], pbp["season"])]

    bbref = bbref_df[~bbref_df["team"].isin(TOTAL_MARKERS)].copy()
    bbref["season"] = bbref["season"].astype(int)
    bbref["norm_name"] = bbref["player"].apply(normalize_name)
    bbref = bbref[pd.MultiIndex.from_frame(bbref[key]).isin(collisions.index)]

    stints = pbp.merge(bbref, on=key + ["team"], how="inner")
    hits = stints.groupby(["EntityId", "season"]).agg(
        player_id=("player_id", "first"),
        player=("player", "first"),
        n_ids=("player_id", "nunique"),
    )
    ambiguous = hits[hits["n_ids"] > 1]
    if len(ambiguous):
        print(f"WARNING: {len(ambiguous)} players still ambiguous after the team "
              f"pass; leaving them UNMATCHED: {list(ambiguous.index[:3])}")
    hits = hits[hits["n_ids"] == 1]

    idx = pd.MultiIndex.from_frame(crosswalk[["EntityId", "season"]])
    fill = crosswalk["player_id"].isna().to_numpy() & idx.isin(hits.index)
    crosswalk.loc[fill, ["player_id", "player"]] = (
        hits.loc[idx[fill], ["player_id", "player"]].to_numpy()
    )
    print(f"Team pass resolved {fill.sum()} same-name player-seasons")
    return crosswalk


def build_crosswalk(pbp_file_path="data/pbpstats_players.csv",
                    bbref_file_path="data/Advanced.csv",
                    output_dir="data"):

    crosswalk_path = os.path.join(output_dir, "player_id_crosswalk.csv")
    report_path = os.path.join(output_dir, "unmatched_report.csv")

    pbp_df = pd.read_csv(pbp_file_path,
                         usecols=["EntityId", "Name", "season", "TeamAbbreviation"],
                         low_memory=False)
    bbref_df = pd.read_csv(bbref_file_path,
                           usecols=["player_id", "player", "season", "team"])

    pbp_seasons = pbp_df[["EntityId", "Name", "season"]].drop_duplicates()
    bbref_seasons = bbref_df[["player_id", "player", "season"]].drop_duplicates()

    # pbpstats only goes back so far and the repo only models 1999+; trimming
    # BBRef to the same window keeps the coverage number meaningful.
    seasons = set(pbp_seasons["season"].astype(int))
    bbref_seasons = bbref_seasons[bbref_seasons["season"].astype(int).isin(seasons)]

    total_pbp_seasons = len(pbp_seasons)

    # Overrides belong on the pbpstats side. Applied to BBRef they match nothing:
    # "Clar. Weatherspoon" appears 7x in pbpstats and 0x in BBRef.
    pbp_seasons["matched_name"] = pbp_seasons["Name"].replace(NAME_OVERRIDES)
    bbref_seasons["matched_name"] = bbref_seasons["player"]

    pbp_seasons["norm_name"] = pbp_seasons["matched_name"].apply(normalize_name)
    bbref_seasons["norm_name"] = bbref_seasons["matched_name"].apply(normalize_name)

    pbp_seasons["season"] = pbp_seasons["season"].astype(int)
    bbref_seasons["season"] = bbref_seasons["season"].astype(int)

    # A duplicated (norm_name, season) on the right side fans the merge out and
    # would duplicate every lineup row downstream. Collapse first, and keep the
    # collisions so they can be inspected rather than silently resolved.
    key = ["norm_name", "season"]
    collisions = (bbref_seasons.groupby(key)["player_id"].nunique()
                               .loc[lambda s: s > 1])
    if len(collisions):
        # These are genuinely different people sharing a name in one season —
        # Chris Johnson 2013 is both johnsch03 (MIN) and johnsch04 (MEM).
        # Name alone can't separate them, so they go to the team pass below.
        print(f"NOTE: {len(collisions)} name+season keys map to multiple BBRef "
              f"players; resolving them by team: {list(collisions.index[:3])}")
        bbref_seasons = bbref_seasons.set_index(key).drop(collisions.index).reset_index()
    bbref_lookup = bbref_seasons.drop_duplicates(subset=key, keep="first")

    # Direction matters: every id in EntityId must resolve, so drive the join
    # from pbpstats and measure coverage over that side.
    crosswalk = pd.merge(
        pbp_seasons,
        bbref_lookup[key + ["player_id", "player"]],
        on=key,
        how="left",
    )

    if len(collisions):
        crosswalk = _resolve_by_team(crosswalk, collisions, pbp_df, bbref_df)

    bbref_names = bbref_df.drop_duplicates("player_id").set_index("player_id")["player"]
    by_id = crosswalk["player_id"].isna() & crosswalk["EntityId"].isin(ID_OVERRIDES)
    crosswalk.loc[by_id, "player_id"] = crosswalk.loc[by_id, "EntityId"].map(ID_OVERRIDES)
    crosswalk.loc[by_id, "player"] = crosswalk.loc[by_id, "player_id"].map(bbref_names)

    crosswalk = crosswalk.rename(columns={
        "EntityId": "nba_player_id",
        "player_id": "bbref_id",
        "player": "player_name",
    })
    crosswalk["player_name"] = crosswalk["player_name"].fillna(crosswalk["Name"])

    matched = crosswalk[crosswalk["bbref_id"].notna()].copy()
    unmatched = crosswalk[crosswalk["bbref_id"].isna()].copy()

    coverage_rate = len(matched) / total_pbp_seasons

    print(f"Successfully matched: {len(matched)} of {total_pbp_seasons} "
          f"pbpstats player-seasons ({coverage_rate:.2%})")

    final_columns = ["bbref_id", "nba_player_id", "player_name", "season"]
    out = matched[final_columns].drop_duplicates()

    dupes = out.duplicated(subset=["nba_player_id", "season"]).sum()
    if dupes:
        raise ValueError(
            f"Pipeline Failure: {dupes} nba_player_id+season rows are duplicated. "
            f"Joining this crosswalk would multiply lineup rows."
        )

    os.makedirs(output_dir, exist_ok=True)
    out.to_csv(crosswalk_path, index=False)

    if len(unmatched) > 0:
        unmatched_sorted = (unmatched[["nba_player_id", "player_name", "season"]]
                            .sort_values(by="player_name"))
        unmatched_sorted.to_csv(report_path, index=False)
        print(f"WARNING: {len(unmatched)} records unmatched. Wrote tracking report "
              f"to '{report_path}'")

    if coverage_rate < COVERAGE_FLOOR:
        raise ValueError(
            f"Pipeline Failure: Coverage rate ({coverage_rate:.2%}) is below threshold "
            f"({COVERAGE_FLOOR:.2%}).\n"
            f"Please check '{report_path}' and append missing strings to your "
            f"NAME_OVERRIDES block (keyed on the pbpstats spelling)."
        )

    print(f"Crosswalk successfully compiled at '{crosswalk_path}' ({len(out)} rows)")
    return out


def load_lineups(path):
    KEY_COLS = [
          "season",             # int, the repo's ending-year convention
          "TeamId",             # NBA team id, join key
          "TeamAbbreviation",   # for canonical_team() -> team_full
          "EntityId",           # "1627750-1629008-..." -> splits into p1..p5
          "Name",               # human-readable lineup, keep for inspection/debug
      ]
    COUNT_COLS = [
          "OffPoss", "DefPoss",          # weights AND the rate denominators
          "Points", "OpponentPoints",    # target numerators
          "Minutes", "GamesPlayed",      # descriptive / filtering
          "PlusMinus",                   # free cross-check on the fill
      ]


    lu = pd.read_csv(path,usecols = KEY_COLS + COUNT_COLS, low_memory= False)

    lu[COUNT_COLS] = lu[COUNT_COLS].fillna(0)

    slots = lu["EntityId"].str.split("-")
    lu = lu[slots.apply(lambda x: len(set(x))) == 5]

    dupes = lu.duplicated(subset = ["season","TeamId","EntityId"]).sum()
    if dupes:
          raise ValueError(f"{dupes} duplicate (season, TeamId, EntityId) rows in {path}")
    drift = (lu["PlusMinus"] - (lu["Points"] - lu["OpponentPoints"])).abs() > 1e-6
    if drift.mean() > MAX_DRIFT_SHARE:
          raise ValueError(f"{drift.sum()} rows where PlusMinus != Points - "
                           f"OpponentPoints; the NaN fill is wrong")
    if drift.any():
          print(f"WARNING: {drift.sum()} lineups where PlusMinus != Points - "
                f"OpponentPoints (seasons {sorted(lu.loc[drift, 'season'].unique())}); "
                f"a pbpstats quirk, ratings use Points/OpponentPoints")

    return lu.reset_index(drop=True)


SLOTS = [f"p{k}" for k in range(1, 6)]

# What each lineup slot carries over from players.csv. Era-relative archetype
# plus position and a box-score impact prior; enough to ask "which archetype
# mixes outscore their parts" without dragging in all 68 columns.
COVARIATES = {"cluster_label": "cluster", "pos": "pos", "bpm": "bpm"}


def build_panel(lineups, crosswalk):
    """One row per (season, team, 5-man unit), players as BBRef ids.

    A lineup with any unresolved player is dropped whole: keeping it with a
    hole would credit its possessions to the other four.
    """
    panel = lineups.copy()
    panel["season"] = panel["season"].astype(int)

    nba_ids = panel["EntityId"].str.split("-", expand=True).astype(int)
    nba_ids.columns = [f"{s}_nba_id" for s in SLOTS]

    lookup = crosswalk.set_index(["nba_player_id", "season"])["bbref_id"]
    for s in SLOTS:
        idx = pd.MultiIndex.from_arrays([nba_ids[f"{s}_nba_id"], panel["season"]])
        panel[f"{s}_id"] = lookup.reindex(idx).to_numpy()

    resolved = panel[[f"{s}_id" for s in SLOTS]].notna().all(axis=1)
    poss = panel["OffPoss"] + panel["DefPoss"]
    print(f"Lineups fully resolved: {resolved.sum()} of {len(panel)} "
          f"({resolved.mean():.2%}), {poss[resolved].sum() / poss.sum():.2%} of possessions")
    panel = panel[resolved].copy()

    # Two pbpstats ids resolving to one BBRef id would put a player on the
    # floor twice. The crosswalk shouldn't allow it; check rather than trust.
    ids = panel[[f"{s}_id" for s in SLOTS]]
    doubled = ids.nunique(axis=1) < 5
    if doubled.any():
        raise ValueError(f"{doubled.sum()} lineups list the same BBRef player twice; "
                         f"the crosswalk maps two NBA ids to one player")

    panel["team_full"] = [canonical_team(bbref_team(a, s), s)
                          for a, s in zip(panel["TeamAbbreviation"], panel["season"])]
    panel["era"] = panel["season"].map(era_of)
    panel["lineup_id"] = (panel["season"].astype(str) + "_" + panel["TeamId"].astype(str)
                          + "_" + panel["EntityId"])

    # Ratings per 100, left NaN where the unit never had the ball (or never
    # defended). A 0 there would read as the worst offense in the league.
    panel["off_rtg"] = 100 * panel["Points"] / panel["OffPoss"].where(panel["OffPoss"] > 0)
    panel["def_rtg"] = 100 * panel["OpponentPoints"] / panel["DefPoss"].where(panel["DefPoss"] > 0)
    panel["net_rtg"] = panel["off_rtg"] - panel["def_rtg"]
    panel["poss"] = panel["OffPoss"] + panel["DefPoss"]

    return panel.reset_index(drop=True)


def attach_covariates(panel, players):
    """Add each slot's archetype, position and BPM from players.csv.

    players.csv is the filtered modelling subset (g >= 20 and the like), so
    deep-bench players have no row. They become cluster "unknown" rather than
    being dropped: the lineup's possessions are still real.
    """
    cov = players[["player_id", "season"] + list(COVARIATES)].copy()
    cov["season"] = cov["season"].astype(int)
    cov = cov.set_index(["player_id", "season"])

    panel = panel.copy()
    for s in SLOTS:
        idx = pd.MultiIndex.from_arrays([panel[f"{s}_id"], panel["season"]])
        vals = cov.reindex(idx)
        for col, short in COVARIATES.items():
            panel[f"{s}_{short}"] = vals[col].to_numpy()
        panel[f"{s}_cluster"] = panel[f"{s}_cluster"].fillna("unknown")

    known = (panel[[f"{s}_cluster" for s in SLOTS]] != "unknown").sum(axis=1)
    panel["n_known"] = known
    full = panel.loc[known == 5, "poss"].sum() / panel["poss"].sum()
    print(f"Possessions with all five players in players.csv: {full:.2%}")
    return panel


def explode(panel):
    """Long form: one row per (lineup, player on the floor).

    Each lineup's counts are repeated on all five of its rows, so sum them per
    lineup, not per row. This is the shape a player-level on/off aggregate or
    an RAPM design matrix is built from.
    """
    keep = ["lineup_id", "season", "era", "team_full", "TeamId",
            "OffPoss", "DefPoss", "Points", "OpponentPoints", "poss"]
    per_slot = [c for c in ("id", "cluster", "pos", "bpm") if f"p1_{c}" in panel]

    parts = []
    for s in SLOTS:
        part = panel[keep + [f"{s}_{c}" for c in per_slot]].copy()
        part.columns = keep + ["player_id" if c == "id" else c for c in per_slot]
        part["slot"] = s
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)

    counts = long.groupby("lineup_id").size()
    if (counts != 5).any():
        raise ValueError(f"{(counts != 5).sum()} lineups did not explode to 5 rows")
    return long.sort_values(["lineup_id", "slot"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pbp", default="data/pbpstats_players.csv")
    ap.add_argument("--bbref", default=os.path.join("data", "Advanced.csv"),
                    help="full player universe; do NOT point this at players.csv, "
                         "which is the filtered modelling subset")
    ap.add_argument("--lineups", default="data/pbpstats_lineups.csv")
    ap.add_argument("--players", default=os.path.join("backend", "data", "players.csv"),
                    help="covariates (cluster, pos, bpm) per player-season")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    crosswalk = build_crosswalk(args.pbp, args.bbref, args.out_dir)
    panel = build_panel(load_lineups(args.lineups), crosswalk)
    panel = attach_covariates(panel, pd.read_csv(args.players))

    out = os.path.join(args.out_dir, "lineup_panel.csv")
    panel.to_csv(out, index=False)
    print(f"Lineup panel written to '{out}' ({len(panel)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
