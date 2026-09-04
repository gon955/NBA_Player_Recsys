
import pandas as pd

from helper import canonical_team

players      = pd.read_csv("players.csv")
teams        = pd.read_csv("teams.csv")
stints       = pd.read_csv("stints.csv")
interactions = pd.read_csv("interactions.csv")


_POS_SHARE_LABELS = [
    ("share_pg", "PG"), ("share_sg", "SG"), ("share_sf", "SF"),
    ("share_pf", "PF"), ("share_c", "C"),
]


def position_phrase(row):
    """Measured minute shares per spot, falling back to the `pos` label.

    rec_sys.py merges pg_percent…c_percent from the play-by-play file into
    players.csv. Saying "68% SG, 29% SF" tells the retriever a combo wing; the
    old `pos` string said "SG" and made him indistinguishable from a pure guard.
    Spots under 10% are dropped as rounding noise.
    """
    parts = [
        (label, row[col]) for col, label in _POS_SHARE_LABELS
        if col in row and pd.notna(row[col]) and row[col] >= 0.10
    ]
    if not parts:
        return f"played {row['pos']}"
    parts.sort(key=lambda p: -p[1])
    return "played " + ", ".join(f"{share:.0%} {label}" for label, share in parts)


# Player Chunks
def row_to_chunk(row):
    three_pt = f"{row['x3p_percent']:.1%}" if pd.notna(row['x3p_percent']) else "N/A"
    team_name = row.get('team_full',canonical_team(row['team_per100'], row['season']))

    age_bin = pd.cut(
        [row['age']],
        bins = [0, 22, 26, 30, 35, 40, 60],
        labels = ["U22", "23-26", "27-30", "31-34", "35-39", "40+"],
        right = False, include_lowest= True
    )[0]

    return (
        f"{row['player']} in the {row['season']} season ({row['era']}) {position_phrase(row)} "
        f"for {team_name} at age {row['age']} (age group: {age_bin}). "
        f"Archetype: {row['cluster_label']} (cluster {row['cluster']}, within {row['era']} era peers) — LightFM item feature pcluster={row['cluster_label']}. "
        f"Scoring: {row['pts_per_100_poss']:.1f} pts per 100 on "
        f"{row['fg_percent']:.1%} FG%, {three_pt} 3P%, {row['ts_percent']:.1%} TS%. "
        f"Playmaking: {row['ast_per_100_poss']:.1f} ast, {row['tov_per_100_poss']:.1f} tov per 100. "
        f"Rebounding: {row['trb_per_100_poss']:.1f} total ({row['orb_per_100_poss']:.1f} off, {row['drb_per_100_poss']:.1f} def). "
        f"Defense: {row['stl_per_100_poss']:.1f} stl, {row['blk_per_100_poss']:.1f} blk per 100. "
        f"Impact: {row['bpm']:.1f} BPM, {row['vorp']:.1f} VORP, {row['ws']:.1f} WS, {row['ws_48']:.3f} WS/48. "
        f"Usage: {row['usg_percent']:.1f}%, ORtg: {row['o_rtg']:.1f}, DRtg: {row['d_rtg']:.1f}."
    )

# Team Chunks
def team_row_to_chunk(row):
    three_pt = f"{row['x3p_percent']:.1%}" if pd.notna(row['x3p_percent']) else "N/A"

    return (
        f"The {row['team']} in the {row['season']} season ({row['era']}) are categorized as: "
        f"{row['cluster_label']} — LightFM user feature tcluster={row['cluster_label']}. "
        f"Pace: {row['pace']:.1f} possessions/game. "
        f"ORtg: {row['o_rtg']:.1f}, DRtg: {row['d_rtg']:.1f}, NRtg: {row['n_rtg']:.1f}. "
        f"Scoring: {row['pts_per_100_poss']:.1f} pts/100 on {row['fg_percent']:.1%} FG%, "
        f"{three_pt} 3P%, {row['ts_percent']:.1%} TS%. "
        f"Playmaking: {row['ast_per_100_poss']:.1f} ast, {row['tov_percent']:.1f}% tov rate. "
        f"Rebounding: {row['trb_per_100_poss']:.1f} total, {row['orb_percent']:.1f}% ORB rate. "
        f"Defense: {row['stl_per_100_poss']:.1f} stl, {row['blk_per_100_poss']:.1f} blk/100, "
        f"opp eFG%: {row['opp_e_fg_percent']:.1%}. "
        f"SOS: {row['sos']:.2f}, SRS: {row['srs']:.2f}."
    )

# Bridge roster comp / players

def add_archetype_composition(team_row,stints,player_archetype):
    roster = stints[(stints['season'] == team_row['season']) &
                   (stints['team_full']==team_row['team'])]
    if roster.empty:
        return team_row['chunk']
    
    labels = roster['player_id'].map(player_archetype).dropna()
    if labels.empty:
        return team_row['chunk']
    
    archetype_counts = labels.value_counts().to_dict()
    archetype_str = ",".join([f"{v}x {k}" for k,v in archetype_counts.items()])

    return team_row['chunk'] + f" Roster archetype composition: {archetype_str}."

def build_all_chunks():
    player_archetype = players.set_index('player_id')['cluster_label'].to_dict()
    players['chunk'] = players.apply(row_to_chunk, axis=1)
    teams['chunk'] = teams.apply(team_row_to_chunk, axis=1)
    teams['chunk'] = teams.apply(
        lambda r: add_archetype_composition(r, stints, player_archetype), axis=1
    )

    return players[['item_id','player','season','era','cluster_label','chunk']], \
           teams[['user_id','team','season','era','cluster_label','chunk']]



player_chunks, team_chunks = build_all_chunks()

# sanity check
# print("=== PLAYER SAMPLES ===")
# for chunk in player_chunks['chunk'].sample(3):
#     print(chunk)
#     print("---")

# print("=== TEAM SAMPLES ===")
# for chunk in team_chunks['chunk'].sample(3):
#     print(chunk)
#     print("---")