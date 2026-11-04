"""Impact-based recommendations: who would make this team-season better?

Replaces LightFM's roster co-occurrence score. For a team-season and a
candidate from the same season (the app's existing framing):

  value(p)    net impact projected to t+1 (project(): shrunk, with the
              previous season). The archetype main effect (fit_model.py, a
              per-season correction such as rim protectors') was dropped: with
              it off, realism 0.475 -> 0.477, quality 0.280 -> 0.281, team R^2
              +0.053 -> +0.051, and the projection's correlation with movers'
              t+1 impact rises 0.632 -> 0.636 -- no measurable value, and it was
              the recommender's last tie to the LightFM-era clustering
              (ARCHETYPES = True restores it).
  score       value(c) + REALISM_WEIGHT * log P(T acquires c), among the
              candidates the team can legally acquire (`quality`). Legal: the
              salary screen and value cap below, passed with some rotation
              player (>= 15% of its floor time, any position) going out who is
              worth less than c -- ungated, Denver's 2024 list was Jokic for
              every other star. `replaces` names the legal outgoing r whose
              departure costs least (share(r) * value(r)).
  creator     a high-usage candidate's value, for team T, also gets
              2 * b2 * (k_T - n_bar): k_T creators on T's floor per possession
              (season-t usage, top 20% of rotation players), n_bar the league
              mean, b2 the within-team curvature (weakness_test.py, pooled over
              seasons <= t; -0.20 early, -0.34 by 2025). Teams short on
              creators value one more. It is a tie-breaker in practice:
              5-95% range -0.33..+0.34 pts/100 against a 2.1 sd in value, and
              no measurable change in evaluate_recs.py (quality rho 0.280 with
              and without; realism 0.477 vs 0.476). Kept because it is the one
              team-specific effect with within-team evidence (p < .005).
  P           acquisition.py: gradient-boosted trees on candidate, team and
              pair features, fit on earlier seasons' actual arrivals
              (held-out AUC 0.63; experience and age carry most of it).
Realism weight (evaluate_recs.py --scoring-grid quality --realism-grid; table
from before AGGREGATE, which moved the default to 8 -- see Salary below):
  w          realism   weighted   quality rho
  0          0.514     0.491      0.321
  1          0.492     0.466      0.306
  2          0.476     0.449      0.280
  4          0.453     0.428      0.212
  6          0.444     0.423      0.169     default until the salary-screen fix
  8          0.439     0.421      0.139
  P alone    0.383     0.369     -0.153
  LightFM    0.473     0.431      0.129
w = 2 matches LightFM's headline realism at twice its quality, but its lists
were stars for everyone: 55% of each top 10 were top-5% players by net
impact (82% top-10%), against 3% of actual veteran arrivals. P is well
calibrated by value (top-2% players: 0.44% predicted per team vs 0.35%
actual; bottom half 1.11% vs 1.16%), so it knows stars rarely move -- at
w = 2 that 3x lower P is worth ~2 pts/100, less than a star's value lead.
w = 6 (17% top-5%, 31% top-10%) beats LightFM on every realism view and
still on quality; Chicago 2025 went from Brunson, Murray, LeBron to Murray,
McConnell, Powell. Tightening the value cap instead (gap <= 2 at w = 2:
39% top-5%) costs more quality per star removed, and rules out 35% of real
arrivals instead of 30%. P is not team-context aware -- it learned that
veterans move -- so a rebuilding team's list leans old as w rises.
Why so little team-specific scoring (N1, evaluate_recs.py, candidate pool
without the salary screen): every team-specific adjustment tried lost
signal against ranking on projected value alone --
                                        quality rho   team R^2 gain
  projected value                       0.428         +0.085   (BPM: 0.392, +0.086)
  x candidate's minutes share           0.430         +0.057
  - value at his position (need)        0.340         +0.065
  both (`expected`)                     0.336         +0.049
and requiring the outgoing player to play within one position of the
candidate ruled out 11% of real arrivals instead of 3% (quality rho 0.223 vs
0.345 with the screen). Positional need as measured here does not predict
who helps, consistent with the pair and style tests below.
There is no archetype-pair "fit" term any more: fit_model.py and
team_style.py found pair and style effects indistinguishable from shuffled
labels, so it was noise in the ranking, and its coefficients were pooled over
every season (a leak into earlier recommendations).

Scoring alternatives (--scoring; evaluate_recs.py --scoring-grid), tried because
the team-level value test credited gain-over-r with almost nothing:
  value      value(c) alone      minutes   share_t(c) * value(c)
  displace   share_t(c) * (value(c) - minutes-weighted value at his position)
Scored on c alone, a team could "swap" its own star for an equal one (Embiid
for Giannis is salary- and surplus-legal), which made `value` look best (team
R^2 +0.062 vs +0.012). Requiring an upgrade (value(c) > value(r)) removes that,
and then all three tie random on realism (~0.50) with R^2 gains of 0.000-0.006
-- while ruling out 80% of actual arrivals, which by this value measure are
mostly not upgrades over anyone at their position.
  replace    share(r) * (value(c) - value(r)), max over r: the default until
             evaluate_recs.py's quality test showed it ranks the arrivals who
             delivered more LOWER (rho -0.10 with the salary screen): the max
             over r mostly measures how weak a team's weakest legal r is.
  expected   share_t(c) * (value(c) - v_pos), v_pos the minutes-weighted
             value of the team's rotation within one position of c: what his
             minutes would otherwise go to. No upgrade gate. Default between
             `replace` and `quality`.
  quality    the default, above; no position rule on the outgoing player
             (pos_rule; every other scoring keeps it).
evaluate_recs.py --scoring-grid expected replace (salary screen, gap <= 1):
                       realism   weighted   quality rho   team R^2 gain
  replace, raw value    0.492     0.519      -0.122        +0.012
  replace               0.497     0.524      -0.103        +0.010
  expected              0.504     0.534      -0.005        +0.035
The same at gap <= 3, after E5 (per-season archetype effects, no pair term):
  expected              0.515     0.530      +0.147        +0.035
  quality, pos rule     0.502     0.491      +0.223        +0.070
  quality               0.509     0.478      +0.345        +0.076

Surplus penalty (SURPLUS_PENALTY, evaluate_recs.py variant(surplus_penalty=)):
a soft version of the value cap, score -= lambda * max(0, surplus(c) -
surplus(r)), so overpaid-for-star swaps (LaVine for LeBron, gap 2.88) fade
instead of passing at 3. Null at w = 6 -- it lowers quality without buying
realism, over all seasons and on 2024 decisions alone (n = 114):
                     realism  weighted  quality   2024: realism  quality
  cap 3, lambda 0    0.444    0.423     0.169           0.429    0.091
  cap 3, lambda 1    0.443    0.426     0.137           0.429    0.086
  cap 3, lambda 2    0.444    0.432     0.108           0.433    0.067
  no cap, lambda 1   0.445    0.427     0.148           0.422    0.119
Dropping the salary screen altogether scores better on every line here
(0.427 / 0.385 / 0.187) but brings back Jokic and Giannis for the 2025
Bulls: the evaluator scores where actual arrivals ranked, and stars almost
never arrive, so it cannot see what the screen is for. Off.

Movability cap (MOVE_CAP_PCT). After the review, lists were still alike:
Royce O'Neale on 23 of 30 2025 lists, McConnell on 20, Bogdanovic on 15 --
not for value (ranks 154, 77, 206 of 309) but for being the most likely to
move at all (P ranks 2, 8, 1). Within a team's pool the realism term spread
3.7x as much as value (sd 4.4 vs 1.2), and P(he moves) is a property of the
player, the same on every list. Veto rules could not help (audit.py: the
team-direction candidates all fail), so the term is capped instead: P(c
moves) above the season's MOVE_CAP_PCT quantile earns nothing more, while
below it a higher weight penalises unmovable players harder. Tuning seasons,
plus the 2025 lists (19 = the review's flags still listed):
                     moved  stars  bias   jaccard  2025 jaccard  max lists  O'Neale  flags
  no cap, w 8        0.306  0.119  0.080  0.156    0.170         23         23       19
  cap 0.75, w 8      0.256  0.125  0.178  0.114    0.079         13          3       13
  cap 0.9, w 12      0.299  0.110  0.085  0.126    0.093         13         11       17
  cap 0.8, w 16      0.291  0.109  0.069  0.111    0.069         12          7       14
  cap 0.8, w 20 (default) 0.293  0.107  0.032  0.109  0.067      12          7       14
  cap 0.75, w 20     0.284  0.108  0.059  0.105    0.066         11          3       12
A cap alone trades availability away (journeymen really do move; at 0.75,
w 8 listed players moved barely more than random, 0.24); raising the weight
with it keeps availability while the duplication halves.

Positional fit (FIT_WEIGHT). With value the same for every team and
destinations close to unpredictable, the lists were nearly identical: the
season's 10 most-listed players held 65% of all slots (T.J. McConnell was on
28 of 30 2025 lists). One team-specific quantity has observable support: how
weak a team's rotation is at a candidate's position (v_pos, stay values).
Among real arrivals, more need means more minutes at the new team (t = 3.4,
for a given value and prior minutes), and real destinations rank above the
middle of a player's possible teams by it (0.538, ~5 se). The score gets
FIT_WEIGHT * -v_pos -- the same for every candidate at a position, so it
reorders positions, not players (adding value - v_pos instead re-adds value
and favoured stars: 17% -> 34% at weight 2). Tuning seasons (rubric.py):
                    moved  joined  stars  bias   jaccard  top10 share  max lists 2025
  fit 0             0.329  0.016   0.166  0.042  0.370    0.650        28
  fit 1             0.318  0.015   0.146  0.039  0.226    0.513        27
  fit 2 (default)   0.306  0.016   0.119  0.080  0.156    0.413        23
  fit 4             0.302  0.016   0.094  0.132  0.107    0.329        18
  fit 2, w 6        0.290  0.015   0.134  0.129  0.145    0.394        22
Exact-destination hits (`joined`) are flat across weights, so real moves
neither support nor reject the weight: how different lists should be is a
product choice, set here at the point where availability and bias have
barely moved.

Salary (fetch_salaries.py). Unrestricted, every team's list converges on the
season's best players, whatever they cost. With salary on, a swap must also be
one the team could make:
  outgoing    salary(r) + the AGGREGATE largest other contracts on the roster
              among players worth less than c or out of the rotation
              (trade_packages(); FILLER, the older bench-only version, is 0)
  incoming    salary(c) <= max(MATCH_PCT * outgoing + pad,  outgoing + room)
              The first term is the over-the-cap trade rule (125% + $100k,
              $250k from the 2023 CBA on); the second is a team under the cap
              absorbing salary into its room = max(0, cap - payroll).
  Candidates with no known salary are dropped; a rotation player with none
  sends out $0.
With r's salary alone (AGGREGATE = 0) the screen ruled out 31% of real trades
-- legal by definition -- because real packages bundle salaries: the trades
it failed sent out 2.35 players on average, and 21% sent no rotation player
at all. On the tuning seasons (rubric.py, decisions 1999-2017):
                          ruled out:  trades  signings  moved  stars  bias
  agg 0, w 6, gap 3 (old)            0.306   0.303     0.315  0.134  0.561
  agg 1, w 6, gap 3                  0.140   0.187     0.302  0.228  0.513
  agg 2, w 6, gap 3                  0.126   0.171     0.304  0.234  0.515
  agg 1, w 8, gap 2 (default)        0.147   0.194     0.327  0.173  0.429
Packages are the cheapest legal one, not the largest (which showed T.J.
McConnell's $9.3M "costing" a $2M player plus Jaylen Brown's $49M). Requiring
the package to be legal for the other team as well (PARTNER_MATCH) doubles
the real trades ruled out (12% -> 25%; agg 2: 22%): the model has one
incoming player, while real deals often send several back, so it is off.
The old screen was doing hidden work: once a team can combine two salaries,
stars become affordable (13% -> 23% of list slots, top-5% by BPM), so the
realism weight (6 -> 8) and value cap (3 -> 2) were retuned to hold that
down; together they also cut the winner's-curse bias (0.56 -> 0.43). Most of
what is still ruled out is the upgrade gate (9% of real trades send out no
one worth less than the player coming in -- salary dumps, not upgrades).

Value cap (--max-gap). Salary matching alone lets Cleveland "trade" Ricky
Davis for Dirk Nowitzki, and makes every star on a rookie deal reachable by
everyone. Teams trade on surplus -- value beyond what the contract normally
buys -- so a swap is also ruled out when
  surplus(c) - surplus(r) > max_gap        points per 100
  surplus(p)  value(p) - (a_t + b_t * salary(p) / cap_t), the line fit each
              season by possession-weighted least squares over players with
              >= MARKET_MIN_POSS (season-t data only; a monotone fit chased
              the single highest salary, zeroing Garnett's '07 surplus).
Garnett for Al Jefferson ('07) is a gap of ~0.7; Dirk for Ricky Davis ('03)
is ~6. Real trades close gaps with picks and prospects this cannot see, so
the tolerance is tuned on evaluate_recs.py, not set by hand. On realism
alone (replace scoring), `--gap-grid 0.5 1 2 3 5` moved the headline
percentile from 0.524 (no cap) to 0.490 / 0.492 / 0.503 / 0.510 / 0.517, and
1.0 was chosen. The quality test showed the price: the cap rules out the
arrivals who go on to deliver most. With expected scoring
(`--scoring-grid expected --gap-grid 2 3 5`):
  gap        realism   weighted   quality rho   ruled out   team R^2 gain
  1          0.504     0.534      -0.005        20%         +0.035
  2          0.509     0.533      +0.077        14%         +0.034
  3          0.514     0.529      +0.139        11%         +0.034
  5          0.521     0.532      +0.178        10%         +0.038
  none       0.525     0.535      +0.185        10%         +0.039
The cap's realism comes almost entirely from the unweighted headline (fringe
arrivals); the weighted view does not move. 3.0 keeps half of it and most of
the quality.
It binds trades only: pending free agents are exempt (below).

Free agency (opt-in, --free-agency; off by default -- see the caveat below). A pending free agent is not traded
for; he signs. So he skips salary matching and the value cap, and instead
  fa_cost     his market price for t+1 (fa_prices): next-season cap share
              regressed on value and current cap share, fit on earlier
              seasons' pending free agents who signed, clipped to
              [FA_MIN_PCT, FA_MAX_PCT]. Running the surplus line backwards
              instead priced +2..+4 players at 32% of the cap against an
              actual 16% (mean error 0.062 of cap vs 0.037).
  budget      max(cap_{t+1} - salary still committed, MLE_{t+1}): offseason
              room once the team's own pending free agents come off its books,
              or the mid-level exception every team has regardless.
  reachable   fa_cost <= budget.
Contract status itself is a proxy (pending_fa): no accessible source lists
contract end dates or re-signings for 1999-2025 (BBRef's transaction log omits
most re-signings; Pro Sports Transactions, RealGM and Spotrac refuse scripted
requests). A contract is taken to end after t when the player's t+1 salary is
outside FA_BAND x his t salary, or he has none -- a new deal, whether he
re-signs or moves, breaks the fixed-raise schedule. That reads t+1 data, but
the same way for players who stay and players who leave, so it says "his
contract was up", not "he moved". Against BBRef's transaction labels it flags
83% of free-agent signings and 41% of traded players (AUC 0.80).

An *extension* breaks the schedule too, and extensions go mostly to stars
(Durant '10 and Cunningham '25 on rookie deals; Harden '16, Mitchell '25 as
veterans). Unchecked, 99% of recommendations arrived "via free agency" and the
stars came back. Two corrections:
  rookie scale  first-round picks are never flagged through draft year +
                ROOKIE_YEARS: on the rookie deal, then restricted free agents
                whose teams match offers.
  p_avail       (availability) P(a flagged player actually changes teams),
                gradient-boosted trees on age, value, cap share, experience,
                floor-time share, whether he was traded mid-season, and his
                team's net rating, fit on earlier seasons' flagged players
                only. Held-out AUC 0.63 (a logistic fit: 0.56 -- stars stay far
                more often than a linear term in value allows: 14% of flagged
                +4s moved vs 38% overall). Weak, but it is the signal there is. A
                free agent's score is gain * p_avail: the expected gain of
                pursuing him, so a star likely to re-sign or extend ranks
                below a lesser player likely to leave. This deliberately ignores the CBA's finer tiers (150-200%
  matching for small salaries, apron limits) and sign-and-trades; it is a
  "could this work on salary at all" screen, not a trade machine.

Usage:
    python impact_recs.py              # write data/recs_impact.csv (+ _open)
    python impact_recs.py --compare    # vs. LightFM (data/recs_lightfm.csv)
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

import impact_model as im
from fetch_salaries import CAP, MLE

SLOTS = im.SLOTS
K = 10
MIN_SHARE = 0.15        # rotation players a candidate can replace
MIN_CAND_POSS = 1500    # offensive possessions; below this impact is mostly prior
POS_WINDOW = 1.0        # |pos_index difference| allowed, 1 = PG ... 5 = C
MATCH_PCT = 1.25        # incoming salary per $ outgoing, over the cap
FILLER = 0              # non-rotation contracts a team may add to the outgoing side
AGGREGATE = 1           # extra lesser players a trade may add to r's outgoing salary
FIT_WEIGHT = 2.0        # score += this * how weak the team is at his position
PARTNER_MATCH = False   # also require the package to be legal for the other team (see Salary)
MAX_GAP = 2.0           # surplus(c) - surplus(r) allowed, pts/100; None = no value cap
SURPLUS_PENALTY = 0.0   # score -= this * max(0, surplus(c) - surplus(r)); 0 = off
SCORINGS = ("replace", "value", "minutes", "displace", "expected", "quality")
SCORING = "quality"
MARKET_MIN_POSS = 500   # players the salary -> value line is fit on
FA_BAND = (0.97, 1.13)  # t+1 / t salary on one contract; outside it, a new deal
FA_MIN_PCT, FA_MAX_PCT = 0.01, 0.35     # a free agent's price, as share of the cap
ROOKIE_YEARS = 4        # first-round rookie scale seasons; RFA right after
AVAIL_MIN_HISTORY = 2   # seasons of flagged players needed before p_avail is fit
SOFT_PENALTY = 1e3      # soft=True: illegal moves rank below every legal one
ARCHETYPES = False      # add fit_model.py's archetype main effects to value (see docstring)
CREATOR_CURVATURE = os.path.join("data", "creator_curvature.csv")
CREATOR_TERM = True     # add weakness_test.py's creator curvature to value
CREATOR_MIN_HISTORY = 3 # seasons pooled before the curvature is used
REALISM_WEIGHT = 20.0   # score += this * log P(acquired) (acquisition.py); 0 = off
MOVE_CAP_PCT = 0.8      # cap P(he moves at all) at this quantile of the season's pool; 1 = no cap
PROJ_AGE = False        # age terms in the legacy linear projection (see project())
PROJ_METHOD = "mover"   # "mover" (project_mover()) or "linear" (project())
PROJ_REL_K = 3000.0     # possessions at which a season's estimate counts as half-reliable
PROJ_BOX = ["bpm", "obpm", "dbpm", "usg_percent", "ts_percent", "ws_48"]
PROJ_PEAK_AGE = 27      # age terms are centred here
PROJ_MIN_POSS = 250     # both seasons of a projection training pair
PROJ_MIN_HISTORY = 3    # season pairs needed before the projection is fit

# Franchise continuity where the NBA's team id does not provide it: Charlotte's
# 1999-2002 Hornets carry 1610612766 (the history went back to Charlotte),
# while the same roster plays 2003 in New Orleans as 1610612740. Keyed by the
# t+1 season. Seattle -> OKC, New Jersey -> Brooklyn and Vancouver -> Memphis
# keep their ids.
PREDECESSOR = {(2003, 1610612740): 1610612766}


def as_of_t(df):
    """Rows keyed (t, TeamId of t+1) -> the same franchise's season-t TeamId."""
    key = list(zip(df["season"] + 1, df["TeamId"]))
    return df.assign(TeamId=[PREDECESSOR.get(k, k[1]) for k in key])


def match_pad(season):
    return 250_000 if season >= 2024 else 100_000


def load(paths):
    panel = pd.read_csv(paths.panel, usecols=SLOTS + ["season", "TeamId", "OffPoss", "DefPoss"])
    impact = pd.read_csv(paths.impact)
    main = main_effects(pd.read_csv(paths.fit)) if ARCHETYPES else pd.Series(dtype=float)
    players = pd.read_csv(paths.players, usecols=[
        "player_id", "season", "player", "cluster_label", "pos_index", "age", "bpm"])
    teams = pd.read_csv(paths.teams, usecols=["season", "TeamId", "user_id", "TeamAbbreviation",
                                              "net_rtg"])
    salaries = pd.read_csv(paths.salaries, usecols=["player_id", "season", "salary"])
    payroll = pd.read_csv(paths.payroll)
    rosters = pd.read_csv(paths.rosters)
    return panel, impact, main, players, teams, salaries, (payroll, rosters)


def load_ages(path):
    """(player_id, season) -> age, from the full BBRef universe: players.csv /
    master_clustered.csv miss 17% of impact rows."""
    adv = pd.read_csv(path, usecols=["player_id", "season", "age"])
    return adv.drop_duplicates(["player_id", "season"]).set_index(["player_id", "season"])["age"]


def _proj_features(x, x_prev, age, use_age=PROJ_AGE):
    has_prev = ~np.isnan(x_prev)
    cols = [np.ones(len(x)), x, np.where(has_prev, x_prev, x), has_prev.astype(float)]
    if use_age:
        a = np.clip(np.nan_to_num(age, nan=PROJ_PEAK_AGE), 19, 38) - PROJ_PEAK_AGE
        cols += [a, a ** 2, x * a]
    return np.column_stack(cols)


def project(impact, ages=None, use_age=PROJ_AGE):
    """Add proj_impact: net impact expected in t+1, from season-t information.

    A season's estimate is one noisy draw, and a recommendation is for the
    season after. Per season t, weighted least squares of net_impact(s+1) on
        net_impact(s), net_impact(s-1) (or s again, flagged)
        [+ age, age^2, net_impact(s) x age   with use_age]
    over every pair with s+1 <= t (expanding window: season t never sees its
    own future), both seasons >= PROJ_MIN_POSS offensive possessions, weighted
    by the smaller of the two. Seasons with fewer than PROJ_MIN_HISTORY pairs
    of history keep the raw estimate.

    Next-season net impact, players >= 1500 possessions, 2004-2025 held out
    season by season (R^2): raw 0.490, shrunk only 0.554, + previous season
    0.576, + age 0.568, all 0.599.

    Why age is off: evaluate_recs.py's team-level test (net-rating change
    explained by arrivals and departures) rejects it. Summing projected value
    instead of raw, R^2 changes by -0.029 [-0.042, -0.014] with age and
    +0.000 [-0.012, 0.012] without; shrink-only is -0.006. Fitting the age
    curve without conditioning on t+1 minutes (a survivorship suspect) does
    not change that. Players do age (+0.4 a season at 18-23, -0.4 to -0.6
    from 30, among those who keep playing), but that has not shown up in
    team outcomes yet, so it is not used for ranking.
    """
    est = impact.set_index(["player_id", "season"])
    x = est["net_impact"]
    poss = est["off_poss"]
    shift = lambda s, k: s.set_axis(pd.MultiIndex.from_arrays(          # noqa: E731
        [s.index.get_level_values(0), s.index.get_level_values(1) + k]))
    x_prev = shift(x, 1).reindex(est.index).to_numpy()
    y = shift(x, -1).reindex(est.index).to_numpy()
    poss_next = shift(poss, -1).reindex(est.index).to_numpy()
    age = ages.reindex(est.index).to_numpy(float) if ages is not None else np.full(len(est), np.nan)
    X = _proj_features(x.to_numpy(), x_prev, age, use_age)
    season = est.index.get_level_values("season").to_numpy()
    w = np.minimum(poss.to_numpy(), poss_next)
    pair = ~np.isnan(y) & (w >= PROJ_MIN_POSS)

    proj = x.to_numpy().copy()
    for t in np.unique(season):
        train = pair & (season + 1 <= t)
        if len(np.unique(season[train])) < PROJ_MIN_HISTORY:
            continue
        sw = np.sqrt(w[train])
        beta = np.linalg.lstsq(X[train] * sw[:, None], y[train] * sw, rcond=None)[0]
        proj[season == t] = X[season == t] @ beta
    return impact.assign(proj_impact=proj)


def move_status(member, min_share=0.05):
    """(player_id, season t) -> where he was in t+1, from lineup memberships.

      moved    >= min_share of some team's floor time in t+1, on no team he
               played for in t
      stayed   >= min_share for one of his season-t teams
      gone     under min_share everywhere: out of the league or the rotation
    share_new is his summed share of the new teams' floor time (moved only).
    The last season has no t+1 and is left out.
    """
    key = ["player_id", "season"]
    now = member[key + ["TeamId"]].drop_duplicates()
    nxt = member[member["share"] >= min_share].assign(season=member["season"] - 1)
    nxt = nxt.merge(now.assign(old=True), on=key + ["TeamId"], how="left")
    nxt["old"] = nxt["old"].fillna(False).astype(bool)
    g = nxt.groupby(key)
    played = g.size().rename("n")
    stayed = g["old"].any()
    share_new = nxt[~nxt["old"]].groupby(key)["share"].sum()
    out = now[key].drop_duplicates().set_index(key)
    out = out[out.index.get_level_values("season") < member["season"].max()]
    out["status"] = "gone"
    hit = played.reindex(out.index).notna()
    out.loc[hit, "status"] = np.where(stayed.reindex(out.index[hit]), "stayed", "moved")
    out["share_new"] = share_new.reindex(out.index).where(out["status"] == "moved")
    return out


def projection_inputs(adv_path, panel):
    """Per (player_id, season): BBRef box score (season totals), age, and
    whether he moved for t+1 (move_status) -- what project_mover() reads."""
    from helper import season_totals
    adv = pd.read_csv(adv_path, usecols=["season", "player_id", "team", "age", "mp"] + PROJ_BOX)
    adv = season_totals(adv[adv["mp"] > 0]).drop(columns=["team", "mp"])
    status = move_status(memberships(panel))["status"]
    out = adv.set_index(["player_id", "season"])
    out["moved"] = (status.reindex(out.index) == "moved").astype(float)
    return out


def project_mover(impact, inputs):
    """Add proj_impact (net impact expected in t+1 if he changes teams) and
    proj_stay (if he stays), from season-t information.

    A recommendation is a move, and movers do worse than their estimates
    say: teams let go of the players they expect to decline. Fit as
    project()'s linear one was -- weighted least squares of net_impact(s+1),
    per season t over pairs with s+1 <= t, weighted by the smaller season's
    possessions -- on
      net_impact(s), net_impact(s-1) (or s again, flagged), each also times
          its reliability poss / (poss + PROJ_REL_K), and the reliability
      BPM, OBPM, DBPM, BPM(s-1), usage, TS%, WS/48           (PROJ_BOX)
      age - 27, its square, net_impact(s) x (age - 27)
      moved for s+1, and moved x {net_impact(s), age - 27, net_impact(s-1)}
    proj_impact sets moved = 1, proj_stay moved = 0. On decisions 2002-2017
    (players >= 1500 possessions; rubric.SPLIT, the holdout unseen), against
    the legacy linear projection:
                                     R^2 t+1   R^2     movers: rho with   bias,  bias, top
                                     (all)     movers  BPM t+1  impact    movers 10% movers
      legacy linear (project())      0.623     0.345   0.525    0.500     0.279  1.004
      + reliability                  0.630     0.353   0.523    0.492     0.212  0.921
      + box score                    0.639     0.360   0.536    0.500     0.230  0.965
      + age                          0.657     0.374   0.550    0.507     0.155  0.858
      + full mover interactions      0.608     0.397   0.531    0.494     0.053  0.402
      + these mover terms (default)  0.623     0.404   0.543    0.500     0.043  0.322
      gradient boosting, all + age   0.638     0.356   0.542    0.501     0.192  1.148
      BPM alone                        -         -     0.512    0.474       -      -
    The top movers by the legacy projection fell a full point per 100 short
    of it (the winner's curse, sharpened by adverse selection); conditioning
    on the move cuts that to a third. BPM t+1 is the independent check.
    Age is in: the team-level test that rejected it (see project()) is no
    longer how the recommender is graded.
    """
    key = ["player_id", "season"]
    est = impact.set_index(key)
    inp = inputs.reindex(est.index)
    x = est["net_impact"].to_numpy(float)
    poss = est["off_poss"].to_numpy(float)
    shift = lambda s, k: s.set_axis(pd.MultiIndex.from_arrays(          # noqa: E731
        [s.index.get_level_values(0), s.index.get_level_values(1) + k]))
    x_prev = shift(est["net_impact"], 1).reindex(est.index).to_numpy(float)
    poss_prev = np.nan_to_num(shift(est["off_poss"], 1).reindex(est.index).to_numpy(float))
    bpm_prev = shift(inp["bpm"], 1).reindex(est.index).to_numpy(float)
    y = shift(est["net_impact"], -1).reindex(est.index).to_numpy(float)
    poss_next = shift(est["off_poss"], -1).reindex(est.index).to_numpy(float)

    has_prev = ~np.isnan(x_prev)
    xp = np.where(has_prev, x_prev, x)
    rel, rel_prev = poss / (poss + PROJ_REL_K), poss_prev / (poss_prev + PROJ_REL_K)
    a = np.clip(np.nan_to_num(inp["age"].to_numpy(float), nan=PROJ_PEAK_AGE), 19, 38) - PROJ_PEAK_AGE
    bpm = inp["bpm"].to_numpy(float)
    box = [np.nan_to_num(inp[c].to_numpy(float)) for c in PROJ_BOX]
    base = [x, xp, has_prev.astype(float), x * rel, xp * rel_prev, rel] + box + \
        [np.nan_to_num(np.where(np.isnan(bpm_prev), bpm, bpm_prev)), a, a ** 2, x * a]
    few = [x, a, xp]
    ones = np.ones(len(x))

    def design(moved):
        return np.column_stack([ones, moved] + base + [f * moved for f in few])

    X = design(inp["moved"].fillna(0.0).to_numpy(float))
    X_mv, X_st = design(ones), design(np.zeros(len(x)))
    season = est.index.get_level_values("season").to_numpy()
    w = np.minimum(poss, poss_next)
    pair = ~np.isnan(y) & (w >= PROJ_MIN_POSS)

    proj, stay = x.copy(), x.copy()
    for t in np.unique(season):
        train = pair & (season + 1 <= t)
        if len(np.unique(season[train])) < PROJ_MIN_HISTORY:
            continue
        sw = np.sqrt(w[train])
        beta = np.linalg.lstsq(X[train] * sw[:, None], y[train] * sw, rcond=None)[0]
        here = season == t
        proj[here], stay[here] = X_mv[here] @ beta, X_st[here] @ beta
    return impact.assign(proj_impact=proj, proj_stay=stay)


def with_projection(impact, adv_path, panel):
    """impact with its t+1 projection, by PROJ_METHOD."""
    if PROJ_METHOD == "mover":
        return project_mover(impact, projection_inputs(adv_path, panel))
    return project(impact, load_ages(adv_path))


def pending_fa(salaries, draft=None):
    """(player_id, season) -> True if his contract looks to end after that season.

    `draft` (Draft Pick History.csv) exempts first-round picks through draft
    year + ROOKIE_YEARS.
    """
    s = salaries.set_index(["player_id", "season"])["salary"]
    nxt = salaries.assign(season=salaries["season"] - 1).set_index(["player_id", "season"])["salary"]
    ratio = nxt.reindex(s.index) / s.where(s > 0)
    flag = ~ratio.between(*FA_BAND)          # NaN (no t+1 salary) -> True
    if draft is not None:
        first = draft[(draft["lg"] == "NBA") & (draft["round"] == 1)]
        drafted = first.drop_duplicates("player_id").set_index("player_id")["season"]
        # 3-year rookie scale for drafts through 1998 (1995 CBA), 4 after.
        until = drafted + np.where(drafted <= 1998, ROOKIE_YEARS - 1, ROOKIE_YEARS)
        until = until.reindex(s.index.get_level_values("player_id")).to_numpy()
        flag &= ~(s.index.get_level_values("season").to_numpy() <= until)   # NaN -> False
    return flag[s.index.get_level_values("season") < max(CAP)]


def memberships(panel):
    """(season, TeamId, player_id, share) for everyone who appears in a lineup."""
    return pd.concat([rotation(df)[0].assign(season=season)
                      for season, df in panel.groupby("season")], ignore_index=True)


def availability(ptabs, member, debut, teams):
    """Add p_avail to every table in {season: player_table} with flagged players.

    Label for season s: a flagged player (>= MARKET_MIN_POSS) appears in s+1
    and on none of his season-s teams. Retiring counts as not available.
    Fit on seasons < t only; until AVAIL_MIN_HISTORY seasons exist, the pooled
    rate so far (0.5 with none).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    on = member[member["share"] > 0].groupby(["player_id", "season"])["TeamId"].apply(set)
    nxt = as_of_t(member[member["share"] > 0].assign(season=member["season"] - 1)) \
        .groupby(["player_id", "season"])["TeamId"].apply(set)
    main_team = member.sort_values("share").drop_duplicates(["player_id", "season"], keep="last") \
        .set_index(["player_id", "season"])["TeamId"]
    share = member.groupby(["player_id", "season"])["share"].sum()
    n_teams = member[member["share"] > 0].groupby(["player_id", "season"])["TeamId"].nunique()
    net = teams.set_index(["season", "TeamId"])["net_rtg"]
    cols = ["age", "value", "cap_pct", "experience", "team_net", "share", "traded"]

    def features(t, season):
        idx = pd.MultiIndex.from_arrays([t.index, [season] * len(t)])
        team = main_team.reindex(idx).to_numpy()
        f = pd.DataFrame({
            "age": t["age"].to_numpy(), "value": t["value"].to_numpy(),
            "cap_pct": t["cap_pct"].to_numpy(),
            "experience": season - debut.reindex(t.index).to_numpy(),
            "team_net": net.reindex(pd.MultiIndex.from_arrays([[season] * len(t), team])).to_numpy(),
            "share": share.reindex(idx).to_numpy(),
            "traded": (n_teams.reindex(idx).fillna(1).to_numpy() > 1).astype(float),
        }, index=t.index)
        return f.fillna(f.median())

    history = []
    for season in sorted(ptabs):
        t = ptabs[season]
        f = features(t, season)
        if len(history) >= AVAIL_MIN_HISTORY:
            h = pd.concat(history)
            model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                   min_samples_leaf=40, random_state=0)
            model.fit(h[cols], h["moved"])
            t["p_avail"] = model.predict_proba(f[cols])[:, 1]
        else:
            t["p_avail"] = pd.concat(history)["moved"].mean() if history else 0.5
        flagged = t["pending_fa"] & (t["off_poss"] >= MARKET_MIN_POSS)
        idx = pd.MultiIndex.from_arrays([t.index[flagged], [season] * int(flagged.sum())])
        before, after = on.reindex(idx), nxt.reindex(idx)
        moved = [isinstance(a, set) and not (a & b) for a, b in zip(after, before)]
        if season + 1 in ptabs or season + 1 < max(CAP):
            history.append(f[flagged].assign(moved=np.array(moved, dtype=float)))
    return ptabs


def team_room(teams, books, flags=None):
    """Per (season, TeamId): in-season cap room, and (given `flags`) the offseason
    free-agent budget. Joins pbpstats' abbreviations to BBRef's."""
    from lineup_features import bbref_team
    payroll, rosters = books
    t = teams.assign(team=[bbref_team(a, s) for a, s in
                           zip(teams["TeamAbbreviation"], teams["season"])])
    t = t.merge(payroll, on=["season", "team"], how="left", validate="1:1")
    if t["payroll"].isna().any():
        raise SystemExit(f"no payroll for {t.loc[t['payroll'].isna(), ['season', 'team']].values[:5]}")
    t["room"] = (t["cap"] - t["payroll"]).clip(lower=0)
    if flags is not None:
        r = rosters.join(flags.rename("fa"), on=["player_id", "season"])
        kept = r[~r["fa"].fillna(True).astype(bool)].groupby(["season", "team"])["salary"].sum()
        t = t.join(kept.rename("committed"), on=["season", "team"]).fillna({"committed": 0.0})
        nxt = t["season"] + 1
        t["fa_budget"] = np.maximum(nxt.map(CAP) - t["committed"], nxt.map(MLE))
    return t[["season", "TeamId", "room"] + (["fa_budget"] if flags is not None else [])]


def main_effects(by_season):
    """(season, archetype) -> net main effect, from fit_model.py's per-season file."""
    return by_season.set_index(["season", "archetype"])["net_effect"]


def archetype_effect(main, seasons, labels):
    """Each row's archetype main effect as of its own season (missing -> 0;
    all zero unless ARCHETYPES)."""
    if main.empty:
        return np.zeros(len(labels))
    idx = pd.MultiIndex.from_arrays([np.asarray(seasons), pd.Series(labels).fillna("unknown")])
    return main.reindex(idx).fillna(0.0).to_numpy()


def creator_inputs(data_dir="data", curvature=CREATOR_CURVATURE):
    """Creator flags ((player_id, season) -> bool, weakness_test.py's main cut,
    judged on that season's box score) and b2 as of each season."""
    import weakness_test as wt
    f = wt.skill_flags(wt.load_box(data_dir), "creator", wt.SKILLS["creator"][2])
    flags = f.drop_duplicates(["player_id", "season"]).set_index(["player_id", "season"])["flag"]
    b2 = pd.read_csv(curvature).set_index("season")["b2"].dropna()
    # One season's curvature is noise (1999 alone: +0.19); wait for history.
    b2 = b2[b2.index >= b2.index.min() + CREATOR_MIN_HISTORY - 1]
    return flags, b2


def player_table(season, impact, players, main, salaries, flags=None, creators=None):
    """value = net impact (projected to t+1 when impact has proj_impact) + the
    archetype main effect."""
    est = impact[impact["season"] == season].set_index("player_id")
    meta = players[players["season"] == season].set_index("player_id")
    sal = salaries[salaries["season"] == season].set_index("player_id")["salary"]
    cols = ["net_impact", "o_impact", "d_impact", "off_poss"] + \
        [c for c in ("proj_impact", "proj_stay") if c in est]
    t = est[cols].join(
        meta[["player", "cluster_label", "pos_index", "age", "bpm"]], how="left").join(sal)
    t["cluster_label"] = t["cluster_label"].fillna("unknown")
    base = t["proj_impact"] if "proj_impact" in t else t["net_impact"]
    arch = archetype_effect(main, np.full(len(t), season), t["cluster_label"])
    t["value"] = base + arch
    # What he is worth to the team he is on now: the side of a swap that stays
    # behind when someone else is the one sent out (project_mover()).
    t["value_stay"] = (t["proj_stay"] + arch) if "proj_stay" in t else t["value"]
    t["cap_pct"] = t["salary"] / CAP[season]
    fit = t[(t["off_poss"] >= MARKET_MIN_POSS) & t["cap_pct"].notna()]
    b, a = np.polyfit(fit["cap_pct"], fit["value"], 1, w=np.sqrt(fit["off_poss"]))
    t["surplus"] = t["value"] - (a + b * t["cap_pct"])
    if flags is not None:
        t["pending_fa"] = flags.xs(season, level="season").reindex(t.index).fillna(False).astype(bool)
    if creators is not None:
        t["creator"] = creators.xs(season, level="season").reindex(t.index).fillna(False).astype(bool)
    return t


def fa_prices(ptabs, salaries):
    """Add fa_cost ($, for season t+1) to every table in {season: player_table}.

    Next-season cap share ~ value + current cap share, fit by least squares on
    the pending free agents of seasons before t who signed a t+1 contract
    (expanding window, so season t never sees its own outcomes). The first
    season has no history and is priced at its current cap share.
    """
    nxt = salaries.assign(season=salaries["season"] - 1).set_index(["player_id", "season"])["salary"]
    train = []
    for season in sorted(ptabs):
        t = ptabs[season]
        X = np.column_stack([np.ones(len(t)), t["value"], t["cap_pct"].fillna(0.0)])
        if train:
            h = pd.concat(train)
            beta = np.linalg.lstsq(h[["one", "value", "cap_pct"]], h["y"], rcond=None)[0]
            pct = X @ beta
        else:
            pct = t["cap_pct"].fillna(FA_MIN_PCT).to_numpy()
        t["fa_cost"] = np.clip(pct, FA_MIN_PCT, FA_MAX_PCT) * CAP[season + 1]
        done = t[t["pending_fa"] & (t["off_poss"] >= MARKET_MIN_POSS) & t["cap_pct"].notna()]
        y = nxt.reindex(pd.MultiIndex.from_arrays([done.index, [season] * len(done)])).to_numpy()
        train.append(pd.DataFrame({"one": 1.0, "value": done["value"].to_numpy(),
                                   "cap_pct": done["cap_pct"].to_numpy(),
                                   "y": y / CAP[season + 1]}).dropna())
    return ptabs


def rotation(df):
    """On-floor share per (TeamId, player), and per-player teammate archetype mix."""
    w = (df["OffPoss"] + df["DefPoss"]).to_numpy(float)
    long = pd.DataFrame({
        "TeamId": np.repeat(df["TeamId"].to_numpy(), 5),
        "player_id": df[SLOTS].to_numpy().ravel(),
        "w": np.repeat(w, 5),
        "lineup": np.repeat(np.arange(len(df)), 5),
    })
    team_w = df.assign(w=w).groupby("TeamId")["w"].sum()
    share = long.groupby(["TeamId", "player_id"])["w"].sum() / 5
    share = share / team_w.reindex(share.index.get_level_values(0)).to_numpy() * 5
    return share.rename("share").reset_index(), long, team_w


def trade_packages(cv, c_sal, r_sal, r_idx, ros_sal, ros_val, ros_rot, k, room, pad,
                   filler=0.0, partner=True):
    """Which (candidate, rotation player r) swaps some salary-legal package
    makes possible, and the cheapest such package.

    A package is r plus up to k other roster players who are worth less than
    the candidate or are out of the rotation -- the ones a team would put in
    a deal for him -- plus `filler` dollars. Sending out `out` is legal for
    this team when c_sal <= max(MATCH_PCT * out + pad, out + room), and (with
    `partner`) for the other team, assumed over the cap, when out <= MATCH_PCT
    * c_sal + pad: it cannot absorb a package far bigger than what it sends.
    Of the legal packages, the one sending out the least salary is chosen.

    cv, c_sal (n_c,) candidates' value and salary; r_sal, r_idx (n_r,) the
    rotation players' salaries and positions in the roster arrays ros_*
    (n_ros,). Missing salaries count as $0. Returns ok (n_c, n_r) bool, pick
    (n_c, n_r) index into combos, combos: tuples of roster positions going
    out besides r, and out_pick (n_c, n_r): the salary that package sends
    out (inf where illegal).
    """
    n_ros = len(ros_sal)
    sal = np.nan_to_num(np.asarray(ros_sal, dtype=float))
    combos = [()] + [c for m in range(1, k + 1) for c in combinations(range(n_ros), m)]
    member = np.zeros((len(combos), n_ros), dtype=bool)
    for j, c in enumerate(combos):
        member[j, list(c)] = True
    csum = member @ sal
    elig = (ros_val[None, :] < cv[:, None]) | ~ros_rot[None, :]                   # (n_c, n_ros)
    combo_ok = ~(member[None, :, :] & ~elig[:, None, :]).any(-1)                   # (n_c, n_combo)
    has_r = member[:, r_idx].T                                                     # (n_r, n_combo)
    out = np.nan_to_num(np.asarray(r_sal, dtype=float))[:, None] + csum[None, :] + filler
    c = np.asarray(c_sal, dtype=float)[:, None, None]
    legal = c <= np.maximum(MATCH_PCT * out + pad, out + room)[None, :, :]
    if partner:
        legal &= out[None, :, :] <= MATCH_PCT * c + pad
    legal &= combo_ok[:, None, :] & ~has_r[None, :, :]
    cost = np.where(legal, np.broadcast_to(out[None, :, :], legal.shape), np.inf)
    pick = cost.argmin(-1)
    return np.isfinite(cost.min(-1)), pick, combos, cost.min(-1)


def recommend_season(season, df, ptab, teams, room=None, filler=FILLER, k=K,
                     min_poss=MIN_CAND_POSS, max_gap=MAX_GAP, fa_budget=None, scoring=SCORING,
                     pos_rule=None, soft=False, p_acq=None, realism=None, creator_b2=None,
                     surplus_penalty=None, aggregate=None, fit_weight=None, partner_match=None,
                     move_cap=None):
    """Top-k per team. `room` (cap room by TeamId) turns the salary screen on;
    `fa_budget` (by TeamId) lets ptab's pending free agents in on free-agent terms.
    `scoring` is one of SCORINGS; the salary and value screens decide who is
    eligible either way. `pos_rule`: must a trade send out a rotation player
    within POS_WINDOW of the candidate? Default: yes, except for `quality`.

    k=None ranks every candidate instead, keeping those with no legal swap
    at score -inf (evaluate_recs.py needs where an actual acquisition fell).
    With soft=True they are ordered among themselves instead, below every
    legal candidate: best gain over any r, minus SOFT_PENALTY. A move this
    screen cannot see (picks, a third team) is still more likely for a
    better player; ranking them all tied last throws that away. The top-k
    lists are unchanged: only legal candidates are listed.

    p_acq ((TeamId, player_id) -> P(acquired), acquisition.py) adds
    realism * log P to every candidate's score (default REALISM_WEIGHT).
    creator_b2 (this season's curvature, weakness_test.py) adds the creator
    term to ptab["creator"] candidates' value for each team; None = off.

    surplus_penalty (default SURPLUS_PENALTY; needs the salary screen) takes
    lambda * max(0, surplus(c) - surplus(r)) off each legal swap, so a trade
    the other side would have to be talked into fades down the list instead
    of passing or failing at max_gap. `replaces` is then the legal r with the
    smallest penalty, ties broken by the least departure cost as before.

    aggregate (default AGGREGATE): how many more players a trade may send out
    with r to match salary (trade_packages()); 0 = r's salary alone.
    partner_match (default PARTNER_MATCH): also require the package to be
    legal for the other team. `with` names who goes out besides r.

    move_cap (default MOVE_CAP_PCT): log P(T acquires c) = log P(c moves) +
    log P(T | c moves), P(c moves) being his P summed over every team. The
    first part is a property of the player, the same on every list; above
    the move_cap quantile of the season's candidates it earns nothing more,
    so "available enough" stops deciding the order. Below it, it penalises
    as before (stars who will not move). `log_p_acq` stays the raw log P.


    fit_weight (default FIT_WEIGHT) adds fit_weight * -v_pos to the score,
    v_pos the minutes-weighted stay value of the team's rotation within
    POS_WINDOW of him (what his minutes would otherwise go to): the same for
    every candidate at a position, so it reorders positions, not players.
    `need` = value(c) - v_pos is reported for explanations, not added to
    `gain`. (Adding need itself mostly re-adds value(c) and favours stars.)
    """
    share, _, _ = rotation(df)
    on_team = share.groupby("TeamId")["player_id"].apply(set)
    cap_q = MOVE_CAP_PCT if move_cap is None else move_cap
    log_move, move_ceiling = None, None
    if p_acq is not None and cap_q < 1.0:
        moves_any = p_acq.groupby(level=1).sum()                 # by player_id
        log_move = np.log(moves_any.clip(lower=1e-9))
        pool = ptab.index[ptab["off_poss"] >= min_poss]
        move_ceiling = float(np.nanquantile(log_move.reindex(pool).to_numpy(), cap_q))
    n_agg = AGGREGATE if aggregate is None else aggregate
    rosters = {t: g for t, g in share.groupby("TeamId")} if room is not None else {}

    cands = ptab[ptab["off_poss"] >= min_poss]
    if room is not None:
        cands = cands[cands["salary"].notna()]
        sal = ptab["salary"]
        bench = share[share["share"] < MIN_SHARE].assign(salary=lambda d: d["player_id"].map(sal))
        filler_sal = bench.groupby("TeamId")["salary"].apply(
            lambda s: s.nlargest(filler).sum() if filler else 0.0)
    c_ids = cands.index.to_numpy()
    c_val = cands["value"].to_numpy()
    c_pos = cands["pos_index"].to_numpy()
    # Floor-time share he carried in season t, across teams if traded.
    c_share = share.groupby("player_id")["share"].sum().clip(upper=1.0) \
        .reindex(c_ids).fillna(0.0).to_numpy()

    # Creator curvature (weakness_test.py): one more high-usage player is
    # worth 2 * b2 * (k - n_bar) more to a team whose lineups average k of
    # them than to the average one; b2 < 0, so teams short on creators gain.
    k_team, n_bar = {}, 0.0
    c_creator = np.zeros(len(cands))
    if creator_b2 and "creator" in ptab:
        flag = ptab["creator"].astype(float)
        n_cre = flag.reindex(df[SLOTS].to_numpy().ravel()).fillna(0.0).to_numpy().reshape(-1, 5).sum(1)
        w_l = (df["OffPoss"] + df["DefPoss"]).to_numpy(float)
        n_bar = np.average(n_cre, weights=w_l)
        k_team = (pd.DataFrame({"TeamId": df["TeamId"].to_numpy(), "kw": n_cre * w_l, "w": w_l})
                  .groupby("TeamId").sum().eval("kw / w").to_dict())
        c_creator = cands["creator"].astype(float).to_numpy()

    out = []
    for team_id, rot in share[share["share"] >= MIN_SHARE].groupby("TeamId"):
        rot = rot.set_index("player_id")
        r_ids = rot.index.to_numpy()
        r = ptab.reindex(r_ids)
        # r's worth to this team if he stays (the candidate's value is as a mover).
        r_val = r["value_stay" if "value_stay" in r else "value"].fillna(0.0).to_numpy()
        r_pos = r["pos_index"].to_numpy()
        pos_win = np.abs(c_pos[:, None] - r_pos[None, :]) <= POS_WINDOW
        pos_win |= np.isnan(c_pos)[:, None] | np.isnan(r_pos)[None, :]
        rule = (scoring != "quality") if pos_rule is None else pos_rule
        pos_ok = pos_win if rule else np.ones_like(pos_win)
        r_share = rot["share"].to_numpy()
        adj = c_creator * creator_b2 * 2 * (k_team.get(team_id, n_bar) - n_bar) if creator_b2 else 0.0
        cv = c_val + adj
        if scoring in ("value", "minutes", "displace", "quality"):
            # Scored on c alone, so the swap must be an upgrade, or a team
            # could "trade" its own star for an equal one (Embiid for Giannis;
            # ungated, Denver's 2024 list was Jokic for every other star).
            pos_ok = pos_ok & (cv[:, None] > r_val[None, :])
        if scoring == "replace":
            gain = r_share[None, :] * (cv[:, None] - r_val[None, :])
        else:
            if scoring in ("value", "quality"):
                cs = cv
            elif scoring == "minutes":
                cs = c_share * cv
            else:   # displace / expected: vs the minutes-weighted value at his position
                w = (pos_ok if scoring == "displace" else pos_win) * r_share[None, :]
                disp = (w @ r_val) / np.where(w.sum(1) > 0, w.sum(1), np.nan)
                disp = np.where(np.isnan(disp), np.average(r_val, weights=r_share), disp)
                cs = c_share * (cv - disp)
            gain = np.repeat(cs[:, None], len(r_ids), axis=1)
        ok = pos_ok
        fa_total = None
        pen = 0.0
        lam = SURPLUS_PENALTY if surplus_penalty is None else surplus_penalty
        if room is not None:
            ros = rosters[team_id]
            ros_ids = ros["player_id"].to_numpy()
            pos_of = {p: i for i, p in enumerate(ros_ids)}
            legal, pick, combos, out_pick = trade_packages(
                cv, cands["salary"].to_numpy(), r["salary"].to_numpy(),
                np.array([pos_of[p] for p in r_ids]), ptab["salary"].reindex(ros_ids).to_numpy(),
                ptab["value_stay" if "value_stay" in ptab else "value"].reindex(ros_ids).to_numpy(),
                (ros["share"] >= MIN_SHARE).to_numpy(), n_agg, room.get(team_id, 0.0),
                match_pad(season), filler_sal.get(team_id, 0.0),
                PARTNER_MATCH if partner_match is None else partner_match)
            ok = ok & legal
            if max_gap is not None or lam:
                gap = cands["surplus"].to_numpy()[:, None] - r["surplus"].to_numpy()[None, :]
            if max_gap is not None:
                # NaN surplus (r with no salary) compares False: ruled out.
                ok = ok & (gap <= max_gap)
            if lam:
                pen = lam * np.clip(np.nan_to_num(gap, nan=0.0), 0.0, None)
            if fa_budget is not None:
                signable = (cands["pending_fa"].to_numpy()
                            & (cands["fa_cost"].to_numpy() <= fa_budget.get(team_id, 0.0)))
                p = cands["p_avail"].to_numpy() if "p_avail" in cands else np.ones(len(cands))
                fa_total = np.where(pos_ok & signable[:, None], p[:, None] * gain, -np.inf)
        total = np.where(ok, gain - pen, -np.inf)
        via_fa = np.zeros(total.shape, dtype=bool)
        if fa_total is not None:
            via_fa = fa_total > total
            total = np.maximum(total, fa_total)
        if scoring == "replace":
            best = total.argmax(1)
        else:
            # The score is c's alone, so every legal r ties. Name the one whose
            # package is closest in salary to c -- how real trades are built:
            # the cheapest departure plus whatever big contract makes up the
            # money read as Queta + Jrue Holiday for T.J. McConnell. Ties (and
            # no salary screen) go to the least departure cost.
            cost = np.broadcast_to(-(r_share * r_val)[None, :], total.shape)
            top = total.max(1, keepdims=True)
            tied = np.isfinite(total) & (total >= top - 1e-9)
            if room is not None:
                imbalance = np.abs(out_pick - cands["salary"].to_numpy()[:, None])
                tied &= imbalance <= np.where(tied, imbalance, np.inf).min(1, keepdims=True) + 1.0
            best = np.where(tied, cost, -np.inf).argmax(1)
        rows = np.arange(len(c_ids))
        score = total[rows, best]
        with_ids = np.full(len(c_ids), "", dtype=object)
        if room is not None:
            chosen = pick[rows, best]
            with_ids = np.array([";".join(ros_ids[j] for j in combos[c]) if np.isfinite(score[i]) else ""
                                 for i, c in enumerate(chosen)], dtype=object)
        if soft:
            score = np.where(np.isfinite(score), score, gain.max(1) - SOFT_PENALTY)
        wpos = pos_win * r_share[None, :]
        v_pos = (wpos @ r_val) / np.where(wpos.sum(1) > 0, wpos.sum(1), np.nan)
        v_pos = np.where(np.isnan(v_pos), np.average(r_val, weights=r_share), v_pos)
        need = cv - v_pos
        fw = FIT_WEIGHT if fit_weight is None else fit_weight
        if fw:
            score = score - fw * v_pos
        w = REALISM_WEIGHT if realism is None else realism
        lp = np.zeros(len(c_ids))
        if p_acq is not None and w:
            idx = pd.MultiIndex.from_arrays([np.full(len(c_ids), team_id), c_ids])
            lp = np.log(p_acq.reindex(idx).to_numpy())
            lp = np.where(np.isnan(lp), np.nanmedian(lp) if np.isfinite(lp).any() else 0.0, lp)
            eff = lp
            if log_move is not None:
                lm = log_move.reindex(c_ids).to_numpy()
                lm = np.where(np.isnan(lm), move_ceiling, lm)
                eff = lp - lm + np.minimum(lm, move_ceiling)
            score = score + w * eff

        rec = pd.DataFrame({
            "player_id": c_ids, "score": score,
            "gain": gain[rows, best], "log_p_acq": lp,
            "creator_adj": adj if np.ndim(adj) else np.zeros(len(c_ids)), "need": need,
            "replaces": r_ids[best],
            "with": with_ids,
            "replaces_value": r_val[best],
            "salary": cands["salary"].to_numpy() if "salary" in cands else np.nan,
            "replaces_salary": r["salary"].to_numpy()[best] if "salary" in r else np.nan,
            "via": np.where(via_fa[np.arange(len(c_ids)), best], "free agency", "trade")
            if fa_total is not None else "",
        })
        rec = rec[~rec["player_id"].isin(on_team.get(team_id, set()))]
        if k is None:
            top = rec.sort_values("score", ascending=False, kind="stable")
        else:
            top = rec[rec["score"] > -SOFT_PENALTY / 2].nlargest(k, "score")
        top = top.assign(TeamId=team_id, season=season)
        top["rank"] = np.arange(1, len(top) + 1)
        out.append(top)
    recs = pd.concat(out, ignore_index=True)
    recs = recs.merge(teams, on=["season", "TeamId"], how="left")
    recs["item_id"] = recs["player_id"] + "_" + recs["season"].astype(str)
    return recs


def build(paths, salary=True, filler=FILLER, max_gap=MAX_GAP, free_agency=False,
          scoring=SCORING, projected=True, realism=None, creator=CREATOR_TERM, k=K,
          aggregate=AGGREGATE, fit_weight=None, partner_match=None, move_cap=None):
    panel, impact, main, players, teams, salaries, books = load(paths)
    if projected:
        impact = with_projection(impact, paths.adv, panel)
    draft = pd.read_csv(paths.draft, usecols=["season", "lg", "round", "player_id"])
    flags = pending_fa(salaries, draft) if salary and free_agency else None
    rooms = team_room(teams, books, flags) if salary else None
    creators, b2 = creator_inputs() if creator else (None, {})
    ptabs = {s: player_table(s, impact, players, main, salaries, flags, creators)
             for s in sorted(panel["season"].unique())}
    if flags is not None:
        fa_prices(ptabs, salaries)
        debut = pd.read_csv(paths.career, usecols=["player_id", "from"]) \
            .drop_duplicates("player_id").set_index("player_id")["from"]
        availability(ptabs, memberships(panel), debut, teams)
    w = REALISM_WEIGHT if realism is None else realism
    p_acq = None
    if w:
        import acquisition
        p_acq = acquisition.p_acq_table(paths, ptabs)
    out = []
    for season, ptab in ptabs.items():
        r = rooms[rooms["season"] == season].set_index("TeamId") if salary else None
        out.append(recommend_season(season, panel[panel["season"] == season], ptab,
                                    teams[["season", "TeamId", "user_id"]],
                                    r["room"] if salary else None, filler,
                                    max_gap=max_gap if salary else None,
                                    fa_budget=r["fa_budget"] if flags is not None else None,
                                    k=k, scoring=scoring, realism=w, creator_b2=b2.get(season),
                                    aggregate=aggregate, fit_weight=fit_weight,
                                    partner_match=partner_match, move_cap=move_cap,
                                    p_acq=p_acq.xs(season, level="season") if w else None))
    return pd.concat(out, ignore_index=True)[
        ["user_id", "rank", "item_id", "score", "gain", "need", "log_p_acq", "replaces", "with",
         "replaces_value", "salary", "replaces_salary", "via"]]


def _jaccard_within_season(recs, rng, n_pairs=2000):
    """Mean overlap between two different teams' lists in the same season."""
    lists = recs.groupby("user_id")["item_id"].apply(set)
    season = lists.index.str.rsplit("_", n=1).str[1]
    vals = []
    for _, group in lists.groupby(season):
        g = group.to_list()
        for _ in range(n_pairs // 27):
            i, j = rng.choice(len(g), 2, replace=False)
            vals.append(len(g[i] & g[j]) / len(g[i] | g[j]))
    return float(np.mean(vals))


def compare(paths, examples):
    _, impact, _, players, _, salaries, _ = load(paths)
    salaries = salaries[salaries["season"] < max(CAP)]
    runs = {"LightFM (today)": pd.read_csv(paths.lightfm),
            "impact": pd.read_csv(paths.out_open),
            "impact + salary": pd.read_csv(paths.out)}
    cap = salaries.assign(cap=salaries["season"].map(CAP))
    meta = impact.merge(players, on=["player_id", "season"], how="left") \
        .merge(cap, on=["player_id", "season"], how="left")
    meta["cap_pct"] = meta["salary"] / meta["cap"]
    meta = meta.set_index(meta["player_id"] + "_" + meta["season"].astype(str))
    rng = np.random.default_rng(0)

    def describe(recs):
        m = meta.reindex(recs["item_id"])
        per_season = recs.assign(season=recs["user_id"].str.rsplit("_", n=1).str[1])
        top10_share = per_season.groupby("season")["item_id"].apply(
            lambda s: s.value_counts().head(10).sum() / len(s)).mean()
        return {
            "net impact (pts/100)": m["net_impact"].mean(),
            "BPM": m["bpm"].mean(),
            "age": m["age"].mean(),
            "salary, % of cap": 100 * m["cap_pct"].mean(),
            "share earning > 25% of cap": (m["cap_pct"] > 0.25).mean(),
            "share with < 1500 poss": (m["off_poss"].fillna(0) < MIN_CAND_POSS).mean(),
            "distinct players / season": per_season.groupby("season")["item_id"].nunique().mean(),
            "share of recs to top-10 players": top10_share,
            "overlap between 2 teams (Jaccard)": _jaccard_within_season(recs, rng),
        }

    print(pd.DataFrame({k: describe(v) for k, v in runs.items()}).round(3).to_string())

    sets = {k: v.groupby("user_id")["item_id"].apply(set) for k, v in runs.items()}
    for a, b in [("LightFM (today)", "impact + salary"), ("impact", "impact + salary")]:
        both = sets[a].to_frame("a").join(sets[b].to_frame("b"), how="inner")
        shared = both.apply(lambda r: len(r["a"] & r["b"]), axis=1)
        print(f"\n{a} vs {b}: {shared.mean():.2f} of 10 shared on average; "
              f"{(shared == 0).mean():.0%} of team-seasons share none")
    lists = runs["impact + salary"].groupby("user_id").size()
    print(f"Team-seasons with fewer than {K} affordable candidates: {(lists < K).sum()}")

    arch = pd.DataFrame({
        k: meta.reindex(v["item_id"])["cluster_label"].value_counts(normalize=True)
        for k, v in runs.items()
    } | {"league (>=1500 poss)": meta[meta["off_poss"] >= MIN_CAND_POSS]["cluster_label"]
         .value_counts(normalize=True)}).fillna(0).sort_values("impact + salary", ascending=False)
    print("\nArchetype share of recommendations:")
    print(arch.round(3).to_string())

    name = meta["player"].fillna(pd.Series(meta.index, index=meta.index)).to_dict()

    def money(x):
        return "?" if pd.isna(x) else f"${x / 1e6:.1f}M"

    for uid in examples:
        season = uid.rsplit("_", 1)[1]
        cols = [v[v["user_id"] == uid].sort_values("rank") for v in runs.values()]
        if cols[2].empty:
            continue
        print(f"\n{uid}")
        print(f"  {'LightFM (today)':<24}| {'impact':<24}| impact + salary: replaces, +pts/100")
        for (_, a), (_, b), (_, c) in zip(*(x.iterrows() for x in cols)):
            rep = name.get(f"{c['replaces']}_{season}", c["replaces"])
            print(f"  {str(name.get(a['item_id'], a['item_id']))[:23]:<24}| "
                  f"{str(name.get(b['item_id'], b['item_id']))[:23]:<24}| "
                  f"{name.get(c['item_id'], c['item_id'])} {money(c['salary'])} -> "
                  f"{rep} {money(c['replaces_salary'])}, {c['score']:+.2f}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--filler", type=int, default=FILLER,
                    help="non-rotation contracts a team may add to the outgoing salary")
    ap.add_argument("--aggregate", type=int, default=AGGREGATE,
                    help="lesser players a trade may add to the outgoing salary (trade_packages)")
    ap.add_argument("--max-gap", type=float, default=MAX_GAP,
                    help="value cap: surplus(candidate) - surplus(replaced) allowed, pts/100")
    ap.add_argument("--lightfm", default=os.path.join("data", "recs_lightfm.csv"),
                    help="the retired LightFM recommender's top 10s, for --compare (local only)")
    ap.add_argument("--examples", nargs="*", default=[
        "Golden State Warriors_2016", "Denver Nuggets_2024",
        "Houston Rockets_2019", "Washington Wizards_2025", "Detroit Pistons_2004"])
    ap.add_argument("--panel", default=os.path.join("data", "lineup_panel.csv"))
    ap.add_argument("--impact", default=os.path.join("data", "player_impact.csv"))
    ap.add_argument("--fit", default=os.path.join("data", "archetype_main.csv"),
                    help="fit_model.py's per-season archetype main effects")
    ap.add_argument("--players", default=os.path.join("data", "player_seasons.csv"),
                    help="player_seasons.py output: name, age, position, BPM per player-season")
    ap.add_argument("--teams", default=os.path.join("data", "team_style.csv"))
    ap.add_argument("--salaries", default=os.path.join("data", "salaries.csv"))
    ap.add_argument("--payroll", default=os.path.join("data", "team_payroll.csv"))
    ap.add_argument("--rosters", default=os.path.join("data", "team_salaries.csv"))
    ap.add_argument("--draft", default=os.path.join("data", "Draft Pick History.csv"))
    ap.add_argument("--career", default=os.path.join("data", "Player Career Info.csv"))
    ap.add_argument("--scoring", choices=SCORINGS, default=SCORING,
                    help="how a legal candidate is scored (see module docstring)")
    ap.add_argument("--realism", type=float, default=None,
                    help=f"weight on log P(acquired) (default {REALISM_WEIGHT:g}; 0 = quality only)")
    ap.add_argument("--no-creator", action="store_true",
                    help="leave out the creator-curvature term")
    ap.add_argument("--no-projection", action="store_true",
                    help="rank on raw season-t impact instead of the t+1 projection")
    ap.add_argument("--adv", default=os.path.join("data", "Advanced.csv"))
    ap.add_argument("--free-agency", action="store_true",
                    help="let flagged pending free agents in on free-agent terms (experimental)")
    ap.add_argument("--out", default=os.path.join("data", "recs_impact.csv"))
    ap.add_argument("--out-open", default=os.path.join("data", "recs_impact_open.csv"),
                    help="the same ranking without the salary screen")
    return ap


def main():
    args = parser().parse_args()

    if args.compare:
        compare(args, args.examples)
        return 0
    for salary, path in [(True, args.out), (False, args.out_open)]:
        recs = build(args, salary=salary, filler=args.filler, max_gap=args.max_gap,
                     aggregate=args.aggregate,
                     free_agency=args.free_agency, scoring=args.scoring,
                     projected=not args.no_projection, realism=args.realism,
                     creator=not args.no_creator)
        recs.round(4).to_csv(path, index=False)
        print(f"Wrote {len(recs)} recommendations for {recs['user_id'].nunique()} "
              f"team-seasons to '{path}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
