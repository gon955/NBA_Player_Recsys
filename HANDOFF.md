# Session handoff: recsys overhaul, phase 2

Branch `recsys_overhaul` (phase 1 commit `9ddd81a`). **Nothing from this session is
committed.** Run everything in the `recsys` conda env:
`conda run -n recsys python ...`

Tests: `pytest tests_pipeline --no-cov` (95 pass) and `pytest tests --no-cov` (13 pass),
run separately (both suites import a `helper.py`); `ruff check *.py tests_pipeline/` is clean.

---

## 1. What the project is trying to do

For a team-season (e.g. *Denver 2024*), recommend players to acquire who are:
1. **good**: they make the team better next season;
2. **realistic**: the team could plausibly make the move;
3. **specific to the team**.

The live app still serves the original LightFM model, which recommends players who
resemble a team's roster. This session built and validated its replacement, which
has two parts:
- a **value model**: projected player impact (`impact_model.py` → `impact_recs.project()`);
- a **realism model**: P(this team acquires this player) (`acquisition.py`).

The score is value + 2 · log P, among moves that are salary-legal and are upgrades.

---

## 2. How results are measured

Every component is trained or fitted only on seasons up to t and judged on what
happened in t+1, over 3,069 veteran arrivals from 1999 to 2025.

`python evaluate_recs.py` reports:
| metric | question | better is |
|---|---|---|
| **Realism** (headline / weighted by minutes) | where did players teams actually acquired rank in that team's list? 0.5 = random | lower |
| **Quality ρ** (added this session) | among those arrivals, did the ones ranked higher deliver more next season? | higher |
| **Team R²** | do arrivals' values explain the change in team net rating? Now sums only the value part of the score | higher |

`python acquisition.py` reports the realism model's metrics within each team-season's
list: mean percentile, within-list AUC, top-10 and top-50 hits, log loss and
calibration, plus a feature-group ablation. Pooled AUC is shown for reference only.

---

## 3. Results

| system | realism | weighted | quality ρ | team R² |
|---|---|---|---|---|
| random | ≈0.50 | ≈0.50 | ≈0.00 | – |
| **LightFM (live today)** | 0.473 | **0.431** | 0.129 | – |
| best available by BPM | 0.511 | 0.451 | 0.392 | +0.086 |
| phase 1 default (start of session) | 0.492 | 0.519 | **−0.122** | +0.012 |
| end of phase 2 session (w = 2) | 0.477 | 0.450 | 0.281 | +0.051 |
| w = 6, see below | 0.444 | 0.423 | 0.169 | +0.051 |

**After the first review (2025, 300 rows, 19 flags).**
- *Duplication* came from the realism term: P(he moves at all) is a player property,
  the same on every list, and it outweighed value about 4 to 1 (O'Neale was on 23 lists
  with below-median value, because he's the 2nd most likely player to move). Fix:
  `MOVE_CAP_PCT = 0.8` caps that part at the season's 80th percentile, with
  `REALISM_WEIGHT` 8 → 20 so unmovable players are still penalised. Tuning seasons:
  players who moved 30.6% → 29.3%, bias 0.08 → 0.03, list overlap 0.156 → 0.109;
  2025 overlap 0.170 → 0.067, max lists per player 23 → 12.
- *Movability calibration* (`acquisition.calibrate_moves`, `CALIBRATE_MOVES`): tested
  and off. The "stars look 2× too available" signal it targeted was an artifact of my
  check (1999–2001 placeholder probabilities and a stricter "moved" label). On the
  model's own label, 2002–2017, P(moves) by tier is already close, and calibrating
  overcorrects and worsens log loss.
- *Team-direction veto rules* from the flags all fail the audit (big contracts go to
  young or bad teams at 0.6–0.7× the normal rate; 30+ players to rebuilders at
  1.06×). They're kept in `audit.RULES`, inactive.
- *Review decisions are one record now:* `data/review/judgments.csv`, one row per
  (season, team, player): `ok` or `flag`, reasons and a note, plus what was shown at
  the time. `python review_app.py` reads the live lists, shows past decisions, marks
  only unjudged entries as new, and has an **Apply flags** button that runs
  `export_recs.py`, so replacements appear as new. Flags remove exactly that entry
  for that season (`export_recs.py`); they're unaudited by design. The first review
  (300 decisions, 19 flags) is imported; a second pass needs only the 124 new entries.
  `audit.py sheet` / `audit.py import` remain for spreadsheet review. `overrides.csv`
  is retired.

**Review app.** `python review_app.py` opens a local page (http://127.0.0.1:8765) over
the newest sheet in `data/review/`. It has a team queue, one checkbox per reason,
"apply to all lists" for player-level flags, and saves into the CSV on every change;
`audit.py summarize` reads it unchanged. Building it exposed a display flaw, now
fixed in `impact_recs.trade_packages()`: lists named a cheap backup as the player
going out while the salary match rested on a big contract nobody saw. Each
recommendation now names its full package (`with`), and of the legal packages the
most salary-balanced one is chosen (Giannis to Boston = White + Porziņģis). That
changes who is named, not who is legal or how lists rank. A partner-side salary
check was tested and left off (`PARTNER_MATCH`): it doubles the real trades ruled
out (12% → 25%).

**Projection and list concentration (follow-up session, latest).** All tuned on
decisions 1999–2017 with `rubric.py`; the holdout (2018+) is still unviewed.
- *Value model:* `impact_recs.project_mover()` (`PROJ_METHOD = "mover"`) replaces the
  single linear shrink. It adds reliability (possessions), box score, age, and terms
  conditioning on the player changing teams, with `proj_stay` for the player who'd be
  replaced. On every mover, it now beats BPM on next-season BPM (ρ 0.536 vs 0.525;
  the old projection was 0.515). The winner's curse on listed movers fell 0.43 → 0.04 pts/100.
  Age is now in: the team-level test that rejected it is no longer how this is graded.
- *Positional fit:* `FIT_WEIGHT = 2` adds how weak the team's rotation is at the
  candidate's position. Real arrivals at weaker positions get more minutes (t = 3.4),
  and real destinations rank above the middle by it (0.538). Overlap between two
  teams' lists fell 0.37 → 0.16; the top 10 players' share of slots 0.65 → 0.41;
  stars 16.6% → 11.9%; players who moved 32.9% → 30.6%.
- Remaining repetition: Royce O'Neale is in 23 of 30 2025 lists, McConnell in 20.
  An exposure cap would be a product rule; nothing observable sets it.

**Salary screen fixed (follow-up session, later).** `audit.py screen` showed the screen
ruled out 31% of real *trades*, because it matched salary with one rotation player
only. `package_salary()` now lets a trade add the largest lesser contract
(`AGGREGATE = 1`): wrongly rejected real trades fell to 14%. That made stars affordable
(13% → 23% of list slots), so the defaults were retuned on the tuning seasons only
(rubric.py, 1999–2017): `REALISM_WEIGHT` 6 → 8 and `MAX_GAP` 3 → 2. Result: stars at
17%, 32.7% of listed players moved (was 31.5%), winner's-curse bias 0.56 → 0.43. The old
rubric's realism/quality numbers in the table above are not recomputed for these
settings. Grading now runs through `rubric.py` (see its docstring) and veto rules
through `audit.py`; the holdout (2018+) has not been viewed yet.

**Realism weight raised 2 → 6 (follow-up session).** At w = 2, 55% of each top 10
were top-5% players by net impact, against 3% of actual veteran arrivals (Chicago
2025: Brunson, Murray, LeBron). P(acquired) is well calibrated by player value, so
the fault was the trade-off, not the model. At w = 6 the share is 17%, the system
beats LightFM on headline realism (CI excludes 0) and ties it on weighted realism.
The quality edge over LightFM is no longer significant (−0.040 vs LightFM,
CI −0.084 to +0.011). Tightening the value cap instead costs more quality per star
removed. The grid is in `impact_recs.py`'s docstring.

- **Quality:** more than twice LightFM's, by +0.15 (95% CI +0.11 to +0.20). The phase 1
  default ranked the arrivals who went on to help *lower*.
- **Headline realism:** tied with LightFM.
- **Minutes-weighted realism:** LightFM is still slightly better, by 0.019 (0.004 to 0.034).
- **Within 2015-2024 only:** realism 0.458 vs LightFM 0.466 (a tie), quality 0.272 vs 0.113.

---

## 4. What was done, in order, and what it changed

### Improvements that were kept
| step | change | effect |
|---|---|---|
| E7 | Quality ρ metric added to `evaluate_recs.py` | Showed the phase 1 scoring and the tight value cap were ranking the best arrivals lowest |
| E1 | `project()`: shrinkage plus the previous season (no age term) | Correlation with movers' t+1 impact 0.606 → 0.633. An age curve predicts players better but *hurt* the team-level test (−0.029, CI excludes 0), so it's off (`PROJ_AGE`) |
| E5 | Archetype effects fit only on seasons ≤ t; pair "fit" term removed | Removed a leak at no cost |
| E6 | Separate O/D ridge strengths tested | Null: both best at λ = 16,000 |
| N1 | Ablation showed every team-specific adjustment (minutes share, positional need, same-position rule) *lost* signal. New default scoring `quality`: value + legality + **upgrade gate** | Without the gate, Denver 2024's list was "Jokić for every other star" |
| — | Value cap MAX_GAP 1 → 3 | It mainly helped by ranking fringe signings higher; quality cost was large |
| N3 | `soft=True`: moves the screens rule out rank below every legal move but in order, not tied last | Quality 0.286 → 0.321; top-10 lists unchanged |
| E3 | `acquisition.py`, a gradient-boosted P(team acquires player), with REALISM_WEIGHT = 2 | Realism 0.514 → 0.476. On its own it ranks real arrivals at the 38th percentile (LightFM: 47th) |
| E4 | Creator curvature term (`weakness_test.py --write-curvature` → `data/creator_curvature.csv`) | Neutral on every metric (±0.3 against a 2.1 spread). Kept as a tie-breaker, since it's the one team-specific effect with within-team evidence |
| — | Scoring fixes: within-list metrics, team R² sums only the value part | — |
| — | **Decoupled from LightFM:** `player_seasons.py` (full universe, 13,104 rows, matches `master_clustered.csv` exactly where they overlap), archetypes dropped (`ARCHETYPES = False`, neutral), dead lineup covariates removed, LightFM baseline optional in the evaluator | Recommender reads nothing from the LightFM or clustering pipelines; results unchanged |

### Tested and rejected (null results, recorded in docstrings)
- **Contract status from salaries** (`contract_status.py`, using era-specific CBA raise
  bands, rookie scale and minimum/partial deals). As a detector it beats the old flat
  band: same 88% recall on known signings, with 5 points fewer false breaks. As a
  realism feature it's neutral.
- **Wikipedia free-agent labels** (`fetch_wikipedia_fa.py` → `data/wikipedia_fa.csv`,
  complete for the 2012-2025 offseasons):
  - a learned P(free agent) reaches AUC 0.80 but adds nothing to the realism model;
  - even the leaky "actual list" upper bound is worth only about 2 percentile points;
  - narrowing to the covered period costs nothing and gains nothing.
- **Team intent** (contender vs rebuilding): it's real as a description (contenders take
  players 1.2 years older), but helps neither the realism model nor quality over two
  seasons.
- **Model forms:**
  - an XGBoost LambdaMART ranker is worse (+0.009, CI excludes 0);
  - a **two-stage** model, P(moves) × P(this team | moves), ties overall. Its stage 2
    shows **destinations are almost unpredictable**: the real team ranks 14.2 of 29,
    against 14.9 for random. That's why nothing team-side has worked.
- **Destination features** (`destination.py`: trade partners, GM and coach changes and
  ties from the cached Wikipedia pages, roster churn). Each lifts the destination
  probability by at most about 1.2×; overall +0.17 places (CI includes 0). Kept as the
  `two-stage + destination` variant, not a default.
- **Surplus penalty** (`SURPLUS_PENALTY`, follow-up session): a soft value cap,
  score −= λ·max(0, surplus gap). Null at w = 6 over all seasons and on 2024
  decisions alone; it lowers quality without improving realism. Off.
- **Quality-of-competition proxy** (`impact_model.py --competition`). Large, consistent
  lineup-level effect (+16% on next-season offence R²), but it moves player estimates
  by only about 0.03. Off by default.

### Repo cleanup (staged, not committed)
- 13 unreferenced BBRef CSVs removed with `git rm`.
- `data/unmatched_report.csv` and `data/archetype_fit.csv` untracked and gitignored
  (still on disk).
- `data/wikipedia_raw/` gitignored.
- `xgboost==2.1.4` added to `environment.yml`.
- **Kept by your decision:** the LightFM path (it still serves the live app) and the
  Ask/RAG tab.

---

## 5. Files

**New:** `acquisition.py`, `contract_status.py`, `destination.py`,
`fetch_wikipedia_fa.py`, `player_seasons.py`, `weakness_test.py`, plus tests for each
in `tests_pipeline/`.

**New data outputs:** `data/player_seasons.csv`, `archetype_main.csv`,
`creator_curvature.csv`, `contract_status.csv`, `wikipedia_fa.csv`.
The `data/wikipedia_raw/` cache is not tracked. Re-fetching runs one request at a time,
6 seconds apart, and stops after repeated refusals.

**Modified:** `impact_recs.py`, `evaluate_recs.py`, `impact_model.py`, `fit_model.py`,
`team_style.py`, `lineup_features.py`, `.gitignore`, `environment.yml`, the regenerated
`data/recs_impact*.csv`, and tests.

**Regenerating everything after an upstream change:**
```
python player_seasons.py
python weakness_test.py --write-curvature
python contract_status.py
python fetch_wikipedia_fa.py --consolidate-only     # no network if the cache exists
python impact_recs.py                               # writes data/recs_impact*.csv
python evaluate_recs.py                             # full report
python acquisition.py [--model-compare | --wiki-compare 2015 2024]
```

---

## 6. What the evidence says about the design

- **Value:** quality comes almost entirely from *who the player is*. The projected
  value alone beats BPM on quality (ρ 0.428 vs 0.392), and every team-specific
  adjustment so far has cost signal.
- **Realism:** this is almost entirely *who moves* (veterans, experience, age,
  contract), not *where they go*. Public structural data barely predicts destinations.
- **Team specificity** comes mainly from what a team can afford (salary screen, value
  cap, upgrade gate) and the acquisition likelihood, not from what it needs.
- **The realism weight** (6) and the value cap (3) are judgment calls on a
  quality-vs-realism trade-off.

---

## 7. Next session

1. **Commit this work** (only when you ask), perhaps as separate commits: cleanup,
   decoupling, the models, the data fetchers.
2. **Decide on the real matchup-data pilot** (pending). pbpstats
   `get-lineup-opponent-summary` is one request per lineup. The proposal is one season,
   the ~300 lineups with 200+ possessions, fetched one at a time 3 seconds apart
   (~15 min) and cached, then test whether true opponent adjustment moves player
   estimates more than the proxy did. Full coverage would be ~786 requests per season
   for 52% of possessions.
3. **Switch the backend and remove LightFM and clustering:** DONE. The backend serves
   `backend/data/recs.csv` (`export_recs.py`, `backend/recommender.py`). The LightFM
   recommender, the clustering pipeline, the Clusters tab, and the Ask index's build
   scripts (they read LightFM-era outputs) are out of the repo, in the gitignored
   `local_archive/`, with the same folder layout. The Ask tab's runtime and shipped
   index stay; rebuilding the index from current data needs a new chunk builder.
   `lightfm`, `joblib` and `scikit-learn` are gone from the backend image, along with
   the compiler stage and `libgomp1` (761 MB → 551 MB).
   **Ask embeddings moved to Bedrock** (Titan Text Embeddings v2, 512 dims): the index
   was re-embedded by `build_ask_index.py` from the chunk text in its metadata, with the
   "LightFM feature" fragments stripped; the old bge-small index is in
   `local_archive/backend/data/index_bge_small/`. Dropping fastembed (onnxruntime,
   sympy, tokenizers, Pillow, 65 MB of weights) and pruning pip/setuptools took the
   image to **310 MB**. On 60 player-season and 30 team-season lookups, Titan finds the
   right chunk first 87% / 93% of the time (bge: 65% / 93%). **Before deploying:** the
   Lambda role needs `bedrock:InvokeModel` on `amazon.titan-embed-text-v2:0`, with model
   access enabled in us-west-2; its 2048 MB memory was sized for the local embedder and
   can likely come down. `data/recs_lightfm.csv` and
   `data/lightfm_scores_full.csv` stay on disk, untracked, as optional baselines.
   Still possible: moving off Python 3.9, now pinned only by `pandas==1.2.4`.
4. **Product choices to settle:**
   - **Shelved: more starter-level players in the lists.** After the movability cap, the
     2025 lists are mostly rotation veterans plus a few superstars who might leave.
     Tier mix by BPM percentile:

     | tier | real veteran arrivals | our top 10s, 1999–2017 | our top 10s, 2025 |
     |---|---|---|---|
     | bench (bottom 50%) | 52% | 27% | 27% |
     | rotation (50–80th) | 34% | 44% | 61% |
     | starter (80–93rd) | 11% | 16% | 7% |
     | fringe all-star (93–98th) | 3% | 8% | 4% |
     | star (top 2%) | 0.5% | 4.5% | 1.7% |

     Starters really do move less (about 25% a season vs 33% for rotation players),
     and the model's P(moves) is close to calibrated by tier, so this is not a model
     error (see `acquisition.calibrate_moves`, tested and off). Options:
     1. *Recommended:* a separate **"Starters & all-stars"** section per list. It
        holds the best-scoring starter-tier and above players the team could legally
        get, ranked by the same score, alongside the realistic targets. It needs a tier
        field in `export_recs.py` and `backend/recommender.py`, and a second list in
        the frontend; the review app would need the same.
     2. Lower `REALISM_WEIGHT` (now 20, with `MOVE_CAP_PCT = 0.8`). Starters rise, but
        so do unmovable stars (the weight-8 trade-off).
     3. Keep the lists as they are, as a reflection of how rarely good players move.
   - the realism weight (now 20 with the movability cap; lower = more stars and
     starters, higher = more journeymen), possibly as a UI toggle;
   - the value cap;
   - whether rebuilding teams should get an age preference as an explicit product
     rule (the data doesn't support it as a model term).
5. **Longer-term options, if more accuracy is wanted:**
   - a better impact prior: a statistical plus-minus fit on multi-year impact from the
     253 pbpstats player columns already on disk;
   - stint-level RAPM, if matchup data is obtained;
   - contract data with real end dates and options (would let free agency, `--free-agency`, be turned on);
   - health and availability;
   - draft prospects (27% of arrivals are rookies, currently out of scope).
6. **Open question:** why the age curve helps player-level prediction but hurts the
   team-level test, even after correcting for survivorship.

### Known caveats
- Many settings (cap, weight, scoring) were tuned against this same evaluator, so the
  numbers are somewhat optimistic.
- Realism measures imitation of teams, and quality only sees players who actually moved.
- The front-office tables undercount a couple of pages (2017-18 coaches, 2022-23 GMs);
  the 2006-07 and 2007-08 pages have none.
- `lineup_panel.csv` is gitignored and was built in phase 1 with the old covariate
  columns. They're harmless, and a rebuild with `lineup_features.py` no longer writes them.
