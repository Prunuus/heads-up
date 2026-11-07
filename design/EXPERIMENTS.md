# EXPERIMENTS.md — Research, Calibration, Benchmarking, and Strategy Selection (v0.4)

**Scope:** A few hours of engineering on an ordinary laptop; Elephant-phase plan only. This plan is intentionally selective and prioritizes *runtime and correctness ahead of strategic complexity*. Bot coding and match testing have **not** yet occurred. Use the official pinned game/evaluator/judge; never alter referee code to change competitive results. The authoritative strategy is **`DESIGN.md` v0.4** (one canonical copy). See `CRITERIA.md` for normative acceptance tests, `RESEARCH_README.md` for provenance, and `READINESS.md` for status.

## 1. Immutable experiment registry

For every run store: UTC timestamp, source commit, agent commit, strategy configuration hash, `FULL_PAIR_v04` or `REDUCED_64_v04`, opponent name/commit, exact seed, # deals, seat/half, actions/calls, start latency, max request latency, charged time, errors, forfeits, time-bank exhaustion, match paired bb/deal, sample SD, paired 95% CI, win/chop/loss, and notes on environment. Keep local output JSON when available. An unlogged run is exploratory only.

**Separate, nonoverlapping data families (literal seed labels):**

| Purpose | Seed labels | May influence policy selection? |
|---|---|---|
| Pilot threat analysis | integer RNG seed `8037` (independent study already conducted) | Descriptive only; not a final match seed |
| Fresh official-table five-card calibration | `threat-cal-13`, `threat-cal-31`, `threat-cal-59` | Yes for threshold calibration; never as final match audit |
| **Development performance** | `dev-11`, `dev-29` | **Yes**, feature/mode-specific performance and promotion |
| **Runtime stress** | `stress-41`, `stress-83` | **Yes for feasibility only**, not competitive promotion |
| **Final sealed holdout** | `final-73`, `final-97` | **No**; report only once after frozen candidate; never retune on them |

All are explicit strings passed as `--seed`. For extra rounds of discovery, create **new** independent `dev-*` labels before testing. If the final audit fails, those final seeds become disclosed and **must never be used as a new private test set**; revert to the prior development-selected candidate and create a distinct final audit set only after any new exploratory work is frozen.

**Opponents:** Four official baselines: `baseline_random`, `baseline_calling_station`, `baseline_always_half`, `baseline_always_pot`. Add two deterministic archetypes when practical: `tight_value` (plays selected top-hand score band; value-bets only strong hands, action rules frozen before testing) and `overfolder` (folds to a raise with a fixed preset deterministic schedule when folding is legal). Add a `call_heavy` stress opponent and a `max_raise` stress opponent specifically for timing. The first four are the minimum competitive comparison suite. Any custom opponent is frozen before feature tuning and never altered to make our candidate look better.

**Unit of statistical analysis:** Per-deal **paired result** across both halves; `n=1,000` paired observations per full match, not 2,000 independent hands. Report `mean ± 1.96×s/√1000` and the exact referee verdict. For comparing policy versions, match opponent + seed + seat schedule, compute paired **candidate-minus-baseline** values and report mean/uncertainty; note that distinct matches can share variation, so pooling blindly across opponents is not valid inference about an unknown tournament field.

## 2. E0 — Runtime eligibility ONLY: no mode selection [P0 FIRST]

**Hypothesis:** FULL fragment enumeration fits the complete eight-second time-bank while preserving useful strategy headroom; REDUCED_64 might offer a feasible lower-cost alternative but cannot be chosen on speed alone.

**Independent variable:** `evaluation_mode` ∈ `{FULL_PAIR_v04, REDUCED_64_v04}`; *all other strategy thresholds identical except the explicitly mode-specific tier calculation*. Use the actual server, official judge, and local one-CPU restricted container. Pre-warm only during startup, not inside `/act`.

**Experiments:** First microbenchmark hero 10/40/100 and opponent 378/1404/3250 vs reduced 64/256/640 probes to diagnose cost, then **for each candidate mode** run four official opponents × `stress-41,stress-83` × full 1,000-deal duplicate matches (16 half-matches), plus call-heavy (frequent river/showdown) and max-raise stress for **each** runtime candidate. Record cache hit rate and `on_hand_end` processing time, but charge **whole HTTP requests** for acceptance.

**E0 outcome:** For **each** mode, emit only `RUNTIME_PASS` or `RUNTIME_FAIL` using `CRITERIA C-MODE` (≤5.5s charged/half, ≤100ms max request, ≤12s startup, no errors/forfeits/bank exhaust). Do not choose, freeze, deploy, or strategically compare modes in E0. E0 has no authority to waive accuracy gates; E1 will test every E0-passing mode. If both fail E0, stop. If one passes, E1 still must approve it. The final choice/freeze occurs **after E1** by the state machine in DESIGN §2.2 and CRITERIA C-FREEZE. Do not silently switch algorithms in any run.

## 3. E1 — Five-card calibration of relative hand strength [P1]

**Hypothesis:** The revised FULL `.03/.12` fragment-risk tiers correlate substantially better with actual uniform-random five-hole-card current-hand threat than the v0.2 `.06/.22` tiers; REDUCED's sampled tiers are sufficiently conservative to avoid falsely labeling dangerous holdings STRONG.

**Reproducible official-table process:** First use `research/calibrate_offline.py recover` to reconstruct the v0.3 **legacy** 240 states with explicit hole/board IDs, then `verify` against a checkout of the pinned official evaluator and record source SHA, script SHA, input/output SHA-256, Python version, and exact card encodings. Do not promote historical pilot outcomes to VERIFIED until **all 240 packed scores, fragment counts, beating-five-card counts, and tie-only counts** match. The historical pilot is not E1 acceptance data.

Generate **fresh independent** calibration states from `threat-cal-*`, at least 60 per street, purposely stratifying FULL STRONG/MEDIUM/WEAK and paired/trips/flush/danger textures until the exact `CRITERIA T08` minimums are met; otherwise report `ACCURACY_INCONCLUSIVE`. Use official packed `get_table()` for hero and each feasible unseen **two-card fragment**, once per pair. Record the BEAT and TIE sets as an edge graph over unseen cards. Enumerate each complete opponent five-card subset and test its ten internal edges for BEAT first, TIE if no BEAT. That produces **exact uniform current-five-card threat** while avoiding repeated five-card poker evaluation. Compare at least three representative states against a **separate slow direct official-score loop** that scores all ten hole-card pairs and board triples for every complete holding. A graph-vs-graph comparison is insufficient. Reproduce `Ah Ac Kc Qd 6s` / `Kh Kd 8c 7s 6h` → 13 beating pair edges, 17,156 beating complete holdings.

**Cost model (60 states per street):** exactly 60×(378+1,404+3,250)=301,920 maximum pair/board-triple score probes, plus 60×(98,280+80,730+65,780)=**14,687,400** five-card subset checks, instead of approximately **647,400,000** repeated five-card score probes in the naive full-holding reference. These are theoretical counts, not measured seconds. Independently slow-verify a small subset, not all 180 states, on ordinary laptop compute.

**Dependent measures:** Spearman ordering of fragment frequency vs exact current-five-card stronger risk; exact current-five-card tie-only risk; per-tier coverage by street and texture; count/percentage of `STRONG` assignments with exact five-card stronger risk ≥.50 and Wilson 95% CI; nuts proof failures; FULL-versus-REDUCED upward/downward tier disagreement, disagreement within ±.015 of `.03` and `.12` pair-risk boundaries; action disagreement on ≥200 official-Game-reachable decision contexts with identical nonmode inputs; stratification by paired/trips/suited/ordinary boards; and separate REDUCED diagnostic salts 0,1,2,3. Store all state cards, scores, pair indexes, sampled risks, and action differences as CSV. The 13/325 case has hypergeometric *uniform-subsample* probability `P(0 beating pairs)=5.439%`, `P(≤1)=23.613%` for 64 of 325 without replacement; an individual deployed affine sample is deterministic and not asserted to follow that distribution across states. No sample-zero nuts proof.

**E1 outcome per E0-passing mode:** `ACCURACY_PASS` only if official-score, graph, sampler/fixture, nuts-proof, coverage, misclassification, FULL/REDUCED disagreement, and action-difference gates listed in `CRITERIA C-THREAT` all pass. `ACCURACY_INCONCLUSIVE` if any required tier/texture/boundary sample count is missing, a 95% interval is too wide, or the official verifier has not finished. `ACCURACY_FAIL` for any wrong exact count, proven invalid nuts claim, or measured violation of an explicit tolerance with adequate coverage. If E1 fails, a threshold adjustment may be considered only on **new or reused development calibration states** with review and a *full rerun* of E1; do not reclassify a failure as PASS. **After** E0 and E1, freeze FULL if it passed both, otherwise REDUCED if it passed both, otherwise STOP. Do not run E2–E4 without a freeze. This is not evidence of match profitability.

## 4. E2 — Frozen strategy baseline and preflop opening aggression

**Hypothesis:** The v0.4 opening-score cutoffs `20/40/60` capture profitable folds/limps/raises better than one more conservative or aggressive band choice.

**Baseline:** Frozen v0.4 selected evaluation mode, opponent adaptation disabled, pure bluffs disabled, fixed call/reraise and value-bet rules. This baseline is also the rollback candidate for all later ablations.

**Single named variant axis:** `opening_aggression` only: default thresholds `(20,40,60)`, looser `(20,35,55)`, tighter `(20,45,65)`. Keep all other functions and thresholds identical. (Changing the two raise cutoffs together is one explicit named opening-policy variant; do not mix it with changes to threat tiers.)

**Runs:** `dev-11`, `dev-29`, four official baselines, 1,000 deals × both halves for each candidate/opponent/seed, **only after mode freeze**. Favor complete fewer variants over half-matches without paired scoring. Each bot must maintain P0 passing. **Decision:** promote only per `CRITERIA C-STRATEGY`; otherwise retain simplest unchanged baseline. Changes in structural opening frequency must be reported; they are not evidence of higher EV.

## 5. E3 — Relative strength and calling / value-bet sizing

**Hypothesis:** Threat-informed relative tiers and controlled value raises improve paired bb/deal compared with simpler category/board-texture decision rules, without increasing time failures.

Test in **separate** ablations on the **same development seeds**:

1. `relative_tier` current `.03/.12` versus a frozen simpler score/category + danger baseline and, if E1 supports it, exactly **one** alternate pair of tier boundaries. Do not introduce untested tiers.
2. `call_bands` current `.12/.27/.34` style bands versus a conservative variant lowering MEDIUM by `.02` on flop/turn/river; all else fixed. Actual half and pot prices are **.25** and **1/3**, so focus on changes crossing those thresholds rather than cosmetic hundredths that never change actions.
3. `value_bet_size` current half/pot rules vs forcing *qualified value raises* to half-pot only; retain legal guards, profile, and no-bluff constraints.

For each: exact independent variable, same frozen baseline and opponents, development results including per-street fold/call/raise frequencies, contribution by opponent, paired bb/deal difference and match points. Do not claim causality if multiple axes are changed simultaneously. Keep better *development-screened* policy only when P0 and promotion thresholds pass; otherwise rollback.

## 6. E4 — Contextual opponent adaptation and optional bluffing

**Hypothesis:** Conservatively smoothed opponent profiles improve against exploitable baselines without catastrophic regressions against disciplined opponents.

Compare three strictly isolated variants: (A) no adaptation/no pure bluff; (B) profile-dependent opening/call/value sizing only, with pure bluffs still disabled; (C) B plus explicitly gated overfolder pure bluffs. Opponents and seeds unchanged, with `tight_value` and `overfolder` if available. Analyze first 100 hands versus final 500 hands separately as diagnostics, but match verdicts always use full 1,000-deal duplicate pairs. Log observed-VPIP coverage and error under intentionally missing first actions. Do not qualify river bluffing from preflop fold responses.

Promotion must meet the development-only threshold and no ≥1-point per-opponent regression; if noisy, keep no-bluff policy. Never retune B/C on final seeds.

## 7. Final one-time selection and release audit

1. **Confirm** the mode already selected and frozen after E0/E1; choose one **development-selected strategy policy** based only on E2–E4. Freeze the final policy/config/implementation hashes; pass all `CRITERIA` P0 requirements. E2–E4 results may never reopen evaluator-mode choice.
2. Run **once**: four official baselines × `final-73`,`final-97`, 1,000 deals and both halves each (8 matches) for the chosen policy, optionally compare with frozen baseline using *the same* final seeds solely to report the audit, **not to promote a different candidate after seeing results**.
3. Report each opponent/seed row: per-half charged time/errors and paired bb/deal/CI/verdict; summary points, wins/chops/losses, worse-opponent regressions, and confidence limits. If results disappoint, write that explicitly. Do not assert competitive superiority without evidence.
4. A final-audit failure does **not** authorize tweaking parameters and retesting on the same `final-*` seeds. Revert if needed, gather new development data, preregister **new** final seeds, and repeat only if development time remains.

## 8. Prioritization under a hard few-hour development window

1. Verify official evaluator and protocol, reconstruct pilot, and establish one correctness-first experimental baseline (no competitive implementation claims yet).
2. E0 benchmark modes; E1 validate each runtime-eligible mode using the graph reference, direct official cross-checks, coverage and disagreement gates; **then freeze mode**.
3. Complete an end-to-end working baseline after the Elephant handoff; spend remaining development measurement time on E2/E3 **before** E4 adaptation.
4. Reserve a non-negotiable final build/smoke/protocol test block; drop optional pure bluffs and fine tuning first.
5. Keep logs, hashes, and failed experiments. A reliable simple bot has a better expected chance of tournament points than an untested advanced bot that forfeits.

## 9. Execution/status and leakage protections

- **Status now:** DESIGN/CRITERIA/EXPERIMENTS are written, raw 240 pilot state reconstruction and **legacy-scorer** hero/pair checks completed, and deterministic sampler fixture values generated. Official `get_table` cross-validation, full opponent-graph verification, E0 timing, E1 fresh calibration, and all opponent matches are **NOT EXECUTED**. Record separately when evidence exists.
- Use separate registries: E0 stress workloads are for runtime feasibility, E1 threat states for accuracy and mode selection, E2–E4 development seeds for feature selection, and sealed final seeds for a single post-freeze audit. Disclose scores from final matches but never tune to them.
- Do not overstate statistical certainty. Wilson bounds quantify binary strong-mislabel uncertainty on observed stratified sets; those sets are *not* random samples of real opponent encounters. If an acceptance stratum is sparse, mark the mode INCONCLUSIVE.
- If time is short, finish legal-action and evaluator correctness before full multi-opponent strategy tuning. Never choose an algorithm from one quick microbenchmark. If both evaluator modes fail E0/E1, stop and request design revision rather than deploy a hidden fallback.

## 10. Explicit uncertainty register

- No bot or official-referee matchup has been run as part of drafting these documents; **no tournament win/loss claims exist**.
- The 240-state independent-score pilot is preliminary, unconditioned on opponent betting, and needs official-table cross-validation; it is not one of the final match seed sets.
- REDUCED_64 can miss rare opponent threats; if it fails calibration, do not silently strengthen it by pretending its sampled zero means nuts.
- Preflop score, relative thresholds, call ceilings, fold priors, value-bet sizing, and bluff gates are all engineering hypotheses.
- Expected tournament opponents are unknown; results against four simplistic official baselines may not transfer to strong competitors.
- An 8-second bank across 1,000 hands is unusually tight; mode selection requires real judge timing on restricted hardware rather than intuition.
