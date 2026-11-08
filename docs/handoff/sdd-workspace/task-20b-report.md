# Task 20b report: hybrid base selector (v3.1)

Status: DONE. Verdict: SHIP v3.1 (default base = "auto").
Commits (not pushed): e5ab8f2 (auto base option, tests, gate --candidate), plus the default flip commit (see git log).

## Signal
Collected the OpponentModel stats of every zoo bot (v2 vs each, 400 deals) and joined them with the per-bot v3-v2 bb from the final gate (TRAIN, p0.1).
Bots where v2 > v3: calling_station, checkfold_after_300, llm_beginner_2h, llm_simple_rules, mc_equity_fast, mc_equity_slow, nit, threshold_cat, llm_opp_model (and llm_gto_inspired / llm_time_careful, mid raise rates).
Their common trait is a PASSIVE preflop: opponent preflop bet/raise rate (OpponentModel.raise_rate(PREFLOP)) is 0.00-0.08 after 400 hands (0.10 at 50 hands for the worst). Every FIELD and FIELD2 bot is 0.32-0.64 (min over time 0.32), llm_aggressive 0.44, maniac/random/always_* 0.5-1.0 (v3 is better there). The rate is stable from ~50 hands on. Rule derived from TRAIN stats; HOLDOUT/FIELD used only as validation.
Unseparated (kept on blueprint, safe default): llm_gto_inspired (0.32), llm_time_careful (0.30), llm_mc_equity (0.15), llm_ml_minded (0.18).

## Selector design (bot/policy.py)
`select_base(opp, current)` pure, deterministic: with >= 60 raise-legal preflop opponent decisions and rate <= 0.12 go "heuristic"; return to "blueprint" at rate >= 0.20 (hysteresis band keeps the current base). Any error/missing/NaN -> "blueprint". `PolicyBot(base="auto"|"blueprint"|"heuristic")`, memory["base"] per half (starts blueprint). `_use_blueprint_now` is exception-safe. Exploit layer unchanged on top of whichever base answers (heuristic base = v2 path, with Features). No new deps; cost is one dict read + a few floats per decision.
Never switched on any FIELD/FIELD2 half (0 heuristic decisions in 21 halves at ts 1 and 2.5, verified), so there v3.1 is decision-for-decision v3; v31 vs v3 head-to-head = exactly +0.000 on all 5 seeds.

## Tests
RED: new selector tests failed with `AttributeError: module 'bot.policy' has no attribute 'select_base'`.
GREEN: 7 new tests in tests/test_policy_blueprint.py (no-evidence default, passive switch, hysteresis, never-raises, per-half switching, fixed bases + ValueError, failing selector keeps blueprint, default is auto). Full fast suite: 668 passed, 52 deselected, 2 xfailed (121 s) before the final test; slow zoo/zero-error tests of agent.py: 25 passed; test_gate_v3 5 passed.

## Gate (fresh dir train_data/gate_v31; cpus 0-7, workers 2; v2 results copied from gate_v3_final; v3 re-run fresh in the same dir for a same-conditions pair)
Sim: `sim/gate_v3.py` gained `--candidate v31` and `--h2h-opponent`.
| split / scale | v3.1 pts | v3 pts | v2 pts | v3.1 - v3 bb [95% CI] | v3.1 - v2 bb [95% CI] |
|---|---|---|---|---|---|
| HOLDOUT ts1 | 0.917 | 0.792 | 1.000 | +0.349 [+0.095, +0.603] | -0.100 [-0.335, +0.134] |
| HOLDOUT ts2.5 | 0.958 | 0.875 | 1.000 | +0.488 [+0.211, +0.764] | -0.036 [-0.292, +0.219] |
| FIELD ts1 | 0.611 | 0.611 | 0.167 | -0.068 [-0.391, +0.254] (noise) | +1.246 [+0.870, +1.622] |
| FIELD ts2.5 | 0.778 | 0.778 | 0.389 | -0.418 [-0.709, -0.127] (noise) | +0.302 [-0.056, +0.660] |
| TRAIN ts1 | 0.933 | 0.933 | 0.967 | +0.305 [+0.209, +0.401] | +0.093 [-0.049, +0.234] |
v3 itself on TRAIN vs v2: -0.213 [-0.362, -0.064]; v3.1 vs v2 now +0.093. Head-to-head v3.1 vs v2: +0.537 [+0.236, +0.838] (same as v3). TRAIN gains: mc_equity_fast +1.73, llm_simple_rules +1.65, llm_beginner +0.76, checkfold +0.53, nit +0.29; llm_mc_equity -0.43, guide_example -0.22 (small n, noise-level).
FIELD bb differences (e.g. field_exploit -0.96 at ts2.5) are wall-clock noise of timed opponents: the policy is provably identical there (0 switches); points are equal.
Errors: 0 errors, 0 forfeits; max bank per half 3.23 s at ts 2.5 (same regime as v3's 3.75 s there), 0.96 s at ts 1; longest request 20.6 ms.

## Verdict
Ship criterion met: FIELD points equal to v3 (no loss), HOLDOUT and TRAIN better than v3 with CIs above 0. SHIP v3.1.
Concerns: threshold tuned on TRAIN bots, a real passive/adaptive opponent in the field could mis-trigger (limited downside: heuristic v2 base is the validated fallback); mid-band bots (llm_gto_inspired, llm_time_careful) remain below v2.
