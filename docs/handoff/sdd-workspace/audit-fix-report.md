# Audit fix report: Importants I1–I6 (foundation, Tasks 1–7)

Fixer: Opus 5.5 (max effort), 2026-11-07. Base `df265bd`, branch `kevin`, nothing pushed.
Requirements: `audit-tasks1-7.md` §Important, `global-constraints.md`, ruling R22.
Scratch scripts and raw outputs (none in the repo) are in
`/private/tmp/claude-501/-Users-kchen-Desktop-code-FALL26-heads-up-bundle/b4276c44-89ab-406f-9078-0e01542ecd72/scratchpad/`
(all `fix_*` files, plus `i3/`).

## Commits

| Finding | Commit | Kind |
|---|---|---|
| I2 | `771e686` fix(bot): fasteval pairs a single row with every row of the other side | code + tests |
| I1 | `817a3d3` fix(bot): equity API defines iters <= 0 as a cheap deterministic estimate | code + tests |
| I6 | `a16af7c` test(bot): strength_now counts the made hand now, not the runout | tests only (code was correct) |
| I3 | `79036c5` fix(sim): load_bot_file gives every load a fresh copy of the bot package | code + tests + fixture |
| I5 | `86d5496` test(sim): pin paired_diff's CI to its analytic width, and the standings | helper refactor + tests |
| I4 | `ab1d1b9` test(sim): differential tests against the organizer's judge | tests + fixtures |
| I2/I3 follow-up | `66a3d73` test(bot): shape errors must name the bad argument; follow-ups to I2/I3 | tests + docstring |

Organizer files (`game.py`, `evaluator.py`, `judge_engine.py`, `baseline-bots/`, `docker/`, `tools/`,
`sim/opponents/llm/`) are unchanged: `git diff --name-only df265bd..HEAD -- <those>` is empty.
`bot/*.py` parse under the 3.12 grammar (`ast.parse(..., feature_version=(3, 12))`) and import only
`__future__`, `itertools`, `typing`, `math`, `time`, `pathlib`, `numpy`, `evaluator`, `game`, `bot`.

Method notes:
- RED for the three test-gap findings (I4, I5, I6) means "the new test fails on the audit's injected
  bug"; the shipped code there was already correct, so the new tests pass on it.
- Mutation runs use `fix_mutrun.py`: it `git archive`s a commit into a temp dir, rewrites the target
  function **in the file** (so subprocesses and spawn workers see the bug too, unlike the audit's
  in-process patching), optionally overlays uncommitted test files, and runs pytest there with the
  repo's `.venv` python. Mutant definitions are the audit's own `mutants.py`, imported unchanged.

---

## I1: `iters <= 0` crashed the equity API (TimeManager returns 0 by design)

**Root cause.** `equity_vs_random` / `strength_now` sampled `(iters, k)` cards and scored them. With
`iters=0` the zero-row batch hit `reshape(len(h), -1)` in `fasteval._best_of` (ValueError); negative
iters failed in `rng.random((iters, n))`. `TimeManager.iters()` returns 0 at the hard stop and when
pacing drops below level 16, so Task 10 would crash on its first starved decision.

A finding of my own while fixing: once I2's reshape fix is in, `iters=0` no longer raises but returns
`nan` (0/0) silently: `test_zero_iters_is_defined[*-0]` failed with `assert (True and 0.0 <= nan)`.
NaN is worse than the crash: every comparison is False (the heuristic check/folds) and `searchsorted`
puts NaN in the top bucket. I1 therefore landed right behind I2 (one commit apart; only `771e686`
itself has the NaN behaviour).

**Decision: iters <= 0 means "no rollouts" and returns a cheap deterministic estimate.**
Options weighed, with measurements (`fix_i1_proto.py`, exact strength by enumerating every opponent
hole on 120 flop, 80 turn and 80 river spots):

| Option for iters <= 0 | Failure mode / quality |
|---|---|
| raise (status quo) | the I1 crash |
| return `None` sentinel | crashes in the caller's arithmetic: the same failure, moved |
| return `nan` | silent: compares False everywhere, lands in the top bucket |
| return 0.5 | defined, but ranks nothing: MAE 0.27 vs exact strength, corr 0; postflop the heuristic becomes a calling station |
| **made-hand percentile (chosen)** | MAE 0.095 / 0.089 / 0.117 (flop/turn/river), corr 0.92 / 0.93 / 0.84 vs exact strength |
| texture-aware percentile | MAE 0.068 / 0.073 / 0.086: about 25% better, but 4x the reference tables, rare classes undersampled, more code; not worth it for an emergency path |

**Fix (`bot/equity.py`).** Any `iters <= 0`: sample nothing, leave `rng` untouched, return a float in
[0, 1], never NaN, never an exception:
- `equity_vs_random` preflop: `bot.preflop.preflop_strength(hole)` (the same quantity from 3,000
  rollouts per class); 0.5 if the table cannot be loaded (OSError/KeyError/ValueError).
- otherwise, both functions: `_made_hand_estimate` = our made hand's rank among the made hands of
  2,048 random holes on random boards of the same width (ties half). The reference is three sorted
  int32 arrays (24 KB), built once with a fixed seed by the new `equity.warmup()` (6 ms) or lazily.
  It ignores this board's texture, and equity gets no credit for draws (documented in the module
  docstring). Board widths other than 0/3/4/5 raise ValueError on this path (impossible game states).
- Cost: 15 us postflop, 0.6 us preflop, vs 62 us for 16 rollouts (ratio about 0.24).

**RED** (new tests on the base commit's code: `git archive df265bd` + new `tests/test_equity.py`):
```
$ .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_equity.py
   8 E       ValueError: cannot reshape array of size 0 into shape (0,newaxis)
   8 E   ValueError: negative dimensions are not allowed
   1 E       AttributeError: 'NoneType' object has no attribute 'random'
17 failed, 9 passed in 5.13s
```
(on top of `771e686` alone: 16 failed, iters=0 now giving `nan`, e.g. `assert nan == 0.41316667199134827`)

**GREEN:**
```
$ .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_equity.py -s
...............MAE 0.12483500817514005 corr 0.8523169176924253
26 passed in 5.42s
```
Tests added: `test_zero_iters_is_defined` (iters 0, -1, -256 x board widths 0/3/4/5: float in [0,1],
same value for every iters <= 0, `rng.bit_generator.state` unchanged); preflop zero budget equals the
table; `test_time_manager_zero_budget_feeds_the_equity_api` (a TimeManager past its hard stop gives
`iters()==0`, fed straight in); the estimate tracks exact enumerated strength on 28 spots (MAE < 0.15,
corr > 0.8; deterministic, so not flaky); royal > trips > pair on one river (exact 1.000 / 0.678 /
0.011); the zero-budget call is under 0.75x the 16-rollout call (measured 0.24x).

**Files:** `bot/equity.py`, `tests/test_equity.py`.

## I2: one hole against N boards returned one silently wrong value

**Root cause.** `_best_of` built the (rows, 10, T) index by broadcasting, then reshaped with
`len(holes)`. For holes (1,5) and boards (N,w) that is `reshape(1, -1)`: the max over all N boards,
shape (1,), no error. The reverse direction happened to work.

**Fix (`bot/fasteval.py`).** `_best_of(holes, boards, widths)` checks shapes and takes the row count
from both sides: equal counts pair row by row; a single row on either side pairs with every row of
the other; anything else raises ValueError. Reshape uses that count (zero rows now give an empty
int32 result instead of a reshape error). Explicit checks for all three entry points: holes must be
(N,5) (a 6-card hole silently ignored the sixth card); `omaha_batch` needs 5-card boards like
`omaha_score`; `made_now_batch` 3/4/5; `score5_batch` needs (N,5) ((N,4)/(N,6) returned garbage).

**RED** (`fix_i2_red.txt`):
```
$ .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_fasteval.py
   6 E       Failed: DID NOT RAISE ValueError
   3 E       assert (1,) == (40,)              # made_now, widths 3/4/5, one hole vs 40 boards
   1 E       assert (1,) == (60,)              # omaha_batch, one hole vs 60 boards
   1 E       assert [7811072] == [7811072, 1594400]   # quad kings, then a pair of aces
   2 E       IndexError: too many indices for array ...
   1 E       IndexError: tuple index out of range
   1 E       IndexError: index 4 is out of bounds for axis 1 with size 4
   1 E       ValueError: cannot reshape array of size 0 into shape (0,newaxis)
16 failed, 25 passed in 1.34s
```
**GREEN:** `tests/test_fasteval.py tests/test_equity.py tests/test_preflop.py`: `55 passed in 6.46s`.

Tests added: every supported row combination, (1,N), (N,1), (N,N), (1,1), against
`evaluator.omaha_score` for `omaha_batch` and against a `score_five` brute force for `made_now_batch`
at widths 3, 4, 5; a hand-built one-hole/two-board case (per board, not the max); zero rows; 7
unsupported `omaha_batch` shapes, 5 `made_now_batch` shapes and 4 `score5_batch` shapes must raise
a ValueError that names the argument at fault (follow-up `66a3d73`: a mutant removing the explicit
row-count check survived, because numpy's broadcast then raises its own, opaque, ValueError).

**Files:** `bot/fasteval.py`, `tests/test_fasteval.py`.

## I3: bot.* module state crossed halves, matches and candidates in the simulator

**Root cause.** `load_bot_file` gave the bot file fresh globals but every module it imported (our
`bot` package) came from the `sys.modules` cache, so module-level state survived into half 2, later
matches and the other candidate. The tournament runs each half in a fresh pod for exactly this reason.

**Fix (`sim/match.py`, ruling R22).** `FRESH_PACKAGES = ("bot",)` and
`load_bot_file(path, fresh_packages=FRESH_PACKAGES)`:
1. evict each listed package and its submodules from `sys.modules` before running the file, so its
   imports run fresh; two bots in one match each get their own copy;
2. afterwards restore `sys.modules` exactly, so the caller keeps its own copy (pickling of the
   caller's classes, `monkeypatch.setattr("bot.x...")` and `isinstance` stay consistent; without the
   restore, a tuner that imported `bot.policy` and later ran a serial match would hit "not the same
   object" PicklingErrors);
3. drop the `_simbot_*` entry once the file has run (it is registered while it runs, which
   dataclasses need). Unique module names are kept.

Why step 3: with R22 every load re-imports `bot` and its tables, and a `_simbot_*` entry kept in
`sys.modules` keeps that private copy reachable forever. Measured with a bot that imports
`bot.equity` and warms it (`i3/mem_probe.py`, peak RSS over 12 loads):
```
shared        42 42 42 ... 42           growth 0.0 MB/load   (old behaviour, but state leaks)
purge_keep    42 44 46 ... 64           growth 2.0 MB/load   (R22 literally, entry kept)
restore_keep  42 44 46 ... 64           growth 2.0 MB/load
restore_pop   42 55 55 ... 55           growth 0.0 MB/load   (shipped)
```
2 MB now; once agent.py loads the preflop (4.5 MB) and blueprint (about 25 MB) tables, keeping the
entry would grow a serial `paired_diff` (default `workers=1`) by about 30 MB per half. The shipped
`load_bot_file` stays flat (`i3/mem_real.py`: 48, 61, 61, ... 61 MB); a load costs 8.6 ms for that
bot (outside the timed region, like tournament startup), 0.2 ms for a light one. This also resolves
most of audit M4 (loaded LLM bots are no longer retained).

**RED** (`fix_i3_red.txt`):
```
$ .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_sim_match.py
E       assert [1, 2] == [1, 1]                       # two loads, shared bot.timing
E       assert [6, 6, 6, 6] == [1, 1, 1, 1]           # 2 halves x 2 bots, one shared module
E       AssertionError: assert (<module 'bot.timing' ...> is not <module 'bot.timing' ...>)
E       TypeError: load_bot_file() got an unexpected keyword argument 'fresh_packages'
4 failed, 14 passed, 1 deselected in 1.20s
```
**GREEN:** `tests/test_sim_match.py tests/test_zoo.py tests/test_tournament.py`: `48 passed, 1
deselected in 15.98s` (all 19 zoo bots load and play; spawn workers unaffected).

Tests added (fixture `tests/fixtures/leaky_bot.py` counts halves in a `bot.timing` attribute): two
loads see [1, 1]; `play_match` with the fixture on both sides gives four bots that each count 1 and
hold four distinct copies; the caller's `sys.modules["bot.timing"]` is untouched and the loaded
module is not left registered; `fresh_packages=()` restores the old shared behaviour.

**Files:** `sim/match.py`, `tests/test_sim_match.py`, `tests/fixtures/leaky_bot.py`.

## I4: 7 of 15 simulator bugs passed the default tests; F10 anchor deselected

**Root cause.** No test exercised several judge rules (each crashing hand result is one error even
when results are batched; on_hand_end time is charged; grace; illegal-but-valid actions; new_half
failure forfeits), the only test touching the button and deal pairing was slow-marked, and nothing
compared the simulator with the judge directly.

**Fix (tests only).**
- `tests/test_sim_match.py`: F10 unmarked (0.2–0.3 s) and now also pins the judge's sd 0.98525 and CI
  (-0.57707, -0.45493); identical bots must give sd 0; new targeted tests (c)–(g) from the audit,
  using the starter (it folds as button, so two results arrive in one request; traced and confirmed
  in both seats): three crashing results = three errors; 5 ms on_hand_end work is charged; a 10 ms
  request is free under a 50 ms grace; FOLD with nothing to call is one error and the default is
  played; a new_half that raises, or overruns a 20 ms timeout, forfeits (points 0, verdict B).
- `tests/test_judge_fidelity.py`, fast (default suite, about 4 s): `judge_engine.play_half` and
  `combine_halves` run unchanged; only `HttpAgentHandle._request` is replaced, by a call into
  **agent.py's own FastAPI app** in process (`fastapi.testclient`, two server copies, one per seat,
  reset per half like a fresh process). Compared with `sim.match` (grace = the judge's 1 ms):
  per-deal nets, errors, forfeits, calls and bank-out flags on 4 deterministic pairs x 2 halves; the
  error-mix fixture (5 errors, forfeit, batched results); 15 odd return values in 4 groups (rejected
  counts pinned: 4, 4, 0, 3); shrunk-timeout overruns and a slow new_half; tiny-bank bank-outs (flags
  only: the exact bank-out call is jitter, 10–15% in calls); and the combined mean, sd, CI, verdict
  and points.
- `tests/test_judge_fidelity.py`, slow (`-m slow`, 7.3 s): real uvicorn servers on free ports
  (`tests/fixtures/serve_bot.py` serves any bot file behind agent.py's plumbing) and
  `judge_engine.py --half 1/2 --json` plus `combine` as subprocesses; 4 matches (starter vs calling
  station on two seeds, the error-mix bot vs the starter: 5 errors and a forfeit, and
  `tests/fixtures/slow_once_bot.py`: one 1.05 s answer, 1 timeout error); per-deal nets, errors,
  forfeits, calls, bank flags and the full verdict compared, and the judge's error counts pinned.

**RED** (`fix_i4_red.txt`: the 15 audit simulator mutants against the two new test files):
every one KILLED, including the seven that survived before; e.g.
```
S1 hand_end errors counted once per request    KILLED  test_counted_errors_and_forfeit_match_the_judge[1], ..., test_each_crashing_hand_result_is_an_error_even_in_one_request[1], [2]
S3 on_hand_end time NOT charged                KILLED  test_bank_out_matches_the_judge[slow-hand-end], test_on_hand_end_time_is_charged_to_the_bank
S8 button always seat 0                        KILLED  test_deterministic_bots_match_the_judge[...] (8), test_match_verdict_matches_judge_combine[...] (2), test_matches_real_judge_value, ...
S9 grace not applied                           KILLED  test_grace_is_free_time_in_every_request
S10 illegal Action accepted                    KILLED  test_counted_errors_and_forfeit_match_the_judge[1], [2], test_illegal_action_is_an_error_and_the_default_is_played
S14 new_half failure does not forfeit          KILLED  test_new_half_failure_forfeits_the_match, test_new_half_past_the_timeout_forfeits, test_timeouts_match_the_judge[slow-new-half]
S15 half-2 deals paired with wrong deal        KILLED  test_identical_bots_chop, test_match_verdict_matches_judge_combine[...] (2), test_matches_real_judge_value
```
**GREEN** on the shipped simulator: `tests/test_sim_match.py tests/test_judge_fidelity.py`: `46
passed, 4 deselected in 5.23s`; `-m slow tests/test_judge_fidelity.py`: `4 passed, 20 deselected in
7.34s`.

**Files:** `tests/test_sim_match.py`, `tests/test_judge_fidelity.py`, `tests/fixtures/errmix_bot.py`,
`tests/fixtures/slow_once_bot.py`, `tests/fixtures/serve_bot.py`.

## I5: paired_diff's CI width was untested (the acceptance gate)

**Root cause.** Nothing pinned the CI arithmetic; a CI sqrt(deals) times too wide (T4) passed, as did
several standings bugs.

**Fix.** The CI math moves into a pure helper, `sim.tournament.mean_ci(d) -> (ci_low, mean,
ci_high)` (mean +- 1.96 * sample sd / sqrt(N)); `paired_diff` returns `mean_ci` of the pooled
per-deal differences (same results as before). Tests:
- `mean_ci` on hand-computed vectors: 1..8 gives 4.5 +- 1.6974097914175 (sd sqrt(6)); -1,1,-1,1 gives
  0 +- 1.1316065276116665; one value or a constant gives zero width.
- `paired_diff` (nit vs threshold_cat, 2 opponents x 2 seeds x 40 deals) against a recomputation that
  shares no code with it: per-deal chips straight from `play_match`, `statistics.stdev`, N = 160.
  Mean, lo, hi and width `2*1.96*sd/sqrt(160)` must match (`hi - lo` to rel 1e-9).
- `resolve_bot("agent").args[0] == "agent.py"` (T10).
- A crash/crash/calling-station round robin: BOTH_FORFEIT gives 0 points each (T2), best first (T7),
  forfeits per bot (T6), the loser's losses (T3), the sign flip of B's mean (T1). BOTH_FORFEIT losses
  are deliberately not asserted: audit M5 says they should count and that is pending the final review.
- A slow bot's `max_bank` is its own (T5; feeds the bank-use gate).

**RED** (`fix_i5_red.txt`, new tests vs the audit's tournament mutants, before the T5 test existed):
T1, T2, T3, T4, T6, T7, T8, T9, T10 KILLED, T5 SURVIVED; with the final test file
(`fix_i5_green_mut.txt`) all ten die, T4 as three re-expressions for the refactored code
(per-match means, no sqrt(N), ddof=0):
```
T4' paired_diff CI over per-match means (N=#matches)   KILLED  test_paired_diff_ci_is_pooled_over_every_deal
T4'' mean_ci half-width without sqrt(N)                KILLED  test_mean_ci_on_hand_computed_vectors, test_paired_diff_ci_is_pooled_over_every_deal
T4''' mean_ci population sd (ddof=0)                   KILLED  test_mean_ci_on_hand_computed_vectors, test_paired_diff_ci_is_pooled_over_every_deal
T5 max_bank attributed to the opponent                 KILLED  test_round_robin_max_bank_is_the_bots_own
```
**GREEN:** `tests/test_tournament.py`: `10 passed in 1.50s`.

**Files:** `sim/tournament.py`, `tests/test_tournament.py`.

## I6: strength_now's "made hand now" semantics was untested

**Root cause.** The only flop test (aces/kings full, > 0.9) and the river equality test cannot tell
current-board strength from full-runout equity (E12 passed all equity tests).

**Fix (test only).** Two flops with a weak made hand and a big draw (wrap plus the nut flush draw):
Ah Jh Td 9c 7s and Ah Jh Td 7c 6s on 9h 8h Kc. Exact strength over all C(28,5) = 98,280 opponent
holes with `made_now_batch`: 0.194 (middle pair) and 0.0045 (ace high). `strength_now` at 4,000
iterations must be within 4 SE of it, and at least 0.15 below `equity_vs_random` (0.644 / 0.572 at
40k iterations).

**RED** (`fix_i6_red.txt`): `E12 strength_now = full-runout equity  KILLED
test_strength_now_is_the_made_hand_not_the_runout[...]`; the unmutated tree passes (28 passed).
**GREEN:** `2 passed`.

**Files:** `tests/test_equity.py`.

---

## Mutation-catch table, before and after

Whole default suite (`pytest -q`, i.e. `-m "not slow"`), source-level mutants, one run each.
Before = `df265bd` (`fix_mut_before_default.txt`), after = `ab1d1b9` (`fix_mut_after_default.txt`).

Mutants are the audit's 44 (`mutants.py`). Where a fix changed the target's text, the same bug is
re-expressed for the new code (T4 as T4'/T4''/T4''', F2 as F2'); the original pattern is reported as
not applicable. "Killed" = at least one default test fails; F5 (int16 table) fails at collection,
which also fails the suite.

**Simulator (the audit's 15 for I4).** Before 8/15, after **15/15**.

| Mutant | Before | After | Caught after by (default suite) |
|---|---|---|---|
| S1 on_hand_end errors counted once per request | survived | **killed** | test_each_crashing_hand_result_is_an_error_even_in_one_request[1,2], test_counted_errors_and_forfeit_match_the_judge[1,2] |
| S2 on_hand_end delivered after choose_action | killed | killed | test_hand_results_delivered_before_next_action_and_flushed |
| S3 on_hand_end time not charged | survived | **killed** | test_on_hand_end_time_is_charged_to_the_bank, test_bank_out_matches_the_judge[slow-hand-end] |
| S4 no end-of-half flush | killed | killed | 18 tests, including every in-process judge comparison |
| S5 bank-limited overrun counted as an error | killed | killed | 4 tests (2 old, 2 judge comparisons) |
| S6 forfeit at 6 errors | killed | killed | 4 tests |
| S7 seats never swap in half 2 | killed | killed | 9 tests |
| S8 button always seat 0 | survived | **killed** | test_matches_real_judge_value (F10, now default), 8 judge-comparison halves, 2 verdict comparisons, ... |
| S9 grace not applied | survived | **killed** | test_grace_is_free_time_in_every_request |
| S10 illegal Action accepted | survived | **killed** | test_illegal_action_is_an_error_and_the_default_is_played, test_counted_errors_and_forfeit_match_the_judge[1,2] |
| S11 >1 s reply still used | killed | killed | 4 tests |
| S12 bank never limits the timeout | killed | killed | 4 tests |
| S13 empty bank never skips calls | killed | killed | test_overrun_of_remaining_bank_is_bankout_not_error |
| S14 new_half failure does not forfeit | survived | **killed** | test_new_half_failure_forfeits_the_match, test_new_half_past_the_timeout_forfeits, test_timeouts_match_the_judge[slow-new-half] |
| S15 half-2 deals paired with the wrong deal | survived | **killed** | test_identical_bots_chop, test_matches_real_judge_value, test_match_verdict_matches_judge_combine[x2] |

The slow real-HTTP test on its own (`fix_mut_after_slow_http.txt`) kills 8/15: S1, S4, S6, S7, S8,
S10, S11, S15. Its survivors are the bank and timeout paths (the CLI judge's 8 s bank and 1 s limit
cannot be shrunk), new_half failure and delivery order; the fast tests cover those.

**Tournament.** Before 2/10 (T8, T9), after **10/10**: T1, T2, T3, T6, T7 by
test_round_robin_both_forfeit_scores_nobody; T4' / T4'' / T4''' by
test_paired_diff_ci_is_pooled_over_every_deal (and test_mean_ci_on_hand_computed_vectors for the
last two); T5 by test_round_robin_max_bank_is_the_bots_own; T8, T9, T10 as listed in I5.

**fasteval / equity.** Before 18/19 (E12 survived), after **19/19**: E12 by
test_strength_now_is_the_made_hand_not_the_runout (which also catches E4 and E10); F2' by the new
row-combination tests and the existing test_made_now_on_flop_equals_bruteforce; every other F and E mutant is caught by the same tests as before plus, for
F1/F4, the new row-combination tests.

**Audit list overall: before 28/44, after 44/44.**

**New mutants aimed at the fixes themselves** (after only; `fix_mut_after_default.txt` on `ab1d1b9`,
N2 re-run on the follow-up in `fix_n2_followup.txt`): 11/11 killed.

| Mutant | Caught by |
|---|---|
| N1 one hole x N boards collapses to the max again (the I2 bug) | row-combination tests (1xN), test_one_hole_against_many_boards_is_per_board_not_the_max |
| N2 mismatched row counts accepted | the rejection tests, once errors must name the argument (survived on `ab1d1b9`: numpy's own ValueError) |
| N3 hole width not checked | 3 rejection tests |
| N4 score5_batch shape not checked | 4 rejection tests |
| N5 zero budget postflop is a flat 0.5 | test_zero_iters_estimate_tracks_exact_strength, test_zero_iters_ranks_hands_on_one_board |
| N6 zero budget strength_now silently samples 16 | test_zero_iters_is_defined (rng state), test_zero_iters_ranks_hands_on_one_board (rng=None) |
| N7 zero budget equity_vs_random silently samples 16 | test_zero_iters_is_defined (rng state), test_zero_budget_is_cheaper_than_the_smallest_rollout |
| N8 zero budget preflop ignores the table | test_zero_iters_preflop_is_the_preflop_table |
| N9 load_bot_file evicts nothing (the I3 bug) | all four I3 tests |
| N10 the bot's private copies left in sys.modules | test_load_bot_file_leaves_the_callers_modules_alone |
| N11 the _simbot_ module kept registered | test_load_bot_file_leaves_the_callers_modules_alone |

## Hot-path timing (omaha_batch, equity_vs_random)

**Interleaved A/B** (`fix_ab_timing.py`): the base commit's module exec'd next to the working tree's
in one process, alternating which runs first (an earlier version always ran "old" first and showed a
spurious 10% gain for "new": the second call of a pair runs on warm caches). Median (p99), us:

| case | old (df265bd) | new | new/old |
|---|---:|---:|---:|
| omaha_batch 10k x 10k | 3471.7 (3685.1) | 3468.3 (3569.5) | 0.999 |
| omaha_batch 128 x 128 | 62.3 (72.1) | 62.4 (66.5) | 1.003 |
| omaha_batch bcast(128,5) x 128 (equity's call) | 64.3 (67.1) | 64.5 (67.3) | 1.003 |
| omaha_batch 1 x 1 | 13.6 (18.0) | 13.8 (15.3) | 1.015 |
| made_now flop 128 | 24.6 (28.4) | 24.8 (26.5) | 1.008 |
| score5_batch 10k | 193.3 (205.2) | 191.3 (204.6) | 0.989 |
| equity pre 128 | 209.3 (230.4) | 209.5 (216.6) | 1.001 |
| equity flop 16 | 61.5 (72.2) | 61.6 (64.4) | 1.001 |
| equity flop 128 | 200.6 (208.0) | 196.4 (206.3) | 0.979 |
| equity flop 256 | 356.4 (368.7) | 356.5 (364.8) | 1.000 |
| equity turn 128 | 194.1 (206.2) | 192.5 (204.3) | 0.992 |
| equity river 128 | 192.0 (198.2) | 187.6 (197.8) | 0.977 |
| strength flop 128 | 122.6 (132.7) | 122.6 (127.9) | 1.000 |
| strength river 128 | 192.8 (204.7) | 195.8 (202.0) | 1.015 |

The added checks cost under 0.2 us per call: within noise everywhere.

**Sequential, same conditions** (`fix_hotpath_timing.py` on a `git archive df265bd` copy and on HEAD,
alternating base/head three times, `fix_timing_sametime.txt`), medians in us:

| case | base x3 | head x3 |
|---|---|---|
| omaha_batch 128 x 128 | 61.0, 61.0, 61.0 | 61.2, 61.0, 61.3 |
| omaha_batch 10k x 10k | 3284.8, 3296.9, 3280.9 | 3296.2, 3289.7, 3352.1 |
| equity flop 128 | 197.2, 198.0, 198.0 | 198.3, 198.3, 199.1 |
| strength flop 128 | 120.1, 120.8, 121.0 | 121.2, 121.5, 122.2 |

The session-start baseline (`fix_timing_before.txt`: equity flop 128 = 187–192 us) is 3–5% below
both of these, on every case including `score5_batch`, whose only change is an O(1) shape check:
that gap is machine drift over the hour between runs, not the fixes.

## Test suite

| | before (`df265bd`) | after |
|---|---|---|
| default `pytest -q` | 95 passed, 1 deselected, 19.4 s | 184 passed, 4 deselected, 24.7 s |
| `pytest -m slow` | 1 passed (F10), 0.35 s | 4 passed (real-HTTP judge comparison; F10 moved to default), 7.4 s |

The 1 warning is pre-existing (`StarletteDeprecationWarning` from `fastapi.testclient`, also imported
by `test_agent.py`).

## Concerns and notes for downstream tasks

1. **The zero-budget estimate is rough, by design** (I1). Mean error about 0.1 against exact
   strength, and texture-heavy spots can be far off (a straight on a four-flush river reads strong).
   Postflop `equity_vs_random(..., 0)` equals `strength_now(..., 0)`: draws get no credit. So at
   `iters == 0` Task 10's `eq` and `now` coincide and Task 16's 2-D bucket falls on the diagonal.
   Acceptable for the emergency regime (6 s hard stop, heavy pacing overshoot); if it matters, a
   texture-aware reference measured about 25% better.
2. **Warmup for Task 11.** agent.py's warmup should call `bot.equity.warmup()` (fasteval table plus
   the zero-budget reference, about 10 ms) and `bot.preflop.load_preflop()`. With R22 every simulated
   half re-imports `bot`, so anything left lazy is rebuilt inside the first timed request of each
   half, just as in a fresh tournament process.
3. **Task 10 test still owed.** The audit asked for a test that drives `compute_features` with a
   TimeManager returning 0; `compute_features` does not exist yet. I added the TimeManager-to-equity
   contract test; Task 10 should add the `compute_features` one.
4. **R22 covers bot files only.** A bot built in-process (for example Task 12's
   `functools.partial(PolicyBot, params)`) shares the process's `bot.*` modules across halves,
   matches and candidates. R22's invariant must hold in code: module-level state in `bot/` is
   read-only tables, per-half state lives in `memory` or the instance. Suggest Task 11 adds a test for
   it. Also, bot code must import at module top: an import that first runs during play resolves
   against the caller's `sys.modules`, not the bot's private copy.
5. **`load_bot_file` no longer leaves `_simbot_*` registered after the file has run** (step 3 of
   I3), a deliberate change to stop each load's private package copy staying reachable. Bot code that
   needs `sys.modules[__name__]` at run time (pickling its own classes, `typing.get_type_hints`)
   would now fail; none of the 19 zoo bots, the baselines or agent.py do (zoo tests pass). Side
   effect: audit M4 (memory growth per load) is largely resolved; worth noting at the final review.
6. **Commit order.** `771e686` (I2) alone makes `iters=0` return NaN; `817a3d3` (I1) fixes it. Do not
   cherry-pick the first without the second.
7. **Fidelity limits of the in-process judge** (I4). The deadline is checked after the call returns
   (it cannot abort a running request), and `TestClient` adds about 0.5 ms per request on the judge
   side, so bank-out cases compare flags, not nets or call counts (call counts differ 10–15% there).
   Audit M3 (a multi-second stall costs several errors in the real judge, one in the sim) is still
   not modelled. The real-HTTP test needs free local ports and subprocesses; it is slow-marked
   (7.4 s) as the brief asked.
8. **M5 left alone.** BOTH_FORFEIT is still not counted as a loss; the new round-robin test leaves
   that unasserted so it does not pin the bug.
9. **Timing tests.** `test_zero_budget_is_cheaper_than_the_smallest_rollout` is relative (median of
   300 interleaved calls; threshold 0.75x, measured 0.24x), so it is robust but still timing-based.
   The default suite grew from 19.4 s to 24.7 s.
10. **Not done (out of scope).** `strength_now` still scores our own hand `iters` times; with the I2
    broadcast it could score it once (worth roughly 15–20% of `strength_now` at 128 iterations).
