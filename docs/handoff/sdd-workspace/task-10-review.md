# Task 10 review: features and heuristic policy (BASE 66a3d73, HEAD 08e541f)

Reviewer: opus/max, read-only. Reviewed at branch tip 85a038e; the two later commits, ccdd6ac and 85a038e, are outside this range. Machine: the user's PC (WSL2, Ryzen 7 9700X 8C/16T, Python 3.12.3, numpy 2.5.4). It was quiet: load average 0.06 at the start, and nothing else ran during the timing work. Scratch scripts and raw outputs are in `/tmp/claude-1000/-home-kchen-code-heads-up-bundle/c0d55f53-d291-4faf-88d2-40523d153176/scratchpad/t10-review/`.

### Spec Compliance
- ✅ **Spec compliant.**
  - `Features` (bot/features.py:71-106) has the spec's 9 fields in order, with the spec's types, and is frozen. The 4 added fields all have defaults.
  - `compute_features(obs, iters, rng)` (bot/features.py:116) uses the table preflop and Task 7's functions postflop.
  - `HeuristicParams` (bot/heuristic.py:30-69) keeps all 8 spec fields at the spec's defaults, and `test_heuristic_params_defaults_are_the_briefs` pins them. Nothing is renamed or removed.
  - `heuristic_action(obs, feats, params, rng, opp=None)` (bot/heuristic.py:141) has the spec's signature.
  - All 4 mandated tests are present, and so is the owed `compute_features(iters=0)` test (`test_compute_features_iters_zero`). The 1,000-deal tests carry `@pytest.mark.slow` (R11). The factory pickles (R3). No mutable module state was added (R22).
  - **Extra (justified):** at the default parameters, the facing-bet rule is no longer the brief's linear formula. It mixes in the range model, and the defaults add thin value bets, a lower thin-value bar in position, and a preflop raise discount (`range_weight=0.7`, `thin_value`, `ip_thin_shift`, `pre_discount`).
    - This is additive and documented.
    - It was required to meet the controller's own targets under R20: v0, the literal rule set (0269604), loses to llm_ml_minded and fails target (b).
    - The brief's postflop rule comes back exactly with `range_weight=0, thin_value=(1,1,1), ip_thin_shift=0`. Preflop, `pre_discount` stands in for the brief's (unspecified) preflop discount.
    - I re-measured that the deviation pays on fresh seeds: v1 − v0 = **+0.202 [+0.041, +0.364]** bb/deal pooled, and W/L/C goes 26/1/3 → 30/0/0.
- ⚠️ **Cannot verify from the diff (rulings and checks for the controller):**
  1. **Record a ruling that the heuristic's default behaviour is v1,** not the brief's literal rules (see the "Extra (justified)" item above for how to recover the literal rules). Also record a ruling on the deviation from the T9 carry-forward note: `nut_flags` still runs when `iters == 0` (Minor M1 below).
  2. **T11 warmup has to absorb the first-decision spike.** It lives in unchanged T7 code: the first `np.unique` call inside `bot.equity.nut_flags` (bot/equity.py:156) takes 3.4-4.0 ms here and about 11 ms on the Mac. Calling `nut_flags` once per board width (3/4/5) in warmup removes it: I measured 0.004 ms afterwards.
  3. **T11/R30a must feed `grace_for_cpu_factor`** into the TimeManager. In my 5×-CPU emulation the default grace let the bank reach **7.1 s** (the CPU-scaled grace gave 5.0 s). This is T9/T11 territory, but the heuristic's fixed cost is part of that budget.
  4. **T11's layered decide() must catch exceptions from this layer.**
     - `compute_features` raises `ValueError` for a board of 1, 2 or 6+ cards.
     - It raises `IndexError` for card ids ≥ 36. Its docstring says "ValueError only", which is slightly too strong.
     - `range_equity` raises `ValueError` for NaN `bet_range`/`raise_step` (Minor M3).
     - The engine never produces any of these.
  5. **T14 threshold mapping.** In v1, unopened betting is driven mostly by `thin_value` (0.62-0.66), not `value_raise` (0.70-0.80). Facing bets, `range_weight`/`bet_range` dominate over `call_margin`. So:
     - The plan's "station: value threshold −0.08" must shift **both** `value_raise` and `thin_value`.
     - The "maniac: call threshold −0.06" should mainly lower `range_weight`/`bet_range`. The report measured `range_weight` 0.5 at +1.3/+1.4 bb/deal against always_half/maniac.

### Strengths
- **Legal by construction.** Raises are requested only under `can_raise`, folds only when something is owed, and `_legal` adds a raise → other raise → call fallback as a second line.
  - Fuzz: 17,365 engine observations, including 709 opponent-all-in spots and 218 raise-capped spots. Each was run against 6 extreme parameter sets plus 40 random ones (thresholds in [−9, 9], `range_weight` up to 1e6, ±inf `bet_range`). Every 7th observation was also run in 13 corrupted shapes: `legal==(CHECK_CALL,)`, my_stack/opp_stack/pot = 0, to_call −5 or 1e6, `raises_this_street` −3 or 99, `big_blind=0`, empty history, reversed legal list.
  - Result: **0 exceptions, 0 illegal actions, 0 folds when checking is free**.
  - Retuned engine constants (stack/blinds/cap = 100/2-4/3, 60/1-2/2, 200/5-10/4, 40/2-4/6; 15,777 observations × 4 parameter sets): **0 errors**. The layer reads sizes only from `obs` (Review Focus 1).
- **Very cheap.**
  - Whole decision: p50 0.22-0.31 ms, p99 about 0.6 ms in a real half.
  - Bank: 0.001-0.004 s per half solo, ≤ 0.052 s in 8-worker evaluations.
  - Review Focus 3 holds: at 3× CPU with 3× plumbing the bank stays at 2.0-3.1 s.
- **`iters == 0` behaves as specified.** It draws nothing from `rng` (tested, and the mutant is killed) and its fixed cost stays inside the ≤ 0.3 ms budget.
- **Strength claims reproduce on a quiet machine and on fresh seeds.**
  - Targets (a) and (b) are met on the report's seeds s1-s3 and again on fresh seeds q1-q3.
  - HOLDOUT goes 12/0/0 (information only).
- **The range model is reproducible bit-for-bit.** `python -m train.fit_range_model --spots 100000 --workers 6` (33 s here) reproduced all 30 rows of `_RANGE_COEF` exactly, with a different worker count, machine and numpy than the implementer's. The table is a read-only tuple.
- **The experiments were done properly.** Paired, documented A/B runs (E1-E4) decided each change, rejected parameters were removed rather than left as dead switches, and the report is thorough and honest about load effects.

### Issues

#### Critical (Must Fix)
None.

#### Important (Should Fix)
None. I found nothing that can raise, play an illegal action, time out or forfeit in a real half. No interface mismatch, no budget violation, no test that misses a real bug, and no measurable strength bug.

#### Minor (Nice to Have)
- **M1. bot/features.py:129: `nut_flags` runs even when `iters == 0`,** against the Task 9 carry-forward note ("skip expensive features (nut_flags, MC) when iters()==0").
  - Why it matters: `iters == 0` means bank pressure. On this PC, `nut_flags` is the largest fixed cost (p50 0.066 ms, p99 0.199 ms), which puts the `iters == 0` path at river p99 0.287 ms, just under the ≤ 0.3 ms target (the report measured 0.14 ms on the Mac). On a 3× CPU in the starved regime it is charged at roughly 0.2-0.6 ms per postflop decision. It is documented and justified: the nut bits are what keep the bot from folding the river nuts and drive the non-nut penalty.
  - Fix: accept it with a controller ruling, and defer to T11, which should either (a) add a critical-bank mode, for example `used > 0.875·bank`, that skips `compute_features` and plays the cheap path, or (b) add a keyword `compute_features(..., nut=True)`.
  - Optional (T7 code): replace `len(np.unique(br)) < w` at bot/equity.py:156 with a Python set check. That removes the first-call spike and some per-call cost.
- **M2. bot/heuristic.py:72-89: `_legal`'s fallbacks are never exercised.** In mutant M1-mut, `_legal` returns `want` unchecked, and it passes all 23 tests in tests/test_heuristic.py and tests/test_heuristic_bot.py.
  - Why it matters: the policy never asks for an illegal action (because of `can_raise` gating), so the fallback is dead in practice. The report's claim that it is "tested on 10k engine observations" overstates what is tested. A later edit (by T12 or T14) that requests a raise without `can_raise` would rely on an untested helper.
  - Fix: add a direct unit test with synthetic legal tuples:
    - want POT, legal (FOLD, CALL) → CALL
    - want HALF, legal (CALL, POT) → POT
    - want FOLD, to_call 0 → CALL
    - want CALL, legal (FOLD,), to_call > 0 → FOLD
- **M3. bot/features.py:57-58: NaN parameters raise.** `int(nan)` raises `ValueError` when `bet_range` or `raise_step` is NaN. A tuple shorter than 3 raises `IndexError` on later streets.
  - Why it matters: T12's random search and T14's shifts could produce such values by accident. T11's try/except would catch it, but the whole heuristic layer would be lost for that decision.
  - Fix: T12 validates parameter sets (finite values, 3-tuples), or add a `HeuristicParams.__post_init__` check. Defer to T12.
- **M4. tests/test_heuristic.py: several strategy branches have no test.** These mutants all survive:
  - bluffing out of position (spec: "in position when checked to") at bot/heuristic.py:136
  - POT and HALF swapped on the value raise at bot/heuristic.py:133
  - `ip_thin_shift` applied out of position at bot/heuristic.py:134
  - the preflop discount ignored at bot/heuristic.py:107
  - `nonnut_penalty` dropped when facing a bet (a spec'd behaviour) at bot/heuristic.py:125-126
  - the river nuts checked instead of bet when unopened at bot/heuristic.py:130

  Why it matters: T12 will rewrite thresholds and could break a branch silently. These behaviours are validated only by sims.

  Fix: add one small test per branch, for example `bluff_freq=1` with a weak hand: IP → HALF, OOP → CHECK_CALL; and `nut=4` vs `nut=0` at a boundary eq flips call → fold.
- **M5. bot/features.py:139-142, 149-150 (docstring at line 90): the added fields are untested, and `facing_frac`'s docstring is wrong when facing a raise.**
  - Mutants that double `facing_frac` or never count `opp_raises_before` survive. These fields are unused today, but T11/T14 may read them.
  - When we bet b into P0 and are raised to B, the denominator `pot − to_call` = P0 + 2b, not "the pot before the opponent's last bet" (P0 + b).
  - Fix: assert all 4 added fields in `test_feature_values_from_history`, and correct the docstring or the formula.
- **M6. bot/heuristic.py:124-126: the non-nut penalty is counted twice on flush-possible boards.**
  - The range model's fitted `d` term already penalizes "flush possible, not the nuts" (−0.8 to −1.3 logits at t ≥ 0.4), and the full `nonnut_penalty` is then subtracted again.
  - Measured thresholds facing a half-pot flop bet: call at eq ≥ 0.707 on flush boards, against 0.407 on plain boards and 0.503 on paired boards (where only the penalty applies). Pot-sized bet on the river: call at eq ≥ 0.769 (flush board) vs 0.716 (plain).
  - This may be fine (it is tunable), but the interaction is undocumented.
  - Fix: document it, and have T12 tune it, for example by applying the penalty to the `(1 − range_weight)` linear share only on flush boards. Defer to T12.
- **M7. bot/features.py:128: `now` is computed by Monte Carlo but nothing uses it.** `strength_now` costs p50 0.047-0.123 ms (p99 up to 0.276 ms) per flop/turn decision at 16-256 iters, about 20-30% of the decision's compute. The report's reason to keep it ("for Task 16's 2-D buckets") is out of date: the T16 ruling says `bucket()` uses deterministic exact features and ignores iters/rng.
  - Fix: defer to T12 as an ablation. Compute `now` with the 0-iteration estimate (or lazily) and give its rollouts to `eq`.
- **M8. Evaluation hygiene (process; for T12).**
  1. **The dev seeds look slightly favourable.** Every E1-E4 decision was made on s1-s2. On fresh seeds q1-q3, the same code on the same quiet machine scores TRAIN mean **+2.619**, against **+2.960** on s1-s3. Most opponents score lower.
  2. **Runs are not reproducible from one run to the next.** Our TimeManager and the time-aware opponents both adapt to wall-clock time. Two runs with the same code and the same q-seeds gave llm_gto_inspired q3 at +0.87 and +0.27, and random q2 at +4.65 and +3.96.
  3. Fix: T12 should tune on a broad seed pool, confirm on disjoint fresh seeds, and either interleave A/B runs (as the implementer did) or tune with a fixed-iteration (deterministic) clock.
- **M9. Small accuracy and quality nits.**
  - bot/heuristic.py:105: the comment says each raise narrows the range "by less every time", but the steps are 0.07, then +0.035, +0.035 (constant after the first raise). The 0.5 slope is an unnamed constant.
  - bot/heuristic.py:59-67: the tuple fields could be typed `tuple[float, float, float]`.
  - sim/heuristic_bot.py:10: the docstring still says `--workers 4` (R32 says 8 on the PC).
  - train/fit_range_model.py:92: `from pathlib import Path` sits inside `main`.
  - The model is fitted on 128-rollout equities, but at `iters == 0` it is applied to the texture-blind estimate. That mismatch only matters in the starved regime; note it for T12/T16.
  - The report's diagnosis of concern 2 is wrong: the `_PAIR_CACHE`/`_TRI_CACHE` build costs 0.05 ms, and the first `np.unique` call costs 3.4-4.0 ms. The remedy it proposes still works.

### Assessment
**Task quality:** Approved
**Reasoning:** The interface matches the spec exactly. Every path I could reach, including heavy fuzzing, extreme and NaN-free random parameters, all-in corners, corrupted observations and retuned engine constants, returns a legal action without raising. Timing and bank use are well inside budget, Review Focus 3 holds at 3× CPU, and the strength targets (a), (b), (c) reproduce on a quiet machine and on fresh seeds. The remaining items are Minors: test gaps, one documented carry-forward deviation that needs a ruling, and tuning notes. They can be deferred to T11/T12/T14.

## Measured numbers vs the report

| Claim | Report (Mac, loaded) | Reviewer (PC, quiet) |
|---|---|---|
| (a) 4 baselines × 3 seeds × 1,000 deals, CI wins | 12/12 (s1-s3), lowest CI low +0.43 (always_pot s2) | **12/12 on s1-s3** (lowest CI low +0.43, always_pot s2). **12/12 on fresh q1-q3** (lowest +0.89, always_pot q3). |
| Baseline means (s1-s3) | random +4.03, calling_station +7.40, always_half +4.33, always_pot +3.25 | +4.24, +7.36, +4.22, +3.25 |
| (b) TRAIN mean bb/deal > 0 vs every opponent | 15/15, mean +2.930, 43/0/2, min ml_minded +0.81 | **s1-s3:** 15/15, mean **+2.960**, 44/0/1, min ml_minded +0.662 (s1 chop). **Fresh q1-q3:** 15/15, mean **+2.619**, 41/0/4, min ml_minded +0.384, gto_inspired +0.507. |
| Thinnest margins over 14 seeds (s1-3, q1-3, m1-8) | ml +0.81, gto +0.97 (3 seeds) | llm_ml_minded **+0.56**, W/L/C **6/0/8**. llm_gto_inspired **+1.07**, **12/0/2**. No losses. |
| HOLDOUT (information only) | 12/0/0, +1.791 | 12/0/0, **+1.754**, max bank 0.031 s |
| v1 − v0 (paired; TRAIN; fresh p1-p2) | v0 +2.344 (36/1/8) → v1 +2.930 (43/0/2), unpaired | **Pooled +0.202 [+0.041, +0.364]**. v0 26/1/3 → v1 30/0/0. Better vs 11 opponents (10 significant). Worse vs always_half −2.37 and maniac −3.13 (both still large CI wins). |
| (c) errors | 0 in 57 matches | 0 in 139 CLI evaluation matches (45 + 45 + 21 + 12 + 16), plus 4 traced solo halves and 8 slow-CPU halves |
| (c) max bank per half | 0.23 s (4 workers, load 7.3); ≤ 0.020 s solo | **≤ 0.052 s** (8 workers); **0.001-0.004 s** solo |
| Max single request in a half | 11.3 ms (first postflop decision) | 4.0-4.3 ms solo (first postflop decision: `np.unique` first call + a GC pass); 6.2-9.8 ms under 8 workers. Every other call ≤ 1.05 ms. |
| Whole decision in a real half | n/a | p50 0.22-0.31 ms, p99 0.60-0.65 ms |
| Fixed cost at `iters == 0`, p50/p99/max (ms) | flop 0.058/0.063, turn 0.083/0.094, river 0.127/0.136, max 0.143 | preflop 0.004/0.015/0.085, flop 0.068/0.203/0.406, turn 0.102/0.254/0.432, river **0.159/0.287**/0.607 |
| `iters == 0` components, p50/p99 (ms) | `nut_flags` about 0.11 | `nut_flags` 0.066/0.199, deterministic eq 0.018/0.072, preflop lookup 0.002/0.008, `heuristic_action` 0.003/0.011 |
| 256 iters, p50/p99 (ms) | flop 0.605/0.633, turn 0.672/0.693, river 0.457/0.478 | flop 0.413/0.781, turn 0.462/0.871, river 0.355/0.656 |
| Unused `now` (`strength_now`), 16/64/256 iters, p50 (ms) | 0.05-0.2 | 0.047 / 0.062 / 0.123 (p99 0.136 / 0.170 / 0.276) |
| Review Focus 3: 3× CPU, plumbing 3× (sim grace 0.13 ms) | n/a | bank **2.0-2.7 s** with CPU-scaled TM grace, 2.3-3.1 s with default grace, 0 errors. At 5×: 4.3-5.0 s and 5.4-7.1 s. |
| Startup | n/a | import 7.3 ms; `equity.warmup()` + `load_preflop()` 22.9 ms |
| Test suite | 207 passed, 8 deselected; slow 4 passed | 207 passed, 8 deselected (38 s); `-m slow tests/test_heuristic_bot.py` 4 passed |
| Range model regenerates deterministically | yes (3 workers, about 2 min) | **30/30 rows identical** (6 workers, 33 s) |
| Mutation check (17 hand-made mutants) | n/a | 8 killed: river-nuts guard, pot odds, raise counting, rng draws at iters 0, in_position, nut-bonus sign, range_weight, spr. 9 survived (M2, M4, M5 above). |

## The report's concerns: fix now or defer

| # | Concern | Decision | Reason |
|---|---|---|---|
| 1 | Measurements ran on a loaded machine | **Resolved here** | Re-measured on the quiet PC: TRAIN +2.960 (s1-s3) and +2.619 (fresh q1-q3), HOLDOUT +1.754, bank ≤ 0.052 s, 0 errors. Fresh-seed numbers are the shipping baseline. T12 should keep confirming on disjoint fresh seeds (M8). |
| 2 | First-decision spike, a T11 warmup item | **Defer to T11** | Measured 4 ms here, about 11 ms on the Mac: once per half and well inside every budget. The proposed remedy (warmup calls `nut_flags` for widths 3/4/5) works, but the root cause is the first `np.unique` call (bot/equity.py:156), not the cache build. Optionally T11 swaps it for a set check. |
| 3 | Points-first trade-off: margin given up against always_half and maniac | **Defer to T13/T14** | Confirmed (−2.37 and −3.13 bb/deal vs v0), yet still large CI wins. The exploit layer should lower `range_weight`/`bet_range` against high-bet-frequency or uninformative bettors. |
| 4 | Thin margins against llm_ml_minded and llm_gto_inspired | **Defer to T12 (tuning) and T14 (adaptive)** | Confirmed on 14 seeds: ml_minded +0.56 with 8 of 14 chops, never a loss. There is no obvious static fix, and it needs opponent modelling against an adaptive bluffer. |
| 5 | `now` computed but not used | **Defer to T12** | It costs about 20-30% of flop/turn compute. The "keep it for T16" reason is out of date (T16's `bucket()` ignores iters/rng). Ablate it in T12 (M7). |
| 6 | The range model assumes "equity vs random ≥ t" betting ranges | **Defer to T14** | `range_weight` is the valve. The exploit layer switches it per opponent class. |

## Evidence (commands; scripts in the scratch dir)
- Strength runs: `.venv/bin/python -m sim.heuristic_bot` with `--deals 1000 --workers 8` in every case.
  - `--opponents train --seeds 3`
  - `--opponents train --seeds 3 --seed-prefix q`
  - `--opponents holdout --seeds 3`
  - `--opponents llm_ml_minded,llm_gto_inspired --seeds 8 --seed-prefix m`
  - a fresh-q run against the baselines plus 3 equity bots

  Outputs: `train_s.txt/json`, `train_q.txt/json`, `holdout_s.txt/json`, `thin_m.txt/json`, `fresh_q.txt/json`.
- v1-vs-v0 comparison: `pkg/paired.py`. It uses v0 = `git show 0269604:bot/heuristic.py` with today's features.py; features.py only gained the range model between the two.
- Fuzz: `fuzz.py`. Retuned engine constants: `retuned.py`. NaN probe: inline.
- Timing: `timing.py`, `spike.py` (plain vs nut-warm), and `pkg/maxcall.py` (per-call trace and GC callbacks in a real half), all pinned with `taskset`. Slow CPU: `pkg/slowcpu.py` (busy-waits (f−1)× the measured compute; sim grace 0.001 − 0.00029·f).
- Mutants: `mutate.py`. Runs on `git archive HEAD` copies under `mut/`; the real checkout is untouched.
- Range model: `train.fit_range_model --spots 100000 --workers 6 --out <scratch>/range_model.npz`, compared row-for-row with `bot.features._RANGE_COEF`.
