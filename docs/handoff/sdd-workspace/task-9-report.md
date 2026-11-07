# Task 9 report: time manager

Commit d5c7812 `feat(bot): pot-weighted time manager` (branch kevin, not pushed).

## Implementation (bot/timing.py, stdlib only)

`TimeManager(bank_s=8.0, hands=1000, target_s=3.0, clock=time.perf_counter)`. It uses `__slots__`, has no globals, and is built for the per-half `memory` dict.

**Methods and attributes**
- `begin()` / `end()` wrap choose_action and charge `max(0, dt - 0.00075)`. A stray `end()` with no `begin()` does nothing.
- `add(dt)` charges on_hand_end time in full, with no grace (R9). Values of 0 or below, and NaN, are ignored.
- `used` holds the charged seconds. `clock` is public, so tests can inject a fake clock. `stop_s` is the hard-stop threshold, 6.0 at the default bank.

**`iters(pot, street, hand_index)`**
- Ladder: (0, 16, 32, 64, 128, 256).
- Base level: 32 when pot <= 8, 64 when pot <= 30, 128 above that.
- `projected = used * hands / (hand_index + 1)`. A hand_index of 0 or less is treated as 0, so the division is always safe.
- If projected > target_s, drop `max(1, ceil(overshoot / 20% - 1e-9))` levels. The epsilon stops an exact 20%, 40%, ... boundary from rounding up an extra level.
- If projected < 0.5 * target_s, raise one level.
- Hard rule: return 0 if `used > stop_s`.
- `street` is accepted for future use and ignored. The brief does not justify a river boost, so there is none.

**Speed** (measured): `iters()` takes 0.04–0.14 µs per call, loop included; the overshoot branch is the slowest. `begin()` + `end()` together take 0.12 µs, and construction takes 0.10 µs. The spec limit is 5 µs.

**Interpretations to check**
1. **"Drop one level per 20% overshoot" means per *started* 20% (ceil).** So 10% over drops 1 level and 30% over drops 2. Ceil gives the `if projected > target_s` guard a purpose, and it is the more conservative reading. With floor, nothing would happen until 20% over.
2. **The hard stop scales with `bank_s`.** It is `6.0 * bank_s / 8.0`, which is exactly 6.0 at the default bank. Review Focus 1 says organizers may retune the bank. A fixed 6.0 would never fire if the bank were 4 s. Without this, `bank_s` would be an unused parameter.
3. **The constructor raises `ValueError`** if bank_s, hands or target_s is not positive. This only catches configuration bugs; valid constants never raise.

## Simulated half (fake clock, judge-style charging)

**Model** (`tests/test_timing.py`)
- 1,000 hands × 4 decisions.
- Pot mix measured with a probe over 7 zoo matchups: 26,848 decisions, about 3,835 per half. Overall, 68.5% of pots were ≤ 8, 19.7% were 9–30, and 11.8% were > 30. The small share ranged from 33% to 98.5% depending on the matchup.
- Decision cost = `BASE + iters × cost_per_iter`:
  - NOMINAL = 0.2 ms / 128 per rollout. Measured: `equity_vs_random` costs 43 µs fixed plus about 1.2 µs per iteration, so 128 iters is about 0.20 ms.
  - BASE = 0.25 ms of non-MC compute. This is an estimate: nut_flags costs 0.04–0.11 ms and strength_now 0.06–0.2 ms.
  - HAND_END = 20 µs.
- Judge charge per request = `max(0, work + 0.23 ms HTTP - 1 ms)`, where work is the previous hand's on_hand_end plus this decision. It is computed independently of `tm.used`.

**Results** (seed 0; seeds 1 and 2 give totals within ±0.003 s). *Superseded:* this table held HTTP plumbing at 0.23 ms at every CPU speed. Fix round 1 below has the table with plumbing slowed.

| clock | judge-charged | tm.used | small pots (≤8) | medium (9–30) | big (>30) | big pots, last 100 hands |
|---|---|---|---|---|---|---|
| 1× | **0.000 s** | 0.020 s (all on_hand_end) | 64: 100% | 128: 100% | 256: 100% | 256 ×43 |
| 3× | **1.421 s** | 1.501 s | 32: 46%, 64: 54% | 64: 46%, 128: 54% | 128: 44%, 256: 56% | min 128, mean 196 |
| 5× | 2.923 s | 3.003 s | 16: 84%, 32: 16% | 32: 83%, 64: 16% | 64: 84%, 128: 16% | min 64, mean 73 |
| rollouts 50×, rest 1×, paced | 3.548 s | 3.602 s | 0: 53%, 16: 47% | 16: 52%, 32: 47% | 32: 51%, 64: 49% | min 32, mean 48 |
| rollouts 50×, **unpaced** baseline | 29.085 s | 29.165 s | 64: 100% | 128: 100% | 256: 100% | 256 |

Mean rollouts per decision: 99.7 at 1×, 76.8 at 3×, 28.9 at 5×, and 15.6 for 50× paced.

**What the table shows**
- **1×:** every decision fits inside the free grace. The manager is always "behind pace" and grants one level above base.
- **3×:** it settles at the raise boundary, 0.5 × target = 1.5 s.
- **5× and the 50× stress:** it holds near target_s.
- **Accounting (corrected in fix round 1):** this claim was wrong. It held only because the model kept HTTP plumbing fixed at 0.23 ms. With plumbing slowed like the rest of the CPU, the default grace under-counts the judge by 0.70 s at 2×, 1.76 s at 3× and 3.60 s at 5×. See Fix round 1.

**Sensitivity to non-MC cost per decision** (BASE on the dev laptop; this is the main bank risk)

| slowdown | BASE 0.25 ms | 0.40 ms | 0.50 ms | 0.70 ms | 0.90 ms |
|---|---|---|---|---|---|
| 3×, 4 decisions/hand | 1.42 s | 2.72 s | 3.45 s | 5.38 s (MC → 0) | 7.78 s |
| 3×, 5 decisions/hand | 1.40 s | 2.90 s | 3.88 s | 6.71 s | **9.71 s** |
| 5×, 4 decisions/hand | 2.92 s | 5.02 s | 7.02 s | **11.0 s** | **15.0 s** |
| 5×, 5 decisions/hand | 3.48 s | 6.25 s | **8.75 s** | **13.8 s** | **18.8 s** |

## Tests (tests/test_timing.py, 9 tests, 0.03 s)

- **Brief's three tests:**
  - `test_slow_clock_stays_under_bank` (3×). It asserts:
    - judge total < 8.0;
    - `tm.used >= total` (fix round 1: now asserted at 1× only, where it is true);
    - late big pots get more than 0 iterations at 3×;
    - at 1×, at least 20 late big pots, all with more than 0 iterations.
  - `test_big_pot_gets_more_iters_than_small`: small < medium < big at three pace states and all streets, plus exact values at pots 8/9/30/31.
  - `test_zero_iters_after_6s`: also asserts iterations are nonzero just before 6 s, with a generous target so only the hard rule can stop them. This makes the test non-vacuous.
- **Required:** `test_on_hand_end_time_counts`.
- **Added:**
  - `test_pacing_holds_bank_when_rollouts_are_far_slower`: unpaced uses more than 8 s; paced uses less than 8 s.
  - `test_pace_moves_one_level_per_20_percent`: exact ladder values for 10%, 30%, 50% and 70% overshoot, both boundaries, and the hand_index projection.
  - `test_hard_stop_scales_with_retuned_bank`
  - `test_begin_end_charges_only_beyond_grace`
  - `test_iters_under_5us`

`simulate_half`, `NOMINAL` and `FakeClock` are test-local helpers.

## TDD evidence

- **RED 1:** collection error `ModuleNotFoundError: No module named 'bot.timing'`.
- **RED 2:** against an interface-only stub (no accounting, `iters` → 0), 8 failed and 1 passed. The pass was the speed-only test. Failure reasons:
  - `assert 0.0 >= 0.0399...` (`used >= total`)
  - `assert 0.0 > 8.0` (unpaced baseline)
  - `0 < 0`
  - `[0, 0, 0] == [64, 128, 256]`
  - `iters(...) > 0` just before the stop
  - the grace/add `approx` checks

  So neither "stays under bank" nor "zero after 6 s" can pass with an always-0 implementation.
- **GREEN:** 9 passed. Full default suite: 89 passed, 1 deselected (slow), 1 pre-existing Starlette deprecation warning, about 20 s. I ran it before committing (it was also green before a final docstring-only edit).
- **Python 3.12:** both files parse with `ast.parse(feature_version=(3, 12))`. No 3.12 interpreter is available locally.

## Files changed

- `bot/timing.py` (new, 89 lines)
- `tests/test_timing.py` (new, 197 lines)

`agent.py` is not wired; that is Task 11.

## Self-review

- **Spec values are verbatim:** ladder, pot thresholds 8 and 30, base levels 32/64/128, the 20% step, the 0.5 × target raise, the 6.0 s stop, and the 0.00075 grace.
- **No path raises** for numeric inputs. Division by zero is guarded.
- **Ladder bounds:** level is always between 0 and 5 (base is at most 4, plus at most 1).
- **The test is independent of the manager's own bookkeeping.** It computes the judge's charge itself, so a manager that under-counted `used` could not pass by grading itself.
- **Cost model limits:**
  - BASE and HAND_END are estimates, because the policy (Task 10) and opponent model (Task 13) don't exist yet.
  - The amortized NOMINAL underestimates small calls: 16 iters really costs 62 µs, against 25 µs in the model. It overestimates 256 iters: 0.36 ms real against 0.40 ms in the model. The net effect is small.

## Concerns

1. **Fixed per-decision cost is the real bank risk, not Monte Carlo.** The manager can only shed rollouts.
   - In the sensitivity table, a non-MC cost of 0.5 ms or more on the dev laptop empties the bank on a 5× machine whatever `iters()` returns, and about 0.9 ms does it on a 3× machine.
   - Recommendation for Task 10/11: keep non-MC decision compute at or below about 0.25–0.4 ms on the dev laptop.
   - Also run a lean feature path (skip strength_now/nut_flags) when `iters()` returns 0 or `tm.used > tm.stop_s`. Both are visible on the public API.
2. **Wiring notes for Task 11:**
   - Construct the manager in `new_half` with `bank_s=config.time_bank_seconds, hands=config.num_hands`.
   - Put `begin()`/`end()` in `try/finally` around choose_action, so a raised exception still charges its time.
   - Time each `on_hand_end` with `perf_counter` and call `add()`.
3. **Two interpretations for the controller to confirm or veto:** ceil rounding of the overshoot, and the hard stop scaling with `bank_s`. Both match the brief exactly at the default settings.
4. **I read one section outside Part A:** the plan's "Review Focus" list (lines 26–35), because the brief's test comment cites "Review Focus 3". I read no other tasks.

---

# Fix round 1 (commit df265bd)

Review of d5c7812: 2 Important findings plus 2 Minor ones folded in by ruling. All four are fixed. I reproduced the reviewer's numbers first: with the default grace and plumbing slowed, the under-count is +0.704 s at 2×, +1.760 s at 3× and +3.600 s at 5×.

## What changed

1. **Plumbing slows with the CPU (Important #1).**
   - `simulate_half` now multiplies HTTP plumbing (0.23 ms) by the same slowdown as BASE and HAND_END. The 50× stress test passes `rest_slowdown=1.0`, so its plumbing stays at laptop speed.
   - `test_slow_clock_stays_under_bank` still asserts `total < 8.0` and that late big pots get more than 0 iterations at 3×. It now asserts `tm.used >= total` only at 1×, where it is true.
   - The `used` comment and module docstring in `bot/timing.py` used to claim the count never falls below the judge's. They now state the condition: plumbing must stay at or below 1 ms − grace_s.
   - The Accounting claim above is corrected.
2. **CPU-factor grace (Important #2).**
   - New constructor parameter `grace_s=0.00075`, used by `end()`.
   - New module-level helper `grace_for_cpu_factor(f)`.
   - New test `test_cpu_factor_grace_stops_under_counting[2, 3, 5]`. It asserts that the default grace under-counts (`default.used < total`), and that with the factor grace `tm.used >= total` and `total < 8.0` at 2×, 3× and 5×.
   - New test `test_grace_for_cpu_factor`: f = 1 gives 0.00075 (the default), f = 3 gives 0.00025, and junk input gives the default.
   - **Deviation from the ruling (please confirm or veto):** the helper returns `0.001 - 0.00025 * f` *without* the `max(0.0, ...)` clamp.
     - With the clamp, the requested 5× assertion cannot hold. At 5× the plumbing alone is 1.15 ms, so the judge charges at least 0.15 ms per request beyond the decision, and a grace of 0 cannot count it.
     - Measured with the ruled clamp: `tm.used` 5.102 s against judge 5.702 s, a gap of −0.600 s (0.15 ms × 4,001 requests). This was RED B below.
     - Without the clamp, the grace at 5× is −0.25 ms. `end()` then charges the overrun, and the count is +0.400 s over the judge.
     - For f ≤ 4 the two formulas give identical results.
     - Reverting is one line, but the `[5]` case would then have to be dropped.
   - The helper also never raises: a non-number, NaN, f ≤ 0 or inf counts as "not measured" and returns the default 0.00075.
3. **Constructor never raises (Minor #3).** Each argument goes through `float()` and falls back instead of raising:
   - **bank_s** that is ≤ 0, NaN or not a number: bank becomes 0 and `stop_s` becomes −1, so `iters()` returns 0 from the very first call. A tiny positive bank would still allow rollouts while `used` is 0.
   - **hands** that is ≤ 0, NaN, inf or not a number: falls back to 1000.
   - **target_s** that is unusable: falls back to the bank-scaled default.
   - **grace_s** that is NaN, inf or not a number: falls back to 0.00075.

   `iters()` also now computes the overshoot as `projected / target − 1`, which cannot divide by zero for a denormal target. It caps the result at 5 levels before `math.ceil`; an infinite overshoot used to raise `OverflowError`. There was no ValueError test to update, so I added `test_degenerate_config_never_raises`.
4. **Target scales with the bank (Minor #4).** `target_s=None` now means `0.375 * bank_s`, which is 3.0 s at the default 8 s bank. New test `test_target_scales_with_retuned_bank`:
   - `TimeManager().target_s == 3.0`;
   - with bank_s = 2.0, the target is 0.75, below the hard stop of 1.5;
   - pacing runs against 0.75 s: at 0.5 s it is on pace (32/64/128), and at 0.85 s it drops one level (16/32/64).

## Updated table: judge charge vs tm.used, plumbing slowed with the CPU (seed 0)

| CPU | grace_s | judge-charged | tm.used | tm.used − judge | mean iters | big-pot mean | late big min |
|---|---|---|---|---|---|---|---|
| 1× | default 0.75 ms (= factor grace at f = 1) | 0.000 s | 0.020 s | +0.020 s | 99.7 | 256 | 256 |
| 2× | default 0.75 ms | 1.126 s | 0.422 s | **−0.704 s** | 99.7 | 256 | 256 |
| 2× | factor 0.50 ms | 1.126 s | 1.286 s | +0.160 s | 99.7 | 256 | 256 |
| 3× | default 0.75 ms | 3.261 s | 1.501 s | **−1.760 s** | 76.8 | 199 | 128 |
| 3× | factor 0.25 ms | 2.753 s | 2.993 s | +0.240 s | 49.8 | 128 | 128 |
| 5× | default 0.75 ms | 6.603 s | 3.003 s | **−3.600 s** | 28.9 | 74 | 64 |
| 5× | factor −0.25 ms (unclamped) | 5.702 s | 6.101 s | +0.400 s | 0.0 | 0 | 0 |
| 5× | ruled clamp 0.00 ms | 5.702 s | 5.102 s | **−0.600 s** | 0.1 | 0 | 0 |

- **Factor grace error:** it over-counts by exactly 0.02 × f ms per request, because plumbing is modelled as 0.25 f ms against a measured 0.23 f ms. Seeds 0–4 give the same differences: +0.16, +0.24 and +0.40 s.
- **3× with the factor grace:** the half costs 2.75 s, and big pots get 128 rollouts all the way to the end.
- **5×:** the fixed cost alone is 1.25 ms of decision plus 1.15 ms of plumbing minus the 1 ms grace, so 1.4 ms × 4,000 = 5.6 s. The manager correctly turns Monte Carlo off. The half ends at 5.70 s, under the bank but with no rollouts. This is the earlier sensitivity concern, and slowed plumbing makes it worse.

## Tests, commands and output

- **RED A** (`.venv/bin/python -m pytest tests/test_timing.py -q`): `ImportError: cannot import name 'grace_for_cpu_factor' from 'bot.timing'`.
- **RED B** (`grace_s` and the helper exactly as ruled, #3 and #4 not yet done): 4 failed, 11 passed.
  - `test_cpu_factor_grace_stops_under_counting[5]`: `assert 5.101750000000129 >= 5.701900000000167`. This is the ruled clamp; [2] and [3] passed.
  - `test_grace_for_cpu_factor`: `assert 0.0 == 0.00075 ± 7.5e-10` (NaN factor).
  - `test_target_scales_with_retuned_bank`: `assert (3.0 == 0.75 ± 7.5e-07`.
  - `test_degenerate_config_never_raises`: `ValueError: bank_s, hands and target_s must be positive`.
- **GREEN** (`.venv/bin/python -m pytest tests/test_timing.py -v`): `15 passed in 0.05s`.
- **Full default suite** (`.venv/bin/python -m pytest -q`): `95 passed, 1 deselected, 1 warning in 19.43s`. The warning is the pre-existing Starlette deprecation.
- **Python 3.12:** both files parse with `ast.parse(..., feature_version=(3, 12))`.
- **Speed:** `iters()` takes 0.04–0.15 µs per call. Construction now takes 0.49 µs, up from 0.10 µs because of the float conversions; it runs once per half.

## Files

- `bot/timing.py`
- `tests/test_timing.py`

The tables come from scratch scripts that are not committed.

## Concerns (fix round)

1. **The clamp deviation above** needs a confirm or veto.
2. **Task 11 must measure f robustly.** For example, take the median of several warmup timings of a fixed workload, divided by the laptop reference.
   - If f is too high, Monte Carlo is throttled more than it needs to be. That is safe but costs EV.
   - If f is too low, the manager under-counts the judge. That is unsafe.
3. **Wiring for Task 11:** `TimeManager(bank_s=config.time_bank_seconds, hands=config.num_hands, grace_s=grace_for_cpu_factor(f))`. `target_s` now comes from the bank by default.
