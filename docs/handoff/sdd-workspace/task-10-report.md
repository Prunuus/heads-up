# Task 10 report: Features and heuristic policy

**Status: DONE.** All three acceptance targets (a), (b), (c) are met by the committed version (v1, `08e541f`); see the final tables below. I wrapped up early at the controller's request. The "next ideas" section lists what I would try next, for Task 12/13/14 to pick up.

Commits on `kevin` (base `66a3d73`):
- `0269604` feat(bot): features and parameterized heuristic policy (v0: the brief's rules, literally)
- `7821d0c` feat(sim): heuristic evaluation bot
- `08e541f` feat(bot): range-aware heuristic v1 (the version evaluated below)

Default suite: **207 passed, 8 deselected** (27 s). Slow tests: `tests/test_heuristic_bot.py -m slow`, **4 passed** (a CI win against each baseline over 1,000 deals).

## Acceptance targets (final v1, `08e541f`)

| Target | Result |
|---|---|
| (a) CI win against all 4 baselines on each of 3 seeds, 1,000 deals | **Met: 12/12 wins.** random +4.03, calling_station +7.40, always_half +4.33, always_pot +3.25 bb/deal. The lowest CI lower bound is +0.43 (always_pot, seed s2). |
| (b) Mean bb/deal positive against every TRAIN opponent | **Met: 15/15 positive.** The minimum is +0.81 (llm_ml_minded). W/L/C 43/0/2; the 2 chops are seed s1 against llm_gto_inspired and llm_ml_minded, both positive. Mean +2.930 bb/deal. |
| (c) 0 errors; max bank < 3.0 s per half | **Met.** 0 errors in all 57 matches (TRAIN plus HOLDOUT). Max bank 0.23 s in the 4-worker TRAIN run (load 7.3). Run alone in one process (load about 6), bank is ≤ 0.020 s per half against maniac, always_half, llm_ml_minded and calling_station. |
| HOLDOUT (information only) | **12/0/0 wins**, mean +1.791 bb/deal, max bank 0.25 s, 0 errors. Never used for development. |

## Design and rationale

### Features (`bot/features.py`)

`Features` is a frozen dataclass with the brief's 9 fields. I added 4 more, each defaulting to "none" so the brief's constructor still works: `opp_raises_before`, `my_raised_this_street`, `facing_frac` and `iters`.

`compute_features(obs, iters, rng)`:
- **pre:** the preflop table value on every street (about 1 µs). It is `None` if the table can't be read.
- **Preflop:** `eq = now = pre`, `nut = 0`, and no Monte Carlo.
- **Postflop:**
  - `eq = equity_vs_random(iters)`.
  - `now = strength_now(iters)`. On the river, and whenever `iters <= 0`, it is the same quantity as `eq`, so I compute it once.
  - `nut = nut_flags`. I keep it even at `iters == 0`, because the measured total stays well under budget (0.14 ms p99 on the river).
- **spr** = effective stack behind after matching the bet ÷ the pot after matching it.
- **pot_odds** = c / (pot + c), where c = min(to_call, my_stack).
- **in_position** = `is_button`.
- **Raise counts** come from `action_history`.
- **Error behaviour:** raises `ValueError` only for a board that isn't 0, 3, 4 or 5 cards. With `iters <= 0` it draws nothing from `rng` (tested).

**Range model.** `range_equity(street, eq, nut, t)` returns P(we win | our eq vs random, our nut bits, an opponent whose own equity vs random is ≥ t):
- Form: a logistic per street on a grid t = 0.0 … 0.9 (interpolated in t), logit P = a + b·logit(eq) + c·[nuts now] + d·[flush possible, not the nuts].
- The c term is clamped at ≥ 0. The river fit comes out negative because of ties at eq ≈ 1, and the policy never folds the river nuts anyway.
- The coefficients are a read-only tuple in the module. `python -m train.fit_range_model --spots 100000` regenerates them deterministically (seed 20261107, about 2 min with 3 workers): 100k random spots per street, both equities from 128 rollouts.
- Why I built it, from the offline data:
  - River, opponent eq ≥ 0.6: a hand with eq 0.6–0.7 vs random wins **10%**, eq 0.7–0.8 wins 30%, eq 0.8–0.9 wins 56%.
  - On the flop and turn the curves are softer and bumpy (draws against made hands).
  - The nut bit and "flush possible without the nuts" matter at high t. On the turn with eq 0.6–0.9, holding the nuts wins 0.79 against 0.375 without.
  - `now` added almost nothing to the fit, so the model doesn't use it.

### Policy (`bot/heuristic.py`)

`HeuristicParams` keeps the brief's 8 fields at the brief's defaults. Six parameters were added, each with a documented default, and each justified by a measurement:

| Param | Default | Meaning |
|---|---|---|
| `range_weight` | 0.7 | Facing a bet, the weight on the range model. The remaining weight goes on the brief's `eq − range_discount·opp_raises`; 0 gives the brief's formula exactly. |
| `bet_range` | (0.50, 0.55, 0.55) | Assumed equity floor of the opponent's first bet or raise on a street (flop, turn, river). |
| `raise_step` | 0.10 | Added to that floor for each further raise on the street. |
| `thin_value` | (0.62, 0.64, 0.66) | Unopened: bet half pot above this effective equity. |
| `ip_thin_shift` | 0.06 | In position after a check, `thin_value` is lowered by this. An equity bot that checks has a capped range. |
| `pre_discount` | 0.07 | Preflop, subtracted from our strength facing the k-th raise as `pre_discount·(1 + 0.5(k−1))`. Measured: the raiser's top 40% costs us about 0.065 equity, top 15% about 0.10, top 5% about 0.14. |

**Preflop.**
- Button first in: raise at `open_min` (pot-sized at `threebet_min`, otherwise half). Below that, call if strength > pot odds + `call_margin`.
- Big blind after a limp: half-pot raise at `threebet_min`.
- Facing k raises: re-raise pot at `threebet_min`. Otherwise call if strength − discount(k) > pot odds + `call_margin`.

**Postflop, facing a bet.**
1. The opponent's assumed floor is t = `bet_range` + `raise_step`·(raises this street − 1).
2. eff = w·range_equity(t) + (1−w)·(eq − `range_discount`·opp_raises), minus `nonnut_penalty` on a paired or flush-possible board without the nuts.
3. Raise above `value_raise` (pot-sized above `pot_vs_half_cut`), call above pot odds + `call_margin`, otherwise fold.
4. With the river nuts: raise pot if allowed, otherwise call. This rule is structural, so no parameter set can fold it.

**Postflop, unopened.**
- Bet pot or half above `value_raise`, and half pot above `thin_value` (minus `ip_thin_shift` in position).
- In position, bluff half pot with probability `bluff_freq`.
- Check otherwise.

**Legalization.** A raise falls back to the other raise size, then to check/call. The policy never folds when nothing is owed, even if a malformed observation lists FOLD as legal. `opp` is accepted and ignored. Randomness comes only from the caller's `rng`.

**Why this shape.** The zoo's strong bots (and, by hypotheses H1/H2, most of the real field) play "equity vs random against pot odds".
- **Their bets are strong.** They bet or raise at equity ≥ 0.55–0.78 and bluff little. The brief's linear discount therefore called far too wide. Against ml_minded, v0's river calls averaged −17.5 chips and won only 29% at showdown, with equity 0.63 vs random.
- **Their calls are loose.** They call with equity ≥ pot odds + 0.02–0.05, so thin, large value bets pay. After they check, their range is capped, so in position we can value-bet thinner still.
- **Position and nut-ness** come in through `ip_thin_shift`, `in_position` and the range model's nut and flush terms.

## Timing

`compute_features` + `heuristic_action` per real decision: 6,001 observations from heuristic-vs-random games, of which 3,091 are preflop. Load 7.4 from other sessions' jobs; numbers on a quiet machine should be equal or lower.

| iters | Preflop p50 / p99 | Flop p50 / p99 | Turn p50 / p99 | River p50 / p99 | All max |
|---:|---|---|---|---|---:|
| 0 | 0.003 / 0.004 ms | 0.058 / 0.063 | 0.083 / 0.094 | 0.127 / 0.136 | 0.143 |
| 32 | 0.003 / 0.004 | 0.191 / 0.204 | 0.222 / 0.238 | 0.195 / 0.207 | 0.311 |
| 64 | 0.003 / 0.004 | 0.250 / 0.265 | 0.288 / 0.302 | 0.234 / 0.248 | 0.387 |
| 128 | 0.003 / 0.004 | 0.371 / 0.391 | 0.418 / 0.441 | 0.310 / 0.326 | 0.512 |
| 256 | 0.003 / 0.004 | 0.605 / 0.633 | 0.672 / 0.693 | 0.457 / 0.478 | 0.784 |

- **Fixed cost outside Monte Carlo** (iters = 0): ≤ 0.14 ms p99 (river, including `nut_flags` at about 0.11 ms). This is inside the ≤ 0.3 ms budget.
- **The policy itself is about 3 µs.** The table value only, about 1 µs, is visible preflop.
- **Whole decisions** are under the 0.75 ms grace at every ladder level except a few turn decisions at 256 iters (max 0.78 ms). The TimeManager gives 256 only while it is under half its target pace, which is why the in-sim bank use is about 0.0–0.2 s per half.

## Evaluation trajectory

All runs: `focus_eval` via `python -m sim.heuristic_bot`, `--workers 4`, seeds s1–s3, 1,000 deals, TRAIN opponents unless stated. Comparisons use `compare.py` (scratchpad): per-opponent paired per-deal differences B − A, seeds s1–s2 × 1,000 deals. ▲/▼ marks a 95% CI that excludes 0.

Caveats:
- **Load affected some runs.** Another session ran a 9-process job during the v1a run and the first comparisons (load average 22–24); later runs saw load 6–9. Bank figures in those runs are inflated.
- **Our iteration counts follow machine load,** through the TimeManager's pace adjustment. So absolute numbers move a little between runs; variants are compared only within one interleaved run.
- **Random-bot diffs are mostly noise.** Any changed decision desynchronizes the random bot's RNG, so its paired diffs are effectively unpaired.

### v0: the brief's rules, literally (`0269604`)

Mean +2.344, W/L/C 36/1/8, max bank 0.03 s. **It loses to llm_ml_minded** and chops 2 of 3 seeds against beginner_2h, mc_equity and gto_inspired.

| Opponent | bb/deal | per-seed bb [95% CI] verdict | W/L/C | Max bank (s) | Errors |
|---|---:|---|:-:|---:|---:|
| random | +3.941 | +3.89 [+2.60, +5.17] W; +3.62 [+2.34, +4.89] W; +4.32 [+3.05, +5.59] W | 3/0/0 | 0.01 | 0 |
| calling_station | +5.709 | +6.07 [+5.35, +6.80] W; +5.43 [+4.73, +6.12] W; +5.63 [+4.94, +6.31] W | 3/0/0 | 0.00 | 0 |
| always_half | +7.049 | +8.00 [+6.55, +9.46] W; +5.47 [+3.91, +7.03] W; +7.68 [+6.16, +9.19] W | 3/0/0 | 0.01 | 0 |
| always_pot | +3.064 | +2.88 [+1.17, +4.60] W; +2.39 [+0.61, +4.17] W; +3.92 [+2.20, +5.63] W | 3/0/0 | 0.01 | 0 |
| guide_example | +0.910 | +0.83 [+0.65, +1.02] W; +1.10 [+0.89, +1.31] W; +0.80 [+0.61, +0.98] W | 3/0/0 | 0.00 | 0 |
| nit | +1.219 | +1.08 [+0.63, +1.52] W; +1.36 [+0.94, +1.77] W; +1.22 [+0.77, +1.68] W | 3/0/0 | 0.00 | 0 |
| maniac | +6.756 | +7.09 [+5.60, +8.59] W; +5.63 [+4.06, +7.19] W; +7.55 [+6.07, +9.03] W | 3/0/0 | 0.00 | 0 |
| mc_equity_fast | +1.413 | +1.29 [+0.73, +1.85] W; +1.42 [+0.83, +2.01] W; +1.53 [+0.96, +2.10] W | 3/0/0 | 0.00 | 0 |
| mc_equity_slow | +1.401 | +1.29 [+1.08, +1.51] W; +1.46 [+1.22, +1.70] W; +1.45 [+1.22, +1.68] W | 3/0/0 | 0.00 | 0 |
| checkfold_after_300 | +1.378 | +1.17 [+0.87, +1.46] W; +1.55 [+1.18, +1.91] W; +1.42 [+1.11, +1.73] W | 3/0/0 | 0.00 | 0 |
| llm_simple_rules | +1.071 | +0.92 [+0.33, +1.50] W; +1.15 [+0.54, +1.76] W; +1.14 [+0.49, +1.80] W | 3/0/0 | 0.01 | 0 |
| llm_beginner_2h | +0.511 | +0.52 [-0.10, +1.13] C; +0.15 [-0.49, +0.79] C; +0.87 [+0.21, +1.53] W | 1/0/2 | 0.01 | 0 |
| llm_mc_equity | +0.656 | +0.60 [-0.13, +1.33] C; +1.16 [+0.46, +1.87] W; +0.21 [-0.49, +0.91] C | 1/0/2 | 0.03 | 0 |
| llm_gto_inspired | +0.493 | +0.38 [-0.27, +1.04] C; +0.79 [+0.15, +1.43] W; +0.30 [-0.31, +0.91] C | 1/0/2 | 0.02 | 0 |
| llm_ml_minded | -0.405 | -0.36 [-0.86, +0.15] C; -0.22 [-0.76, +0.32] C; -0.64 [-1.19, -0.09] L | 0/1/2 | 0.02 | 0 |
| **mean** | **+2.344** | | **36/1/8** | 0.03 | 0 |

**Diagnosis.** I logged every decision in v0 against ml_minded (s1, 2,000 hands). Calls facing a bet lost on every street:

| Spot | Calls | Avg net (chips) | Showdown win rate | Avg eq vs random |
|---|---:|---:|---:|---:|
| River | 204 | −17.5 | 29% | 0.63 |
| Turn | 249 | −13.7 | 36% | 0.62 |
| Flop | 330 | −7.3 | 43% | 0.56 |

River checks with eq 0.7–0.9 won 94–95% at showdown, so they were missed value bets.

### v1a: range model + thin value + preflop discount (uncommitted prototype; load 22.5)

Mean +2.672, **W/L/C 45/0/0**. It gained against every equity-based bot:
- llm_ml_minded −0.41 → +0.66
- llm_mc_equity +0.66 → +1.86
- llm_gto_inspired +0.49 → +1.21
- llm_beginner_2h +0.51 → +2.67
- llm_simple_rules +1.07 → +2.29
- mc_equity_fast +1.41 → +3.59

It gave back margin against the bots whose bets carry no information (always_half 7.05 → 3.53, maniac 6.76 → 4.45), which are still easy wins.

| Opponent | bb/deal | per-seed bb [95% CI] verdict | W/L/C | Max bank (s) | Errors |
|---|---:|---|:-:|---:|---:|
| random | +3.516 | +3.79 [+2.53, +5.04] W; +2.46 [+1.15, +3.77] W; +4.30 [+3.03, +5.57] W | 3/0/0 | 0.01 | 0 |
| calling_station | +7.039 | +7.20 [+6.45, +7.94] W; +6.84 [+6.12, +7.55] W; +7.09 [+6.35, +7.82] W | 3/0/0 | 0.00 | 0 |
| always_half | +3.525 | +4.04 [+2.36, +5.72] W; +2.06 [+0.31, +3.81] W; +4.48 [+2.80, +6.15] W | 3/0/0 | 0.03 | 0 |
| always_pot | +3.015 | +3.57 [+1.86, +5.28] W; +1.85 [+0.07, +3.63] W; +3.63 [+1.94, +5.31] W | 3/0/0 | 0.02 | 0 |
| guide_example | +1.254 | +1.14 [+0.99, +1.29] W; +1.38 [+1.17, +1.59] W; +1.24 [+1.06, +1.43] W | 3/0/0 | 0.00 | 0 |
| nit | +1.460 | +1.55 [+1.07, +2.03] W; +1.51 [+1.06, +1.96] W; +1.31 [+0.81, +1.82] W | 3/0/0 | 0.00 | 0 |
| maniac | +4.449 | +4.86 [+3.29, +6.44] W; +3.32 [+1.63, +5.01] W; +5.16 [+3.56, +6.77] W | 3/0/0 | 0.04 | 0 |
| mc_equity_fast | +3.585 | +3.32 [+2.70, +3.93] W; +3.55 [+2.91, +4.19] W; +3.89 [+3.24, +4.54] W | 3/0/0 | 0.09 | 0 |
| mc_equity_slow | +1.542 | +1.58 [+1.37, +1.79] W; +1.57 [+1.34, +1.80] W; +1.47 [+1.28, +1.66] W | 3/0/0 | 0.21 | 0 |
| checkfold_after_300 | +2.006 | +1.91 [+1.58, +2.24] W; +2.08 [+1.71, +2.45] W; +2.03 [+1.62, +2.44] W | 3/0/0 | 0.71 | 0 |
| llm_simple_rules | +2.290 | +1.98 [+1.50, +2.47] W; +2.78 [+2.28, +3.29] W; +2.10 [+1.63, +2.57] W | 3/0/0 | 0.85 | 0 |
| llm_beginner_2h | +2.666 | +2.54 [+1.74, +3.35] W; +2.54 [+1.73, +3.34] W; +2.92 [+2.05, +3.78] W | 3/0/0 | 1.48 | 0 |
| llm_mc_equity | +1.858 | +1.67 [+0.88, +2.46] W; +1.86 [+1.05, +2.67] W; +2.04 [+1.21, +2.88] W | 3/0/0 | 1.44 | 0 |
| llm_gto_inspired | +1.210 | +0.91 [+0.22, +1.59] W; +1.24 [+0.58, +1.89] W; +1.49 [+0.77, +2.20] W | 3/0/0 | 1.29 | 0 |
| llm_ml_minded | +0.661 | +0.57 [+0.04, +1.10] W; +0.85 [+0.31, +1.40] W; +0.56 [+0.01, +1.12] W | 3/0/0 | 1.31 | 0 |
| **mean** | **+2.672** | | **45/0/0** | 1.48 | 0 |

(Bank values up to 1.48 s come from the load-22 contention; compare the final run's 0.23 s and the 0.02 s solo figure.)

### Experiment E1: `range_weight` (A = 0.7)

| Variant | Pooled diff B − A [95% CI] | Decision |
|---|---|---|
| 0.5 | **+0.230 [+0.095, +0.365]** | Rejected. The gain is all in the information-free bettors, and it costs the equity bots. |
| 0.9 | −0.324 [−0.462, −0.185] | Rejected. |

I kept 0.7. Points come from CI wins: against always_half, maniac and random we already win by about 3–5 bb/deal, while the margins against the equity bots (the likely real field) are thinner. 0.5 is significantly worse against mc_equity_fast, mc_equity_slow and simple_rules. Making this trade-off per opponent is the opponent model's job.

| Opponent | A bb/deal | range_weight 0.5 | range_weight 0.9 |
|---|---:|---:|---:|
| random | +3.20 | +0.69 | -1.13 ▼ |
| calling_station | +7.01 | +0.08 | -0.01 |
| always_half | +3.31 | +1.33 ▲ | -1.15 ▼ |
| always_pot | +3.07 | +0.59 ▲ | -0.83 ▼ |
| guide_example | +1.25 | +0.00 | +0.02 |
| nit | +1.53 | -0.01 | +0.04 |
| maniac | +4.09 | +1.44 ▲ | -1.01 ▼ |
| mc_equity_fast | +3.15 | -0.79 ▼ | -0.83 ▼ |
| mc_equity_slow | +1.66 | -0.17 ▼ | -0.23 ▼ |
| checkfold_after_300 | +2.21 | -0.15 | -0.08 |
| llm_simple_rules | +2.37 | -0.21 ▼ | -0.08 |
| llm_beginner_2h | +2.19 | +0.43 | +0.35 |
| llm_mc_equity | +1.64 | +0.62 | +0.02 |
| llm_gto_inspired | +1.24 | -0.31 | -0.02 |
| llm_ml_minded | +0.47 | -0.08 | +0.09 |

### Experiment E2: `ip_thin_shift` (thinner value in position after a check; A = 0)

| Variant | Pooled diff B − A [95% CI] | Decision |
|---|---|---|
| 0.06 | **+0.161 [+0.043, +0.279]** | **Adopted** as the default. |
| 0.12 | +0.108 [−0.013, +0.230] | Not significant. |

| Opponent | A bb/deal | ip_thin_shift 0.06 | ip_thin_shift 0.12 |
|---|---:|---:|---:|
| random | +3.15 | +0.14 | -0.67 |
| calling_station | +7.03 | +0.34 ▲ | +0.58 ▲ |
| always_half | +3.05 | +0.00 | +0.00 |
| always_pot | +2.71 | +0.00 | +0.00 |
| guide_example | +1.26 | +0.01 | +0.12 ▲ |
| nit | +1.53 | -0.07 | -0.10 |
| maniac | +4.09 | +0.37 | +0.08 |
| mc_equity_fast | +3.27 | +0.35 | +0.53 ▲ |
| mc_equity_slow | +1.59 | +0.08 | +0.11 ▲ |
| checkfold_after_300 | +2.13 | -0.03 | +0.16 |
| llm_simple_rules | +3.25 | +0.10 | +0.12 |
| llm_beginner_2h | +2.28 | +0.54 ▲ | +0.43 |
| llm_mc_equity | +1.87 | +0.29 | +0.26 |
| llm_gto_inspired | +1.19 | +0.15 | -0.14 |
| llm_ml_minded | +0.43 | +0.14 | +0.15 |

### Experiment E3: preflop restructure

A = the raise-count-based discount schedule, with a constant re-raise threshold.

| Variant | Pooled diff B − A [95% CI] |
|---|---|
| `limp_min` 0.42 (fold weak button hands instead of limping) | −0.175 [−0.304, −0.047] |
| `reraise_step` 0.05 (tighter 4-bets and 5-bets) | −0.323 [−0.475, −0.172] |
| Both | −0.307 [−0.459, −0.154] |

Both parameters were **rejected and removed**. Re-raising less costs about 1.7 bb/deal against always_half, which re-raises with random hands. Against the equity bots it changed nothing measurable.

| Opponent | A bb/deal | limp_min .42 + reraise_step .05 | limp_min .42 | reraise_step .05 |
|---|---:|---:|---:|---:|
| random | +4.29 | -1.48 ▼ | -1.61 ▼ | -1.77 ▼ |
| calling_station | +7.39 | -0.02 | -0.02 | -0.01 |
| always_half | +3.85 | -1.81 ▼ | +0.08 | -1.68 ▼ |
| always_pot | +2.88 | -0.68 | +0.05 ▲ | -0.73 |
| guide_example | +1.27 | -0.03 | -0.03 | +0.00 |
| nit | +1.39 | +0.14 | -0.08 | +0.14 |
| maniac | +4.61 | -0.33 | -0.46 | -0.26 |
| mc_equity_fast | +3.59 | -0.00 | -0.11 | -0.03 |
| mc_equity_slow | +1.65 | -0.03 | -0.01 | +0.01 |
| checkfold_after_300 | +2.10 | +0.01 | +0.07 | +0.06 |
| llm_simple_rules | +3.43 | -0.05 | -0.13 | -0.01 |
| llm_beginner_2h | +2.57 | +0.02 | +0.04 | -0.01 |
| llm_mc_equity | +2.46 | -0.29 | -0.69 ▼ | -0.76 ▼ |
| llm_gto_inspired | +1.21 | -0.21 | +0.14 | -0.03 |
| llm_ml_minded | +0.34 | +0.17 | +0.13 | +0.22 |

(Baseline A ran lucky against random, +4.29, so all three random diffs are biased negative. See the RNG-desync caveat above.)

### Experiment E4: bluffing and out-of-position thin value (A = defaults)

| Variant | Pooled diff B − A [95% CI] | Decision |
|---|---|---|
| `bluff_freq` 0.2 | +0.082 [−0.038, +0.202] | Not significant. The brief's 0.10 is kept. |
| River bluffs only below eq 0.3 (`river_bluff_max`) | **−0.164 [−0.283, −0.044]** | Rejected and removed. "Bluffs" with medium hands work as thin value bets. |
| Both | −0.004 | Rejected. |
| `bluff_freq` 0 | −0.048 | Rejected. |
| OOP thin value +0.04, IP unchanged | **−0.179 [−0.307, −0.051]** | Rejected. |

| Opponent | A bb/deal | bluff_freq .2 | river_bluff_max .3 | both | bluff_freq 0 | OOP thin +.04 (IP same) |
|---|---:|---:|---:|---:|---:|---:|
| random | +4.25 | -0.03 | -0.77 | +0.08 | -0.72 | -1.18 ▼ |
| calling_station | +7.41 | +0.11 | -0.04 | -0.07 | +0.07 | -0.14 |
| always_half | +3.85 | +0.00 | +0.00 | -0.01 | +0.00 | +0.07 |
| always_pot | +2.88 | +0.00 | +0.00 | +0.00 | +0.00 | +0.14 |
| guide_example | +1.27 | +0.13 ▲ | -0.01 | +0.06 | -0.08 | +0.02 |
| nit | +1.39 | +0.03 | +0.02 | +0.00 | +0.02 | -0.02 |
| maniac | +4.61 | +0.29 | +0.04 | +0.97 ▲ | +0.47 | -0.60 |
| mc_equity_fast | +3.53 | -0.21 | -0.63 ▼ | -0.74 ▼ | -0.30 | -0.43 ▼ |
| mc_equity_slow | +1.61 | -0.02 | -0.11 ▼ | -0.00 | -0.05 | -0.10 ▼ |
| checkfold_after_300 | +2.10 | +0.07 | +0.14 | +0.09 | +0.14 | +0.01 |
| llm_simple_rules | +3.43 | +0.02 | -0.39 ▼ | -0.10 | -0.07 | -0.15 |
| llm_beginner_2h | +2.62 | +0.06 | -0.12 | -0.15 | +0.18 | -0.08 |
| llm_mc_equity | +1.72 | +0.94 ▲ | -0.42 | +0.14 | -0.18 | -0.08 |
| llm_gto_inspired | +1.31 | -0.17 | +0.01 | -0.36 | -0.09 | +0.11 |
| llm_ml_minded | +0.60 | +0.02 | -0.18 | +0.02 | -0.11 | -0.24 |

### Final v1 (`08e541f`), TRAIN: 3 seeds × 1,000 deals, load 7.3

| Opponent | bb/deal | per-seed bb [95% CI] verdict | W/L/C | Max bank (s) | Errors |
|---|---:|---|:-:|---:|---:|
| random | +4.032 | +4.19 [+2.90, +5.49] W; +3.35 [+2.08, +4.63] W; +4.55 [+3.27, +5.83] W | 3/0/0 | 0.09 | 0 |
| calling_station | +7.402 | +7.50 [+6.75, +8.25] W; +7.30 [+6.54, +8.07] W; +7.40 [+6.66, +8.14] W | 3/0/0 | 0.06 | 0 |
| always_half | +4.334 | +5.12 [+3.47, +6.77] W; +2.68 [+0.94, +4.42] W; +5.20 [+3.54, +6.86] W | 3/0/0 | 0.18 | 0 |
| always_pot | +3.250 | +3.63 [+1.94, +5.32] W; +2.21 [+0.43, +3.98] W; +3.91 [+2.20, +5.63] W | 3/0/0 | 0.13 | 0 |
| guide_example | +1.261 | +1.25 [+1.12, +1.38] W; +1.30 [+1.13, +1.46] W; +1.24 [+1.09, +1.38] W | 3/0/0 | 0.00 | 0 |
| nit | +1.403 | +1.48 [+1.01, +1.96] W; +1.42 [+0.95, +1.88] W; +1.31 [+0.78, +1.84] W | 3/0/0 | 0.00 | 0 |
| maniac | +5.251 | +5.58 [+3.97, +7.19] W; +3.86 [+2.17, +5.55] W; +6.31 [+4.70, +7.92] W | 3/0/0 | 0.19 | 0 |
| mc_equity_fast | +3.645 | +3.57 [+2.94, +4.21] W; +3.40 [+2.75, +4.06] W; +3.96 [+3.29, +4.64] W | 3/0/0 | 0.10 | 0 |
| mc_equity_slow | +1.643 | +1.60 [+1.39, +1.81] W; +1.64 [+1.42, +1.87] W; +1.69 [+1.45, +1.93] W | 3/0/0 | 0.02 | 0 |
| checkfold_after_300 | +2.135 | +1.94 [+1.56, +2.31] W; +2.22 [+1.84, +2.60] W; +2.25 [+1.86, +2.65] W | 3/0/0 | 0.04 | 0 |
| llm_simple_rules | +3.266 | +3.32 [+2.67, +3.97] W; +3.24 [+2.54, +3.94] W; +3.24 [+2.53, +3.95] W | 3/0/0 | 0.13 | 0 |
| llm_beginner_2h | +2.749 | +2.55 [+1.85, +3.25] W; +2.76 [+2.07, +3.44] W; +2.94 [+2.22, +3.66] W | 3/0/0 | 0.21 | 0 |
| llm_mc_equity | +1.802 | +1.42 [+0.66, +2.18] W; +2.14 [+1.40, +2.89] W; +1.84 [+1.06, +2.62] W | 3/0/0 | 0.23 | 0 |
| llm_gto_inspired | +0.971 | +0.68 [-0.05, +1.40] C; +0.84 [+0.14, +1.54] W; +1.39 [+0.65, +2.14] W | 2/0/1 | 0.16 | 0 |
| llm_ml_minded | +0.808 | +0.39 [-0.15, +0.94] C; +1.19 [+0.64, +1.74] W; +0.84 [+0.29, +1.39] W | 2/0/1 | 0.14 | 0 |
| **mean** | **+2.930** | | **43/0/2** | 0.23 | 0 |

### Final v1, HOLDOUT (information only; never used for development)

| Opponent | bb/deal | per-seed bb [95% CI] verdict | W/L/C | Max bank (s) | Errors |
|---|---:|---|:-:|---:|---:|
| threshold_cat | +2.396 | +2.36 [+1.90, +2.82] W; +2.46 [+1.98, +2.95] W; +2.37 [+1.88, +2.85] W | 3/0/0 | 0.02 | 0 |
| llm_aggressive | +2.609 | +2.46 [+1.20, +3.72] W; +1.60 [+0.27, +2.94] W; +3.76 [+2.48, +5.04] W | 3/0/0 | 0.25 | 0 |
| llm_opp_model | +1.071 | +0.99 [+0.50, +1.48] W; +1.02 [+0.48, +1.56] W; +1.20 [+0.72, +1.68] W | 3/0/0 | 0.13 | 0 |
| llm_time_careful | +1.086 | +1.03 [+0.42, +1.64] W; +1.28 [+0.74, +1.82] W; +0.95 [+0.34, +1.55] W | 3/0/0 | 0.11 | 0 |
| **mean** | **+1.791** | | **12/0/0** | 0.25 | 0 |

**Summary.** v0 → v1 mean bb/deal on TRAIN went +2.344 → +2.930, and W/L/C went 36/1/8 → 43/0/2. On the equity-bot group (the 5 llm_* bots plus mc_equity_fast) the mean went +0.62 → +2.21.

## TDD evidence

1. **RED:** `tests/test_heuristic.py` written first. `pytest tests/test_heuristic.py` → `ModuleNotFoundError: No module named 'bot.features'` (collection error).
2. **First implementation:** 2 failed, `test_calls_with_nuts_on_river[1]` and `[2]`. The PASSIVE parameter set (call_margin 1.0) folded the river nuts, because my first guard went through `eff`. I made the guard structural: never fold the river nuts. GREEN: **15 passed**.
3. **Range-model tests:**
   - `test_folds_medium_river_hands_to_a_pot_bet` pins the v1 behaviour. With `range_weight=0` (the brief's formula, i.e. v0's logic) the same 15 engine spots, eq 0.55–0.65 vs random facing a river pot bet, are all CHECK_CALL; v1 folds all 15.
   - `test_range_equity_is_a_monotone_probability` first failed on my own over-strict bound (the flop drop at eq 0.3 is 0.28, not > 0.3). I loosened the bound to 0.2, with that measurement as the reason.
4. **Eval-bot tests** (`tests/test_heuristic_bot.py`) were written together with the bot, not test-first: 5 pass. The 4 slow tests pass with `-m slow`.
5. **Final:** default suite **207 passed, 8 deselected**. All new and changed files parse with `ast.parse(feature_version=(3, 12))`.

Tests owed by the brief and the controller, all present:
- `test_always_legal_on_10k_random_observations`: 10,000 observations from heuristic-vs-random games, both seats, all four streets, × 3 parameter sets (defaults, an all-raise/bluff set and an all-fold set), iters 0 and 16.
- `test_never_folds_when_free`: also covers a malformed observation that lists FOLD when nothing is owed.
- `test_all_in_corners_are_legal` (Review Focus 2), built with the engine where reachable:
  - raise_to all-in below min_raise_to, preflop (80 < 90) and on the flop;
  - to_call ≥ my_stack with the opponent all-in and raises capped;
  - legal == (CHECK_CALL,), preflop and river (synthetic);
  - a short stack facing a bigger bet.
- `test_calls_with_nuts_on_river`: facing 1 and 2 raises, all 3 parameter sets, iters 0/16/128.
- `test_compute_features_iters_zero`: every street, no draws from `rng`, finite values, median < 0.5 ms.

Also added:
- `test_raises_with_nuts_on_river_when_allowed`
- `test_feature_values_from_history`
- `test_features_are_frozen`
- `test_heuristic_params_defaults_are_the_briefs`
- `test_heuristic_is_deterministic_given_rng_and_ignores_opp`
- `test_preflop_open_and_fold_behaviour`
- `test_calls_or_raises_strong_river_hands_facing_a_pot_bet`

## Files changed

- `bot/features.py` (new): Features, compute_features, the range model table and `range_equity`.
- `bot/heuristic.py` (new): HeuristicParams and heuristic_action.
- `train/fit_range_model.py` (new, offline): regenerates the range table.
- `sim/heuristic_bot.py` (new, dev-only): HeuristicBot, the bot-file interface and the eval CLI.
- `tests/test_heuristic.py` (new): 18 tests.
- `tests/test_heuristic_bot.py` (new): 5 tests plus 4 slow.

Dev scripts in `t10/` under the scratchpad (not committed): `diag.py`, `analyze*.py`, `compare.py`, `range_eq*.py`, `fit_*.py`, `timing_table.py`, plus raw results in `t10/evals/`.

## Self-review

- **R22.**
  - `bot.*` is imported at module top in `bot/` and in `sim/heuristic_bot.py`. The lazy imports in `main()` are `sim.*` only.
  - There is no mutable module-level state; the range table is a tuple of tuples.
  - The eval bot keeps the TimeManager and rng in `memory`. Its instance holds only the frozen params.
  - Nothing in `bot.*` is pickled during play. `HeuristicParams` is pickled only when jobs are submitted.
- **Runtime purity:** `bot/` uses stdlib, numpy and `game` only. Python 3.12 syntax is verified.
- **Legality:** every return value goes through `_legal`, which never folds when free and falls back raise → other raise → call. Tested on 10k engine observations × 3 extreme parameter sets, plus 7 corner observations.
- **Brief interface:** exact. I added fields and params with defaults and renamed nothing. The brief's default values are unchanged; my measurements of them are recorded for Task 12.
- **`range_discount` semantics:** it is still "subtracted from equity per opponent raise". It now acts on the `(1 − range_weight)` share of the facing-bet equity; `range_weight=0` reproduces the brief's formula exactly. Preflop uses the new `pre_discount` instead, because the brief's tuple is per postflop street (flop, turn, river).
- **Removed:** the rejected params (`limp_min`, `reraise_step`, `river_bluff_max`) are gone from the code; I kept no dead switches.

## Concerns and notes

1. **The machine was loaded throughout.** Another session's 9-process job ran at load 22–24, and later sessions' pytest jobs at load 6–9.
   - Bank numbers above are therefore upper bounds. The 0.02 s single-process figure was measured at load about 6.
   - Time-adaptive opponents (ml_minded, beginner_2h, gto_inspired, mc_equity) get fewer rollouts under load. Their numbers may be slightly optimistic for us.
   - Re-run the final table on a quiet machine before shipping v1 (Task 12).
2. **First-decision spike (Task 11 warmup).** The first postflop decision of a process took 11.3 ms. `bot.equity.nut_flags` builds `_PAIR_CACHE`/`_TRI_CACHE` lazily per board size, and `equity.warmup()` doesn't touch them. Task 11's warmup should call `nut_flags` once for each board width (3/4/5).
3. **Points-first trade-off.** v1 gives up bb against the information-free bettors (always_half 7.0 → 4.3, maniac 6.8 → 5.3 against v0) to beat the value bettors. Those are still large CI wins. This is the clearest win for Task 13/14: detect a high bet frequency and lower `range_weight` (0.5 measured +1.3/+1.4 bb/deal against always_half/maniac).
4. **Thinnest margins:** llm_ml_minded (+0.81) and llm_gto_inspired (+0.97). Each chopped one seed. ml_minded adapts its bluff rate to our fold-to-raise rate; that is countered by opponent modelling, not by a static heuristic.
5. **`now` is computed but not used** by this policy (it didn't improve the range fit). It costs about 0.05–0.2 ms, mostly inside the free grace. Keep it for Task 16's 2-D buckets, or drop it from the hot path if the bank ever binds.
6. **The range model assumes "equity vs random ≥ t" betting ranges.** That is accurate for the zoo's equity bots and wrong for maniacs. `range_weight` is the safety valve.

## Next ideas (for Task 12 tuning, Task 13/14 exploits)

**Parameters (Task 12).** Only E1–E4 were measured.
- The brief params were never tuned: `value_raise`, `pot_vs_half_cut` (POT sizing threshold 0.85, which may be too high against callers), `call_margin`, `open_min`, `threebet_min`, `range_discount`.
- New params: `bet_range` per street, `raise_step`, `thin_value` per street. `ip_thin_shift` has a peak between 0.06 and 0.12. `range_weight` lies in 0.6–0.8.
- Suggested objective, per plan: points + 0.01·bb, on TRAIN with common seeds.

**Logic.**
- **(a)** Probe bets out of position after the opponent checked back the previous street. Their range is capped then, too; add a `opp_checked_prev` feature.
- **(b)** Value against a capped range, computed exactly from the range model: R_capped = (R(e, c)·F(c) − R(e, cap)·F(cap)) / (F(c) − F(cap)), where F(t) is the share of hands with equity ≥ t. Use it instead of a flat `ip_thin_shift`.
- **(c)** Size by expected value against a calling range c(s) = s/(1+2s) + margin. Choose pot or half from P(call)·(2R − 1)·s.
- **(d)** An SPR commitment rule. With SPR < 1 after a bet, call or jam with lower thresholds.
- **(e)** Check-raise with the nuts out of position against bots that bet when checked to.
- **(f)** Exploit hook: when the opponent model reports a high bet frequency or that bets are uninformative, lower `range_weight` and `bet_range`. When it reports a check/fold bot (I2), bet any two cards.
- **(g)** Measure how fast the opponent bots react to our fold rate (ml_minded's `bluffp`), and keep fold-to-raise below about 0.3 against adaptive bots.
