# Task 13 report: opponent model

**Status:** DONE_WITH_CONCERNS. The one deviation from the brief is listed under Concerns: the check-folder raise window is scoped to the run of folds.
**Branch / worktree:** `kevin-opp` in `/Users/kchen/Desktop/code/FALL26/heads-up-bundle-opp`. Not pushed.
**Commit:** `371e137 feat(bot): opponent model with reversible detectors` (parent `66a3d73`).

## Files changed
- `bot/opponent.py` (new, 393 lines): `OpponentModel`. Imports only stdlib (`collections`), `game` and `evaluator`. It has no mutable module-level state, only int/float constants.
- `tests/test_opponent.py` (new, 433 lines): 20 default tests and 2 slow tests.

`agent.py` is not wired up: the brief lists only these two files, and on this branch `agent.py` is still the starter. The integration contract is below.

## Implementation

**Integration contract (for Tasks 11 and 14).**
- In `new_half`, create `memory["opp"] = OpponentModel(raise_cap=config.raise_cap, big_blind=config.big_blind)`. Both arguments are optional; they default to the `game.py` constants, and bad values fall back to the defaults, so the constructor never raises.
- Call `memory["opp"].observe(obs)` first thing in `choose_action`.
- Call `memory["opp"].finish(result)` in `on_hand_end`.
- `observe()` and `finish()` never raise. Every exception is swallowed and counted in the read-only `errors` property, which is 0 in every simulated half.
- `omaha_score`'s table must already be built. `agent.py` calls `warmup()` at import, which does this.

**observe(obs)**
- Replays `obs.action_history[seen:]` once per record, keyed by hand_index. A repeated or older observation adds nothing.
- Each opponent record becomes one decision: `(street, facing_bet, action, raise_legal)`.
- `facing_bet` means the opponent owed chips. It compares the opponent's street total before the action (`street_total - amount`) with our current street total. Preflop, our total starts at our blind, since the history has no blinds.
- `raise_legal` is the engine's `_can_raise`: fewer than `raise_cap` raises on the street, the opponent's stack greater than what it owed, and our stack greater than 0.
- Stacks before each record are rebuilt backwards from `obs.my_stack` and `obs.opp_stack`, so no config is needed. Without this, all-in and capped spots would deflate the maniac's raise rate.

**finish(result)** handles F11. At most one opponent decision can follow our last action in a hand. It is inferred from our last observation `o` and `added = result.pot - o.pot`:

| Result | Inferred opponent decision |
|---|---|
| no showdown, we won, we have an observation | fold to our bet or raise, on `o.street` |
| no showdown, we won, no observation, `pot < 2*BB` | button folded its first preflop decision (we were BB and never acted) |
| showdown, `added > min(o.to_call, o.my_stack)` | called our bet or raise, possibly all-in for less; `raise_legal` found by matching `o.raise_to` sizes to `added` |
| showdown, `added == call`, river, `to_call == 0`, no river action yet | checked behind |
| anything else | nothing; our action closed the hand, or we folded |

- Results are keyed by hand_index. Order doesn't matter, and a repeated hand_index is ignored (first one wins).
- At most 8 unfinished hands are kept; normally there is 1.
- Showdowns add `(action class, omaha_score(opp_hole, board) >> 20)` for each street on which the opponent decided.

**Action class** is `(street: int, aggressive: bool)`. `aggressive` means the opponent bet or raised at least once on that street of the hand. `OpponentModel.current_class(obs)` returns the opponent's class on the current street, which is what Task 14 looks up.

**Statistics.** All are cumulative over the half and return `(rate, n)`, with rate 0.0 when n == 0.
- `fold_to_bet(street)`
- `raise_rate(street)`: over decisions where a raise was legal.
- `vpip` (property): hands where it called or raised preflop, over hands where it made a preflop decision.
- `showdown_min_category(cls, min_n=5)`: returns None below `min_n` showdowns, so the I6 rule "requires n >= 5" is the default. Use `min_n=1` for the raw minimum.
- `showdown_count(cls)`.
- Read-only extras: `decisions`, `fold_streak`, `errors`, `recent_postflop_fold_to_bet`, `recent_raise_rate`.

**Detectors** are computed from the current evidence on every call, so each one reverts as soon as the evidence goes away.
- `is_check_folder()`: at least 12 bet-facing folds in a row, with no call, bet or raise since the first of them. Any call, bet or raise starts the run over.
- `is_station()`: postflop fold-to-bet < 0.05 with n >= 60, over the last 150 bet-facing postflop decisions.
  - It is postflop only. Preflop equities are compressed (F6), so mc_equity_fast almost never folds preflop (0.4%) and would otherwise look like a station.
- `is_maniac()`: raise rate > 0.55 with n >= 80, over the last 200 decisions where a raise was legal (all streets).
- The station and maniac windows keep them reversible late in a half. A cumulative 0.70 rate after 600 decisions would need about 260 passive decisions to drop below 0.55.

## Accuracy against the engine's ground truth
`test_stats_match_engine_ground_truth` wraps the opponent and logs every decision it actually made, with `to_call > 0` and legal raises taken from its own observation. Against calling_station, maniac, nit, mc_equity_fast, random, always_pot and guide_example, the model's `fold_to_bet` and `raise_rate` for every street, its `vpip`, and its decision count match the ground truth exactly. Every opponent decision is recovered, including the ones only the result reveals.

## Detection latency and false positives
Our side is the zoo `maniac` as a bettor. One seed per opponent, 1000 deals (300 for mc_equity_slow), no bank limit. Each cell gives the first hand flagged after `finish()`, then the share of hands flagged.

| Opponent | check-folder | station | maniac | postflop fold-to-bet | raise rate | vpip |
|---|---|---|---|---:|---:|---:|
| **checkfold_after_50** | **60** / 94.0% | - | - | 0.676 | 0.017 | 0.04 |
| **checkfold_after_300** | **311** / 68.9% | - | - | 0.299 | 0.092 | 0.25 |
| **calling_station** | - | **28** / 97.2% | - | 0.000 | 0.000 | 0.84 |
| **maniac** | - | 55 / 94.5% | **26** / 97.4% | 0.000 | 0.696 | 0.95 |
| always_half | - | 70 / 93.0% | 26 / 97.4% | 0.000 | 1.000 | 1.00 |
| always_pot | - | 138 / 86.2% | 34 / 96.6% | 0.000 | 1.000 | 1.00 |
| random | - | - | 32 / 39.5% (flickers) | 0.339 | 0.548 | 0.75 |
| llm_aggressive | - | - | 192 / 5.4% | 0.150 | 0.491 | 0.91 |
| **nit** | 122 / **2.9%** | - | - | 0.205 | 0.048 | 0.16 |
| **mc_equity_fast** | - | - | - | 0.179 | 0.137 | 0.85 |
| mc_equity_slow | - | - | - | 0.181 | 0.139 | 0.85 |
| guide_example, threshold_cat | - | - | - | 0.67 / 0.45 | 0.007 / 0.026 | 0.67 / 0.85 |
| llm_simple_rules, beginner_2h, mc_equity, opp_model, gto_inspired, ml_minded, time_careful | - | - | - | 0.22-0.34 | 0.14-0.28 | 0.47-0.84 |

**Check-folder latency against checkfold_after(50):**
- First detection at hand 60 or 61 on 8 seeds.
- The tested seeds cf1-cf4 are asserted to fall in [50, 70], with no flag at any decision of hands 0-50.
- Once on, it stays on for the rest of the half.

**False positives:**
- No station or maniac flag at any hand or decision against nit or mc_equity_fast. This is asserted for the nit (default suite) and for 2 mc_equity_fast seeds (slow tests).
- No check-folder flag against mc_equity_fast.
- The nit's check-folder rate across 3 seeds was 8-25 of 1000 hands.

**Errors:** 0 swallowed errors in every simulated half.

## Cost
Each trace was replayed 3 times; times are in µs. A background simulation was running on the same CPU during the measurement.

| Trace (1000 deals) | observe median / p99 | finish median / p90 / p99 | finish at showdown, median | model total per half |
|---|---|---|---|---|
| random | 1.7 / 3.7 | 1.3 / 42 / 59 | 36 | 19 ms |
| calling_station | 1.9 / 4.1 | 35.5 / 51 / 73 | 35.5 | 48 ms |
| maniac | 1.7 / 3.9 | 33.3 / 49 / 65 | 33.3 | 43 ms |
| nit | 1.8 / 4.0 | 1.4 / 42 / 57 | 39 | 13 ms |

- Both calls are under the 50 µs typical target.
- finish costs about 1.3 µs without a showdown. At a showdown it costs about 35 µs, almost all of it one `omaha_score` call (100 table lookups).
- The worst case is 48 ms per half, 0.6% of the 8 s bank.
- `test_cost_per_call_is_small` asserts medians under 50 µs.

## TDD evidence
- **RED:** `pytest tests/test_opponent.py` gave `ModuleNotFoundError: No module named 'bot.opponent'`, with 1 collection error.
- **GREEN:**
  - `tests/test_opponent.py`: 20 passed, 2 deselected in 6.8 s.
  - `-m slow`: 2 passed in 28.7 s.
  - **Full default suite: 204 passed, 6 deselected in 57.6 s.**
- **Mutation check:** each mutation was applied to a copy and then reverted. The suite catches all 6:

| Mutation | Failed tests |
|---|---|
| no button-fold inference | 5 |
| no fold inference | 13 |
| no showdown-call inference | 8 |
| no check-behind inference | 8 |
| `finish` without try/except | 1 (the batch test) |
| bets don't reset the run | 1 |

  The last mutation survived at first. That showed the check-folder revert test gave the opponent a bet-facing raise rather than a bet when not facing one. I fixed the test so the opponent's only decision is a bet from the big blind after our limp.

**Specified tests:**
- `test_detects_checkfolder_after_switch`, run on 4 seeds.
- `test_station_and_maniac_detection_thresholds`: exact boundaries. The station is off at n=59 and on at n=60, still on at 3/63 folds and off at 4/64. The maniac is on at 80 and off between 80/145 and 80/146.
- `test_infers_fold_to_our_bet_from_result`: fold to our flop bet, the BB-never-acted fold (pot 3, my_net +1), our own fold, a river call seen only at showdown, a river check behind, and a duplicate result.
- `test_hand_end_batch_never_raises`: one delivery of 20 items. It holds 6 real results in shuffled order with one duplicate, a BB walk, a split, `opp_hole None`, invalid and duplicate cards, a result with garbage fields, an object with only `hand_index`, `None`, a string, an int, `object()`, and negative and huge hand indexes. After the batch, junk observations and junk arguments to the getters also don't raise, and the model still records correctly.

**Extra tests:**
- engine ground truth (7 bots)
- check-folder reverts on a call and on a bet
- station and maniac found in real traces (by hand 60, and they stay on)
- showdown_min_category per action class, plus `current_class`
- cost
- runtime import purity
- 1000-deal false-positive checks: nit in the default suite, mc_equity_fast in the slow tests (about 16 s each)

## Self-review
- **Never-raise paths:** both public feeders are wrapped. Getters validate their argument and return `(0.0, 0)`, `0` or `None` on junk.
  - A bad record is never replayed twice: `seen` is set before the replay.
  - Negative street indexes are rejected, so the per-street lists can't be silently corrupted.
  - Eviction of unfinished hands uses insertion order, which can't fail on mixed key types.
- **The `pot < 2*BB` test** for hands where we never acted can't misfire. If we acted, the pot is at least 2 BB. BB is learned from `obs.big_blind`.
- **The inference relies on "at most one opponent decision after our last action".** This holds because any non-terminal opponent action gives us another decision, and all-in run-outs end in a showdown. The exact match with the engine's ground truth confirms it in practice.
- **The raise sizes that `finish` matches assume `o.raise_to`** reflects what we played. If the policy ever returns an illegal action, the judge's default replaces it. A mismatch then only marks the call as "raise not legal", which removes it from the raise-rate denominator; fold-to-bet is unaffected.

## Concerns
1. **Deviation: check-folder raise window.**
   - The brief says "0 raises in the last 40 decisions", but read literally it can't meet the required "detected by hand 70". A bot that empties its bank mid-half raised shortly before the switch. The opponent makes about 1.3 decisions per hand after the switch, so those raises stay in a 40-decision window for about 30 hands.
   - Measured against checkfold_after(50) on 8 seeds: the literal window first detects at hands 60, 61, 68, 68, 70, 71, 75 and 77 (3 of 8 miss hand 70). Scoped to the run, it detects at 60-61.
   - Implemented rule: no call, bet or raise since the first fold of the run. Any bet or raise restarts the run.
   - Cost: against the nit, the literal window flags 3-14 hands per 1000 and the run-scoped rule flags 8-25. Twelve folds in a row from an 80%-preflop folder look exactly like check-folding.
   - The I2 exploit (RAISE_HALF) is still +EV against such over-folders: about 50% break-even preflop against an 80% fold rate. It also turns off on the nit's first call.
2. **Precedence is Task 14's call.**
   - Bots that never fold and always raise (maniac, always_half, always_pot) trip both maniac and station, which is correct by both definitions.
   - The maniac detector flickers against `random` (39.5% of hands) and `llm_aggressive` (5.4%), whose raise rates sit near 0.55. Calling lighter against them is arguably right.
   - Suggested order: check-folder > maniac > station.
3. **The model is only as good as its feed.** If `choose_action` fails before `observe()`, the last observation of that hand is stale, and an inferred terminal action can land on an earlier street. Call `observe()` first, inside its own guard.
4. **Not done:** `agent.py` wiring, which is out of scope for the brief. Task 14 should pass `config.raise_cap` and `config.big_blind` to the constructor so that retuned constants are handled.
