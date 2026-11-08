# Task 14 report: exploit modifiers, v2

Track C: worktree `/home/kchen/code/heads-up-bundle-opp`, branch `kevin-opp`, BASE 3d8f162.

**Status: DONE_WITH_CONCERNS.**
- **Commit:** a9ffa6b "Add the exploit layer on top of the heuristic and feed the opponent
  model from PolicyBot for v2." Not pushed, not submitted (R12).
- **Gates:** the M2 gate and both plan sim gates pass.
- **Concerns:**
  - I6 deviates from the plan's literal ">=", with TRAIN evidence (concern 1).
  - The full suite has 1 failure: a latency test in an untouched module, under load 22 (concern 2).
  - The sim's wall-clock banks are contention-dominated on today's shared machine (concern 3).

## What I implemented

- **`bot/exploit.py`** (new, 147 lines).
  - `apply_exploits(base, obs, feats, opp, rng, params=None, stats=None) -> Action`, plus
    `active_mode(opp)`, `station_params(p)` and `maniac_params(p)`.
  - Module tunables: `STATION_VALUE_SHIFT = 0.08`, `MANIAC_CALL_SHIFT = 0.06`,
    `MANIAC_RANGE_WEIGHT = 0.5`, `BIG_BET_FRACTION = 0.4`, `I6_CATEGORY_MARGIN = 1`.
    `STAT_KEYS` names the per-half counters.
  - Two optional keyword arguments extend the plan's signature. `params` are the heuristic
    params to shift: PolicyBot passes `self.params`, so T12's tuned params get shifted,
    not the defaults. `stats` is the counter dict.
  - `rng` is accepted but never drawn from (guards below).
- **`bot/policy.py`**, the T11/T13 wiring.
  - `new_half` builds `OpponentModel(raise_cap=config.raise_cap,
    big_blind=config.big_blind)` (defaults on a bad config) and `memory["exploits"]`
    (zeroed counters).
  - `_decide` calls `memory["opp"].observe(obs)` first, before any layer, inside its own
    try.
  - `on_hand_end` calls `finish(result)`, then charges the time with `add(dt)`.
  - The identity exploit slot now calls `apply_exploits(base, obs, feats, opp,
    memory["rng"], self.params, memory["exploits"])`.
  - The model is fed with `use_exploits=False` too. It is cheap, and it means v1 and v2
    differ only in the layer.
  - T11 M5: the docstring now reads "base = blueprint else heuristic; exploits then modify
    the base", and the 132-character line is gone.
- **`agent.py`**: the T11 M5 comment, in the YOUR CODE section only.
- **Tests.**
  - `tests/test_exploit.py` (new): 16 fast tests and the plan's 2 slow sim gates. The
    slow gates print their CI.
  - `tests/test_policy.py`: T11 M1 (`tm.used > before`), the T11 M3 positive-path test
    `test_a_legal_exploit_change_replaces_the_base` (a legal change is played; an illegal
    one is a fault and the base stands), and the opp slot now asserts an OpponentModel.
- **`docs/results/v2.md`**: the results (tables, gates, activation, timing, method).

## The exploit modes: trigger and effect

At most one mode applies per decision, by precedence check-folder > maniac > station (T13
note). I6 applies on top of any mode. Each decision re-reads the detectors, so every mode
stops the moment its evidence does.

| Mode | Trigger (bot/opponent.py) | Effect |
|---|---|---|
| check-folder (I2) | the last 12 bet-facing decisions were folds, with no call/bet/raise since the run began | RAISE_HALF whenever legal, otherwise the base |
| maniac | bet/raise rate > 0.55 over the last 200 raise-legal decisions (n >= 80) | the heuristic re-run with `call_margin` -0.06, `bet_range` -0.06 per street, `range_weight` capped at 0.5 (T10 note: these dominate call_margin facing a bet), `bluff_freq` 0 |
| station | postflop fold-to-bet < 0.05 over the last 150 bet-facing postflop decisions (n >= 60) | the heuristic re-run with `value_raise` AND `thin_value` -0.08 (T10 note), `bluff_freq` 0 |
| I6 | river; facing a bet >= 0.4 x the pot before it; the weakest category shown after betting/raising the river (>= 5 showdowns) >= our category + `I6_CATEGORY_MARGIN` (1); no nuts | FOLD instead of CHECK_CALL |

Guards and design choices:
- **Never fold more.** The re-run modes never fold a hand the base plays on.
- **No rng draws.** The re-run uses a stand-in rng (with bluff_freq 0 the draw cannot
  matter). A decision the layer leaves alone is therefore exactly v1's, and so is the
  rest of the Monte Carlo stream. Unit test:
  `test_no_evidence_means_the_base_and_no_rng_draws`.
- **feats None.** With a blueprint base (Task 20) there are no Features. Only
  check-folder and I6 then apply; the re-run modes keep the base.
- **I6 is river only.** On earlier streets the opponent's showdown category is its river
  category. It is also seen only for hands that reached showdown (survivor bias).
- **Big bet.** to_call / (pot - to_call) >= 0.4, which means a half-pot bet (0.5) or more,
  with slack for odd-pot rounding.

## Per-mode activation rates: every TRAIN and HOLDOUT bot

Activation is the share of v2's decisions with the mode's detector on. "Changed" is the
share where the layer changed the action. The data is the final paired runs:
- TRAIN: 3 seeds x 1000 deals, both halves, load 16-20;
- HOLDOUT: 6 seeds x 1000 deals, load 17-19.

| Split | Opponent | check-folder % | maniac % | station % | changed % | I6 folds |
|---|---|---:|---:|---:|---:|---:|
| TRAIN | always_half | 0.0 | 97.4 | 0.0 | 8.2 | 0 |
| TRAIN | always_pot | 0.0 | 96.7 | 0.0 | 11.6 | 0 |
| TRAIN | calling_station | 0.0 | 0.0 | 93.9 | 8.5 | 0 |
| TRAIN | checkfold_after_300 | 23.3 | 0.0 | 0.0 | 12.3 | 1 |
| TRAIN | guide_example | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| TRAIN | llm_beginner_2h | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| TRAIN | llm_gto_inspired | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| TRAIN | llm_mc_equity | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| TRAIN | llm_ml_minded (adaptive) | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| TRAIN | llm_simple_rules | 15.1 | 0.0 | 0.0 | 8.0 | 1 |
| TRAIN | maniac | 0.0 | 97.5 | 0.0 | 8.8 | 0 |
| TRAIN | mc_equity_fast | 21.9 | 0.0 | 0.0 | 11.6 | 0 |
| TRAIN | mc_equity_slow | 77.3 | 0.0 | 0.0 | 40.3 | 0 |
| TRAIN | nit | 1.7 | 0.0 | 0.0 | 0.9 | 0 |
| TRAIN | random | 0.0 | 80.0 | 0.0 | 7.3 | 0 |
| HOLDOUT | llm_aggressive | 0.0 | 96.6 | 0.0 | 9.2 | 0 |
| HOLDOUT | llm_opp_model (adaptive) | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| HOLDOUT | llm_time_careful | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| HOLDOUT | threshold_cat | 0.0 | 0.0 | 0.0 | 0.0 | 2 |

False positives:
- **check-folder.** It fires only where the opponent really plays default actions:
  checkfold_after_300 after hand 300, and mc_equity_slow, mc_equity_fast and
  llm_simple_rules. Those three ran out of their 8 s bank in 6/6 halves (the sim's
  bank_exhausted flag, checked in the final run). The one false positive is nit (1.7%):
  T13's ruled-harmless case, which reverts on the first call. v2 - v1 there is -0.017
  [-0.075, +0.041].
- **maniac and station.** No false positive. Maniac fires only on bots raising > 55%;
  station only on calling_station.
- **Adaptive bots** (llm_ml_minded, llm_opp_model): no mode fired.

## Paired CIs, v2 - v1 (bb/deal, 95%)

| Comparison | Seeds | Result |
|---|---|---|
| checkfold_after(300), plan test `test_gain_vs_checkfold_after_300` | cf300-1..3 | **+0.203 [+0.027, +0.379]** (gate: lower > 0, pass) |
| checkfold_after(300), first run (load ~14) | t14-1..3 | +0.078 [+0.035, +0.121] |
| checkfold_after(300), final TRAIN run (load 16-20) | t14-1..3 | +0.127 [-0.053, +0.306] |
| TRAIN pooled, 15 bots, final run | t14-1..3 | **+0.836 [+0.713, +0.959]** (45,000 deals) |
| HOLDOUT pooled, 4 bots, final run | t14-1..6 | **+0.320 [+0.147, +0.494]** (24,000 deals) |
| HOLDOUT, plan test `test_no_regression_vs_untargeted` | ho-1..3 | **+0.315 [+0.067, +0.564]** (gate: lower > -0.05, pass) |

The per-bot TRAIN and HOLDOUT rows (CI, points, bb/deal, errors, bank) are in
`docs/results/v2.md`. Highlights:
- maniac +3.92, always_half +2.91, random +1.80, always_pot +1.42, calling_station +0.78;
- held-out llm_aggressive +1.42 [+0.78, +2.06];
- the bank-out bots: mc_equity_fast +0.30 [+0.12, +0.49], llm_simple_rules +0.27
  [+0.09, +0.46], mc_equity_slow +0.15 [+0.08, +0.23].

With no mode on, the spread is wall-clock Monte Carlo noise, for example llm_opp_model
-0.18 [-0.40, +0.04] with no mode and no I6 fold. See concern 3.

## M2 gate table

From the HOLDOUT final run: 4 bots x 6 seeds, the same 24 matches for v1 and v2.

| Opponent | Points v1 | Points v2 | W/C/L v1 | W/C/L v2 | errors v1 / v2 | layer faults v2 | max bank v1 / v2 (s, sim wall clock) |
|---|---:|---:|---|---|---|---:|---|
| llm_aggressive | 0.92 | 1.00 | 5/1/0 | 6/0/0 | 0 / 0 | 0 | 3.19 / 3.09 |
| llm_opp_model | 1.00 | 1.00 | 6/0/0 | 6/0/0 | 0 / 0 | 0 | 1.58 / 1.23 |
| llm_time_careful | 1.00 | 1.00 | 6/0/0 | 6/0/0 | 0 / 0 | 0 | 0.65 / 0.73 |
| threshold_cat | 1.00 | 1.00 | 6/0/0 | 6/0/0 | 0 / 0 | 0 | 0.89 / 0.94 |

| M2 criterion | Result |
|---|---|
| Expected points on held-out >= v1 | v1 0.979, **v2 1.000**: pass |
| No new errors | 0 errors, 0 forfeits and 0 v2 layer faults in all 276 halves (TRAIN + HOLDOUT, both versions). The slow whole-zoo test of agent.py (now v2): 19/19 pass. Pass |
| Bank < 3.0 s | **Compute:** CPU-time bank (judge's charge rule on `thread_time`) max 1.22 s per half for v2, 1.08 s for v1. Pass. **Sim wall clock under load:** 3.09 s for v2, 3.19 s for v1, one held-out half each; see concern 3 |

TRAIN: expected points v1 0.967, v2 1.000. Against llm_ml_minded v1 went 0/3/0 and v2
3/0/0, but no mode fired there: that is noise in a thin-margin match, not an exploit
gain. FIELD is not registered at gate time: no FIELD set exists in `sim/opponents`, and
`../competitor-bots` is still being built. So there is no v2 vs FIELD.

## Timing

`timing_probe.py`: v1 and v2 alternate half by half in one process, 2 halves each,
1000 deals, CPUs 0-7, load 14.7-21.9. Figures are CPU time (`time.thread_time`) per
choose_action call, as median / p99 in ms, then bank_cpu max in s.

| Opponent | v1 | v2 |
|---|---|---|
| maniac | 0.650 / 4.67 ms, 1.08 s | 0.663 / 4.80 ms, 1.22 s |
| calling_station | 0.637 / 4.50 ms, 0.99 s | 0.657 / 4.68 ms, 1.01 s |
| checkfold_after_300 | 0.714 / 4.80 ms, 0.43 s | 0.705 / 4.21 ms, 0.36 s |
| llm_aggressive | 0.687 / 4.93 ms, 1.04 s | 0.713 / 5.24 ms, 1.14 s |

- **Overhead.** v2's median per decision is within -0.01 to +0.03 ms of v1's, at most +4%.
- **on_hand_end.** The median is 0.005-0.04 ms for both: v1 feeds the model too.
- **Layer cost.** apply_exploits in a re-run mode costs 3.7 us per decision
  (`dataclasses.replace` + the re-run).

## Tests and results

| Run | Command | Result |
|---|---|---|
| Focused, at HEAD a9ffa6b | `taskset -c 0-7 nice -n 10 .venv/bin/python -m pytest -q tests/test_exploit.py tests/test_policy.py test_agent.py` | **60 passed**, 21 deselected, in 20 s (load 19.5) |
| Full fast suite | `taskset -c 0-7 nice -n 10 .venv/bin/python -m pytest -q` | **274 passed, 1 failed**, 31 deselected, in 123 s (load 14-22). The failure is `tests/test_equity.py::test_timing_p99_under_1_5ms_at_128_iters_flop` (concern 2) |
| Plan sim gates | `taskset -c 0-7 nice -n 10 .venv/bin/python -m pytest -m slow -q -s tests/test_exploit.py` | **2 passed** in 218 s (load 15.5), CIs above |
| agent.py (v2) vs the whole zoo, 19 bots x 200 deals: 0 errors, 0 faults, bank < 3 s | `... -m pytest -m slow -q tests/test_policy.py -k whole_zoo` | **19 passed** in 55 s (load 21-22) |

What the fast tests in `tests/test_exploit.py` cover:
- **check-folder.** Only RAISE_HALF or the base, always legal (the plan test).
- **check-folder in real halves.** It bets every legal spot vs checkfold_after(0) and
  takes the blinds. It stops at a fighting-back bot's first raise
  (`_FoldThenFight`: check/fold 100 hands, then raise every hand; read as a maniac
  later).
- **station.** It bets exactly above the bar lowered by 0.08, never bluffs, folds exactly
  when the base folds, raises whenever the base raises, and leaves preflop unchanged.
- **maniac.** It never folds what the base plays, continues where the base folds (more
  than 10 spots), never bluffs, and its param shifts are the ones specified.
- **Precedence and identity.** Precedence is check-folder > maniac > station. With no
  evidence the result is the base and no rng draw is ever taken. The layer follows the
  detectors both ways. Without Features, only check-folder and I6 apply.
- **I6.** It folds below every shown category, keeps calling with the same category,
  needs the river, a big bet and no nuts, and works without Features.
- **Wiring.** The model is fed by choose_action (observe) and on_hand_end (finish, seen
  through showdown counts). Counters are kept. new_half takes the config's raise_cap,
  with defaults on garbage. use_exploits=False never calls the layer.

## TDD evidence

RED 1, before `bot/exploit.py` existed, with `new_half` still writing `memory["opp"] = None`:

```
$ taskset -c 0-7 nice -n 10 .venv/bin/python -m pytest -q tests/test_exploit.py tests/test_policy.py -x
E   ImportError: cannot import name 'exploit' from 'bot' (/home/kchen/code/heads-up-bundle-opp/bot/__init__.py)
ERROR tests/test_exploit.py
$ ... -m pytest -q tests/test_policy.py -k "planned_memory or legal_exploit_change or charges_time"
>       assert isinstance(memory["opp"], OpponentModel)
E       assert False
E        +  where False = isinstance(None, OpponentModel)
FAILED tests/test_policy.py::test_start_half_sets_the_planned_memory - assert...
```

This was the expected failure: the module did not exist, and the slot was still None.
The T11 M1 (`>`) and M3 positive-path tests passed at once. They pin behaviour T11
already had (the slot replaces a legal answer, and add(dt) is always > 0), so they are
characterization tests, as the carry-forwards asked.

GREEN 1, in three runs:
1. `... -m pytest -q tests/test_exploit.py tests/test_policy.py` gave 44 passed and 2
   failed. The 2 failures were a test-data shortfall: 80 sampled hands gave only 4
   qualifying river calls for the I6 tests.
2. With the sample widened to 200 hands, `... -m pytest -q tests/test_exploit.py` gave
   **14 passed**, 2 deselected.
3. After the observe guard, both files with `-k "not first_decision and not
   warmup_under"` (the two subprocess timing tests skipped) gave **45 passed**, 23
   deselected.

RED 2 is the I6 margin. These are the I6 tests run under the plan's literal rule (margin
patched to 0):

```
$ .venv/bin/python -c "import bot.exploit as e; e.I6_CATEGORY_MARGIN = 0; import pytest; \
    raise SystemExit(pytest.main(['-q', 'tests/test_exploit.py', '-k', 'same_category or always_beat']))"
E               assert <Action.FOLD: 0> == <Action.CHECK_CALL: 1>      (a same-category river call folded)
E       assert 0 == 1                                                  (the margin pin)
2 failed, 16 deselected
```

GREEN 2: with margin 1, `... -m pytest -q tests/test_exploit.py` gave **16 passed**, 2
deselected. The fast tests grew from 14 to 15 to 16: first the reversibility test, then
the same-category pin.

## Files changed (a9ffa6b)

- `bot/exploit.py` (new, 147 lines)
- `bot/policy.py` (+55/-16, now 500 lines)
- `agent.py` (a comment in YOUR CODE)
- `tests/test_exploit.py` (new)
- `tests/test_policy.py`
- `docs/results/v2.md` (new)

Untouched, as the brief requires: HeuristicParams defaults, `sim/match.py`, and every
organizer file.

Scratch runners, not committed (the controller can reuse them for the gate re-run), in
`/tmp/claude-1000/-home-kchen-code-heads-up-bundle/c0d55f53-d291-4faf-88d2-40523d153176/scratchpad/`:
- `measure.py`: the paired runner with v2 counters,
- `summarize.py`, `make_doc.py`: tables,
- `i6_diag.py`: the I6 counterfactual check,
- `timing_probe.py`: CPU-time timing,
- `train3_c.json`, `holdout6_c.json`: the final runs.

## Self-review findings (fixed before the commit)

- **I6 at margin 0 folded coin flips.** The plan's literal rule folded same-category
  river hands; a counterfactual check showed 6 of 11 would have won. Fixed with
  `I6_CATEGORY_MARGIN = 1` (concern 1).
- **Merged pairs were invalid.** My first attempt paired a v2 re-run with older v1
  halves. Under different CPU contention the wall-clock-paced time manager draws different
  rollout counts, which gave bogus results (cf300 -0.10, ml_minded points 0.50). I
  discarded it and re-ran v1 and v2 side by side in one run.
- **Belt and braces.** `observe()` in `_decide` got its own try, although observe never
  raises: `_decide` sits outside any `except` in choose_action, and agent.py calls
  `BOT.choose_action` directly.
- **Junk slots.** The exploit slot passes `_opponent(memory)` (OpponentModel or None),
  not `memory["opp"]`, so a junk slot yields the base without a fault.
- **Small fixes.** I removed an unused `MODES` constant and renamed a test variable that
  shadowed a counter. I wrapped the over-long lines, and I made the docstring's
  bet-sizing claim precise ("no mode adds a link between bet size and hand strength").

## Issues and concerns

1. **I6 deviates from the plan's literal rule (needs a ruling).** The plan says fold when
   "showdown_min_category >= our made category". I fold only when it is at least one
   category above ours (`I6_CATEGORY_MARGIN = 1`).
   - The evidence is TRAIN only: 11 margin-0 folds on 6 TRAIN bots, all same-category
     spots. Calling would have won 6 of them, at 25-33% pot odds: -292 chips in all.
   - HOLDOUT (threshold_cat: 18 margin-0 folds, -0.009 [-0.043, +0.024]) was seen before
     the diagnostic and did not drive the decision.
   - Setting the constant to 0 restores the literal rule.
   - With margin 1, I6 fires rarely (0-2 folds per bot per run). Its main value is
     against bots that bet the river only with a category above ours.
   - I also made I6 river only, because earlier-street showdown categories are
     river-final and survivor-biased.
2. **Full-suite failure, unrelated to this diff.**
   `tests/test_equity.py::test_timing_p99_under_1_5ms_at_128_iters_flop` measured median
   0.30 ms but p99 2.47 ms (limit 1.5) at load 22.5. It failed 3/3 when re-run alone at
   that load. `bot/equity.py` and `tests/test_equity.py` are untouched (`git diff
   3d8f162` is empty for both). It is a preemption tail; re-run it on a quiet machine.
3. **The sim's wall-clock bank is contention-dominated today.** Task 12's tuner ran eight
   nice-0 processes on CPUs 0-7 against my two nice-10 workers.
   - Paired held-out maxima were v1 3.19 s and v2 3.09 s.
   - In one v2-only re-run at that contention, v2 banked out (8.01 s) in one half vs
     llm_beginner_2h; v1 had reached 3.03 s on that opponent and seed in a lighter run.
   - v2's own compute is +2-4% per decision, and its CPU-time bank peaks at 1.22 s.
   - Still: wall-clock pressure on a shared host can exhaust the bank whatever the bot
     does. The 6 s hard stop only removes Monte Carlo; per-call overhead above the grace
     still drains the bank. That is a T9/T11/T24 question, worth a quiet-machine or
     docker `--cpus 1` re-check before submission.
   - The same contention makes paired diffs noisy where no mode fires (about +-0.2 bb/deal).
4. **The gate needs a re-run on the merged head after T12 (R34).** Expected merge
   conflicts in `bot/policy.py`:
   - the docstring's memory-key list;
   - `new_half`'s try block (T12 adds `time_scale` and a scaled clock);
   - `on_hand_end`. T12 switches it to `clock = tm.clock` and `t0 = clock()`. Keep
     `opp.finish(result)` between `t0 = clock()` and `tm.add(clock() - t0)`, so its time is
     charged on the scaled clock.

   T12's new `HeuristicParams.__post_init__` validation is compatible with
   `station_params` / `maniac_params`, which build tuples of 3 floats. The re-run then
   costs a few us more per in-mode decision.
5. **Residual adaptive risk.** Station and maniac read T13's 150/200-decision windows. A
   bot that plays maniac and then tightens keeps us calling wider for up to about 100 of
   its decisions, until its window rate falls below 0.55. Check-folder reverts at once.
6. **Not run:** `./tools/check_build.sh` (R12: the controller's), and v2 vs FIELD (not
   registered).
7. **Code size.** `bot/policy.py` is now 500 lines. T11 M6 (move the CPU-factor machinery
   out) was not in my carry-forwards and is untouched.

## Late addendum

Background task `bef6soxhp` (the I6 counterfactual diagnostic) was reported "failed, exit
144" after the report was delivered. Exit 144 is from the waiting wrapper, killed in its
sleep loop. The diagnostic itself had finished and written complete logs (`i6_a.log`,
`i6_b.log` in the scratchpad). The report's figures are exactly its output, so concern
1's evidence is unchanged.

The diagnostic ran v2 with I6 at margin 0 (the code before the change). It used TRAIN
bots only, seeds i6-1..3, 1000 deals per half, both halves.

| TRAIN bot | I6 folds | calling would have won | would have lost | chips (lost calls saved - pots given up) |
|---|---:|---:|---:|---:|
| llm_beginner_2h | 0 | 0 | 0 | 0 |
| guide_example | 4 | 1 | 3 | +16 |
| nit | 0 | 0 | 0 | 0 |
| llm_mc_equity | 3 | 3 | 0 | -132 |
| llm_gto_inspired | 3 | 1 | 2 | -60 |
| mc_equity_fast | 1 | 1 | 0 | -116 |
| **total** | **11** | **6** | **5** | **-292** |

All 11 folds were same-category spots (our category == the weakest shown).
