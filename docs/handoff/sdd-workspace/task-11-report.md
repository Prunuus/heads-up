# Task 11 report: layered policy and agent.py wiring

**Status: DONE_WITH_CONCERNS.** Everything in the brief and the controller additions is implemented and tested. The concerns are listed at the end; none of them blocks T12. The main one is that the online CPU-factor sample depends on what ran just before it, so sims read a higher factor than the tournament will.

Commits on `kevin`:
- `4636d2c` feat: layered policy wired into agent.py. This is the old message format with a body and trailer; it predates the coordinator's one-sentence rule and was not rewritten.
- `09bd939` Return a CPU factor of 1.0 when warmup measures no runs. (`measure_cpu_factor(runs=0)` returned F_MAX before; this commit adds its test.)

## What I implemented

### `bot/policy.py` (new)
- **`PolicyBot(params=None, use_blueprint=True, use_exploits=True)`**, BotLike (R5).
  - `new_half` sets these memory keys every time:
    - `tm`: `TimeManager(bank, hands, grace_s=grace_for_cpu_factor(cpu_factor))`
    - `rng`: `default_rng(20261107)`
    - `opp`: `None` until T13
    - `config_ok`
    - `cpu`: the online estimator
    - `faults`: per-layer counters
  - `new_half` never raises, even with a garbage config.
  - The class itself and `functools.partial(PolicyBot, ...)` are picklable factories.
  - The instance keeps settings only. Per-half state lives in memory (R22).
- **Layers in `choose_action`.** The order of precedence is blueprint → exploits → heuristic.
  1. Blueprint: only when one is loaded and `memory["config_ok"] is True`.
  2. Heuristic: `compute_features` + `heuristic_action`, whenever the blueprint gives no answer.
  3. Exploits: modify the base action, so this layer runs last.

  Each layer has its own `try/except Exception`, and each result must be a legal `Action`. Anything else (an exception, an illegal action, a non-Action) counts in `memory["faults"]` and moves on:
  - a failed blueprint falls to the heuristic;
  - a failed exploit keeps the base action;
  - with no legal answer, the result is `fallback_action(obs)`: CHECK_CALL if `to_call == 0`, else FOLD, filtered by `legal_actions`; it never raises.

  The final result is checked against `legal_actions` again. The fault counters exist so a sim can tell "0 errors" apart from "every layer silently fell back". The zoo tests assert 0 faults.
- **Empty slots (R10).**
  - `_blueprint_action` returns None, the "no answer" contract T20 fills.
  - `_exploit_action` is the identity (T14).
  - `_load_blueprint()` returns None (T19/T20).
  - `config_ok` is computed against `blueprint.meta` when a blueprint is loaded, else against `DEFAULT_META` (stack 80, sb 1, bb 2, cap 4, the game the T15 tree is built for). It compares stack, blinds and cap; `num_hands` and the time bank don't change the game. `config_matches` never raises: a missing key or a garbage config gives False.
- **Time (R9 / T9).**
  - `tm.begin()/end()` wrap every `choose_action` in try/finally.
  - `on_hand_end` charges its own time with `tm.add(dt)`, with no grace, and never raises.
  - The fixed seed rng and the TM live in memory.
  - Monte Carlo is skipped at `iters == 0` by `compute_features`; nut_flags still runs, per ruling M1.
  - No lazy init happens in the first decision (measured below).
- **CPU factor (R30a + T9 grace ruling).**
  - **Warmup measurement.**
    - Workload: `reference_workload` is a table-free stand-in for the server plumbing the bot cannot see: JSON in → `Observation.from_dict` → `to_dict` → JSON out, 20 reps.
    - Statistic: best of 60 runs (about 25 ms in all), divided by `REFERENCE_WORKLOAD_S`.
    - The factor is floored at 1.0, clamped to `F_MAX = 8` (where `grace_for_cpu_factor(8)` is −1 ms/call), and 1.0 on any failure.
    - The grace passed to the TimeManager is `grace_for_cpu_factor(max(warmup factor, online factor))`, unclamped as T9 ruled.
  - **Online re-estimate (`CpuFactorEstimator`).**
    - Sample: the time of every preflop heuristic layer call. That is a fixed workload, with no Monte Carlo preflop.
    - Statistic: the median of a sliding window of the last 64 samples, compared with the dev-PC median `PREFLOP_REF_S`. It needs at least 16 samples, is floored at 1 and clamped to F_MAX.
    - Bad samples are dropped: non-finite, ≤ 0 and non-numbers. An unusable reference gives None, and the method never raises.
    - The grace update runs in its own try, so it can never discard the heuristic's answer.
    - Why max(): the online estimate can only shrink the grace, never grow it past what warmup measured. A stale or noisy online reference therefore errs in the safe direction (over-counting). A sliding median also forgets old samples, so it recovers after a slow spell.
- **`warmup()`.**
  - Steps:
    - the evaluator table (omaha_score, which T13's showdown scoring needs);
    - `equity.warmup()`;
    - the preflop table, touched through `preflop_strength`;
    - `nut_flags` once per board width 3/4/5;
    - the blueprint slot;
    - 8 dummy `compute_features` + `heuristic_action` spots (every street, facing a bet and not, at iters 0 and 16);
    - the CPU factor last.
  - Each step has its own try. A failed step is listed in `WARMUP.failed`, and warmup never raises.
  - It is idempotent, once per process, and records `WARMUP = Warmup(seconds, cpu_factor, blueprint, failed)`.
- **Module-level `warmup / start_half / decide / hand_finished`.** Thin wrappers over a default `PolicyBot` (the plan's interface). `decide` and `hand_finished` never raise.
- **Documented (AUDIT-FIX note).** Never pickle bot.* objects during play, and import at module top. I verified that pickling a TimeManager from a `load_bot_file` copy raises `PicklingError: ... not the same object as bot.timing.TimeManager`; `copy.deepcopy` works.

### `agent.py` (warmup area + YOUR CODE section only)
- Warmup area: `from bot import policy` and `policy.warmup()`, after the organizer's `warmup()`.
- YOUR CODE: `BOT = policy.PolicyBot()`. `new_half / choose_action / on_hand_end` delegate to it.
- PARTICIPANT stays "3XKO". SERVER PLUMBING and PORT are untouched.

### `sim/match.py`
- `load_bot_file`'s docstring now states that pickling objects of the private package copy during play fails, and that factories are unaffected.

### `tests/test_policy.py` (new): 31 default tests + 19 slow
- The plan's four:
  - `test_layer_exception_falls_back_legally`
  - `test_config_mismatch_uses_heuristic`: a fake blueprint plus a spy on the slot. Under a matching config it is consulted. With `game.STARTING_STACK` monkeypatched to 100 and `AgentConfig(starting_stack=100)`, 40 hands of a real 100-chip game: the spy is never called, all actions are legal, 0 faults.
  - `test_warmup_under_6s`: a cold `import agent` in a subprocess.
  - `test_zero_errors_vs_whole_zoo`: slow, 19 parametrized halves of 200 deals through the agent.py file the sim loads. It asserts `errors_a == 0`, no forfeit, `bank_a < 3.0` and 0 faults.
- Fast smoke version in the default suite: 5 bots × 25 deals.
- Also:
  - illegal and odd layer results (an Action that is legal only sometimes; `"RAISE_POT"`, `3`, `None`, `7.5`);
  - exploit and blueprint faults keep the heuristic;
  - corrupt observations (ValueError, IndexError, TypeError) fall through;
  - `fallback_action` filtering;
  - `hand_finished` never raises (garbage results and memories) and charges time;
  - decisions without `start_half`;
  - the memory keys, fixed seed and grace;
  - `config_ok` false for each game change;
  - `config_matches` never raises;
  - picklable factories;
  - the CPU factor: floor, clamp, NaN, 0, `runs=0`;
  - online estimator robustness: zeros, −0, negatives, NaN, ±inf, None, str, list, a third of samples being 1000× spikes, the floor, the clamp, forgetting old samples, a bad reference;
  - the online estimate drives the grace, and warmup's factor wins when it is higher;
  - warmup is idempotent and recorded;
  - a missing or corrupt preflop table (truncated npz, a BadZipFile) never crashes warmup or decisions;
  - a failing warmup step doesn't stop the rest;
  - **the first decision per street costs the steady state**: a cold subprocess per run, up to 3 cold starts, threshold `first ≤ 2·steady + 1 ms`;
  - **agent.py end to end through the FastAPI TestClient**: /new-half, 40 hands of /act with hand results batched into the next /act, and a /hand-end batch of the pending result plus 20 more (incl. hands we never acted in, splits and showdowns). Then /new-half with stack 100 gives `config_ok False`;
  - `resolve_bot("agent")` pickles and plays a sim half.

## Test results

- **Default suite (final tree):** `taskset -c 0-7 .venv/bin/python -m pytest -q` → **238 passed, 27 deselected, 1 warning in 73.23 s** (final tree `09bd939`, load 7.95 → 10.27 with T17 back on 8–15). On `4636d2c` it was 238 passed in 50.25 s at load 3.69 → 2.42, and 44.20 s at load 4.76 → 2.78. Before my change it was 207 passed in 38.8 s. The one warning is the pre-existing StarletteDeprecationWarning (httpx).
- **Slow policy tests:** `taskset -c 0-7 .venv/bin/python -m pytest -q -m slow tests/test_policy.py` → 19 passed in 32.5 s (load 15.5–16.6) and 28.62 s (load 2.4 → 2.1). Final tree `09bd939`: **19 passed, 31 deselected in 37.50 s** (load 10.3). `tests/test_policy.py` default: **31 passed, 19 deselected in 4.23 s**.
- **`sim.tournament --candidate agent` end to end:** `taskset -c 0-7 .venv/bin/python -m sim.tournament --candidate agent --opponents holdout --seeds 1 --seed-prefix e --deals 200 --workers 4` ran in 5 s (spawned workers, so the factory pickles). Expected points 0.875, mean +2.211 bb/deal.

### TDD evidence
- **RED:** `taskset -c 0-7 .venv/bin/python -m pytest -q tests/test_policy.py` → `ImportError: cannot import name 'policy' from 'bot'` (collection error, 1 error in 0.44 s). Expected: `bot/policy.py` did not exist yet.
- **First green attempt:** 2 failed, 29 passed. Both were my tests' own bugs:
  - `memory["cpu"].n == len(pre)` ignored the window cap of 64; it is now `min(len(pre), WINDOW)`;
  - the session's first `PolicyBot()` ran warmup inside a test that had already monkeypatched `heuristic_action`, so warmup recorded `failed=('decisions',)`. The test module now calls `policy.warmup()` at import, as agent.py does.
- **GREEN:** the same command → `31 passed, 19 deselected, 1 warning in 3.52s`.
- **RED/GREEN for the follow-up fix:** a new assertion `measure_cpu_factor(runs=0) == 1.0`. The committed HEAD returned 8.0 (F_MAX); the fixed tree returns 1.0. I loaded HEAD's policy.py in a throwaway module because the working tree was in use by a background run. GREEN: the full default suite on the fixed tree passed (238 passed, above), with the new assertion included in `test_cpu_factor_measurement_floors_and_clamps`.

## Measured numbers (all pinned `taskset -c 0-7`; load average with each)

T17 started, paused and restarted on CPUs 8–15 during the session, so measurements alternate between quiet (load about 2) and loaded (load 8–16) periods. A busy PC is measurably slower on CPUs 0–7 too: about 23% on the reference workload (shared L3 and lower boost clocks).

### Warmup and startup
| What | Value | Load |
|---|---|---|
| `policy.warmup()` (after the organizer's evaluator warmup) | 0.054–0.063 s | 2.3–2.5 |
| `policy.warmup()` (after the organizer's evaluator warmup) | 0.072 s | 8.4 |
| `policy.warmup()` (after the organizer's evaluator warmup) | 0.103–0.117 s | 3.4 (T17 restarting) |
| Full cold `import agent` (fastapi/uvicorn imports, evaluator table, policy warmup, PolicyBot) | 0.59–0.65 s | 3.4–3.5 |
| Budgets | ≤ ~2 s (R30c); test < 6 s | |

Component costs (load 0.05): numpy import 37 ms, evaluator table 126–147 ms, fastapi+uvicorn import 148–234 ms, equity.warmup 15–26 ms, preflop load 9–11 ms, the first nut_flags 3.4–6.1 ms.

### CPU-factor reference and how it was measured
- **`REFERENCE_WORKLOAD_S = 0.336 ms`.**
  - Each process: `policy.warmup()`, then the same best-of-60 runs of `reference_workload(20 reps)` that warmup uses. One fresh process per measurement, 12 processes, `taskset -c 0-7`.
  - Quiet machine (load 1.99 → 2.47): range 0.331–0.341 ms; I took the median.
  - With T17 on 8–15 (load 8.5): 0.403–0.423 ms, median 0.413. A loaded PC therefore reads factor 1.17–1.25, which I observed directly at load 3.5–4.4.
  - Why best-of-60 short runs: one run of 100 reps varied 2.2–5.9 ms within a process, from environment noise (GC off gave the same spread). Best-of-15 long runs varied ±15% across processes; best-of-60 short runs varied ±2.5%.
- **`PREFLOP_REF_S = 13.7 µs`.**
  - Setup: 3 real-judge halves of 1000 deals each. `judge_engine.py --half 1` ran on CPU 6 against uvicorn agent.py on CPU 2, instrumented to record every estimator sample, with the baseline on CPU 4.
  - Medians: 13.52 / 14.04 / 13.72 µs vs random / calling station / always pot (load 2.3–2.5). The same at load 12.5–14.9: 15.92 / 16.05 / 15.72 µs.
  - The sample depends on what ran just before it: in-process sims read 13–34 µs depending on the opponent's code, the TestClient 60 µs, and a tight loop 4 µs. So the reference must come from the real server.
  - With these constants the bot reads factor 1.0 (grace 0.75 ms) in real-judge play on this PC.
- **Real-judge play with the final policy** (quiet, load about 2.3):
  - judge-charged bank 0.01 / 0.00 / 0.03 s per half, 0 errors, 0 faults;
  - per-decision p50 / p99 (ms): preflop 0.022 / ≤ 0.057, flop 0.32–0.42 / 0.51–0.76, turn 0.37–0.53 / 0.56–0.82, river 0.34–0.43 / 0.49–0.67; max 1.18 ms.

  At load 12–15 the judge charged 0.08–0.09 s per half.

### First decision vs steady state per street
Cold process, `policy.warmup()`, then 25 calls per street in play order, each with a fresh TimeManager so iters are equal. The first call vs the median of the next 24, in ms:

| Street | load 3.9 | load 15.7 |
|---|---|---|
| Preflop | 0.030–0.031 / 0.004 | 0.031–0.071 / 0.004–0.006 |
| Flop | 0.419–0.423 / 0.245–0.293 | 0.450–0.552 / 0.282–0.318 |
| Turn | 0.307–0.530 / 0.317–0.388 | 0.320–0.550 / 0.325–0.434 |
| River | 0.260–0.320 / 0.284–0.427 | 0.335–0.434 / 0.379–0.436 |

The T10 spikes are gone: the first-np.unique 3.4–4 ms spike and the 9–15 ms preflop table load. What remains is a first-call cold-code overhead of ≤ 0.25 ms (preflop +0.03 ms, flop +0.15–0.25 ms). It is not a lazy cache, and it sits inside the free 1 ms grace, so it is charged 0.

### Fixed per-decision cost outside Monte Carlo (iters = 0, `PolicyBot.choose_action`)
- p50: preflop 0.005–0.007 ms, flop 0.063–0.090, turn 0.097–0.163, river 0.174–0.200 (load 5.6–6.6).
- The PolicyBot wrapper's own overhead is about 2.5–4.7 µs p50. Preflop is the clean measurement; postflop differences of medians are noise-dominated.
- The p99 in those runs reached 0.38–0.49 ms on the river. The bare `compute_features + heuristic_action` had the same p99 (0.31–0.50) in the same runs, so that is machine contention, not the policy. T10 measured a 0.136 ms river p99 when quieter.

### Zoo errors and bank use
- `test_zero_errors_vs_whole_zoo` (slow; 19 bots × 200 deals; agent.py through `load_bot_file`): 19/19 passed. 0 errors, `bank_a < 3.0`, 0 layer faults. 32.5 s at load 15.5–16.6.
- Full 1000-deal survey: `focus_eval(resolve_bot("agent"), ZOO, ["t11a"], 1000, workers=4)`, 47 s, load 10.1 → 7.1:
  - **0 errors in all 19 matches; max bank 0.04 s per half** (mc_equity_fast 0.03, llm_simple_rules 0.04, llm_beginner_2h 0.04, the rest ≤ 0.02);
  - 18 W / 0 L / 1 C, mean +2.778 bb/deal, expected points 0.974;
  - the one chop: llm_ml_minded +0.44 [−0.10, +0.99]. T10 also flagged that margin as thin.

## Files changed
- `bot/policy.py` (new)
- `tests/test_policy.py` (new)
- `agent.py` (warmup area + YOUR CODE section)
- `sim/match.py` (docstring only)

## Self-review findings (fixed before reporting)
- The brief says "blueprint → exploit → heuristic", but exploits need a base action, so they run last. The docstring now says this order is the order of precedence.
- `measure_cpu_factor(runs=0)` returned F_MAX, because an empty min stays inf. It now returns 1.0, with a test, in a separate commit.
- In the `sim/match.py` note I first wrote "pickle (or copy.deepcopy via pickle)". deepcopy passes class objects directly and works, which I verified, so I removed it.
- I checked the brief and every controller addition against the code: R5, R10, R9/T9, R30a, R30c, warmup caches, AUDIT-FIX, Review Focus 1, error budget, the tests, CPU etiquette, the agent.py edit regions, and the T10 carry-forwards (Exception everywhere, nut_flags at iters 0).

## Issues and concerns
1. **The online CPU-factor sample depends on what ran just before it.** The calibration targets real HTTP play, where the bot reads about 1.0 on this PC. But in-process sims read about 1–2.5, depending on the opponent's code in the same process, and the TestClient reads about 4.4.
   - Because the grace uses `max(warmup, online)`, this only shrinks the grace in sims (to as low as ~0.375 ms). The bot then over-counts its own bank there.
   - That is harmless at today's bank use, far below the 3 s target, so pacing is unchanged. T12 should know it when adding `time_scale` (R30b): the bot's own TimeManager count in sims is a little more conservative than in the tournament.
   - If the preflop path changes (T12/T14 code inside `compute_features`/`heuristic_action` preflop), re-measure `PREFLOP_REF_S` with the real judge; the procedure is above and in the comment next to the constant. A cheaper preflop path would make the online factor read low, and then warmup's factor still floors it.
2. **The warmup factor reads 1.17–1.25 on this PC while T17 runs on CPUs 8–15** (grace about 0.70 ms instead of 0.75). That is the correct direction (a busy PC is slower), but sims run alongside T17 see a slightly smaller grace than quiet runs. T12's quiet-machine rerun before shipping (T10 note) covers this.
3. **The exploit layer gets `feats=None` when the base came from the blueprint.** T20 decides whether to compute features for exploits; the Monte Carlo cost sits in that choice.
4. **T14 wiring still to do** (per the T13 note): construct `OpponentModel(raise_cap=config.raise_cap, big_blind=config.big_blind)` in `new_half`, call `observe()` first in `choose_action`, and `finish(result)` in `on_hand_end` (that time is already charged with `add(dt)`).
5. **First-commit message format:** `4636d2c` has the old body + Co-Authored-By format, written before the coordinator's one-sentence rule. Per that instruction it was not rewritten; later commits follow the new rule.
