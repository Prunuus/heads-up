# Task 21 Report: Local Best Response probe (validation)

Status: DONE_WITH_CONCERNS (see Issues and concerns)
Commit: `a07c84c` Add a local best response probe that measures how exploitable a bot is (branch `kevin-lbr`, not pushed)

## What was implemented

`sim/opponents/lbr.py` (new): the LBR probe, its target contract, an evaluation runner, and a CLI.

- **`make_lbr(target, rollouts=500, particles=PARTICLES, seed=SEED) -> BotFactory`.** This is the plan's
  interface plus two keyword arguments. It returns `functools.partial(LBRBot, ...)`, which is picklable (R3).
- **Belief.** Particles are uniform samples of target holes that avoid LBR's hole and the visible board.
  - After each target action, every particle is reweighted by the target's probability of that action
    with that hole.
  - The probability is floored at `TREMBLE = 1e-6`, and particles below `PRUNE = 1e-4` of the heaviest are
    dropped. As a result, an action that no particle explains keeps the particles that explain the most,
    instead of emptying the belief.
  - A board card that a particle holds kills that particle (exact rejection sampling). If none is left, a
    fresh set is drawn and the hand's target actions are replayed on it.
- **Decision (the plan's rule).** LBR plays the action with the highest immediate EV, assuming:
  - the target folds to a raise with its policy's fold probability `fp`, and otherwise calls (a re-raise
    counts as a call);
  - after that, both players check/call to showdown.

  In chips:

  | Action | EV |
  |---|---|
  | fold | `-mine` |
  | check/call | `wp * pot - mine` (after the call) |
  | raise | `fp * theirs + (1 - fp) * (wp' * pot - mine)` (after the target's call) |

  Every amount comes from `Game.clone()/apply()` on LBR's mirror of the hand. The mirror rebuilds the
  engine state from LBR's own observation and is checked against it on every decision (a divergence
  raises).
- **Win probability.** `particle_wins` gives each live particle a win probability: exact on the river, and
  otherwise `ceil(rollouts / live)` random completions per particle. It is computed once per decision.
  `wp` for check/call and `wp'` for each raise are weighted averages of the same samples (common random
  numbers).
- **Target contract** (documented in the module docstring):
  - **BotLike targets: deterministic re-simulation.**
    - LBR keeps a private shadow of the target: `target()`, `new_half` with LBR's config, and
      `on_hand_end` with the target's side of every result (`target_result`).
    - Each particle carries a deep copy of the shadow memory, with the top-level rngs reseeded per
      particle.
    - The judge's handling is mirrored: an exception, a non-action or an illegal action plays
      `default_action`.
    - LBR's candidate raises run on memory copies. The branch that actually happens is reused when the
      target answers, so a particle answers the same way both times.
    - After the hand, a surviving particle's memory becomes the shadow memory.
  - **Distribution targets.**
    - Any bot with `action_probs(obs, memory)` is weighted exactly by those probabilities.
    - `DistributionTarget(probs_fn)` is the adapter: `probs_fn(obs, hole) -> probabilities over
      obs.legal_actions`, where T20 plugs in the blueprint's `probs`. The adapter plays by sampling, with
      an rng kept in memory.
- **Evaluation.** `lbr_half` / `lbr_eval(target, seeds, deals, workers, ...)` plays LBR as agent1 through
  `sim.match.play_half`.
  - Both bots get `LBR_BANK_S = 1e6` s as bank and as per-request timeout. LBR is never cut short, and a
    time-managed target (the heuristic's `TimeManager`) always gets its full budget, so the
    re-simulation matches the target that plays.
  - Results are pooled per deal with `sim.tournament.mean_ci`, exactly like `paired_diff`.
  - Any LBR error raises `RuntimeError`, so a bug cannot silently become a low exploitability number.
  - `LBRResult` carries `per_deal`, so T20 can compare LBR(v3) against LBR(v2) per deal on common seeds.
- **CLI.** `python -m sim.opponents.lbr --target heuristic|agent|<zoo>|<file> --seeds --deals
  --workers --particles --rollouts --out`.

Not touched: `sim/opponents/__init__.py` (LBR is not in `ZOO`), `bot/`, and every organizer file.

## Particle/rollout choice

**Defaults: 1,000 particles (the plan says 2,000) and 500 rollouts (as planned).**

- The same 1,000 deals (seeds `to1..to8` × 125) were played against heuristic v1-untuned at
  three particle counts, on a nearly idle CPU 8–15 set, 8 workers.
- The paired difference compares each deal's value at N particles with its value at 2,000:

| Particles | LBR bb/deal [95% CI] | Paired vs 2,000 [95% CI] | ms/decision | LBR s/deal | Wall, 1,000 deals | Inconsistent target actions |
|---:|---|---|---:|---:|---:|---:|
| 2,000 (plan) | +3.486 [+2.586, +4.387] | n/a | 412 | 2.04 | 265 s | 3/4,245 |
| **1,000** | +3.448 [+2.519, +4.377] | −0.038 [−0.800, +0.723] | 198 | 0.98 | 131 s | 14/4,274 |
| 500 | +2.962 [+2.029, +3.896] | −0.524 [−1.429, +0.381] | 91 | 0.46 | 62 s | 32/4,313 |

- **Why 1,000.** The cost is linear in particles. Against the plan's setting, LBR's value moved
  by −0.04 bb/deal, which is not measurable. 500 hints at a weaker LBR (−0.52). The saved
  budget buys deals, which is what narrows the CI: the per-deal SD is ~14.5 bb.
- **Why the plan's 500 rollouts stay.** Win probabilities cost a few ms per decision
  (vectorized `fasteval`), against 100–400 ms of target re-simulation. There is no cost
  reason to cut them.
- **The brief's "hours per match" estimate did not hold.**
  - A postflop heuristic decision costs ~0.3–1 ms, but a preflop one only ~5 µs, plus ~22 µs
    for the memory copy.
  - Only live particles are re-simulated: 768 of 2,000 on average (filtered by board cards
    and the target's actions).
  - The target's answer to LBR's raise reuses that branch.
  - Measured: one 1,000-deal match costs 2,036 s of LBR CPU at 2,000 particles and 984 s at
    1,000.
- **Bigger accuracy fix: common random numbers.** The first version gave each candidate action
  its own independent 500-completion `wp`. A raise that nobody folds to then got a noisily
  different `wp` than the call (±0.03–0.04, seen in a decision trace), which flipped close
  choices. LBR vs always_pot read +1.7 [−1.0, +4.4] (200 deals). Sharing one per-particle
  estimate across actions gave +5.04 [+3.75, +6.33] (1,000 deals, a superset). TDD:
  `test_particle_wins_share_samples_across_weightings` and
  `test_a_raise_nobody_folds_to_is_valued_with_the_calls_win_probability`.

## Measured numbers (defaults: 1,000 particles, 500 rollouts)

| Target | LBR bb/deal | 95% CI | Deals | Seeds × deals |
|---|---:|---|---:|---|
| heuristic v1-untuned (`HeuristicBot`, defaults) | **+3.891** | [+3.445, +4.338] | 4,000 | `lbr1–16` × 250 |
| always_pot | +6.085 | [+5.181, +6.989] | 2,000 | `lbr1–20` × 100 |
| calling_station | +13.140 | [+12.387, +13.893] | 2,000 | `lbr1–20` × 100 |

- **Cost per decision** (nearly idle cores):
  - 198 ms against the heuristic (trade-off run, 1,000 particles);
  - 8–10 ms against the trivial baselines (2,000 particles), and 5.9 ms in a 200-particle CLI
    smoke run.
- **Cost per evaluation:** 0.98 s of LBR CPU per deal against the heuristic, i.e. 4,000 deals in
  ~9 min on 8 free workers. The headline run took **1,211 s wall (20 min)** on 8 workers while sharing CPUs 8–15
  with Track B's 8 `gen_deals` workers: 19,770 decisions, 9,403 s of request wall time,
  ~3,900 CPU-s. Its belief stats: mean 385 live particles, 35/17,242 inconsistent target
  actions, 1 rebuild, 21.5M target re-simulations, 0 target errors.
- **Wall-clock caveat:** the `ms_per_decision` printed by the CLI is the wall time of LBR's
  requests, so it inflates under contention. The baselines run reported 17–21 ms per decision
  while sharing cores with Track B's 8 `gen_deals` workers.

## TDD evidence

RED 1 (`.venv/bin/python -m pytest tests/test_lbr.py -q -m "slow or not slow"`, before
`sim/opponents/lbr.py` existed, with the plan's `test_lbr_beats_always_pot_and_calling_station`
in the file):
```
tests/test_lbr.py:15: in <module>
    from sim.opponents.lbr import (DistributionTarget, LBRBot, action_values, lbr_eval,
E   ModuleNotFoundError: No module named 'sim.opponents.lbr'
ERROR tests/test_lbr.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.23s
```
Expected: the module did not exist yet.

RED 2 (the common-random-numbers change; `.venv/bin/python -m pytest tests/test_lbr.py -q`):
```
E   ImportError: cannot import name 'particle_wins' from 'sim.opponents.lbr'
1 error in 0.13s
```
Expected: `particle_wins` and `last_decision["wp_after"]` did not exist yet.

GREEN:
```
$ .venv/bin/python -m pytest tests/test_lbr.py -q        # final state, at commit a07c84c
18 passed, 1 deselected in 1.19s
$ taskset -c 8-15 nice -n 19 .venv/bin/python -m pytest tests/test_lbr.py -q -m slow   # at a07c84c
1 passed, 18 deselected in 39.55s
```

The slow test is deterministic (fixed seeds and LBR seed). Its margins in that exact
configuration (`lbr1, lbr2` × 200 = 400 deals, default settings):
- always_pot: +6.375 [+4.376, +8.374]
- calling_station: +13.300 [+11.611, +14.989]

Full default suite (`taskset -c 8-15 .venv/bin/python -m pytest -q`):
```
225 passed, 9 deselected, 1 warning in 57.98s
```
The warning is the pre-existing starlette/httpx deprecation from `fastapi.testclient`.

## Tests (`tests/test_lbr.py`)

Fast (default suite, ~1 s):
- `win_prob` is exact on the river against weighted particles, and matches exact enumeration
  on the turn.
- `particle_wins` shares samples across weightings.
- `action_values` follows the check/call-afterwards rule, checked by hand on two spots
  (unopened, and facing a pot raise).
- `target_result` mirrors LBR's HandResult.
- Deterministic target (`AceRaiser`): after its preflop raise or limp, every live particle
  agrees with the action, and none overlaps LBR's hole.
- Distribution target: the weight ratio of ace and non-ace holes equals 0.9/0.1, exactly.
- Stochastic target (`CoinFolder`, rng in memory): the fold probability averages to ½ over
  particles.
- Branch reuse: after the target answers LBR's raise, exactly the predicted callers (minus
  flop conflicts) survive.
- A raise that nobody folds to is valued with exactly the call's `wp` (common samples).
- When the board kills every particle, the belief is redrawn (avoiding every known card) and
  the target's preflop action is replayed on it.
- An action that no particle explains keeps the belief (tremble floor), and the next action
  filters it again.
- Against a target that always folds to bets, LBR scores exactly +1.5 bb/deal: the blinds on
  every deal, which tests the raise EV and the paired accounting.
- LBR plays the real heuristic without errors.
- `lbr_eval` raises when LBR errs.
- Factories are picklable.
- `DistributionTarget` plays legally by sampling.

Slow (`-m slow`): `test_lbr_beats_always_pot_and_calling_station`, which requires a CI win
against both on 400 deals each.

## Files changed

- `sim/opponents/lbr.py` (new)
- `tests/test_lbr.py` (new)
- `docs/results/lbr.md` (new; per the ruling, not `docs/results/v3.md`)

## Self-review findings (fixed before commit)

- **Noisy per-action `wp`.** This was found in a decision trace against always_pot and fixed with
  common random numbers (see above). It was the largest accuracy issue.
- **Missing coverage.** The rebuild path (the board kills every particle) and the tremble floor
  (no particle explains an action) had no test. I added
  `test_belief_is_redrawn_and_replayed_when_the_board_kills_every_particle` and
  `test_an_action_no_particle_explains_keeps_the_belief`. A mutation check confirmed each fails
  when its mechanism is disabled (`w * p` without the floor; the rebuild guard disabled).
- **Test-file tidy-ups.** I removed an unused import, fixed a wrong helper docstring, and
  simplified `ace_probs`.
- **Unreported counter.** `target_calls` (re-simulations, the cost driver) was counted but not
  reported. It is now in the belief stats and the CLI.
- **Checked, no change needed:**
  - The plan's interface is kept: `make_lbr(target, rollouts=500)`, with keyword extras only.
  - LBR is not in `ZOO`. `sim/opponents/__init__.py`, `sim/match.py`, `bot/` and the organizer
    files are unchanged.
  - Every heavy run was pinned and niced (R32). Factories are picklable (R3). The slow test is
    marked (R11).

## Issues and concerns

- **LBR's myopia is inherent to the plan's rule (check/call afterwards).**
  - Its numbers are lower bounds. Against aggressive targets they are weaker: vs always_pot,
    LBR calls pot bets with 1/3 < wp < 1/2 and then faces more bets.
  - Gate (d) compares two lower bounds. That is the plan's design; it is worth remembering when
    the values are close.
- **Costs scale with the target's per-decision cost.** The heuristic costs about 0.3–1 ms
  postflop. A slower v2/v3 (exploit layers, blueprint bucketing with MC) makes LBR
  proportionally slower. Re-measure ms/decision on the first T20 run and lower `--particles`
  or `--deals` if needed. The trade-off table above says what 500 particles cost in accuracy.
- **BotLike contract.** Per-half state must live in `memory`, and the rngs must be stored at the
  top level of `memory`. A T11 `PolicyBot` that keeps decision-relevant state on the instance or
  in module globals would be modelled approximately. The inconsistent count printed by the CLI
  is the tell: it was 14/4,274 for the heuristic at 1,000 particles, and a broken model would
  show many.
- **Targets with on_hand_end-driven opponent models.** These are tracked through the shadow
  memory. If the target's last action ends the hand after LBR's last decision, it is not
  replayed into the shadow; the hand result still is.
- **Unlimited bank for the target.** An LBR match gives both bots `LBR_BANK_S`. The target
  plays its full-budget strategy, which may differ slightly from its tournament-bank strategy.
  This is consistent across targets.
- **Only wall time depends on CPU contention, not the LBR value.**
  - With the unlimited bank, the target's rollout budget does not depend on time.
  - LBR's randomness is seeded per hand, so a run is deterministic for fixed seeds: per-deal values and stats were identical between a 2-worker and an
    in-process run (heuristic, seeds `det1, det2` × 10, 150 particles).
  - The headline run shared CPUs 8–15 with Track B's 10M-deal `gen_deals` job (8 workers,
    nice 19), which roughly doubled its wall time.

---

## Fix round 1: the probe now fails closed when its model of the target breaks

Status: DONE. Commit `20bd2dd` "Make the local best response probe fail closed when its model of the target breaks", on `kevin-lbr`, on top of the controller's `ba6f886`. Not pushed.

### What changed (`sim/opponents/lbr.py`)

- **Re-simulation failures are counted, not swallowed.**
  - `_judged` now returns `(action, ok)`. The judge's handling is unchanged (an exception, a
    non-action or an illegal answer plays `default_action`), but `ok` is False in those cases.
  - `_Resim` counts them in `errors`.
  - Both models count the shadow's `on_hand_end` exceptions in `hand_end_errors`; before, these
    were dropped.
  - `lbr_half` merges both into the stats as `model_errors` and `shadow_errors`, next to
    `target_calls`.
- **`lbr_eval` fails closed.** `model_problems(stats, target_errors, max_inconsistent)` is a pure
  function, and `lbr_eval` raises `ModelCheckError` (a `RuntimeError` subclass) when it returns
  anything. The error carries `.result` and `.problems`, and survives pickling. It lists a
  problem when:
  - `model_errors > 0` or `shadow_errors > 0` while the real target had no errors
    (`errors_b == 0`);
  - `inconsistent > max_inconsistent × target_actions + 2`, with default `MAX_INCONSISTENT = 0.02`.
    The `+ 2` slack keeps one or two misses in a short run from failing it. The heuristic
    measured 0.2%.

  `LBRResult.model_problems` lists the problems either way. `check_model=False` is the explicit
  opt-out, and `max_inconsistent=None` drops only the rate limit.
- **The CLI.** New flags are `--max-inconsistent` and `--no-model-check`. It prints a
  `model: re-simulation errors … shadow on_hand_end errors … problems …` line and one
  `MODEL PROBLEM:` line per problem, writes `model_problems` to the `--out` JSON, and exits
  with **code 2** when the check fails.
- **The `_Dist` gap is closed structurally.** Only a `DistributionTarget` instance (the adapter)
  is weighted by its probabilities: `isinstance` replaces the old duck-typed `action_probs`.
  The adapter's `choose_action` samples from its own `action_probs`, i.e. from the pure
  `probs_fn`, so the two cannot disagree. Any other bot, one with its own `action_probs`
  included, is now re-simulated and so covered by the two checks above.
- **Docs.** The module docstring documents the checks and the opt-out; `docs/results/lbr.md`
  has a short note.

### New tests (`tests/test_lbr.py`)

- `test_lbr_eval_fails_closed_when_the_re_simulated_target_raises`. This is the reviewer's
  case: `CrashesOffDeal` checks/calls with the holes the deal gave it and raises for any other
  hole, so only re-simulation ever sees the exception. Expected: `ModelCheckError` with
  "re-simulated", `target_errors == 0`, `model_errors > 0`, and the error pickles.
  `check_model=False` returns the result with `model_problems` listed.
- `test_lbr_eval_fails_closed_when_re_simulation_disagrees_with_the_target`. `RaisesOnDeal`
  never errs, but it raises only with its dealt holes. Expected: `ModelCheckError` with
  "no particle", and 0 model errors.
- `test_cli_exits_non_zero_when_the_model_check_fails`: exit code 2 and `MODEL PROBLEM` output;
  0 with `--no-model-check`.
- `test_model_problems_rules`: the pure rules, i.e. the target-errs exemption, the 2% + 2
  limit, `None`, and the short-run slack.
- `test_shadow_hand_end_errors_are_counted`.
- `test_only_the_adapter_is_weighted_by_its_probabilities`: a non-adapter bot with
  `action_probs` gets `_Resim`.
- `test_with_hole_is_dataclasses_replace`: guards the `_with_hole` fast path against the
  regression the reviewer named.

### RED evidence

Before the fix, `.venv/bin/python -m pytest tests/test_lbr.py -q` gave:
```
E   ImportError: cannot import name 'ModelCheckError' from 'sim.opponents.lbr' (/home/kchen/code/heads-up-bundle-lbr/sim/opponents/lbr.py)
1 error in 0.32s
```

The fail-open behaviour itself, on the unfixed code (the same two targets, defined inline,
`lbr_eval(cls, seeds=["fc"], deals=12, particles=30)`):
```
CrashesOffDeal returned silently: -1.33 bb/deal, target_errors=0, inconsistent=84/120
RaisesOnDeal returned silently: +7.75 bb/deal, target_errors=0, inconsistent=57/57
```

### GREEN, covering tests

```
$ .venv/bin/python -m pytest tests/test_lbr.py -q
25 passed, 1 deselected in 3.56s
$ taskset -c 8-15 nice -n 19 .venv/bin/python -m pytest tests/test_lbr.py -q -m slow
1 passed, 25 deselected in 92.69s (0:01:32)
```

The slow run confirms the plan's baseline test passes through the new check. A mutation check,
reverted afterwards and confirmed byte-identical with `cmp`:
- with the `ModelCheckError` raise disabled, 3 tests fail (both fail-closed tests and the CLI test);
- with `_judged` reporting ok for everything, the re-simulation-raises test fails.

### Determinism evidence (the controller's request)

`lbr_eval(HeuristicBot, seeds=["det"], deals=50, workers=2)` run twice, at default settings
(1,000 particles, 500 rollouts):
```
$ taskset -c 8-15 nice -n 19 .venv/bin/python det2.py   # lbr_eval(HeuristicBot, seeds=["det"], deals=50, workers=2), twice
run 1: +2.7000 bb/deal [-1.978, +7.378] deals 50 per_deal[:8] [3.0, -15.0, -30.0, 3.0, 49.0, 17.0, 0.0, 14.0] sum +135.0
       model_errors 0 shadow_errors 0 inconsistent 3/230 target_errors 0 model_problems []
run 2: +2.7000 bb/deal [-1.978, +7.378] deals 50 per_deal[:8] [3.0, -15.0, -30.0, 3.0, 49.0, 17.0, 0.0, 14.0] sum +135.0
       model_errors 0 shadow_errors 0 inconsistent 3/230 target_errors 0 model_problems []
per_deal identical across the two runs: True
stats identical: True
```

The script is
`/tmp/claude-1000/-home-kchen-code-heads-up-bundle/c0d55f53-d291-4faf-88d2-40523d153176/scratchpad/det2.py`.
The heuristic passes the new check with 0 re-simulation errors.

The healthy CLI path, `taskset -c 8-15 nice -n 19 .venv/bin/python -m sim.opponents.lbr --target always_pot --seeds 2 --deals 15 --particles 200`:
```
model: re-simulation errors 0  shadow on_hand_end errors 0  problems 0
exit code 0
```

### Notes

- The headline numbers in `docs/results/lbr.md` were measured before this fix. Its value path
  is unchanged:
  - `_judged` returns the same actions, and the counters only observe;
  - the heuristic and the baselines are not `DistributionTarget`s, so they were already
    re-simulated;
  - their inconsistency (35/17,242 for the headline run) is far under the limit.
- **Known residual.** A re-simulation that diverges systematically, but still leaves some
  particle explaining most actions, raises the inconsistency rate only partly. The 2% limit
  catches gross divergence such as the two new test targets; subtle divergence is bounded only
  by how far the rate rises. For T20, compare the CLI's "inconsistent" line with the
  heuristic's 0.2%.
- The 7 Minor findings were not touched, as the controller instructed.
