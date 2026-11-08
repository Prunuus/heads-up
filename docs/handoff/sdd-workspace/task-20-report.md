# Task 20 report: the CFR blueprint in the bot (v3) and the v3 gate

Status: DONE_WITH_CONCERNS (see Concerns). Branch kevin, on top of d0b35d5.

Commits (not pushed):
- c6b971a Integrate the CFR blueprint as the base layer of the policy with heuristic fallbacks.
- fea1920 Add the resumable v3 gate script.
- 94076b5 Default the v3 gate's LBR section to sixteen seeds like Task 21.
- 9c33d51 Add the Task 20 blueprint integration and provisional v3 results to the v3 results.
- e85525a Record the v3 zero-error zoo run in the v3 results.
`data/blueprint.npz` is NOT committed (dev export only, per the brief).

## What I built

### The blueprint layer (`bot/policy.py`)

v3 = `PolicyBot()` with a blueprint loaded. The layers are: blueprint base, then the exploit
layer on top. The heuristic (v2's base) answers whenever the blueprint can't. v2 is
`PolicyBot(use_blueprint=False)`, decision for decision: the heuristic path is unchanged.

- **Warmup** (never raises; each step is in `WARMUP.failed` if it fails):
  - `tree` = `get_tree()`;
  - `abstraction` = `abstraction.warmup()`;
  - `blueprint` = `_load_blueprint()`, which reads `policy.BLUEPRINT_PATH` (default
    `data/blueprint.npz`):
    - no file means `None` and v2 behaviour (not a failure);
    - a corrupt or stale file (`Blueprint.load` raises `ValueError`) is a failed step and
      gives `None`.
  - `blueprint_selftest`: every spot of the fixed warmup hand (all four streets, facing a
    bet) must map through `node_of`, get a bucket, and get a legal answer from `act`.
    Otherwise the blueprint is dropped. This also warms every lazy cache of the blueprint
    path, so the cold-start first-decision test passes with the blueprint present.
  - The blueprint is loaded only if both the tree and the abstraction steps succeeded.
  - `Warmup` gained a `tree` field (default None).
- **Per decision** (`_blueprint_action`):
  1. Skip it if the street is not in `blueprint_streets`.
  2. `node_of(obs, tree)`. On a miss, count it in `memory["layers"]["node_miss"]` and return
     None, so the heuristic answers. A miss is not a fault.
  3. Bucket: `abstraction.bucket(street, hole, board, iters)`, where iters comes from the
     TimeManager as the plan says. The bucket ignores it because it is deterministic (T16).
     It is cached in `memory["bucket"]` as `((street, hole, board), bucket)`. Keying on the
     hole too means LBR's per-particle re-simulations, or a hypothetical hole, never reuse a
     wrong bucket.
  4. `Blueprint.act(node, bucket, obs.legal_actions, memory["rng"], self.purify)`. The rng is
     the per-half `memory["rng"]` (LBR reseeds it per particle).
- **Fallbacks**, each to the heuristic:
  - node miss;
  - `config_ok` False: `config_matches(config, blueprint.meta)` compares stack, sb, bb and
    cap. This was already wired, and it is now tested with a retuned raise cap, retuned
    blinds, a retuned stack, and a blueprint whose meta differs;
  - no blueprint;
  - any exception (including from the bucket), an illegal action or a non-Action answer.
    These count `faults["blueprint"]`, the same way the other layers count theirs.
- **Exploits on a blueprint base.** This is the design choice the brief asked me to justify.
  - check-folder (RAISE_HALF when legal) and I6 (which computes its own nut flag) need no
    Features. They apply exactly as on the heuristic base.
  - The maniac and station modes re-run the heuristic with shifted params, so they need
    Features. `_blueprint_extras` computes Features (`compute_features` with the
    TimeManager's iters, as the heuristic would) only while `exploit.active_mode(opp)` is
    maniac or station. The common blueprint decision therefore pays only node_of, the
    bucket and act.
  - Why: those two modes are v2's largest validated gains (+2.9 / +3.9 / +0.8 bb/deal vs
    always_half / maniac / calling_station, docs/results/v2.md). The re-run's guard ("never
    fold a hand the base plays on") now protects the blueprint's action. In those modes v3
    plays v2's exploit, never tighter than the blueprint.
  - The knob `exploit_features=False` turns those modes off on a blueprint base (gate label
    `v3_nofeat`).
- **CPU sample (T11 M2).** Every preflop decision runs the preflop heuristic layer
  (`_heuristic_action`), whichever layer answered. With a blueprint base its action is
  discarded and only its Features are kept. Preflop it draws no random numbers (checked:
  `compute_features` with an empty board and `heuristic_action` preflop never touch the
  rng). So the sample is the same fixed workload as the dev-PC reference
  (`PREFLOP_REF_S`), at a cost of about 14 µs per preflop decision.
- **Counters.** `memory["layers"]` = {blueprint, heuristic, node_miss} counts which base
  layer answered.
- **Knobs.** All are constructor arguments, so `functools.partial` factories pickle:
  `purify` (default `PURIFY = 0.1`), `blueprint_streets` (default all streets), and
  `exploit_features` (default True).
- **No bank-critical skip.** I did not add "skip the blueprint when the bank is critical".
  The blueprint path costs about the same as the heuristic at iters 0 and has no Monte
  Carlo, so skipping it would save nothing.
- **Pinning tests and the tuner to v2.** The heuristic-layer tests in `tests/test_policy.py`
  (via `_fresh_bot`), the v1/v2 sim tests in `tests/test_exploit.py`, and `sim/tune.py`
  now pin `use_blueprint=False`. Otherwise, once the blueprint file is committed, they
  would silently test or tune the blueprint instead of the heuristic.

### The gate script (`sim/gate_v3.py`)

- **Bots.**
  - v3 = `agent.py` exactly as shipped (`bot_file_factory`, fresh package per half).
  - v2 = `PolicyBot(use_blueprint=False)`.
  - Purify variants are `v3_p<p>` = `PolicyBot(purify=p)`.
  - Tuning variants: `v3_post` (blueprint postflop only) and `v3_nofeat`.
- **Sections:**
  - (a) h2h v3 vs v2, 5 seeds × 1000, pooled per-deal CI over the duplicate halves.
  - (b) HOLDOUT + FIELD for v3 and v2 at scales 1.0 and 2.5 (3 seeds × 1000 by default):
    expected points, bb/deal, paired v3 − v2 CI per split, and a per-opponent line with the
    opponent's max bank per half.
  - (c) errors, forfeits, our max bank per half (per scale) and the longest request, over
    every match of (a) and (b).
  - (d) LBR(v3) and LBR(v2) at 2000 particles, 16 seeds × 250 (T21's protocol), plus the
    paired per-deal difference.
    - LBR runs in seed groups of `--lbr-workers/2` seeds, each cached.
    - `model_problems` is re-checked on the pooled stats of the whole run: the same check
      as `lbr_eval`'s ModelCheckError, with MAX_INCONSISTENT.
    - A failed check prints "MODEL CHECK FAILED: ..." next to the value and the script
      exits 2.
  - (e) purify sweep {0, 0.05, 0.1, 0.2} vs v2 on TRAIN, 2 seeds × 1000. v2's matches are
    played once and shared. The CI is paired_diff's (per-deal A - B differences of every (opponent, seed) pooled into mean_ci), computed from cached matches so v2 is not replayed per purify value. `--extra v3_post v3_nofeat` adds the variants.
- **Resumable.** Each match is written atomically as one JSON file under `--out` as soon as
  it finishes. A rerun skips cached jobs, and `--summary-only` reprints the table from the
  cache. The summary also goes to `<out>/summary.txt`.
- **One blueprint per cache.** Job keys don't name the blueprint, so `<out>/blueprint.json`
  records its identity (iterations, seed, buckets sha). A run with a different
  `data/blueprint.npz` is refused (exit 3), so results from two blueprints never mix.
- `--cpus 0-7` sets the affinity of the script and its spawned workers (R39). The load
  average is printed with every job.
- `tests/test_gate_v3.py` (4 tests, about 14 s):
  - the labels resolve and pickle;
  - a tiny a+e run caches, resumes with nothing re-run, and prints (a), (c) and (e);
  - a tiny LBR run is chunked, cached and resumed;
  - the cache refuses another blueprint.

## TDD evidence

RED, the plan's test (Step 1), before any implementation:

```
$ .venv/bin/python -m pytest -q tests/test_policy_blueprint.py -x -k test_blueprint_layer_used_when_config_ok
E       TypeError: PolicyBot.__init__() got an unexpected keyword argument 'purify'
1 failed, 24 deselected in 1.21s
```

Behavioural RED on the same HEAD (a fake blueprint without the new kwarg, config ok):

```
config_ok True blueprint calls 0 of 233 | layers key present: False
```

GREEN after the implementation:

```
$ .venv/bin/python -m pytest -q tests/test_policy_blueprint.py
25 passed in 3.34s
```

Related suites, with the blueprint file present:

```
$ pytest -q tests/test_policy.py tests/test_exploit.py tests/test_policy_blueprint.py tests/test_tune.py tests/test_strategy.py tests/test_lbr.py
153 passed, 25 deselected in 22.17s
```

The same suites with `data/blueprint.npz` moved away (heuristic-only state):

```
$ pytest -q tests/test_policy.py tests/test_exploit.py tests/test_policy_blueprint.py test_agent.py
90 passed, 24 deselected in 53.01s
```

Full default suite, blueprint present, trainer on all CPUs (load about 15), nice 19:

```
6 failed, 642 passed, 46 deselected, 2 xfailed in 207.64s
```

- One of the six was my gate test: its `*.json` glob also counted `blueprint.json`. I
  fixed it, and `tests/test_gate_v3.py` passes 4/4.
- Four were `test_judge_fidelity` timeout/bank-out tests (20 ms action timeouts) and one
  was `test_sim_match::test_grace_is_free_time_in_every_request`. These are environment
  timing flakes.
  - Rerun alone without nice: the grace test and both bank-out tests pass. The two
    `test_timeouts_match_the_judge` tests still fail under this load.
  - They fail the same way at HEAD d0b35d5. I checked in a temporary worktree:
    `1 failed, 1 passed`, with the judge-side calling_station also timing out at 20 ms.
  - None of these tests touches bot/.
- Per R42 they need a rerun on quiet cores.

LBR smoke against v3 (`--target agent --seeds 1 --deals 20 --particles 300`):

```
model: re-simulation errors 0, shadow on_hand_end errors 0, problems 0
inconsistent 1/64
```

## Per-decision timing

In-process, 300-deal halves, trainer on all 16 threads (load 10-16), blueprint at 1.56B
iterations, so these are upper bounds:

| Bot vs | median | p99 | max | bank/half |
|---|---:|---:|---:|---:|
| v3 vs calling_station (station re-run active) | 0.673 ms | 3.52 ms | 16.0 ms | 0.233 s |
| v3 vs llm_gto_inspired (no mode) | 0.240 ms | 1.78 ms | 11.3 ms | 0.041 s |
| v3 vs maniac (maniac re-run active) | 0.448 ms | 1.73 ms | 10.8 ms | 0.140 s |
| v2 vs calling_station / gto_inspired / maniac | 0.37 / 0.38 / 0.36 ms | 1.7 / 1.7 / 1.2 ms | 7-9 ms | 0.03-0.05 s |

- **Per street** (v3, no mode), median: preflop 0.11 ms, flop 0.68, turn 0.56, river
  0.38 ms. The flop bucket is the largest fixed cost.
- **Medians are within the 0.7 ms target even under load.** p99 is ≤ 3.5 ms, well under 10 ms.
- **Warmup** took 0.89 s under load: the tree is 0.84 s of it (0.36 s quiet), and the
  blueprint load is 0.04 s.
- **Full GC** with agent.py loaded: 24-30 ms.
- **Isolated 150-260 ms requests.**
  - Sim matches showed single requests of 150-260 ms for both v2 (213 ms) and v3 (257 ms)
    at load 12-16 with nice 19. The same v3 halves in the later run read 37 ms max.
  - I read these as preemption by the nice-0 trainer, not compute.
  - (c) on quiet cores will show the real number.

## Provisional sanity numbers

These are informational only. The machine was loaded (R39 does not apply), there were few
seeds, and the blueprints were checkpoints.

Blueprint checkpoint at 1.555B iterations:

- **(a) h2h v3 vs v2, 5 seeds × 1000:** **+0.307 bb/deal [+0.002, +0.612]**, verdicts
  CHOP CHOP A CHOP CHOP. With 2 seeds × 500 it was +0.325 [-0.368, +1.017].
- **(e) TRAIN, 1 seed × 500, paired v3 − v2:**

  | purify | v3 − v2 [95% CI] | points (v2 0.967) |
  |---|---|---:|
  | 0 | -0.521 [-0.813, -0.229] | 0.900 |
  | 0.05 | -0.377 [-0.670, -0.085] | 0.967 |
  | 0.1 | -0.456 [-0.749, -0.163] | 0.900 |
  | 0.2 | -0.645 [-0.936, -0.353] | 1.000 |

  0 errors. v3 trails mainly vs passive and over-folding bots: mc_equity_fast -2.5,
  checkfold_after_300 -1.2, llm_beginner_2h -1.0, llm_simple_rules -1.2, nit -0.6
  bb/deal. It leads vs maniac, random and llm_ml_minded. The cause, from a 500-deal
  diagnostic vs nit:
  - Postflop facing a bet, v2 folds 100% and v3 calls 33% on the flop, paying off the
    nit's value bets.
  - Preflop facing a raise, v3 folds 12-16% and v2 folds 0-2%.
  - Equilibrium defence is the wrong play against these opponents. Also, TRAIN is
    in-sample for v2's heuristic and exploit design (Tasks 10-14 were developed against
    TRAIN), so this comparison is biased toward v2.
- **(b) HOLDOUT + FIELD, 1 seed × 500, scale 1.0:**

  | split | v3 pts | v2 pts | v3 bb | v2 bb | v3 − v2 [95% CI] |
  |---|---:|---:|---:|---:|---|
  | HOLDOUT | 0.875 | 1.000 | +2.10 | +2.57 | -0.465 [-1.041, +0.111] |
  | FIELD | 0.500 | 0.333 | +0.38 | -0.25 | +0.626 [-0.277, +1.529] |

  Per FIELD bot, v3 vs v2 (bb/deal): field_gto +0.55 vs -1.47; field_equity +0.74 vs
  -0.69; field_exploit -0.16 vs +1.42.
- **(c)** 0 errors, 0 forfeits, max bank per half 0.78 s.

Blueprint checkpoint at 4.0B iterations (export of `blueprint_raw.ckpt-004000000000.npz`):

- (a) h2h v3 vs v2, 5 seeds x 1000: **+0.437 bb/deal [+0.126, +0.747]**, verdicts A CHOP CHOP
  CHOP CHOP (per seed +0.98, +0.34, +0.72, +0.36, -0.22).
- (b) 1 seed x 1000, scale 1.0:
  - FIELD: v3 **1.000** pts vs v2 0.333; v3 - v2 **+1.137 [+0.526, +1.748]**. v3 wins all
    three FIELD bots (bb/deal v3 vs v2):
    - field_equity: +0.97 vs -1.22 (v2 lost);
    - field_gto: +0.96 vs -0.99 (v2 lost);
    - field_exploit: +1.21 vs +1.94.
    Opponents' max bank per half was 0.33-0.93 s.
  - HOLDOUT: 1.000 vs 1.000 pts; v3 - v2 -0.623 [-1.075, -0.172]. v3 wins every match by
    less: llm_aggressive +3.96 vs +4.86, threshold_cat +1.61 vs +2.53, llm_time_careful
    +0.75 vs +1.40, llm_opp_model +1.02 vs +1.03.
- (e) TRAIN, 1 seed x 500:
  - purify 0.1: -0.338 [-0.640, -0.035], points 1.000 (v2 0.967);
  - v3_post (blueprint postflop only): -0.380 [-0.661, -0.100], points 0.933. Postflop-only
    is no better.
  - Per opponent (v2 / v3): mc_equity_fast +3.26 / +1.00, llm_beginner_2h +3.11 / +1.33,
    llm_simple_rules +2.38 / +0.99, checkfold_after_300 +2.40 / +1.23. v3 leads vs random
    +4.03 / +5.60 and llm_ml_minded +0.34 / +1.83.
- (c) 0 errors, 0 forfeits, max bank per half 1.24 s (loaded), longest request 273 ms
  (preemption: see Timing).
- v3 vs the whole zoo (`pytest -m slow tests/test_policy.py::test_zero_errors_vs_whole_zoo`, agent.py with the 4.0B blueprint, 200 deals per bot): **22 passed in 61.87 s**, i.e. 0 errors, no forfeit, bank < 3.0 s, 0 layer faults, and the blueprint answered in every half.

**Reading.** The later checkpoint is better everywhere it was measured: h2h +0.31 to +0.44,
FIELD +0.63 to +1.14, TRAIN -0.46 to -0.34. v3 already beats v2 head-to-head and on the FIELD
proxies, which is the user's target. It gives up bb/deal (not points) against weak, passive
or over-folding bots, where v2's in-sample heuristic and exploits extract more. The purify
sweep at 1 seed x 500 is all noise-level (-0.38 to -0.65, overlapping CIs), so I left
`PURIFY = 0.1`. The full sweep (2 seeds x 1000) on the final blueprint decides.

## How to run the gate

After the final export into `data/blueprint.npz` (and only then, because the cache is per
blueprint):

```
# (a) (b) (c) (e): quiet protocol R39, CPUs 0-7, 2 workers, nothing else on 0-7
.venv/bin/python -m sim.gate_v3 --out train_data/gate_v3 --cpus 0-7 --workers 2 --sections abce
# (d) LBR has no clock: run it on the other physical cores (8-15 = cores 4-7), in parallel
.venv/bin/python -m sim.gate_v3 --out train_data/gate_v3 --cpus 8-15 --lbr-workers 8 --sections d
# one table from the cache
.venv/bin/python -m sim.gate_v3 --out train_data/gate_v3 --summary-only
```

- Optional tuning rows: add `--extra v3_post v3_nofeat` to the `abce` run. `--extra` only
  adds jobs to (e).
- Kill and rerun at any time to resume.
- Exit codes: 2 means an LBR model check failed (value unreliable); 3 means `--out` holds
  another blueprint's results.

Rough cost (estimates):

| Sections | Matches or deals | Expected time |
|---|---|---|
| (a) | 5 matches | a few minutes |
| (b) | 84 matches (FIELD bots spend their bank) | probably 20-40 min on 2 workers |
| (e) | 150 matches | about 15-25 min |
| (d) | 2 targets × 4000 deals at 2000 particles | about 35 min on 8 workers (T21: about 1 s of LBR per deal at 1000 particles) |

## Files changed

- `bot/policy.py`: the blueprint layer, warmup loading and self-test, the CPU sample
  regardless of layer, the knobs, `memory["layers"]` and `memory["bucket"]`, and the
  docstrings.
- `tests/test_policy_blueprint.py` (new, 25 tests).
- `tests/test_policy.py`: heuristic-layer tests pin `use_blueprint=False`; the warmup test
  accepts either state of the file; the slow zoo test also asserts that the blueprint
  answered.
- `tests/test_exploit.py`: V2/V1 and the recorder pin `use_blueprint=False`.
- `sim/tune.py`: the tuner pins `use_blueprint=False`.
- `sim/gate_v3.py` (new) and `tests/test_gate_v3.py` (new).
- `docs/results/v3.md`: the Task 20 section.
- Not committed: `data/blueprint.npz`, a dev export of the 4.0B checkpoint, 51 MB.

## Concerns

1. **Every strength number is provisional.** They were measured on checkpoints (1.56B and
   4.0B iterations), under heavy load and nice 19, with few seeds. R39 says loaded runs
   inflate edges vs time-adaptive bots by 1-3 bb/deal. Only the quiet gate on the final
   blueprint counts.
2. **v3 trails v2 in bb/deal vs HOLDOUT** (-0.62 [-1.08, -0.17] at 4.0B, 1 seed) and vs
   TRAIN (-0.34), while matching or beating it on points.
   - R44's "clearly lower expected points vs held-out + FIELD" did not trigger: HOLDOUT
     points were equal and FIELD points better.
   - If the quiet gate shows a HOLDOUT points loss, the tuning lead is exploit gating vs
     passive or over-folding opponents. Example: fall back to the heuristic base when the
     opponent model shows a high fold-to-bet, or an early check-folder trigger.
   - The current knobs did not help on TRAIN: postflop-only was -0.38, and purify 0 to 0.2
     was noise. I did not build a new detector: it is outside the brief and would need its
     own validation.
3. **Environment timing failures in the default suite.** The judge-fidelity timeout tests
   (20 ms action timeouts) and the sim grace test fail under the trainer's load. They also
   fail at HEAD d0b35d5 (checked in a temp worktree) and do not touch bot/. R42 (1) needs
   them rerun alone on quiet cores.
4. **Isolated 150-270 ms single requests** appeared in loaded sims for both v2 and v3. I
   attribute them to preemption (a full GC is about 30 ms). Section (c) on quiet cores, and
   the real-judge docker halves (R42 (3)), must confirm max request < 100 ms and bank
   < 3.0 s with the blueprint in the image.
5. **The blueprint is not committed.** `data/blueprint.npz` in this worktree is my dev export
   of the 4.0B checkpoint. The controller exports and commits the final one.
   - The default suite passes with and without the file.
   - Once it is committed, `tools/check_build.sh` must be re-run: the image grows by 51 MB,
     and startup adds the tree build plus the blueprint load and self-test, about 0.5 s
     quiet.
6. **Not run:**
   - the plan's purify sweep at full size;
   - LBR(v3) vs LBR(v2) at 2000 particles (only a 300-particle smoke, healthy: 0 model
     errors, 1/64 inconsistent);
   - scale 2.5.
   All of these are in the gate script for free CPUs.
7. **Small extra changes outside the brief:**
   - `sim/tune.py` and the existing v1/v2 tests now pin `use_blueprint=False`. Without it,
     committing the blueprint would silently change what they measure.
   - The gate refuses mixing blueprints in one cache.
8. **Process discipline.** For about 2 minutes, a 300-particle LBR smoke (1 process, nice 19)
   ran alongside the 2-worker sweep, so 3 of my processes were busy at once. Everything else
   kept to 2 or fewer, at nice 19. I never touched the trainer: pid 210070 was still running
   at the end.

## Fix round 1

sim/gate_v3.py main now exits 4 (message on stderr) when no blueprint loads and sections a, b, d or e are requested; new `--allow-no-blueprint` flag bypasses it. Test `test_gate_refuses_without_blueprint` in tests/test_gate_v3.py (mocked warmup, no real runs).
Command: `nice -n 19 .venv/bin/python -m pytest -q tests/test_gate_v3.py tests/test_policy.py` -> 42 passed, 22 deselected in 23.50s.
