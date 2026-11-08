# Task F2 report

## Done
- Copied runtime files unmodified into `sim/opponents/field2/{field2_solver,field2_bayes,field2_adaptive}/` (no dev/, harness, sdcore.py). README with sha256 table and descriptions: `sim/opponents/field2/README.md` (same 10 hashes as sources; game.py/evaluator.py cmp-identical to the repo root).
- `sim/opponents/__init__.py`: ZOO entries via `bot_file_factory`, new `FIELD2`, `TRAIN = ZOO - HOLDOUT - FIELD - FIELD2`.
- `sim/tournament.py`: `field2` in OPPONENT_SPLITS / select_opponents (heuristic_bot uses the same list). Tuner already refuses anything outside TRAIN; added test cases.
- `tests/test_field2.py` (new): names, partition `HOLDOUT|TRAIN|FIELD|FIELD2 == set(ZOO)` and pairwise disjoint, select_opponents, tuner refuses, both CLIs accept `field2`, README hashes match copies, data tables really loaded (solver PF_PCT/RIVER/PF_E; bayes/adaptive tables equal the .npy files, since both silently rebuild on failure), smoke per bot (10 deals vs calling_station, sys.monitoring swallowed-exception guard + raises > 0), slow 200-deal per bot.
- Edits to existing tests: `swallowed_exceptions` in tests/test_field.py takes optional `bot_dir` (default unchanged); test_tournament.py pinned partition test now includes FIELD2; test_tune.py refuse-list adds "field2"; test_zoo.py excludes FIELD2 from the load-only test.

## sha256
| file | sha256 |
|---|---|
| field2_solver/agent.py | b106228f0f395b6fbced5cedd30e2a6c700c0cf17afe179e561e4d27590185f0 |
| field2_solver/preflop.npz | cae918c089b0bdbf8bc145e9475895df1dc163262f71e394cd457480c5b7bdf4 |
| field2_solver/river.npz | eaf8fc83ced66ae3224e0c939af708d07c030fe3406116991da2a0805c7af177 |
| field2_solver/pf_range.npy | 0851eea61c6da6d79c7358029c809d93ce007c859798a45f6639e15a443d4002 |
| field2_bayes/agent.py | 9847e88874e18c55769b81e281789934b13940a8d326126e1ecbac988f5387a6 |
| field2_bayes/tp_rank.npy | d5e3d01b4a34821070d85fe094a313727cb8b19c14fc473bebd959868eb7e26a |
| field2_bayes/pf_hs.npy | 6ab8ef067af2cedb08b7fa18bf671e8eb66b3f5a0dacb301c0b9745133197c8c |
| field2_adaptive/agent.py | 76e2ddeb65b2c2c2fd58fbb5305c286a786bf565b0df21b7d99a4dad0b81440e |
| field2_adaptive/ptab.npy | 37edbb5a5b2aaad08d17dd0be4d3572b441dff0a81c1041802ff62d7c2c5b966 |
| field2_adaptive/pfeq.npy | 73102322789096d6bb95bee2e1eb40fd1affc7d1df98aee05c44e4e59a113281 |

## Tests
- RED: `pytest tests/test_field2.py` -> ImportError: cannot import name 'FIELD2' from 'sim.opponents'.
- GREEN: test_field2 + test_field + test_tune + test_zoo (not slow): 87 passed. Slow test_field2: 3 passed (2 s).
- Full suite `-m "not slow"` minus tests/test_judge_fidelity.py: 629 passed, 2 xfailed. test_judge_fidelity::test_timeouts_match_the_judge[slow-every-9th] failed once under CPU load (timing-based, unrelated); not re-run alone.
- Slow tests of other files not run.

## Concerns
- Bayes and adaptive folders had no REPORT.md; their README descriptions come from agent.py strategy comments.
- Bots throttle under CPU load; evaluate on a quiet machine.
- `docs/results` not updated (no information run, per brief).
