# Task 18 report: external-sampling MCCFR trainer

**Status: DONE_WITH_CONCERNS.** The trainer is correct, resumable and verified. The concerns (below) are:
- the real-data exploitability diagnostic is only a sanity monitor;
- throughput swings with machine load;
- the file size;
- three interpretations to confirm.

Branch `kevin-cfr` in `/home/kchen/code/heads-up-bundle-cfr`. Nothing pushed.

## Commits

| sha | subject |
|---|---|
| `605a4bc` | Accept numpy integers as action codes in node_from_actions. |
| `5b5949e` | Make load_deals refuse a dataset whose features hash differs from the bucket table's. |
| `e67e787` | Add the external-sampling MCCFR trainer with resumable checkpoints and an exact best-response evaluator. |
| `90cfd02` | Document the MCCFR verification run, bucket-count throughput and the six-hour launch in the training runbook. |
| `7d1a0f1` | Make the slow MCCFR behaviour tests robust to run-to-run variation of hogwild training. |
| `14c83f4` | Set the runbook's six-hour launch to an iteration target the wall clock reaches first. |

## What I implemented

### `train/mccfr.py` (new, 792 lines)

The trainer, its CLI, the evaluator, checkpoints and the loader. The module docstring documents every convention.

- **Game and info sets.** Chance draws one dataset row uniformly. The info set is (decision node, `buckets[row, actor seat, node street]`), so the game has imperfect recall.
- **Seat mapping.** Seat 0 = button, matching both the dataset and the tree arrays. `terminal_value(tid, player, showdown, term_fold_by_button, term_contrib)` (njit) gives the button:
  - `-button_in` if the button folded;
  - `+bb_in` if the BB folded;
  - at showdown, `+bb_in` if the button wins, `-button_in` if it loses, 0 for a split.
  - The BB gets the negative. It is tested against `game.Game` nets for both seats and both engine button seats.
- **Layout (R26).**
  - `regret` and `avg` are float32 `[n_info, 4]`. **[Corrected in fix round 1: `avg` is now float64.]** Node n owns rows `offset[n] .. offset[n+1]-1`, one per bucket of its street, so infoset (n, b) is row `offset[n] + b`.
  - The counts come from the dataset's meta. They are verified equal to `bot.abstraction.NUM_BUCKETS` when the table is `data/buckets.npz`.
  - No row is unused. 1,601,400 infosets at 200/100/100/100. **[Corrected in fix round 1: the original sentence called each street's `[nodes, B_s, 4]` block "a slice" of this array. That is false. Rows are node-major in DFS node-id order and the streets interleave (3,182 street changes), so a street's block is a gather: `street_rows(tree, offset, s)`.]**
  - `offset` is stored in every file.
  - Illegal actions are never written; a test checks they stay exactly 0.
- **Algorithm.** Lanctot 2009 external sampling.
  - Iteration t draws its row and the traverser `t % 2` from its own rng stream.
  - At a traverser node, the walk explores every legal action, then adds `regret += w*(v(a) - Σ σ v)`.
  - At an opponent node, it adds `avg += w*σ` (the reach-weighted average, since the opponent's reach sampled the node), then samples one action.
  - Regret matching uses positive regrets, uniform over legal actions when none is positive.
  - `linear=True` gives Linear CFR (Brown & Sandholm 2019): `w = t + 1` on both the regret and the average increments. `linear=False` gives w = 1.
  - The walk is iterative: one frame per traverser node, opponent nodes pass straight through. Max depth is 14 decisions.
- **RNG.** Counter-based: splitmix64 keyed by (seed, iteration). An iteration's randomness depends only on (seed, t), never on threads or block boundaries.
- **Threads (R23).** One numba `parallel=True` kernel (`_run_block`) runs a block of iterations, `prange` over threads.
  - Iterations are handed out in (even, odd) pairs, so every thread alternates the traverser.
  - Lock-free (hogwild) updates to the shared arrays. The docstring documents why the races are benign: a lost update drops one sample and is rare, and each row is read once and normalized from that copy, so σ is always a valid distribution.
  - Per-thread frame buffers are allocated outside the `prange`, padded so no two threads share a cache line.
  - With threads=1 a run is deterministic.
- **Checkpoints and resume (R23).**
  - Atomic writes: `<out stem>.ckpt-<iterations>.npz` every 10%, on SIGINT/SIGTERM at the end of the current block (blocks auto-size to about 0.5 s), and when `--max-hours` (training time summed over resumes) runs out. The final state goes to `out`.
  - Rerunning the same command resumes from the newest of the checkpoints and `out`. It refuses another run's files, comparing seed, linear, count_visits, dataset n / seed / shard size / format, buckets and features sha256, and the counts.
  - SIGINT raises KeyboardInterrupt (CLI exit 130); SIGTERM gives SystemExit(143). Both come after the checkpoint.
  - A run that finishes during a signal returns normally.
- **Metadata for T19.** Every checkpoint and the final output carry:
  - `n_dec`, `n_term` and `tree_child_sha256` (sha256 of `child.tobytes()`);
  - `buckets_sha256`, `features_sha256`, `num_buckets` and the flop and turn grids;
  - `starting_stack`, `small_blind`, `big_blind`, `raise_cap`;
  - `iterations`, `target`, `seed`, `linear`, `count_visits`, `train_seconds`;
  - `dataset_path`, `dataset_n`, `dataset_seed`, `dataset_shard_size`, `dataset_format`;
  - the numpy and numba versions;
  - `diag` (the diagnostic history) and `visits` (when counted).

  `load_checkpoint(path, tree=None, buckets_table=DEFAULT_PATH)` refuses a mismatch with:
  - the current tree (n_dec, n_term, child hash);
  - the game constants;
  - the bucket table: its sha256, the features sha256 (via `gen_deals.open_table`) and the counts. Pass `None` for synthetic data.
  - It also checks offsets, shapes and dtypes. T19 can call it directly.
- **Evaluator.**
  - `best_response(tree, strategy, offset, buckets, showdown, responder, w_fit, w_val)` is backward induction over the tree for a weighted set of rows. It is an explicit-stack DFS in numba with per-row reach and value vectors. At a responder node, each bucket picks the action that maximizes reach-weighted value.
  - `profile_value` gives the self-play value. `evaluate` gives (value, br_button, br_bb, exploitability = (br_button + br_bb)/2, exact), all in bb per hand.
  - It is exact on perfect-recall rows.
  - The checkpoint diagnostic runs it on all rows (identical rows merged with counts; exact if perfect recall), or on `--eval-rows` (default 8,192) fixed sampled rows of a big dataset, labelled approximate.
- **CLI:** `python -m train.mccfr --deals --iterations --out --seed [--no-linear] [--threads] [--buckets PATH|none] [--eval-rows] [--max-hours] [--log-every] [--count-visits]`.

### Changes to existing files

- **`bot/tree.py`** (T15 deferred minor): `node_from_actions` accepts numpy integers (`_codes(..., numpy_ints=True)`). It still refuses bools, numpy bools and floats. `node_of` is unchanged and still refuses numpy ints (the existing lookalike test requires it).
- **`train/gen_deals.py`** (T17 M1): when a table is given, `load_deals` now also checks `features_sha256` against `open_table(buckets_table)`. A changed `data/preflop.npz` under an unchanged `buckets.npz` is refused. One table read, 65 ms.
- **`train/README.md`:**
  - a note on the features check in the loader section;
  - a new MCCFR section: algorithm summary, long-run launch / monitor / stop / resume, throughput, the verification run with visit statistics, checks.
- **Tests:**
  - `tests/test_tree.py`: numpy-int acceptance.
  - `tests/test_gen_deals.py`: features-hash refusal, using a modified preflop table via monkeypatched `PREFLOP_TABLE`; the M4 guard `gd.PREFLOP_TABLE == bot.preflop._DEFAULT_PATH`.
  - `tests/test_mccfr.py` (new): 12 default tests and 4 slow ones.

## Tests

### Default tests (`tests/test_mccfr.py`, 12 tests, about 10 s)

- `test_terminal_values_match_the_engine_for_both_seats`: 24 real dataset deals (gen_deals), each with the button in engine seat 0 and seat 1, 6 random hands each. Trainer value = engine net for both players. Covers folds by either player, and showdowns won and lost.
- `test_offsets_give_each_node_its_streets_bucket_count`.
- `test_best_response_known_answers`: against "fold to any bet", BR_button = 1 bb, BR_bb = 0.5 bb, exploitability 0.75, value −0.5 (hand-derived).
- `test_best_response_beats_random_strategies_and_matches_its_own_policy`:
  - the BR policy is legal everywhere;
  - playing it against σ gives exactly the BR value;
  - none of 10 random responder strategies does better (both seats).
- `test_regret_matching_never_picks_illegal` (plan): avg and regret are exactly 0 at every illegal action of every infoset, and the normalized strategy sums to 1.
- `test_checkpoint_roundtrip` (plan): every metadata field; nine 10% checkpoints at the right iterations, each with an exact diagnostic; save then load gives back identical arrays and meta.
- `test_resume_from_a_checkpoint_equals_an_uninterrupted_run`:
  - delete everything but the 40% checkpoint, resume, and regret / avg / visits / diag are bit-identical;
  - a finished run is not redone;
  - a larger target extends it.
- `test_loading_refuses_another_tree_table_or_run`: another tree, the real table on synthetic data, a bad num_buckets, a retuned raise cap, another seed, another dataset.
- `test_training_refuses_buckets_outside_the_counts`: numba does not bounds-check, so this is validated up front.
- `test_real_dataset_run_with_the_approximate_diagnostic`: the real table on 64 generated deals. Checks the counts, sha and offsets; the diag is finite and labelled not exact.
- **`test_exploitability_falls_on_a_perfect_recall_game[1, 2]`** (the brief's known-answer test). Three classes, the same bucket on every street, the higher class wins 80%. 200k iterations, exact diagnostic at every 10%. The final value must be below 0.25 bb (set from measurement, below) and below a third of the first. On one thread, which is deterministic, it must fall at every checkpoint (10% tolerance). Two threads cover the hogwild path.

### Slow tests (`-m slow`, 4 tests, about 21 s pinned to 8-15)

- `test_exploitability_keeps_falling_with_hogwild_threads`: 8 threads, 4M iterations, below 0.035 (measured 0.026).
- `test_one_bucket_game_is_seed_stable` (plan): two seeds, 4 threads, 2M iterations, self-play values within 0.02 bb.
- `test_strong_bucket_raises_more_than_weak` (plan; the design is under Concerns).
- `test_sigterm_checkpoints_and_resume_is_exact`: a subprocess CLI run is sent a real SIGTERM. It exits 143 with one checkpoint at k iterations. Resuming in-process to k + 3000 is bit-identical to an uninterrupted run.

### TDD evidence

**RED** (before any implementation):

    $ .venv/bin/python -m pytest -q tests/test_mccfr.py tests/test_tree.py::test_node_from_actions_accepts_numpy_integers_but_not_bools \
        tests/test_gen_deals.py::test_load_deals_rejects_features_from_another_preflop_table tests/test_gen_deals.py::test_preflop_table_is_the_one_the_runtime_reads
    E   ImportError: cannot import name 'mccfr' from 'train' (/home/kchen/code/heads-up-bundle-cfr/train/__init__.py)
    $ .venv/bin/python -m pytest -q tests/test_tree.py::test_node_from_actions_accepts_numpy_integers_but_not_bools tests/test_gen_deals.py::...features... tests/test_gen_deals.py::...preflop_table...
    E       assert None == 2     (node_from_actions(array([1, 1]), tree))
    E       Failed: DID NOT RAISE ValueError
    FAILED tests/test_tree.py::test_node_from_actions_accepts_numpy_integers_but_not_bools
    FAILED tests/test_gen_deals.py::test_load_deals_rejects_features_from_another_preflop_table
    2 failed, 1 passed

Why each failure was expected:
- the module did not exist;
- `_codes` rejected numpy ints;
- `load_deals` ignored `features_sha256`.

The M4 guard passed at once: it is a regression guard on an invariant that already holds.

The first green attempt gave 10 passed and 1 failed. The 0.05 exploitability threshold had been a guess before any measurement; one thread measured 0.194 at 200k. I set the threshold from measurement at 0.25.

**RED found by measurement: two threading bugs, now guarded.** The 1-vs-8-thread curve showed 8 threads at 20M iterations (0.241) worse than one thread at 200k (0.194).

| kernel threads | numba threads | block | exploitability at 1M |
|---|---|---|---|
| 1 | 1 | 4096 | 0.0545 |
| 2 | 1 (sequential) | 4096 | **1.02** |
| 2 | 1 (sequential) | 64 | 0.052 |
| 2 | 2 | 4096 | 0.103 |

1. **numba allocation hoisting.** `_run_block.parallel_diagnostics(level=4)` reported the five `np.empty` frame buffers inside the `prange` body as "hoisted out of the parallel loop ... and reused inside the loop". Every thread shared one traversal stack. Fixed with per-thread slices allocated before the loop; the diagnostics now report "No allocation hoisting found".
2. **One player per thread.** The interleave `j ≡ k (mod threads)` with an even thread count dedicated each thread to one traverser. Long one-player runs against a frozen opponent are best-response dynamics, and they oscillate: the table above isolates this. Fixed by handing out (even, odd) pairs. The single-thread order is unchanged, so determinism and resume are unaffected.

The default 2-thread test fails on each bug when it is reintroduced in a scratch copy of the module (same test helpers, 200k iterations):

| reintroduced bug | exploitability | result |
|---|---|---|
| one player per thread | 1.31 bb (curve 2.6 → 1.3, non-monotone) | FAILS |
| shared stack | 11.7 bb, then NaN | FAILS |
| fixed kernel, 4 runs | 0.184-0.207 | passes |

**GREEN** (final code):

    $ .venv/bin/python -m pytest -q                                   # full default suite, unpinned
    311 passed, 12 deselected, 1 warning in 132.18s                   # the warning: pre-existing starlette/httpx deprecation (T17 M6)
    $ .venv/bin/python -m pytest -q tests/test_mccfr.py
    12 passed, 4 deselected in 9.65s
    $ taskset -c 8-15 .venv/bin/python -m pytest -q -m slow tests/test_mccfr.py
    4 passed, 12 deselected in 21.38s

The seed-stability and strong-bucket tests were made robust after a re-run failure (Concerns, items 5 and 6). Each then passed 3 of 3 re-runs.

## Measured numbers

### Throughput: 2M dataset, 200/100/100/100 buckets, normal priority

| threads | CPUs | session 1 (from scratch, load avg 11.6-18.3) | session 2 (from a common 18.7M-iteration state, load avg 17.7-21.9) |
|---|---|---|---|
| 1 | 8-15 | 104,425 it/s | 71,116 it/s |
| 4 | 8-15 | 552,755 it/s | 199,894 it/s |
| 8 | 8-15 | 1,610,452 it/s | 351,357 it/s (397,151 with `--count-visits`) |
| 16 | all (overlapped Track A on 0-7; under 60 s) | 1,101,863 it/s | 741,515 it/s |

- Session 1 measured each count at a different training stage (25 s from scratch). Session 2 is the fair comparison, but it ran while Track A started six normal-priority workers on 0-7, two at nice 10, and the 10M generation's eight nice-19 workers sat on 8-15.
- The kernel is memory-latency-bound: random access into 49 MB of arrays plus the dataset. It shares the single 32 MB L3 and the memory bandwidth with everything else on the PC, hence about 4× swings. **Size runs by wall clock.**
- The verification run below averaged 642,725 it/s on 8 threads over 300M iterations, at load 19-25. That is the best long-horizon number.
- **Bucket counts barely matter.** I split every bucket 4 ways (800/400/400/400, 6.4M infosets, 195 MB): 310,906 it/s on 8 threads, against 346,572 and 270,497 at 1× measured just before and after.
- **Pure numpy/Python:** the same kernel with `NUMBA_DISABLE_JIT=1` runs 649 it/s. That is far below the plan's 20k it/s rule, so numba is used, 100-2,500× faster. Compile on first call takes about 1.8 s, then it is cached (`cache=True`).

### Memory per infoset

- 32 bytes: regret + avg, float32 × 4 actions each. Plus 8 bytes with `--count-visits` (int64).
- Default table: 1,601,400 infosets = 48.9 MB of arrays. A checkpoint is 64 MB with visits.
- **[Corrected in fix round 1: 48 bytes per infoset (16 regret + 32 float64 avg), plus 8 with visits. The default table needs 73 MB of arrays, and a checkpoint is 90 MB with visits.]**
- Process max RSS on the 2M dataset: 274 MB. Before the arrays it was 190 MB: interpreter, numba, tree and dataset.

### Exploitability curve on the perfect-recall synthetic game

Three classes, 3,000 rows; exact best response.

| iterations | 1 thread (deterministic) | 8 threads, linear (20M run) | 8 threads, uniform weights |
|---|---|---|---|
| 20k | 1.355 | | |
| 40k | 0.835 | | |
| 60k | 0.604 | | |
| 80k | 0.429 | | |
| 100k | 0.355 | | |
| 120k | 0.311 | | |
| 140k | 0.268 | | |
| 160k | 0.225 | | |
| 180k | 0.199 | | |
| 200k | 0.194 | | |
| 1M | 0.0545 | | |
| 2M | 0.0322 | 0.0323 | |
| 4M | | 0.0258 | |
| 6M | | 0.0204 | |
| 8M | | 0.0179 | |
| 10M | | 0.0148 | |
| 12M | | 0.0128 | |
| 14M | | 0.0114 | 0.0143 |
| 16M | | 0.0116 | 0.0131 |
| 18M | | 0.0105 | 0.0127 |
| 20M | | 0.0103 | 0.0114 |

- The decay is about 1/√T, as MCCFR should give: 10× the iterations gives 3.1× lower exploitability.
- The button's game value settles at +0.103 bb inside a shrinking bracket [−br_bb, br_button] = [0.092, 0.112].
- At 200k, 2 threads give 0.193 and 8 threads 0.209. Four 2-thread runs gave 0.184-0.207.

### Verification run

Run on the stable 2M set in `train_data/` (not d10m), 8 threads, CPUs 8-15, normal priority:

    nohup taskset -c 8-15 .venv/bin/python -m train.mccfr --deals train_data --iterations 300_000_000 \
        --out train_data/mccfr-verify/blueprint_raw.npz --seed 0 --threads 8 --count-visits --max-hours 0.30 \
        --log-every 60 > train_data/mccfr-verify/train.log 2>&1 &

- **Progress.** 300M iterations in 7 min 46 s of training (466.8 s), 642,725 it/s on average. The rate climbed over the run:

  | elapsed | iterations | rate now | rate since start |
  |---|---|---|---|
  | 1:00 | 27.2M | 451k | |
  | 2:00 | 48.3M | 351k | 401k |
  | 3:00 | 88.2M | 661k | 487k |
  | 4:01 | 125.0M | 613k | 519k |
  | 5:01 | 162.2M | 617k | 538k |
  | 6:01 | 210.6M | 799k | 582k |
  | 7:02 | 265.7M | 915k | 629k |
  | 7:46 | 300M | 769k | 643k |

- **Outputs.** Nine checkpoints (30M … 270M) and the final output, 64 MB each, 0.1-0.2 s to write, 613 MB in all. They are left in `train_data/mccfr-verify/`, useful for developing T19. No errors.
- **Diagnostics.**
  - The self-play value is +0.086 to +0.103 bb for the button.
  - The logged approximate exploitability was the first design (fit on half the sample, value on the other half). It read −1.28 to −1.46 "bb", which is meaningless (Concerns, item 1), so I replaced it.
  - The replacement (greedy response in-sample on the fixed sample), recomputed on the checkpoints, reads 3.26 → 3.46 bb on 8,192 rows: flat, dominated by the sample.
  - On the 30M and 300M checkpoints it reads 0.84 and 0.90 bb on 65,536 rows (16-18 s), and 0.296 and 0.294 bb on 262,144 rows (76-79 s).
- **Visits** (traverser), after 300M iterations:

  | street | infosets | visits / iteration | zero-visit | visit-weighted median | weighted 10th pct | visits in infosets with <1k / <10k |
  |---|---|---|---|---|---|---|
  | preflop | 12,400 | 1.83 | 0% | 254,622 | 28,215 | 0.1% / 3.6% |
  | flop | 81,000 | 5.63 | 0% | 97,031 | 11,294 | 0.5% / 9.0% |
  | turn | 367,800 | 9.94 | 0% | 35,037 | 4,167 | 2.1% / 23.1% |
  | river | 1,140,200 | 15.03 | 0% | 23,132 | 2,228 | 4.4% / 32.4% |

  In all, 32.4 traverser visits per iteration.
- **Blueprint sanity** (average strategy, real data, 300M iterations):
  - **Root, button:** bucket 0 folds 1.00; 20 folds 0.885; 50 limps 0.68; 100 raises 0.62; 150 raises 0.97; 199 raises 0.95.
  - **BB facing a pot raise:** 0 and 50 fold; 100 folds 0.42 and calls 0.58; 150 calls 0.95; 199 re-raises pot 0.94.
  - **Checked-down river node, BB:** P(raise) by river bucket:

    | bucket | 0 | 25 | 50 | 75 | 94 | 95 | 96 | 97 | 98 | 99 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | P(raise) | 0 | 0 | 0 | 0.26 | 0 | 0.05 | 0.30 | 0.98 | 0.67 | 0.62 |

  Seat mapping and signs hold on the real data.

### Sizing recommendation for a ~6 h run on 12-16 threads

**Recommend 400/400/400/400 buckets** (preflop 400, flop grid 20×20, turn grid 20×20, river 400 including the 5 nut buckets) with **`--iterations 24_000_000_000 --max-hours 6`**. The wall clock decides; expect 12-24B iterations.

Reasoning:
1. **Iterations are cheap.** At the measured 0.55-1.1M it/s (8 threads under load to 16 threads), 6 h is 12-24B iterations: 60-120× the plan's 200M floor, which takes about 5 minutes. Throughput is nearly independent of the bucket counts (the 4× split cost about nothing). So the budget should buy a finer abstraction, not more iterations on the default one.
2. **Visits scale as iterations / buckets.** Scaling the measured visit-weighted statistics by (T / 300M) × (B_default / B), 400/400/400/400 at the conservative T = 12B gives:

   | street | median | 10th percentile |
   |---|---|---|
   | preflop | 5.1M | 564k |
   | flop | 970k | 113k |
   | turn | 350k | 41.7k |
   | river | 231k | 22.3k |

   At 24B, double that. The river 10th percentile is about 10× what the 2M verification blueprint got (2.2k), and every street stays at or above 2×10^4.
3. **The dataset supports it.** d10m gives 25k deals per bucket per seat per street at 400 (R37). Finer river buckets (e.g. 800) would halve both that and the visits (river p10 about 11k at 12B), with thinner chance sampling per infoset. That is an option only with a 20M dataset.
4. **Footprint.** 6,380,000 infosets (62×400 + 15,890×400): 204 MB of arrays plus 51 MB of visits, under 1.5 GB of RAM. Checkpoints are 256 MB (2.6 GB for ten). A uint8 export (T19) would be about 26 MB, under the 100 MB file limit. **[Corrected in fix round 1, with float64 avg: 306 MB of arrays plus 51 MB of visits. Checkpoints are 357 MB, 12-20 of them in 6 h counting the 30-minute ones.]**
5. ~~**float32 holds.** At this scale the hottest infosets see 10^7-3×10^7 visits. A sequential-accumulation simulation (k = 10M and 30M visits, linear weights) shows float32 shifting a 0.70/0.25/0.05 average by at most 0.5 percentage points (0.7048/0.2462/0.0490). That is acceptable; float64 avg would remove it at 2× avg memory.~~ **[Withdrawn in fix round 1. The review's exact single-rounding simulation gives 2.2 pp at 1.5e7 updates and 7.9 pp at 6e7 with constant σ, systematic across seeds. My simulation understated it. Fixed: avg is float64, regret stays float32.]**

If the controller keeps the default 200/100/100/100 instead, a 6 h run gives river p10 about 90k (overkill), and the abstraction is the bottleneck.

### Long run: exact commands

From the worktree root, with the venv's python, once `train_data/d10m` is complete.

    # optional, for the recommended abstraction: refit (same --seed 0 and design sizes) then re-bucket (seconds per 1M deals)
    python -m train.fit_buckets --per-street 1000000 --preflop 400 --flop-grid 20x20 --turn-grid 20x20 \
        --river 400 --workers 8 --seed 0 --out data/buckets.npz
    python -m train.gen_deals --rebucket --out train_data/d10m

    # launch, detached, with a log (16 threads if Track A is idle; else prefix taskset -c 8-15 and use --threads 8)
    mkdir -p train_data/mccfr
    nohup python -m train.mccfr --deals train_data/d10m --iterations 24_000_000_000 \
        --out train_data/mccfr/blueprint_raw.npz --seed 0 --threads 16 --count-visits \
        --max-hours 6 > train_data/mccfr/train.log 2>&1 &

    # monitor
    tail -f train_data/mccfr/train.log      # progress each minute; a diagnostic + checkpoint line every 10%
    ls -la train_data/mccfr/
    ps -o pid,etime,pcpu,args -C python | grep train.mccfr

    # stop (checkpoints at the end of the current ~0.5 s block, exits 143) and resume (the identical command)
    pkill -TERM -f train.mccfr
    nohup python -m train.mccfr --deals train_data/d10m --iterations 24_000_000_000 \
        --out train_data/mccfr/blueprint_raw.npz --seed 0 --threads 16 --count-visits \
        --max-hours 6 >> train_data/mccfr/train.log 2>&1 &

- `--max-hours` counts training time summed over resumes. When it runs out, the run writes a checkpoint and `blueprint_raw.npz` at the iterations done.
- A tmux variant is in the README.
- A refit of `data/buckets.npz` also needs `tests/test_abstraction.py` re-run, and a commit of the table: the README's existing "After any refit" section.

## Files changed

- `train/mccfr.py` (new)
- `tests/test_mccfr.py` (new)
- `bot/tree.py`
- `tests/test_tree.py`
- `train/gen_deals.py`
- `tests/test_gen_deals.py`
- `train/README.md`

## Self-review findings (fixed before reporting)

1. The two threading bugs above (allocation hoisting, one player per thread). Both were found by measuring the exploitability with threads > 1, and both are now guarded by a default 2-thread test, shown to fail on each.
2. The real-data diagnostic's first design (split-sample "lower bound") was meaningless at −1.3 bb. It is replaced by the in-sample greedy response. Its measured floor is documented, and it is labelled a sanity monitor.
3. Block-size adaptation collapsed after a 10% mark. Growth was capped at 2×count; it is now 2×max(block, count).
4. Removed an unused import, and fixed the `Evaluation.exact` comment.
5. Two slow plan tests were flaky under hogwild variation: seed stability at 300k iterations, and the strong-bucket margin at the BB's node. Both are reworked from measurement (Concerns, items 5 and 6).

## Issues and concerns

1. **The real-data exploitability diagnostic is a sanity monitor, not a convergence measure.**
   - The brief expected "approximate, because of imperfect recall". The bigger problem is the sample: with 8,192 rows over 1.6M infosets, the greedy response mostly exploits the sample's noise. It reads 3.3-3.5 bb flat from 30M to 300M iterations; 0.29 bb at 262k rows (76 s), still flat.
   - Fitting on half and valuing on the other half gives a negative "bound".
   - A real number needs the whole training game (2M rows: about 1.2 GB of buffers and an estimated 10+ min single-threaded per evaluation; not implemented), or T21's LBR probe.
   - Exact checking on perfect-recall data is unaffected.
2. **Throughput depends on machine load**, 4× between sessions minutes apart (memory-bound kernel, shared L3). The 16-thread points overlapped Track A (as the brief allowed, under 60 s). Size the long run by `--max-hours`.
3. **Interpretation: linear weighting.** It applies to both the regret and the average increments (Linear CFR as in Brown & Sandholm 2019 and Pluribus); the plan text names only the average. `--no-linear` gives uniform weights. Both converge on the test game: 0.0103 vs 0.0114 at 20M.
4. **Interpretation: layout.** One flat `[n_info, 4]` array with per-node offsets, instead of either suggested option (`[n_dec, max_B, 4]` or separate per-street arrays). T19 should read `offset` from the file. **[Corrected in fix round 1: this originally said each street's array was "a contiguous slice". It is not; the streets interleave in node-id order. A street's array is the gather `a[street_rows(tree, offset, s)].reshape(-1, B_s, 4)`.]**
5. **Plan test `test_strong_bucket_raises_more_than_weak`: a different synthetic game, and the > 0.8 check moved to the button's node.**
   - With information only on the river, the check-down line is almost never played: (first river node, 99) had 2,400 visits in 10M iterations, so its average stays near the prior.
   - Giving only the BB information makes the button fold preflop.
   - The test uses a perfect-recall class game with buckets {0, 33, 66, 99}, where the higher wins. At the BB's first river node the equilibrium mixes (the nuts check about 18% to check-raise): measured 0.796-0.85 over 8 runs, so "> 0.8" there would be a coin flip.
   - The test asserts > 0.8 at the button's first river node (checked to; equilibrium 1.000 in every run). At the BB's first river node it asserts > 0.7 and at least 0.4 above every weaker class.
6. **Plan test `test_one_bucket_game_is_seed_stable` needed 2M iterations.** Both seeds converge to +0.345 bb: the 4,000 random showdowns do not average 0, so the BB folds to avoid big pots. At 300k the 4-thread values differed by 0.04 in one run. The difference is ≤ 0.003 from 1M on.
7. **Resume equivalence is bit-exact only with one thread.** With N threads, hogwild interleaving is nondeterministic, so a resumed run is statistically equivalent, not identical: same iteration count, rng streams and weights.
8. **`train/mccfr.py` is 792 lines**: trainer, evaluator, checkpoint I/O and CLI in the one file the plan names. I did not split it. The evaluator could become its own module if T19 or T21 want it separately.
9. **`node_of` still refuses numpy ints**, by design and per the existing test; only `node_from_actions` accepts them.
10. **`--count-visits` must be chosen at the start of a run** (it is part of the run identity). I recommend it for the long run: about 0% throughput cost, and T19/T20 can see which infosets are under-trained.

---

## Fix report: round 1

Commits on `kevin-cfr`, not pushed:

| sha | subject |
|---|---|
| `e3bbeaf` | Accumulate the MCCFR average strategy in float64 and document the interleaved node-major layout with a per-street gather. |
| `d75dc9d` | Write an MCCFR checkpoint whenever 30 minutes of wall clock have passed since the last one. |

### Important 1: `avg` is float64, `regret` stays float32 (controller ruling)

- The trainer allocates `avg` as float64 `[n_info, 4]`. The numba kernels specialize on the dtype, so the update `avg[idx, a] += w * σ` now accumulates in float64. `average_strategy`, save and load handle it unchanged.
- **Old float32 checkpoints are refused, not upcast.**
  - `FORMAT` is bumped to 2.
  - `load_checkpoint`, and so also any resume over such a file, rejects a format-1 file with: "checkpoint format 1 keeps avg in float32, which biases the average strategy of heavily visited infosets; this code reads format 2 (float64 avg). Retrain: format-1 runs cannot be resumed."
  - A format-2 file whose `avg` is not float64 is rejected by the dtype check ("avg is float32 ..., expected float64").
  - Why refuse: T19 gets one exact on-disk contract, and no biased float32 accumulation can reach an export silently.
  - The only format-1 files in existence are the 8-minute verification run's, in `train_data/mccfr-verify/`. Rerunning its command regenerates them; the README says so.
- **T19 contract:** `regret` float32 and `avg` float64, both `[n_info, 4]`. The module docstring (Layout), the README and the report state it.
- **Cost:** 48 bytes per infoset (was 32), plus 8 with visits.
  - Default table: 73 MB of arrays, a 90 MB checkpoint with visits.
  - 400/400/400/400: 306 MB of arrays, a 357 MB checkpoint with visits.
- The report's "float32 holds / at most 0.5 pp" item is withdrawn and marked so in place.

### Important 2: the layout claim is corrected

- The module docstring's Layout section now says:
  - rows are node-major in DFS node-id order;
  - the streets interleave (3,182 street changes along the ids, recounted);
  - so a street's `[nodes, B_s, 4]` array is a gather, never a slice.
- New helper `street_rows(tree, offset, street)`: int64 rows, by node then bucket, so `a[street_rows(...)].reshape(-1, B_s, 4)` is the per-street array. It equals the reviewer's `np.concatenate([np.arange(offset[n], offset[n+1]) for n in np.flatnonzero(tree.street == s)])`.
- New test `test_street_rows_gather_each_streets_per_street_array`. It checks:
  - equality with that concatenate;
  - per-street arrays built from it hold exactly each node's rows;
  - the four streets' rows partition all rows;
  - the 3,182 street changes;
  - for flop, turn and river, the rows are not contiguous, so a slice would be wrong.
- The README and the report (lines 37 and 356, original numbering) are corrected, each marked "[Corrected in fix round 1]".
- Two more stale statements in the same docstring are fixed: the Threads paragraph still described the old `t0 + k, t0 + k + threads` assignment (it is (even, odd) pairs), and the usage line said 200M iterations.

### Elevated: a checkpoint every 30 minutes of wall clock at most

- `train(..., checkpoint_minutes=30.0)` and CLI `--checkpoint-minutes` (default 30; 0 or None disables). After each block (about 0.5 s), if that much wall clock has passed since the last checkpoint of any kind, a checkpoint is written at the current iteration count.
- These use the same `save` path as the 10% marks: the same atomic `gd._write` (temp file, fsync, rename) and the same `<stem>.ckpt-<iterations>.npz` naming. Resume picks the newest by iterations.
- At the 24B target, the 10% marks fall 36-73 minutes apart, so the clock bounds a WSL2 shutdown or Windows sleep to 30 minutes of lost training.
- The README notes the disk cost: 357 MB per checkpoint at 400 buckets with visits, 12-20 per 6-hour run (4-7 GB). Any but the newest may be deleted.
- New test `test_wall_clock_checkpoints_between_the_marks`. With `checkpoint_minutes=1e-9` every block writes one. It checks:
  - at least 2 checkpoints, strictly increasing, all between the marks, no `.tmp` left;
  - the arrays are identical to a run without them, so checkpoint writes do not perturb training;
  - the 30-minute default writes none in a short run;
  - resuming from the first clock checkpoint reproduces the uninterrupted bits.

### TDD evidence

RED, with the new and changed tests run against the pre-fix trainer (`git stash push train/mccfr.py`, run, `git stash pop`):

    $ .venv/bin/python -m pytest -q tests/test_mccfr.py -k "street_rows or float32_average or checkpoint_roundtrip"
    E           AttributeError: module 'train.mccfr' has no attribute 'street_rows'
    E       Failed: DID NOT RAISE ValueError
    FAILED tests/test_mccfr.py::test_street_rows_gather_each_streets_per_street_array
    FAILED tests/test_mccfr.py::test_checkpoint_roundtrip - AssertionError: asser...     (avg dtype float32)
    FAILED tests/test_mccfr.py::test_loading_refuses_a_float32_average - Failed: ...
    3 failed, 15 deselected in 1.30s

    $ .venv/bin/python -m pytest -q tests/test_mccfr.py -k wall_clock               # before implementing it
    E       TypeError: train() got an unexpected keyword argument 'checkpoint_minutes'
    1 failed, 18 deselected in 0.79s

GREEN, the covering tests, pinned:

    $ taskset -c 8-15 .venv/bin/python -m pytest -q tests/test_mccfr.py tests/test_gen_deals.py
    31 passed, 4 deselected in 8.45s
    $ taskset -c 8-15 .venv/bin/python -m pytest -q -m slow tests/test_mccfr.py
    4 passed, 15 deselected in 10.45s          # one-bucket seed 3.59 s, strong bucket 3.54 s, 8-thread convergence 1.79 s, SIGTERM 0.97 s
    $ taskset -c 8-15 .venv/bin/python -m pytest -q tests/test_mccfr.py -k "wall_clock or roundtrip"   # after a final docstring rewrap
    2 passed, 17 deselected in 3.34s

The slow suite exercises the recompiled float64 kernel end to end:
- the 8-thread convergence test (still below 0.035 at 4M);
- the CLI with the new default `--checkpoint-minutes` under a real SIGTERM, then an exact resume.

### Throughput after the fix

The machine was idle (load 0.1-1), unlike the earlier loaded sessions.

    $ taskset -c 8-15 .venv/bin/python -m train.mccfr --deals train_data --iterations 10_000_000_000 \
        --out train_data/mccfr-tput/blueprint_raw.npz --seed 0 --threads 8 --eval-rows 0 \
        --max-hours 0.0125 --log-every 10 --checkpoint-minutes 0
    mccfr: 17,379,404/10,000,000,000 iterations (0.2%), 1,709,157 it/s now, 1,709,157 it/s since start, elapsed 0:00:10, eta 1:37:20
    mccfr: 36,321,801/10,000,000,000 iterations (0.4%), 1,809,794 it/s now, 1,760,203 it/s since start, elapsed 0:00:20, eta 1:34:20
    mccfr: 55,950,520/10,000,000,000 iterations (0.6%), 1,871,427 it/s now, 1,797,685 it/s since start, elapsed 0:00:31, eta 1:32:11
    mccfr: 75,131,286/10,000,000,000 iterations (0.8%), 1,911,337 it/s now, 1,825,395 it/s since start, elapsed 0:00:41, eta 1:30:37
      final output train_data/mccfr-tput/blueprint_raw.npz at 82,463,691 iterations (0.1 s)
    mccfr: --max-hours 0.0125 reached at 82,463,691 iterations
        Elapsed (wall clock) time: 0:47.99      Maximum resident set size: 321,368 kB

- **8 threads, float64 `avg`: 1.83M it/s on average over 45 s, 1.91M it/s by the end** (idle PC, CPUs 8-15, normal priority).
- I measured the dtype cost itself with a same-conditions A/B: a kernel harness from scratch, 8 threads, 20 s each, alternated, with the pre-fix kernel from 14c83f4.

  | kernel | run 1 | run 2 |
  |---|---|---|
  | float32 `avg` | 1,735,341 it/s | 1,731,142 it/s |
  | float64 `avg` | 1,631,614 it/s | 1,706,534 it/s |

  float64 costs about 4% (1.5-6%).
- With an idle PC, the 6-hour run on 8 threads would reach about 39B iterations (1.8M × 21,600 s). The recommended `--iterations 24_000_000_000 --max-hours 6` would then finish in about 3.7 h. To spend the full 6 h, raise `--iterations` to 40_000_000_000.
- The scratch output `train_data/mccfr-tput/` was deleted.

### Not done in this round (deferred by the controller)

Reviewer Minors M3, M4, M6, M7 and M8.
