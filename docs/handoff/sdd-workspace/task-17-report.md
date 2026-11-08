# Task 17 report: training deal dataset

Branch `kevin-cfr` (worktree /home/kchen/code/heads-up-bundle-cfr), base cf30916.

## What I implemented

`train/gen_deals.py` (new). It generates the dataset, loads it, and re-buckets it.

### Shard format (FORMAT 1)

Each shard is `train_data/deals_<k>.npz`, uncompressed, 50,000 deals by default (`--shard-size`). One row is one uniform random deal:

- `cards` int8 [N, 15] (R13): the button's hole (5, sorted), the BB's hole (5, sorted), then the board as dealt.
- `buckets` int16 [N, 2, 4]: seat 0 = button, by street.
  - Computed per row through the runtime's own `BucketTables.features()` + `bucket_from_features()`, i.e. exactly `bucket()`.
- `showdown` int8 [N]: +1 button, 0 split, -1 BB, from `evaluator.omaha_score` (the engine's own showdown function).
- `features` float64 [N, 2, 6]: preflop strength, flop strength, flop equity, turn strength, turn equity, river equity (`FEATURE_COLUMNS`).
- `river_nut` bool [N, 2] and `river_category` int8 [N, 2].
- Meta:
  - `format`, `seed`, `shard`, `shard_size`, `numpy_version`.
  - `buckets_sha256`: the sha256 of the bucket-table file bytes, with the `num_buckets`, `flop_grid` and `turn_grid` it implies.
  - `features_sha256`: the sha256 of the table's `design_*` arrays plus `data/preflop.npz`.

### Generation

`python -m train.gen_deals --n 2_000_000 --workers 8 --iters 1000` (the plan's command, verbatim).

- Spawn Pool, as in fit_buckets. Workers write their own shards and return only (shard, count).
- Shard k draws its cards only from `default_rng([seed, 17, k])`, so the output is independent of `--workers`.
  - The `17` tag keeps the streams apart from fit_buckets' `[seed, 1]` design stream. Without it, shard 1 with seed 0 would reuse the rng stream that drew the table's opponent-hole designs.
- Atomic writes: `deals_<k>.npz.<pid>.tmp`, then fsync, then `os.replace`.
- A rerun deletes leftover `*.tmp` files, skips finished shards and makes the missing ones.
- A larger `--n` adds shards. A short last shard is redone at full size; it is a prefix of the same stream.
- A run refuses a directory with another seed, shard size, features hash or bucket table, and hints at the fix (another `--out`, or `--rebucket`).
- Every shard self-checks before it is written: the vectorized `buckets_from_features` must equal the per-row scalar buckets, or it raises.
- A Pool-initializer exception is deferred to the first job. An exception escaping an initializer makes the Pool respawn workers forever, which would hang an overnight run.
- `--iters` is accepted and ignored, with a printed note (T16: deterministic features).

### `load_deals(path_or_dir, buckets_table=data/buckets.npz, n=None, cards=False, features=False) -> Deals`

- Returns `Deals(buckets, showdown, meta, cards, features)`: contiguous preallocated arrays.
- Reads only the `buckets` and `showdown` members unless asked.
- Refuses with ValueError:
  - a gap in the shard numbers (an interrupted run);
  - a short shard before the last;
  - shards that disagree on seed, shard size, hashes or counts;
  - a `buckets_sha256` that is not the sha256 of `buckets_table` (`None` skips that check, for synthetic data);
  - a wrong member shape or dtype.
- `n=` takes a prefix.

### `rebucket(path, buckets_table, verify=4)`, also `--rebucket` on the CLI

- Refuses, before writing anything, if any shard's `features_sha256` differs from the table's: other designs or another preflop table mean stale features.
- Recomputes `buckets` with the vectorized `buckets_from_features` (np.searchsorted side=right, the same cut as bisect_right).
- Recomputes `verify` evenly spaced deals per shard from their cards with bot.abstraction. Their features and new buckets must match exactly, which catches a change in the feature code, something no hash can see.
- Rewrites each shard atomically with the new buckets, hash and counts. Shards already on the table are skipped, so it is resumable and idempotent.

### Docs

- `train/README.md`: a new runbook section. Also fixed two now-stale lines: "each script writes into data/" and "regenerate both" after a refit.
- `docs/results/v3.md`: created.

## Deviations from the brief (deliberate)

1. **Features are float64, not float32.** The brief allows float32, but float32 breaks the brief's own precondition (bucket-from-stored-features == bucket()).
   - Strengths are exact in float32: multiples of 1/8192 or 1/32768, and preflop is float32 in data/preflop.npz.
   - Flop and turn equity are multiples of 1/6048 and 1/16416, which are not dyadic.
   - fit_edges cuts at sample values, so many deals sit exactly on an edge. A float32 copy rounds below the edge about half the time, and the deal drops a bucket.
   - A mutation probe (features rounded through float32) failed the precondition test and the rebucket == fresh test.
   - On the real 2M dataset, float32 rounding would move 0.25% of flop buckets and 0.13% of turn buckets; preflop and river would not change.
   - Cost: 48 extra bytes per deal on disk (about 1 GB at 20M), none in load_deals RAM.
2. **`--workers` default** is `len(os.sched_getaffinity(0))` where it exists, else `os.cpu_count()`. On an unpinned machine that is the same number. Under `taskset -c 8-15` it gives 8, not 16, so it never oversubscribes a pinned job.
3. **The rng stream is `(seed, 17, shard)`**, not literally `(seed, shard)`. The reason is above.

## Tests (`tests/test_gen_deals.py`, 14 tests, 3.6 s; none needs `@slow`)

Most tests share a module fixture: n=200, seed 3, shard size 64, so shards of 64, 64, 64 and 8. The default suite generates in-process; the 2-worker run takes 0.9 s.

- `test_small_dataset_shapes_and_outcomes` (the plan's test):
  - shapes and dtypes of all arrays, contiguity, meta and hash;
  - the cards equal the deals regenerated from `deal_cards(seed, shard)`;
  - showdown == the `omaha_score` comparison on every row;
  - every street's ids are in range.
- `test_showdown_is_the_engines_check_down_winner`: 40 deals played through `game.Game` as a check-down, with the button on seat 0 or 1. This pins the seat mapping and the sign.
- `test_every_row_has_15_distinct_cards_in_range`: cards distinct and in 0..35, all 36 ids dealt, holes sorted.
- `test_buckets_equal_runtime_bucket_on_the_stored_cards`: `ab.bucket()` for 200 rows x 2 seats x 4 streets.
- `test_stored_features_are_the_runtime_features_and_cut_to_the_stored_buckets`: the precondition. Stored features exactly equal `features()` on the cards, and `buckets_from_features` equals the stored buckets.
- `test_vectorized_cut_equals_bucket_from_features_on_and_beside_the_edges`: 4,000 x 2 synthetic feature rows with values on the edges, one ulp either side, uniform, and 0/1, nut categories 0..8. Each must equal the scalar `bucket_from_features`.
- `test_output_depends_on_the_seed_not_on_the_worker_count`: same seed gives identical shards, 1 vs 2 workers identical, and another seed differs.
- `test_resume_ignores_a_partial_temp_file_and_finishes_identically`:
  - shards 1 and 3 deleted and a truncated `.tmp` left behind;
  - `load_deals` refuses the gap;
  - a rerun gives identical shards, removes the temp file, and does not rewrite shard 0.
- `test_raising_n_adds_shards_on_top_of_the_existing_ones`: n=100 then n=200 is identical to n=200 directly; also checks `load_deals(n=150)` as a prefix.
- `test_generation_refuses_to_mix_seeds_or_shard_sizes`.
- `test_load_deals_rejects_a_bucket_table_mismatch`: a refit table, and a table with one changed byte but the same counts.
- `test_load_deals_rejects_mixed_shards`: a foreign seed's shard; an empty path.
- `test_rebucket_equals_fresh_bucketing`:
  - rebucket to a 40/20/20/30 refit (seed 0, same designs), then compare with `t.bucket()` on every row;
  - the rebucketed shards are identical to a fresh generation with that table, every array and the meta included;
  - an idempotent second run; rebucketing back gives the original shards.
- `test_rebucket_refuses_a_table_with_other_feature_designs`: a seed-1 fit is refused and the files are untouched.

Mutation probes (a throwaway pytest plugin, no edits to the code). Every probe was caught:

| probe | tests failed |
|---|---|
| `<` for `<=` in the vectorized equity cut | the edge test, plus the generation self-check, which raised in every fixture |
| showdown sign flipped | 3 |
| features rounded through float32 | 3 |
| resume regenerates everything | 3 |
| seats swapped in buckets | 4 |

## TDD evidence

RED, written before the module existed. The failure was expected: `train/gen_deals.py` did not exist yet.

    $ .venv/bin/python -m pytest -q tests/test_gen_deals.py
    E   ImportError: cannot import name 'gen_deals' from 'train' (/home/kchen/code/heads-up-bundle-cfr/train/__init__.py)
    1 error in 0.18s

GREEN:

    $ .venv/bin/python -m pytest -q tests/test_gen_deals.py --durations=20
    14 passed in 3.62s

After the final edits, pinned to CPUs 8-15:

    14 passed in 4.62s

Full default suite before the commit, `taskset -c 8-15 .venv/bin/python -m pytest -q`:

- First run: `1 failed, 295 passed, 8 deselected in 52.81s`.
  - The failure was `tests/test_judge_fidelity.py::test_timeouts_match_the_judge[slow-every-9th]`, which tests the judge with a 20 ms action timeout, so it is timing jitter. It is not in my diff.
  - Run alone right after, it gave `2 passed`.
- Second full run: `296 passed, 8 deselected, 1 warning in 58.20s`. The warning is the existing starlette/httpx deprecation.

## Measurements

All runs used `taskset -c 8-15 nice -n 19`. CPUs 8-15 are 4 physical cores with SMT; Track A was on 0-7.

Throughput (`--shard-size 5000`, wall time including Pool start-up):

| workers | deals | wall | deals/s |
|---|---|---|---|
| 1 | 10,000 | 12.5 s | 803 |
| 4 | 40,000 | 14.2 s | 2,823 |
| 8 | 80,000 | 21.3 s | 3,757 |

About 1.25 ms of one core per deal. T16's Mac figure implied about 770 deals/s per core, so this matches.

2M acceptance run, `--n 2_000_000 --workers 8 --iters 1000`:

- Wall: 510.9 s (3,916 deals/s). User CPU 3,980 s; GNU time max RSS 127 MB.
- Disk: 40 shards of 6.6 MB, 264.2 MB total (132 B/deal), in `train_data/deals_00000..00039.npz`.
- Checks on the finished 2M:
  - 2,000 random rows, 8 buckets each vs `ab.bucket()` and showdown vs `omaha_score`: 0 mismatches.
  - Vectorized rebucket on all 2M rows equals the stored buckets (0.78 s).
  - 0 rows with a repeated card.
  - Showdown: button / split / BB = 47.81% / 4.45% / 47.74%.
  - Occupancy: preflop 0.83-1.22, flop 0.92-1.09, turn 0.93-1.05, river 0.55-1.85 (the river nut buckets, as T16 found).

`load_deals` of the 2M (warm page cache):

| what | time | RAM |
|---|---|---|
| buckets + showdown | 0.05-0.07 s | 34 MB of arrays (17 B/deal); peak RSS 67 MB, 33 MB of it interpreter and imports |
| `cards=True` | 0.09 s | 96 MB |
| `cards=True, features=True` | 0.19 s | 291 MB |

20M projection:

- Generation: about 85 min on CPUs 8-15 at 8 workers, or about 45 min on all 16 CPUs of an idle PC.
- Disk: 2.6 GB.
- `load_deals`: about 0.6 s warm, 340 MB arrays, about 375 MB peak RSS.
- Re-bucketing: 20-50 s.

Rebucket speed (2M, CPUs 8-15):

- Rebucketing to a 400/200/200/200 refit (fit_buckets `--per-street 20000 --seed 0`, written to train_data and deleted afterwards) took 2.2 s (wall 2.29 s).
- Rebucketing back to `data/buckets.npz` took 5.0 s (wall 5.14 s; user CPU 1.6 s both ways, so it is disk-write bound).
- That is 1.1-2.5 s per million deals.
- 500 random rows matched the refit's `bucket()` with 0 mismatches.
- While the dataset was on the refit, `load_deals` with the default table was refused.
- The round trip gave bit-identical buckets: sha256 prefix cf86876f64901b63 before and after.
- A second `--rebucket` rewrote 0 of 40 shards.

20M extension: not run. It needs about 77 more minutes on CPUs 8-15, which does not fit the time box. The resume command for the controller is:

    cd /home/kchen/code/heads-up-bundle-cfr && taskset -c 8-15 nice -n 19 /home/kchen/code/heads-up-bundle/.venv/bin/python -m train.gen_deals --n 20_000_000 --workers 8

It adds shards 40..399 on top of the 2M ones.

## Self-review findings (fixed before the commit)

- The `_agree` error text first ended with "A dataset has one seed, shard size and bucket table". That would have made the tests' `match="seed"` and `match="shard size"` pass vacuously, so I removed it. `generate` now gives a per-field hint instead of string matching.
- Pool-initializer exceptions are deferred to the first job, so the Pool cannot respawn workers forever.
- No idle workers are spawned: workers = min(workers, jobs), and a run with no jobs starts no Pool.
- Added a hole-sortedness assertion; no test covered it.
- Fixed a README overclaim: only 40 showdowns go through the engine check-down, every one through `omaha_score`.
- Wrapped the long lines.

## Issues / concerns

- **float64 features instead of float32** (deviation 1 above). The brief allowed float32, but float32 would break the precondition the brief asks to verify. The disk cost is real: 2.6 GB vs 1.6 GB at 20M.
- **`ev.THREE_OF_A_KIND` is restated** in `buckets_from_features`. The equivalent constant in bot.abstraction is private (`_LOWEST_NUT_CATEGORY`). The edge test compares vectorized vs scalar on categories 0..8, so any drift fails it.
- **What the hashes cannot see**: a change to the feature code in bot/abstraction.py. `--rebucket` re-derives `--verify` deals per shard from the cards and raises on any difference. Generation into an existing directory does not re-verify old shards.
- **Rebucketing needs a refit that keeps the designs** (`fit_buckets --seed 0` and the default design sizes). The committed table satisfies this: a seed-0 refit's features hash matched. A refit with another seed means regenerating, about 9 min per 2M here.
- The judge-fidelity timing test flaked once under the full suite (see above). It is unrelated to this change, but worth knowing for Track A.

## Commit

- dc4d6f5 feat(train): training deal dataset, loader and re-bucketing

## Files changed

- `train/gen_deals.py` (new)
- `tests/test_gen_deals.py` (new, 14 tests)
- `train/README.md` (runbook section + 2 stale lines)
- `docs/results/v3.md` (new)
