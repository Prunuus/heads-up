# Task 16 report: card abstraction (buckets)

Two commits on `kevin-cfr` in the worktree `/Users/kchen/Desktop/code/FALL26/heads-up-bundle-cfr`, base `fcde753`. Nothing is pushed.

- `9ae259d` `feat(bot): 2-D potential-aware card abstraction`
- `d93b93a` `test(bot): a bucket count that disagrees with its grid is rejected` (self-review follow-up)

New files:
- `bot/abstraction.py` (481 lines)
- `train/fit_buckets.py` (287)
- `train/README.md` (59)
- `tests/test_abstraction.py` (485; 61 default tests + 4 slow)
- `data/buckets.npz` (0.46 MB)

No existing file was touched.

## The main decision: deterministic features, so runtime bucket == offline bucket

The plan's design computes bucket features with Monte Carlo: offline at 2,000 rollouts, at runtime with whatever the time manager allows. I measured that design and it cannot meet the brief's agreement test. I replaced the Monte Carlo with deterministic features computed by the same code offline and at runtime, so the two buckets are equal by construction.

**What the plan's MC bucketing would get.** Same final 10 x 10 grid, 3,000 fresh spots per street, runtime MC at `iters` vs offline MC at 2,000:

| iters | flop exact | flop same-or-adjacent | turn exact | turn same-or-adjacent |
|---|---|---|---|---|
| 0 (zero-budget estimates) | 0.036 | 0.149 | 0.037 | 0.188 |
| 32 | 0.138 | 0.501 | 0.143 | 0.446 |
| 64 | 0.209 | 0.612 | 0.192 | 0.540 |
| **128** | **0.287** | **0.714** | **0.277** | **0.642** |
| 256 | 0.378 | 0.809 | 0.346 | 0.741 |
| **adopted, every iters** | **1.000** | **1.000** | **1.000** | **1.000** |

- "Adjacent" means a king's move on the grid: |Δrow| ≤ 1 and |Δcol| ≤ 1.
- The plan's design fails the brief's ≥ 80% at iters=128 on both streets.
- Even offline MC-2000 vs the near-exact features agrees only 0.648 exact / 0.934 adjacent (flop) and 0.647 / 0.940 (turn).
- An intermediate prototype with exact strength rows and MC equity columns reached 0.77 (flop) / 0.73 (turn) at 128, still below 80%.
- The equity columns are fitted within a strength row, so they are narrow: about 0.06 equity wide on the flop. MC noise at 128 rollouts is about 0.04 rms.

**Recommended minimum runtime iters: none.**
- `bucket(street, hole, board, iters, rng)` keeps the plan's signature but ignores `iters` and `rng`, and never touches `rng`.
- iters=0, the TimeManager's bank-exhausted answer, returns exactly the offline bucket, never raises, and stays in range. Tests cover this, including a real exhausted TimeManager.
- This is in `bucket()`'s docstring and the module docstring.
- The cost is fixed (see Timing) and fits in the judge's free 1 ms per request on the dev laptop.

**How the features stay exact and cheap**
- *Pair values.* Each unseen pair's best 2+3 hand is scored once per board. An opponent hole's made hand is then the max over its 10 pairs, which is 10 gathers and no scoring.
- *Scoring.* A 0.7 MB pair-class x triple-class table is built at load from `fasteval.score5_batch`. With fancy indexing it is 3.5x faster than gathering from fasteval's 15.6 MB table with `np.take`.
- *strength* (= `strength_now`'s quantity): our made hand against fixed, balanced designs of opponent holes. There are 4,096 holes on flop and turn and 16,384 on the river, mapped onto the spot's unseen cards in ascending order.
- *Turn equity*: exact over the 27 river cards, with 304 design holes per river card (8,208 rows).
- *Flop equity*: exact over all 378 (turn, river) runouts, with 8 design holes each (3,024 rows), evaluated only at the cells the design reads.
  - A pair's full-board hand is max(pair+flop, pair+one runout card+2 flop cards, pair+both runout cards+1 flop card).
  - The first two terms are shared across cells, so a cell costs 3 lookups.
- *Nut flag*: exact, `ours >= max over all unseen pairs`, the same as `nut_flags` bit 0. A test checks it against `nut_flags` on 750 spots.

**Accuracy** (committed designs, 100 random spots per street):

| street | strength vs exact enumeration of every opponent hole | equity vs MC 40k rollouts (MC sd 0.0025) |
|---|---|---|
| flop | rms 0.0039, max 0.015, mean −0.0004 | rms 0.0067, max 0.019, mean −0.0001 |
| turn | rms 0.0048, max 0.013, mean +0.0005 | rms 0.0036, max 0.010, mean +0.0003 |
| river | rms 0.0027, max 0.009, mean +0.0007 | = strength |

The fit's own validation, 2,000 spots per street against 2,000-rollout MC (whose sd is ≤ 0.0112), found:
- strength rms 0.0097 / 0.0096 / 0.0086 (flop / turn / river);
- equity rms 0.0118 / 0.0102 / 0.0088;
- |mean| ≤ 0.0011.

That is the MC noise floor.

## Bucket definitions (adopted)

- **Preflop (200).** Quantile buckets of `preflop_strength`. The edges are cut on all 376,992 holes from `data/preflop.npz`, exactly, with no sampling. The runtime value is the same float32 table entry, so preflop matches exactly.
- **Flop and turn (10 x 10).** bucket = row·10 + col.
  - row = strength decile.
  - col = equity decile **among that row's hands**: each row has its own 9 equity edges.
  - This is the controller's suggested "equity quantiles conditioned on strength". It is the plan's grid in spirit and passes the occupancy rule.
- **River (100).**
  - Hands that are not the nuts: 95 equity quantile buckets, 0–94.
  - The nuts (nut_flags bit 0, ties count) take 95–99 by category: 95 trips, 96 full house, 97 flush, 98 quads, 99 straight flush.
  - Trips beat a straight in this game, and the opponent can always make some set. So the river nuts are never below trips; a lower category would clamp to 95.
- **Edges.** A value equal to an edge goes to the upper bucket. This is `searchsorted(side="right")`, implemented with `bisect_right` on tuples, and it is always in range.
  - `train/fit_buckets.fit_edges` cuts at the boundary between distinct values nearest each quantile.
  - A run of tied values is never split; it goes whole to the nearer side. A plain quantile cut would leave buckets empty when there are atoms (tested).
- **Configurable (R26).**
  - `num_buckets`, `flop_grid` and `turn_grid` live in `data/buckets.npz`.
  - `NUM_BUCKETS` is a read-only mapping set at import from the file's metadata. Without a readable file it falls back to `DEFAULT_NUM_BUCKETS` = 200/100/100/100, and never raises at import.
  - The fit CLI takes `--preflop N --flop-grid RxC --turn-grid RxC --river N`.

## Measured alternatives (the "quality improvement if measured" comparison)

Grid definitions use the deterministic features, fitted on 30k spots and evaluated on a fresh 30k per street. Variance is within-bucket variance divided by total variance, so lower is finer; "potential" is equity − strength. Occupancy is the per-bucket count divided by the uniform expectation (expected 300 per bucket here).

| definition | flop occupancy | flop var: equity / strength / potential | turn occupancy | turn var: equity / strength / potential |
|---|---|---|---|---|
| G1, plan literal: independent deciles of s and e | **0.00–7.76** | 0.0216 / 0.0099 / 0.0358 | **0.00–8.37** | 0.0127 / 0.0085 / 0.0735 |
| **G2, adopted: s rows, e deciles within row** | 0.75–1.25 | **0.0082** / 0.0098 / 0.0279 | 0.84–1.21 | **0.0041** / 0.0082 / 0.0561 |
| G3: e rows, s deciles within row | **0.00–4.59** (ties in s) | 0.0199 / 0.0089 / 0.0350 | **0.00–4.60** | 0.0116 / 0.0056 / 0.0644 |
| G5: s rows, potential deciles within row | 0.82–1.22 | 0.0338 / 0.0096 / 0.0067 | 0.79–1.23 | 0.0194 / 0.0091 / 0.0134 |
| G6: 20 s rows x 5 e cols | 0.80–1.21 | 0.0239 / 0.0025 / 0.0241 | 0.80–1.19 | 0.0101 / 0.0023 / 0.0419 |
| G4: 1-D equity, 100 quantiles (EHS only) | 0.82–1.21 | 0.0006 / 0.3938 / **0.8932** | 0.80–1.26 | 0.0003 / 0.1529 / **0.8206** |

- G1 (the plan's literal grid) and G3 fail the occupancy rule.
- Among the 2-D grids with uniform occupancy, G2 has the finest equity resolution, 2.6–3x finer than G1, G5 and G6.
- 1-D equity (G4) barely separates potential at all (0.89 / 0.82 of its variance remains within buckets).
- Correlation of strength and equity: flop 0.767, turn 0.915.

**River nut split.**
- Nut fraction 6.14%; 82.5% of nut hands have equity exactly 1.0.
- Splitting the 5 nut buckets by equity quantile gives occupancy [1.07, 0, 0, 0, 5.07] x expected, which fails.
- By category: trips 1.16, full house 1.38, flush 1.04, quads 1.97, straight flush 0.58 x expected.

**Feature estimators** (prototype numbers, flop):
- Plain MC at 128 rollouts: rms 0.040, 0.19 ms.
- MC plus a strength control variate: rms 0.038 (flop) and 0.0325 (turn) for +35–50% cost. Rejected: the same gain as just adding rollouts.
- Runout designs with balanced matchings: K=56 → 0.020 at 0.27 ms; K=112 → 0.018 at 0.36 ms; K=224 → 0.010 at 0.56 ms.
- Exact over all 378 runouts:
  - rank-class version: 0.0055 at 0.85 ms;
  - cell version, 4/8/16 holes per runout: 0.0091 at 0.26 ms, **0.0055 at 0.35 ms** (adopted, 8; final implementation 0.25 ms), 0.0044 at 0.55 ms.
- The cell version gives identical numbers to the rank-class version.
- Strength designs (rms vs exact): 1,024 rows ≈ 0.010; 4,096 ≈ 0.005; 8,192 ≈ 0.0035. Balanced blocks beat iid rows (0.0049 vs 0.0053 at 4,096).

## Fit: command, wall time, throughput

    python -m train.fit_buckets --per-street 200000 --iters 2000 --workers 4 --seed 0 --out data/buckets.npz

- Features: 600,000 spots (200k per postflop street) in 32.6 s, **18,382 spots/s on 4 workers**, about 4,600 per core.
- Validation: 6,000 spots at 2,000 rollouts in 8.3 s.
- **Total 41.6 s wall.** Output 0.46 MB.
- The brief's full size (200k per street) is the local fit, so nothing was scaled down.

How the script works:
- It uses a spawn Pool with an initializer that sends the designs once.
- Seeds are deterministic per chunk: `(seed, street, chunk)`. The output does not depend on `--workers`; a test compares the files written with 1 and 2 workers.
- `--iters` and `--validate` drive only the MC validation. Its results are stored in the file as `validation` [3, 4].
- `train/README.md` has the runbook, including big-machine notes and bucket-count refits with exact commands. Task 18 extends it.

## Occupancy (x the uniform expectation)

| sample | preflop | flop | turn | river |
|---|---|---|---|---|
| exact population (all 376,992 holes) | 0.83–1.21 | — | — | — |
| fit sample (200k) | — | 0.95–1.05 | 0.99–1.00 | 0.53–1.87 |
| fresh 20k, slow test | 0.68–1.39 | 0.81–1.26 | 0.84–1.18 | 0.58–1.94 |
| default suite: 20k preflop, 6k postflop | 0.74–1.38 | 0.72–1.32 | 0.72–1.32 | 0.72–1.73 |

All are within the brief's [0.4, 2.5].
- River extremes are the nut categories: straight flush 0.58, quads 1.9–2.0.
- Preflop ties: 2,216 distinct values; the largest tie is 648 holes, 0.34 of a bucket.

## Timing (dev laptop; p99 over 400 spots, each spot's cost = best of 3 calls, iters=128)

| street | p50 | p99 |
|---|---|---|
| preflop | 0.002 ms | 0.003 ms |
| flop | 0.247 ms | 0.298 ms |
| turn | 0.150 ms | 0.164 ms |
| river | 0.091 ms | 0.116 ms |

- With the other agent's 9-worker job running (load average 23): flop 0.355 / 0.893 ms, turn 0.215 / 0.486, river 0.127 / 0.283.
- Single-call wall p99 reached 3.6 ms under that load, which is why the test takes the best of 3 per spot. Preemption is filtered out, but a spot that is slow every time still counts.
- Load: import 10 ms, including the NUM_BUCKETS metadata read. `load_buckets()` takes 27 ms, building the score table and checking the designs. The tables hold 5.0 MiB live, 8.3 MiB peak.

## TDD evidence

1. **RED 1.** Wrote `tests/test_abstraction.py` first. Collection failed with `ImportError: cannot import name 'abstraction' from 'bot'`.
2. **RED 2.** Wrote the module and the fit script, with no table yet: 49 failed, 2 passed. Failures were `FileNotFoundError: card abstraction table not found: .../data/buckets.npz (build it with: python -m train.fit_buckets)`.
   - The fit-script test also failed on a test bug: `np.array_equal` treats the NaN "not validated" marker as unequal. Fixed with `equal_nan=True`.
3. **GREEN.** Ran the fit: 51 passed, 4 deselected (6.7 s). The 4 slow tests passed.
4. **RED 3.** Added corrupt-table tests: the 4 design-integrity cases failed. The 5 cases for table validation that already existed passed.
5. **GREEN 2.** Added the design-integrity check: 60 passed.
6. **Mutation check, `bot/abstraction.py`: 17/17 caught**, each applied and restored byte-identical:
   - edge to the lower bucket; rows/columns swapped; one equity edge set for all rows;
   - nut categories off by one; river ignores the nut flag; ties count as losses;
   - flop without the both-runout-cards term; flop ours ignores runouts; turn opponent ignores the old board; turn equity = strength;
   - strict nut flag; no duplicate-card check; no cache; NUM_BUCKETS always defaults;
   - writeable edges; strength on the wrong design; river card allowed in the hole.
   - The occupancy tests act as a distribution-shift detector: any feature change moves values off the committed edges.
7. **Mutation check, `train/fit_buckets.py`: 6/6 caught**, after one fix.
   - "River edges include the nuts" was initially MISSED, because the synthetic nut hands had ordinary equity. They now sit at equity 1.0, as real ones do.
8. **Validation lines: 7/7 caught**, after one fix.
   - "No grid check" was initially MISSED, because the 5x10-grid corruption also trips the edge-shape check. The case now corrupts the count instead (50 for a consistent 10x10). That is commit `d93b93a`.
9. **Honesty notes.**
   - `test_fit_cuts_rows_by_strength_then_equity_within_each_row` was written after `fit()`. It was then mutation-verified: it catches 3 of the fit mutations.
   - The 5 existing table-validation cases also came after their code. They are mutation-verified as above.
10. **Full default suite** (worktree, at HEAD `d93b93a`): **282 passed, 8 deselected, 1 warning** (the existing starlette/httpx deprecation) in 33.7 s. `-m slow` on the new file: 4 passed.
11. **Python 3.12:** all three .py files pass `ast.parse(..., feature_version=(3, 12))`.

## Files

- `bot/abstraction.py`, the runtime. Imports: stdlib, numpy, `bot.fasteval`, `bot.preflop`, `game`.
  - Constants: `NUM_BUCKETS`, `DEFAULT_NUM_BUCKETS`, `NUT_BUCKETS`, `FORMAT`.
  - `Features(strength, equity, nut, category)`.
  - `Featurizer(designs)`.
  - `BucketTables`, with `.bucket`, `.features`, `.bucket_from_features`, `.num_buckets`, `.grid` and the read-only edge arrays.
  - `load_buckets(path=None)`, `features()`, `bucket()`, `warmup()`.
  - The module docstring has the definitions, accuracy, cost and the npz format.
- `train/fit_buckets.py`: `make_designs`, `fit_edges`, `fit`, CLI `main(argv)`.
- `train/README.md`: runbook.
- `tests/test_abstraction.py`.
- `data/buckets.npz`.

## Self-review

- **Brief interface kept.** `NUM_BUCKETS`, `bucket(street, hole, board, iters, rng) -> int`, and the four required tests:
  - range/determinism;
  - occupancy at 20k (default runs 20k preflop + 6k postflop; the full 20k-per-street version is `@slow`);
  - agreement ≥ 80% at iters=128 vs 2,000 (actual 100%; the test also asserts exact equality);
  - p99 < 1.5 ms.
- **Deviation from the plan: iters/rng are ignored.** The bucket never samples (see the main decision). That makes "recommend a minimum iters" moot; the docstring says so.
- **R22.** Module state is lazy, read-only data:
  - non-writeable arrays;
  - MappingProxyType for `NUM_BUCKETS`, edges, grids, cuts and position tables;
  - a single cached `BucketTables`.
- **R22 consistency.** If the default file changes after import, `load_buckets()` raises RuntimeError rather than return ids outside `NUM_BUCKETS`.
- **Validation at load.** The loader rejects:
  - a wrong format or counts that disagree with the grids;
  - edge shapes or orders that are wrong, or missing edges;
  - design rows that are not the 10 pairs of one 5-card hole;
  - a river or runout card that also sits in its hole.
- **Input validation.** Bad street, wrong board length, 4 hole cards, out-of-range or duplicate cards all raise ValueError.
- **Self-review fixes.**
  - The commit is a single code-and-data commit, following the preflop-table precedent, so every commit is green.
  - One self-inflicted slip: a chained command committed `63b4ae6` before the docstring edit applied, so the message overstated it. I amended that unpushed, unreported commit to `d93b93a`, which now contains both changes.

## Concerns / notes for later tasks

1. **Plan deviation (iters ignored, deterministic features).** It is strictly better on every measured axis: agreement, accuracy (6–10x lower error than MC-128), cost and occupancy. Still, it changes the plan's model of "runtime bucketing gets whatever iters the time manager allows". Please confirm in review.
2. **Fixed per-call cost, independent of the bank.** About 0.25 ms on the flop. Under bank exhaustion (iters=0) the bucket still costs this, which is within the free 1 ms on the dev laptop. A 3x slower tournament CPU would charge about 0.1–0.4 ms per postflop decision.
   - For Task 20: call `abstraction.warmup()` in startup (about 27 ms plus preflop/fasteval).
   - Cache the bucket per (street, hole, board) within a hand.
   - Consider skipping the blueprint when the bank is critical.
3. **For Task 17.** `bucket()` needs no `--iters` (deterministic). Throughput is about 4,600 postflop spots/s per core.
   - With 6 postflop buckets per deal (2 seats x 3 streets), 2M deals is about 2,600 core-seconds: about 11 min on 4 workers, under 1 min on 64 cores.
   - The deal dataset should record a hash of `data/buckets.npz`.
4. **For Task 19.** Store a hash of `data/buckets.npz`, or of its edges and designs, in the blueprint meta, and verify it at load. Bucket ids are only meaningful with that exact table, the same idea as the Task 15 note about hashing `child`. A refit invalidates datasets and blueprints (README).
5. **River nut buckets are uneven** (0.58x to 1.97x). That is inside the brief's bounds, with less margin than other streets. The split by category is strategically meaningful (board texture) and is the only deterministic split that worked, since 82% of nut hands tie at equity 1.0.
6. **Default suite runs 6k (not 20k) postflop spots** for occupancy, to keep `pytest -q` fast (+7 s for the whole new file). The brief's full 20k-per-street check is the slow test, and it passes.
7. **Timing tests are wall-clock.** Best-of-3 per spot makes them robust to preemption on this shared Mac. Extreme sustained contention could still flake them, like the existing equity timing test.
8. **Refit reproducibility** needs the same numpy version (Generator streams). The runtime never regenerates designs; they are stored in the file.
9. **Strategy validation (R20).** No sim matches are possible yet: nothing consumes buckets until Tasks 17–20. Quality is argued through within-bucket variance, occupancy, agreement and accuracy, as the controller specified for this task.
