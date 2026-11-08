# Offline training runbook

Scripts in `train/` run offline only; the tournament container never imports them.

- Requirements: `requirements-train.txt`, Python 3.12 or newer, on Linux or macOS.
- Run every command from the repo root with the venv's interpreter.
- The table scripts write into `data/`, which is committed. Every file there must stay under 100 MB.
- The training deal dataset goes to `train_data/`, which is git-ignored.

## Card abstraction: `train/fit_buckets.py` writes `data/buckets.npz`

`bot/abstraction.py` turns (street, hole, board) into the bucket the CFR blueprint is indexed by. Its module docstring has the definitions and the file format. The fit does four things:

1. Draws the fixed opponent-hole designs behind the card features. Every feature is a deterministic function of the cards, so the runtime bucket is exactly the offline bucket.
2. Cuts the preflop edges at exact quantiles of `data/preflop.npz`, over all 376,992 holes.
3. Computes features for `--per-street` random spots per postflop street and cuts the edges:
   - flop and turn: strength rows, then equity columns within each row;
   - river: equity for the hands that are not the nuts.
4. Checks the features on `--validate` spots per postflop street against `bot.equity`'s Monte Carlo at `--iters` rollouts, and stores the rms errors in the file.

### The committed table

400 buckets a street (ruling R38): preflop 400, flop and turn 20 x 20, river 395 equity buckets + 5 nut buckets.

    python -m train.fit_buckets --per-street 1000000 --preflop 400 --flop-grid 20x20 --turn-grid 20x20 \
        --river 400 --workers 8 --seed 0 --out data/buckets.npz

Measured on the PC (Ryzen 7 9700X, WSL2, numpy 2.5.4), as `taskset -c 8-15 nice -n 19`, sharing CPUs 8-15 with a dataset generation for most of the run:

| step | spots | wall |
|---|---|---|
| features | 3,000,000 (1M per postflop street) | 364.3 s, 8,235 spots/s on 8 workers |
| validation at 2,000 rollouts | 6,000 | 9.3 s |
| total | | 379.6 s, file 0.46 MB |

- The fit kept `--seed 0` and the default design sizes, so the designs, the features sha256 and the validation errors are those of the previous table: deal datasets re-bucket, with no regeneration.
- Validation rms against 2,000-rollout Monte Carlo was 0.009–0.012 per street, the Monte Carlo noise floor (sd ≤ 0.011), with mean bias ≤ 0.0011.
- 1M spots give each flop and turn cell and each river equity bucket about 2,500 fit samples.
- Occupancy (x the uniform share), on all 376,992 holes preflop and on the 10M deal dataset's 20M spots a street postflop: preflop 0.55–1.41, flop 0.87–1.12, turn 0.12–1.93, river equity buckets 0.20–2.18. The river's nut buckets hold 0.55–1.84% of hands each.
- Below 0.4x are four buckets next to runs of tied feature values: turn cell 398 (0.12x) and river buckets 392, 367 and 365 (0.20, 0.30, 0.37x). `fit_edges` never splits a run, and at 400 buckets some runs outweigh a bucket: in the turn's top strength row, 9.6% of the hands have equity exactly 1.0 (1.9 cells' worth), and some river equity values hold 1.3–2.3 buckets' worth. At 100 buckets these runs were under half a bucket. The occupancy tests record the four as known failures (`FLOOR_FAILS` in `tests/test_abstraction.py`).
- The previous table, the plan's 200/100/100/100, was `--per-street 200000 --iters 2000 --workers 4 --seed 0`: 41.6 s on the dev laptop (Apple Silicon, 4 performance cores), 0.46 MB.

### On a big machine

Run the same command with `--workers` set to the core count. The workers come from a `spawn` Pool, so the script behaves the same on Linux and macOS.

- Throughput scales with cores. The features cost about 0.2 ms per spot per core, so `--per-street 2000000` on 64 cores takes roughly 30 s plus Pool start-up.
- Beyond about 200k spots per street the edges barely move: each 10 x 10 cell already holds about 2,000 samples. A bigger sample mostly helps if you add buckets.
- The output depends only on `--seed`, the sample sizes and the numpy version, never on `--workers`. Every chunk of spots has its own seed `(seed, street, chunk)`, and `tests/test_abstraction.py` checks that 1 and 2 workers write identical files.

### More buckets (ruling R26)

The bucket counts live in the file, and `bot.abstraction.NUM_BUCKETS` reads them at import. To refit with more buckets:

    python -m train.fit_buckets --per-street 1000000 --preflop 400 --flop-grid 20x10 \
        --turn-grid 20x10 --river 200 --workers 64 --seed 0 --out data/buckets.npz

- The flop and turn counts are rows x cols.
- The river's top 5 buckets are always the nut categories, so `--river 200` leaves 195 equity buckets.
- Runtime cost does not depend on the bucket counts. It depends only on the design sizes, the constants at the top of `train/fit_buckets.py`.
- New bucket ids invalidate every training deal dataset (Task 17) and blueprint (Task 18) built from the old table. Re-bucket the dataset (below: seconds, if the refit kept `--seed 0` and the design sizes), then retrain the blueprint.

### After any refit

    python -m pytest -q tests/test_abstraction.py               # ranges, occupancy, agreement, timing
    python -m pytest -q -m slow tests/test_abstraction.py       # occupancy on 200 fresh spots per postflop bucket

- The occupancy tests size their fresh sample to the table's counts (per bucket: 60 postflop spots by default or 200 slow, 500 preflop hands or 1,000 slow), so the [0.4x, 2.5x] rule means the same at any count.
- The river's 5 nut buckets hold the nuts by category, a mass the game sets: about 0.6% (straight flush) to 2% (quads) of river hands each, whatever the count. At 400 river buckets that is 2-8x a uniform bucket, so instead of the [0.4x, 2.5x] rule each must hold 0.4-2.5% of river hands, the band the rule gave them at the plan's 100.

Commit the new `data/buckets.npz` along with whatever was rebuilt on it.

## Training deals: `train/gen_deals.py` writes `train_data/deals_<k>.npz`

The MCCFR trainer (Task 18) samples one deal per iteration, so this dataset is the game the blueprint is solved for. `train_data/` is git-ignored: the dataset is rebuilt from its seed, never committed. Each row is one uniform random deal (the module docstring has the format):

- `cards` int8 [15]: the button's hole, the big blind's hole, then the board.
- `buckets` int16 [2, 4]: seat (0 = button) x street, exactly `bot.abstraction.bucket()` on those cards.
- `showdown` int8: +1 the button wins, 0 split, -1 the big blind wins (`evaluator.omaha_score`).
- `features`, `river_nut`, `river_category`: the raw inputs of `BucketTables.bucket_from_features`, kept so that a refit is a re-bucketing, not a regeneration. They are float64 because float32 is not exact for the flop and turn equities. A deal on a bucket edge would move down a bucket.

Shards hold 50,000 deals each (`--shard-size`). Each one records the seed, its shard number, the sha256 of the bucket table, and the sha256 of the inputs its features depend on: the table's designs and `data/preflop.npz`.

### The acceptance dataset

    taskset -c 8-15 nice -n 19 python -m train.gen_deals --n 2_000_000 --workers 8 --iters 1000

- `taskset` and `nice` keep the job off logical CPUs 0-7 while timing-sensitive work runs there (ruling R32). On an idle machine, drop them and use every CPU.
- `--iters` is accepted and ignored: the bucket features are deterministic (Task 16).
- `--workers` defaults to the CPUs the process may use, so it respects `taskset`.

Measured on the PC (Ryzen 7 9700X, WSL2), pinned to logical CPUs 8-15 (4 physical cores with SMT):

| workers | deals/s |
|---|---|
| 1 | 803 |
| 4 | 2,823 |
| 8 | 3,757 (3,916 on the 2M run) |

- A deal costs about 1.25 ms of one core: 8 feature calls through `bot.abstraction`, plus 2 `omaha_score` calls.
- 2,000,000 deals: 511 s wall, 40 shards of 6.6 MB, 264 MB in all (132 bytes a deal).
- `load_deals` of the 2M takes 0.05-0.07 s from a warm page cache. It holds 34 MB of arrays (17 bytes a deal), with peak RSS 67 MB.
- 20M projection, from the same per-deal numbers:
  - generation: about 85 min pinned to CPUs 8-15, or about 45 min on all 16 CPUs of an idle PC (`--workers 16`, if SMT scales the same over 8 cores);
  - 2.6 GB on disk;
  - `load_deals` about 0.6 s warm, 340 MB of arrays.

### Interrupt, resume, extend

- Kill the job at any time. Shards land atomically (a temp file, then a rename). Rerun the same command and it skips the finished shards, deletes leftover temp files and makes the rest.
- Raising `--n` later adds shards on top of the existing ones. The first 2M deals of a 20M dataset are the 2M dataset.
- The output depends only on `--seed`, `--shard-size` and the numpy version, never on `--workers`. Shard k draws from the rng stream `(seed, 17, k)`, and `tests/test_gen_deals.py` checks that 1 and 2 workers write identical shards.
- A run refuses a directory whose shards have another seed, shard size or bucket table. Use another `--out`, or re-bucket first.

### Loading (Task 18)

    from train.gen_deals import load_deals
    d = load_deals("train_data")       # d.buckets int16 [N, 2, 4], d.showdown int8 [N], d.meta

- `load_deals` concatenates the shards into contiguous arrays. It reads only `buckets` and `showdown` unless you pass `cards=True` or `features=True`, and `n=` takes a prefix.
- It refuses a gap (an unfinished run), shards that disagree (seed, shard size, table), and a dataset bucketed with a table other than `buckets_table` (default `data/buckets.npz`, compared by sha256).
- It also refuses a dataset whose features came from other designs or another `data/preflop.npz` (`features_sha256`). A changed preflop table with an unchanged bucket table would otherwise pass, and every preflop bucket would differ from the runtime's.

### After a refit of `data/buckets.npz`

    python -m train.gen_deals --rebucket                           # against data/buckets.npz
    python -m train.gen_deals --rebucket --buckets path/to/other.npz

- Re-bucketing recomputes `buckets` from the stored features with the vectorized `buckets_from_features` and rewrites each shard atomically with the new hash and counts. Shards already on the table are skipped, so an interrupted re-bucket resumes.
- It needs the features to still be valid. The refit must keep the designs: the same fit `--seed` (0) and the design sizes in `train/fit_buckets.py`. Otherwise the features sha256 differs and `--rebucket` refuses. Regenerate into a new `--out` instead.
- `--verify` deals per shard (default 4) are recomputed from their cards with `bot.abstraction`. Their features and new buckets must match exactly, which catches a change to the feature code (the hashes cannot see code).
- Measured on the 2M dataset (CPUs 8-15):
  - re-bucketing to a 400/200/200/200 refit took 2.2 s, and back 5.0 s: 1-2.5 s per million deals, bound by disk writes;
  - the round trip gave back bit-identical buckets;
  - 500 random deals matched the refit's `bucket()` on their cards.
- Re-bucket a copy (`cp -r`, not hard links) when the original must stay on its table. Measured on the 10M dataset's copy `train_data/d10m-b400`, for the committed 400-a-street table:
  - `--verify 100` (20,000 deals recomputed from their cards) took 47.8 s, `taskset -c 8-15 nice -n 19`;
  - 3,000 more random deals matched `bucket()` on their cards;
  - `load_deals` takes the new table and refuses the old one;
  - every bucket holds at least 3,000 deals per seat (the thin turn cell), and the average is 25,000.

### Checks

    python -m pytest -q tests/test_gen_deals.py      # format, engine agreement, determinism, resume, loader, rebucket

The suite takes a few seconds and needs no `-m slow`. It generates small datasets in temp directories. Among other checks, it confirms that every bucket equals `bot.abstraction.bucket()` on the stored cards, that every showdown equals `omaha_score`, and that 40 showdowns equal a check-down played through `game.Game`.

## Blueprint training: `train/mccfr.py` (external-sampling MCCFR)

The trainer solves the abstract game of the deal dataset: chance draws a row uniformly, then both players bet on the exact tree (`bot/tree.py`), each seeing only its own bucket on the current street. The module docstring has the algorithm, the array layout, the thread semantics and the checkpoint format.

- Iteration t samples a row and the traverser `t % 2`, then makes one external-sampling walk. Regret matching uses positive regrets, and linear weighting (Linear CFR) weighs iteration t by t + 1 in both the regrets and the average.
- The arrays are `regret` and `avg`, float32 `[n_info, 4]`. Node n owns rows `offset[n] .. offset[n+1]-1`, one per bucket of its street. The counts come from the dataset (`NUM_BUCKETS` of `data/buckets.npz`): 1,601,400 infosets at 200/100/100/100. Each infoset costs 32 bytes, plus 8 with `--count-visits`.
- Threads share the arrays without locks (hogwild). Each thread alternates the traverser; threads that each served one player oscillated instead of converging (module docstring). With `--threads 1` a run is deterministic.
- Outputs, all atomic and all the same format:
  - a checkpoint `<out stem>.ckpt-<iterations>.npz` every 10%, on SIGINT/SIGTERM, and when `--max-hours` runs out;
  - the final state in `--out`.
  - Each one carries the tree hash, the bucket-table and features hashes, the counts, the game constants, the iterations, seed, linear flag and the dataset identity. `load_checkpoint` refuses a mismatch with the current tree or table (Task 19 exports from it).
- At each checkpoint it logs a diagnostic: the self-play value, both best responses, and the exploitability (bb/hand) of the average strategy.
  - It is exact on a perfect-recall dataset used whole (the tests).
  - On the real dataset it is approximate: a greedy best response on `--eval-rows` (default 8,192) fixed sampled rows. Under imperfect recall it is not the exact best response, and with few rows per infoset it mostly exploits the sample's noise.
  - Measured on the 2M dataset from 30M to 300M iterations: 3.3-3.5 bb/hand on 8,192 rows, 0.84-0.90 on 65,536 and 0.29 on 262,144 (76 s), flat throughout.
  - Treat it as a sanity monitor: it, and the self-play value, should stay finite and steady. It is not a convergence measure; the Task 21 LBR probe measures the blueprint.

### The long run

    mkdir -p train_data/mccfr-b400
    nohup taskset -c 8-15 python -m train.mccfr --deals train_data/d10m-b400 --iterations 24_000_000_000 \
        --out train_data/mccfr-b400/blueprint_raw.npz --seed 0 --threads 8 --count-visits \
        --max-hours 6 > train_data/mccfr-b400/train.log 2>&1 &

The plan's floor of 200M iterations takes about 5 minutes at the measured rates. A 6-hour run is 10-30 billion iterations, so give it `--max-hours 6` and an iteration target it will not reach before that. The surplus goes to finer buckets: the committed table has 400 a street (sizing below).
- The dataset must be bucketed with the committed table. `train_data/d10m-b400` is a copy of the 10M dataset re-bucketed to it (the "After a refit" section above):

      cp -r train_data/d10m train_data/d10m-b400
      python -m train.gen_deals --rebucket --out train_data/d10m-b400 --verify 100

- A 2-minute smoke run of this command (6,380,800 infosets; 8 threads on CPUs 8-15, normal priority, load average 6-11) made 72.3M iterations in 120 s of training, 600k it/s, in line with the 200/100/100/100 table (643k it/s over its 300M-iteration verification run below). Peak RSS was 1.5 GB. Checkpoints are 255 MB and took 0.3-1.6 s to write; each diagnostic took 1.3-2.5 s. Traverser visits were 31.9 per iteration.
- Detached with `nohup ... &`, the run survives the terminal. `tmux new -s mccfr` then the same command without `nohup`/`&` works too (detach with Ctrl-b d, reattach with `tmux attach -t mccfr`).
- `--threads` defaults to the CPUs the process may use, so `taskset -c 8-15` gives 8 threads. Use 16 only when nothing timing-sensitive shares the PC.
- `--max-hours` caps the training time of the run, summed over resumes. When it runs out, the trainer writes a checkpoint and the final output at the iterations done.
- `--count-visits` stores traverser visits per infoset in every checkpoint, so Task 19 and the sizing below can see which infosets are under-trained.

Monitor:

    tail -f train_data/mccfr-b400/train.log             # a progress line a minute, checkpoint lines every 10%
    ls -la train_data/mccfr-b400/                       # checkpoints so far
    ps -o pid,etime,pcpu,args -C python | grep train.mccfr

A progress line gives the iterations done, the rate over the last minute and since the start, the elapsed training time and the ETA. A checkpoint line follows its diagnostic.

Stop and resume:

    pkill -TERM -f train.mccfr      # or Ctrl-C in tmux: checkpoints the current block, then exits (143, 130)

- To resume, rerun the identical command; `--threads` may differ (for example 16 without `taskset` once nothing timing-sensitive shares the PC), it is not part of the run's identity. It continues from the newest of the checkpoints and `--out`, so it continues the iteration count. The per-iteration rng streams and the linear weights then continue exactly: with one thread, an interrupted-and-resumed run is bit-identical to an uninterrupted one (`tests/test_mccfr.py`).
- A larger `--iterations` extends a finished run.
- A different seed, linear flag, `--count-visits` or dataset (rows, seed, table, features) is refused: use another `--out`.

### Throughput (PC: Ryzen 7 9700X, WSL2; 2M dataset, 200/100/100/100 buckets)

The kernel is memory-latency-bound: random reads and writes into 49 MB of arrays plus the dataset. Its rate depends on how loaded the PC's shared L3 and memory are, as well as on the thread count. Two sessions, minutes apart:

| threads | CPUs | light load (load avg 12-15) | heavy load (load avg 18-22), from one common 18.7M-iteration state |
|---|---|---|---|
| 1 | 8-15 | 104k it/s | 71k it/s |
| 4 | 8-15 | 553k it/s | 200k it/s |
| 8 | 8-15 | 1.61M it/s | 351k it/s (397k with `--count-visits`) |
| 16 | all | 1.10M it/s (CPUs 0-7 busy with Track A) | 742k it/s |

- The same code in pure Python (`NUMBA_DISABLE_JIT=1`) runs 649 it/s. Numba is required: the plan's rule is numba below 20k it/s.
- `--count-visits` costs nothing measurable.
- The rate also drifts as training proceeds, because the walk's length depends on the strategies. The verification run below gives the rate over a long horizon.
- Bucket counts barely matter: with every bucket split four ways (800/400/400/400, 6.4M infosets, 195 MB), 8 threads ran 311k it/s, against 347k and 270k for the default table measured before and after it.
- Numba compiles the kernels on first use (about 2 s) and caches them in `train/__pycache__`.

### The verification run (2M dataset, default table)

    taskset -c 8-15 python -m train.mccfr --deals train_data --iterations 300_000_000 \
        --out train_data/mccfr-verify/blueprint_raw.npz --seed 0 --threads 8 --count-visits \
        --max-hours 0.30 --log-every 60

- 300M iterations in 7 min 46 s of training: 643k it/s on average, 769k it/s by the end. 8 threads ran on CPUs 8-15 at load average 19-25.
- Nine checkpoints and the final output: 64 MB each with visits, 0.1-0.2 s to write; each diagnostic took 1.5-1.8 s.
- Traverser visits per iteration: 1.8 preflop, 5.6 flop, 9.9 turn, 15.0 river. Every infoset was visited.
- The visit-weighted median visits per infoset (and its 10th percentile) after 300M iterations:

  | street | median | 10th percentile |
  |---|---|---|
  | preflop | 255k | 28k |
  | flop | 97k | 11k |
  | turn | 35k | 4.2k |
  | river | 23k | 2.2k |

  Visits per infoset scale as iterations / buckets. Use that to size a run.
- The average strategy is sensible already:
  - the button folds preflop bucket 0, and raises buckets 150-199 at the root 95-97% of the time;
  - facing a pot raise, the big blind folds 0 and 50, calls 150, and re-raises 199 94% of the time;
  - at the checked-down river node, the nut buckets bet and the weak ones check.

### Checks

    python -m pytest -q tests/test_mccfr.py             # engine payoffs, exact best response, convergence, checkpoints
    python -m pytest -q -m slow tests/test_mccfr.py     # 8-thread convergence, the plan's behaviour tests, SIGTERM + resume

The default tests take about 10 s. The key one trains on a perfect-recall game: three hand classes, the same bucket on every street. On that game the best response is exact. The test asserts that the exploitability falls below a measured threshold, on one thread (falling at every checkpoint) and on two (hogwild). It falls about as 1/sqrt(T): 0.194 bb/hand at 200k iterations, 0.032 at 2M, 0.010 at 20M. Either threading bug the trainer had during development fails the two-thread case (module docstring of `_run_block`).
