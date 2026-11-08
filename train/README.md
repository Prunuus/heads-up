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

    python -m train.fit_buckets --per-street 200000 --iters 2000 --workers 4 --seed 0 --out data/buckets.npz

Measured on the dev laptop (Apple Silicon, 4 performance cores, numpy 2.5):

| step | spots | wall |
|---|---|---|
| features | 600,000 (200k per postflop street) | 32.6 s, 18,400 spots/s on 4 workers, about 4,600 per core |
| validation at 2,000 rollouts | 6,000 | 8.3 s |
| total | | 41.6 s, file 0.46 MB |

Validation rms against 2,000-rollout Monte Carlo was 0.009–0.012 per street, the Monte Carlo noise floor (sd ≤ 0.011), with mean bias ≤ 0.0011.

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
    python -m pytest -q -m slow tests/test_abstraction.py       # occupancy on 20k fresh spots per street

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

### Checks

    python -m pytest -q tests/test_gen_deals.py      # format, engine agreement, determinism, resume, loader, rebucket

The suite takes a few seconds and needs no `-m slow`. It generates small datasets in temp directories. Among other checks, it confirms that every bucket equals `bot.abstraction.bucket()` on the stored cards, that every showdown equals `omaha_score`, and that 40 showdowns equal a check-down played through `game.Game`.
