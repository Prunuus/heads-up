# Offline training runbook

Scripts in `train/` run offline only; the tournament container never imports them.

- Requirements: `requirements-train.txt`, Python 3.12 or newer, on Linux or macOS.
- Run every command from the repo root with the venv's interpreter.
- Each script writes into `data/`, which is committed. Every file there must stay under 100 MB.

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
- New bucket ids invalidate every training deal dataset (Task 17) and blueprint (Task 18) built from the old table. Regenerate both.

### After any refit

    python -m pytest -q tests/test_abstraction.py               # ranges, occupancy, agreement, timing
    python -m pytest -q -m slow tests/test_abstraction.py       # occupancy on 20k fresh spots per street

Commit the new `data/buckets.npz` along with whatever was rebuilt on it.
