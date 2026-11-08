# Task 19 report: blueprint export and runtime sampler

Commit (kevin-cfr): 4f71c90 "Add the blueprint export and the runtime sampler."
Files: train/export.py (new), bot/strategy.py (new), tests/test_strategy.py (new), train/README.md (section appended).
data/blueprint.npz NOT committed (default).

## Built
- train/export.py: `export(src, out, tree=None, buckets_table=...)` and `python -m train.export SRC [OUT]`. Reads via mccfr.load_checkpoint (all checks), average_strategy -> float16 [n_info,4], trainer's own row order (node-major, streets interleaved; no per-street slicing, offsets recomputed at load from counts + tree.street). Atomic write; re-loads with Blueprint.load before the rename.
- File stores: probs, stack/sb/bb/cap, iterations, seed, linear, n_dec, n_term, tree_child_sha256, buckets_sha256, features_sha256, buckets (per-street counts from the checkpoint), dataset_* identity.
- bot/strategy.py (stdlib+numpy): `Blueprint.load(path, *, tree=None, buckets_path=default)` verifies format, game constants, n_dec/n_term/tree hash, sha256 of data/buckets.npz, bucket counts vs ab.NUM_BUCKETS, array dtype/shape; ValueError on any mismatch or unreadable file. `Blueprint.try_load` and `strategy.load_default` return None, never raise. `probs(node,bucket)` -> float64[4] (normalized, 0 on illegal; IndexError out of range; wraps easily as (obs,hole)->probs over legal). `act(node,bucket,legal,rng,purify=0.1)`: mask to legal, renormalize, zero < purify, renormalize, sample with rng.random() (numpy Generator or random.Random); falls back to most probable legal (first legal if all zero). Arrays non-writeable, meta is a mappingproxy, no mutable module state.
- `buckets_path=None` skips the table check (synthetic tests).

## TDD
RED: `pytest -q tests/test_strategy.py` -> "ImportError: cannot import name 'strategy' from 'bot'", collection error.
GREEN: 10 passed in 1.64s (after one test typo fix: key name tree_child_sha256).
Tests: load<1s+meta, export == average_strategy per street (float16 tol), read-only, act legal incl. all-mass-illegal, purify=0 chi-square over 10k draws, purify drops small probs, refuses changed tree/stack/table, missing/corrupt never raises via try_load, probs p99 < 50us, export refuses wrong table.

## Real checkpoint (blueprint_raw.ckpt-001555397459.npz, 1.555e9 iterations, ran on CPUs 0-7 nice 19)
- Export: 51.1 MB (uncompressed float16; 6,380,800 infosets x 8 bytes), export took well under a minute.
- Load: 0.016 s with the tree already built (get_tree is cached/warmup); 0.96 s cold in a fresh process including the 0.8 s tree build and imports.
- probs p50 2.3 us, p99 17 us; act p50 7.8 us, p99 72 us (20k random decisions, machine loaded).
- Real meta verified against data/buckets.npz (400/400/400/400). Root: bucket 0 -> [1,0,0,0], bucket 399 -> mostly RAISE_HALF.
- Exported file is in the scratchpad only: /tmp/claude-1000/-home-kchen-code-heads-up-bundle/c0d55f53-d291-4faf-88d2-40523d153176/scratchpad/blueprint.npz

## Concerns
- 51 MB, not the plan's ~13 MB (plan assumed 200 buckets); under the 100 MB limit.
- Cold-process load is ~0.96 s only because of the tree build; call get_tree() in warmup (load itself is 16 ms).
- Infosets the trainer never updated export as uniform over legal actions (no visit mask exported; checkpoints with --count-visits hold visits if T20 wants one).
- Hashed features_sha256 is stored but only checked at export time (needs design arrays, not available at runtime).
- Commit has no trailers per brief, despite a later system reminder suggesting them.
