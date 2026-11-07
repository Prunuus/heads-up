# Task 8 report: preflop strength table

Commit f4dfb8e `feat(bot): preflop strength table`.

Implementation
- bot/preflop.py: canonical_key (min over 24 suit perms, base-36), load_preflop(path=None) (default path relative to repo, FileNotFoundError if missing, no import-time load), preflop_strength (lazy load, mask + searchsorted, KeyError if hand absent).
- train/build_preflop.py: spawn Pool, per-chunk rng seeded [seed, chunk_idx], 3000 rollouts per class, expand to 376,992 masks sorted; npz stores masks int64, strength float32, rollouts, n_classes, seed. train/__init__.py added.
- Build: `python -m train.build_preflop --rollouts 3000 --workers 4 --seed 0`: 19,998 classes, wall 23.8 s. File data/preflop.npz is 1.09 MB (compressed).
- Table: min 0.099, median 0.5003, max 0.6993.
- Lookup median 0.9 us (limit 50 us).

Range finding (R8)
- Min 0.099 is NOT a bug: hands with four of a kind (288 hands, 9*32) range 0.099-0.322. Omaha uses only 2 hole cards, so quads play as one pair. Verified with a 20k-rollout re-estimate of four sixes (0.106).
- Non-quad min is 0.222 (three sixes + low cards; 124 hands < 0.25). Max 0.699.
- So the test asserts: non-quad >= 0.20, all <= 0.75, all >= 0.05, median in [0.45,0.53]. This deviates from the ruled [0.25,0.75]; flagging for the controller.

TDD
- RED: collection ImportError (no bot.preflop); then with module but no data, 3 failed (FileNotFoundError), 2 passed.
- GREEN: after build, full default suite 80 passed, 1 deselected (~20 s).
- Note: first commit attempt was made with the range test still failing (my sed edit failed silently); I amended the commit once the test was fixed, so the final commit is green.

Files: bot/preflop.py, train/__init__.py, train/build_preflop.py, tests/test_preflop.py, data/preflop.npz.

Self-review / concerns
- 3000 rollouts gives ~0.009 std error per class; fine for bucketing.
- Class members share one estimate (by design).
- Quads being low should be handled sensibly by later bucketing (they are just weak hands).
