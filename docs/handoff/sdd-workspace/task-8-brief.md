### Task 8: Preflop strength table

**Files:** Create `bot/preflop.py`, `train/build_preflop.py`, `data/preflop.npz`. Test: `tests/test_preflop.py`.

**Interfaces — Produces:**
- `canonical_key(hole: Sequence[int]) -> int`: invariant under all 24 suit permutations. Map each suit to its lexicographic minimum over permutations, then pack the 5 sorted cards as base-36.
- `preflop_strength(hole) -> float`: equity vs random.
- `load_preflop(path="data/preflop.npz") -> None`, called from warmup.

The offline script enumerates all 376,992 hands, groups them by canonical key, and runs ≥ 3,000 rollouts per class with `bot.equity` (multiprocessing).

- [ ] **Step 1: Write the failing tests**
```python
def test_canonical_invariant_under_suit_permutation():   # 1000 random hands × 24 perms
def test_table_covers_all_hands():                       # every 5-subset of 36 has a key in the table
def test_strength_range():                               # all values in [0.30, 0.70]  (F6 spread)
def test_lookup_under_50us():
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement both files. Run `.venv/bin/python -m train.build_preflop --rollouts 3000 --workers 8` and commit the `.npz` (must be < 100 MB).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat(bot): preflop strength table`.

