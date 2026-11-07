### Task 6: Vectorized evaluator (I4)

**Files:** Create `bot/__init__.py`, `bot/fasteval.py`. Test: `tests/test_fasteval.py`.

**Interfaces — Produces:**
- `score5_batch(cards: np.ndarray) -> np.ndarray`: shape (N,5) of int → (N,) int32, **identical** to `evaluator.score_five`.
- `omaha_batch(holes: np.ndarray, boards: np.ndarray) -> np.ndarray`: (N,5),(N,5) → (N,), identical to `omaha_score`.
- `made_now_batch(holes: np.ndarray, boards: np.ndarray) -> np.ndarray`: boards of width 3, 4 or 5. Best hand using exactly 2 hole + 3 board.
- `PAIR_IDX`, `TRIPLE_IDX`: the combination index arrays.

Table: index = base-9 encoding of the five ranks sorted descending, plus a flush bit → `int32[2·9⁵]`. Build it with `evaluator._score_ranks`.

- [ ] **Step 1: Write the failing tests**
```python
def test_score5_matches_reference_on_200k_random():   # exact equality
def test_omaha_batch_matches_reference_on_20k_random():
def test_made_now_on_flop_equals_bruteforce():
def test_speed():   # 10_000 Omaha evaluations in one batch
    t = perf_counter(); omaha_batch(H, B); assert perf_counter() - t < 0.010   # ≥ 10× faster than F5
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): vectorized numpy hand evaluator`.

