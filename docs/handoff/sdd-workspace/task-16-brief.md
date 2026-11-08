### Task 16: Card abstraction (buckets)

**Files:** Create `bot/abstraction.py`, `train/fit_buckets.py`, `data/buckets.npz`. Test: `tests/test_abstraction.py`.

**Interfaces — Produces:**
- `NUM_BUCKETS = {PREFLOP: 200, FLOP: 100, TURN: 100, RIVER: 100}`
- Bucket definitions:
  - preflop: quantile of `preflop_strength`
  - flop/turn: a 10×10 grid of (quantile of `strength_now`) × (quantile of `equity_vs_random`), so made hands and draws separate (Johanson 2013 / Ganzfried 2014 rationale)
  - river: 100 quantiles of equity, with the top 5 buckets reserved for `nut & 1`
- `bucket(street, hole, board, iters: int, rng) -> int`
- Offline `fit_buckets.py`: samples 200k situations per street at 2,000 iterations and writes the quantile edges.

- [ ] **Step 1: Write the failing tests**
```python
def test_bucket_range_and_determinism():
def test_bucket_occupancy_roughly_uniform():          # fresh 20k sample: every bucket within [0.4×, 2.5×] expected
def test_runtime_bucket_agrees_with_offline():        # iters=128 vs iters=2000: same or adjacent bucket ≥ 80% (flop, turn)
def test_bucket_time_p99_under_1_5ms():
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): 2-D potential-aware card abstraction`.

