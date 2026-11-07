### Task 7: Vectorized equity and strength

**Files:** Create `bot/equity.py`. Test: `tests/test_equity.py`.

**Interfaces — Consumes:** Task 6. **Produces:**
- `equity_vs_random(hole: Sequence[int], board: Sequence[int], iters: int, rng: np.random.Generator) -> float`: win + ½ tie probability, rolling out the opponent's 5 cards and the rest of the board.
- `strength_now(hole, board, iters, rng) -> float`: same, but on the *current* board using `made_now_batch`, with no future cards. On the river it equals equity.
- `nut_flags(hole, board) -> int`, a bitmask:
  - bit0: we hold the current best possible made hand given this board
  - bit1: board pairs (full houses possible)
  - bit2: ≥ 3 board cards of one suit (flush possible)
  - bit3: our flush draw uses the ace of the suit

  Computed exactly by checking all two-card combos from the unseen cards.

- [ ] **Step 1: Write the failing tests**
```python
def test_equity_agrees_with_slow_reference():   # 50 spots, 4000-iter reference, |diff| < 0.03 at iters=2000
def test_river_equity_is_deterministic_given_rng_seed():
def test_nut_flag_royal_example():              # guide example: board Ah Kh Qh Jh 6c, hole Th 7s 7d 8c 9c
    assert not (nut_flags(H, B) & 1)            # a king-high straight is not the nuts here
def test_timing_p99_under_1_5ms_at_128_iters_flop():
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): vectorized equity, strength, nut flags`.

