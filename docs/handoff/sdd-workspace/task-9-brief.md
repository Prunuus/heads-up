### Task 9: Time manager (I3)

**Files:** Create `bot/timing.py`. Test: `tests/test_timing.py`.

**Interfaces — Produces:**
`class TimeManager(bank_s=8.0, hands=1000, target_s=3.0, clock=time.perf_counter)` with:
- `begin() -> None` and `end() -> None`: wrap each request. They accumulate charged time `max(0, dt − 0.00075)`.
- `iters(pot: int, street: Street, hand_index: int) -> int`: returns a value from the ladder `{0, 16, 32, 64, 128, 256}`.
  - Base level: 32 when pot ≤ 8, 64 when pot ≤ 30, 128 above that.
  - Pace adjustment from `projected = used / (hand_index + 1) × hands`: if projected > `target_s`, drop one level per 20% overshoot; if projected < 0.5 × `target_s`, raise one level.
  - **Hard rule:** if used > 6.0 s, always return 0.

- [ ] **Step 1: Write the failing tests**
```python
def test_slow_clock_stays_under_bank():          # Review Focus 3: fake clock, each decision costs 3× nominal
    total = simulate_half(TimeManager(clock=fake), cost_per_iter=3*NOMINAL)
    assert total < 8.0
def test_big_pot_gets_more_iters_than_small():
def test_zero_iters_after_6s():
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): pot-weighted time manager`.

