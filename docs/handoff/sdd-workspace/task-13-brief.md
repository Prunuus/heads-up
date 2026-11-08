### Task 13: Opponent model

**Files:** Create `bot/opponent.py`. Test: `tests/test_opponent.py`.

**Interfaces — Produces:**
`class OpponentModel`, which lives in memory:
- `observe(obs: Observation) -> None`: records the opponent actions in `obs.action_history` not seen yet, keyed by (street, facing_bet: bool), and remembers the last observation of the hand.
- `finish(result: HandResult) -> None`: infers the opponent's terminal action if they folded to our last bet (F11), and records showdown pairs `(opp action class per street, omaha_score(opp_hole, board))`.
- Properties: `fold_to_bet(street) -> tuple[float, int]` (rate, n), `raise_rate(street) -> tuple[float, int]`, `vpip`, `showdown_min_category(action_class) -> int | None`.
- Detectors:
  - `is_check_folder()`: ≥ 12 consecutive bet-facing folds and 0 raises in the last 40 decisions
  - `is_station()`: fold_to_bet < 0.05 with n ≥ 60
  - `is_maniac()`: raise rate > 0.55 with n ≥ 80

  Each detector **re-evaluates every hand**, so it reverts if the evidence goes away.

- [ ] **Step 1: Write the failing tests**
```python
def test_detects_checkfolder_after_switch():      # feed checkfold_after(50) traces; detected by hand 70, not before 50
def test_station_and_maniac_detection_thresholds():
def test_infers_fold_to_our_bet_from_result():
def test_hand_end_batch_never_raises():           # Review Focus 4: 20 results at once incl. opp_hole None, split, hands we never acted in
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): opponent model with reversible detectors`.

