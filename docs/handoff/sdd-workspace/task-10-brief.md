### Task 10: Features and heuristic policy

**Files:** Create `bot/features.py`, `bot/heuristic.py`. Test: `tests/test_heuristic.py`.

**Interfaces — Consumes:** Tasks 6–9. **Produces:**
- `@dataclass(frozen=True) Features(street: Street, pre: float | None, eq: float, now: float, nut: int, spr: float, pot_odds: float, in_position: bool, opp_raised_this_street: int)`
- `compute_features(obs: Observation, iters: int, rng) -> Features`. Preflop uses the table; postflop uses Task 7.
- `@dataclass(frozen=True) HeuristicParams` with tunable fields and defaults:
  - `open_min=0.44`, `threebet_min=0.53`, `call_margin=0.02`
  - `range_discount=(0.10, 0.08, 0.06)` (subtracted from equity per opponent raise, by street)
  - `value_raise=(0.70, 0.75, 0.80)` flop/turn/river
  - `pot_vs_half_cut=0.85`, `bluff_freq=0.10`
  - `nonnut_penalty=0.08` (applied when `nut & 0b110` is set and `nut & 1` is clear)
- `heuristic_action(obs, feats: Features, params: HeuristicParams, rng, opp=None) -> Action`, always ∈ `obs.legal_actions`.
  - Effective equity = `eq − range_discount·opp_raises − nonnut_penalty·[non-nut on dangerous board]`.
  - Raise when effective equity > `value_raise[street]`; choose RAISE_POT if > `pot_vs_half_cut`, else RAISE_HALF.
  - Call when effective equity > pot odds + `call_margin`.
  - Otherwise check or fold (never fold when free). Bluff with `bluff_freq` in position when checked to.

- [ ] **Step 1: Write the failing tests**
```python
def test_always_legal_on_10k_random_observations():   # drawn from simulated games vs random bot
def test_never_folds_when_free():
def test_all_in_corners_are_legal():                  # Review Focus 2: to_call >= my_stack; opp all-in; only CHECK_CALL legal
def test_calls_with_nuts_on_river():                  # nut & 1 on river, to_call>0 → not FOLD
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): features and parameterized heuristic policy`.

