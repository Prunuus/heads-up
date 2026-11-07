### Task 15: Exact betting tree

**Files:** Create `bot/tree.py`. Test: `tests/test_tree.py`.

**Interfaces — Produces:**
- `@dataclass BettingTree`:
  - `n_dec: int`, `street: np.ndarray[int8]`, `actor_is_button: np.ndarray[bool]`, `legal: list[tuple[Action, ...]]`
  - `child: np.ndarray[int32] [n_dec, 4]`: −1 = illegal, ≥0 = decision id, ≤ −2 = terminal `−2 − tid`
  - `term_fold_by_button: np.ndarray[int8]`: 1 = button folded, 0 = BB folded, −1 = showdown
  - `term_contrib: np.ndarray[int16] [n_term, 2]` (button, bb)
  - `key_to_node: dict[tuple[int, ...], int]`
- `build_tree() -> BettingTree`: exhaustive DFS with `game.Game` (default constants).
- `node_of(obs: Observation, tree) -> int | None`: key = `tuple(int(a.action) for a in obs.action_history)`.

- [ ] **Step 1: Write the failing tests**
```python
def test_tree_size():  t = build_tree(); assert t.n_dec == 15952 and n_terminals(t) == 25538   # F1
def test_preflop_nodes(): assert (t.street == 0).sum() == 62
def test_every_live_obs_maps_to_node_with_same_legal_actions():   # 2000 simulated decisions
def test_unknown_history_returns_none():                          # Review Focus 5
def test_terminal_contributions_match_game():                     # replay 500 random action paths
```
- [ ] **Steps 2–4:** fail → implement → pass. Build time must be < 2 s (it runs at startup).
- [ ] **Step 5:** Commit: `feat(bot): exact betting tree with history lookup`.

