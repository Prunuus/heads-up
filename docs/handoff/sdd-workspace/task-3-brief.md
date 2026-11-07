### Task 3: Archetype opponent zoo

**Files:** Create `sim/opponents/__init__.py` and `sim/opponents/archetypes.py`. Test: `tests/test_zoo.py`.

**Interfaces — Produces:** `ZOO: dict[str, BotFactory]` containing:
- `random`, `calling_station`, `always_half`, `always_pot` (load from `baseline-bots/`)
- `guide_example` (the PDF §4.1 strategy)
- `nit` (folds unless preflop strength is in the top 20% / flop+ two-pair+)
- `maniac` (raises whenever legal 70% of the time, otherwise calls)
- `mc_equity_fast` (64-rollout equity vs random hand + pot odds; raise if equity > 0.7)
- `mc_equity_slow` (500 rollouts: same logic, and it will empty its bank, which is realistic per H1)
- `threshold_cat` (category thresholds, ignoring board texture)
- `checkfold_after(n)` (normal `mc_equity_fast` for n hands, then `default_action`)

Each opponent is a class with the three methods. Opponent state lives in `memory`.

- [ ] **Step 1: Write the failing test**
```python
@pytest.mark.parametrize("name", list(ZOO))
def test_zoo_bot_is_legal_and_errorless(name):
    h = play_half(ZOO[name], ZOO["calling_station"], "z1", 1, deals=100)
    assert h.errors_a == 0 and not h.forfeit_a
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Use `evaluator.omaha_score` (deliberately slow, as LLM bots would be).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat(sim): archetype opponent zoo`.

