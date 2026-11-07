### Task 11: Layered policy and `agent.py` wiring

**Files:** Create `bot/policy.py`. Modify `agent.py` (YOUR CODE section + warmup area only). Test: `tests/test_policy.py`.

**Interfaces — Produces:**
- `warmup() -> None`: loads all tables; must finish < 6 s.
- `start_half(config: AgentConfig, memory: dict) -> None`: sets `memory["tm"]`, `memory["rng"] = np.random.default_rng(<fixed>)`, `memory["opp"]`, and `memory["config_ok"] = (config matches the blueprint metadata)`.
- `decide(obs, memory) -> Action`: layers blueprint (if loaded and `config_ok`) → exploit → heuristic. **Each layer is in its own try/except.** The final fallback is `CHECK_CALL if obs.to_call == 0 else FOLD` filtered by `legal_actions`. The result is asserted to be in `legal_actions`; if it isn't, return the fallback.
- `hand_finished(result, memory) -> None`: never raises (the whole body is in try/except).

- [ ] **Step 1: Write the failing tests**
```python
def test_layer_exception_falls_back_legally(monkeypatch):    # heuristic raises → still a legal action
def test_config_mismatch_uses_heuristic():                   # Review Focus 1: starting_stack=100 → blueprint never consulted
def test_warmup_under_6s():
def test_zero_errors_vs_whole_zoo():                         # play_half vs each ZOO bot, 200 deals, errors_a == 0, bank_a < 3.0
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement, and wire `agent.py`.
- [ ] **Step 4:** Run → PASS. Also run `.venv/bin/python -m pytest -q` (all tests, including root `test_agent.py`) → PASS.
- [ ] **Step 5:** Commit: `feat: layered policy wired into agent.py`.

