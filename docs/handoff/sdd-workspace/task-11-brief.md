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


---

## Controller additions (binding; from the ledger — read the ledger lines cited)

- **R5 (API).** `class PolicyBot(params: HeuristicParams | None = None, use_blueprint: bool = True, use_exploits: bool = True)` in `bot/policy.py`, with `new_half/choose_action/on_hand_end` (BotLike). `agent.py` delegates to a module-level `PolicyBot()`. Module-level `start_half/decide/hand_finished/warmup` remain as thin wrappers (the plan's interface). `sim.tournament --candidate` accepts `agent` (load_bot_file("agent.py")), a ZOO name, or a bot-file path; make sure `agent` works end to end. PolicyBot factories must be picklable (R3) — e.g. a top-level factory or functools.partial — because T12 tunes in a multiprocessing pool.
- **R10 (empty slots).** blueprint = None (until T20), exploits = identity (until T14), `memory["opp"] = None` (until T13 is merged from kevin-opp before T14). `config_ok` is computed against blueprint metadata when one exists; with no blueprint it must still be computed so T20 can rely on it, and the blueprint layer is skipped. Missing or corrupt data files must never crash warmup or a half: fall back to the heuristic.
- **R9 + T9 carry-forward.** TimeManager `begin()/end()` wrap each choose_action call in try/finally (grace applied per call). on_hand_end time goes in via `add(dt)` with no grace. Floor the measured cpu factor at 1.0. `tm.hands` is a float (never pass it to range()). Keep ALL lazy initialization out of the first choose_action (hand-0 amplification), and keep the fixed per-decision cost outside Monte Carlo at or below ~0.3 ms on this PC. Skip expensive features (nut_flags, MC) when iters()==0.
- **T9 ruling (grace).** TimeManager takes `grace_s`. `bot.timing.grace_for_cpu_factor(f) = 0.001 − 0.00025·f`, unclamped. T11 measures the cpu factor in warmup and passes the grace.
- **R30a (tournament hardware).** At warmup, time a fixed reference workload and compare it with a constant measured on THIS PC (the new "dev laptop"). Measure that constant on a quiet machine, pinned `taskset -c 0-7`, best-of-N, and record how. cpu_factor = max(1.0, measured/reference) → grace_for_cpu_factor. ALSO re-estimate cpu_factor online during play from measured decision times. TimeManager pacing already self-corrects on measured time; the online estimate closes the unseen-plumbing hole. It must be robust to preemption spikes: use a robust statistic, clamp, never raise. Context: real HTTP plumbing on this PC = 0.29 ms/request median (judge keep-alive → uvicorn agent.py), consistent with plan F3.
- **R30c (startup).** warmup ≤ ~2 s on this PC, including all table loads. The hard limit is 15 s; the plan's test says < 6 s.
- **Warmup must touch every lazy cache before the first /act:**
  - `bot.preflop.load_preflop()` (T8 note: otherwise the first /act pays ~15 ms);
  - `bot.equity.warmup()` (AUDIT-FIX note);
  - `bot.equity.nut_flags` once per board width 3/4/5 (T10 note: the first postflop decision otherwise pays an 11 ms cache build);
  - one dummy `compute_features` + `heuristic_action` per street.
- **AUDIT-FIX notes.** PolicyBot relies on no mutable module state in `bot/` (R22: module state = read-only tables; per-half state in memory or instance). Pickling bot.* objects during play fails in the sim, because load_bot_file gives each half private module copies: document this in sim/match.py and here, and don't rely on it.
- **Review Focus 1.** `test_config_mismatch_uses_heuristic` (starting_stack=100 → blueprint never consulted). The heuristic reads sizes only from obs.
- **Error budget.** 5 errors in a half forfeit the half and lose the match. Every layer has its own try/except. hand_finished never raises. The final result is checked against legal_actions, with the guaranteed-legal fallback (CHECK_CALL if to_call == 0 else FOLD, filtered by legal).
- **Tests.** The plan's 4 tests. `test_zero_errors_vs_whole_zoo` plays sim halves (it is slow: mark it per R11 if it takes ≥ a few seconds, and keep a fast smoke version in the default suite). Also add: online cpu-factor estimator robustness (spikes, zeros, NaN); warmup touches every lazy cache (the first decision per street costs ≈ steady state); agent.py end-to-end via the FastAPI TestClient (/new-half, /act, /hand-end batch) with the real PolicyBot; and the root `test_agent.py` passes. The default `pytest -q` must stay fast.
- **CPU etiquette (ruling R32).** Track B (T17 dataset generation) runs heavy jobs pinned to CPUs 8–15 at the same time as you. Run your timing measurements and sims pinned `taskset -c 0-7`, with sims at `--workers ≤ 4`. Report the load average next to every timing number.
- **Agent.py edits.** Only the YOUR CODE section plus the warmup area. Never touch SERVER PLUMBING or the PORT lines. PARTICIPANT is now "3XKO" (the user's team name, ruling R1 resolved): keep it.

### From the Task 10 review (ledger "Task 10:" lines)
- The first-decision spike is the first `np.unique` call at `bot/equity.py:156`: 3.4–4.0 ms on this PC, about 11 ms on the Mac. Warmup calling `nut_flags` once per board width (3/4/5) brings it to 0.004 ms. Verify that the first decision per street costs about the same as the steady state.
- Pass `grace_for_cpu_factor(cpu_factor)` to the TimeManager. The reviewer's 5× CPU emulation reached a bank of 7.1 s with the default grace and 5.0 s with the scaled grace.
- Every layer catches `Exception`, never a narrower type:
  - `compute_features` raises `ValueError`/`IndexError` on malformed boards or card ids;
  - `range_equity` raises on NaN params.
  - The engine never produces these, but a corrupt observation must fall through to the next layer.
- Ruling (M1): `nut_flags` keeps running at iters==0 (p99 0.199 ms). If you add a critical-bank cheap path, skip the Monte Carlo but keep the nut bits.
