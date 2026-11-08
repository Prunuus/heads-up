### Task 2: In-process duplicate match simulator

**Files:** Create `sim/__init__.py`, `sim/match.py`. Test: `tests/test_sim_match.py`.

**Interfaces — Produces:**
- `load_bot_file(path: str) -> ModuleType`: imports a bot file under a unique module name each call, so every half gets fresh globals.
- `@dataclass HalfStats(nets_a: list[int], errors_a: int, errors_b: int, forfeit_a: bool, forfeit_b: bool, bank_a: float, bank_b: float, calls_a: int, calls_b: int, max_call_a: float)`
- `@dataclass MatchStats(mean_bb: float, sd_bb: float, ci_low: float, ci_high: float, verdict: str, points_a: float, halves: tuple[HalfStats, HalfStats])`, with `verdict ∈ {"A", "B", "CHOP", "BOTH_FORFEIT"}`.
- `play_half(fa: BotFactory, fb: BotFactory, seed: str, half: int, deals: int = 1000, grace_s: float = 0.00075) -> HalfStats`
- `play_match(fa: BotFactory, fb: BotFactory, seed: str, deals: int = 1000) -> MatchStats`

These mirror `judge_engine.py` exactly:
- seats: `agent1_seat_for_half`; deals: `make_deal(seed, d)`; button: `button_for_deal(d)`
- hand results are queued and delivered before the bot's next `choose_action`, then flushed at the end of the half
- an exception, a non-Action or an illegal action counts as an error and plays `default_action`
- a forfeit at 5 errors; a bank that empties switches the bot to `default_action`
- a call slower than 1 s counts as an error
- forfeit-loses-match logic as in `combine_halves`

- [ ] **Step 1: Write the failing tests**
```python
def test_matches_real_judge_value():          # F10
    m = play_match(lambda: load_bot_file("agent.py"),
                   lambda: load_bot_file("baseline-bots/baseline_calling_station.py"), seed="1")
    assert m.mean_bb == pytest.approx(-0.516, abs=5e-4) and m.verdict == "B"

def test_identical_bots_chop():
    f = lambda: load_bot_file("baseline-bots/baseline_calling_station.py")
    m = play_match(f, f, seed="s1", deals=200)
    assert m.mean_bb == 0 and m.verdict == "CHOP"

def test_crashing_bot_forfeits_and_loses():   # bot raising on every call
    m = play_match(crash_factory, call_factory, seed="s1", deals=50)
    assert m.halves[0].forfeit_a and m.points_a == 0

def test_on_hand_end_errors_count_individually():
    h = play_half(handend_crash_factory, call_factory, "s1", 1, deals=10)
    assert h.errors_a == 5 and h.forfeit_a
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Charge time `max(0, elapsed − grace_s)` with `time.perf_counter()`, around `on_hand_end` deliveries plus `choose_action` together, as one "request".
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat(sim): in-process duplicate match simulator matching judge`.

