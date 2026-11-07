### Task 5: Round robin, head-to-head, paired comparison

**Files:** Create `sim/tournament.py`. Test: `tests/test_tournament.py`.

**Interfaces — Produces:**
- `run_round_robin(entrants: dict[str, BotFactory], seeds: list[str], deals: int, workers: int = os.cpu_count()) -> Standings`, where `Standings` holds rows of `(name, points, mean_bb, losses, forfeits, max_bank)`.
- `focus_eval(candidate: BotFactory, opponents: dict[str, BotFactory], seeds, deals, workers) -> FocusReport`. Gives per-opponent `MatchStats` per seed, plus expected points (mean of `points_a`).
- `paired_diff(cand_a, cand_b, opponents, seeds, deals) -> tuple[float, float, float]`: mean and 95% CI of the per-deal difference in bb, using the same seeds and opponents (I10).
- CLI: `python -m sim.tournament --candidate bot.policy --opponents train|holdout|all --seeds N --deals D`, plus `--entrants zoo --report <path.md>`. The latter runs a zoo-only round robin and writes per-bot bank use and fold/raise rates.

- [ ] **Step 1: Write the failing tests**
```python
def test_round_robin_points_sum():
    s = run_round_robin({k: ZOO[k] for k in ("random", "calling_station", "nit")}, ["t1"], 100, workers=1)
    assert sum(r.points for r in s.rows) == pytest.approx(3.0)   # 3 matches, 1 point each

def test_paired_diff_of_identical_is_zero():
    lo, mid, hi = paired_diff(ZOO["nit"], ZOO["nit"], {"cs": ZOO["calling_station"]}, ["t1"], 100)
    assert lo == mid == hi == 0
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement with `multiprocessing.Pool`. **Step 4:** Run → PASS.
- [ ] **Step 5:** Run `.venv/bin/python -m sim.tournament --entrants zoo --seeds 1 --deals 300 --report docs/zoo_report.md`. Record in the report how many zoo bots show H1–H6.
- [ ] **Step 6:** Commit: `feat(sim): round robin, focus eval, paired comparison, zoo report`.

## Stage 1 — Fast evaluation and the robust heuristic bot (ship v1)

