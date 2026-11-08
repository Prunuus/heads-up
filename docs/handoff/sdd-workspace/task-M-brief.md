### Task M (controller-added, R34/R39/R41): assemble v2 on kevin and re-gate it

**Goal.** Produce the v2 submission candidate on `kevin`: v1 (Task 12) plus the exploit layer (Task 14), with FIELD (Task F) and the LBR probe (Task 21) merged in. Then measure it under the quiet protocol.

**Merges** (worktree `/home/kchen/code/heads-up-bundle`, branch `kevin`, HEAD 9fbaf8b). Merge in this order, with default one-line merge messages:
1. `kevin-opp` (a9ffa6b: T13 opponent model + T14 exploit layer).
2. `kevin-field` (d78414a: FIELD split).
3. `kevin-lbr` (20bd2dd: sim/opponents/lbr.py).

Expected conflicts:
- **`bot/policy.py`.** T12 changed the clock lines in `new_half`/`on_hand_end` (the scaled TimeManager clock); T14 added the OpponentModel wiring. Keep both behaviours. `opp.finish(result)` stays between `t0` and `tm.add(...)`, measured on T12's scaled clock (T14 review merge note).
- **`tests/test_tournament.py:138`** and possibly `tests/test_policy.py`, `pytest.ini`, `sim/opponents/__init__.py`. Keep both sides. `TRAIN = ZOO − HOLDOUT − FIELD` must hold after the merge, and T12's tuner must still refuse FIELD.

Never change strategy logic while resolving.

**Verification after the merges** (`taskset -c 0-7`; CPUs 8–15 run the overnight training):
- Run the full default suite plus the root `test_agent.py`. A few timing-threshold tests can fail under load; re-run those alone on a quiet moment before calling them real.

**M2 re-gate on the merged head.** v2 = `PolicyBot()` (exploits on); v1 = `PolicyBot(use_exploits=False)`. Quiet protocol R39: `taskset -c 0-7`, ≤ 2 workers, opponents' per-half bank reported next to every result. Run at time_scale 1.0; add 2.5 where time allows.
- v2 vs v1 paired on HOLDOUT. Expected points on held-out must be v2 ≥ v1.
- `checkfold_after(300)` gain with ≥ 5 seeds. It was load-sensitive with 3 seeds.
- v2 vs each FIELD bot, 3 seeds × 1,000 deals, serial or 2 workers, with the FIELD bank check. Information, but critical: report it plainly.
- 0 errors; bank < 3.0 s per half; median `/act` ≤ 0.7 ms on a quiet core.

**T24 release-checklist items** (except check_build and push, which are the controller's):
- Real judge (`judge_engine.py`), with our bot built from `docker/Dockerfile.agent` and run `--cpus 1`: vs every baseline + 2 LLM bots, 2 seeds, halves run separately with restarts.
- Required result: 0 errors, no FORFEITED, judge-charged bank < 3.0 s per half.
- Reuse the in-container harness T12 documented in `docs/results/v1.md`. Use Docker via `sg docker -c "…"`.

**Outputs.**
- Update `docs/results/v2.md` with a "merged head" section.
- Commit messages are one simple sentence with no trailers.
- Report: write `task-M-report.md` in the SDD folder. If the harness blocks that write, put the full report in your final message instead; don't route around the block.
- Never push.
