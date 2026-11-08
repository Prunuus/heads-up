### Task 20: Integrate the blueprint, then the v3 gate

**Files:** Modify `bot/policy.py`: blueprint layer = `node_of` → `bucket` (with `iters` from `TimeManager`) → `Blueprint.act`, followed by `apply_exploits`. Results go in `docs/results/v3.md`.

- [ ] **Step 1:** Write the failing test `test_blueprint_layer_used_when_config_ok` and confirm it fails.
- [ ] **Step 2:** Implement; the test passes; run the full suite.
- [ ] **Step 3:** Sweep the purify threshold {0, 0.05, 0.1, 0.2} with `paired_diff` vs v2 on train opponents.
- [ ] **Step 4: Milestone M3 (go/no-go).** v3 ships only if all of these hold:
  - (a) head-to-head vs v2 over 5 seeds: mean > 0, with no CI loss
  - (b) expected points on held-out ≥ v2
  - (c) 0 errors and bank < 3.0 s
  - (d) LBR (Task 21) exploitability of v3 < that of v2

  Otherwise **keep v2** and log why.
- [ ] **Step 5:** If it's a go: `./tools/check_build.sh`, commit, push, **submit v3**.


---

## Controller additions (binding; from the ledger — read "NOTES FOR T20 (consolidated)", R39, R42, R44)

- **User directive (R44).** The final bot USES the CFR blueprint: v3 = blueprint as the base layer, the exploit layer on top, and the heuristic as fallback. The M3 gate becomes a safety check. v3 ships unless it is broken or clearly worse than v2: a CI loss head-to-head, or clearly lower expected points vs held-out + FIELD. If v3 underperforms, tune it rather than drop it: purify sweep, per-street use of the blueprint (e.g. blueprint postflop only), exploit gating.
- **State.**
  - `kevin` (this worktree, HEAD d0b35d5) now holds everything: T10–T14 (v2, pushed as 2115725), FIELD, the LBR probe (`sim/opponents/lbr.py`, with the `DistributionTarget` adapter for `probs`), and the CFR track (`bot/tree.py`, `bot/abstraction.py` with the 400/400/400/400 table, `train/mccfr.py`, `train/export.py`, `bot/strategy.py` Blueprint).
  - The overnight training run writes checkpoints to `/home/kchen/code/heads-up-bundle-cfr/train_data/mccfr-b400/blueprint_raw.ckpt-*.npz`: format 2, about every 30 min, final output at ~02:15.
  - For development, export the newest checkpoint into this worktree's `data/blueprint.npz` with `train/export.py`, per its CLI and the runbook. Don't commit the blueprint; the controller exports and commits the final one.
- **Integrate (`bot/policy.py`; `agent.py` only in YOUR CODE / warmup).**
  - Warmup loads `get_tree()`, `abstraction.warmup()` and the Blueprint. It never raises: a missing or corrupt file means the heuristic-only path.
  - Per decision: `node_of(obs)` → the bucket for (street, our hole, board), cached per (hand, street) → `Blueprint.act(node, bucket, legal, rng, purify)` → `apply_exploits`.
  - Fall back to the heuristic on a node miss, `config_ok` False (compare AgentConfig with the blueprint meta: stack, sb, bb, cap), a missing blueprint, or any exception. Count faults the way the other layers do.
  - T11 M2: take the online CPU sample independent of which layer answered.
  - When the blueprint answers, decide how the exploit layer gets what it needs (cheap features or mode-only exploits) and justify it. Keep fixed per-decision cost within budget (median `/act` ≤ 0.7 ms target, p99 ≤ 10 ms).
  - The rng stays per-half in memory.
- **Tests.**
  - The plan's `test_blueprint_layer_used_when_config_ok`.
  - A positive blueprint path skips the heuristic (T11 M3).
  - Config mismatch, missing/corrupt blueprint, and node-miss fallbacks.
  - 0 errors vs the whole zoo, slow.
  - Default `pytest -q` stays fast.
- **CPU.** Until ~02:15 the trainer owns all 16 threads. Keep your runs light: `nice -n 19`, ≤ 2 workers. Do NOT stop or signal the trainer.
  - Do a small provisional sanity run on the checkpoint blueprint (e.g. v3 vs v2, 1–2 seeds × 500 deals), only to catch gross problems.
- **The gate as a script.** Write `sim/gate_v3.py`, or an equivalent documented command set, that the controller runs on free CPUs after the final export. It must produce:
  - (a) v3 vs v2 head-to-head, 5 seeds × 1,000 deals, paired CI;
  - (b) expected points vs HOLDOUT and vs FIELD for v3 and v2, quiet protocol R39 (≤ 2 workers on reserved CPUs, opponents' per-half bank reported), at time_scale 1.0 and 2.5;
  - (c) 0 errors and our bank per half;
  - (d) LBR(v3) vs LBR(v2), 2,000 particles (T21 M5), with ModelCheckError respected;
  - (e) a purify sweep {0, 0.05, 0.1, 0.2} via paired_diff vs v2 on TRAIN.
  
  Make it resumable or chunked, and print one summary table.
- **Never push.** Commit messages are one simple sentence, with no body and no trailers.
