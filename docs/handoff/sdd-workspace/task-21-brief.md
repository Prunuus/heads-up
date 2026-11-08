### Task 21: Local Best Response probe (validation)

**Files:** Create `sim/opponents/lbr.py`. Test: `tests/test_lbr.py`.

**Interfaces — Produces:** `make_lbr(target: BotFactory, rollouts=500) -> BotFactory`.
- It keeps a sampled belief over the target's holdings: 2,000 particles, reweighted after each target action by the target's own action probabilities. For the blueprint these come from `probs`; for the heuristic, from deterministic re-simulation.
- At each decision it picks the action that maximizes immediate EV, assuming the target calls or folds per its policy, with check/call afterwards [L: Lisý & Bowling 2017].
- Offline only, with no time limit.

- [ ] **Step 1:** Write the failing test `test_lbr_beats_always_pot_and_calling_station` (CI win) and confirm it fails.
- [ ] **Step 2:** Implement; the test passes.
- [ ] **Step 3:** Report LBR bb/deal vs v1, v2 and v3 in `docs/results/v3.md`.
- [ ] **Step 4:** Commit.

## Stage 4 — Stretch (each item is independently gated by `paired_diff` on held-out)


---

## Controller additions (binding; from the ledger)

- **Why now (R15 + the user's 18:35 request to parallelize).** T21 runs before T20, because the v3 gate's criterion (d) compares the LBR exploitability of v3 with that of v2. You start early, on your own track (Track D: branch `kevin-lbr`, worktree `/home/kchen/code/heads-up-bundle-lbr`, forked from kevin@85a038e). The blueprint (T19) and the final v1/v2 bots don't exist yet, so design for targets that plug in later.
- **Target interface (design it explicitly).** `make_lbr(target: BotFactory, rollouts=500) -> BotFactory` per the plan. The belief update needs the target's action probabilities for hypothetical target holdings:
  - **BotLike targets** (zoo bots, the T10 heuristic, later the T11 `PolicyBot`): deterministic re-simulation. Call the target's decision path on the hypothetical observation with a copied memory and a fixed per-particle rng. A stochastic policy becomes deterministic per particle.
  - **Distribution targets:** an adapter taking a callable `(obs, hole) -> probabilities over obs.legal_actions`. T20 will plug the blueprint's `probs(node, bucket)` into it.
  
  Document the adapter contract in the module docstring.
- **Practical budget (we have a deadline; "no time limit" means not bound by the tournament's bank).**
  - The plan says 2,000 particles and 500 rollouts. Re-simulating a ~0.3–1 ms heuristic for 2,000 particles at every target decision can cost hours per 1,000-deal match.
  - Measure, then choose particle/rollout counts and any batching or caching, so that one LBR evaluation of a target (several seeds, enough deals for a useful CI) finishes in ≤ ~30 min on 8 workers.
  - Report the accuracy/cost trade-off you measured: how much the LBR value moves between your setting and the plan's.
- **Deliverables.**
  - `sim/opponents/lbr.py`.
  - `tests/test_lbr.py`, with the plan's `test_lbr_beats_always_pot_and_calling_station` (CI win). Slow parts are `@pytest.mark.slow` (R11); keep the default suite fast.
  - The LBR bb/deal (with CI and deals) vs the T10 heuristic: `sim/heuristic_bot.py`'s factory at v1 defaults, "v1-untuned". v1/v2/v3 are measured later, by the controller at T20.
- **Ruling (file conflict).** Write results to `docs/results/lbr.md`, not `docs/results/v3.md`: Track B's T17 creates v3.md on another branch, and this avoids a merge conflict. Don't register LBR in `ZOO`, and don't edit `sim/opponents/__init__.py` (shared across tracks) unless it is unavoidable; if it is, say why.
- **Engine semantics.** `game.py` is the rules authority. Blinds don't count toward the 4-raise cap; a short all-in counts as a raise (`game.py:382,413,485`). Use the engine (clone/step) for hypothetical play rather than re-implementing rules.
- **CPU etiquette (ruling R32).** Track A runs timing-sensitive work on CPUs 0–7. Run every heavy job as `taskset -c 8-15 nice -n 19 …`, with `--workers` ≤ 8; you share 8–15 with Track B's dataset generation. The default test suite may run unpinned.
- **Constraints.** Commit on `kevin-lbr` only; never push. Runtime `bot/` is untouched: this is offline sim code.
