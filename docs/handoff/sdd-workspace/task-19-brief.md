### Task 19: Blueprint export and runtime sampler

**Files:** Create `train/export.py`, `bot/strategy.py`, `data/blueprint.npz` (float16, ~13 MB). Test: `tests/test_strategy.py`.

**Interfaces — Produces:**
- `Blueprint.load(path) -> Blueprint` with `meta` = {stack, sb, bb, cap, buckets, iterations}.
- `probs(node, bucket) -> np.ndarray[4]`
- `act(node, bucket, legal, rng, purify: float = 0.1) -> Action`: masks illegal actions, zeroes probabilities below `purify` (I9), renormalizes, and if everything is zero falls back to the most probable legal action.

- [ ] **Step 1: Write the failing tests**
```python
def test_load_under_1s_and_meta_present():
def test_act_always_legal_even_if_all_mass_illegal():
def test_purify_zero_equals_raw_sampling_distribution():   # χ² over 10k draws
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): blueprint export and sampler`.


---

## Controller additions (binding; from the ledger — read the lines cited)

- **Blueprint metadata and the load-time check (T15, T16 and T18 notes).** Beyond the plan's {stack, sb, bb, cap, buckets, iterations}, `data/blueprint.npz` stores:
  - `n_dec`, `n_term` and the hash of the tree's `child.tobytes()`;
  - the sha256 of `data/buckets.npz` and the per-street bucket counts;
  - the T17 dataset identity and the T18 training meta (seed, linear flag).
  
  `Blueprint.load` verifies every one of these against the runtime tree and the runtime buckets table, and refuses a mismatch. A refit or a tree change invalidates the blueprint. T20's `config_ok` relies on these fields (Review Focus 1).
- **R26 (per-street counts).** The counts come from the checkpoint and the table, not a hard-coded 200. Shape the exported arrays per street, or with a documented padding, so the file stays well under 100 MB (global constraint). Measure and report the size, and the load time against the plan's < 1 s.
- **Runtime rules for `bot/strategy.py`.**
  - It imports only stdlib + numpy.
  - Module-level state is read-only (R22): non-writeable arrays, no mutable globals.
  - A missing or corrupt blueprint file must never raise at import or warmup. T11's PolicyBot treats "no blueprint" as the heuristic-only path (R10).
  - `act()` always returns a member of `legal` (plan test).
  - `probs()` is cheap: report p99, which counts against the ~0.3 ms fixed per-decision budget.
- **Export source.** `train/export.py` exports from a T18 checkpoint or final output, through T18's loader and its checks.
  - The long training run is in progress while you work, so develop and test against an intermediate checkpoint or T18's verification-run output. Ask the controller for paths if they aren't in `train/README.md`.
  - The controller re-runs the export on the final checkpoint.
  - Commit `data/blueprint.npz` only if the controller asks. The default is code only.
- **Purification (I9).** The `purify` argument behaves per the plan, so the purify sweep in T20 can run. Note that T21's LBR probe consumes the blueprint through a probability function `(obs, hole) → probs over obs.legal_actions` (`DistributionTarget` in `sim/opponents/lbr.py`, on branch kevin-lbr). Make `probs` easy to wrap that way. Don't modify lbr.py.
- **CPU (R32).** The overnight MCCFR run owns CPUs 8–15 (and later all of them). Keep your jobs light. Run anything heavy as `nice -n 19`, and state where you ran it.
- **Constraints.** Branch `kevin-cfr` in `/home/kchen/code/heads-up-bundle-cfr`. Commit messages are one simple sentence, with no body and no trailers (R36). Never push.
- **CORRECTION (Task 18 review, Important #2).** The trainer's arrays are ONE flat `[n_info, 4]` array, node-major in DFS node-id order, with the streets INTERLEAVED (`tree.street` changes value 3,182 times along node ids).
  - Each node's rows are the contiguous block `offset[n] .. offset[n+1]`. A street's rows must be GATHERED via the offsets, e.g. `np.concatenate([np.arange(offset[n], offset[n+1]) for n in np.flatnonzero(tree.street == s)])`.
  - Never slice a street as one contiguous range. `train/mccfr.py`'s docstring said otherwise; it is being corrected in T18's fix round.
  - The average-strategy array becomes float64 in that fix round; regret stays float32.
