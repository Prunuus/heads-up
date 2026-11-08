### Task 18: MCCFR trainer

**Files:** Create `train/mccfr.py`. Test: `tests/test_mccfr.py`.

**Interfaces — Consumes:** Tasks 15 and 17. **Produces:** `train(tree, deals_path, iterations: int, out: str, seed: int, linear: bool = True) -> None`.

The trainer runs **external-sampling MCCFR** [L: Lanctot 2009]:
- Each iteration samples one deal row from the dataset and alternates the traverser.
- Regret matching uses positive regrets; the average strategy uses linear weighting [L: Brown & Sandholm 2019].
- Info set = `(node, bucket[actor seat, node street])`, which is imperfect recall.
- Terminal utility comes from `term_contrib` plus fold/showdown and the deal's `showdown` value.
- Arrays: `regret float32 [n_dec, 200, 4]` and `avg float32 [n_dec, 200, 4]`.
- Checkpoint every 10% of iterations. Use numba `@njit` if pure numpy is < 20k iterations/s.

- [ ] **Step 1: Write the failing tests**
```python
def test_one_bucket_game_is_seed_stable():           # all buckets = 0, random showdowns: self-play value of avg strategy differs < 0.02 bb between 2 training seeds
def test_regret_matching_never_picks_illegal():      # avg strategy has zero mass on child == -1 at every node
def test_strong_bucket_raises_more_than_weak():       # synthetic dataset where bucket 99 always wins: P(raise | top bucket, first river node) > 0.8
def test_checkpoint_roundtrip():
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Train with `--iterations` chosen from the measured throughput (target ≥ 200M sampled iterations or 6 h wall, whichever comes first). Commit the code.


---

## Controller additions (binding; from the ledger — read the lines cited)

- **R14 as amended by R19 (who runs what).** You verify throughput plus one ≤ 20-minute training run on the real dataset. The controller launches the long run (plan target: ≥ 200M sampled iterations or 6 h wall, whichever comes first) in the background after review, from your runbook.
- **T15 deferred minor, required here.** `bot/tree.py:161` `node_from_actions`/`_codes` reject numpy integers. Accept `np.integer` (not bool) or convert at the boundary. Never pass numpy bools as actions.
- **Metadata for T19 (T15 + T16 notes).** Every checkpoint and the final output carry:
  - `n_dec`, `n_term`, and a hash of the tree's `child.tobytes()`;
  - the sha256 of `data/buckets.npz` that the dataset was bucketed with, and the per-street bucket counts;
  - the game constants (stack, sb, bb, raise cap);
  - the iterations done, seed, linear flag and dataset identity.
  
  Loading refuses a mismatch with the current tree or buckets table. T19 exports `data/blueprint.npz` from this, with the same checks.
- **R26 (configurable bucket counts).** Size the arrays from the dataset/table's per-street counts (`bot.abstraction.NUM_BUCKETS`), not a hard-coded 200. Either use per-street arrays, or use `[n_dec, max_B, 4]` with the unused bucket rows never touched; document which. Use `load_deals` from `train/gen_deals.py` (Task 17) for the data, and read its docstring for the arrays and meta it returns.
- **R23 (portable, many-core, resumable).**
  - Linux/macOS, `requirements-train.txt` only (numba is allowed offline).
  - Scale to every core. Recommended: numba `parallel=True`/`prange` worker threads over shared regret/avg arrays with lock-free (hogwild-style) updates, which is standard practice for parallel MCCFR. Document the race semantics and why they're benign, or choose another design and justify it.
  - Checkpoint every 10% and on SIGINT/SIGTERM. Resume from the latest checkpoint continues the iteration count, the RNG streams and the linear-averaging weights, so an interrupted-and-resumed run is equivalent to an uninterrupted one. Test this.
  - Extend `train/README.md` with the runbook: how to launch the long run detached (nohup/tmux) with logs, monitor progress, resume, and what throughput to expect.
- **Correctness test with a known answer (required, beyond the plan's four).** The plan's tests can't catch a wrong sign, regret update, traverser alternation or averaging weight.
  - Add an exact best-response / exploitability evaluator for the abstract game.
  - Test it on a synthetic dataset with PERFECT RECALL: each row gives each player one hand class that is the same bucket on every street (e.g. K = 2–4 classes), with showdowns correlated to the classes. Backward-induction best response is then exact.
  - Assert that the trained average strategy's exploitability falls below a threshold you set from measurement, and decreases across checkpoints.
  - Keep the evaluator reusable as a diagnostic on the real dataset. There it is approximate, because of imperfect recall; label it as such. It will be logged during the long run.
- **Sizing for this PC.** AMD Ryzen 7 9700X, 8 cores / 16 threads (SMT siblings = logical CPUs 2k, 2k+1), 15 GB RAM in WSL2. Measure iterations/s at 1, 4, 8 and 16 threads, and memory per infoset. Then recommend a (per-street bucket counts, iteration target) pair for a ~6 h run on 12–16 threads, with the reasoning (expected visits per infoset at each street).
  - The controller decides, and can refit `data/buckets.npz` with more buckets and re-bucket the Task 17 dataset from its stored features in minutes (`train/fit_buckets.py --preflop/--flop-grid/--turn-grid/--river`; Task 17's rebucket path).
  - Plan's rule: use numba `@njit` if pure numpy gives < 20k iterations/s.
- **CPU etiquette (ruling R32).** Track A runs timing-sensitive sims on CPUs 0–7 at the same time as you. Run every heavy job (throughput runs, the ≤ 20-min verification run) as `taskset -c 8-15 nice -n 19 …`. For the 16-thread throughput point only, keep the run under 60 s and note that it overlapped Track A. The default test suite may run unpinned; keep it fast (R11: anything ≥ a few seconds is `@pytest.mark.slow`).
- **Constraints.** Branch `kevin-cfr` in `/home/kchen/code/heads-up-bundle-cfr`. Training outputs go under `train_data/` (git-ignored). Commit code and docs only. Never push. Offline code may use numba and multiprocessing; runtime `bot/` stays stdlib + numpy. Export and purification are T19, not this task.

### From the Task 17 review and the controller (ledger lines "Task 17:" and R37)
- **Seat mapping.** The dataset's seat 0 is the button: `buckets[row, 0, street]` is the button's bucket, and `showdown = +1` means the button wins. Map the tree's `actor_is_button` / `term_fold_by_button` / `term_contrib` (button, bb) onto that consistently. Add a test that replays a few random paths with `game.Game` on a dataset row's real cards and checks the trainer's terminal utility for both seats against the engine's nets.
- **Feature hash (T17 Minor M1, required).** `train.gen_deals.load_deals` checks the buckets-table sha but ignores the stored `features_sha256`. A changed `data/preflop.npz` with an unchanged `buckets.npz` would therefore pass silently. Add that check to the load path the trainer uses (in `load_deals`, or in the trainer right after it) and test it. Optionally also guard `gen_deals.PREFLOP_TABLE == bot.preflop._DEFAULT_PATH` (T17 Minor M4).
- **Datasets.**
  - `train_data/` in the worktree holds a stable 2M-deal set (40 shards). Use it for your verification run.
  - The controller is extending a copy to 10M deals in `train_data/d10m/`, in the background on CPUs 8–15 at nice 19. Don't use `d10m` until it's complete, since it is growing. The long run will use it.
- **CPU, revised (supersedes the CPU-etiquette bullet above).** CPUs 8–15 also carry background jobs at nice 19: the 10M generation, the Task 21 LBR probe and three other builds.
  - Run your throughput points and your ≤ 20-minute verification run at NORMAL priority, pinned `taskset -c 8-15`. The nice-19 jobs then yield, and your numbers stay close to true capacity. Note the load average next to each number anyway.
  - The 16-thread point is the only one that may use CPUs 0–7. Keep it under 60 s.
- **Commits.** One simple sentence per commit message, with no body, no trailers and no Co-Authored-By line (user rule, R36; see global-constraints.md).
