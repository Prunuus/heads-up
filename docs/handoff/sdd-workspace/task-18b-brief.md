### Task 18b (controller-added, ruling R38): refit the abstraction to 400/400/400/400 and re-bucket the 10M dataset for the overnight run

**Why.** Task 18 measured 643k MCCFR iterations/s on 8 threads, and found that bucket count barely affects speed. It recommends 400 buckets on every street: a 6-hour run (12–24B iterations) still gives ≥ 2×10^4 visits at the 10th percentile on every street. The night's surplus compute therefore goes to a finer card abstraction. Read `task-18-report.md` (sizing section) and the `train/README.md` runbook ("More buckets", "For finer buckets").

**Work in:** branch `kevin-b400`, worktree `/home/kchen/code/heads-up-bundle-b400`, forked from kevin-cfr@14c83f4, which holds Tasks 15–18. Task 18 is in review on kevin-cfr at the same time. Do not touch the kevin-cfr worktree, except the git-ignored dataset copy described in step 3.

**Steps:**
1. **Refit `data/buckets.npz`:** preflop 400, flop grid 20×20, turn grid 20×20, river 400 (395 equity + 5 nut categories).
   - Keep the fit `--seed 0` and the default design sizes in `train/fit_buckets.py`. T17's stored features stay valid only if the designs are unchanged, and `--rebucket` refuses otherwise.
   - Choose `--per-street` large enough that every cell has ≥ ~2,000 samples (the runbook uses 1,000,000).
   - Report the wall time, the occupancy ranges per street, the file size (must be < 100 MB), and the runtime `bucket()` p99 per street. It should be unchanged, since runtime cost depends on the designs, not the counts.
2. **Make the test suite pass with the new counts without weakening what the tests check.**
   - Occupancy tests should scale their sample to the bucket count (or a slow variant should hold the full check) so that [0.4×, 2.5×] stays meaningful.
   - Any test that hard-codes 200/100 must read the counts from the table.
   - Don't change `bot/abstraction.py` logic. If something there truly must change, stop and report NEEDS_CONTEXT.
   - Keep the default `pytest -q` fast; R11 applies.
3. **Re-bucket a full COPY of the 10M dataset**, not hard links: rebucket may rewrite shard files in place.
   - Copy `/home/kchen/code/heads-up-bundle-cfr/train_data/d10m` to `/home/kchen/code/heads-up-bundle-cfr/train_data/d10m-b400`. The controller's 10M generation finishes about 20:35; its log `train_data/d10m.log` ends with `exit=0` when it's done. Wait for that.
   - Run `python -m train.gen_deals --rebucket --out <copy> --buckets <this worktree's data/buckets.npz> --verify <n>`.
   - Confirm `load_deals(<copy>)` accepts the new table and refuses the old one.
4. **Smoke-train** about 2 minutes with the new table on the re-bucketed copy, 8 threads pinned `taskset -c 8-15`, at normal priority. Use a scratch `--out` under the cfr worktree's `train_data/`. Report iterations/s and that checkpoints are written and load.
5. **In your report, write the exact launch command for "run B"**: `--deals` = the re-bucketed copy, `--iterations 24_000_000_000 --max-hours 6`, `--out train_data/mccfr-b400/blueprint_raw.npz`, 8 threads on CPUs 8–15, nohup and log. Include the resume command with 16 threads for later.

**Constraints.**
- Commit only `data/buckets.npz`, the test changes and README notes, on `kevin-b400`. Never push.
- Commit messages are one simple sentence, with no body and no trailers (R36).
- Don't modify organizer files.
- Run heavy jobs as `taskset -c 8-15 nice -n 19 …`, except the smoke run in step 4. CPUs 0–7 are Track A's.
