### Task 17: Training deal dataset  (plan text, verbatim)

**Files:** Create `train/gen_deals.py`. Test: `tests/test_gen_deals.py`.

**Interfaces — Produces:** `train_data/deals_<k>.npz` containing:
- `buckets: int16 [N, 2, 4]` (seat 0 = button)
- `showdown: int8 [N]` (+1 button wins, 0 split, −1 BB wins), using `omaha_score`
- CLI: `--n 2_000_000 --workers 8 --iters 1000`

- [ ] **Step 1: Write the failing test**
```python
def test_small_dataset_shapes_and_outcomes():   # n=200: shapes; showdown equals omaha_score comparison on regenerated deals
```
- [ ] **Steps 2–4:** fail → implement → pass. Then generate the full dataset and record wall time in `docs/results/v3.md`.
- [ ] **Step 5:** Commit the code only (`train_data/` is gitignored).

---

## Controller additions (binding; from the ledger)

- **R13.** Each shard also stores `cards: int8 [N, 15]` (button hole 5, BB hole 5, board 5), so outcomes are verifiable and buckets recomputable.
- **T16 deterministic buckets.** `bot.abstraction.bucket()` ignores `iters`/`rng` (exact deterministic features). `--iters` therefore means nothing here: drop it, or accept and ignore it with a documented note.
- **T16 note (hash).** Every shard stores a sha256 of `data/buckets.npz` (and the counts it implies). The loader verifies it against the table the trainer will use and refuses a mismatch. A refit invalidates datasets and blueprints.
- **R23 (portable, many-core, resumable).**
  - Linux/macOS, `requirements-train.txt` only, spawn Pool (as in `train/fit_buckets.py`).
  - Scales to every core: `--workers` defaults to `os.cpu_count()`.
  - Checkpoint/resume: write shards atomically (temp file, then rename). A rerun skips finished shards. Raising `--n` later adds shards on top of the existing ones.
  - The output depends only on `--seed`, the shard size and the numpy version, never on `--workers`: per-shard seeds `(seed, shard)`.
  - Add a runbook section to `train/README.md`.
- **R26 + PC sizing (decouple bucket counts from generation).**
  - Bucket counts are NOT final: Task 18 chooses them from measured trainer throughput, and may refit `data/buckets.npz` with more buckets.
  - So each shard also stores the raw per-seat, per-street bucket features: whatever `BucketTables.bucket_from_features` needs (preflop strength; flop/turn strength + equity; river equity + nut/category). Choose compact dtypes; float32 is fine.
  - Provide a re-bucket path (CLI flag or function) that recomputes `buckets` from the stored features against a given buckets table in seconds per million deals, without regenerating cards or features, and updates the stored hash.
  - **Precondition to verify and test:** for a given table, bucket-from-stored-features == `bucket()` on the stored cards.
- **For Task 18 (consumer).**
  - Provide `load_deals(path_or_dir, ...)`: concatenates shards, verifies hashes/meta, and returns at least the contiguous `buckets` int16 [N,2,4] and `showdown` int8 [N] plus meta, without materializing the features or cards unless asked.
  - The trainer will likely be one process running numba parallel threads over shared arrays.
  - Keep shards uncompressed or cheap to load: loading 20M deals must take seconds, not minutes.
- **Sizing on this PC.** AMD Ryzen 7 9700X, 8 cores / 16 threads (SMT siblings = logical CPUs 2k, 2k+1), 15 GB RAM in WSL2, Ubuntu 24.04, Python 3.12.3.
  - Measure throughput (deals/s) at 1, 4 and 8 workers. T16 measured about 4,600 postflop spots/s per Mac core, and a deal needs 6 postflop + 2 preflop features.
  - **Acceptance run:** the plan's 2,000,000 deals. Record wall time and throughput in `docs/results/v3.md` (create it) and `train/README.md`.
  - **Full target:** 20,000,000 deals. The controller extends to it in the background after review (unless you finish it within your time box). Report the disk size and the `load_deals` time/RAM for 2M and the projection for 20M.
- **CPU etiquette (ruling R32).** Track A runs timing-sensitive work on logical CPUs 0–7 at the same time as you.
  - Run every heavy job (generation, throughput runs) as `taskset -c 8-15 nice -n 19 <cmd>`, with `--workers` ≤ 8.
  - Never put heavy load on CPUs 0–7. The default test suite may run unpinned.
- **Tests (beyond the plan's).**
  - Shapes/dtypes.
  - showdown == `omaha_score` comparison on the stored cards.
  - buckets == `bot.abstraction.bucket()` on the stored cards (both seats, all 4 streets).
  - The 15 cards of a row are distinct and in range.
  - Determinism (same seed → identical shards; 1 vs 2 workers identical).
  - Resume after a simulated interruption (a partial temp file is ignored, and the finished result is identical).
  - Hash mismatch rejected by `load_deals`.
  - Rebucket == fresh bucketing.
  - Mark anything ≥ a few seconds `@pytest.mark.slow` (R11).
- **Constraints.** Branch `kevin-cfr` in `/home/kchen/code/heads-up-bundle-cfr`. Commit code (and docs) only, since `train_data/` is gitignored. Never push. Offline code may use numba/multiprocessing; runtime `bot/` stays stdlib + numpy (do not modify `bot/` beyond what is strictly necessary; if you must, say why).
