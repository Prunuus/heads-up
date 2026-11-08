### Task 12: Tune the heuristic, then ship v1

**Files:** Create `sim/tune.py`. Modify the `HeuristicParams` defaults. Results go in `docs/results/v1.md`.

**Interfaces — Produces:** `tune(param_space: dict[str, list], opponents="train", seeds: list[str], deals: int, budget: int) -> HeuristicParams`. Random search plus a coordinate-descent refinement, with the objective **expected points + 0.01·mean_bb** (D1). Each candidate is evaluated with common seeds.

- [ ] **Step 1:** Run the tuner: `.venv/bin/python -m sim.tune --budget 200 --seeds 4 --deals 500`.
- [ ] **Step 2:** Run the held-out evaluation: `python -m sim.tournament --candidate bot.policy --opponents holdout --seeds 5 --deals 1000`.
- [ ] **Step 3: Milestone M1 (gate).** Must hold:
  - a CI **win** vs all 4 baselines on 5/5 seeds
  - **0 errors** in every match
  - max bank use < 3.0 s per half
  - expected points on held-out opponents ≥ points on training opponents − 10%, which shows no overfit

  If it fails, iterate on Task 10's logic before moving on.
- [ ] **Step 4: Real-judge check.** Run 2 seeds vs `baseline_random` with separate halves and restarts (README §4), then `./tools/check_build.sh` → `ALL CHECKS PASSED`.
- [ ] **Step 5:** Commit, push, `git rev-parse HEAD` → **submit v1** through the form.

## Stage 2 — Exploit layer (ship v2)


---

## Controller additions (binding; from the ledger — read the lines cited)

- **R5/R3.** Candidates are `bot.policy.PolicyBot(params)` factories: picklable, evaluated in a multiprocessing pool. `python -m sim.tournament --candidate agent` exercises agent.py end to end.
- **R20 (user directive).** Strategy work is validated with real sim matches and iterated. Size the budget to the time box below.
- **R30b.**
  - Add a per-bot `time_scale` to `sim/match.play_half` / `play_match`. It scales charged time and timeouts, emulating a slower vCPU for BOTH bots.
  - Run tuning and evaluation at `time_scale ≈ 2.5` as the primary condition.
  - T11 note: the bot's online CPU factor reads about 1.0 under real uvicorn on this PC but about 1–2.5 in in-process sims. max(warmup, online) sets the grace, so sims over-charge, which is the safe direction. Make `time_scale` and the bot's own CPU estimate compose sensibly, and document how.
- **T10 notes for T12 (ledger "Task 10:" lines).**
  - **M3:** validate every param set the tuner generates: finite values, 3-tuples per street (`HeuristicParams.__post_init__` or a `validate()` the tuner calls). A NaN or a short tuple raises inside the policy, which gets caught as a layer fault and falls back: a silent strength loss.
  - **M6:** the non-nut penalty counts twice on flush boards (the range model's d term plus `nonnut_penalty`). Expose `nonnut_penalty` to the tuner and document the effect.
  - **M7:** ablate `now`. It is unused Monte Carlo, about 20–30% of flop/turn compute.
  - **M8 (protocol):**
    - The dev seeds s1–s3 read favourable (+2.960 vs +2.619 on fresh seeds).
    - Wall-clock-adaptive bots make runs non-reproducible.
    - So tune on a broad seed pool, confirm on DISJOINT fresh seeds, and compare candidates paired (common seeds, `paired_diff`).
    - Consider a fixed-iters clock for tuning determinism.
  - **Baselines** (T10 review, quiet PC, fresh seeds q1–q3): TRAIN mean +2.619 bb/deal, 15/15 opponents positive. Thin margins: llm_ml_minded +0.56 (6W/0L/8C over 14 seeds), llm_gto_inspired +1.07 (12/0/2). `task-10-report.md` "Next ideas" lists untuned params and logic ideas.
- **T11 notes for T12.**
  - **In-container validation** (T11 review ⚠️3; required in Step 4):
    - Play real-judge halves against our bot built from `docker/Dockerfile.agent`, run with `--cpus 1`.
    - Record the judge-charged bank per half, and the CPU factor warmup measured vs the online estimate inside the container (T11 M4: it reads 4.4 under the TestClient, which is context, not CPU).
    - Repeat the T10 reviewer's 5× CPU emulation with the T11 grace.
    - Use Docker as `sg docker -c "<cmd>"` (your shell lacks the docker group otherwise).
    - Never edit tools/ or docker/.
  - **M7:** add a targeted `filterwarnings` for the StarletteDeprecationWarning in `pytest.ini`, so test output is pristine.
- **Adaptive opponents (user info).** Other teams' bots track our stats over each 1,000-deal half and adapt.
  - Avoid static tells, e.g. bet size revealing strength (RAISE_POT iff equity > `pot_vs_half_cut` is one); consider mixing.
  - Weigh results against the adaptive zoo bots.
  - Report whether any adaptive bot gains on us over the half: first-500 vs last-500 deals, bb/deal.
- **Compute budget.**
  - The plan's `--budget 200 --seeds 4 --deals 500` over 15 TRAIN opponents is about 6–7 h on this PC, which is too long.
  - Size the search so the tuning run itself takes ≤ ~90 min wall, with `--workers ≤ 6` pinned `taskset -c 0-7`.
  - Spend the rest of the time box on confirmation (fresh disjoint seeds) and the M1 gate. Document the budget you chose and why.
- **CPU etiquette (R32).**
  - Track A owns CPUs 0–7.
  - Task 14 (Track C, in parallel) runs its sims on 0–7 at nice 10 with ≤ 2 workers.
  - CPUs 8–15 carry Track B training work and background jobs: don't use them.
  - Report load averages with every timing number.
- **R12.** Never push, never submit. Plan Step 5 ("commit, push, submit v1") becomes: commit and report the SHA. The controller runs check_build on it and asks the user about pushing and submitting.
- **Holdout discipline.**
  - HOLDOUT is for the M1 gate only, never for tuning decisions.
  - The controller may register a new FIELD split (3 agentic competitor bots) during your task. If it exists at gate time, also report v1 vs FIELD (information only; target: no CI loss).
- **Commits.** One simple sentence each, with no body and no trailers (R36).
- **Time box.** About 2.5 h of wall time in total.
