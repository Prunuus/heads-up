# Audit: Tasks 1–7 foundation (2189b5b..e4739a1)

Read-only audit, Opus (max effort), 2026-11-07. HEAD moved to d5c7812 during the audit (Tasks 8–9 committed concurrently). `git diff e4739a1 --stat` touches only Task 8 files, so every audited file is unchanged since e4739a1.

**Method.** All scratch scripts are in `/private/tmp/claude-501/-Users-kchen-Desktop-code-FALL26-heads-up-bundle/b4276c44-89ab-406f-9078-0e01542ecd72/scratchpad/`.

- `judge_xcheck.py`, `judge_xcheck_edges.py`, `f10.py`: these run the **organizer's** `judge_engine.play_half` / `combine_halves` with only `HttpAgentHandle._request` overridden. The override is an in-process copy of agent.py's SERVER PLUMBING: JSON round trip of request and reply, on_hand_end delivery with an error list, the answer mapping, Starlette's `allow_nan=False` rendering, and the deadline. The results are compared against `sim.match` on the same bot factories.
- `fasteval_exhaustive.py` checks all 376,992 five-card hands plus adversarial Omaha sets. `equity_exact.py`, `nutflags_check.py` and `timing.py` cover exact enumeration, brute force, constructed nut/tie/blocker cases and latency.
- `mutants.py`, `slow_mut.py` and `harness_vs_mutants.py`: 44 realistic bugs, each injected in a throwaway subprocess, then run against the repo's own Task 1–7 tests. This tests which bugs the tests would let through.
- Repo tests: 63 Task 1–7 tests pass. That is 6.8 s for `test_sim_match/fasteval/equity/repo_hygiene/tournament` plus 11.9 s for `test_zoo`, with 1 slow test deselected. Run with `-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1`.

---

### Critical

None found. I found no defect that currently produces wrong results or can forfeit a match:
- The simulator matches the real judge exactly on every deterministic scenario I tried, including all error and timeout paths.
- `fasteval` is bit-exact on every five-card hand.
- MC equity is unbiased against exact enumeration.
- `nut_flags` is bit-exact against brute force.

I1 is the closest to Critical: it is a live cross-task contract break that the next task will hit.

---

### Important

**I1: `iters=0` crashes the equity API, and Task 9's TimeManager returns 0 by design.** Owner: Task 7 (API). It hits Task 10, which consumes the API, and is triggered by Task 9.
- **Where.** `bot/equity.py:33-43` (`equity_vs_random`) and `:46-54` (`strength_now`). The crash happens at `bot/fasteval.py:90` (`.reshape(len(h), -1)` on an empty array).
- **What's wrong.** With `iters=0`, `_sample` returns shape (0, k) and `_best_of` raises `ValueError`. Task 9 (committed d5c7812) defines `LADDER = (0, 16, 32, 64, 128, 256)` at `bot/timing.py:28`. `iters()` returns 0 at the 6 s hard stop (`:78`) and when pacing drops below level 16 (`:86`).
- **Failure scenario.**
  1. Task 10 calls `compute_features(obs, iters=tm.iters(...), rng)`, which calls `equity_vs_random(hole, board, 0, rng)` and gets a `ValueError`.
  2. Under Task 11's per-layer try/except, the heuristic layer then fails on every MC decision for the rest of the half.
  3. Play falls back to `CHECK_CALL if to_call == 0 else FOLD`. Our bot becomes exactly the check-folder that our own I2 detector exploits. The 6 s stop is reachable on a slower tournament CPU.
  4. If any path lets the exception escape `choose_action`, each one is a counted error; 5 errors forfeit the half and lose the match.
- **Evidence.** `equity_vs_random([0,4,8,12,16], [], 0, rng)`, the same with board `[20,24,28]`, and `strength_now(..., 0, ...)` all print `raises ValueError cannot reshape array of size 0 into shape (0,newaxis)`.
- **Fix.** Give the API a defined zero-budget contract before Task 10 consumes it. Either `if iters <= 0: return None` (callers must branch), or route to a cheap deterministic proxy such as made-hand category or the preflop table, and document the choice. Add `test_zero_iters_is_defined`, plus a Task 10 test that drives `compute_features` with a TimeManager stub returning 0.

**I2: `omaha_batch` / `made_now_batch` silently return one wrong value for a broadcast hole.** Owner: Task 6.
- **Where.** `bot/fasteval.py:74-90` (`_best_of`, `reshape(len(h), -1)` at line 90).
- **What's wrong.** Broadcasting one board works: (N,5) holes against a (1,w) board gives a correct (N,) result, so the API looks broadcast-friendly. The reverse does not: one hole (1,5) against N boards reshapes with `len(h)=1` and returns shape (1,), the **max over all N boards**. There is no error.
- **Failure scenario.** A Task 10/16/17 helper scoring one hand over many sampled boards calls `made_now_batch(np.array([hole]), boards)`. It gets `array([6771760])` instead of 50 per-board scores. Any feature or bucket built on that is silently wrong.
- **Evidence.** `fasteval_exhaustive.py` prints `hole[None] x 50 boards -> (1,) [6771760] | correct per-board result shape (50,)` and `omaha hole[None] x 50 boards -> (1,)`.
- **Fix.**
  - Set `n = max(h.shape[0], b.shape[0])`.
  - Raise when `h.shape[0]` or `b.shape[0]` is not in `(1, n)`.
  - Use `reshape(n, -1)`, or simply require equal leading dims.
  - Add a test for both broadcast directions.

**I3: the simulator's "fresh process per half" covers only the top-level bot file.** Everything that file imports (`bot.*`) is shared across halves, matches and candidates. Owner: Task 2. Settle this before Task 11 wires agent.py to bot.policy.
- **Where.** `sim/match.py:37-51` (`load_bot_file`) and `:159-170` (`start_half` calls the factory once per half).
- **What's wrong.**
  - `load_bot_file` gives agent.py fresh globals.
  - But `import bot.policy` (and bot.timing, bot.opponent, …) hits the `sys.modules` cache. Their module-level objects survive into half 2 and into every later match in the same worker or process.
  - The tournament runs each half in a fresh pod precisely because half-2 cards are the half-1 cards with seats swapped (`judge_engine.py:10-19`).
- **Failure scenario.**
  - R5 has agent.py delegate to a module-level `PolicyBot()`, with module-level `start_half/decide/hand_finished` wrappers in bot/policy.py.
  - Suppose that instance, or any `bot.*` global, holds state: an opponent-model cache, a showdown log, the TimeManager, an RNG. Then half-1 knowledge reaches half 2. The bank looks partly spent, or the bot recognises boards whose opponent holding it saw at showdown in half 1.
  - The sim then reports an edge, or a slowdown, that the tournament cannot reproduce, and tuning optimises for it.
  - The same leak crosses candidates when `paired_diff` runs A and B in one process (its default is `workers=1`).
- **Evidence.** Scratch `leaky_bot.py` imports `bot.fasteval` and counts halves in a module attribute. Loaded twice via `load_bot_file`, it prints `[1, 2]`; fresh processes would give `[1, 1]`. `sys.modules["bot.fasteval"]` is the same object both times.
- **Fix.**
  - In `load_bot_file`, evict repo-local runtime packages (`bot`, `bot.*`) from `sys.modules` before `exec_module`. The import happens outside the timed region, so this costs nothing in sim terms. Alternatively, run candidate halves in fresh subprocesses.
  - Add a test with a scratch bot that keeps a module-level counter in a `bot.*` module and asserts it reads 1 in both halves.
  - Put the invariant "no per-half state outside `memory`" in the Task 11 brief.

**I4: `tests/test_sim_match.py` lets real judge-fidelity bugs through, and the F10 anchor is deselected although it takes 0.29 s.** Owner: Task 2.

Mutation evidence: each bug was injected into `sim.match` in a fresh subprocess, then all 14 default tests were run.

| Mutant | Result | Why |
|---|---|---|
| S1: on_hand_end errors counted once per request, not once per hand | **survives** | Tests never carry more than one result per request. The judge counts each (`judge_engine.py:388-392`). |
| S3: on_hand_end time not charged to the bank | **survives** | Plan Task 2 Step 3 requires it; nothing asserts it. |
| S8: button fixed at seat 0 | **survives** the default suite | Killed only by `test_matches_real_judge_value`, which is `@pytest.mark.slow` (`tests/test_sim_match.py:50`) and never runs by default, yet takes **0.29 s** (`pytest -m slow --durations`). |
| S9: grace not applied | **survives** | `test_slow_bot_error_and_bank` (`:84`) asserts only `bank_a > 0`. |
| S10: illegal-but-valid Action accepted | **survives** | No test returns, say, FOLD when to_call == 0, the commonest LLM-bot bug. With the mutant the sim raises `IllegalActionError` instead of counting an error. |
| S14: new_half failure does not forfeit | **survives** | Not tested. |
| S15: half-2 nets paired with the wrong deal (reversed) | **survives even the slow F10 test** | Mean is unchanged (−0.5160), sd is 1.0922 vs 0.9852, verdict still B. `test_identical_bots_chop` (`:57`) checks only mean and verdict. |

Killed: S2, S4, S5, S6, S7, S11, S12, S13.

Missing tests (each kills the mutant in brackets):
- (a) Unmark F10 and also assert `m.sd_bb == approx(0.98525, abs=1e-4)` [S8, S15].
- (b) Add `assert m.sd_bb == 0` to `test_identical_bots_chop` [S15].
- (c) Play an on_hand_end crasher against `sim/opponents/starter.py`. The starter folds as button, so several results arrive in one request; assert errors equal the number of crashing results [S1].
- (d) A bot whose on_hand_end busy-waits 5 ms must be charged that time [S3].
- (e) `play_half(..., grace_s=0.05)` with a bot that waits 10 ms per action must give `bank_a == 0` [S9].
- (f) A bot that returns FOLD when to_call == 0: errors are counted, the default action is played, nothing raises [S10].
- (g) A new_half that raises, or sleeps past the action timeout, must give `forfeit_a` and `points_a == 0` [S14].
- (h) **Best single guard:** a judge-equivalence test. Run `judge_engine.play_half` with `HttpAgentHandle._request` overridden by an in-process copy of agent.py's plumbing (my `judge_xcheck.py`, about 80 lines, under 2 s). Assert identical nets, errors, forfeits and calls versus `sim.match.play_half` for deterministic bots plus an error-mix bot. It kills S1, S8, S10 and S15 in one run (`harness_vs_mutants.py`).

**I5: `paired_diff`'s CI width is untested, and `paired_diff` is the acceptance gate.** Owner: Task 5.
- **Where.** `sim/tournament.py:146-167` and `tests/test_tournament.py:13-22`.
- **Mutation evidence.**
  - T4 survives all 6 tests: `half = Z*sd/sqrt(#matches)` instead of `#deals`, a CI √deals ≈ 22× (500 deals) to 31× (1000 deals) too wide.
  - Also surviving: T2 (BOTH_FORFEIT gives B a point), T7 (standings sorted worst-first) and T10 (`resolve_bot("agent")` pointing at the wrong file; `test_resolve_bot:43` only checks `callable`).
  - T1, T3, T5 and T6 survive too, confirming the already-logged B-side gaps.
- **Failure scenario.** Task 14's gates (`paired_diff(...)` lower CI > 0; HOLDOUT lower CI > −0.05) and every v1/v2/v3 acceptance read this CI. A width regression would make gates pass or fail for the wrong reason, with no test signal.
- **Current math is right.** `paired_diff` gives (1.038458, 2.948958, 4.859459) with N=480, identical to my manual recomputation from `_per_deal_bb`. Pooling across heterogeneous opponents is mildly conservative: pooled SE 0.975 vs stratified 0.961.
- **Fix and missing tests.**
  - Factor the CI into a pure helper and test it against a hand-computed vector.
  - Add one integration assert `hi - lo == approx(2*1.96*sd/sqrt(N))`, recomputed from `_per_deal_bb`.
  - Assert `resolve_bot("agent").args[0] == "agent.py"`.
  - Add a BOTH_FORFEIT round-robin case (see M5).

**I6: `strength_now`'s "made hand now, no future cards" semantics is untested.** Owner: Task 7. This was logged as minor ("no strength_now timing/width-4/5 value tests"); I judge it **more severe than logged**.
- **Where.** `bot/equity.py:46-54` and `tests/test_equity.py:64-77`.
- **Mutation evidence.** E12 replaces the body of `strength_now` with `return equity_vs_random(hole, board, iters, rng)`, and all 9 equity tests still pass. The only flop test (AAA+KK on Kh Kd 7c, > 0.9) and the river-equality test cannot tell strength from equity.
- **Why it matters.** `strength_now` is one axis of the planned 2-D buckets (Task 16) and a heuristic feature (Task 10). A regression would silently collapse the 2-D abstraction to 1-D.
- **Current code is correct.** Exact made-hand enumeration on 2 flops, 2 turns and 2 rivers agrees within |z| ≤ 1.5 at 200k iterations (`equity_exact.py`).
- **Missing test.** Use a flop with a weak made hand but high equity, such as a big wrap plus the nut flush draw. Compute the exact answer over C(28,5) = 98,280 opponent hands with `made_now_batch` (about 30 ms). Assert `|strength_now − exact| < 4·SE` and `strength_now < equity_vs_random − 0.15`.

---

### Minor

**M1: the simulator sets `AgentConfig.num_hands = deals`** (`sim/match.py:238`). Owner: Task 2.
- 5 of the 8 LLM bots pace their time by `config.num_hands`: `beginner_2h.py:71/119`, `aggressive.py:80/89`, `mc_equity.py:68/124`, `ml_minded.py:205/309`, `opp_model.py:135/197`. Our TimeManager will do the same if it is fed the config.
- In 100–500-deal sims (zoo test 100, zoo report 300, plan Task 12 tuning 500) these opponents get 2–10× the per-decision thinking budget they would have in the tournament.
- A fixed-cost bot like `mc_equity_slow` exhausts its bank after a roughly fixed number of decisions, so the exhausted share of a half grows with `deals`. Short-match measurements of the I2 check-fold exploit therefore don't transfer to 1000 deals.
- Fix: add an option such as `config_hands=1000` to `play_half`/`play_match` for tuning runs, or document that time-related conclusions need `deals=1000` (as R20 requires).

**M2: no `evaluator.warmup()` before play in `sim.match`.** The judge calls it at `judge_engine.py:1000`. Owner: Task 2/3.
- In each fresh process (pytest, or every spawn worker), the first archetype that calls `omaha_score` builds the 376,992-entry table inside its timed request. That is McEquity, CheckFoldAfter, or GuideExample on the river.
- Evidence: `mc_equity_fast` bank after 3 deals is 134.8 ms cold vs 14.1 ms warm; max single request 126.2 ms vs 3.7 ms. Real agent.py-style bots warm up at import.
- Fix: call `evaluator.warmup()` at `sim.match` import or at the top of `play_half`.

**M3: one error per stall in the sim, several in the real judge.** `sim/match.py:146-151` vs `judge_engine.py:273-275`. Owner: Task 2.
- A request that stalls ≥ 2 s gives one error in the sim. In the real judge, later requests queue behind the still-running computation and time out too, so a stall of more than 5 s forfeits for real but not in the sim. The bank charge is roughly equivalent.
- This only affects opponents with multi-second stalls; there are none in the zoo (H6 = 0/19). Document it, or count `ceil(elapsed/timeout)` errors when not bank-limited.
- Stats-only nit: on a factory crash the judge reports `calls=1` (the failed connection) and the sim reports 0. Errors and forfeit match.

**M4: the `sys.modules` leak, quantified** (`sim/match.py:37-51`; already logged as minor).
- `llm_ml_minded` leaks **6.1 MB RSS per `load_bot_file`** (2 per match, 143 ms import); other bots leak 0.05–0.2 MB.
- It is bounded when `workers > 1`, because `_run_jobs` builds a new Pool per call.
- It is unbounded in-process: `paired_diff` defaults to `workers=1` (`sim/tournament.py:147`), so a long-lived tuner or gate process accumulates about 12 MB per match against ml_minded. A growing heap may also lengthen GC pauses that land in timed requests (not measured).
- Keep the logged severity unless serial long runs are planned; in that case pop the module after the half or use `maxtasksperchild`.

**M5: BOTH_FORFEIT is not counted in `losses` for either side** (`sim/tournament.py:112-113`). Owner: Task 5.
- Evidence: a crashA/crashB/cs round robin gives each crash bot `losses=1` (only the loss to cs), while points 0/0 are correct.
- Fix: `losses[a] += m.verdict in ("B", "BOTH_FORFEIT")`, and the same for `b`.

**M6: default `workers = os.cpu_count()` (10 here)** in `run_round_robin` (`sim/tournament.py:73`), `focus_eval` (`:127`) and the CLI (`:275`). Owner: Task 5.
- This contradicts R17. Contention inflated the zoo report's `mc_equity_fast` max bank to 5.3 s at 300 deals; I measured 1.9 s per **1000-deal** half single-process.
- Fix: default to `min(4, cpu_count)`.

**M7: seeds are always `s1..sN`** (`sim/tournament.py:263-264`, `--seed-prefix` default `"s"` at `:273`). Owner: Task 5/12.
- Task 12 tuning (`--seeds 4`) and held-out evaluation (`--seeds 5`) share s1–s4 unless someone remembers the flag. The plan asks for held-out seeds as well as held-out opponents.
- Fix: use distinct default prefixes per mode, or a guard.

**M8: `nut_flags` bit3 semantics** (`bot/equity.py:100-104`). Owner: Task 7.
- It is set on the river for a dead two-board-card "draw": As Ts with board Ks 9s 6d Jc Qd gives `1000`.
- It is clear for the effective nut-flush draw when the suit's ace is on the board: Ks Ts with board As 9s 6d gives `0000`.
- This follows the literal spec but is misleading as a "nut flush draw" feature. Fix: require `len(board) < 5` when `bsuit[s] == 2`, and test for "highest unseen card of the suit" instead of the ace.

**M9: invalid inputs pass silently in `fasteval`** where `evaluator` raises (`bot/fasteval.py:57-68`). Owner: Task 6.
- A duplicate card, `[6c,6c,7c,8c,9c]`, scores 0 (`score_five` raises `ValueError`). Card −1 is read as As, giving a royal flush.
- The internal sampler is fine (verified: no duplicates). A debug-mode assert, or a test helper, would catch upstream card-generator bugs in Tasks 10/16/17.

**M10: one zoo-report classification is an artifact** (`docs/zoo_report.md:62`; rows at `:20` and `:46`). Owner: Task 5.
- `mc_equity_slow`'s H4 "folds a lot preflop" (50.3%) and its 64.6% fold-facing rate are bank-out default folds: its bank was exhausted in 36/36 halves. That is H1 behaviour, not a strategy trait.
- Exclude bank-exhausted bots from H2/H4, or footnote it.

**M11: `.DS_Store` is tracked** (from the initial commit), is not in `.gitignore`, and ships in the image. Harmless; fix with `git rm --cached .DS_Store` and an ignore entry. Owner: Task 1.

---

### Verified correct

Simulator fidelity:
- **Seats, deals and button.** `play_half` uses `agent1_seat_for_half`, `make_deal(seed, d)` and `button_for_deal(d)` exactly as the judge does (`sim/match.py:236-248` vs `judge_engine.py:661-677`). Nets were identical in 24 half-by-half comparisons: 6 deterministic pairs × 2 seeds × 2 halves × 150 deals (`judge_xcheck.py`, all OK).
- **Error and forfeit counting.** Errors and forfeits match the organizer judge for:
  - a crash every 7th call;
  - on_hand_end crashes on every hand, and on every 50th;
  - a new_half crash;
  - 13 odd return values: `np.int64(1)`, `1.0`, `"1"`, `True`, `None`, `"CHECK_CALL"`, int `1`, a StrEnum, a foreign IntEnum, a tuple, `np.float64`, an illegal RAISE_POT, and `"fold"` (`judge_xcheck_edges.py`).
- **Timeout vs bank-out.** Results are identical for 40 ms requests under a 20 ms limit, for a slow new_half (which forfeits), and for bank-limited overruns. Under a 0.3 s bank the exact bank-out point differs by a few calls. That is timing jitter, because the judge emulation adds JSON overhead per request; it is not a logic difference.
- **Time charging.** One timed region covers the on_hand_end deliveries plus `choose_action` (`sim/match.py:207-211`), and the end-of-half flush is timed too (`:225-228`). Grace is 0.75 ms, per ruling.
- **Pending results.** They are queued only when not forfeited, delivered before the next action, and flushed at the end only if the bot is usable, matching `judge_engine.py:447-480`. Mutants S2 and S4 were killed.
- **Verdict and points.** The sim calls `judge_engine.combine_halves` directly. F10 reproduces exactly: mean −0.516, sd 0.985249, CI (−0.57707, −0.45493), AGENT2_WIN, 0 points (`f10.py`). Nit vs maniac (0.75875, CHOP) and threshold_cat vs random (2.17625, A) are identical, and the per-deal paired vectors are identical.

`bot/fasteval.py`:
- **Table key.** The base-5 rank-count key maps one-to-one onto (rank multiset, flush) across all 376,992 hands. The largest index, 2,441,250, is below the table size of 3,906,250. int32 is safe: the largest score is below 2^24.
- **Exactness.**
  - `score5_batch` equals `evaluator`'s table on all 376,992 hands, also with shuffled card order.
  - `omaha_batch` equals `omaha_score` on 150k random, 60k two-suit, 60k wheel-ranks-only and 30k monotone-board deals.
  - `made_now_batch` equals brute force on 40k hands each for widths 3, 4 and 5.
  - The wheel and the wheel straight flush are correct (mutants F4 and F6 killed).
- **Omaha flush detection.** It requires the hole pair to be monochrome and the board triple to be monochrome in the same suit (`bot/fasteval.py:84-89`). Mutant F1, which ignores suit equality, is killed by 4 tests.

`bot/equity.py`:
- **Unbiased MC.** Against exact enumeration on 6 river, 3 turn and 6 made-hand spots at 200k iterations: max |z| 1.76, mean z −0.42.
- **Sampler.** An argsort of iid uniforms gives uniform draws without replacement. Per-card inclusion was 0.03547–0.03586 against 1/28 = 0.03571. There are no duplicates, seen cards are never drawn, and results are deterministic given the Generator.
- **Ties count 0.5** (`bot/equity.py:29-30`). Mutants E1 and E2 were killed.
- **`nut_flags` is bit-exact** against brute force on 1,500 random spots (108 with bit0 set) and 12 constructed cases. Those include card removal (holding Th makes a K-high straight flush the effective nuts), a full house where quads are blocked, and a tie (KKK77 vs KKK77 sets bit0). The opponent is held to exactly 2 hole + 3 board cards, and our hole cards are removed.
- **Performance.** Median timings: equity at 128 iterations 0.18–0.21 ms (p99 ≤ 0.21); at 256 iterations 0.33–0.37 ms. `strength_now` at 128 iterations 0.12–0.19 ms. `nut_flags` 0.04–0.11 ms. `fasteval` warmup 2.8 ms. This is consistent with the ≤ 0.7 ms median `/act` budget on this laptop.

`sim/tournament.py`:
- Points (including BOTH_FORFEIT 0/0), the sign flip of B's mean_bb, and forfeit and max_bank attribution are correct, by reading and by the crash round robin.
- `paired_diff` equals a manual recomputation.
- Parallel results equal serial results for deterministic bots, factories pickle, and `resolve_bot` resolves agent, ZOO names and paths.

Opponents:
- The archetypes match their specs. GuideExample matches guide §4.1 line for line; nit, maniac, the pot-odds MC bots, threshold_cat and checkfold all behave as specified.
- The archetype MC sampler and all 8 LLM bots' samplers draw without replacement from the correct unseen set.
- The LLM bot files are byte-identical to the generator outputs in `llm-raw/`.
- `mc_equity_fast` uses 1.89–1.97 s of bank per 1000-deal half single-process, so it is not a disguised bank-out bot.
- HOLDOUT (4 bots) and TRAIN (15) partition ZOO, and `select_opponents` uses them. No tuning code exists yet that could misuse them.

Global constraints:
- `bot/*.py` parses under the 3.12 grammar and imports only `__future__`, `itertools`, `typing`, `numpy`, `evaluator` and `bot`.
- `game.py`, `evaluator.py`, `judge_engine.py`, `baseline-bots/` and `docker/` are unchanged since e9ce98d. The only change under tools/ is the untracked `tools/__pycache__` .pyc, as the plan says.
- `requirements.txt` keeps its pins and adds numpy.
- `.dockerignore` keeps agent.py, bot/, data/, game.py, evaluator.py and requirements.txt.
- All 10 commits carry the trailer. `kevin` is 13 commits ahead of `origin/kevin`, so nothing was pushed.

Test status:
- All 63 Task 1–7 tests pass (18.7 s).
- Of the 44 mutants, the `bot/` tests kill 18 of 19; only E12 survives. The `sim/` and `tournament` tests kill 10 of 25; details are in I4 and I5.
- The equity reference test is seeded and its margins are about 3.6σ per spot and 5σ on the mean, so it is not flaky.
- The sleep-based sim tests have margins of 2× or more.
- The machine-dependent speed tests were already logged.
