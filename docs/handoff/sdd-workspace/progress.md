# SDD ledger — plan: docs/superpowers/plans/2026-11-07-plo5-poker-bot.md

Branch: kevin (user's own branch; origin = github.com/Prunuus/heads-up). Plan committed at 2189b5b.
Spec: context/prompt.md + context/Heads Up Participant Guide.pdf + README.md (reachable, untracked in git).
Env: .venv = Python 3.14 (tournament = 3.12, no local 3.12/uv; Docker NOT installed). 10 CPU cores.
Baseline tests: 12 passed (test_agent.py).

## Pre-flight scan

| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T2 ↔ T11 (agent.py) | T2 test loads agent.py expecting starter bot (-0.516); T11 rewires agent.py | CONFLICT → R2 |
| T3 ↔ T5 (ZOO factories) | T3 ZOO values; T5 multiprocessing Pool needs picklable factories | CONFLICT (lambdas unpicklable on macOS spawn) → R3 |
| T3 ↔ T13/T14 | `checkfold_after(n)` used as factory by T13/T14 | needs exported function → R3 |
| T4 internal | plan says implementer dispatches 8 subagents; SDD forbids implementers dispatching subagents | CONFLICT → R4 |
| T5 ↔ T11/T12 | `--candidate bot.policy`; bot.policy exposes start_half/decide/hand_finished, not BotLike; T12 tuner needs param-built bots | CONFLICT → R5 |
| T5 internal | `paired_diff` "mean and CI" vs test unpack `lo, mid, hi` | order ambiguous → R6 |
| T6 internal | interface text: sorted base-9 table; speed test < 10 ms for 1M 5-card evals | sort likely too slow → R7 |
| T7 internal | consistent | ok |
| T8 internal | `test_strength_range` [0.30,0.70] is a guess from F6 p5–p95 | may fail on extremes (e.g. four sixes) → R8 |
| T9 ↔ T11 | begin()/end() "wrap each request" but bot can't see request boundaries | → R9 |
| T10 ↔ T13 | heuristic `opp=None` param; OpponentModel arrives T13 | ok (optional param) |
| T11 ↔ T13/T14/T19/T20 | policy layers blueprint/exploit/opp not yet built at T11 | → R10 |
| T11/T14/T20 tests | sim-heavy tests in default suite → very slow `pytest -q` | → R11 |
| T12/T14/T20/T24 | "push", "submit", `./tools/check_build.sh` | push = stop-class side effect; submission is user's form; Docker missing → R12 |
| T15 internal | consistent; build < 2 s | ok |
| T16 ↔ T17/T18 | bucket(...) ints; NUM_BUCKETS dims [n_dec,200,4] | ok |
| T17 internal | test needs "regenerated deals" but schema stores no cards | → R13 |
| T18 internal | 6 h training run inside an implementer | → R14 |
| T19 internal | blueprint size: 15,952×200×4 fp16 ≈ 25.5 MB (plan says ~13 MB) | harmless; < 100 MB ok |
| T20 ↔ T21 | T20 gate (d) needs LBR from T21, which comes later | ORDER CONFLICT → R15 |
| T22/T23 | stretch | gated; run only if budget allows → R16 |
| T1 internal | team name unknown | → R1 |

## Rulings

- Ruling R1: PARTICIPANT="KChen", AGENT_NAME="ShortDeckHybrid" placeholders — user never gave a team name — cost if wrong: 1-line edit before submission (must match registration form name).
- Ruling R2: T2 cross-check uses a frozen copy of the starter strategy at `sim/opponents/starter.py` (check-if-free-else-fold, same 3 functions), NOT agent.py — agent.py is rewired in T11 — cost if wrong: none (same strategy as starter).
- Ruling R3: every ZOO value and every factory passed to multiprocessing must be picklable (a top-level class, or functools.partial of a top-level function); `sim.opponents.archetypes.checkfold_after(n) -> BotFactory` is exported and ZOO registers `checkfold_after_300` — macOS uses spawn — cost if wrong: refactor of zoo registration.
- Ruling R4: the controller (not the implementer) dispatches the 8 LLM-bot generator subagents (sonnet), each writing one file into `sim/opponents/llm/raw/`; a T4 implementer then adapts/registers them — SDD forbids implementer-spawned subagents — cost if wrong: none.
- Ruling R5: T11 adds `class PolicyBot(params: HeuristicParams | None = None, use_blueprint: bool = True, use_exploits: bool = True)` in bot/policy.py with new_half/choose_action/on_hand_end (BotLike); agent.py delegates to a module-level `PolicyBot()`; module-level `start_half/decide/hand_finished` remain as thin wrappers. T5 `--candidate` accepts `agent` (load_bot_file("agent.py")), a ZOO name, or a bot file path — cost if wrong: small API change.
- Ruling R6: `paired_diff` returns `(ci_low, mean, ci_high)` — matches the test's `lo, mid, hi` — cost if wrong: none.
- Ruling R7: T6 may use any table encoding (recommended: additive rank-count key Σ 5^rank, table int32[2·5^9], no sort) as long as results are identical to evaluator; the < 10 ms speed test uses best-of-5 warm runs — sort-based base-9 likely misses the speed spec, which is binding — cost if wrong: a few MB more table memory.
- Ruling R8: if any preflop table value falls outside [0.30, 0.70], the implementer reports actual min/max and may widen to [0.25, 0.75]; beyond that is a bug — the bound was an estimate — cost if wrong: a weak sanity check.
- Ruling R9: TimeManager.begin()/end() wrap each choose_action call (grace applied per call); on_hand_end time is added via `add(dt)` with no grace — bot can't see HTTP request boundaries; conservative — cost if wrong: slight over-counting (safe direction).
- Ruling R10: T11 builds the layer chain with empty slots: blueprint=None (until T20), exploits=identity (until T14), memory["opp"]=None (until T13). Missing data files must never crash warmup — cost if wrong: none.
- Ruling R11: tests that play ≥ 1 full sim match or tuning runs get `@pytest.mark.slow`; a root `pytest.ini` registers the marker and sets `addopts = -m "not slow"`; run slow ones with `-m slow` — keeps README's `pytest -q` fast — cost if wrong: slow tests need explicit runs.
- Ruling R12: never `git push` and never submit (user's form, user's call); Docker isn't installed so `./tools/check_build.sh` is skipped and flagged to the user; milestone gates use the in-process sim + real `judge_engine.py` runs instead — cost if wrong: an image-build problem surfaces only when the user runs check_build/submits.
- Ruling R13: T17 dataset also stores `cards: int8 [N, 15]` (button hole 5, BB hole 5, board 5) so outcomes are verifiable and buckets can be recomputed — cost if wrong: ~15 bytes/row.
- Ruling R14: T18 implementer only verifies throughput + a ≤ 20-min training run; the long run is launched by the controller in the background after T18 review, capped at 3 h wall (user flagged usage concerns) — cost if wrong: weaker blueprint.
- Ruling R15: execute T21 (LBR) before T20 so the T20 gate can use it — plan order defect — cost if wrong: none.
- Ruling R16: T22/T23 stretch run only after T20 completes and only if the session is still alive — cost if wrong: they stay unimplemented.

## Progress
Task 1: implementer done (8617b76), concerns: commit trailer model name; placeholder team name (R1). Under review.
LLM generators (R4): 8 dispatched in parallel → workspace llm-raw/ (not repo).
Task 1: minor (deferred): test_names_set only checks old names absent, not new ones present (plan-mandated weak test)
Task 1: minor (deferred): hygiene tests use cwd-relative paths
Task 1: minor (deferred): .dockerignore __pycache__ not recursive (**/__pycache__)
Task 1: complete (commits 2189b5b..8617b76, review clean)
LLM generators: all 8 written to llm-raw/ (simple_rules, beginner_2h, aggressive, mc_equity, opp_model, gto_inspired, ml_minded, time_careful). numpy installed into .venv (was missing).
Task 2: Ruling: grace_s=0.00075 in-process (not judge's 0.001) — plan-specified: 1 ms grace minus ~0.25 ms measured HTTP overhead — cost if wrong: sim bank slightly pessimistic (safe direction).
Task 2: Ruling: actions pass through judge_engine.coerce_action (str/int accepted) — the real judge accepts what the HTTP plumbing forwards; judge is the authority over the brief's wording — cost if wrong: none (real-judge-faithful).
Task 2: minor (deferred): sim/match.py:137 _request docstring says 3-tuple, returns 2-tuple
Task 2: minor (deferred): sim/match.py:68 loaded bot modules never evicted from sys.modules (leak in long tuning runs)
Task 2: minor (deferred): sim/match.py:43 sys.path mutation at import uncommented
Task 2: fix round 1/5 (1 addressed, 0 open — timing/bank/flush tests added; commits 8340224..8fe07a5)
Task 2: minor (deferred): tests/test_sim_match.py:217 test_flush_is_one_request doesn't assert one request
Task 2: complete (commits 8617b76..8fe07a5, review clean)
Task 3: minor (deferred): tests/test_zoo.py:263 checkfold_after test doesn't assert the switch to default_action
Task 3: minor (deferred): archetypes.py:44 _legal_or unused; archetypes.py:152 nit preflop raise threshold (extra, untested)
Task 3: minor (deferred): no behavioural tests for maniac/guide_example/threshold_cat/MC bots
Task 3: complete (commits 8fe07a5..459fca4, review clean)
Task 4: Ruling: HOLDOUT = {llm_time_careful, llm_aggressive, llm_opp_model, threshold_cat} — one strong time-aware MC bot, one haiku aggressive bot, one adaptive bot: diverse styles kept unseen by tuning — cost if wrong: holdout less representative.
Task 4: minor (deferred): sim/opponents/__init__.py HOLDOUT/TRAIN mutable sets (frozenset suggested); FORFEIT_PRONE empty-on-purpose undocumented
Task 4: minor (deferred): ml_minded ~6 s per 100 deals (import-time fit) — sim budget note
Task 4: complete (commits 459fca4..543cf05, review clean)
Ruling R17: gating/tuning sim runs use --workers 4 (this Mac = 4 P-cores + 6 E-cores; 10 workers inflate bank use via E-cores and make time-sensitive opponents exhaust banks unrealistically); 10 workers OK for coarse surveys — cost if wrong: slower evaluation runs.
Task 5: minor (deferred): action-count test lacks default-after-error/bank-out case; tests/test_tournament.py:635 tautological assert; no test for B-side sign flip/losses/max_bank; CLI untested; f-string w/o placeholder (sim/tournament.py:485)
Task 5: complete (commits 543cf05..ecbb48d, review clean)
Task 6: minor (deferred): tests/test_fasteval.py:155 duplicate edge-case hand / mislabeled; bot/fasteval.py:43 TRIPLE_IDX unused duplicate of _TRIPLES[5]; speed test machine-dependent
Task 6: complete (commits ecbb48d..269a2ea, review clean) — 10k omaha = 3.2 ms (~140x ref), table 15.6 MB, build 5 ms
Task 7: implementer hit ECONNRESET mid-task (partial uncommitted bot/equity.py, tests/test_equity.py); resumed same agent.
Task 7: Ruling: reference test per-spot tolerance 0.03 → 0.05 plus new aggregate check |mean signed diff over 50 spots| < 0.01 — plan's 0.03 at 2000-vs-4000 iters is ~2.2σ, ~1.4 expected false failures per 50 spots (implementer seed-shopped to pass); 0.05 ≈ 3.6σ and the mean check (SE≈0.002) still catches systematic bias — cost if wrong: per-spot errors in 0.03–0.05 that average out go undetected.
Task 7: minor (deferred): nut_flags brute-force test rarely exercises bit0=True (add constructed nut/tie cases); odd .replace in test; set() rebuilt in loop bot/equity.py:119; no strength_now timing/width-4/5 value tests
Task 7: fix round 1/5 (2 addressed, 0 open — slow marker removed; tolerance 0.05 + mean check; commits 7068757..e4739a1)
Task 7: complete (commits 269a2ea..e4739a1, review clean) — p99 equity@128 flop 0.20 ms
Task 8: Ruling: runtime preflop lookup may key the full 376,992-hand table by 36-bit card mask (sorted int64 masks + float32 values, np.searchsorted) built offline from canonical classes; canonical_key still provided/tested — avoids 24-permutation canonicalization at runtime — cost if wrong: ~5 MB data file instead of ~0.1 MB.
USER DIRECTIVE (2026-11-07): "each task implemented with as much effort as possible. Don't be shy on usage."
Ruling R18: from Task 8 review onward, ALL implementers/reviewers/re-reviewers = model opus, effort max (supersedes Model Selection cost rules) — user directive — cost if wrong: higher usage (user accepted).
Ruling R19: R14 superseded — training budget back to plan target (≥200M sampled iterations or 6 h wall); R16 superseded — Tasks 22/23 are planned work, not optional — user directive — cost if wrong: longer run time.
Ruling R20: strategy tasks (10, 12, 14, 16–23) require implementers to validate strength with real sim matches (sim.tournament, --workers 4, full 1000-deal matches, multiple seeds) and iterate, not just pass unit tests; tuning budgets enlarged — user directive.
Audit: Opus/max read-only audit of Tasks 1–7 (2189b5b..e4739a1) dispatched in parallel with Task 8 implementer.
Task 8: Ruling: R8 bounds replaced — test asserts all ≥ 0.05, non-quad ≥ 0.20, all ≤ 0.75, median ∈ [0.45, 0.53] — 288 four-of-a-kind hands legitimately score 0.099–0.322 (Omaha uses exactly 2 hole cards → quads play as one pair; 20k-rollout re-check of four sixes = 0.106) — cost if wrong: none (evidence-backed).
Task 8: minor (deferred): bot/preflop.py:47,21,26 narrow numpy int dtypes overflow `1 << c` (use int(c)); no test that canonical_key distinguishes classes (assert 19998 unique keys, n_classes/rollouts metadata, 24-relabel strength equality)
Task 8: note for Task 11: agent.py warmup must call preflop load (first /act otherwise pays ~15 ms)
Task 8: complete (commits e4739a1..f4dfb8e, review clean; reviewer regenerated table bit-for-bit, Burnside count 19,998 confirmed)
Task 9: Ruling: "drop one level per 20% overshoot" rounds UP (10% over → 1 level) — conservative toward bank safety — cost if wrong: slightly fewer rollouts when marginally over pace.
Task 9: Ruling: hard stop = 0.75 × bank_s (= 6.0 s at default 8.0) — Review Focus 1 (bank may be retuned) — cost if wrong: none at default config.
Task 9: carry-forward to Tasks 10/11: fixed per-decision cost outside MC must stay ≤ ~0.3 ms on dev laptop; skip expensive features (nut_flags, MC) when iters()==0; begin()/end() in try/finally.
AUDIT (Tasks 1–7, opus/max): 0 Critical, 6 Important, 11 Minor — file audit-tasks1-7.md
  I1 equity_vs_random/strength_now raise on iters=0 (TimeManager returns 0) — load-bearing for Task 10
  I2 omaha_batch one-hole-vs-N-boards broadcasting returns a single value
  I3 load_bot_file: only top-level module fresh; bot.* module state leaks across halves/matches
  I4 test_sim_match: 7/15 injected sim bugs pass default tests; F10 anchor marked slow but 0.29 s
  I5 test_tournament: paired_diff CI √deals-too-wide bug passes tests
  I6 test_equity: strength_now replaced by full-runout equity passes tests
Ruling R21: audit Importants I1–I6 fixed in ONE opus/max fix dispatch right after Task 9 review closes (before Task 10); audit Minors logged for final review — cost if wrong: none.
Ruling R22 (for I3): load_bot_file purges `bot` and `bot.*` from sys.modules before exec (configurable package list) so each half re-imports our package fresh, mirroring the tournament's fresh process; bot/ module-level state must be read-only tables; per-half state lives in memory/instance — cost if wrong: ~20–50 ms re-import per half in sims.
Task 9: Ruling (Important #2, plan-mandated GRACE_S=0.00075): TimeManager takes `grace_s` ctor param (default 0.00075) + module helper `grace_for_cpu_factor(f) = max(0.0, 0.001 − 0.00025·f)`; Task 11 measures cpu_factor in warmup and passes it — fixed grace under-counts the judge by ~0.44 ms/request at 3× CPU (≈1.8 s/half) — cost if wrong: slight over-conservatism on fast CPUs.
Task 9: Ruling: fold two reviewer Minors into fix round 1 because they bear on forfeits/Review Focus 1: (a) ctor clamps degenerate config instead of raising (ValueError in new_half = forfeited half); (b) target_s default = 0.375 × bank_s — cost if wrong: none.
Task 9: minor (deferred): 3× total<8.0 check can't fail for any ladder (pacing only proven by 50× test); begin() overwrite undocumented; hand-0 amplification (Task 11: no lazy init in first choose_action); _EPS boundaries 3.6/4.2 untested
Task 9: Ruling (amends prior): grace_for_cpu_factor(f) = 0.001 − 0.00025·f WITHOUT max(0,…) clamp — a negative grace correctly charges plumbing time beyond the judge's free 1 ms that the bot cannot observe (needed for f > 4) — cost if wrong: over-charging on very slow CPUs (safe direction).
Task 9: fix round 1/5 (4 addressed, 0 open — plumbing-scaled sim, grace_s param + grace_for_cpu_factor (unclamped), ctor never raises, target=0.375×bank; commits d5c7812..df265bd)
Task 9: minor (deferred): iters() ZeroDivision at bank_s=5e-324 (derived target underflow); grace_for_cpu_factor(inf) fails open; report stale lines
Task 9: note for Task 11: floor measured cpu factor at 1.0 (fixed plumbing share on fast hosts); tm.hands is float (don't pass to range()); wrap begin/end in try/finally; keep lazy init out of first choose_action
Task 9: complete (commits f4dfb8e..df265bd, review clean)
AUDIT-FIX: dispatching opus/max fixer for I1–I6 (base df265bd)
USER INFO (2026-11-07): user has access to a high-end CPU machine for training (not limited to this Mac).
Ruling R23: Tasks 17/18 scripts must be portable (Linux/macOS, requirements-train.txt only), scale to all available cores (parallel dataset gen; shared-memory multi-process/numba-parallel MCCFR), checkpoint/resume, and ship train/README.md runbook; iteration target and bucket counts to be sized for the big machine at Task 18 (ask user: cores, OS, who runs it) — cost if wrong: extra engineering in T17/T18.
USER DIRECTIVE (2026-11-07): set up two-track plan; re-reviews of small fixes on Opus at high.
Ruling R24: TWO TRACKS. Track A = branch kevin in /Users/kchen/Desktop/code/FALL26/heads-up-bundle: audit-fix → T10 → T11 → T12 (v1) → T13 → T14 (v2). Track B = branch kevin-cfr in sibling worktree /Users/kchen/Desktop/code/FALL26/heads-up-bundle-cfr (forked at df265bd): T15 → [merge kevin into kevin-cfr after audit-fix closes] → T16 → T17 → T18 (+ long training on user's big CPU) → T19. Then merge kevin-cfr into kevin (local, no push) → T21 → T20 → T22 → T23 → T24 → final review. One implementer per track at a time; ledger shared; review packages generated from the main checkout by SHA — cost if wrong: merge conflicts (low: tracks touch disjoint files except possibly requirements-train.txt).
Ruling R24b: worktree placed as SIBLING dir, not .worktrees/ inside repo — pytest without testpaths would collect the worktree's tests from the main checkout — cost if wrong: none.
Ruling R25: scoped re-reviews of small fix diffs (≲300 changed lines) use opus at effort high; full task reviews, implementers, audits stay opus max — user directive.
Track B BASE for T15: df265bd
Task 15 (Track B): Ruling: node_of also validates obs consistency with the node (legal, pot, to_call, seat, street, chips) and returns None on mismatch — strengthens Review Focus 1/5 (retuned stack/blinds → heuristic fallback) — cost if wrong: none (a raise-cap retune still needs the T20 config check).
Task 15: Ruling: legal as tuple and key_to_node as read-only mapping (with pickle support) — R22 read-only module state — cost if wrong: none.
Task 15: note for T19: blueprint must store n_dec, n_term and a hash of `child`; loader verifies before use. note for T20/T11: call get_tree() in warmup/new_half, never inside choose_action (0.36 s build).
Task 15: minor (deferred → T18 required): bot/tree.py:161 node_from_actions/_codes reject numpy ints (trainer input) — T18 must accept np.integer (not bool) or convert.
Task 15: minor (deferred): get_tree() caches first-call rules; tree should store sb/bb/raise_cap and rebuild/assert on mismatch (also gives T20 a config to compare); bot/tree.py:116 uses private g._to_call
Task 15: complete (commits df265bd..51ee0ae on kevin-cfr, review clean — reviewer exhaustively verified all nodes/terminals vs engine, both seats, JSON path)
Track B: waiting for audit-fix to close, then merge kevin → kevin-cfr, then T16.
AUDIT-FIX: Ruling: load_bot_file additionally restores sys.modules after loading and drops _simbot_* entries (prevents ~2–30 MB/half leak; fixes audit M4) — cost if wrong: a bot needing sys.modules[__name__] at play time breaks (none do).
AUDIT-FIX: Ruling: iters<=0 → preflop table value preflop; postflop texture-blind made-hand percentile (mean err ~0.1) — cheap deterministic fallback — cost if wrong: weaker decisions only when bank-starved.
AUDIT-FIX: note for T16/T20: at iters=0 equity==strength → 2-D buckets collapse to diagonal; runtime bucketing must guarantee a minimum iters or accept degradation (decide in T16). note for T10/T11: compute_features(iters=0) test owed; warmup must call bot.equity.warmup() and bot.preflop.load_preflop(); PolicyBot factories rely on no mutable module state in bot/.
AUDIT-FIX: minor (deferred): audit M5 BOTH_FORFEIT not counted as a loss in standings; commit 771e686 alone makes iters=0 NaN (never cherry-pick without 817a3d3).
AUDIT-FIX: implementer done (commits df265bd..66a3d73: 771e686 817a3d3 a16af7c 79036c5 86d5496 ab1d1b9 66a3d73); re-review dispatched (opus max; 1,059-line diff > small threshold).
Track B: merged kevin@66a3d73 into kevin-cfr → fcde753; worktree suite 221 passed. T16 BASE = fcde753.
Ruling R26: T16 makes NUM_BUCKETS configurable via data/buckets.npz metadata (defaults 200/100/100/100 per plan) so T17/T18 can scale up on the user's big CPU; T16 must measure runtime-vs-offline bucket agreement at iters ∈ {0, 32, 64, 128} and define the iters=0 behaviour explicitly (audit-fix note) — cost if wrong: bucket rework in T17.
AUDIT-FIX: fix round 1/5 (6 addressed, 0 open; commits df265bd..66a3d73) — re-review clean; reviewer hand-applied 7 surviving sim mutants: all killed.
AUDIT-FIX: minor (deferred): pickling bot.* objects during play fails in sim (private module copies) — document in sim/match.py + T11; memory lags GC (gc.collect() at load_bot_file start suggested); bot/equity.py:60 except misses zipfile.BadZipFile; slow-every-9th timeout test flake risk under CPU contention
AUDIT-FIX: complete. Track A next: T10 (BASE 66a3d73).
USER INFO (2026-11-07 16:10 CST): DEADLINE = 2026-11-08 12:00 (local). User's PC (high-end CPU + dedicated GPU) available ~17:10; not reachable before.
Ruling R27 (user decision): Tasks 22/23 CUT for now (stretch) — revisit only if v3 is done with time left; T21 kept in full — user directive.
Ruling R28: Track C added — T13 (opponent model) in sibling worktree ../heads-up-bundle-opp on branch kevin-opp (from kevin@66a3d73); merges into kevin before T14 — saves ~1 h critical path — cost if wrong: small merge (disjoint files).
Ruling R29: plumbing tasks (T11, T19, T24) implementers at opus effort high; strategy tasks + full reviews stay opus max; every strategy implementer gets an explicit wall-clock time box — deadline pressure — cost if wrong: slightly less polish on plumbing.
Target timeline: v2 submission-ready + reviewed ~01:00; CFR training overnight on user's PC; v3 gate ~08:00; final ~10:00; deadline 12:00.
Note: foreign worktree 'licheng' (another session) exists in git worktree list — not ours; leave alone.
