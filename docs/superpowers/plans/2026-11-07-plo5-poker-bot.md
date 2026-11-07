# Short Deck PLO5 Tournament Bot — Research, Strategy & Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the bot most likely to finish first in the TAMU Datathon heads-up Short Deck PLO5 round robin. Ship a crash-proof bot early, then upgrade it in gated stages: an exploitative heuristic, then an opponent-model layer, then a CFR blueprint.

**Architecture:** `agent.py` stays a thin shell around a `bot/` package. Its layered policy is: CFR blueprint lookup → exploit modifiers → tuned heuristic → guaranteed-legal default. Every layer is wrapped in its own `try/except`, and the work is bounded by a time manager. Offline tooling lives in `train/`. An in-process duplicate-match simulator and opponent zoo live in `sim/`. Neither ships in the image.

**Tech Stack:** Python 3.12 (tournament base image; local `.venv` is 3.14, so avoid 3.13+-only features), numpy (pre-installed in the base image), FastAPI/uvicorn (pinned), pytest. numba is optional and used offline only (`requirements-train.txt`).

**Spec:** `context/prompt.md` (research brief) + `context/Heads Up Participant Guide.pdf` + `README.md`. Rules source of truth: `game.py`, `judge_engine.py`.

## Global Constraints

- Deliverable entry point: `agent.py` in repo root. Edit only between `# ===== YOUR CODE HERE =====` and `# ===== END OF YOUR CODE =====`, plus the `warmup()` area. Never touch SERVER PLUMBING or the last lines that read `PORT`.
- `requirements.txt` stays in repo root. `fastapi==0.156.1` and `uvicorn==0.54.0` must not change. Add `numpy` without a version (the base image already has it).
- Game constants (from `game.py`): `BIG_BLIND = 2`, `SMALL_BLIND = 1`, `STARTING_STACK = 80`, `RAISE_CAP_PER_STREET = 4`, `NUM_DEALS = 1000`.
- Time: `TIME_BANK_SECONDS = 8.0` per half, `NETWORK_GRACE_SECONDS = 0.001` free per request, `ACTION_TIMEOUT_SECONDS = 1.0` per request, `STARTUP_TIMEOUT_SECONDS = 15.0`.
- `ERROR_LIMIT = 5` errors in a half → forfeit → **match loss**. Every `on_hand_end` exception counts separately.
- Verdict: 95% CI (`CHOP_Z = 1.96`) on paired bb/deal; win 1 / chop 0.5 / loss 0; tie-break = average bb/deal.
- Container: 1 CPU, 4 GB RAM, no network, read-only files except `/tmp` (512 MB), non-root user, ≤ 3 GB added image size, ≤ 30 min build, fresh process every half, `OMP_NUM_THREADS=1`.
- Model and data files are committed normally, each < 100 MB (Git LFS is not downloaded).
- Repo must be public. Submit the full 40-character commit SHA.
- Engineering budgets for this plan: median `/act` compute ≤ 0.7 ms, p99 ≤ 10 ms, max single request ≤ 100 ms, total bank use ≤ 3.0 s per half on a dev laptop (leaves ×2.5 headroom for a slower or shared tournament CPU), startup ≤ 6 s.

## Review Focus

1. **Organizers retune constants** (stack, blinds, raise cap or time bank change before the tournament; the guide says they may). `new_half` receives `AgentConfig`. If it differs from the blueprint's training metadata, the bot must silently use the heuristic layer, which reads sizes only from `obs`. Test: Task 11 `test_config_mismatch_uses_heuristic`.
2. **Short-stack and all-in corners:** `to_call >= my_stack` (no raises legal), opponent already all-in, an all-in `raise_to` below `min_raise_to`, `legal_actions == (CHECK_CALL,)`. The bot must still return a legal action without raising an exception. Test: Task 10 `test_all_in_corners_are_legal`.
3. **Slower tournament CPU:** if every decision takes 3× longer, the bot must still finish the half without emptying the bank. Test: Task 9 `test_slow_clock_stays_under_bank`.
4. **End-of-half `/hand-end` batch and odd results:** several results arrive at once, `opp_hole is None`, `winner == "split"`, and hands where we never acted (the opponent folded as button preflop). `on_hand_end` must never raise. Test: Task 13 `test_hand_end_batch_never_raises`.
5. **Action history not in the betting tree** (a rules change, or a corrupt observation): the node lookup misses and the bot must fall back to the heuristic, not raise. Test: Task 15 `test_unknown_history_returns_none`.

---

# Part A — Research & Strategy

Labels: **[V]** verified in this session against the source code or by measurement. **[D]** mathematical deduction. **[H]** hypothesis to be tested by a named task. **[L]** published literature, cited from memory: verify the citations before quoting them externally.

## A0. Verified facts about this exact game (measured 2026-11-07)

| # | Fact | Source |
|---|---|---|
| F1 | **The full betting tree is tiny and card-independent: 15,952 decision nodes** (preflop 62, flop 810, turn 3,678, river 11,402) **and 25,538 terminals.** | [V] exhaustive enumeration with `game.Game.clone()` |
| F2 | Preflop pot-raise ladder: raise-to 6 → 18 → 54 → 80. Four pot raises preflop = all-in. | [V] |
| F3 | The judge charges `max(0, elapsed − 1 ms)` per request. Measured server and HTTP overhead is ≈ 0.23 ms per call (2,502 calls in 0.58 s wall), so **plumbing costs nothing against the bank**. Compute that fits inside the remaining ≈ 0.7 ms per request is also free. | [V] `judge_engine.py:358-404` + real-judge run |
| F4 | Decisions per half for one bot: 2,502 (starter vs calling station, real judge) and 3,550–4,130 (in-process, various pairs). → **Average bank ≈ 2 ms per decision** on top of the free ~0.7 ms. | [V] |
| F5 | `omaha_score` costs ≈ 11.8 µs per call. A 200-rollout pure-Python Monte Carlo equity takes ≈ 3.5–4.4 ms, which is **over budget for every decision.** | [V] |
| F6 | Equity vs a uniformly random hand, 5th–95th percentile: **preflop 0.38–0.60** (sd 0.071), **flop 0.23–0.92** (sd 0.218), **turn 0.14–0.95** (sd 0.245). | [V] Monte Carlo, 300 situations × 200 rollouts |
| F7 | Best river hand of a random holding: straight 27.2%, two pair 23.6%, full house 22.6%, trips 14.9%, flush 7.8%, quads 2.3%, straight flush 0.5%, one pair 1.1%. | [V] 20k samples |
| F8 | Duplicate paired sd (bb/deal) depends on how differently the two bots play: 0 for identical deterministic bots, ≈ 0.8 vs a calling station, 8.5–11 for threshold bots vs random/each other, ≈ 15.8 for random vs calling station. | [V] in-process duplicate matches, 2 seeds each |
| F9 | The guide's example strategy loses to random (−1.95 and −1.56 bb/deal on 2 seeds), as the guide itself says. | [V] |
| F10 | Starter bot vs calling station, judge seed `1`, 1,000 deals: **−0.516 bb/deal, std 0.99**. Use this as the simulator's cross-check value. | [V] real judge |
| F11 | `HandResult` contains **no action history**. The opponent's final action (for example, folding to our bet) must be inferred from our last observation plus the result. | [V] `game.py:272-313` |
| F12 | Tournament base image runs Python 3.12 and includes numpy and torch. | [V] `docker/Dockerfile.base` |

### Deductions

- **D1 — win threshold.** CI half-width = 1.96·sd/√1000 = 0.062·sd. By F8, the edge needed to *win* is about **0.05 bb/deal against passive bots** and **0.5–1.0 bb/deal against aggressive or random ones**. Against a near-equal opponent, expect a chop. The points come from (a) never losing, (b) converting medium edges into significant ones.
- **D2 — no action abstraction.** By F1, every betting sequence can be represented exactly. Only **card abstraction** is lossy. That puts this game structurally closer to heads-up *limit* hold'em (where Cepheus applied) than to no-limit (Libratus/DeepStack/Pluribus).
- **D3 — infoset budget.** With B buckets per street, the infoset count is 62·B₀ + 810·B₁ + 3,678·B₂ + 11,402·B₃. At B = 200/100/100/100 that is ≈ 1.6 M infosets × 4 actions ≈ 25 MB as float32 (13 MB as float16). That fits the 4 GB RAM and 100 MB-per-file limits easily.
- **D4 — where the edge is.** By F6, preflop equities are compressed, so preflop mistakes are cheap in equity terms but expensive in variance. Postflop equities are spread wide, so most of the EV comes from postflop and river decisions.
- **D5 — stack depth.** After a pot 3-bet preflop (each player in for 18), the stack-to-pot ratio (SPR) is 62/36 ≈ 1.7. Many hands become one-decision commitment spots on the flop. Correct commitment thresholds, which depend on how often our hand is the nuts, matter more than fancy multi-street play.
- **D6 — runtime design.** By F3–F5, runtime work must be table lookups plus a **vectorized** evaluator with a small Monte Carlo (≤ ~128 rollouts). The extra Monte Carlo goes to big pots, where errors cost the most chips.

## A1. Historical research — what transfers

| System | Key idea | Demonstrated | Transfer here |
|---|---|---|---|
| CFR [L: Zinkevich et al., NIPS 2007] / MCCFR [L: Lanctot et al., NIPS 2009] | Regret minimization over information sets. Sampling variants scale it up. | Foundation of every ACPC champion since ~2007 [L] | **Direct.** By D2/D3, tabular external-sampling MCCFR over the exact tree is feasible. |
| Cepheus [L: Bowling et al., Science 2015; CFR+ Tammelin 2014] | CFR+ on a nearly lossless heads-up limit hold'em game → essentially solved | Yes, HULHE "essentially weakly solved" | **Closest analogue** (fixed bet sizes, raise cap). But the lossless approach doesn't fit: PLO5 hands (C(36,5) = 376,992 preflop) × boards need *lossy* buckets. |
| Discounted/Linear CFR [L: Brown & Sandholm, AAAI 2019] | Weight later iterations more heavily → faster convergence | Yes, empirically | **Direct.** Use linear averaging in MCCFR. |
| Libratus [L: Brown & Sandholm, Science 2018] | Blueprint + nested real-time subgame solving + self-improvement | Beat top pros over 120k hands of no-limit hold'em | Blueprint idea yes. **Real-time solving no**: by F4 there are ~2 ms per decision in single-core Python. |
| DeepStack [L: Moravčík et al., Science 2017] | Continual re-solving with learned counterfactual-value networks; AIVAT evaluation | Beat pros in heads-up no-limit (statistically significant) | Re-solving is impractical (same reason). **AIVAT-style variance reduction for *our* evaluation is useful** [L: Burch et al., AAAI 2018]. |
| Pluribus [L: Brown & Sandholm, Science 2019] | MCCFR blueprint (linear weighting, pruning) + depth-limited search, cheap compute | Beat pros in 6-max | MCCFR-with-linear-weighting blueprint yes. Search no. |
| Deep CFR [L: Brown et al., ICML 2019] / ReBeL [L: Brown et al., NeurIPS 2020] | Neural function approximation instead of abstraction; RL + search | Strong in research benchmarks | Deep CFR could remove abstraction loss but has high build risk. Kept as a stage-4 option. ReBeL depends on search, so no. |
| Robust counter-strategies [L: Johanson, Zinkevich & Bowling, NIPS 2007; Johanson & Bowling, AISTATS 2009] | Restricted Nash response: exploit a model while limiting your own exploitability | Yes, in limit hold'em experiments | **Direct** for safe exploitation (Task 22). |
| Implicit agent modelling [L: Bard et al., AAMAS 2013] | Pick among precomputed counter-strategies online by observed performance or likelihood | Yes, ACPC-style | **Direct**: choose a counter-strategy at runtime from observed opponent stats. |
| Abstraction quality [L: Johanson et al., AAMAS 2013; Ganzfried & Sandholm, AAAI 2014] | Potential-aware, distribution-based buckets beat expected-hand-strength-only buckets | Yes | Use 2-D buckets (current strength × equity with potential), not equity alone. |
| Purification [L: Ganzfried, Sandholm & Waugh, AAMAS 2012] | Rounding abstract mixed strategies toward pure ones often *improves* play in the real game | Yes, empirically | Cheap to test (Task 19 threshold). |
| Local Best Response [L: Lisý & Bowling, 2017] | Cheap lower bound on exploitability | Showed many no-limit bots were highly exploitable | Validation tool (Task 21). |
| PLO specifically | — | I know of **no peer-reviewed superhuman PLO agent** to cite. Commercial PLO solvers exist, but their methods aren't published here. | Transfer to PLO is **untested**, so treat every PLO-specific claim as [H]. |

**Conclusion.** The winning recipe in fixed-action poker is a CFR blueprint over a good card abstraction (Cepheus lineage), plus safe opponent exploitation (counter-strategy lineage). The search-heavy no-limit breakthroughs don't fit a ~2 ms Python budget.

## A2. Competitive analysis — predicted AI-generated bots

Each line is a hypothesis about LLM-generated bots, followed by the test that checks it and our exploit.

- **H1 — Monte Carlo equity vs a *random* hand, with fixed iterations (200–1,000) in pure Python.**
  - Test: Task 4 zoo. Count how many generated bots do this, and measure their bank usage.
  - Exploit: by F5 these bots **empty their 8 s bank part-way through the half** and become check/fold bots. Detecting that and betting relentlessly is free EV (Innovation I2).
- **H2 — Equity vs random, compared directly with pot odds.** Medium hands look like ~50% equity, so these bots over-call and stack off with non-nut hands (straights and small flushes; F7: straights are the most common river hand).
  - Test: Task 4 action stats at showdown.
  - Exploit: value-bet thinner and larger, cut our bluffs (Task 14 "station" mode).
- **H3 — Category thresholds** ("raise flush+, call two pair"), ignoring board texture: flushes on four-flush boards, a straight on a paired board.
  - Test: Task 4.
  - Exploit: the CFR/heuristic layer conditions on nut status, not just category.
- **H4 — Hold'em-style preflop logic.** Too tight, folds to raises.
  - Test: fold-to-preflop-raise rate from opponent stats (Task 13).
  - Exploit: by F6 our preflop equity is ≥ 0.38 almost always, so we open and 3-bet light against folders.
- **H5 — No use of position, `action_history`, or the half-pot vs pot distinction.**
  - Exploit: positional aggression, and sizing chosen per spot by the blueprint.
- **H6 — Fragile code** (rare exceptions, 1 s timeouts in the first hands, crashes in `on_hand_end`).
  - Our bot can't cause these. Being robust ourselves converts their forfeits into wins.
- **H7 — A minority of sophisticated teams** (CFR or RL bots).
  - Defense: blueprint near-equilibrium → chop, not loss.
  - Never deviate from the blueprint far enough to be exploited by an adaptive bot. Exploit modes switch on only with high-confidence detection, and the bot reverts if the evidence weakens.

**Anti-overfitting:** the zoo is built from different generators (hand-written archetypes, LLM-generated bots from several prompts, past versions of ours). Tuning uses a **train split**. Acceptance decisions use a **held-out split** of opponents and seeds.

## A3. Architecture comparison

| # | Approach | Expected strength vs LLM field | Vs strong bot | Feasibility (datathon time) | Runtime fit | Main weakness |
|---|---|---|---|---|---|---|
| 1 | Tuned heuristic (vectorized equity + pot odds + nut awareness) | Good | Weak to fair | **High** (hours) | Good with fast eval | Exploitable; hand-tuned thresholds overfit |
| 2 | Tabular MCCFR blueprint, exact betting tree, 2-D card buckets | Good (wins slowly vs weak bots) | **Good** (chop or win) | Medium (≈1–2 days incl. data generation) | **Excellent** (bucket + lookup) | Abstraction loss; runtime bucket noise vs offline buckets |
| 3 | Deep CFR / NFSP / RL self-play with networks | Potentially best | Potentially best | Low (training risk, tuning time) | OK (small MLP on 1 thread) | High variance in outcome; hard to debug in time |
| 4 | Pure opponent modelling + best response | **Best vs weak/static bots** | Poor (exploitable, slow to adapt) | Medium | Good | Misclassification costs big pots; overfits to the zoo |
| 5 | Real-time subgame solving (Libratus/DeepStack-style) | — | — | Low | **Fails** (~2 ms/decision, F4) | Rejected |
| 6 | Distillation of an expensive offline policy (deep-MC or LBR policy → lookup or MLP) | Good | Fair | Medium | Good | Inherits the teacher's flaws; no equilibrium guarantee |
| 7 | **Hybrid: #2 blueprint + #4 safe exploit layer + #1 as fallback** | **Best expected points** | Good | Medium, **staged with ship gates** | Good | Most moving parts; mitigated by layered fallbacks and tests |

## A4. Recommended architecture

**#7, the staged hybrid.** Why: by D1, points come from (i) never forfeiting, (ii) clearly beating the many weak bots, (iii) not losing to the few strong ones. #4 delivers (ii), #2 delivers (iii), and #1 plus the layered fallbacks deliver (i) and give a shippable bot within hours.

**Ship gates:**
- **v1** = heuristic, after Stage 1.
- **v2** = v1 + exploit layer, after Stage 2.
- **v3** = blueprint + exploit, after Stage 3, **only if** it beats v2 by the Task 20 criteria. Otherwise v2 stays the submission.

**Self-challenge:**
- *Is CFR worth it?* Against a weak LLM field, a well-tuned heuristic plus exploits may earn nearly the same points. CFR's value is insurance against strong bots and a better tie-break bb/deal. That's why it's gated rather than assumed.
- *Will equity-vs-random buckets suffice in PLO, where how often you hold the nuts dominates?* Not guaranteed. The 2-D (strength now × equity) buckets partly capture draws. Task 16 measures how much bucket count and bucket definition change head-to-head results.
- *Could the exploit layer backfire?* It only switches on with high-confidence detection and has bounded deviations. Task 14 requires no regression against non-target opponents.

## A5. Creative advantages (each testable)

- **I1 — Exact-tree CFR.** No action abstraction at all (F1). Test: Task 20 head-to-head vs heuristic.
- **I2 — Opponent-time-bank-exhaustion / check-fold detector.** A bot that has run out of bank (or forfeited) plays `default_action` deterministically: it checks when that's free and folds otherwise. After N consecutive bet-facing folds with zero raises, switch to "bet whenever a raise is legal (RAISE_HALF)". Expected gain against such a bot: roughly the pot on every hand. Test: Task 14 vs `checkfold_after(300)`.
- **I3 — Pot-weighted thinking.** Spend Monte Carlo rollouts where pot × uncertainty is largest. Small pots, and the many trivial checks, get 0–32 rollouts and stay inside the free 1 ms. Test: Task 9 + Task 12 ablation.
- **I4 — Vectorized numpy evaluator.** Index 5-card ranks in base 9 into a 2·9⁵ table, so evaluation is batched instead of dict-per-hand. Target ≥ 10× throughput, which makes runtime equity affordable. Test: Task 6 speed assertion.
- **I5 — Variance-aware play for points.** Avoid near-zero-EV, high-variance preflop all-ins (F2, F6) when our edge is postflop. That shrinks the CI and turns chops into wins. Test: Task 23 compares *points*, not bb, on the held-out set.
- **I6 — Showdown-calibrated opponent model.** Use `opp_hole` revealed at showdown to estimate each opponent's minimum strength per action (for example "raises the river only with flush or better"), then tighten or loosen our calls. Test: Task 13/14.
- **I7 — LLM-generated opponent population.** Prompt fresh agents with only the participant guide, which is what our real competitors will do, to build a realistic zoo with a held-out split. Test: Task 4 diversity report.
- **I8 — Free startup compute.** Load the tables and the evaluator during the 15 s startup window, which isn't charged to the bank. Test: Task 11 startup timing.
- **I9 — Purification/thresholding of the blueprint.** Test: Task 19 sweep of the threshold over {0, 0.05, 0.1, 0.2}.
- **I10 — Paired, common-seed candidate comparison** (duplicate on both sides, plus AIVAT-like control variates optional), so we detect smaller improvements with fewer hands. Test: Task 5 reports the paired difference CI between two candidates.

---

# Part B — Implementation Roadmap

## File Structure

| Path | Responsibility | Ships in image? |
|---|---|---|
| `agent.py` | Thin shell: calls `bot.policy` from the three functions; `warmup()` area calls `bot.policy.warmup()` | yes |
| `bot/fasteval.py` | Vectorized 5-card / Omaha evaluator (I4) | yes |
| `bot/equity.py` | Vectorized Monte Carlo equity and current-strength estimates | yes |
| `bot/preflop.py` | Suit-canonical preflop key + strength table lookup | yes |
| `bot/features.py` | `Features` dataclass + `compute_features` | yes |
| `bot/timing.py` | `TimeManager` (I3) | yes |
| `bot/heuristic.py` | `HeuristicParams` + `heuristic_action` | yes |
| `bot/opponent.py` | `OpponentModel` stats + detectors (I2, I6) | yes |
| `bot/exploit.py` | Exploit modifiers applied to a base action/distribution | yes |
| `bot/tree.py` | Exact betting tree + history → node lookup | yes |
| `bot/abstraction.py` | Bucket functions + bucket edges | yes |
| `bot/strategy.py` | `Blueprint` loader and sampler | yes |
| `bot/policy.py` | Layered `decide`, `start_half`, `hand_finished`, `warmup` | yes |
| `data/*.npz` | preflop table, bucket edges, blueprint | yes |
| `train/*.py` | Offline table / bucket / dataset / MCCFR / export scripts | no (`.dockerignore`) |
| `sim/match.py`, `sim/tournament.py`, `sim/tune.py` | In-process judge clone, round robin, tuning | no |
| `sim/opponents/` | Zoo: baselines, archetypes, LLM-generated bots | no |
| `tests/` | Unit and integration tests (root `test_agent.py` unchanged) | no |
| `requirements-train.txt` | `numba` (optional) etc. for offline only | no |

**Shared types used across tasks:**
- `BotLike`: any object with `new_half(config, memory)`, `choose_action(obs, memory) -> Action` and `on_hand_end(result, memory)`. Modules qualify.
- `BotFactory = Callable[[], BotLike]`: returns a **fresh** instance per half, emulating the tournament's fresh process.

## Stage 0 — Infrastructure

### Task 1: Repo hygiene and identity

**Files:** Create `.gitignore`, `.dockerignore`, `requirements-train.txt`. Modify `requirements.txt` (add `numpy`) and `agent.py` (`PARTICIPANT`, `AGENT_NAME`). Test: `tests/test_repo_hygiene.py`.

- [ ] **Step 1: Write the failing test**
```python
def test_dockerignore_excludes_dev_dirs_but_not_bot():
    lines = Path(".dockerignore").read_text().split()
    for d in (".venv", "context", "train_data", "sim", "tests", "train", "docs"):
        assert d in lines or f"{d}/" in lines
    for keep in ("agent.py", "bot", "data", "game.py", "evaluator.py", "requirements.txt"):
        assert keep not in lines and f"{keep}/" not in lines

def test_names_set():
    src = Path("agent.py").read_text()
    assert "TotallyAwesomeDatathonParticipant" not in src and "JustAsAwesomeAgent" not in src

def test_requirements_pins_untouched():
    req = Path("requirements.txt").read_text()
    assert "fastapi==0.156.1" in req and "uvicorn==0.54.0" in req and "numpy" in req
```
- [ ] **Step 2:** `.venv/bin/python -m pytest tests/test_repo_hygiene.py -q` → FAIL.
- [ ] **Step 3:** Create the files. `.gitignore` covers `.venv/`, `__pycache__/`, `train_data/`, `*.json` match logs. Set team and bot names; **ask the user for the team name** (it must be identical on every resubmission).
- [ ] **Step 4:** Run the tests → PASS. Also run `.venv/bin/python -m pytest -q test_agent.py` → PASS.
- [ ] **Step 5:** `git rm -r --cached tools/__pycache__`, then commit: `chore: repo hygiene, dockerignore, team identity`.

### Task 2: In-process duplicate match simulator

**Files:** Create `sim/__init__.py`, `sim/match.py`. Test: `tests/test_sim_match.py`.

**Interfaces — Produces:**
- `load_bot_file(path: str) -> ModuleType`: imports a bot file under a unique module name each call, so every half gets fresh globals.
- `@dataclass HalfStats(nets_a: list[int], errors_a: int, errors_b: int, forfeit_a: bool, forfeit_b: bool, bank_a: float, bank_b: float, calls_a: int, calls_b: int, max_call_a: float)`
- `@dataclass MatchStats(mean_bb: float, sd_bb: float, ci_low: float, ci_high: float, verdict: str, points_a: float, halves: tuple[HalfStats, HalfStats])`, with `verdict ∈ {"A", "B", "CHOP", "BOTH_FORFEIT"}`.
- `play_half(fa: BotFactory, fb: BotFactory, seed: str, half: int, deals: int = 1000, grace_s: float = 0.00075) -> HalfStats`
- `play_match(fa: BotFactory, fb: BotFactory, seed: str, deals: int = 1000) -> MatchStats`

These mirror `judge_engine.py` exactly:
- seats: `agent1_seat_for_half`; deals: `make_deal(seed, d)`; button: `button_for_deal(d)`
- hand results are queued and delivered before the bot's next `choose_action`, then flushed at the end of the half
- an exception, a non-Action or an illegal action counts as an error and plays `default_action`
- a forfeit at 5 errors; a bank that empties switches the bot to `default_action`
- a call slower than 1 s counts as an error
- forfeit-loses-match logic as in `combine_halves`

- [ ] **Step 1: Write the failing tests**
```python
def test_matches_real_judge_value():          # F10
    m = play_match(lambda: load_bot_file("agent.py"),
                   lambda: load_bot_file("baseline-bots/baseline_calling_station.py"), seed="1")
    assert m.mean_bb == pytest.approx(-0.516, abs=5e-4) and m.verdict == "B"

def test_identical_bots_chop():
    f = lambda: load_bot_file("baseline-bots/baseline_calling_station.py")
    m = play_match(f, f, seed="s1", deals=200)
    assert m.mean_bb == 0 and m.verdict == "CHOP"

def test_crashing_bot_forfeits_and_loses():   # bot raising on every call
    m = play_match(crash_factory, call_factory, seed="s1", deals=50)
    assert m.halves[0].forfeit_a and m.points_a == 0

def test_on_hand_end_errors_count_individually():
    h = play_half(handend_crash_factory, call_factory, "s1", 1, deals=10)
    assert h.errors_a == 5 and h.forfeit_a
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Charge time `max(0, elapsed − grace_s)` with `time.perf_counter()`, around `on_hand_end` deliveries plus `choose_action` together, as one "request".
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat(sim): in-process duplicate match simulator matching judge`.

### Task 3: Archetype opponent zoo

**Files:** Create `sim/opponents/__init__.py` and `sim/opponents/archetypes.py`. Test: `tests/test_zoo.py`.

**Interfaces — Produces:** `ZOO: dict[str, BotFactory]` containing:
- `random`, `calling_station`, `always_half`, `always_pot` (load from `baseline-bots/`)
- `guide_example` (the PDF §4.1 strategy)
- `nit` (folds unless preflop strength is in the top 20% / flop+ two-pair+)
- `maniac` (raises whenever legal 70% of the time, otherwise calls)
- `mc_equity_fast` (64-rollout equity vs random hand + pot odds; raise if equity > 0.7)
- `mc_equity_slow` (500 rollouts: same logic, and it will empty its bank, which is realistic per H1)
- `threshold_cat` (category thresholds, ignoring board texture)
- `checkfold_after(n)` (normal `mc_equity_fast` for n hands, then `default_action`)

Each opponent is a class with the three methods. Opponent state lives in `memory`.

- [ ] **Step 1: Write the failing test**
```python
@pytest.mark.parametrize("name", list(ZOO))
def test_zoo_bot_is_legal_and_errorless(name):
    h = play_half(ZOO[name], ZOO["calling_station"], "z1", 1, deals=100)
    assert h.errors_a == 0 and not h.forfeit_a
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Use `evaluator.omaha_score` (deliberately slow, as LLM bots would be).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat(sim): archetype opponent zoo`.

### Task 4: LLM-generated opponent population (I7)

**Files:** Create `sim/opponents/llm/<name>.py` (≥ 8 bots), `sim/opponents/llm/PROMPTS.md`, and a `sim/opponents/__init__.py` registration under the `llm_` prefix with `HOLDOUT: set[str]`. Test: extend `tests/test_zoo.py`.

- [ ] **Step 1:** Write 8 prompt variants and record them in `PROMPTS.md`:
  - "simple rules bot"
  - "Monte Carlo equity bot"
  - "opponent-modelling bot"
  - "aggressive bot"
  - "GTO-inspired bot"
  - "beginner team, 2 hours"
  - "ML-minded team"
  - "careful about time bank"

  Each prompt gets only the guide PDF text and `agent.py`.
- [ ] **Step 2:** Dispatch one fresh subagent per prompt. Each writes a complete `agent.py`-style file. Save only the YOUR CODE section plus imports, adapted into a class.
- [ ] **Step 3:** Mark 3 of the 8 bots, plus `threshold_cat`, as `HOLDOUT`. Never tune against them.
- [ ] **Step 4:** Run the parametrized zoo test → PASS. Bots that crash stay in the zoo and are recorded as "forfeit-prone", because they're realistic opponents. The test for those asserts `forfeit_a or errors_a == 0`.
- [ ] **Step 5:** Commit: `feat(sim): LLM-generated opponent population with holdout split`. (The diversity report is produced in Task 5, Step 5.)

### Task 5: Round robin, head-to-head, paired comparison

**Files:** Create `sim/tournament.py`. Test: `tests/test_tournament.py`.

**Interfaces — Produces:**
- `run_round_robin(entrants: dict[str, BotFactory], seeds: list[str], deals: int, workers: int = os.cpu_count()) -> Standings`, where `Standings` holds rows of `(name, points, mean_bb, losses, forfeits, max_bank)`.
- `focus_eval(candidate: BotFactory, opponents: dict[str, BotFactory], seeds, deals, workers) -> FocusReport`. Gives per-opponent `MatchStats` per seed, plus expected points (mean of `points_a`).
- `paired_diff(cand_a, cand_b, opponents, seeds, deals) -> tuple[float, float, float]`: mean and 95% CI of the per-deal difference in bb, using the same seeds and opponents (I10).
- CLI: `python -m sim.tournament --candidate bot.policy --opponents train|holdout|all --seeds N --deals D`, plus `--entrants zoo --report <path.md>`. The latter runs a zoo-only round robin and writes per-bot bank use and fold/raise rates.

- [ ] **Step 1: Write the failing tests**
```python
def test_round_robin_points_sum():
    s = run_round_robin({k: ZOO[k] for k in ("random", "calling_station", "nit")}, ["t1"], 100, workers=1)
    assert sum(r.points for r in s.rows) == pytest.approx(3.0)   # 3 matches, 1 point each

def test_paired_diff_of_identical_is_zero():
    lo, mid, hi = paired_diff(ZOO["nit"], ZOO["nit"], {"cs": ZOO["calling_station"]}, ["t1"], 100)
    assert lo == mid == hi == 0
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement with `multiprocessing.Pool`. **Step 4:** Run → PASS.
- [ ] **Step 5:** Run `.venv/bin/python -m sim.tournament --entrants zoo --seeds 1 --deals 300 --report docs/zoo_report.md`. Record in the report how many zoo bots show H1–H6.
- [ ] **Step 6:** Commit: `feat(sim): round robin, focus eval, paired comparison, zoo report`.

## Stage 1 — Fast evaluation and the robust heuristic bot (ship v1)

### Task 6: Vectorized evaluator (I4)

**Files:** Create `bot/__init__.py`, `bot/fasteval.py`. Test: `tests/test_fasteval.py`.

**Interfaces — Produces:**
- `score5_batch(cards: np.ndarray) -> np.ndarray`: shape (N,5) of int → (N,) int32, **identical** to `evaluator.score_five`.
- `omaha_batch(holes: np.ndarray, boards: np.ndarray) -> np.ndarray`: (N,5),(N,5) → (N,), identical to `omaha_score`.
- `made_now_batch(holes: np.ndarray, boards: np.ndarray) -> np.ndarray`: boards of width 3, 4 or 5. Best hand using exactly 2 hole + 3 board.
- `PAIR_IDX`, `TRIPLE_IDX`: the combination index arrays.

Table: index = base-9 encoding of the five ranks sorted descending, plus a flush bit → `int32[2·9⁵]`. Build it with `evaluator._score_ranks`.

- [ ] **Step 1: Write the failing tests**
```python
def test_score5_matches_reference_on_200k_random():   # exact equality
def test_omaha_batch_matches_reference_on_20k_random():
def test_made_now_on_flop_equals_bruteforce():
def test_speed():   # 10_000 Omaha evaluations in one batch
    t = perf_counter(); omaha_batch(H, B); assert perf_counter() - t < 0.010   # ≥ 10× faster than F5
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): vectorized numpy hand evaluator`.

### Task 7: Vectorized equity and strength

**Files:** Create `bot/equity.py`. Test: `tests/test_equity.py`.

**Interfaces — Consumes:** Task 6. **Produces:**
- `equity_vs_random(hole: Sequence[int], board: Sequence[int], iters: int, rng: np.random.Generator) -> float`: win + ½ tie probability, rolling out the opponent's 5 cards and the rest of the board.
- `strength_now(hole, board, iters, rng) -> float`: same, but on the *current* board using `made_now_batch`, with no future cards. On the river it equals equity.
- `nut_flags(hole, board) -> int`, a bitmask:
  - bit0: we hold the current best possible made hand given this board
  - bit1: board pairs (full houses possible)
  - bit2: ≥ 3 board cards of one suit (flush possible)
  - bit3: our flush draw uses the ace of the suit

  Computed exactly by checking all two-card combos from the unseen cards.

- [ ] **Step 1: Write the failing tests**
```python
def test_equity_agrees_with_slow_reference():   # 50 spots, 4000-iter reference, |diff| < 0.03 at iters=2000
def test_river_equity_is_deterministic_given_rng_seed():
def test_nut_flag_royal_example():              # guide example: board Ah Kh Qh Jh 6c, hole Th 7s 7d 8c 9c
    assert not (nut_flags(H, B) & 1)            # a king-high straight is not the nuts here
def test_timing_p99_under_1_5ms_at_128_iters_flop():
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): vectorized equity, strength, nut flags`.

### Task 8: Preflop strength table

**Files:** Create `bot/preflop.py`, `train/build_preflop.py`, `data/preflop.npz`. Test: `tests/test_preflop.py`.

**Interfaces — Produces:**
- `canonical_key(hole: Sequence[int]) -> int`: invariant under all 24 suit permutations. Map each suit to its lexicographic minimum over permutations, then pack the 5 sorted cards as base-36.
- `preflop_strength(hole) -> float`: equity vs random.
- `load_preflop(path="data/preflop.npz") -> None`, called from warmup.

The offline script enumerates all 376,992 hands, groups them by canonical key, and runs ≥ 3,000 rollouts per class with `bot.equity` (multiprocessing).

- [ ] **Step 1: Write the failing tests**
```python
def test_canonical_invariant_under_suit_permutation():   # 1000 random hands × 24 perms
def test_table_covers_all_hands():                       # every 5-subset of 36 has a key in the table
def test_strength_range():                               # all values in [0.30, 0.70]  (F6 spread)
def test_lookup_under_50us():
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement both files. Run `.venv/bin/python -m train.build_preflop --rollouts 3000 --workers 8` and commit the `.npz` (must be < 100 MB).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat(bot): preflop strength table`.

### Task 9: Time manager (I3)

**Files:** Create `bot/timing.py`. Test: `tests/test_timing.py`.

**Interfaces — Produces:**
`class TimeManager(bank_s=8.0, hands=1000, target_s=3.0, clock=time.perf_counter)` with:
- `begin() -> None` and `end() -> None`: wrap each request. They accumulate charged time `max(0, dt − 0.00075)`.
- `iters(pot: int, street: Street, hand_index: int) -> int`: returns a value from the ladder `{0, 16, 32, 64, 128, 256}`.
  - Base level: 32 when pot ≤ 8, 64 when pot ≤ 30, 128 above that.
  - Pace adjustment from `projected = used / (hand_index + 1) × hands`: if projected > `target_s`, drop one level per 20% overshoot; if projected < 0.5 × `target_s`, raise one level.
  - **Hard rule:** if used > 6.0 s, always return 0.

- [ ] **Step 1: Write the failing tests**
```python
def test_slow_clock_stays_under_bank():          # Review Focus 3: fake clock, each decision costs 3× nominal
    total = simulate_half(TimeManager(clock=fake), cost_per_iter=3*NOMINAL)
    assert total < 8.0
def test_big_pot_gets_more_iters_than_small():
def test_zero_iters_after_6s():
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): pot-weighted time manager`.

### Task 10: Features and heuristic policy

**Files:** Create `bot/features.py`, `bot/heuristic.py`. Test: `tests/test_heuristic.py`.

**Interfaces — Consumes:** Tasks 6–9. **Produces:**
- `@dataclass(frozen=True) Features(street: Street, pre: float | None, eq: float, now: float, nut: int, spr: float, pot_odds: float, in_position: bool, opp_raised_this_street: int)`
- `compute_features(obs: Observation, iters: int, rng) -> Features`. Preflop uses the table; postflop uses Task 7.
- `@dataclass(frozen=True) HeuristicParams` with tunable fields and defaults:
  - `open_min=0.44`, `threebet_min=0.53`, `call_margin=0.02`
  - `range_discount=(0.10, 0.08, 0.06)` (subtracted from equity per opponent raise, by street)
  - `value_raise=(0.70, 0.75, 0.80)` flop/turn/river
  - `pot_vs_half_cut=0.85`, `bluff_freq=0.10`
  - `nonnut_penalty=0.08` (applied when `nut & 0b110` is set and `nut & 1` is clear)
- `heuristic_action(obs, feats: Features, params: HeuristicParams, rng, opp=None) -> Action`, always ∈ `obs.legal_actions`.
  - Effective equity = `eq − range_discount·opp_raises − nonnut_penalty·[non-nut on dangerous board]`.
  - Raise when effective equity > `value_raise[street]`; choose RAISE_POT if > `pot_vs_half_cut`, else RAISE_HALF.
  - Call when effective equity > pot odds + `call_margin`.
  - Otherwise check or fold (never fold when free). Bluff with `bluff_freq` in position when checked to.

- [ ] **Step 1: Write the failing tests**
```python
def test_always_legal_on_10k_random_observations():   # drawn from simulated games vs random bot
def test_never_folds_when_free():
def test_all_in_corners_are_legal():                  # Review Focus 2: to_call >= my_stack; opp all-in; only CHECK_CALL legal
def test_calls_with_nuts_on_river():                  # nut & 1 on river, to_call>0 → not FOLD
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): features and parameterized heuristic policy`.

### Task 11: Layered policy and `agent.py` wiring

**Files:** Create `bot/policy.py`. Modify `agent.py` (YOUR CODE section + warmup area only). Test: `tests/test_policy.py`.

**Interfaces — Produces:**
- `warmup() -> None`: loads all tables; must finish < 6 s.
- `start_half(config: AgentConfig, memory: dict) -> None`: sets `memory["tm"]`, `memory["rng"] = np.random.default_rng(<fixed>)`, `memory["opp"]`, and `memory["config_ok"] = (config matches the blueprint metadata)`.
- `decide(obs, memory) -> Action`: layers blueprint (if loaded and `config_ok`) → exploit → heuristic. **Each layer is in its own try/except.** The final fallback is `CHECK_CALL if obs.to_call == 0 else FOLD` filtered by `legal_actions`. The result is asserted to be in `legal_actions`; if it isn't, return the fallback.
- `hand_finished(result, memory) -> None`: never raises (the whole body is in try/except).

- [ ] **Step 1: Write the failing tests**
```python
def test_layer_exception_falls_back_legally(monkeypatch):    # heuristic raises → still a legal action
def test_config_mismatch_uses_heuristic():                   # Review Focus 1: starting_stack=100 → blueprint never consulted
def test_warmup_under_6s():
def test_zero_errors_vs_whole_zoo():                         # play_half vs each ZOO bot, 200 deals, errors_a == 0, bank_a < 3.0
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement, and wire `agent.py`.
- [ ] **Step 4:** Run → PASS. Also run `.venv/bin/python -m pytest -q` (all tests, including root `test_agent.py`) → PASS.
- [ ] **Step 5:** Commit: `feat: layered policy wired into agent.py`.

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

### Task 13: Opponent model

**Files:** Create `bot/opponent.py`. Test: `tests/test_opponent.py`.

**Interfaces — Produces:**
`class OpponentModel`, which lives in memory:
- `observe(obs: Observation) -> None`: records the opponent actions in `obs.action_history` not seen yet, keyed by (street, facing_bet: bool), and remembers the last observation of the hand.
- `finish(result: HandResult) -> None`: infers the opponent's terminal action if they folded to our last bet (F11), and records showdown pairs `(opp action class per street, omaha_score(opp_hole, board))`.
- Properties: `fold_to_bet(street) -> tuple[float, int]` (rate, n), `raise_rate(street) -> tuple[float, int]`, `vpip`, `showdown_min_category(action_class) -> int | None`.
- Detectors:
  - `is_check_folder()`: ≥ 12 consecutive bet-facing folds and 0 raises in the last 40 decisions
  - `is_station()`: fold_to_bet < 0.05 with n ≥ 60
  - `is_maniac()`: raise rate > 0.55 with n ≥ 80

  Each detector **re-evaluates every hand**, so it reverts if the evidence goes away.

- [ ] **Step 1: Write the failing tests**
```python
def test_detects_checkfolder_after_switch():      # feed checkfold_after(50) traces; detected by hand 70, not before 50
def test_station_and_maniac_detection_thresholds():
def test_infers_fold_to_our_bet_from_result():
def test_hand_end_batch_never_raises():           # Review Focus 4: 20 results at once incl. opp_hole None, split, hands we never acted in
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): opponent model with reversible detectors`.

### Task 14: Exploit modifiers, then ship v2

**Files:** Create `bot/exploit.py`. Modify `bot/policy.py`. Test: `tests/test_exploit.py`. Results go in `docs/results/v2.md`.

**Interfaces — Produces:** `apply_exploits(base: Action, obs, feats: Features, opp: OpponentModel, rng) -> Action`:
- **check-folder** → RAISE_HALF whenever legal (I2), otherwise base.
- **station** → no bluffs (a bluff raise becomes CHECK_CALL); value threshold −0.08 (thinner value).
- **maniac** → call threshold −0.06; no bluff raises.
- **I6:** if `showdown_min_category` for the opponent's current action class is ≥ our made category, then fold or check instead of calling a big bet (requires n ≥ 5 showdowns).

- [ ] **Step 1: Write the failing tests**
```python
def test_checkfolder_mode_only_returns_legal_raises_or_base():
def test_gain_vs_checkfold_after_300():        # paired_diff(v2, v1, {checkfold_after(300)}, 3 seeds) lower CI > 0
def test_no_regression_vs_untargeted():        # paired_diff on HOLDOUT opponents: lower CI > -0.05 bb/deal
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5: Milestone M2.** Expected points on held-out ≥ v1, no new errors, bank < 3.0 s. Run `./tools/check_build.sh`, then commit, push and **submit v2**.

## Stage 3 — CFR blueprint (ship v3 only if it beats v2)

### Task 15: Exact betting tree

**Files:** Create `bot/tree.py`. Test: `tests/test_tree.py`.

**Interfaces — Produces:**
- `@dataclass BettingTree`:
  - `n_dec: int`, `street: np.ndarray[int8]`, `actor_is_button: np.ndarray[bool]`, `legal: list[tuple[Action, ...]]`
  - `child: np.ndarray[int32] [n_dec, 4]`: −1 = illegal, ≥0 = decision id, ≤ −2 = terminal `−2 − tid`
  - `term_fold_by_button: np.ndarray[int8]`: 1 = button folded, 0 = BB folded, −1 = showdown
  - `term_contrib: np.ndarray[int16] [n_term, 2]` (button, bb)
  - `key_to_node: dict[tuple[int, ...], int]`
- `build_tree() -> BettingTree`: exhaustive DFS with `game.Game` (default constants).
- `node_of(obs: Observation, tree) -> int | None`: key = `tuple(int(a.action) for a in obs.action_history)`.

- [ ] **Step 1: Write the failing tests**
```python
def test_tree_size():  t = build_tree(); assert t.n_dec == 15952 and n_terminals(t) == 25538   # F1
def test_preflop_nodes(): assert (t.street == 0).sum() == 62
def test_every_live_obs_maps_to_node_with_same_legal_actions():   # 2000 simulated decisions
def test_unknown_history_returns_none():                          # Review Focus 5
def test_terminal_contributions_match_game():                     # replay 500 random action paths
```
- [ ] **Steps 2–4:** fail → implement → pass. Build time must be < 2 s (it runs at startup).
- [ ] **Step 5:** Commit: `feat(bot): exact betting tree with history lookup`.

### Task 16: Card abstraction (buckets)

**Files:** Create `bot/abstraction.py`, `train/fit_buckets.py`, `data/buckets.npz`. Test: `tests/test_abstraction.py`.

**Interfaces — Produces:**
- `NUM_BUCKETS = {PREFLOP: 200, FLOP: 100, TURN: 100, RIVER: 100}`
- Bucket definitions:
  - preflop: quantile of `preflop_strength`
  - flop/turn: a 10×10 grid of (quantile of `strength_now`) × (quantile of `equity_vs_random`), so made hands and draws separate (Johanson 2013 / Ganzfried 2014 rationale)
  - river: 100 quantiles of equity, with the top 5 buckets reserved for `nut & 1`
- `bucket(street, hole, board, iters: int, rng) -> int`
- Offline `fit_buckets.py`: samples 200k situations per street at 2,000 iterations and writes the quantile edges.

- [ ] **Step 1: Write the failing tests**
```python
def test_bucket_range_and_determinism():
def test_bucket_occupancy_roughly_uniform():          # fresh 20k sample: every bucket within [0.4×, 2.5×] expected
def test_runtime_bucket_agrees_with_offline():        # iters=128 vs iters=2000: same or adjacent bucket ≥ 80% (flop, turn)
def test_bucket_time_p99_under_1_5ms():
```
- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5:** Commit: `feat(bot): 2-D potential-aware card abstraction`.

### Task 17: Training deal dataset

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

### Task 22: Robust counter-strategies (restricted Nash response)
Extend `train/mccfr.py` with `--fixed-opponent <archetype> --p 0.5`. The opponent plays the archetype's abstracted policy with probability p and plays freely otherwise [L: Johanson 2007]. Train counters for `station`, `maniac` and `mc_equity`. At runtime, `exploit.py` switches to a counter only when the matching detector fires [L: Bard 2013]. Test: `paired_diff` vs v3 on held-out, lower CI ≥ −0.02, plus a gain vs the target archetype.

### Task 23: Variance-aware play (I5)
Add `HeuristicParams.allin_variance_guard`: preflop, with `pre` in [0.42, 0.58] facing an all-in-sized raise, prefer fold or call over re-raising all-in when the opponent model says we have a postflop edge. Test: compare **expected points** (not bb) over held-out with 10 seeds; adopt only if points increase.

### Task 24: Release checklist (run before every submission)
- [ ] `.venv/bin/python -m pytest -q` passes.
- [ ] Real judge: our bot vs every baseline plus 2 LLM bots, 2 seeds, halves run separately with restarts. Result: 0 errors, no FORFEITED, `thinking` < 3.0 s per half.
- [ ] `./tools/check_build.sh` → `ALL CHECKS PASSED`, and the image size is reported.
- [ ] `git status` is clean, `data/*.npz` are committed, and each file is < 100 MB.
- [ ] `git push`; submit the full SHA from `git rev-parse HEAD`; keep the same team name.

## Validation protocol (summary)

1. **Unit correctness:** fast evaluator ≡ reference evaluator; tree ≡ game engine; the policy is always legal (Tasks 6, 15, 10, 11).
2. **Simulator fidelity:** the simulator reproduces the real judge's F10 value exactly (Task 2).
3. **Comparisons:** every decision about a candidate uses `paired_diff` with common seeds and opponents (I10). Accept a change only when the lower 95% bound > 0 on the **training** split and there's no significant regression on the **held-out** split.
4. **Tournament proxy:** expected points from a full round robin with our candidate inserted into the zoo (incl. LLM bots). The tie-break proxy is mean bb/deal.
5. **Robustness:** LBR exploitability (Task 21); 3× slow-clock time test (Task 9); zero errors across ≥ 50 full-length halves (Task 24).
6. **Deployment:** `check_build.sh` on the exact submitted commit.

## Open decisions for the user

- **Team name** (Task 1). It must be identical on every resubmission.
- **Deadline.** The guide's `[[Date and time]]` is still a placeholder. If it's within ~24 h, stop after Stage 2 (v2). Stage 3 needs roughly a day, including dataset generation and training.
