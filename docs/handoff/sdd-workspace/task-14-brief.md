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


---

## Controller additions (binding; from the ledger — read the lines cited)

- **Track C (R34).**
  - You work on branch `kevin-opp` in `/home/kchen/code/heads-up-bundle-opp`. It now holds T13 (`bot/opponent.py`, OpponentModel) merged with kevin's T10/T11 (merge 3d8f162).
  - Task 12 tunes HeuristicParams in parallel on Track A. Develop and gate against the CURRENT defaults: v1-untuned = `PolicyBot()` at 3d8f162.
  - After T12 lands, the controller merges kevin-opp into kevin and re-runs your M2 gate on the merged head.
  - Do not modify HeuristicParams defaults (T12 owns them) or `sim/match.py` (T12 adds `time_scale`). If an exploit needs a tunable, put it in your own module.
- **T13 notes (ledger "Task 13:" lines).**
  - Construct `OpponentModel(raise_cap=config.raise_cap, big_blind=config.big_blind)` in new_half, call `observe()` first in choose_action, and call `finish()` in on_hand_end.
  - Detector precedence: check-folder > maniac > station. Station and maniac both trip on never-fold-always-raise bots.
  - `is_station()` is postflop-only with a 150-decision window; `is_maniac()` uses a 200-decision window. Do not assume `is_station()` == fold_to_bet()/vpip.
  - Check-folder rule: ≥ 12 consecutive bet-facing folds, with no call/bet/raise since the run began. It detects `checkfold_after(50)` at hands 60–61.
- **T10 notes for T14.**
  - Station mode's "value threshold −0.08" must move BOTH `value_raise` and `thin_value`; v1's unopened bets are driven by `thin_value`.
  - Maniac mode's "call threshold −0.06" should mainly lower `range_weight` and `bet_range`; facing a bet, they dominate `call_margin`.
  - Against high-frequency bettors, lower `range_weight`: 0.5 measured +1.3 / +1.4 bb/deal vs always_half / maniac. v1 gives up −2.37 / −3.13 vs v0 there.
- **T11 notes for T14 (wiring, plus Minors in the code you touch).**
  - Wire OpponentModel into PolicyBot's new_half / choose_action / on_hand_end (the `memory["opp"]` slot, R10).
  - Replace the identity exploit slot in `bot/policy.py` with `apply_exploits`.
  - Fix T11 M1: tests/test_policy.py:245 `>=` → `>`.
  - Add the T11 M3 positive-path test: a legal exploit change replaces the base.
  - Fix the T11 M5 wording (policy.py:10, agent.py:53) to "base = blueprint else heuristic; exploits then modify the base".
- **Adaptive opponents (user info).** Other teams' bots track our stats over each 1,000-deal half and adapt.
  - Exploit modes must stay confidence-gated and reversible, must not create static tells, and must not over-commit to a misclassification.
  - Test against the adaptive zoo bots.
  - Report per-mode activation rates against every TRAIN and HOLDOUT bot; a false positive is a risk.
- **Plan tests and gate stand.**
  - Tests: `test_checkfolder_mode_only_returns_legal_raises_or_base`, `test_gain_vs_checkfold_after_300`, `test_no_regression_vs_untargeted`.
  - Milestone M2: expected points on held-out ≥ v1, no new errors, bank < 3.0 s.
  - Results go to `docs/results/v2.md`.
  - If the controller has registered the FIELD split (3 agentic competitor bots) by gate time, also report v2 vs FIELD (information only).
- **CPU (R32).** Run sims as `taskset -c 0-7 nice -n 10` with `--workers ≤ 2`; Task 12's tuner owns most of CPUs 0–7, and CPUs 8–15 belong to Track B. Report load averages with timing numbers.
- **R12.** Never push, never submit; the controller runs check_build.
- **Commits.** One simple sentence each, with no body and no trailers (R36).
- **Time box.** About 2 h of wall time.
