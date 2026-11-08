### Task 20b (controller-added, R44/R47): hybrid base selector, v3.1

**Measured facts** (final blueprint, quiet gate, `train_data/gate_v3_final`; see the "GATE (a,b,c,e) DONE" ledger line):
- v3 (blueprint base + exploits) beats v2 (heuristic base + exploits) head-to-head: +0.54 bb/deal.
- v3 is far better against the strong competitor bots (FIELD): 0.67 vs 0.17 points at 1.0×, 0.83 vs 0.39 at 2.5×.
- v3 is worse against weaker held-out and train bots:
  - HOLDOUT: 0.92 vs 1.00 points, −0.41 bb/deal. It loses points to llm_opp_model and llm_time_careful, and bb to threshold_cat.
  - TRAIN: −0.15 to −0.24 bb/deal, whatever the purify setting.

**Goal.** Build v3.1, which keeps v3's edge against strong bots and v2's edge against weaker, exploitable bots. Have PolicyBot pick its base per opponent during the half: blueprint by default, heuristic when online evidence says the opponent is the kind v2 beats harder. The exploit layer stays on top of whichever base answers.

**Do.**
1. Find a cheap, robust online signal from data.
   - Look at per-opponent activation of the existing OpponentModel detectors (check-folder, station, maniac, fold_to_bet, raise rates, vpip); T14's report has activation tables.
   - Identify what separates the bots where v2 > v3 from FIELD/strong bots, where v3 > v2.
   - Prefer simple, monotone rules with minimum sample sizes and hysteresis. Defaulting to the blueprint is the safe choice.
2. Implement it in `bot/policy.py` as an option of PolicyBot (`base="auto"|"blueprint"|"heuristic"`), defaulting to `"auto"` only if it wins. It must be exception-safe and deterministic per half, with no new runtime dependencies, keeping per-decision cost within budget.
3. Add tests for the selector logic and its fallbacks.
4. Evaluate with the gate script.
   - Add a candidate label for v3.1 if needed.
   - Run sections a, b and e in a FRESH `--out` directory, at time_scale 1.0 at least (2.5 if time allows), with `--cpus` and `--workers 2`.
   - Report v3.1 against both v2 and v3.
   - Ship criterion: v3.1 ≥ v3 on FIELD (no loss of points), and better than v3 on HOLDOUT and TRAIN. Otherwise report "keep v3".

**CPU.** CPUs 8–15 are busy with the LBR probe until about 03:30. CPUs 0–7 carry a controller FIELD2 evaluation until about 03:15. Until then do light work only (`nice -n 19`). After that, use `--cpus 0-7 --workers 2`.

**Time box.** Results by ~04:30.

**Constraints.**
- Branch `kevin`, worktree `/home/kchen/code/heads-up-bundle`, HEAD 480cc66, which includes the committed final `data/blueprint.npz`. Don't modify or re-export the blueprint.
- Commit messages are one simple sentence, with no trailers.
- Never push.
