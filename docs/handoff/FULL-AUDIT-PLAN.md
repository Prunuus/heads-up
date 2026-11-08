# Full audit plan: final pre-submission check of the PLO5 bot

**Deadline:** submission 2026-11-08 12:00 CST. The audit's findings are due by ~09:45; any approved fix must be re-verified by ~11:15.

**Verified submission candidate:** `c1ded1661937b4ffce15d875e65b3f0ccaff89a4` on `kevin`. `origin/kevin` is at `f4868c6`, which adds only `docs/handoff/` and is identical for the bot.

## 0. Ground rules (binding)

- **Read-only by default.** Report findings; don't change code unless the user approves a specific fix.
- **The release bar for any change.** A change may replace the submission SHA only if the NEW commit passes all of the following. Otherwise `c1ded16` stays.
  1. Full default suite green (`.venv/bin/python -m pytest -q`; timing-threshold tests may flake under load, so re-run them alone).
  2. `./tools/check_build.sh` → ALL CHECKS PASSED, on a clean `git archive` snapshot of that commit.
  3. The blueprint probe inside the built image (`docs/handoff/verify/bp_probe.py`, command in `docs/handoff/README.md` §4) shows `blueprint loaded: True` and blueprint-answered decisions with 0 faults.
  4. Real-judge halves in the image at `--cpus 1` (`docs/handoff/verify/realjudge.sh`): 0 errors, no forfeits, judge-charged bank < 3 s per half.
  5. Strength ≥ c1ded16 on common seeds (paired CIs), measured with ≤ 2 workers on an otherwise idle machine. Machine load inflates our edge against time-adaptive bots by 1–3 bb/deal (ledger R39).
- Never push or submit without asking the user. Commit messages are one simple sentence, with no co-author or other trailers.
- Don't modify organizer files: `game.py`, `evaluator.py`, `judge_engine.py`, `baseline-bots/`, `docker/`, `tools/`.
- Use tokens deliberately: read the docs below instead of re-deriving them, and sample the code where a risk is named.

## 1. Read first (context)

1. `docs/handoff/README.md`: what ships, results, how to verify, where things are, open items.
2. `docs/superpowers/plans/2026-11-07-plo5-poker-bot.md`: the plan, its Global Constraints and Review Focus 1–5.
3. `docs/handoff/sdd-workspace/progress.md`: the ledger. Every `Ruling:` is a decision made on the user's behalf. The end has the final gate numbers, the final review verdict, and 65 `minor (deferred)` items with their triage.
4. `docs/handoff/sdd-workspace/task-*-report.md` and `task-*-review.md` as needed (T20 / T20b for the blueprint integration).
5. Organizer sources of truth: `README.md`, `game.py`, `judge_engine.py`, `docker/Dockerfile.agent`, `tools/check_build.sh`, the participant guide in `context/`. Upstream is https://github.com/tamu-datathon-org/heads-up-bundle; it was verified identical at 08:05 today.

## 2. Audit checklist

### A. Tournament correctness (highest priority; anything here is a must-fix)

- [ ] `agent.py`: edits only inside YOUR CODE and the warmup area; SERVER PLUMBING and the `PORT` lines untouched (diff against `main`).
- [ ] No path from `/new-half`, `/act` or `/hand-end` can raise into the plumbing. Every return value is in `obs.legal_actions`. `hand_end` never raises.
- [ ] Startup well under 15 s on 1 slow CPU, with every table loaded in warmup: `bot/policy.py` warmup, the tree, buckets, preflop and the blueprint self-test. No lazy load in the first `/act`.
- [ ] Time bank: per-decision cost, pacing, the hard stop, and behaviour under a 2–3× slower CPU. Note that the blueprint path has no cheaper mode near exhaustion; the worst half at 2.5× emulation was 3.2–3.75 s.
- [ ] Data files under `/app/data` on a read-only filesystem are loaded relative to `__file__`. The blueprint is ~51 MB with no Git LFS, in the commit, and not excluded by `.dockerignore`.
- [ ] `config_ok` vs `AgentConfig`: a retuned stack, blinds, raise cap or bank → silent heuristic-only, with no errors. That must be the case. Is it?
- [ ] `node_of` relies on the judge preserving legal-action order. Confirm in `judge_engine.py` and `game.py`.
- [ ] Memory over 1,000 hands (4 GB limit; the final review measured RSS flat at 179 MB), and a fresh process per half.
- [ ] `requirements.txt`: exactly `fastapi==0.156.1`, `uvicorn==0.54.0`, `numpy` unpinned, no torch.
- [ ] Team name `3XKO` in `agent.py` matches the registration.

### B. Rules and submission hygiene

- [ ] Repo public; every file < 100 MB; `git status` clean; the submitted commit is pushed (it is: an ancestor of `origin/kevin`).
- [ ] The form fields per the guide (form link, team, repo URL format, 40-char SHA).

### C. Strategy gaps and improvement candidates (rank by expected points; risk-weighted)

- [ ] **Draws vs strong bots.** 24 of 30 matches against the 6 competitor bots were chops. Is there a low-risk lever to turn chops into wins? Candidates: sizing, exploit gating, purify. Any lever must not create losses.
- [ ] **Exploit layer on top of the blueprint.** Maniac mode overrode 33–44% of blueprint actions against random and always_pot (the final review's note). Check whether it ever fires against strong or adaptive bots, and whether that helps or hurts.
- [ ] **Auto base switch** (preflop raise rate ≤ 0.12 → heuristic base). Check its robustness to an unseen strong-but-passive bot, and how close llm_mc_equity (0.15) and llm_ml_minded (0.18) sit to the line.
- [ ] **Exploitability.** The v3 LBR estimate is inconclusive: the model check failed because LBR re-simulation can't reproduce the blueprint's randomization. Wiring v3 as a `DistributionTarget` in `sim/opponents/lbr.py` would give a valid number; this is validation only.
- [ ] **Abstraction and training.** 400 buckets per street, with 4 underfilled buckets (strict xfails); 16.2B MCCFR iterations. Is more training or a refit worth it in the time left? Likely not before noon. Say why.

### D. Test and quality gaps

- [ ] Triage the 65 `minor (deferred)` ledger items again. The final review marked all shipped-code items LEAVE; challenge any you disagree with.
- [ ] Flaky timing tests (judge-fidelity 20 ms timeouts, p99 thresholds) and the 2 strict xfails.
- [ ] Any test that asserts nothing or can't catch the bug it claims to.

## 3. Deliverable

A findings report, in a file or in chat, with:
1. **Must fix before submit.** Section A or B problems, with file:line, evidence (command plus output) and a concrete fix.
2. **Should fix if time allows.** Expected gain, risk, and the time needed including re-verification.
3. **Leave.** A short list with reasons.
4. **Recommendation:** submit `c1ded16` as is, or apply named fixes and then re-run the full release bar (§0).

## 4. Time plan

| Time (CST) | Step |
|---|---|
| 08:30–09:45 | Audit (A and B first, then C and D) and the findings report |
| 09:45 | The user decides on fixes |
| 09:45–11:15 | Approved fixes plus the full release bar on the new commit (about 45–60 min per pass) |
| ≤ 11:30 | The user submits (the new SHA only if it passed everything; otherwise `c1ded16`) |
