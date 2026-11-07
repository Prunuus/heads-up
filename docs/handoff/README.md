# Handoff: PLO5 poker bot, moving from the Mac to the PC

This is a snapshot taken on 2026-11-07 (CST). **The submission deadline is 2026-11-08 at 12:00.**

This folder carries everything a fresh Claude Code session on another machine needs to continue the work exactly where it stopped:
- the plan
- the progress ledger
- every task report
- the controller's rulings
- the memory notes

## 1. What this project is
- **The challenge.** We are writing a bot for the TAMU Datathon heads-up Short Deck 5-card Pot-Limit Omaha bot challenge. `README.md` in the repo root is the organizer's README.
- **The plan.** `docs/superpowers/plans/2026-11-07-plo5-poker-bot.md` holds the research, the strategy and 24 tasks. Part A explains why the plan looks the way it does; Part B is the tasks.
- **How the plan is run.** The plan is executed with the **superpowers:subagent-driven-development** skill. A controller session sends each task to a fresh implementer subagent, then a reviewer, then a fix loop, and records everything in a ledger.

## 2. State at the snapshot

Branches live on `origin` (github.com/Prunuus/heads-up). Each branch was worked in its own worktree on the Mac:

| Branch | Purpose | Contents |
|---|---|---|
| `kevin` | **Track A**: heuristic bot (v1/v2) | Tasks 1–9 done and reviewed, plus the audit fixes (I1–I6). **Task 10** (features and heuristic) was in progress. |
| `kevin-cfr` | **Track B**: CFR pipeline (v3) | Task 15 (betting tree) done and reviewed, and `kevin` merged in at `66a3d73`. **Task 16** (card buckets) was in progress. |
| `kevin-opp` | **Track C**: opponent model | Forked from `kevin@66a3d73`. **Task 13** (opponent model) was in progress. |
| `kevin-handoff` | this handoff | `docs/handoff/` only, on top of `kevin` |

`licheng` belongs to a teammate. It is not part of this work, so don't touch it.

**The ledger is the source of truth for progress.** It is `sdd-workspace/progress.md`. A task is done only if the ledger has a `Task N: complete` line for it. Every `Ruling:` line is a decision the controller made on the user's behalf.

### Tasks that were in progress when the snapshot was taken
For each of Tasks 10, 13 and 16, check the ledger and that branch's git log on the PC:
- **Ledger says `complete`:** nothing to do.
- **There are commits and a `task-N-report.md`, but no review yet:** generate the review package and dispatch the task reviewer.
- **No commits (the agent died mid-task):** re-dispatch an implementer with the same brief, `sdd-workspace/task-N-brief.md`. Point it at the existing report file if there is one, and tell it to resume.

## 3. Setting up the PC

```bash
git clone https://github.com/Prunuus/heads-up.git heads-up-bundle
cd heads-up-bundle
git fetch --all
git checkout kevin
# The worktrees must be SIBLING folders, not inside the repo (ruling R24b: pytest would collect duplicate tests).
git worktree add ../heads-up-bundle-cfr kevin-cfr
git worktree add ../heads-up-bundle-opp kevin-opp

python3 -m venv .venv            # Python 3.12+; the tournament image runs 3.12
.venv/bin/pip install -r requirements-dev.txt      # fastapi, uvicorn, pytest, httpx, numpy
.venv/bin/pip install -r requirements-train.txt    # + numba, for offline training only
.venv/bin/python -m pytest -q                      # should pass, with a few deselected slow tests
```

On Windows, use PowerShell with `.venv\Scripts\python`, or use WSL. WSL is recommended, since the organizer's `tools/check_build.sh` needs bash and Docker.

**If the PC has Docker, run `./tools/check_build.sh` there.** The Mac had no Docker, so the tournament-identical image build and smoke test have never been run. This must pass before submission.

### Restoring the ledger so the skill resumes instead of restarting
The skill keeps its workspace at `<repo>/.superpowers/sdd/2026-11-07-plo5-poker-bot/`. That folder is git-ignored, which is why a copy travels here:

```bash
mkdir -p .superpowers/sdd/2026-11-07-plo5-poker-bot
cp docs/handoff/sdd-workspace/* .superpowers/sdd/2026-11-07-plo5-poker-bot/
```

Paths inside the copied files point at the Mac (`/Users/kchen/Desktop/code/FALL26/...`). Read them as the matching paths on the PC.

### Restoring the memory notes (optional)
`docs/handoff/memory/` holds the two memory notes plus their index. Either copy them into the PC's Claude Code project memory folder (it is created under `~/.claude/projects/<name derived from the repo path>/memory/` after the first session in this folder), or just have the new session read them.

## 4. Starting the new Claude Code session
You need Claude Code with the **superpowers** plugin installed, since the skills `subagent-driven-development` and `using-git-worktrees` come from it. Open the repo folder and send:

> Read docs/handoff/README.md, then docs/handoff/memory/*.md, then the ledger at .superpowers/sdd/2026-11-07-plo5-poker-bot/progress.md. Resume executing docs/superpowers/plans/2026-11-07-plo5-poker-bot.md with superpowers:subagent-driven-development on the three tracks, following every ruling in the ledger. Deadline: 2026-11-08 12:00.

## 5. User directives (binding)
- **Maximum effort.** Implementers, task reviewers and audits run on **opus at effort max**. Scoped re-reviews of small fix diffs (≲300 lines) run on **opus at effort high**. Plumbing tasks (T11, T19, T24) may use opus at high.
- **Never push or submit without the user's OK.** Commit only on the track branches.
- **Tasks 22 and 23 are cut** (stretch), to be revisited only if v3 finishes with time to spare. Task 21 (the LBR probe) stays in full.
- **Training runs on the PC,** which has a high-end CPU and a dedicated GPU.
  - Task 17 and Task 18 scripts must be portable and scale to every core, with checkpoint/resume and a `train/README.md` runbook (ruling R23).
  - With a CUDA GPU available, weigh a GPU (full-width, vectorized) CFR trainer against the CPU numba MCCFR plan in Task 18, using measured throughput.
  - Size the bucket counts (`data/buckets.npz` metadata, R26) and the iteration target to this machine.
- **The team name is a placeholder.** `agent.py` has `PARTICIPANT = "KChen"` (ruling R1). It must match the registered team name before submission.

## 6. What comes next, and the timeline

**Track A** (`kevin`):
1. Finish T10.
2. T11: layered `PolicyBot` plus agent.py wiring. Read the T9, T15 and audit-fix carry-forward notes in the ledger first.
3. T12: tuning, which gives **v1**.
4. Merge `kevin-opp`.
5. T14: exploits, which gives **v2**.
6. Run T24 (the release checklist) and the final whole-branch review on v2, so **v2 is submission-ready by about 01:00**. It is the safety net.

**Track B** (`kevin-cfr`):
1. Finish T16.
2. T17: dataset.
3. T18: the trainer, then the **long training run on the PC overnight**.
4. T19: export.

**Then:**
1. Merge `kevin-cfr` into `kevin`.
2. T21: the LBR probe.
3. T20: the v3 gate. Ship CFR only if it beats v2 by the plan's criteria, around 08:00.
4. Run T24 and the final review again, around 10:00.
5. The user submits, before 12:00.

## 7. Submission (the user does this)
Push the final branch. Run `git rev-parse HEAD` to get the full 40-character SHA, then submit it with the team name and the repo URL through the form linked in the participant guide. The repo must be public. The tournament builds exactly that commit.
