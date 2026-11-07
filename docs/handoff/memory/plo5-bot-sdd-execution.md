---
name: plo5-bot-sdd-execution
description: TAMU Datathon PLO5 poker bot — plan executing via subagent-driven development; where the ledger lives and how to resume
metadata:
  node_type: memory
  type: project
  originSessionId: b4276c44-89ab-406f-9078-0e01542ecd72
  modified: 2026-11-07T22:15:50.470Z
---

**DEADLINE: 2026-11-08 12:00 local (CST).** Target: v2 submission-ready by about 01:00, CFR training overnight on the user's PC (high-end CPU plus dedicated GPU), the v3 gate around 08:00, and final checks around 10:00. Tasks 22 and 23 are cut; revisit them only if time remains (R27). Track C, branch `kevin-opp` in `../heads-up-bundle-opp`, holds Task 13 (R28).

The work runs on TWO TRACKS (ruling R24): Track A is branch `kevin` in this repo; Track B is branch `kevin-cfr` in the sibling worktree `../heads-up-bundle-cfr` (CFR pipeline, Tasks 15–19), merged back locally later. Re-reviews of small fixes use opus at high effort; everything else uses opus max.

Since 2026-11-07 the user has had me execute `docs/superpowers/plans/2026-11-07-plo5-poker-bot.md` with superpowers:subagent-driven-development. The work happens on branch `kevin` (the user's branch; remote github.com/Prunuus/heads-up).

The ledger is at `.superpowers/sdd/2026-11-07-plo5-poker-bot/progress.md` (git-ignored). It holds the pre-flight rulings R1–R16 and per-task completion lines. To resume, trust the ledger and `git log`, and continue at the first task that has no `complete` line.

**Why:** the user warned the session may be terminated over usage limits, so progress has to survive in files.

**How to apply:**
- Use `.venv/bin/python`. The local venv is Python 3.14 but the tournament runs 3.12, so avoid 3.13+ features.
- Docker isn't installed, so `tools/check_build.sh` can't run.
- Never push or submit; both are the user's call.
- The team name is still a placeholder (ruling R1).
- On 2026-11-07 the user said: "each task implemented with as much effort as possible. Don't be shy on usage." So every subagent (implementer, reviewer, re-reviewer, audit) runs on opus at effort max. This supersedes the earlier cost caution; see ledger rulings R18–R20.
