---
name: plo5-bot-sdd-execution
description: TAMU Datathon PLO5 poker bot — plan executed via subagent-driven development on several track worktrees; where the ledger lives, how to resume, model/usage policy
metadata:
  node_type: memory
  type: project
  originSessionId: c0d55f53-d291-4faf-88d2-40523d153176
  modified: 2026-11-08T05:03:21.493Z
---

**DEADLINE: 2026-11-08 12:00 CST.** The user has me execute `docs/superpowers/plans/2026-11-07-plo5-poker-bot.md` as the controller, using the superpowers `subagent-driven-development` skill. The plugin has been installed since 2026-11-07 18:11. The loop is: implementer → task review → fix rounds → ledger line.

**Where things live (PC, WSL2):**
- Repo: `~/code/heads-up-bundle` (branch `kevin`, Track A).
- Sibling worktrees:
  - `-cfr` (`kevin-cfr`, CFR)
  - `-b400` (`kevin-b400`, the 400-bucket refit)
  - `-opp` (`kevin-opp`, exploits)
  - `-lbr` (`kevin-lbr`, the LBR probe)
  - `-field` (`kevin-field`, FIELD opponents)
  - `-handoff`
- The ledger is the source of truth: `~/code/heads-up-bundle/.superpowers/sdd/2026-11-07-plo5-poker-bot/progress.md`. It is git-ignored; the worktrees symlink to it.
- Interpreter: `~/code/heads-up-bundle/.venv/bin/python` (Python 3.12.3).
- Git pushes go through the user's `gh` login, and only with the user's OK each time.
- The competitor bots live in `~/code/competitor-bots/`.

**Why:** sessions can die on usage limits. One did at 20:45–23:00 on 2026-11-07. Progress has to survive in files: trust the ledger plus `git log`, and resume at the first task with no `complete` line. Use SendMessage to resume agents that were stopped by a limit.

**How to apply:**
- Never push or submit without asking the user.
- Model policy since 2026-11-07 ~23:03 (ruling R40, the user asked for token efficiency). It replaces "opus max everywhere".
  - Use the cheapest model that fits each role, and don't force an effort level.
  - Sonnet for plumbing implementers, reviews and re-reviews.
  - Opus only for judgment-heavy tasks (T20) and the final whole-branch review.
  - The controller runs merges, gate scripts and Docker checks itself.
- Team name: "3XKO". Commit messages: one sentence, no trailers.
- Tasks 22 and 23 are cut.

Related: [[training-hardware]], [[tournament-environment]], [[commit-message-style]], [[submission-process]].
