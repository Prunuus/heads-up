---
name: submission-process
description: "How the TAMU Datathon PLO5 bot gets submitted — the current form link, the exact fields, and what happens on a failed build"
metadata:
  node_type: memory
  type: reference
  originSessionId: c0d55f53-d291-4faf-88d2-40523d153176
  modified: 2026-11-08T01:12:46.531Z
---

**The form.** The current submission form, from the latest participant guide (user-provided, 2026-11-07): https://forms.gle/a6Nr9VxXD9RYvi9x7. It replaces the older link https://forms.gle/Jx9RWE5VgR8dbbcj9.

**What the form needs (guide page 13):**
- The team name, identical on every resubmission. Ours is "3XKO", which must also match `PARTICIPANT` in agent.py and the registration.
- The repo URL in exactly the form `https://github.com/owner/repo`. Ours is https://github.com/Prunuus/heads-up, and it must be public.
- The full 40-character SHA of a pushed commit (`git rev-parse HEAD`). Short IDs are rejected.

**Build behaviour.** Every submission is built and smoke-tested under tournament limits. A failed build keeps the previous working bot, and the latest successful build is the one that plays. So an early known-good submission is a free safety net.

**Rules source.** If the organizers retune settings, the upstream starter repo's `game.py` is the source of truth; check it before the final submission. The guide gives no deadline (it still shows a placeholder). The user's deadline is 2026-11-08 12:00 CST.

**Who submits.** Only the user pushes and submits; never do either without their OK.

Related: [[plo5-bot-sdd-execution]], [[tournament-environment]].
