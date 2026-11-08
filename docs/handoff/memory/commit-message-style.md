---
name: commit-message-style
description: "User's rule for every git commit — one simple sentence, no body, and no Claude co-author/attribution trailer"
metadata:
  node_type: memory
  type: feedback
  originSessionId: c0d55f53-d291-4faf-88d2-40523d153176
  modified: 2026-11-08T01:02:08.815Z
---

Every commit message is one simple sentence, for example `Add the context folder with the participant guide`. No body, no trailers, and never a `Co-Authored-By: Claude …` line or any other attribution to me.

**Why:** the user said so on 2026-11-07, for "this and future sessions". It overrides the harness's default commit-attribution reminder and the old ledger constraint that required the trailer.

**How to apply:**
- This covers commits I make and commits by every subagent I dispatch. Put the rule in each implementer brief and in fix-round messages.
- Merge commits keep git's default one-line message.
- Existing commits are not rewritten unless the user asks.

Related: [[plo5-bot-sdd-execution]].
