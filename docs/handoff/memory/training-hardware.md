---
name: training-hardware
description: "User has access to a high-end CPU machine for offline training — don't size training to the Mac"
metadata:
  node_type: memory
  type: user
  originSessionId: b4276c44-89ab-406f-9078-0e01542ecd72
  modified: 2026-11-07T20:48:44.709Z
---

The user has access to a high-end CPU machine for offline training (stated 2026-11-07), so training doesn't have to fit their Mac (4 P-cores + 6 E-cores). The tournament bot itself still runs on 1 CPU at match time. Only offline work (dataset generation, MCCFR training) can use the big machine.

**Why:** they don't want training compute capped by the laptop.

**How to apply:**
- Make training scripts portable and scalable to many cores, with checkpoint/resume and a runbook (ledger ruling R23).
- Before a long training run, ask the user for the machine's core count and OS, and whether they run it themselves.
- See [[plo5-bot-sdd-execution]].
