---
name: tournament-environment
description: "Where and how the tournament actually runs bots (GCP k8s VM, ~96 vCPUs) and the bot changes that implies (ruling R30)"
metadata:
  node_type: memory
  type: project
  originSessionId: b4276c44-89ab-406f-9078-0e01542ecd72
  modified: 2026-11-07T22:43:44.065Z
---

On 2026-11-07 the organizer said the bots run on a **Kubernetes cluster on a GCP VM with roughly 96 vCPUs** (the organizer was unsure of the exact count). Each bot is a Docker container built from `docker/Dockerfile.agent`. The guide still gives each bot 1 CPU and 4 GB. The 96 vCPUs are shared across many concurrent matches, so a bot does not get more compute.

**Why it matters:** one GCP vCPU is a hyperthread, so a contended VM likely runs our single-threaded bot 1.5–3× slower than the dev Mac's performance cores, with timing jitter from Kubernetes CPU limits. Other teams' bots slow down the same way, so more of them will empty their 8 s time bank mid-half.

**How to apply (ledger ruling R30):**
- **Task 11.** Measure a CPU-speed factor at warmup (floor 1.0) and feed it to `bot.timing.grace_for_cpu_factor`. Re-estimate the factor online from measured decision times.
- **Task 12.** Add a per-bot `time_scale` to `sim.match` and run tuning and evaluation at about 2.5× as the primary condition.
- **Startup.** Keep it at about 2 s or less on the Mac; the hard limit is 15 s.
- **Docker check.** `tools/check_build.sh` must run on the user's PC (it has Docker; the Mac doesn't) before any submission.
- **Training.** No change to the CFR training or blueprint design.
- **Open questions for the organizer:** the per-container CPU request and limit, the machine type, concurrent matches per VM, and whether the judge and both bots share one pod.

Related: [[plo5-bot-sdd-execution]], [[training-hardware]].
