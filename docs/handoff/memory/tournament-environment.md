---
name: tournament-environment
description: "Where the tournament runs bots (GCP k8s VM, ~96 shared vCPUs, 1 CPU/bot) and the bot changes that implies (ruling R30)"
metadata:
  node_type: memory
  type: project
  originSessionId: c0d55f53-d291-4faf-88d2-40523d153176
  modified: 2026-11-08T00:09:15.487Z
---

**What the organizer said (2026-11-07).** Bots run on a Kubernetes cluster on a GCP VM with roughly 96 vCPUs. Each bot is a Docker container built from `docker/Dockerfile.agent`, with 1 CPU and 4 GB per bot.

**Why it matters.** A vCPU is a hyperthread, so expect our single-threaded bot to run 1.5–3× slower than on a dev machine, with jitter. Other teams' bots slow down the same way and will empty their 8 s banks more often.

**How to apply (ledger ruling R30):**
- **T11:** calibrate a CPU-speed factor at warmup (floor 1.0) and feed it to `bot.timing.grace_for_cpu_factor`. Re-estimate it online during play.
- **T12:** add a per-bot `time_scale` to the sim, and tune and evaluate at about 2.5×.
- **Startup:** keep it at about 2 s or less.
- **Before any submission:** `tools/check_build.sh` must pass, which needs Docker.

**Measured on the PC.** Real uvicorn plumbing is 0.29 ms per request. The in-process TestClient is 1.9 ms per request, so it is not a valid proxy for bank usage (R31).

Related: [[plo5-bot-sdd-execution]], [[training-hardware]].
