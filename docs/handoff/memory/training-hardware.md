---
name: training-hardware
description: "The PC that runs this project and all offline training — AMD Ryzen 7 9700X 8C/16T, 15 GB RAM in WSL2, AMD GPU (no CUDA)"
metadata:
  node_type: memory
  type: user
  originSessionId: c0d55f53-d291-4faf-88d2-40523d153176
  modified: 2026-11-08T00:09:10.840Z
---

**The machine.** The user's home PC is now both the controller machine and the training machine (since 2026-11-07):
- WSL2 Ubuntu 24.04.
- AMD Ryzen 7 9700X, 8 cores / 16 threads. SMT siblings are logical CPUs 2k and 2k+1.
- 15 GB RAM visible to WSL, about 950 GB disk.
- AMD GPU, not CUDA, so plan training for the CPU (numba).

**CPU split (ledger ruling R32).** When two tracks run at once, timing-sensitive Track A work is pinned to `taskset -c 0-7`. Bulk jobs (dataset generation, training) run as `taskset -c 8-15 nice -n 19`.

**Sim worker counts (R32).** Use `--workers 8` when the machine is otherwise idle, and 4 while another heavy job runs.

**Why:** the user doesn't want training capped by the old Mac (4 performance cores). The tournament bot itself still runs on 1 CPU.

**How to apply:** size dataset and training jobs to 16 threads and 15 GB RAM, with checkpoint/resume and a `train/README.md` runbook (R23).

Related: [[plo5-bot-sdd-execution]].
