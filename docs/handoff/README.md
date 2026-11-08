# Handoff: PLO5 bot, moving from the PC back to the Mac (2026-11-08 morning)

The bot is **finished, verified and pushed**. Deadline: **2026-11-08 12:00 CST**.

## 1. What to submit

| Field | Value |
|---|---|
| Form | https://forms.gle/a6Nr9VxXD9RYvi9x7 (the latest guide; the old form link is dead) |
| Team | `3XKO` (must match the registration exactly) |
| Repo | `https://github.com/Prunuus/heads-up` (public; no branch in the URL) |
| Commit SHA | **`c1ded1661937b4ffce15d875e65b3f0ccaff89a4`** (branch `kevin`) |

The tournament builds exactly that commit. A failed build keeps your previous working bot, and the latest successful build is the one that plays.

Earlier fallbacks, all on GitHub:
- v2 `2115725fd2a0f9fac4f0dab081ae84c7a60f93b7`
- failsafe `ff537d2a13dd0927508de34fe61b8e79d070b572` (already submitted)

## 2. What the bot is (v3.1)

- **CFR blueprint.** External-sampling MCCFR with Linear CFR weighting, run for 16.2 billion iterations. It uses 400 card buckets per street over the exact betting tree (15,952 decision nodes, 6.38M infosets). The export is `data/blueprint.npz` (51 MB, float16 average strategy).
- **Layers.** The blueprint is the base for every opponent. The base switches to the heuristic (v2's base) only against passive-preflop opponents: raise rate ≤ 0.12 after ≥ 60 hands, back at ≥ 0.20. The exploit layer (opponent-model detectors) sits on top.
- **Fallbacks.** On a node miss, a config mismatch, a missing or corrupt blueprint, or any exception, the bot falls back to the heuristic, and then to a guaranteed-legal action.
- **Startup.** Everything loads in `warmup()`: the tree, the buckets and the blueprint, with hash checks. The upstream organizer repo was checked at 08:05: `game.py`, the judge and the Dockerfiles are byte-identical to ours, so nothing was retuned.

## 3. Results (all reproducible; numbers in `sdd-workspace/progress.md`)

- **Real judge, in the tournament container.** `docker/Dockerfile.agent`, `--cpus 1`, read-only, no capabilities, a fresh process every half.
  - 12 matches: **11 wins, 1 chop** against the 4 baselines (+5.1 to +8.1 bb/deal), llm_aggressive (+3.7) and llm_mc_equity (+1.5 / +0.6 chop).
  - 0 errors, 0 forfeits, time bank at most **1.9 s** per half (8 s allowed).
- **vs 6 strong competitor-style bots** (built by top-tier agents as rival teams; `sim/opponents/field*`), 5 seeds each:
  - v3.1: **6 wins, 0 losses, 24 chops**.
  - v2: 0 wins, 19 losses.
- **v3.1 vs v2 head-to-head:** +0.54 bb/deal [+0.24, +0.84].
- **Held-out and train zoo bots:** about equal to v2 (TRAIN +0.09, not significant).
- **Exploitability (LBR probe).** v2 is beaten by **+3.68** bb/deal. The v3 estimate is **inconclusive**: the probe's model check failed (3.1% of actions unpredicted), so it is not claimed.
- **Checks on c1ded16:**
  - 669 tests pass;
  - `tools/check_build.sh` ALL CHECKS PASSED (startup 1.8 s, +21 MB image);
  - the final Opus review says **ready to ship** with no fixes.

## 4. Verify it yourself on the Mac: the tournament-like Docker run

Requirements: Docker Desktop running, and the repo on branch `kevin` at the SHA above.

```bash
git fetch origin && git checkout kevin && git pull     # HEAD must be c1ded16...
git ls-tree -l HEAD data/blueprint.npz                 # must show ~51052391 bytes: the trained weights are in the commit
```

**Step 1: the organizer's own check.** It builds the tournament base image and our image, checks the size cap, then starts the bot like a match: no network, read-only files except /tmp, 1 CPU, 4 GB, non-root. It then runs the smoke test.

```bash
./tools/check_build.sh                       # expect: ALL CHECKS PASSED
PLATFORM=linux/amd64 ./tools/check_build.sh  # optional on Apple Silicon: the tournament's CPU type (emulated, so slow timings are expected)
```

**Step 2: prove the trained weights load and answer inside that image.** Run this after step 1, which tags the image `poker-agent-check:latest`:

```bash
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=512m --cap-drop ALL \
  --security-opt no-new-privileges --cpus 1 --memory 4g -e PYTHONPATH=/app \
  -v "$PWD/docs/handoff/verify/bp_probe.py:/probe/bp_probe.py:ro" -w /app \
  poker-agent-check:latest python /probe/bp_probe.py
# expect: blueprint loaded: True | iterations: 16221206540
#         layers: {'blueprint': 132, 'heuristic': 0, 'node_miss': 0} | faults: all 0
```

If it ever says `blueprint loaded: False` or shows `heuristic` answers against this random opponent, the weights are not being used. Check `git ls-tree` above and that `data/` is not in `.dockerignore`.

**Step 3: real judge vs real opponent servers, halves separate, fresh processes.**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # once
bash docs/handoff/verify/realjudge.sh /tmp/realjudge    # ~5 min; prints per-half errors/bank and each match verdict
```

Expect 0 errors, `forfeited False`, bank well under 3 s per half, and AGENT1 WINS against the baselines.

**Optional: strength vs the 6 competitor bots** (CPU-heavy, about 10–30 min):

```bash
.venv/bin/python docs/handoff/verify/strong_eval.py
```

## 5. Where everything is

| Thing | Location | In git? |
|---|---|---|
| Submission code | branch `kevin` (`agent.py`, `bot/`, `data/`) | yes |
| This handoff | branch `kevin-handoff`, `docs/handoff/` | yes |
| SDD ledger, briefs, reports, review packages | `docs/handoff/sdd-workspace/` (copy of the PC's `.superpowers/sdd/2026-11-07-plo5-poker-bot/`) | yes, here |
| Claude memory notes | `docs/handoff/memory/` | yes, here |
| Verify scripts | `docs/handoff/verify/` | yes, here |
| Competitor bots (FIELD, FIELD2) | `sim/opponents/field/`, `sim/opponents/field2/` (originals in the PC's `~/code/competitor-bots/`) | yes |
| Training data (10M deals) and MCCFR checkpoints | PC only: `~/code/heads-up-bundle-cfr/train_data/` (d10m-b400, about 1.3 GB; checkpoints 357 MB each) | no; not needed to run or submit |
| Trainer and runbook | `train/mccfr.py`, `train/README.md` | yes |

## 6. Open items and known limits (none block submission)

- **Draws.** Against strong bots most results are chops (CI-based verdicts over 1,000 deals). v3.1 never lost one in 30, but wins only some.
- **Time bank on a slow host.** Under a 2.5× slow-CPU emulation, the worst half used 3.2–3.75 s (8 s limit). The real container measured ≤ 1.9 s.
- **Upstream retunes.** If the organizers change `game.py` constants (stack, blinds, raise cap), the bot silently plays heuristic-only, since the blueprint no longer matches. Re-check upstream (https://github.com/tamu-datathon-org/heads-up-bundle) before resubmitting.
- **Cut.** Tasks 22 and 23 (stretch) were not done, by decision.

## 7. Resuming with Claude Code on the Mac

Paths: the Mac repo is `~/Desktop/code/FALL26/heads-up-bundle`, and the PC repo is `~/code/heads-up-bundle`. Install the superpowers plugin (`/plugin`). Then copy `docs/handoff/memory/*.md` into the Mac's Claude project memory folder, or just have the session read them. Start the new session with:

> Read docs/handoff/README.md (branch kevin-handoff), docs/handoff/memory/*.md and docs/handoff/sdd-workspace/progress.md (especially the end). The bot is finished and pushed at c1ded16; help me verify and submit it. Never push or submit without asking me. Commit messages are one simple sentence, no co-author lines.

Every decision made on your behalf is a `Ruling:` line in `sdd-workspace/progress.md`. The morning summary lists them with their costs.
