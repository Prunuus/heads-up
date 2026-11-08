### Task F2 (controller-added, ruling R49): register the three second-wave competitor bots as a held-out FIELD2 split

Same rules and pattern as Task F. Read `task-F-brief.md` for them; the FIELD registration it produced is already on `kevin`. This brief changes only the following.

| ZOO key | source folder (read-only) | runtime files |
|---|---|---|
| `field2_solver` ("Solverline") | `/home/kchen/code/competitor-bots/hard_solver/` | agent.py, preflop.npz, river.npz, pf_range.npy |
| `field2_bayes` ("Posterior") | `/home/kchen/code/competitor-bots/hard_bayes/` | agent.py, tp_rank.npy, pf_hs.npy |
| `field2_adaptive` ("Metagame") | `/home/kchen/code/competitor-bots/hard_adaptive/` | agent.py, ptab.npy, pfeq.npy |

- **Copy.** Copy into `sim/opponents/field2/<name>/` unmodified, with a sha256 table in `sim/opponents/field2/README.md`. Don't copy `dev/`, harness scripts or leftover dev files (e.g. `sdcore.py`).
- **Split.** Register a new `FIELD2` split.
  - TRAIN = ZOO − HOLDOUT − FIELD − FIELD2.
  - The tuner must refuse FIELD2.
  - The CLIs accept `field2`.
  - The test that pins `HOLDOUT | TRAIN | FIELD | FIELD2 == set(ZOO)` must hold.
- **Tests.** Reuse Task F's guard (sys.monitoring swallowed-exception hook + raises > 0) in a fast smoke test per bot (few deals vs calling_station) and a slow 200-deal test.
- **No information run.** The controller evaluates candidates against FIELD2 itself.
- **CPU.** `nice -n 19`, ≤ 2 processes. A gate job runs on all CPUs.
- **Constraints.** Work on branch `kevin` in `/home/kchen/code/heads-up-bundle`. Don't touch `bot/`, `agent.py`, `data/` (an untracked `data/blueprint.npz` is there; leave it alone), or organizer files. Commit messages are one simple sentence, with no trailers. Never push.
