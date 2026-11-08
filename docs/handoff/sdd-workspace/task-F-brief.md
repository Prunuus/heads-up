### Task F (controller-added, ruling R35): register the agentic competitor bots as the FIELD split

**Why.** Three competitor bots were built by separate agents acting as independent teams. Each had only the organizer's starter kit and the participant guide, which mimics how other teams build their bots with AI coding tools. They become a held-out **FIELD** split for ship decisions: we test against them and never tune on them.

**Sources (read-only; never modify anything there):** `/home/kchen/code/competitor-bots/`

| name (ZOO key) | source folder | runtime files | notes |
|---|---|---|---|
| `field_equity` ("EquityEdge") | `agentic_equity/` | `agent.py`, `preflop_eq.npy` | data file loaded next to `agent.py` |
| `field_gto` ("Equilibrium") | `agentic_gto/` | `agent.py`, `river_cfr.json` | data file loaded next to `agent.py` |
| `field_exploit` ("SharkTank") | `agentic_exploit/` | `agent.py` | builds tables at startup, 1.3–2.2 s |

- Each folder's `REPORT.md` describes the bot: strategy, results, known weaknesses.
- The `dev/` folders are development tools. Don't copy them.
- The bots import the organizer's `game.py` and `evaluator.py`. Those are byte-identical to this repo's root files (sha256 checked by the controller).

**Deliverables** (branch `kevin-field`, worktree `/home/kchen/code/heads-up-bundle-field`, base ff537d2):

1. **Copy the runtime files.** Copy each bot's runtime files, unmodified, into `sim/opponents/field/<name>/`.
   - Record each source file's sha256 in `sim/opponents/field/README.md`, with a one-paragraph description taken from that bot's REPORT.md.
   - If a bot needs a change to load in our sim harness (a path, an import), make the smallest adapter you can outside the copied file. Only if that's impossible, make a minimal documented edit. Never change a bot's strategy.
2. **Register them** in `sim/opponents/__init__.py`.
   - Add ZOO entries with picklable factories (R3; e.g. `sim.match.bot_file_factory`).
   - Add a new `FIELD` split.
   - **TRAIN must exclude FIELD explicitly.** Today `TRAIN = set(ZOO) - HOLDOUT`, which would silently make them training opponents.
   - Make the CLIs that take an opponent split accept `field`: `sim.tournament --opponents`, and `sim.heuristic_bot --opponents`.
   - `all` may include FIELD; document that in the README.
3. **Tests** in `tests/test_field.py`:
   - Each FIELD bot loads through ZOO and plays a short half vs calling_station with 0 errors. This is the fast smoke test: few deals, keep it to a few seconds total.
   - A `@pytest.mark.slow` 200-deal version per bot.
   - FIELD is disjoint from TRAIN and from HOLDOUT.
4. **Information run (not a gate).**
   - v1-untuned (`bot.policy.PolicyBot()` at this branch's base) vs each FIELD bot: 3 seeds × 1,000 deals, duplicate, with paired CIs.
   - Also record each FIELD bot's bank use per half and its errors in our sim. A FIELD bot that runs out of bank plays check/fold, which would overstate our edge: flag every half where that happens.
   - Results go to `docs/results/field.md`.

**CPU.** Run heavy jobs as `taskset -c 8-15 nice -n 19 … --workers ≤ 4`. CPUs 0–7 belong to Track A's timing-sensitive work.

**Constraints.**
- Commit on `kevin-field` only. Never push.
- Commit messages are one simple sentence, with no body and no trailers (R36).
- Don't modify organizer files, `bot/`, or `agent.py`.
- Python 3.12 syntax.
