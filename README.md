# 3XKO poker bot v0.5

Deterministic heads-up short-deck PLO5 bot for the TAMU Datathon challenge.
Repository work is on **`licheng`**. Participant: `3XKO`; bot: `3XKO`.
Use your registered team name when submitting.

This builds on the supplied [v0.4 design](design/DESIGN.md). The user-authorized
v0.5 revision corrects aggression classification and overfolding, with controlled
development comparisons and a new sealed audit. Read
[the current release report](reports/v05/RELEASE.md) for results and limitations.
The [v0.4 report](reports/RELEASE.md) is historical. Passing unit tests
alone does not establish tournament readiness.

Install with Python 3.12 (the official container runtime) or Python 3.13:

```powershell
git switch licheng
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest -q
.venv/Scripts/python agent.py
```

On Linux/macOS use `.venv/bin/python`. The server reads `PORT` (default 5008)
and `AGENT_TOKEN`. It binds to `0.0.0.0` with the starter's unchanged health,
new-half, action, and hand-end protocol. No internet, writable game files,
ML packages, downloaded models, or absolute local paths are needed during play.
Runtime dependencies remain the starter's FastAPI and uvicorn versions.

In separate terminals, start an opponent and the official judge:

```powershell
.venv/Scripts/python baseline-bots/baseline_calling_station.py
.venv/Scripts/python judge_engine.py --seed practice-independent --deals 1000 --json practice.json
```

The judge plays two halves with seats swapped. Practice commands reuse processes;
the research runner restarts both servers for every half. Do not use sealed
`final-*` seeds for exploratory play.

The implementation has four small modules:

- `agent.py`: identity, fixed configuration, hooks, original HTTP server.
- `poker_eval.py`: structural preflop score, exact visible-board lookup scoring,
  FULL fragment enumeration, deterministic REDUCED sampler, danger/draw features.
- `poker_policy.py`: first-matching rules, call-price ceilings, legality/investment
  guards, identical-raise canonicalization, legal fallback.
- `poker_memory.py`: observed-opportunity counters, smoothed labels, history
  deduplication, pending responses, transactional terminal-result deduplication.

The deployed strategy is `v05_limp20`: after sufficient observed evidence,
frequent raisers take priority over the calling-station label; MEDIUM/STRONG hands
can defend ordinary pot prices; preflop hands scoring at least 20 defend against
confirmed aggression; calls and button limps avoid unnecessary reraises. UNKNOWN,
TIGHT, STATION and OVERFOLDER pricing remains unchanged. Pure bluffs stay disabled.
These are conditional heuristics, not equity estimates or a guarantee of future wins.

The exact evaluator remains `FULL_PAIR_v04`. Both evaluator modes are independently
testable, with no dynamic switching. `STRATEGY` is fixed before each half;
`v04` remains available as the frozen control for reproduction.
The experimental launcher under `research/` sets a mode before `new_half`; it is
excluded from the submission image. All mutable match state resets in `new_half`.
Diagnostics are disabled in the production server. The research server exposes
bounded diagnostics after matches to detect contained errors.

Reproduce correctness and calibration from the repository root:

```powershell
.venv/Scripts/python -m pytest -q --junitxml=reports/correctness.xml
git clone https://github.com/tamu-datathon-org/heads-up-bundle .work/upstream
.venv/Scripts/python research/check_sources.py
.venv/Scripts/python research/verify_design.py check-fixtures --path design/research/sampler_fixtures_v04.json
.venv/Scripts/python research/verify_design.py verify --repo .work/upstream --input design/research/pilot_raw_reconstructed_v04.csv --output reports/pilot_official.csv --slow 3
.venv/Scripts/python research/verify_design.py verify --repo .work/upstream --input design/research/fresh_stratified_inputs_v04.csv --output reports/fresh_official.csv --slow 3
.venv/Scripts/python research/verify_design.py generate --seed threat-cal-31 --per-stratum 20 --output reports/fresh_cal31_inputs.csv
.venv/Scripts/python research/verify_design.py verify --repo .work/upstream --input reports/fresh_cal31_inputs.csv --output reports/fresh_cal31_official.csv --slow 0
.venv/Scripts/python research/qualify_accuracy.py
```

The upstream checkout must be at `c83ab6eeb7660a2fd1fae7df224315d41fe985a9`.
`design/` is preserved byte-for-byte and its manifest is checked. The corrected
offline runner fixes supplied verifier defects; see [implementation notes](reports/IMPLEMENTATION.md).
Research never runs in the decision path. Fragment risk is **not** five-card
opponent equity; current-board strength is **not** final showdown equity.

Build and smoke-test using Docker Desktop and Git Bash (or Linux Bash):

```bash
./tools/check_build.sh
```

The official check builds both Dockerfiles, enforces the 3 GiB image increment,
and runs without network, with a read-only root, writable `/tmp`, one CPU,
4 GB RAM, and non-root UID 10001. `.dockerignore` excludes development files.

`research/run_matches.py` records complete official duplicate matches, per-request
latency percentiles, charged time, startup, errors, and source hashes. It requires
a new output directory and refuses to overwrite registries. Run inside the official
image with `research/` mounted read-only and output under `/tmp`. E0 uses
`stress-41`/`stress-83`; development uses `dev-11`/`dev-29`; final audit uses
`final-73`/`final-97` once after an approved freeze. A short debug run cannot pass E0.

The [current requirements matrix](reports/v05/REQUIREMENTS.md) maps this revision
to evidence; [v0.4 acceptance evidence](reports/REQUIREMENTS.md) remains archived. Raw CSVs, JSON registries, and JUnit output are in `reports/`.
Thresholds remain provisional hypotheses. The v0.5 comparisons and rejected
variants are archived under `reports/v05/`; original final seeds were not reused.
The initial v0.5 audit (19 wins, one chop) is retained under `reports/v05/`.
The subsequent single-axis refinement and its separately registered seeds,
selection and final freeze are under `reports/v05b/`. Both phases are included
in the current release report. No disclosed final seed was reused. The old `freeze_mode.py`,
`select_policy.py`, and `finalize_configuration.py` scripts reproduce the historical
v0.4 experiment; they intentionally refuse to overwrite existing frozen evidence.

For submission, complete mandatory gates, verify registration and team name,
push the desired `licheng` commit, and copy the full SHA with `git rev-parse HEAD`.
The participant guide's form is <https://forms.gle/Jx9RWE5VgR8dbbcj9>. Supply the
public repository URL, team name, and full 40-character SHA yourself; this bot
does not submit the form.
