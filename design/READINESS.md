# READINESS.md — Elephant v0.4

**Task boundary:** No tournament bot implementation has been performed. The only executable artifact in this package is an **offline research/reproducibility utility**, not imported by any submission or referee process.

## Design handoff checklist

| Gate | Current evidence | Status |
|---|---|---|
| One canonical deterministic design | `DESIGN.md` v0.4, first-match and alias precedence spelled out | **Specified** |
| Rule/evaluator contracts | Official game/evaluator references and pinned commit specified | **Specified; tests pending** |
| FULL/REDUCED algorithm contracts | Two versioned modes and byte-level deterministic sampling | **Specified** |
| Terminal memory atomicity | `processed_result` bitset and transaction order in DESIGN | **Specified; tests pending** |
| Sampler fixture reproducibility | Six fixture records compared to research spec locally | **PASS (research utility only)** |
| Legacy raw input reconstruction | All 240 row hero scores and beating pair counts match v0.3 independent scorer | **PASS (legacy scorer only)** |
| Official-table validation of all 240 states | No official repository checkout in local calculation environment; run `verify` | **NOT EXECUTED** |
| Exact graph versus independent slow reference | Planned at least 1 state each street | **NOT EXECUTED** |
| E0 runtime for BOTH modes | Restricted, full-referee, charge entire requests | **NOT EXECUTED** |
| E1 accuracy, coverage, tiers, action-difference | Official scorer + fresh strata + confidence bounds | **NOT EXECUTED** |
| Mode frozen after E0/E1 | No implemented mode to freeze | **NOT SELECTED** |
| Goldfish implementation and P0 protocol tests | Explicitly prohibited by current user request | **NOT STARTED** |
| Strategy ablations and final independent holdout | E2–E4 and final match audit defined | **NOT EXECUTED** |
| Tournament release | Needs official build, time, correctness, calibration, performance gates | **BLOCKED** |

## First actions *if separately authorized in a future turn*

1. Validate official repository commit and benchmark both modes under official judge via E0 (no mode freeze yet).
2. Re-run `research/calibrate_offline.py verify` against pinned official evaluator; review all discrepancies and checksum output.
3. Run new E1 official calibration, stratum coverage, graph/direct checks, FULL-vs-REDUCED comparison and salt sensitivity.
4. Freeze preferred eligible mode only after **both** gates pass. If neither passes, block handoff and request an explicit new design decision.
5. Only after a distinct implementation authorization, produce the Goldfish bot, test reachable states, run development comparisons, and perform one untouched final evaluation.

**Never claim the poker bot has been tested or is ready for submission based only on this documentation.**
