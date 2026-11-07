# v0.5 verification matrix

The original [v0.4 acceptance matrix](../REQUIREMENTS.md) is historical. The user
authorized the strategic revisions in [STRATEGY.md](STRATEGY.md); exact old policy
outputs are retained under the explicit `v04` control instead of being represented
as v0.5 requirements. Shared poker/protocol/reliability contracts still apply.

| Requirement | Evidence |
|---|---|
| Exact official rules, evaluator and protocol retained | `source_verification.json`; no changes to game/evaluator/judge or server plumbing |
| Hero 2+3 allocation, categories, wheel, kickers and ties | Original correctness tests rerun on current source |
| 10,000 distinct river scores and 1,000 independent fragment references | Full current test suite, Python 3.12 and 3.13 JUnit reports |
| Exact preflop distribution over 376,992 holdings | Exhaustive score-distribution test; scorer unchanged |
| Full pair graph, draws, nuts and sampler contracts | Shared evaluator unchanged; original research cross-validation remains applicable |
| Legacy policy reproducible | All original policy fixtures remain under `v04`; control screened and HTTP-tested on the new development seeds |
| Overfolding regression fixed for confirmed aggression | New price-boundary tests: MEDIUM/STRONG can call 1/3-priced bets, including four-raise contexts |
| Weak hands still fold; ordinary opponents retain price rules | New WEAK and UNKNOWN/TIGHT/STATION/OVERFOLDER regression cases |
| Preflop widening has an explicit boundary | Tests immediately below/at scores 20/25/30/35/40/45; default UNKNOWN remains unchanged |
| Aggressor is not mistaken for a passive station | High-raise/nonfold profile fixture; CallingStation control fixture |
| Evidence thresholds and per-half reset preserved | Original smoothing/count/reset tests; current protocol tests and fresh server per measured half |
| Trapping and button limping require known aggression | New conditional preflop fixtures; no opponent-ID access |
| Every returned action legal; no uncaught internal failures | Full official HTTP matches plus three >=10,000-decision Game-driven v0.5 workloads |
| All streets, raise cap and aggressive profile exercised | `reachability_v05_aggressive30.json`, `reachability_v05_aggressive_limp.json`, `reachability_v05_limp20.json` |
| Duplicate/history/terminal atomicity unchanged | Original memory suite rerun; internal-error diagnostics in every HTTP half |
| No new runtime dependencies, no online simulation or hidden information | requirements/evaluator unchanged; production policy reads only Observation and memory |
| Direct screens independently match HTTP behavior | `screen_validation.json`: 24 matches, 24,000 paired outcomes exactly equal |
| Controlled development promotion | `selection.json` and `../v05b/selection.json`: matched train seeds, paired-delta CIs/variances, points and opponent regressions |
| New seeds separated by purpose | `experiment_plan.json`: development, confirmation, stress and final sets registered before use |
| Failed candidate runs disclosed | `http_confirm_v05_aggressive30/registry.json` retains its 102.53 ms request-margin failure; candidate was not deployed |
| Frozen source/configuration before final audit | `../v05b/final_freeze.json`; final runner verifies every runtime source hash |
| One CPU, 4 GB, read-only root, no network, non-root, bounded PIDs | `container_configuration.txt`, `memory_peak_bytes.txt`, and Docker run commands |
| Full official build and restricted smoke test | `../v05b/official_build.log` |
| Full serial runtime stress, complete HTTP charging | `../v05b/stress/registry.json`, per-half timing/error/latency fields |
| One-time untouched final audit, all outcomes retained | Both original `final/registry.json` and new `../v05b/final/registry.json`; each frozen policy has a separate five-seed audit |
| No universal-win claim | Report distinguishes observed audit wins from a guarantee about future matches |

See [RELEASE.md](RELEASE.md) for actual counts, pass/fail status and final results.
Python 3.10 is NOT RUN because it is not installed; available 3.12/3.13 runtimes
are tested. Organizer-hosted builds/tournament and form submission are NOT RUN.
No competition form is submitted by this work.
