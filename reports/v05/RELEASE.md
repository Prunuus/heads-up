# Current v0.5 release evidence

**Latest frozen audit: 20 wins, 0 chops, 0 losses in 20 full duplicate matches. Observed match win rate: 100%.**

Deployment: `v05_limp20`, FULL_PAIR_v04, adaptation on, pure bluffs off, salt 0. Branch `licheng`; participant/bot `3XKO`. Engineering gates: PASS.

This is an observed audit result, not a guarantee that every future match or every individual hand will be won. All matches use the official 1,000-deal duplicate format (2,000 hands), including its confidence-interval verdict. The tournament form was not submitted.

## What changed

- Prioritize demonstrated aggression over the calling-station label, using the original evidence thresholds. AlwaysPot was previously liable to become STATION merely because it never folds.
- Against confirmed aggressive opponents, let MEDIUM/STRONG hands call ordinary pot prices without the contradictory pot/repeated-raise deductions.
- Defend preflop scores >=20 against confirmed aggression; use calls and button limps to avoid unnecessary recursive reraising. Lower scores and nonaggressive price policies retain their original guards.
- Keep the exact evaluator, feature score, sampler, draws, nuts proofs, legal guards, protocol and per-half memory contracts unchanged. No opponent identity, hidden cards, match seed or baseline RNG state is used by tournament logic.

See [STRATEGY.md](STRATEGY.md) and [REQUIREMENTS.md](REQUIREMENTS.md).

## Honest experiment history

The original v0.4 bot lost to RandomBot and AlwaysPot. Phase A isolated price, preflop-defense, trapping and profile-priority changes on new development seeds. Its selected score-30 policy won all 16 development/confirmation HTTP matches, but its sealed audit was **19 wins, 1 chop, 0 losses**. The non-perfect audit is retained in [the phase-A registry](final/registry.json); it was not overwritten or rerun.
Phase A improved the matched original-control development score by +4 match points and +2.9271 mean paired BB/deal, with no opponent point regression.

Phase B was preregistered after that audit. It tested only the conditional preflop defense score, holding all other changes fixed, on new development/confirmation seeds. Because both policies were already winning all development matches, its promotion criterion required a BB improvement and no regression instead of an impossible increase beyond the match-point ceiling.
The selected score-20 revision won all 16 phase-B development/confirmation matches and added +0.4709 mean paired BB/deal versus score 30. Per-opponent changes were baseline_random: +0.4327, baseline_calling_station: +0.0000, baseline_always_half: +0.9223, baseline_always_pot: +0.5287 BB/deal. These are development screening results, with per-match delta confidence intervals/variance disclosed in the selection files, not proof of population-wide superiority.

Only after promotion and a new source/configuration freeze did the final runner open five new audit seeds. No phase-A final seeds were reused, and no parameter was changed after reading the new final outcomes.

Evidence: [phase-A plan](experiment_plan.json), [phase-A selection](selection.json), [phase-B plan](../v05b/experiment_plan.json), [phase-B selection](../v05b/selection.json), [new final freeze](../v05b/final_freeze.json).

One rejected finalist (`v05_aggressive30`) had a 102.53 ms request during parallel development confirmation, exceeding the 100 ms acceptance margin despite zero referee errors and all match wins. That failed guard remains in its raw registry and was not hidden. The deployed policy passed its own complete checks.

## Latest untouched final audit

| Opponent | Wins / matches | Mean paired BB/deal across five matches | Lowest per-match CI bound |
|---|---:|---:|---:|
| baseline_random | 5 / 5 | 3.3963 | 1.6753 |
| baseline_calling_station | 5 / 5 | 3.4503 | 2.8138 |
| baseline_always_half | 5 / 5 | 4.3423 | 2.7173 |
| baseline_always_pot | 5 / 5 | 4.1355 | 1.6674 |

Every row below is one independent 1,000-deal duplicate match.

| Opponent | Seed | BB/deal | 95% paired CI | Verdict | Charged s, halves 1 / 2 | Max POST ms, halves 1 / 2 |
|---|---|---:|---|---|---|---|
| baseline_random | final-v05b-503 | 4.1300 | [3.0224, 5.2376] | AGENT1_WIN | 0.534 / 0.333 | 17.06 / 9.44 |
| baseline_random | final-v05b-509 | 3.0710 | [1.9934, 4.1486] | AGENT1_WIN | 0.393 / 0.490 | 6.50 / 10.09 |
| baseline_random | final-v05b-521 | 2.7415 | [1.6753, 3.8077] | AGENT1_WIN | 0.613 / 0.517 | 11.49 / 15.69 |
| baseline_random | final-v05b-541 | 3.6325 | [2.5380, 4.7270] | AGENT1_WIN | 0.313 / 0.238 | 6.91 / 4.64 |
| baseline_random | final-v05b-547 | 3.4065 | [2.2960, 4.5170] | AGENT1_WIN | 0.311 / 0.473 | 6.21 / 9.22 |
| baseline_calling_station | final-v05b-503 | 3.5130 | [3.0939, 3.9321] | AGENT1_WIN | 0.787 / 0.817 | 8.68 / 10.26 |
| baseline_calling_station | final-v05b-509 | 3.2070 | [2.8138, 3.6002] | AGENT1_WIN | 0.944 / 0.856 | 10.40 / 12.36 |
| baseline_calling_station | final-v05b-521 | 3.6710 | [3.2498, 4.0922] | AGENT1_WIN | 0.990 / 0.912 | 9.28 / 10.39 |
| baseline_calling_station | final-v05b-541 | 3.5365 | [3.1325, 3.9405] | AGENT1_WIN | 0.836 / 1.103 | 4.95 / 32.54 |
| baseline_calling_station | final-v05b-547 | 3.3240 | [2.9065, 3.7415] | AGENT1_WIN | 1.136 / 0.761 | 11.41 / 26.18 |
| baseline_always_half | final-v05b-503 | 4.4340 | [3.2607, 5.6073] | AGENT1_WIN | 0.824 / 0.704 | 9.90 / 16.69 |
| baseline_always_half | final-v05b-509 | 3.9420 | [2.7572, 5.1268] | AGENT1_WIN | 0.477 / 0.512 | 8.60 / 13.55 |
| baseline_always_half | final-v05b-521 | 3.9150 | [2.7173, 5.1127] | AGENT1_WIN | 0.529 / 0.724 | 8.27 / 11.82 |
| baseline_always_half | final-v05b-541 | 4.4015 | [3.2107, 5.5923] | AGENT1_WIN | 0.471 / 0.614 | 8.04 / 11.71 |
| baseline_always_half | final-v05b-547 | 5.0190 | [3.8628, 6.1752] | AGENT1_WIN | 0.738 / 0.537 | 23.59 / 11.75 |
| baseline_always_pot | final-v05b-503 | 4.3630 | [2.6972, 6.0288] | AGENT1_WIN | 0.374 / 0.352 | 14.02 / 8.35 |
| baseline_always_pot | final-v05b-509 | 3.2970 | [1.6674, 4.9266] | AGENT1_WIN | 0.491 / 0.479 | 29.53 / 11.41 |
| baseline_always_pot | final-v05b-521 | 3.6040 | [1.9474, 5.2606] | AGENT1_WIN | 0.561 / 0.672 | 18.90 / 22.18 |
| baseline_always_pot | final-v05b-541 | 4.5555 | [2.9367, 6.1743] | AGENT1_WIN | 0.583 / 0.333 | 41.21 / 6.50 |
| baseline_always_pot | final-v05b-547 | 4.8580 | [3.2559, 6.4601] | AGENT1_WIN | 0.413 / 0.783 | 12.03 / 13.16 |

Raw final results: [registry](../v05b/final/registry.json). All development, stress and both audits are mapped to [all_http_results.csv](all_http_results.csv), including per-half errors, forfeits, time-bank status, startup, POST percentiles and source/configuration hashes.

## Correctness and runtime

- **337 tests pass on Python 3.12 and 3.13**, including the original 305 tests and new aggression, price-boundary, conditional trapping/limping and defense-floor cases.
- Three revised configurations each undergo >=10,000 Game-driven decision checks, exercising every street, aggressive profiles and raise caps. Legacy scoring/fragment/draw/memory tests remain intact; the old policy is an explicit reproducible control.
- Independent direct-hook screens and official HTTP results match all 24 compared training matches / 24,000 paired outcomes exactly. Direct screens are clearly labelled as screening, not runtime qualification.
- The unchanged official Docker build and restricted smoke test pass. No runtime dependencies or referee-controlled files were added/modified.
- The deployed strategy completed 48 full HTTP matches / 96,000 hands across development, confirmation, serialized stress and final audit: 0 referee errors, 0 internal errors, no illegal-action reports, no forfeits and no time-bank exhaustion.
- Across those runs: maximum charged time 2.390 s/half (acceptance <=5.5; official bank 8), maximum complete POST 41.21 ms (acceptance <=100; official timeout 1,000), maximum startup 2.56 s (acceptance <=12; official limit 15).
- Qualification containers enforce one CPU, 4 GiB RAM, non-root UID, read-only root, writable `/tmp`, no external network, dropped capabilities and PID limit 256. Resource/image snapshots are retained beside the phase-B report.

Evidence: [Python 3.12 tests](../v05b/final_tests_python312.xml), [Python 3.13 tests](../v05b/final_tests_python313.xml), [official build](../v05b/official_build.log), [source verification](../v05b/source_verification.json), [runtime stress](../v05b/stress/registry.json), [screen/HTTP cross-check](screen_validation.json).

## Limits and not-run items

- A finite winning audit cannot establish a 100% future win probability. Different opponents, action-conditioned ranges and card variance can produce losses or chops.
- The price ceilings remain heuristic risk policies. Fragment risk is not five-card equity, and current strength is not guaranteed final showdown strength.
- Python 3.10: NOT RUN because it is not installed; available Python 3.12/3.13 were tested.
- Custom tight-value/overfolder opponents and real tournament field: NOT RUN. Pure bluffs remain disabled. The measured claim covers the four official baselines.
- Organizer-hosted build/tournament/form: NOT RUN; requires registration and submission. No form was submitted.
- This revision was made and published on `licheng`, preserving the registered identity and public-repository workflow. Use the final response's full pushed commit SHA when submitting.
