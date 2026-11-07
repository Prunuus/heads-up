# Release evidence

**Engineering qualification: PASS. Competitive performance: mixed; this is an experimental submission candidate, not a demonstrated strong tournament bot.**

Identity: `3XKO / 3XKO`. Branch: `licheng`. Mode: `FULL_PAIR_v04`; salt 0; adaptation enabled; pure bluffs disabled (development policy B).

The participant guide requires team registration and a public repository/full pushed commit SHA. The repository is public; registration was not verified and no competition form was submitted.

Completed **56 full duplicate matches / 112,000 hands** across E0, development, and the one-time final audit. Candidate errors: **0**; illegal-action reports, internal hook errors, forfeits, and time-bank exhaustion: **0**. No strategic parameter was changed after viewing final results.

## Correctness and official build

- 305 tests passed on Windows Python 3.13 and on Linux Python 3.12 in an offline, read-only, one-CPU, 4-GB test container. Final frozen flags also passed all 305 Windows tests.
- Includes 10,000 distinct complete river comparisons, 1,000 independent fragment references, all 376,992 preflop holdings, 100 runout-enumerated draw states, 100 malformed observations, and deterministic action/memory matrices.
- The official Game-driven generator checked 10,006 decisions: 4,266 preflop, 3,485 flop, 1,657 turn, 598 river; 437 at the raise cap. Both seats and all four actions were covered.
- Unchanged official `tools/check_build.sh` passed for both the initial and final configuration, including the restricted smoke test. Final smoke startup was approximately 0.8 seconds.
- Official game/evaluator/judge and server plumbing are unchanged; the supplied design manifest passes. Source commit: `c83ab6eeb7660a2fd1fae7df224315d41fe985a9`.

Evidence: [final tests](final_correctness.xml), [Python 3.12 tests](correctness_python312.xml), [final build](final_build.log), [source verification](final_source_verification.json), [coverage](reachability.json).

## Evaluator qualification

E0 ran both modes independently: six opponents (four official baselines plus behaviorally identical call-heavy/max-raise aliases), two stress seeds, both 1,000-hand halves, fresh HTTP servers per half. Both passed before mode selection.

| Mode | Matches / halves | Max charged s/half | Max request ms | Max startup s | E0 | E1 |
|---|---:|---:|---:|---:|---|---|
| FULL_PAIR_v04 | 12 / 24 | 0.778 | 26.92 | 1.76 | PASS | PASS |
| REDUCED_64_v04 | 12 / 24 | 0.263 | 47.21 | 1.76 | PASS | PASS |

Resource controls: one CPU quota, 4 GiB RAM, no external network, non-root UID 10001, read-only root, writable `/tmp`, dropped capabilities, and PID limit 256. Captured cgroup peak during E0 was 167,211,008 bytes (about 159.5 MiB) for judge and both bots together. The final official size check passed; the initial bot image increment was only 50,937 bytes.

Across all E0/development/final runs, maximum charged time was 2.570 s/half and maximum measured complete POST latency was 53.83 ms, below the 5.5 s / 100 ms acceptance margins and 8 s / 1 s official limits. Per-half percentiles, startup, charged time and errors are in [match_results.csv](match_results.csv).

E1 used 480 fresh states (160 per street), including 102 FULL STRONG, 143 MEDIUM, 219 WEAK, and 16 NUTS. Coverage included 273 paired boards, 120 trips boards, 140 suited boards, 60 states near .03 and 37 near .12.

- FULL severe STRONG mislabels: 0/102; Wilson 95% upper 3.63%. Mean exact current-five-card stronger risk in STRONG was 9.81%, versus 42.79% MEDIUM and 76.64% WEAK.
- REDUCED severe STRONG mislabels: 0/100; Wilson upper 3.70%. Upward tier disagreement: 24/480 = 5.00% (Wilson upper 7.33%); downward: 67/480 = 13.96%.
- Action disagreement: 44/1,004 = 4.38% on reachable postflop contexts with fixed nonmode inputs. Salts 1/2/3 disagreed in 46/50/41 contexts respectively. Salts were diagnostic only.
- All six sampler fixtures pass on Python 3.12 and 3.13. Python 3.10 was NOT RUN because it was not installed; the acceptance requirement permits available runtimes.
- All 240 reconstructed historical states pass official packed-score, pair-count, five-card-beat and tie-only checks. Independent slow direct enumeration agrees on one state per street for both historical and fresh sets.
- Kings-full regression: exactly 13/325 beating fragments and 17,156/65,780 beating complete holdings. Salt 0 is MEDIUM; diagnostic salt 1 labels it STRONG, illustrating sampling risk.

Fragment frequency, uniform current-five-card threat and final showdown equity are distinct. Calibration does not establish betting EV against real ranges. See [accuracy](accuracy.json), [action comparisons](action_comparison.csv), [salt diagnostics](salt_diagnostics.csv), [historical verification](pilot_official.csv), [fresh verification](fresh_official.csv), and [ordered freeze](freeze.json).

## Development selection

| Variant | Adaptation | Pure bluffs | Points / 8 | Mean paired BB/deal across match means |
|---|---|---|---:|---:|
| A | off | off | 3.0 | -0.1825 |
| B | on | off | 3.5 | 0.1669 |
| C | on | on | 3.5 | 0.1669 |

B met the prespecified promotion screen: +0.5 match points, +0.3494375 mean paired BB/deal versus A, and no opponent point regression. C was identical to B and was not promoted. All numeric preflop, tier, call and sizing thresholds remain v0.4 values. No E2/E3 threshold alternatives were proposed or run.

Per-match candidate-minus-baseline means, variances and confidence intervals are in [policy_selection.json](policy_selection.json). The mean improvement averages these four selected opponents; it is not evidence about the unknown tournament field. Early 100 and final 500 paired-outcome diagnostics are included in the CSV; small-window results are not match verdicts.

## One-time final audit

Frozen before access to final seeds: [final_freeze.json](final_freeze.json). Final score: **4.0/8 points**, 4 wins, 0 chops, 4 losses.

| Opponent | Seed | Paired BB/deal | 95% CI | Verdict | Charged s, halves 1 / 2 | Max POST ms, halves 1 / 2 |
|---|---|---:|---|---|---|---|
| baseline_random | final-73 | -1.8205 | [-2.2254, -1.4156] | LOSS | 0.090 / 0.115 | 7.55 / 7.12 |
| baseline_random | final-97 | -1.0980 | [-1.5466, -0.6494] | LOSS | 0.201 / 0.528 | 19.64 / 9.60 |
| baseline_calling_station | final-73 | 3.3010 | [2.8963, 3.7057] | WIN | 1.190 / 1.043 | 10.22 / 15.16 |
| baseline_calling_station | final-97 | 3.8110 | [3.3770, 4.2450] | WIN | 0.543 / 0.659 | 6.38 / 7.27 |
| baseline_always_half | final-73 | 1.1430 | [0.0084, 2.2776] | WIN | 0.267 / 0.175 | 7.12 / 7.88 |
| baseline_always_half | final-97 | 1.8530 | [0.6622, 3.0438] | WIN | 0.168 / 0.297 | 6.59 / 17.66 |
| baseline_always_pot | final-73 | -2.6260 | [-2.6958, -2.5562] | LOSS | 0.041 / 0.050 | 6.10 / 8.01 |
| baseline_always_pot | final-97 | -2.6090 | [-2.6823, -2.5357] | LOSS | 0.032 / 0.036 | 6.66 / 7.20 |

**The bot loses to RandomBot and AlwaysPot on the sealed audit.** Its conservative call ceilings can fold strong current hands to pot-sized aggression. This is a measured strategic limitation, not an engineering failure. The audit was not used to change the selected policy or thresholds. No claim of overall superiority, GTO play or tournament-winning strength is supported.

## Not run and remaining submission actions

- Optional alternative opening/call/sizing threshold experiments (E2/E3): NOT RUN; coefficients were kept frozen and the controlled E4 variants supplied the selection evidence.
- Optional tight-value and overfolder archetypes: NOT RUN; the required official suite and timing stress opponents were completed. Pure-bluff exploitation of a deliberate overfolder therefore remains unvalidated and is disabled.
- Python 3.10 fixture reproduction: NOT RUN; runtime not installed. Python 3.12 and 3.13 passed.
- Remote organizer build and actual tournament: NOT RUN; require organizer submission. Registration status was not checked. No submission form was sent.
- Use the full pushed commit SHA from the final response, verify the registered team name `3XKO`, and submit the public repository URL plus SHA if you choose to enter.

The complete implementation, reproducible evidence and requirement mapping are committed on `licheng`. See [REQUIREMENTS.md](REQUIREMENTS.md) and [IMPLEMENTATION.md](IMPLEMENTATION.md).
