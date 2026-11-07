# Acceptance requirements matrix

Status refers to executed evidence, not the historical package's readiness text.
`PASS` means the specified check passed locally; organizer-hosted builds and actual
tournament results remain outside this execution. The selected policy is FULL,
adaptation on, pure bluffs off. Exact numerical outcomes are in [RELEASE.md](RELEASE.md).

| Requirement | Status | Evidence |
|---|---|---|
| R01 deck, ranks, suits, distinct cards | PASS | Official source verification; legal-deal and physical-card validation tests |
| R02 all categories, flush/full-house, trips/straight, short-deck wheel | PASS | `test_categories_wheel_kickers_and_allocation` |
| R03 exactly 10/40/100 hero probes | PASS | `test_probe_counts` on all streets |
| R04 10,000 distinct full-score river comparisons | PASS | `test_ten_thousand_distinct_rivers` |
| R05 wheel, kickers, ties, correct allocation, false flushes | PASS | Category/allocation fixtures and 1,000 independent pair references including tie counts |
| R06 incomplete-board lookup scoring and malformed fallback | PASS | Hero table implementation, probe tests, 100 malformed observations |
| R07 official betting actions, min raise, cap, all-ins, opener/BB | PASS | Official Game reachability, boundary tests, aggressive protocol tests |
| R08 dynamic blind/stack/config data | PASS | `test_configuration_driven_blinds_and_half_reset`, dynamic config memory tests |
| T01 full fragment counts and caching | PASS | Probe counts 378/1,404/3,250; per-hand cache; E0 cache hit/miss diagnostics |
| T02 independent full-score/B/E/N comparison on >=1,000 boards | PASS | `test_one_thousand_full_pair_references` using separate `score_five` combinations |
| T03 kings full 13/325 and 17,156/65,780 | PASS | Regression/graph test; [sampler regression](kings_sampler.json) |
| T04 full-score ties and unbeaten versus untied distinction | PASS | Independent reference E counts, calibration tie-only counts; sizing predicates |
| T05 exact deterministic sampler, all fixtures, available Python versions | PASS | Six fixtures on Python 3.12 and 3.13; Python 3.10 unavailable |
| T06 current/vulnerable/locked nuts, sampled-zero prohibition | PASS | `test_current_locked_and_reduced_zero`; exact locked proof and mode-specific tiers |
| T07 vulnerable nuts action precedence | PASS | `test_vulnerable_nuts_never_automatic_pot_or_reraise` |
| T08 fresh street/tier/texture/boundary coverage | PASS | [accuracy.json](accuracy.json): 480 states, all minimum strata satisfied |
| T09 exact five-card risk and severe-label Wilson intervals | PASS | 0/102 FULL, 0/100 REDUCED severe STRONG mislabels; Wilson bounds in accuracy report |
| T10 graph versus independent direct official-table reference | PASS | Three streets for both historical and fresh sets; all selected exact counts agree |
| T11 FULL/REDUCED upward/downward tiers and coverage | PASS | 5.00% upward (Wilson upper 7.33%); 13.96% downward; >=40 each reduced tier |
| T12 reachable action impact and salts 0/1/2/3 | PASS | [action_comparison.csv](action_comparison.csv): 1,004 postflop contexts, 4.38% salt-0 divergence; diagnostic salt outputs |
| T13 historical pilot reconstruction and official verification | PASS | All 240 official score/pair/five-card/tie checks; output SHA recorded in mode freeze |
| C-DRAW 1 visible-card denominators | PASS | Exact formulas use U=28/27; runout reference |
| C-DRAW 2 flush regression values | PASS | `test_flush_examples` (three parameterized fixtures) |
| C-DRAW 3 same-suit hole blockers and exactly two used | PASS | Physical-card route enumeration and allocation tests |
| C-DRAW 4 six straight sequences, one/two missing ranks, rank blockers | PASS | Explicit six-sequence implementation; 100-state independent route/runout enumeration |
| C-DRAW 5 maximum individual route and made-category suppression | PASS | Route-max assessment; category suppression tests |
| C-DRAW 6 >=100 physically consistent runout comparisons at 1e-12 | PASS | `test_one_hundred_draw_states_by_runout_enumeration` |
| C-POT 1 half-pot .25 and pot 1/3 | PASS | Real Game observations on every postflop street |
| C-POT 2 exact actual call price and stack cap | PASS | `test_reachable_half_and_pot_prices`, partial-all-in price fixture |
| C-POT 3 >=10,000 reachable decisions and coverage | PASS | [reachability.json](reachability.json): 10,006; both seats/all streets/actions; 437 cap states |
| C-POT 4 >=100 inconsistent observations and protocol failures | PASS | 100 malformed dataclass cases, mapping matrices, empty legal-set protocol test |
| C-POT 5 identical raise aliases canonicalize to HALF | PASS | A05 matrices across streets, both evaluator modes and locked branches |
| C-POT 6 all call ceilings, first-match order, actual investment caps | PASS | Parameterized pre/postflop boundaries and action matrices |
| C-POT 7 rejected call versus failed raise/check fallback | PASS | Boundary and unavailable-raise tests; no free-check folding |
| C-ACTION A01 both true -> POT | PASS | STRONG/LOCKED matrices |
| C-ACTION A02 pot only -> POT | PASS | Explicit synthetic legal-set intersection on Game context |
| C-ACTION A03 half only -> HALF | PASS | STRONG/LOCKED matrices |
| C-ACTION A04 neither -> CHECK/CALL | PASS | STRONG/LOCKED matrices |
| C-ACTION A05 same effective amount -> HALF | PASS | Explicit synthetic alias overlays |
| C-ACTION A06 pot illegal/over cap -> legal half | PASS | Legality/cap overlays; ordered independent branches |
| C-ACTION A07 malformed raise mappings -> CHECK/CALL | PASS | Malformed mapping overlays |
| C-MEM new-half initialization, first/last hands, no cross-half leakage | PASS | Official tests, dynamic config/reset tests, all complete HTTP half results |
| C-MEM repeated/shrinking/backwards histories | PASS | History idempotence/shrink tests; safe fallback and no replay |
| C-MEM immediate reply resolution, called-then-folded, terminal folds | PASS | Pending-response unit tests; direct fold and all-in showdown cases |
| C-MEM unobserved button folds and observed-only denominators | PASS | Unseen fold test; profile denominator tests; per-half coverage in CSV |
| C-MEM free checks and raise aliases | PASS | Observed-action logic, post-nonfold counters, one pending effective response |
| M01 identical terminal duplicates | PASS | Deep equality after duplicate delivery |
| M02 conflicting duplicate diagnosed once | PASS | Duplicate transaction tests |
| M03 malformed/out-of-range results leave state unchanged | PASS | Ten malformed terminal variants; scratch-delta commit |
| M04 batch (5,3,5,4) exactly once | PASS | `test_atomic_duplicates_out_of_order_and_reset` |
| M05 processed bits reset every half | PASS | Reset tests and fresh server per measured half |
| M06 contained hooks and unusable required protocol inputs distinguished | PASS | Invalid config flag, fallback tests, protocol rejection tests |
| M07 diagnostics bounded 32 x 160 | PASS | `test_diagnostics_and_invalid_config` |
| M08 terminal errors cannot partially corrupt counters/pending/bitset | PASS | Malformed transaction tests; no internal errors in measured matches |
| C-MODE full E0 suite for BOTH modes before selection | PASS | [E0 registry](E0/registry.json): 12 duplicate matches / 24 halves per mode |
| C-MODE charged <=5.5s, POST <=100ms, startup <=12s | PASS | Entire official-referee HTTP requests, percentiles and per-half timings in [CSV](match_results.csv) |
| C-MODE no errors, illegal moves, forfeits, exhaustion | PASS | All 56 full matches and internal diagnostics |
| C-MODE resource/read-only/network/image-size/build/smoke | PASS | [final build](final_build.log), [container limits](E0_container.json), cgroup peak memory evidence |
| C-SCORE complete preflop structural distribution | PASS | Exhaustive 376,992-holding test: published suit counts, min/max/mean/tail/opening frequencies |
| C-STRATEGY >=60 deterministic branch edge assertions | PASS | 159 parameterized policy cases with matching/failing/unavailable and boundary assertions |
| C-STRATEGY frozen no-adaptation/no-pure-bluff baseline | PASS | [development A](dev_A/registry.json) |
| C-STRATEGY controlled promotion on matched dev seeds | PASS | B versus A +0.5 points/+0.3494375 BB/deal; no opponent point regression; C not promoted |
| C-STRATEGY paired-difference uncertainty/variance | PASS | [policy selection](policy_selection.json), per-match delta CIs and variances |
| C-STRATEGY final audit once after freeze, no retuning | PASS | [final freeze](final_freeze.json), [final registry](final/registry.json); losses disclosed |
| C-FREEZE 1 E0 alone does not select mode | PASS | Separate completed E0 registry before mode freeze |
| C-FREEZE 2 E1 accuracy states | PASS | Calibration diagnostics and [ordered freeze](freeze.json) |
| C-FREEZE 3 FULL preferred only after both gates pass | PASS | Both qualify; FULL selected |
| C-FREEZE 4 source/config/salt recorded before strategy experiments | PASS | Mode freeze precedes development; final source/config freeze precedes holdout |
| C-FREEZE 5 no later evaluator/sampler/salt changes | PASS | Core module hashes match across E0, development and final; identity-only edit documented |

Additional evidence is under [IMPLEMENTATION.md](IMPLEMENTATION.md). The original
research runner's defects were corrected in a separate utility while preserving
the supplied package and manifest. P0/P1 qualification is local; it does not
predict tournament EV. Optional E2/E3 threshold alternatives, custom tight-value/
overfolder opponents, Python 3.10 and organizer-hosted execution are explicitly
NOT RUN with reasons in the release report. No current failed test remains.
