# Zoo diversity report

Round robin of 19 zoo bots, seeds ['s1'], 300 deals per half (both halves per match), wall time 151 s.

## Standings

| Bot | Points | mean bb/deal | Losses | Forfeit halves | Max bank (s) |
|---|---:|---:|---:|---:|---:|
| llm_ml_minded | 17.0 | +3.582 | 0 | 0 | 2.742 |
| llm_opp_model | 15.5 | +3.170 | 1 | 0 | 4.370 |
| llm_mc_equity | 15.0 | +3.132 | 1 | 0 | 2.543 |
| llm_gto_inspired | 15.0 | +2.134 | 0 | 0 | 1.211 |
| llm_time_careful | 14.5 | +1.782 | 0 | 0 | 1.480 |
| llm_beginner_2h | 13.0 | +2.710 | 3 | 0 | 3.618 |
| llm_simple_rules | 11.5 | +2.521 | 3 | 0 | 3.912 |
| mc_equity_fast | 11.0 | +2.136 | 5 | 0 | 5.309 |
| checkfold_after_300 | 10.5 | +2.120 | 6 | 0 | 5.461 |
| llm_aggressive | 8.0 | +0.493 | 8 | 0 | 1.630 |
| threshold_cat | 7.0 | -0.184 | 9 | 0 | 0.042 |
| mc_equity_slow | 5.5 | -0.246 | 10 | 0 | 8.033 |
| guide_example | 5.0 | -0.935 | 11 | 0 | 0.269 |
| always_pot | 5.0 | -2.547 | 10 | 0 | 0.000 |
| always_half | 5.0 | -4.996 | 11 | 0 | 0.000 |
| nit | 4.0 | -0.965 | 11 | 0 | 0.033 |
| maniac | 4.0 | -5.399 | 12 | 0 | 0.001 |
| random | 2.5 | -4.521 | 14 | 0 | 0.003 |
| calling_station | 2.0 | -3.987 | 14 | 0 | 0.000 |

## Behaviour

Rates pool every decision the bot played across all its halves. Fold-facing = folds / decisions with to_call > 0; Pre fold = folds / preflop decisions.

| Bot | Fold % | Call % | Half % | Pot % | Fold-facing % | Pre fold % | Errors | Bank-exhausted halves |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llm_ml_minded | 8.0 | 63.2 | 17.4 | 11.4 | 18.4 | 2.7 | 0 | 0/36 |
| llm_opp_model | 8.0 | 74.6 | 8.5 | 8.9 | 18.5 | 4.1 | 0 | 0/36 |
| llm_mc_equity | 7.3 | 65.9 | 12.4 | 14.3 | 16.9 | 4.9 | 0 | 0/36 |
| llm_gto_inspired | 9.1 | 61.5 | 15.6 | 13.8 | 21.5 | 12.7 | 0 | 0/36 |
| llm_time_careful | 19.0 | 52.5 | 18.2 | 10.3 | 44.4 | 31.6 | 0 | 0/36 |
| llm_beginner_2h | 6.2 | 70.8 | 11.5 | 11.5 | 14.4 | 3.4 | 0 | 0/36 |
| llm_simple_rules | 5.7 | 76.5 | 8.2 | 9.6 | 13.0 | 1.7 | 0 | 0/36 |
| mc_equity_fast | 4.4 | 81.5 | 5.9 | 8.2 | 10.1 | 0.4 | 0 | 0/36 |
| checkfold_after_300 | 4.5 | 81.6 | 5.7 | 8.2 | 10.3 | 0.4 | 0 | 0/36 |
| llm_aggressive | 2.6 | 43.7 | 36.1 | 17.6 | 5.8 | 0.4 | 0 | 0/36 |
| threshold_cat | 8.4 | 88.9 | 0.0 | 2.7 | 19.9 | 0.0 | 0 | 0/36 |
| mc_equity_slow | 27.4 | 67.3 | 2.0 | 3.3 | 64.6 | 50.3 | 0 | 36/36 |
| guide_example | 12.9 | 85.6 | 0.0 | 1.5 | 32.1 | 11.2 | 0 | 0/36 |
| always_pot | 0.0 | 7.8 | 0.0 | 92.2 | 0.0 | 0.0 | 0 | 0/36 |
| always_half | 0.0 | 7.6 | 92.4 | 0.0 | 0.0 | 0.0 | 0 | 0/36 |
| nit | 24.3 | 71.0 | 0.0 | 4.7 | 58.6 | 52.6 | 0 | 0/36 |
| maniac | 0.0 | 36.6 | 30.4 | 33.0 | 0.0 | 0.0 | 0 | 0/36 |
| random | 13.5 | 30.1 | 28.1 | 28.3 | 30.2 | 20.8 | 0 | 0/36 |
| calling_station | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0/36 |

## Hypotheses H1-H6

Counts of zoo bots (out of 19) showing each behaviour, with the stat used.

- **H1, exhausts its time bank**: 1/19: mc_equity_slow. Stat: a half where the bank hit zero (`bank_exhausted` / timed_out flag). Max bank use per bot is in Standings.
- **H2, rarely folds facing bets**: 5/19: llm_aggressive, always_pot, always_half, maniac, calling_station. Stat: fold-facing rate < 10%.
- **H3, category thresholds**: n/a from stats. From archetype code: `threshold_cat` (hand-category thresholds); the llm_* bots were not audited for this.
- **H4, folds a lot preflop**: 2/19: mc_equity_slow, nit. Stat: preflop fold rate > 40%.
- **H5, never uses RAISE_HALF or never uses RAISE_POT**: 6/19: threshold_cat, guide_example, always_pot, always_half, nit, calling_station. Stat: zero count of either action over all its halves.
- **H6, errors or forfeits**: 0/19: none. Stat: error count > 0 or forfeited halves > 0.

Note: matches ran 10 in parallel on a laptop, so bank figures include CPU contention and overstate single-process use. Bank figures for the llm bots were measured under the same load.
