# Task 5 report

## Implemented
- sim/tournament.py: run_round_robin, focus_eval, paired_diff (returns (lo, mean, hi); pooled per-deal A-B bb over all opponents/seeds, 1.96*sd/sqrt(N)), resolve_bot, select_opponents, CLI (focus mode and `--entrants zoo --report`). Spawn Pool; workers=1 runs in-process.
- sim/match.py (additive): HalfStats gained actions_a/b, ctx_a/b (faced, faced_fold, pre, pre_fold counters, needed for H2/H4), bank_exhausted_a/b (timed_out flag). All have defaults.
- Tests: tests/test_tournament.py (6 tests), test_action_counts_calling_station in tests/test_sim_match.py.

## TDD
RED: collection error `No module named 'sim.tournament'`. GREEN: 20 passed targeted; full default suite 59 passed, 1 deselected (14 s). No test exceeded 15 s so no slow marks added.

## Zoo run
`--entrants zoo --seeds 1 --deals 300`: 19 bots, 171 matches, wall 2:31 on 10 cores. Report docs/zoo_report.md.
H1 1/19 (mc_equity_slow), H2 5/19, H3 n/a (threshold_cat by code), H4 2/19 (mc_equity_slow, nit), H5 6/19, H6 0/19.

## Commits
feat(sim): round robin, focus eval, paired comparison; docs: zoo diversity report.

## Concerns
- Bank figures are measured under 10-way parallel load, so they overstate single-process use (e.g. mc_equity_fast 5.3 s, mc_equity_slow exhausts in all 36 halves); noted in the report.
- Thresholds for H2 (<10% fold-facing) and H4 (>40% preflop fold) are my choice; mc_equity_fast sits at 10.1% for H2.
- Standings sort order for ties depends on name; slow-bot results are timing-dependent so not bit-reproducible.
- CLI --seeds N gives seeds "s1".."sN" (override with --seed-prefix).
- paired_diff takes an extra optional workers=1 kwarg.
