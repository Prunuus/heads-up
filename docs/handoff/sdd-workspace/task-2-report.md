# Task 2 report
Commit 8340224. Files: sim/__init__.py, sim/opponents/__init__.py, sim/opponents/starter.py, sim/match.py, tests/test_sim_match.py.
RED: with sim/match.py absent, pytest tests/test_sim_match.py -> "1 error during collection".
GREEN: default suite 22 passed, 1 deselected; `-m slow tests/test_sim_match.py` 1 passed (-0.516 bb, verdict B).
Design: _Handle mirrors HttpAgentHandle (_request = _post, with bank-limited timeout, errors from on_hand_end each counted, discarded action on timeout). Verdict computed by building judge HalfResults and calling judge combine_halves, so stats/verdict are identical. Action coerced via judge coerce_action (str/int accepted as the HTTP server would pass them). Default grace_s=0.00075 per brief signature (judge constant is 0.001).
Concerns: obs/results are passed as objects, not JSON round-tripped; a slow in-process call cannot be aborted, only judged after the fact; factory exception counts as new_half forfeit; bank_* = seconds charged.

## Fix report (timing semantics tests)
Added keyword overrides time_bank_s and action_timeout_s to play_half (defaults = game constants, so behaviour is unchanged). Added tests: >timeout request is an error and discarded (opponent never faces a raise), bank exhaustion is no error and stops calls, overrun of remaining bank is bank-out (no error, no more calls), on_hand_end(k) precedes act of hand k+1 and last hand arrives via final flush, flush errors counted. Command: `.venv/bin/python -m pytest -q` -> all pass; `-m slow tests/test_sim_match.py` -> 1 passed.
