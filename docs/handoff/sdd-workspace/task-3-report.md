# Task 3 report: archetype zoo
Commit 459fca4 feat(sim): archetype opponent zoo
Files: sim/opponents/archetypes.py, tests/test_zoo.py
RED: pytest tests/test_zoo.py -> collection error (module sim.opponents.archetypes missing).
GREEN: full default suite 43 passed, 1 deselected. mc_equity_slow 100-deal test ~3.7 s; no test marked slow.
Nit calibration (100k random hands): threshold 10.3 passes 19.8%; raise threshold 13.8 passes 5.0%. A test asserts 15-25%.
Design: ZOO values are classes/partials/bot_file_factory partials (pickle test). RNG seeded Random(12345) per half in memory. checkfold_after(n) exported; ZOO["checkfold_after_300"].
Self-review: raises fall back to the other raise size if the preferred is illegal; FOLD falls back to CHECK_CALL if illegal (cannot happen given engine). Preflop MC equity samples a random 5-card board.
Concerns: the 100-deal tests only check legality, not that mc_equity_slow exhausts its bank (a full half would). Nit features are my own heuristic.
