# Implementation record

The user's Goldfish task authorizes implementation; their subsequent instruction
changes the repository branch to `licheng`. Historical statements inside the
Elephant package describing a documentation-only handoff are artifact status,
not restrictions overriding that request.

The official upstream HEAD remained `c83ab6eeb7660a2fd1fae7df224315d41fe985a9`.
Game, evaluator, judge, requirements, official tests, Dockerfiles and smoke/build
tools retain official source. `agent.py` retains the entire server plumbing.
The fork began at `e9ce98dc5749672c1d85fc9c12fcc3e901256969`.

The strategy is split into evaluation, memory, and policy modules. Evaluation
uses full packed table scores for every legal two-hole/three-board combination.
Cache entries belong to the current hand and visible street/board. Changed hole
cards, backwards streets, or inconsistent board prefixes fail safely. FULL
enumerates all feasible fragments. REDUCED uses the fixture sampler with salt 0.
Only the specified maximum straight-flush proof, and FULL's zero beating pairs
on river, establish locked nuts.

Statistics use observed first actions and resolved immediate responses.
Unobserved button folds do not become VPIP observations. Terminal results are
validated into scratch deltas before committing counters and a fixed per-half
deduplication bit. Malformed results leave strategic counters and pending
responses unchanged. Repeated histories/results do not replay statistics.
All hooks contain expected failures; a bounded internal-error count makes
fallbacks visible to tests and research. An empty/unparseable legal action set
is an explicit protocol error.

Defensive validation precedes history ingestion so malformed observations cannot
alter statistics. This refines DESIGN §2.5's defensive ordering without changing
valid-state actions. Call ceilings are rounded to 12 decimal places after
arithmetic to preserve equality boundaries against binary floating-point noise
(for example .12 + .02). No numeric strategic threshold was tuned.

The supplied research utility has executable defects. Its original bytes remain
under `design/research/`. `research/verify_design.py` corrects:

- A tuple accidentally used as the `slow_reference_match` dictionary key, causing
  CSV writer failure and incorrect subsequent access.
- Slow-reference selection taking only three flop rows; the corrected runner
  takes one state per street and checks exact counts against the direct oracle.
- Missing legacy columns in fresh inputs being treated as failed legacy checks.
- A shared locked flag allowing FULL river proof to label REDUCED nuts.
- Board trips failing the danger test when hero had a full house or higher.

All 240 recovered historical rows match official packed scores, fragment counts,
five-card beating counts, and tie-only counts. Fresh verification is separate
from historical provenance. Graph and independent slow full-hand reference agree
on all selected states, including nonzero beats and ties. The kings-full
regression yields 13/325 fragments and 17,156/65,780 complete holdings.

Tests use official Game observations for protocol, prices, transitions and
reachability. Predicate matrices inject assessments for rare intersections;
synthetic legal-set/stack/mapping overlays are identified in test docstrings.
The 10,006-decision coverage harness drives additional legal actions to reach
late streets and raise caps. If it declines a proposed raise, it discards that
unexecuted proposal's pending response rather than inventing an opponent action.

The only runtime dependencies are the starter's FastAPI and uvicorn. Research,
PDF tooling, tests, design documents, and reports are excluded from the image.
No neural networks, Monte Carlo, CFR, unseen-card reconstruction, cross-half
persistence, or runtime mode switching were introduced.
