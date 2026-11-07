# CRITERIA.md — Acceptance and Release Criteria for Poker Bot DESIGN v0.4

**Purpose:** This is the independent Goldfish coding agent's objective acceptance contract. All conditions are **targets/tests**, not claims that any code currently passes. No implementation is authorized by merely creating these documents. Source of truth: the official `heads-up-bundle` at pinned commit `c83ab6eeb7660a2fd1fae7df224315d41fe985a9` (reverify at submission). Reference design: `DESIGN.md`; research procedure: `EXPERIMENTS.md`.

**Verification state at drafting:** these are future acceptance tests; no bot implementation, official cross-validation report, restricted-container runtime result, or match result has been produced for v0.4. The 240-state raw pilot recovery was checked only against the **legacy independent scorer**.

## Acceptance severity

- **P0 release blocker:** Any failure prevents tournament submission, regardless of poker performance.
- **P1 strategy/calibration blocker:** Prevents claiming design v0.4 competitive readiness; an explicitly reviewed simpler baseline may be shipped **only if** its P0 passes and the owner accepts the unresolved strategic limitation.
- **P2 improvement:** Optional research; never spend the remaining time on it at the expense of P0.

## C-RULE — Official game and evaluator correctness [P0]

| ID | Setup / operation | Exact pass condition |
|---|---|---|
| R01 | Deck representation | Exactly 36 distinct cards, ranks 6–A × four suits; no duplicate physical cards |
| R02 | Category ranking | Official score ordering, including **flush > full house**, **trips > straight**, A-6-7-8-9 low wheel |
| R03 | Current hero evaluation, flop/turn/river | Exactly **10 / 40 / 100** legal 2-hole/3-visible-board combinations, never any other allocation |
| R04 | At least **10,000 distinct legal complete river hero+board states** generated independently | Full packed integer score equals official `omaha_score` in **100%** of cases (not just hand category) |
| R05 | Known board/wheel/tie/kicker tests | Compare full packed score (including kickers), handle A-high and A-low straights, shared-board ties, same-rank alternatives, four-flush false positives |
| R06 | Incomplete boards | Never call `omaha_score()` on flop/turn; use official `get_table` and legal combinations; malformed cards/lengths trigger fallback |
| R07 | Official actions | Legal sets, min raise, max pot raise, raise cap, short-stack all-in, button small-blind opener and big-blind option match `game.py` |
| R08 | Configuration | Blind/stack/raise-limit data from actual config; no reliance on 1/2/80 when the referee supplies different settings |

Independent reference may be slower than the agent; correctness takes precedence over speed during tests.

## C-THREAT — Opponent fragments, complete holdings, and nuts [P0/P1]

| ID | Requirement | Pass condition |
|---|---|---|
| T01 [P0] | FULL enumerates unseen two-card pairs | Flop `N=378`, turn `N=351`, river `N=325`, correct board-triple probes 378 / 1,404 / 3,250 at most once per `(hand,street,board)` |
| T02 [P0] | Independent comparison | FULL `B`, `E`, `N`, `hero_score` exactly match separate legal-combination reference for at least 1,000 boards including trips, paired boards, suited boards, and wheel boards |
| T03 [P0] | Reviewer's regression hand | Board **Ah Ac Kc Qd 6s**, hero **Kh Kd 8c 7s 6h** ⇒ `B=13`, `N=325`, fragment risk `0.04`; exhaustive uniform opponent five-card comparison ⇒ **17,156/65,780 = 26.0808756%** better holdings; FULL tier `MEDIUM` |
| T04 [P0] | Ties | `E` counts equal full-score pairs distinctly from beating pairs; `B==0` on river proves unbeatable, **not necessarily untied** |
| T05 [P0] | REDUCED sampler | Exact SHA-256 + affine-permutation byte contract of DESIGN §2.2; 64 unique indices, fixture `research/sampler_fixtures_v04.json` matches **all** indices for 3 boards × salts 0 and 1 across Python 3.10/3.12/3.13 where available; cache hits consistent; sample zero **never** implies CURRENT/LOCKED/NUTS |
| T06 [P0] | Nuts categories | Maximum ace-high straight flush on current board can be `LOCKED_NUTS`; FULL `B==0` flop/turn is `CURRENT_NUTS+VULNERABLE_NUTS`, not LOCKED; FULL river `B==0` is locked; other REDUCED states never claim nuts |
| T07 [P0] | Vulnerable-nuts action precedence | No automatic pot bet or reraise on flop/turn due solely to CURRENT_NUTS; conditional half bet cap/board guard; rejected calls never transition to raises |
| T08 [P1] | New independent calibration coverage | At least 60 **fresh** states per flop/turn/river, ≥60 FULL-STRONG across all streets, ≥30 each FULL-MEDIUM and FULL-WEAK, ≥20 states each with paired board, trips board, ≥3 same-suit board, and ≥20 states within 0.015 of each pair-threshold `.03` and `.12` (overall); record coverage per street/tier/texture. Stratify deliberately; **if not met: INCONCLUSIVE, not PASS**. All reference values use official `get_table` and source commit. |
| T09 [P1] | Exact risk and uncertainty | Report per-tier exact `p5_current_stronger`, `p5_tieonly`, ≥50% risk misclassification rate for STRONG, and 95% **Wilson** interval using the number of sampled STRONG states. Provisional acceptance: observed severe STRONG misclassification ≤5% **and** Wilson upper ≤10%; no missing category accepted. This does not prove the true rate ≤5%. |
| T10 [P0] | Optimized graph reference | For each state, construct exact BEAT/TIE pair graph and count a five-card holding as beating iff any of ten internal pairs is BEAT, else tie-only iff any is TIE. Check the 13/325 and 17,156/65,780 example. Verify against independent direct full-five-card official-table enumeration on ≥3 states (one per street, including nonzero beating and tying counts across chosen cases); require exact count equality. |
| T11 [P1] | FULL/REDUCED tiers | On same ≥180 fresh states, report false stronger-tier and false weaker-tier separately (FULL serves as tier reference), disagreements near both `.03/.12` boundaries, and breakdown across paired/trips/suited/ordinary boards. Provisional REDUCED acceptance: upward-tier disagreement ≤10% overall **and** Wilson upper ≤20%; ≥40 each REDUCED tier across test set or INCONCLUSIVE. Also report (not hide) if FULL tier is itself inaccurate versus five-card risk. |
| T12 [P1] | Action-impact and salt sensitivity | For ≥200 **reachable game decision states**, compare FULL/REDUCED actions with all other policy inputs held fixed; report action disagreement by street and action class, overall ≤5% provisional target. Run REDUCED with offline diagnostic salts 0,1,2,3 (salt 0 deployed); report class/action variance, especially the 13/325 regression. No automatic salt optimization against final seeds. |
| T13 [P0] | Pilot reproducibility | Recover 240 raw card inputs via published legacy generator; require legacy hero-score and pair-count agreement in all rows. The **official** verification script must independently confirm official packed hero score, pair counts, five-card BEAT/TIE counts for all 240 rows; expected output checksum is recorded **only after execution**. If official check has not run, mark the historical pilot PRELIMINARY / C-RESEARCH INCOMPLETE, not PASS. |

**Critical distinction:** Testing T08/T09 does not establish actual poker EV, opponent betting ranges, or future-runout showdown equity. Do not falsely claim it does.

## C-DRAW — Exact per-route completion [P0]

1. Formula denominators are conditional on visible hero+board cards: `U=28` flop, `U=27` turn, **not** a full 52-card deck or a pool with unobserved opponent cards removed.
2. Flush regression: exactly two same-suit hero cards and two matching flop board cards, no other suit blockers ⇒ `1−C(23,2)/C(28,2) = 0.33068783...` (33.07% rounded). On turn with two suit-matching board cards ⇒ `5/27=0.185185...` (18.52%). With exactly one matching flop board card and exactly two suit-matching hole cards ⇒ `C(6,2)/C(28,2)=0.03968254...` (3.97%). Verify independently via enumerate remaining community runouts.
3. Support hole-card blockers: a third/fourth matching suit in hero's five reduces available flush outs and must change the value. More than two hole cards of a suit do not permit using three of them at showdown.
4. Straight regression: exactly six valid short-deck straight rank sets including `{A,6,7,8,9}`; exactly two distinct hero ranks plus three distinct board ranks. Verify rank blockers and one-/two-rank completion probabilities by enumeration, including duplicated board ranks and blocked outs.
5. For multiple possible draw routes take their **maximum**, not falsely their union. If a made hand already meets/exceeds the target category, do not double-count that target draw. `draw_p` must be labelled completion chance, not win equity.
6. For at least 100 synthetic but physically consistent hole/board draw states, exact formula agrees with independent runout enumeration to `1e-12` absolute tolerance.

## C-POT — Pot-limit mathematics and reachability [P0]

1. In a normal half-pot bet into pre-bet pot `P0`, `price=0.25` (exact if bet equals `P0/2`); a full pot bet gives `price=1/3` (short stack/minimum raise may change actual available amount). The *20%* figure sometimes used for a half-pot bet is **incorrect**.
2. Assert the agent computes `price=min(to_call,my_stack)/(pot+min(to_call,my_stack))`, never `to_call/pot` or `to_call/(pot+2*to_call)`. Test 0-cost check, partial all-in call, button's initial one-chip complete, and fully legal maximum pot reraises.
3. Generate **≥10,000 agent decision observations by driving the official `Game` state machine through legal action sequences**, balanced across both seats/streets and including 4-raise caps. Every returned action must be in the actual generated `legal_actions`; no exceptions. Record achieved coverage for each street and action category, not only the count.
4. Separate **≥100 deliberately inconsistent/adversarial synthetic observations** (wrong card count, duplicated card, invalid raise mapping, stale history, impossible chip state, missing keys): whenever a legal action set is still recoverable, choose precisely documented fallback; when it isn't, record a protocol-test failure rather than synthesize an action.
5. Test alias example with `RAISE_HALF` and `RAISE_POT` same `raise_to`: agent returns `RAISE_HALF` for any intended raise. Never compare nominal half/pot sizes when amounts are identical.
6. Test every preflop/postflop call ceiling with boundary conditions just below/at/above threshold and contextual deductions. At least check typical `.25` and `1/3` prices on actual reachable referee states; verify first-matching-rule precedence and cap comparisons use `raise_cost`, not `extra_vs_call`.
7. A failed raise after approved call ⇒ `CHECK_CALL`; a failed first bet when free ⇒ `CHECK_CALL`; a rejected call ⇒ `FOLD` where legal, no opportunistic reraise. Never fold a free check.

**C-ACTION — Binding deterministic branch-precedence matrix [P0]**

For each relevant STRONG free-action branch in DESIGN §§8.1 and 9.1, use a *reachable* `Game` observation plus injected fixed precomputed assessment to isolate predicate combinations. All expected actions are after the alias canonicalization rule:

| Case | Pot predicate and legality | Half predicate and legality | Effective amounts | Required action |
|---|---|---|---|---|
| A01 | true | true | distinct | `RAISE_POT` |
| A02 | true | false | distinct | `RAISE_POT` |
| A03 | false | true | distinct | `RAISE_HALF` |
| A04 | false | false | — | `CHECK_CALL` (check) |
| A05 | true | true | same `raise_to` | `RAISE_HALF` |
| A06 | sizing conditions true but pot is illegal or fails its cap | true and legal | distinct | `RAISE_HALF` |
| A07 | sizing conditions true but both raise mappings malformed | no valid raise | — | `CHECK_CALL` |

Repeat the matrix for LOCKED-NUTS free and facing-wager branches; a rejected call must never be rescued by a raise. For any `STRONG` predicate involving modes, explicitly evaluate `((mode==FULL and full_risk<=limit) OR (mode==REDUCED and reduced_sample_risk==0))`, with no cross-mode leakage. Generate test states from official `Game` when feasible; use tightly scoped synthetic/assessment stubs only for otherwise rare predicate intersections, and identify those cases as synthetic rather than achievable poker outcomes.


## C-MEM — Per-half opponent model and hooks [P0]

| Test case | Required behavior |
|---|---|
| Fresh `new_half`, valid dynamic config | All memory fields explicitly initialized; chosen eval mode fixed and not inferred from previous half |
| Repeat same observation with identical `action_history` | No opponent statistics or raise-response counters increment twice |
| Longer history in same hand | Only newly appended actions processed, single response resolved per own raise |
| Hand index jumps or history shrinks | Skip ambiguous entries and avoid replay; reset per-hand cursor safely |
| Own raise, direct opponent fold ending hand | Unique pending response + terminal winner `me`, no showdown ⇒ exactly one fold-to-raise response |
| Own raise, showdown without another decision | Unresolved immediate response ⇒ nonfold if uniquely justified; not a fold; no fabricated VPIP |
| Opponent called then later folded | Resolve immediate response as nonfold when observed; do not retroactively count fold-to-first-raise |
| Big blind receives hand result after opponent button folds before any action request | Do **not** invent observed preflop action or true VPIP participation count |
| Batch of hand results delivered twice or out of order | `processed_result[hand_index]` fixed-size per half (length `config.num_hands`), keyed by the half instance and index; each accepted unique terminal result processed at most once; no counter increments on duplicate |
| Terminal result with no pending raise | No inferred opponent action or fold response |
| Same numeric half run twice locally | Identical memory initialization; no leakage from prior process/global state |
| Malformed `on_hand_end` payload | Hook catches expected failure without throwing; counters stay consistent |
| One versus both raises having same final amount | No duplicate size-specific fold-response statistics |
| Postflop free `CHECK_CALL` | Count as observed *check*, not opponent voluntary payment |
| Opponent profile with deliberately missing first actions | `observed_vpip` denominator only qualifying observed opportunities; TIGHT/STATION not inferred from unseen actions |

Diagnostics: disabled by default; enabled test mode keeps ≤32 events of ≤160 characters. No unbounded logs or unbounded hand history retained.

**C-MEM atomicity requirements [P0]:** (M01) duplicate byte-identical terminal result causes no counter or bit changes; (M02) conflicting duplicate same index causes no counter changes and emits at most one bounded diagnostic; (M03) out-of-range and malformed terminal results do **not** mark any bit or partially update aggregate/pending state; (M04) valid batch delivered as `(5,3,5,4)` processes exactly indices 3,4,5 once irrespective of order; (M05) on new half all processed bits reset; (M06) `new_half`, `choose_action`, `on_hand_end` all independently catch expected malformed optional inputs, while missing nonrecoverable required protocol fields fail harness acceptance; (M07) diagnostics cap at 32 ×160 bytes/characters as specified; (M08) no silent counter corruption when `on_hand_end` errors.

## C-MODE — Runtime, health, and shipping [P0]

Test both modes **before mode selection**, in restricted local container matching the tournament's CPU and memory environment. For each of four official opponents on two stress-only seeds, play a complete **1,000-deal duplicate match (both halves)**, plus at least one call-heavy and one max-raise stress workload for **each** mode (no mode has been selected at E0).

Every half must have: **0 referee errors, 0 forfeits, 0 time-bank exhaustion, 0 illegal actions**, charged time **≤5.5 seconds**, and maximum entire HTTP request **≤100 milliseconds**. Startup reachability ≤12 seconds; hard official 1-second and 15-second limits are never intentionally approached. Read-only deployment and `/tmp` behavior must pass official build/smoke tests. Confirm <4 GB resident memory and image-size increment <3 GB.

`FULL_PAIR_v04` passes only if all its runs meet this gate. Also measure `REDUCED_64_v04` with its *own* complete E0 runtime suite. **Do not choose either mode until E1 accuracy eligibility has separately passed.** If neither passes, **no implementation is approved for release** without another reviewed design revision. The mode choice is a recorded constant, frozen for every later dev/final tournament match. Never change modes during a hand/half.

Use the provided `test_agent.py` / `pytest`, and the official Docker build/smoke check. No judge/evaluator/game modifications in deliverable code. Zero unexpected exceptions in `new_half`, `choose_action`, or `on_hand_end` in full testing; exceptions surfaced by a local unit test must be fixed rather than merely caught.

## C-SCORE — Exhaustive preflop structural checksum [P0]

Independent enumeration must produce exactly `C(36,5)=376,992` holdings and the following suit-pattern counts: `2-2-1=139,968`; `2-1-1-1=104,976`; `3-2=36,288`; `3-1-1=81,648`; `4-1=13,608`; `5=504`. Score minimum 0, max 93, mean **37.9097±0.0001**, `S≥40=40.129%`, `S≥60=9.211%`, `S≥70=2.902%`, `S≥80=0.439%` (tolerance ±0.01 percentage points). Default button open actions: fold 6.82%, limp 53.05%, raise half 30.92%, pot 9.21% (±0.01 pp). These are *distribution tests only*, not poker profitability tests.

## C-STRATEGY — Decision boundaries and tournament evidence [P1]

- For each branch in §§6–9 of DESIGN, create one matching state, one just-failing state, and one raise-unavailable state; **≥60 total deterministic edge-case assertions** with exact expected actions.
- Freeze a no-adaptation/no-pure-bluff policy baseline. Compare ablations only with matched configurations, complete paired duplicate games, development seeds and identical opponent suite. No benchmark success inferred from `--deals 200` debug runs.
- Promotion on DEVELOPMENT seeds, not final: all P0 green, **≥0.5 total added match points** across official four opponents × two dev seeds, **≥+.02 mean paired bb/deal**, and **no opponent loses ≥1.0 match point** across its two dev matches; report CI of per-deal paired outcomes and variance of candidate-minus-baseline difference. These are engineering screening criteria, not significance guarantees.
- The **FINAL holdout** is run once **after** all tuning and mode choice are frozen. Its scores are a report/audit; never iterate strategy values on its results. If poor, revert and use new untouched seeds only after any new development. Track match points, win/chop/loss counts, paired bb/deal and CI, plus runtime/error outcomes for every bot-versus-opponent matchup.
- No numeric win rate or competitive superiority claim is permitted before experiments actually run and are logged.

## C-FREEZE — Ordered mode-selection gate [P0]

1. No mode is approved before E0. E0 outputs only `RUNTIME_PASS` / `RUNTIME_FAIL` for **each** mode; it must not freeze a selected mode.
2. E1 is attempted for each runtime-passing mode. E1 outputs `ACCURACY_PASS`, `ACCURACY_FAIL`, or `ACCURACY_INCONCLUSIVE` based on the official scorer, exact graph, coverage, tier/action disagreements and sampler fixtures. INCONCLUSIVE cannot be treated as PASS.
3. Prefer FULL iff `FULL.runtime==PASS && FULL.accuracy==PASS`; otherwise choose REDUCED iff its two gates pass; otherwise block implementation release, *even if FULL is merely faster*.
4. Freeze mode, salt=0, source/evaluator/config/policy hashes and exact test evidence **before** E2–E4; do not silently change the mode during match, during tuning, or after reading final holdout.
5. Any change in mode, evaluator semantics, REDUCED sampler, or its salt requires E0 and E1 again. If a subsequent change invalidates prior E2–E4 comparisons, redo them or explicitly mark them invalid.

## Release readiness checklist

- [ ] P0 rule/evaluator, pot/reachability, memory/protocol, runtime and build checks **all pass**.
- [ ] FULL or REDUCED selected and frozen with mode/config hash (not changed after final audit).
- [ ] P1 five-card threat calibration and full decision boundaries complete; unresolved failures explicitly documented.
- [ ] Development-selected candidate documented with all opponents/seeds; no retrospective seed selection.
- [ ] Final holdout conducted **once**, archived with raw outputs; no tuning on it.
- [ ] Every `[TUNE]` constant is either frozen with justification or marked an acknowledged uncertainty.
- [ ] No claim of GTO optimality, optimal thresholds, or tournament wins without data.
