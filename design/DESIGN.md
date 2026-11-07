# DESIGN.md v0.4 — Deterministic Balanced-Exploitation Short-Deck PLO5 Bot

**Status:** Elephant-phase design, **not bot implementation** and **not a demonstrated winning strategy**. This focused v0.4 revision resolves the senior engineering review of v0.3 without rewriting the policy. All numeric policy thresholds are frozen *provisional hypotheses* (`[TUNE]`), not estimated equity or verified optimal choices. The authoritative rule reference is the official [`heads-up-bundle`](https://github.com/tamu-datathon-org/heads-up-bundle) at inspected commit `c83ab6eeb7660a2fd1fae7df224315d41fe985a9` (2026-11-07), together with the 15-page participant guide. Recheck upstream and record a new commit before implementation/submission; the organizer reserves the right to change limits.

**Companion documents:** [`CRITERIA.md`](CRITERIA.md), [`EXPERIMENTS.md`](EXPERIMENTS.md), [`CHANGELOG.md`](CHANGELOG.md), [`RESEARCH_README.md`](RESEARCH_README.md), [`READINESS.md`](READINESS.md), and [`MANIFEST.json`](MANIFEST.json). They are part of this specification. In any conflict: official judge/game/evaluator code > this design's `[RULE]` statements > explicit action trees > examples > experiment suggestions. Do not silently rewrite strategy or referee code to resolve inconsistencies.

## 1. Objective, rules, and deployment constraints

1. **[RULE]** Heads-up, 36-card ranks 6–A (4 suits), five private cards per player, five community cards; showdown requires **exactly two private plus exactly three board cards**. Ranking: straight flush, quads, flush, full house, trips, straight, two pair, pair, high card; A-6-7-8-9 is the lowest straight. Compare the official packed **full scores**, including kickers.
2. **[RULE]** Initial stacks 80 chips (40 BB); small blind 1, BB 2; reset stacks every hand. Button acts first preflop, out of position after flop. `FOLD`, `CHECK_CALL`, `RAISE_HALF`, `RAISE_POT` only; four voluntary raises/street maximum. **`obs.legal_actions` and `obs.raise_to` are authoritative**, including all-ins, minimum raises and raise aliases.
3. **[RULE]** Each match has 1,000 deals replayed with seats swapped. For each deal pair the bot's two chip results, divide by the BB, compute mean ± `1.96 × sample_SD/sqrt(1000)`. A wholly positive interval wins one match point; wholly negative loses; zero-touching interval chops for 0.5. Forfeiting either half loses the match regardless of chips.
4. **[RULE]** Eight charged seconds per 1,000-hand half, 1-second hard request timeout, first millisecond per request uncharged, five errors/half ⇒ forfeit. Charging covers `new_half`, `choose_action`, and batched `on_hand_end` as included in HTTP requests. Startup health check must complete within 15 seconds. One CPU, 4 GB memory, no GPU/internet; files read-only outside `/tmp`. A fresh server/memory per half.
5. **Goal:** maximize expected tournament match points versus unknown opponents, with paired chip results, robustness, and reliability as important secondary objectives. No Monte Carlo online, CFR solver, neural model, persistent opponent database, or changes to referee-owned components.

## 2. Definitions, policy modes, and universal action contract

### 2.1 Exact state quantities

| Name | Definition |
|---|---|
| `P` | `obs.pot`, *including* the opponent's current wager and all prior chips already contributed |
| `C` | `min(obs.to_call, obs.my_stack)`; the actual additional investment to call |
| `price` | If `C>0`, `C/(P+C)`; else 0. This is only the immediate showdown break-even equity **if no future betting is possible** |
| `SPR` | `min(obs.my_stack,obs.opp_stack)/max(P,1)`; a heuristic pressure indicator |
| `raise_cost(a)` | `obs.raise_to[a] - obs.my_bet`, the *total new investment* for the offered action `a` |
| `extra_vs_call(a)` | `raise_cost(a)-C` |
| `hero_score` | Max packed five-card score obtainable from exactly 2 hero hole cards and 3 *visible* board cards |
| `K` | `hero_score >> 20`, category 0 to 8 |
| `N,B,E` | In FULL mode: available unseen two-card opponent pairs, pairs beating hero's **current** full score, and pairs tying it, respectively |
| `frag_risk`, `tie_fragment` | FULL: `B/N`, `E/N`; REDUCED: sampled pair proportions, **never mathematical probabilities of a five-card opponent winning** |
| `draw_p` | Largest exact *single-route* flush or straight completion chance; not the chance to win at showdown |
| `danger_board` | Three or more board cards of a suit; or board trips; or board paired while `K<5` |

Validate card IDs (0–35), uniqueness, street/board-length agreement, and required observation fields before strategic calculations. Do not try to correct inconsistent referee states. On malformed-but-respondable input, return the legal fallback; if `legal_actions` itself is empty or unparseable, classify as a protocol failure in tests rather than inventing a legal move.

### 2.2 Exactly two *preapproved* evaluator configurations

**`FULL_PAIR_v04`**: exhaustive currently feasible opponent two-card pair enumeration on each new board using the official five-card lookup table. Score all `C(28,2)=378` flop pairs × one board triple, `C(27,2)=351` turn pairs × four triples = 1,404 probes, and `C(26,2)=325` river pairs × ten triples = 3,250 probes. Also compute hero score (10/40/100 probes). Cache by `(hand_index, street, visible_board)`; never recompute after a second action on the same board. Exact `B==0` proves **current** unbeatable hand, not future showdown safety on flop/turn.

**`REDUCED_64_v04`**: same **exact hero score**, board features, draw formulas, and opponent policy, but evaluate exactly `min(64,N)` distinct opponent two-card fragments per new board. Pair indexing is lexicographic over sorted unseen card IDs. To avoid Python-version-dependent `random.sample` behavior, select indices using the version-independent affine permutation specified next. A zero-beating sample **never** establishes `CURRENT_NUTS`, `LOCKED_NUTS`, or a safe all-in. The only non-FULL `LOCKED_NUTS` proof is the explicit maximum ace-high straight-flush condition in §2.4.

**REDUCED sampler contract, byte-for-byte:** form `payload = ASCII("PB4/REDUCED64") || 0x00 || bytes(sorted(hole_ids)) || bytes([street_id, board_length]) || bytes(ordered_visible_board_ids) || salt.to_bytes(4,"big")`, where official `Street` IDs are flop 1, turn 2, river 3; `salt` is a nonnegative 32-bit integer **frozen to 0 for deployment**, and `||` means concatenation. Compute the 32-byte SHA-256 digest `d`; let `offset=int.from_bytes(d[0:8],"big") mod N`; let `step=1+(int.from_bytes(d[8:16],"big") mod (N−1))`. While `gcd(step,N) != 1`, set `step=(step mod (N−1))+1`. Select the **sorted** indices `{(offset+j×step) mod N : j=0,...,min(64,N)−1}` without replacement. These indices address lexicographically enumerated unseen two-card pairs (zero-based). No Python RNG, platform hash, global state, or per-request salt change is permitted. The research fixture `research/sampler_fixtures_v04.json` is normative, including full 64-index lists and hashes; validate multiple Python runtimes before approval. The sampler is deterministic, but **not guaranteed to produce an unbiased subset** for every state; calibration across independent offline salts is required.

**Mandatory mode-selection state machine:** (1) **E0 runtime eligibility**: independently measure both configurations and mark each runtime PASS or FAIL; do not freeze yet. (2) **E1 correctness/accuracy eligibility**: officially cross-validate each E0-passing mode against exact complete five-card research, tier disagreement, and coverage; mark each accuracy PASS, FAIL, or INCONCLUSIVE. (3) **Freeze** `FULL_PAIR_v04` if it passed *both*, otherwise `REDUCED_64_v04` if it passed *both*, otherwise **STOP** (no approved evaluator mode). (4) **E2–E4 strategy experiments** may now begin; freeze one mode, one sampler salt and one configuration/policy hash before evaluating development matches. (5) Final sealed holdout is a one-time audit, never a mode-selection input. A later mode change restarts E0/E1 and invalidates subsequent strategy comparisons until re-run. Never silently change modes during a half or after examining holdout results. FULL retains the v0.3 exact evaluator algorithm; the `_v04` suffix identifies this clarified configuration contract.

### 2.3 Exact hero evaluation contract

Use the starter's `evaluator.get_table()` (already populated by `warmup()` during `agent.py` startup). The 36-bit key is the OR of the bits for **five distinct physical cards**. For every private two-card combination (10) and every three-card subset of the *visible* board (1/4/10 for flop/turn/river), look up the full packed score; take the maximum. **Never call `omaha_score()` on fewer than five board cards.** On river it must equal `omaha_score(hero_hole, full_board)` bit-for-bit. Rank/straight/tiebreak rules come from the official evaluator only. Each opponent pair uses the same table with exactly two cards in that pair, not a fabricated five-hole-card hand.

### 2.4 Relative strength and nuts terminology

A player's opponent actually holds **five** private cards, allowing up to ten two-card choices. Thus `frag_risk` is a fragment-frequency feature only. Do **not** convert it to opponent-five-card risk using `1-(1-frag_risk)^10`: the events overlap and are dependent.

Use these provisional **mode-specific** class rules on flop/turn/river (`[TUNE]`):

| Condition, evaluated in order | Tier/flag |
|---|---|
| Proven `LOCKED_NUTS` | Tier `NUTS`, `locked=true` |
| FULL: `B==0` on flop/turn | Tier `STRONG`, `CURRENT_NUTS=true`, `VULNERABLE_NUTS=true` |
| FULL: `B==0` on river | Tier `NUTS`, `CURRENT_NUTS=true`, `LOCKED_NUTS=true` |
| FULL: `0<B/N≤.03` | `STRONG` |
| FULL: `.03<B/N≤.12` | `MEDIUM` |
| FULL: `B/N>.12` | `WEAK` |
| REDUCED: sampled `frag_risk≤.03` **and** (`not danger_board` **or** `K≥5`) | `STRONG`; no nuts inference |
| REDUCED: sampled `frag_risk≤.12` (including a low-risk dangerous board failing the preceding rule) | `MEDIUM` |
| REDUCED: sampled `frag_risk>.12` | `WEAK` |

These thresholds are **not** opponent five-card winning odds. The not-yet-officially-cross-validated legacy pilot in §4 is descriptive only; the recovered raw card inputs have been checked against the independent legacy scorer, but official-evaluator validation remains a mandatory research gate. The tiers do not establish betting profitability. The tiers are used conservatively and must be validated against held-out exact five-card enumeration (CRITERIA C-THREAT).

`LOCKED_NUTS`: a sufficient proof that no future legal runout/opponent holding can produce a superior showdown hand. The **only** accepted flop/turn proof in v0.4 is that our current best hand is the globally maximum **ace-high straight flush** (official category 8 with straight top rank A). For FULL on river, `B==0` also proves locked unbeaten status, with possible ties. No other pattern may be called locked without a separate mathematical proof and criteria update. **If uncertain, treat as vulnerable.** `CURRENT_NUTS` means no presently feasible stronger hand on the *visible* board; `VULNERABLE_NUTS` is CURRENT but not LOCKED. REDUCED cannot assert CURRENT except where the maximum straight-flush proof also establishes LOCKED.

For sizing conditions requiring `tie_fragment`, FULL uses `E/N`, REDUCED uses its 64-pair sample value, with no claim of exact opponent tie probability. “Tie-heavy” means `tie_fragment>=.30` as a provisional sizing guard. A river `NUTS` still need not win the full pot because ties remain possible.

### 2.5 Absolute action precedence and fallback

**In each `choose_action`:** (1) ingest newly observed history once; (2) validate inputs and detect game/config mode; (3) calculate score/cache if postflop; (4) calculate profile from resolved historical opportunities only; (5) run the street's **first matching** action rule; (6) validate/canonicalize; (7) return one actual member of `legal_actions`.

For a free check, safe fallback is `CHECK_CALL`. When chips are owed, safe fallback is `FOLD` if listed, otherwise `CHECK_CALL`. If a value raise fails legality/investment guard **after calling was approved**, return `CHECK_CALL` (call), not fold. If a free-action bet fails its guard, return `CHECK_CALL` (check). Never fold free checks.

If BOTH raises are listed with identical `raise_to`, normalize any intended raise to `RAISE_HALF` (the actions are economically identical). Otherwise choose only the specified offered action. Reject raises with missing mapping, nonpositive `raise_cost`, `raise_cost>my_stack`, or `extra_vs_call<0`. Never synthesize a bet amount or assume a missing action is legal. Every comparison uses the exact recorded raise cost, including all-in aliasing.

**Ordered-branch semantics for every policy branch:** assess all predicates for the *current action* before selecting it. Do not return a nominal pot raise unless it is legal and meets every cap and mapping condition. When a pot branch fails, independently test any explicitly specified half branch; when both fail, return the approved call or free check. When a raise is an alias (`raise_to[HALF]==raise_to[POT]`), a fully qualified pot branch still resolves to the legal `RAISE_HALF` action. There is no unreviewed alternative bet size, and the sizing rule must not retroactively alter a prior call/fold decision. Test both sizes eligible, pot-only, half-only, neither, alias, and a disallowed pot with legal half; examples are normative acceptance fixtures in `CRITERIA.md`.


## 3. Preflop score and exact structural distribution (preserved from v0.2)

**Correction:** v0.1 gave +8 for *any* suit appearing twice. Every five-card hand has that property because there are only four suits, so that feature was constant. v0.2 replaces it with the complete suit multiplicity pattern and explicitly penalizes redundant suits.

For a five-card hand, let `rank(c)` be 0–8 for 6–A and `suit(c)` be 0–3. Define **[TUNE]**:

`S = clamp(2*H + PAI + SU + CON - DUP, 0, 100)`.

- `H`: sum per physical hole card, counting duplicates: A=6, K=4, Q=3, J=2, T=1, 6–9=0.
- `PAI`: take the **highest rank** present at least twice and award AA=18, KK=15, QQ=12, JJ=9, TT=7, 99=5, 88=4, 77=3, 66=2; add 6 if another **different** rank also appears twice or more. Otherwise 0. A triple counts as a pair for this purpose but is separately penalized.
- `SU`: sort four suit counts in descending order and discard zero counts; use the full table below. Additionally add **+4 once** if at least one ace has another hole card of the same suit.
- `CON`: use *distinct hole-card ranks* (not physical duplicate cards). For every pair of different ranks, add 3 if their rank indices differ by 1, 1 if they differ by 2, and 1 if they are the special `(A,6)` wheel pair. Otherwise add 0. Cap at 15. This modest wheel feature does not claim A-6 alone makes a good straight draw.
- `DUP`: subtract 7 per rank occurring exactly three times or 14 per rank occurring four times. Otherwise 0.

| Suit pattern (descending nonzero counts) | `SU` before ace bonus | Reason |
|---|---:|---|
| 2-2-1 | +12 | Two distinct usable suit pairs |
| 2-1-1-1 | +4 | One usable suit pair, little redundancy |
| 3-2 | +7 | Two usable suits, but one redundant third card |
| 3-1-1 | +2 | One usable suit, extra redundancy |
| 4-1 | −4 | Heavily redundant suit concentration |
| 5 | −9 | All cards one suit; only two can be used |

### 3.1 Exhaustive structural enumeration — actual calculation, not equity testing

All `C(36,5)=376,992` legal five-card private-card combinations were independently enumerated using exactly the v0.2 formula above. This is a count of feature scores, **not** a measured win rate or demonstration that the score is predictive of equity.

| Suit pattern | Hand count |
|---|---:|
| 2-2-1 | 139,968 |
| 2-1-1-1 | 104,976 |
| 3-2 | 36,288 |
| 3-1-1 | 81,648 |
| 4-1 | 13,608 |
| 5 | 504 |
| **Total** | **376,992** |

| Minimum score | Maximum score | Mean score | `S≥40` | `S≥45` | `S≥60` | `S≥65` | `S≥70` | `S≥80` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 93 | 37.910 | 40.13% | 28.01% | 9.21% | 5.77% | 2.90% | 0.44% |

**Recalibrated button-first-action distribution** for the provisional cutoffs in §6.1: fold `S<20` **6.82%**; limp `20≤S<40` **53.05%**; half-pot `40≤S<60` **30.92%**; pot `S≥60` **9.21%**. These are structural frequencies, not forecasts of decisions in full matches. Counts should be independently reproduced before using the score in production. `S` is *not* a percentile or equity probability.

**Research implication:** the old cutoff `S≥93` almost never reraises under v0.2 (0.016% of hands); the v0.2 premium-raise threshold is instead `S≥70`, still only 2.90% of holdings before situation filters. Both opening and reraising frequencies remain **[TUNE]**.

## 4. Offline five-card calibration and interpretation

### 4.1 Exact calibration reference

For a *fixed* hero hole/visible board state, first score each feasible unseen **two-card fragment** once with the official table. Make a graph whose vertices are unseen physical cards (`|U|`=28 flop, 27 turn, 26 river). Put an edge `(a,b)` in **BEAT** if that fragment's best current score exceeds the hero's full packed score, or in **TIE** if equal. Enumerate all `C(|U|,5)` unordered opponent five-card subsets `H` and count `H` as **beating** if any of its ten internal edges belongs to BEAT; else count it as **tie-only** if any edge belongs to TIE. Compute `p5_current_stronger = beating_count/C(|U|,5)` and `p5_tieonly=tieonly_count/C(|U|,5)`. This is **exact**, because the strongest legal showdown combination for a five-card opponent is the maximum across those ten fragment scores. Avoid ten repeated poker-score lookups for each five-card holding. Validate the graph result against a slower direct-official-evaluator five-card loop on at least three stratified states (flop, turn, river); do not compare the graph only with another copy of itself. This uniform current-hand statistic is **not** range-conditioned equity or final showdown odds on flop/turn.

**Mandatory regression example:** board `Ah Ac Kc Qd 6s`; hero `Kh Kd 8c 7s 6h`. Hero has kings full of aces (full packed category 5). Of `C(26,2)=325` unseen opponent fragments, exactly **13** beat hero ⇒ `13/325=4.00%` fragment risk. But exactly **17,156** of the `C(26,5)=65,780` complete opponent holdings contain a beating pair ⇒ **26.0808756%** uniform current five-card risk. Under the preserved v0.3 (and v0.4) FULL threshold, `0.04` is `MEDIUM`, not `STRONG`. These exact reference counts are correctness requirements, not evidence about realistic betting ranges.

### 4.2 Small offline pilot: evidence, not optimization

An additional *research-only* enumeration of 80 reproducibly sampled hero/board states **per street** (240 total, seed 8037, cards sampled without replacement) using the published ranking logic produced the following mean exact random-five-card CURRENT threat by cheap fragment-risk bucket:

| Board length | `frag_risk <= .03` | `.03 < frag_risk <= .12` | `frag_risk > .12` |
|---|---:|---:|---:|
| Flop (3) | 9.64% (26 states) | 45.78% (14) | 74.71% (40) |
| Turn (4) | 12.21% (33) | 41.47% (12) | 74.85% (35) |
| River (5) | 8.64% (23) | 39.15% (24) | 72.69% (33) |

Within the `<=.03` bucket, some exact-current-nuts states have risk zero and others do not. **These are pilot measurements, not match results**; the reference implementation must still be cross-checked with the official `get_table()` before using this pilot as a release gate. The states are unconditioned on opponent betting and not necessarily representative of reachable *decision* states. All cutoffs remain `[TUNE]`. Original pilot: `research/legacy_street_calibration_samples_v03.csv` (historical, independently scored). The reconstructed raw cards are in `research/pilot_raw_reconstructed_v04.csv`, regenerated from v0.3's recorded Python `Random(8037).sample(range(36),5+board_length)` recipe, in the documented street/row order. The reconstructed states matched all 240 original hero tuple scores and beating-pair counts under the *legacy independent* scorer (not the official `get_table()`). `research/calibrate_offline.py verify` must still cross-validate **all** packed official scores and five-card counts; no official validation claim is made until its report is produced. The original pilot file lacked raw card IDs, so reconstruction is an explicit provenance recovery, not magically recovered original bytes. No historical result is used as an acceptance gate.

**Release calibration rule:** Follow `CRITERIA.md` C-THREAT and `EXPERIMENTS.md` E1 using a **new stratified official-table** batch. Require minimum street/tier/texture coverage, Wilson intervals, direct FULL-versus-REDUCED disagreement rates and action-impact checks, and multiple diagnostic REDUCED salts; *insufficient coverage is INCONCLUSIVE*, never PASS. Count exact uniform complete-five-card stronger risk by the pair-graph combinatorial method in `research/calibrate_offline.py` (§4.1); validate against slower separate official scoring on representative samples. No salt other than the frozen 0 can ship. Do **not** tune on the final sealed match holdout; this calibration cannot certify match EV.

## 5. Exact draw-completion probabilities (preserved from v0.2)

Only compute on flop/turn; board has `b=3` or `4` visible cards, so `k=5-b` future community cards. The unseen pool `U` has `36−5−b` cards, assuming no hidden opponent cards are known. Unknown opponent cards are **not** removed from the denominator: these are marginal card-runout probabilities conditional only on our visible information.

### 5.1 Flush completion by a specified suit

For each suit with at least **two of our private cards**, let `h_s` be our hole cards of that suit, `b_s` board cards of the suit, `m_s=9−h_s−b_s` unseen cards of that suit, and `need_s=max(0,3−b_s)` additional board cards of the suit. If `need_s==0`, that suit's flush is already made, so it is **not a new draw**. If `need_s>k` or `m_s<need_s`, completion probability is zero. Otherwise use exact hypergeometric probability:

`p_flush(s) = Σ_{j=need_s..min(k,m_s)} C(m_s,j)*C(U−m_s,k−j)/C(U,k)`.

Use `p_flush=max_s p_flush(s)` for the strongest *individual* feasible flush route, unless a later experiment computes a union probability. The maximum of several suit-specific event probabilities is a **deliberate conservative aggregate**, not their exact union or showdown equity. No standard 52-card flush-outs shortcut is allowed.

### 5.2 Straight completion by a specified sequence

Consider precisely these rank sets: `{6,7,8,9,T}`, `{7,8,9,T,J}`, `{8,9,T,J,Q}`, `{9,T,J,Q,K}`, `{T,J,Q,K,A}`, `{A,6,7,8,9}`. For each set choose any **two distinct hole-card ranks** inside it; the other **three distinct ranks** must appear among three cards selectable from the final board. (A repeated rank on board cannot fill two required rank positions.)

Count required ranks not yet on the visible board. If zero, the straight is already constructible and is not a new straight draw. If missing-rank count exceeds remaining community-card draws, that route has zero probability. For a missing rank `x`, count its unseen outs `m_x = 4 − #visible_hole_cards_of_rank_x − #visible_board_cards_of_rank_x`.

- **One missing rank, `k` draws:** `p = 1 − C(U−m_x,k)/C(U,k)`.
- **Two distinct missing ranks, flop `k=2`:** `p = 2*m_x*m_y/(U*(U−1))`.
- All other cases: 0. If the required three board ranks are already present, 0 **new** straight-completion probability.

Take `p_straight` as the **maximum over individual valid hole-pair+straight-sequence routes** (not their event union). This prevents double-counting overlapping straight opportunities. The result is exact for each line, approximate as an aggregate. If current hand category is already a straight or stronger, set effective `p_straight=0` to avoid counting a non-improving lower-category draw.

### 5.3 Effective draw quantity used by the decision policy

If `K>=6`, set `draw_p=0`. Otherwise let `draw_p=max(p_flush if K<6 else 0, p_straight if K<3 else 0)`; cap to [0,1]. Compute only once per new board and cache. This is a conservative probability of completing **one selected target category**, not the probability of improving overall, being best at showdown, or getting paid when winning.

**Example checks:** With exactly two hole cards of a suit and two flop cards of that suit, zero additional same-suit hole blockers and no other known cards of that suit, there are five same-suit cards among 28 unseen, so the chance of at least one appearing by river is `1−C(23,2)/C(28,2)=33.07%`. On turn with two suitable board cards, five of 27 unseen cards complete the flush (`18.52%`). With only one suitable flop board card, six of 28 unseen cards are of that suit; needing **both** future cards gives `C(6,2)/C(28,2)=3.97%`. These probabilities are conditional on the specified hole composition; extra same-suit hole cards reduce outs.

## 6. Preflop deterministic decision policy

### 6.1 Button's first preflop decision — NOT a voluntary raise faced

Identify `street=PREFLOP`, button true, action history empty, no voluntary raises, `my_bet==config.small_blind`, `opp_bet==obs.big_blind`, `to_call==obs.big_blind-config.small_blind`. Config takes precedence over hard-coded blinds. If unexpected but valid state, use fallback rather than assume this branch.

Default opening bands `[TUNE]` (preserved v0.2): `S<20` ⇒ fold; `20≤S<40` ⇒ complete via `CHECK_CALL`; `40≤S<60` ⇒ `RAISE_HALF`; `S≥60` ⇒ `RAISE_POT`. For `STATION`, raise the two raise cutoffs to `45/65`; otherwise, **only if `resolved_pre_raise_responses>=12` and `fold_hat_pre>=.52`**, lower them to `35/55`; otherwise use defaults. A postflop `OVERFOLDER` label **alone** does not justify preflop opening aggression. The fold/limp boundary remains 20. The structural frequencies in §3 apply to `UNKNOWN`, not these adjustments. Any intended raise failing its legal guard becomes a *previously approved complete* (CHECK_CALL) only if S≥20; if S<20 follow fold branch.

### 6.2 Big blind after limp

Identify non-button, `to_call=0`, both street bets exactly `obs.big_blind`, no voluntary raises, and exactly one visible opponent `CHECK_CALL` completing the initial small blind. `S≥65` ⇒ pot raise; `45≤S<65` ⇒ half raise; otherwise free check. If raise legality/cost invalid, check.

### 6.3 Facing a *voluntary* preflop raise

Compute `price`, then select the **frozen provisional willingness-to-pay ceiling**, not equity:

| `S` | Base max `price` | Typical effective behavior |
|---|---:|---|
| `0–19` | .12 | Fold most raises |
| `20–34` | .21 | Defend only unusually cheap calls |
| `35–49` | .27 | Defend typical half-pot, not pot |
| `50–64` | .31 | Defend half-pot and smaller; generally not pot |
| `65–100` | .34 | Accept ordinary legal half/pot calls before adjustments |

Subtract `.05` if at least two raises occurred this street; subtract `.04` against `TIGHT`; add `.02` against `AGGRESSIVE`; no other preflop price adjustments. Clamp to `[.05,.34]`. First: if `price>ceiling`, fold. If approved to continue: with `S≥80`, exactly one voluntary raise so far, `price<=ceiling-.04`, legal pot raise and `raise_cost(POT)<=.65*my_stack`, **pot reraise**. Else with `S≥70` under the same conditions except legal half raise and `raise_cost(HALF)<=.60*my_stack`, **half reraise**. Else `CHECK_CALL`. A failed reraise guard returns the *approved* call.

**Price audit:** A nominal half-pot bet into pre-bet pot `P0` costs `P0/2` and gives `price=(P0/2)/(P0+P0)=.25`; a pot-sized bet gives `price=1/3`. Actual `obs.raise_to` min-raise/short-stack truncation can produce smaller or intermediate prices, so always compare the actual `price`. Values above 1/3 are functionally indistinguishable for ordinary legal calls under the current pot-limit rules; `.34` means “call any typical legal price,” not 34% estimated equity. **The review's informal 20% half-pot figure was corrected to 25%.** A half-pot call can still be unprofitable after future play despite an allowed ceiling.

### 6.4 Other preflop states

If `to_call==0`, check; if nonzero and not covered above, apply the safe legal fallback. If recognized as a post-raise decision but with inconsistent history, do **not** reconstruct an arbitrary premium-reraise opportunity.

## 7. Postflop call / fold acceptance

For any offered call (`C>0`), choose base ceiling by **relative class**, not by category alone:

| Tier | Flop | Turn | River | Typical effect |
|---|---:|---:|---:|---|
| `WEAK` | .12 | .10 | .08 | Fold most meaningful bets |
| `MEDIUM` | .27 | .26 | .26 | Call half-pot or smaller, usually fold pot |
| `STRONG` | .34 | .34 | .34 | Call typical legal bets unless context deducts |
| `NUTS` (PROVEN LOCKED) | .34 | .34 | .34 | Call every ordinarily legal price; evaluate raise separately |

On flop/turn with `draw_p>=.10`, compute `draw_ceiling=min(.34,.08+.70*draw_p)` and raise the base ceiling to `max(base,draw_ceiling)`. This is a **heuristic** for draw play, not implied or true showdown equity. No draw bonus on river.

Context adjustments applied **only when a wager is faced**: against `AGGRESSIVE` add `.02`; against `TIGHT` subtract `.04`; last opponent action on this street was `RAISE_POT` subtract `.04`; `raises_this_street>=2` subtract another `.04`; if class WEAK/MEDIUM and `danger_board` subtract `.03`. `STATION`, `OVERFOLDER`, `UNKNOWN`, button position, and high-SPR receive **zero** price adjustments in v0.4. Clamp to `[.03,.34]`. If `LOCKED_NUTS`, set final ceiling to `.34` even after context (the unbeatable final-hand logic overrides those deductions).

**Decision:** if `price>ceiling`, fold. Otherwise approve `CHECK_CALL` and only then evaluate separately qualified **value raises** in §8/9. A class (including `CURRENT_NUTS`) is *not* proof that calling is profitable versus an arbitrary action-conditioned range. The pot-odds comparisons are exact; the ceilings are `[TUNE]` risk-policy proxies.

**Boundary audit examples:** Typical half-pot price `.25`: MEDIUM river `.26` calls, WEAK `.08` folds; typical pot price `1/3`: MEDIUM `.26` folds, STRONG `.34` calls; STRONG facing observed opponent pot raise is `.30` and folds a full-pot-priced call, unless LOCKED. MEDIUM turn on a danger board has `.23` and folds even a half-pot price of `.25`. All must be tested on actual reachable referee states.

## 8. Flop / turn free-betting and reraising decision tree

Never automatically bet solely because the hand's five-card **category** is high. `CURRENT_NUTS` on flop or turn is assumed **VULNERABLE** unless the narrow LOCKED proof succeeds. Use the first matching condition; all bet/raise amounts are exact offered amounts.

### 8.1 No chips owed, no prior voluntary raise this street

1. **LOCKED_NUTS (free):** If `(tie_fragment<.30 and profile in {STATION,AGGRESSIVE})` and a valid `RAISE_POT` exists, select pot; else if a valid `RAISE_HALF` exists, select half; otherwise check. The pot option is deliberately vetoed when tie-heavy. No additional investment fraction cap on proven locked nuts; both sizes remain value-bet hypotheses.
2. **VULNERABLE_NUTS (FULL only):** Half-pot if `not danger_board`, legal, `SPR>=.35`, and `raise_cost(HALF)<=.50*my_stack`; otherwise free check. **Never pot automatically** for current-but-vulnerable nuts.
3. **STRONG other (explicit size precedence):** Evaluate these branches in order; each chosen raise must be legal, mapped to a positive actual investment, and pass §2.5's universal checks:
   1. If `profile == STATION` **and** `K>=5` **and** `SPR<=2` **and** `not danger_board` **and** `((mode==FULL_PAIR_v04 and frag_risk<=.015) or (mode==REDUCED_64_v04 and sampled_frag_risk==0))` **and** `RAISE_POT` is legal **and** `raise_cost(POT)<=.65*my_stack`, choose `RAISE_POT` (subject to raise-alias canonicalization).
   2. Else if `SPR>=.35` **and** `not(danger_board and K<5)` **and** `RAISE_HALF` is legal **and** `raise_cost(HALF)<=.65*my_stack`, choose `RAISE_HALF`.
   3. Else choose `CHECK_CALL` (free check). This branch intentionally prefers pot when both guards pass; there is no post-selection override.
4. **MEDIUM thin value:** Half-pot if `profile==STATION`, `K>=1`, `frag_risk<=.10`, not dangerous, and `raise_cost(HALF)<=.40*my_stack`; OR on **flop only**, hero was last preflop aggressor, `K>=1`, `frag_risk<=.08`, not dangerous, `SPR>=.75`, and `raise_cost(HALF)<=.40*my_stack`.
5. **Semibluff:** Half-pot if class WEAK/MEDIUM, `draw_p>=.20`, opponent not STATION, `SPR>=.75`, not dangerous, and `raise_cost(HALF)<=.40*my_stack`.
6. **Overfolder pure bluff:** Half-pot if `OVERFOLDER`, `fold_hat_post>=.50`, `SPR>=.75`, current hand `K<=2`, `draw_p<.20`, not dangerous, hero was preflop aggressor on flop or button on turn, and `raise_cost(HALF)<=.30*my_stack`.
7. Otherwise check.

### 8.2 Facing a wager (only after §7 approves call)

1. **LOCKED_NUTS (facing wager; call already approved):** If `(tie_fragment<.30 and profile in {STATION,AGGRESSIVE})` and valid pot raise exists, return pot; else if a valid half raise exists, return half; otherwise return the already-approved call. Tie-heavy vetoes pot. Apply the alias rule in §2.5.
2. VULNERABLE_NUTS: **call, not reraise**, even if FULL `B==0` currently; future redraw risk has not been proven absent.
3. STRONG other: raise half only if `frag_risk<=.02`, `K>=5`, opponent is STATION/AGGRESSIVE, `price<=ceiling-.05`, not danger, and `raise_cost(HALF)<=.55*my_stack`.
4. Otherwise call. No bluff-reraises flop/turn.

If a desired raise fails availability, amount, or cap checks, return **approved** CHECK_CALL. If the initial §7 call was rejected, fold and never try to raise.

## 9. River decision tree

On river the exact hero hand is final. FULL `B==0` proves opponent cannot beat it (but may tie); REDUCED sample `B==0` proves **nothing** about unbeatable status. Classify using full score and the selected mode before betting.

### 9.1 River, no chips owed; first matching rule

1. **LOCKED_NUTS (river free):** If `(profile==STATION and tie_fragment<.20)` and a valid pot raise exists, return pot; else if a valid half raise exists, return half; otherwise check. This is a first-matching rule, not a half-then-override rule. No investment cap for a proven locked hand; sizing remains unvalidated.
2. **STRONG (explicit size precedence):** First, if `profile==STATION` **and** `K>=5` **and** `tie_fragment<.20` **and** `SPR<=2` **and** `not danger_board` **and** `((mode==FULL_PAIR_v04 and frag_risk<=.015) or (mode==REDUCED_64_v04 and sampled_frag_risk==0))` **and** `RAISE_POT` is legal **and** `raise_cost(POT)<=.65*my_stack`, choose `RAISE_POT`. Otherwise, if `not danger_board` **and** `RAISE_HALF` is legal **and** `raise_cost(HALF)<=.65*my_stack`, choose `RAISE_HALF`. Otherwise choose `CHECK_CALL` (check). When both choices meet their guards, **pot wins** unless the canonical legal raise amounts are identical, in which case §2.5 returns `RAISE_HALF`. REDUCED's sampled zero is never a nuts proof.
3. MEDIUM thin value: half-pot if STATION, `K>=1`, `frag_risk<=.10`, not dangerous, and `raise_cost(HALF)<=.40*my_stack`.
4. Overfolder pure bluff: half-pot only if `OVERFOLDER`, ≥12 resolved postflop fold responses, `fold_hat_post>=.52`, button, `SPR>=.75`, `K<=1`, *observed opponent free check on river*, `raise_cost(HALF)<=.30*my_stack`, and the v0.2 blocker test: board has ≥3 suit `s`; hero holds ace of suit `s` but **fewer than two** suit-`s` hole cards and does not make a flush. This is not a proof of opponent folding.
5. Otherwise check.

### 9.2 River, facing wager; only after §7 approves call

1. **LOCKED_NUTS (river facing wager, call already approved):** If `(profile in {STATION,AGGRESSIVE} and tie_fragment<.30)` and a valid pot raise exists, return pot; else if a valid half raise exists, return half; otherwise call. Alias canonicalization still applies.
2. STRONG: half-pot reraise only versus STATION, `K>=5`, `frag_risk<=.015`, tie-fragment<.20, and `raise_cost(HALF)<=.60*my_stack`; otherwise call.
3. Otherwise call. No river bluff-reraises.

## 10. Opponent model, observation bias, and state lifecycle

### 10.1 Bounded memory contract

All per-half historical state lives in the `memory` dictionary: primitive `AgentConfig` fields; currently tracked hand index and action-history cursor; per-hand first-opponent-action and last-preflop-aggressor flags; a single pending response to our latest raise; a **fixed `processed_result[0..num_hands−1]` byte/bit array initialized to false** (no rolling cache), with a bounded fingerprint for conflicting duplicate diagnostics; fixed integer counters; cached per-board features for the current hand only; and optional diagnostics capped at **32 events × 160 characters**, disabled by default in tournament play. No cross-half filesystem/model state. On `new_half`, initialize *every* field and conservative defaults before optional work; a valid config must not cause an exception.

### 10.2 Observable statistics: exact denominators

| Estimate | Qualified successes | Denominator |
|---|---|---|
| `observed_vpip` | First **observed** opponent preflop action voluntarily invested more than forced blind (`CHECK_CALL` with positive added amount or raise) | Hands with first opponent preflop action directly **observed** |
| `observed_pre_raise` | Hands with an observed opponent preflop raise (at least one per hand) | Hands with directly observed first opponent preflop action (acknowledge later raises may be unobserved) |
| `post_aggr` | Observed opponent flop/turn/river raises | Observed opponent postflop *nonfold* actions (both free checks and calls counted; folds excluded) |
| `fold_hat_pre` | Opponent folded as *immediate response* to our preflop raise, observed or uniquely reconstructed | Resolved immediate opponent responses to our preflop raises |
| `fold_hat_post` | Opponent folded as immediate response to our flop/turn/river raise | Resolved immediate responses to our postflop raises |

**Bias warning:** A hand can finish before our agent receives any observation (e.g., opponent small blind immediately folds while we're BB). `HandResult` has no complete action history. Do **not** count that hand as an observed VPIP opportunity and do **not** infer its unobserved first action from a generic winner/pot. Therefore `observed_vpip` is **not all-hand VPIP**, and a low observed rate does not prove tightness. Record `opponent_first_action_observed_count` separately from `completed_hands_count`, display their ratio as a **coverage diagnostic only** (not a correction factor), and test profile sensitivity to missing observations. Size-specific fold-to-half and fold-to-pot counters may be recorded as diagnostic subgroups; use them for sizing choices only after ≥10 resolved responses to that exact effective size, otherwise keep the pooled estimate. When two raises have the same actual final total, do not invent separate size categories.

### 10.3 Exact transition / deduplication order

1. `new_half`: reset counters, `history_cursor=0`, hand sentinel, cache, pending raise response, processed-terminal dedupe tracker, profile UNKNOWN, and chosen **frozen** evaluation mode. Missing optional config fields use defaults only for diagnostics, never to override actual legal actions.
2. At start of `/act`, the supplied plumbing calls `on_hand_end` once for each queued finished hand **before** `choose_action`. Terminal identity is `(current_half_instance, result.hand_index)` with `0<=hand_index<config.num_hands` (currently 1,000). If `processed_result[index]` is true, skip without mutating counters; log at most one bounded conflict diagnostic if the payload fingerprint differs. For an invalid/out-of-range hand index or malformed result, skip without setting the bit or altering counters. No result is processed twice even when batches are delivered repeatedly or out of order.
3. For a **new** `hand_index`, reconcile any last pending response using terminal results already received; discard an unresolved stale one with **no inferred statistic**; reset cursor and per-hand flags; preserve aggregate per-half counters. If history length shrank or the index moved backwards unexpectedly, do not replay statistics from uncertain entries.
4. For the current hand, consume **only** new suffix `action_history[history_cursor:]`. Update directly observed first preflop opponent action, preflop raise-per-hand, postflop action stats once. Resolve our pending raise as nonfold when the immediate next opponent response is observed as call/raise; as fold if an observed fold is present (typically terminal). Never count our own actions as opponent actions. Advance cursor to the consumed length.
5. **Before returning our own raise**, store one pending response with hand/street/history-index/actual offered total and effective size. A later own raise must not overwrite an unresolved old response without a diagnostic and a safe conservative resolution (no fabricated reply).
6. `on_hand_end` **transaction protocol:** validate entire primitive result (including valid hand index) and determine any unambiguous pending-response resolution **in a scratch delta**. If the result uniquely matches the pending response with no observed immediate reply, infer fold only if no showdown and `winner==me` and the pending opponent response could be the only terminal action. Infer nonfold from showdown only when uniquely attributable to that immediate response; otherwise infer nothing. Only after all validation and scratch calculations succeed, atomically apply counter deltas, update the pending state, and set `processed_result[index]=true`. If validation fails, do **not** set its bit, update any counter, or partially resolve pending state. Never reconstruct full VPIP from a result without observations.
7. All three hooks must contain their own safe error paths: `new_half` initializes a complete conservative memory structure even if optional configuration is malformed; `choose_action` protects the entire policy and returns a legal fallback when the legal set is recoverable; `on_hand_end` uses the transaction above and catches errors without mutating counters. `new_half` must not claim success if the referee supplies an actually unusable required configuration; such a case is a P0 protocol failure in test. Avoid unhandled exceptions; tests must reveal any swallowed internal failures, rather than treating fallback as good strategy. No diagnostic collection may exceed 32 entries ×160 characters.

### 10.4 Smoothed labels and conservative exploitation

Preserve v0.2 priors as provisional: `observed_vpip_hat=(vpip+12*.55)/(observed_first_actions+12)`; `pre_raise_hat=(observed_pre_raise_hands+12*.30)/(observed_first_actions+12)`; `post_aggr_hat=(post_raises+12*.25)/(post_nonfold_actions+12)`; `fold_hat_pre=(pre_folds+12*.32)/(resolved_pre+12)`; `fold_hat_post=(post_folds+12*.32)/(resolved_post+12)`.

Evaluate labels in this fixed order:

1. `OVERFOLDER`: at least 12 resolved postflop raise responses and `fold_hat_post>=.50`.
2. `STATION`: at least 20 **observed first actions**, `observed_vpip_hat>=.55`, ≥10 resolved postflop responses, and `fold_hat_post<=.25`.
3. `AGGRESSIVE`: ≥16 observed postflop nonfold actions with `post_aggr_hat>=.42`, **or** ≥20 observed first preflop actions with `pre_raise_hat>=.50`.
4. `TIGHT`: at least **30** observed first preflop actions, `observed_vpip_hat<=.27`, and `pre_raise_hat<=.30`. This is an **observed-only** proxy, not proof of true all-hand tightness; its sole v0.4 call-price effect is −.04.
5. Else `UNKNOWN`.

A preflop overfolder can widen our opening bands only when ≥12 specifically resolved preflop responses and `fold_hat_pre>=.52`. An `OVERFOLDER` label from **postflop** data must not alone establish a preflop overfolder. The basic button opening adjustment must therefore check **STATION first, then independently verified preflop overfolder, then default**. Postflop bluff gates use the postflop OVERFOLDER label. All exploit shifts remain `[TUNE]`; disable as one named experimental ablation, not ad hoc.

## 11. Reliability, performance, and configuration freeze

- **Critical correctness:** legal action every time, no unhandled hook exception, no time-bank exhaustion, no forfeits, no network/file writes outside allowed paths, stable CPU-only startup; keep the agent's existing web-server plumbing intact. If memory/config corruption is detected mid-hand, use the legal fallback rather than reconstruct hidden information.
- **FULL upper workload:** 378+1,404+3,250 = **5,032** opponent-score table probes per hand through river, or **5,032,000** across 1,000 such hands, *excluding* hero scoring, draws, action updates, HTTP, and multi-request latency.
- **REDUCED upper workload:** 64×(1+4+10)=**960** opponent-score table probes per hand through river, or **960,000** across 1,000 such hands, excluding the same overhead. Its approximation can miss important beating pairs, which is why it never asserts current nuts.
- **P0 targets (unmeasured):** ≥1 full 1,000-deal duplicate match per official baseline on two **stress-only** seeds, zero errors, zero illegal actions, zero exhaustion/forfeits, per-half charged time `<=5.5s`, startup `<=12s`, maximum HTTP request `<=100ms` in the local restricted container. Run the same stress set against a deliberately call-heavy opponent (frequent rivers) and an aggressive 4-raise-cap opponent, or justify a stronger reachable-state stress generator. A microbenchmark is never a substitute.
- **Mode sequence:** E0 **runtime only** checks both modes; E1 tests accuracy/calibration for each E0-passing mode; **then** freeze FULL if both gates pass, else REDUCED if both pass, else stop. Record evaluator/source/config/sampler hashes, 0 salt, Python runtime, and complete pass reports. No E2–E4 experiment may begin while mode status is unresolved. Do not switch at runtime or mix data across modes as if they tested one policy.
- **Failure policy:** expected malformed optional data is skipped in `new_half`/`on_hand_end`; legal fallback in `choose_action`; malformed `legal_actions` is a failing harness test. No strategy decision depends on parsing debug strings or on absent cross-half state.

## 12. Evaluation and acceptance policy

All exact test cases, reproducible setups, and pass/fail criteria are in `CRITERIA.md`; all tuning/ablation/holdout protocols are in `EXPERIMENTS.md`. Both are normative.

**Order:** validate official evaluator and reachability; reproduce preflop structural counts; independently verify two-card / five-card regression; E0 benchmark both modes in the full referee; E1 calibrate **runtime-eligible** modes; only **then** choose and freeze mode; evaluate a *frozen no-adaptation/no-pure-bluff* baseline; test one major feature at a time **only on development seeds**; lock a candidate; run **untouched** final seeds exactly once; report results without using those results to choose new constants. If the final set exposes failure, revert to the prior development-selected candidate and gather **new independent final seeds** for any revised contender. Never reuse a disclosed final holdout as a new tuning set.

**Feature promotion on development set `[RESEARCH POLICY]`:** require all P0 gates, ≥0.5 more cumulative match point across four official baselines × two development seeds, ≥+.02 average paired bb/deal, and no archetype regression ≥1 match point across its two development seeds. These are practical screening thresholds, **not statistical proof**. Compare paired-difference distributions and uncertainty; noisy or ambiguous evidence favors the simpler baseline. Final holdout is an **audit**, not another feature-selection filter.

## 13. Explicitly deferred work and known strategic risks

- Full opponent five-card ranges conditional on betting, exact future showdown equity, and exact flop/turn LOCKED_NUTS testing remain offline research, not expensive online policy requirements.
- The 80-state-per-street pilot provides descriptive calibration only. A threatening fragment frequency can translate to a substantially higher five-card risk, and even uniform five-card risk does not equal range-conditioned showdown equity.
- Relative tiers, call-price ceilings, opening score, size policy, draw-to-call interpolation, and opponent priors all remain provisional. Freeze the basic preflop thresholds, relative-tier boundaries, call bands, and value-bet size as the **only initial strategic tuning knobs**; do not simultaneously optimize small position bonuses or specialized river-bluff constants.
- Reduced sampling can miss rare threats, so `NUTS` status from it is expressly prohibited. The full evaluator may be too slow; whole-referee measurement decides.
- A hand that is CURRENT_NUTS now may lose on future cards; even a locked final unbeaten hand may split. A full house/flush/straight is not automatically the best relative hand.
- Current policy is deterministic, exploitable by sophisticated opponents, and does not promise GTO performance. Randomized mixing is deferred until measured benefit is clear.
- Opponent actions and card revelations are selection-biased; observed VPIP is not actual VPIP. Lack of observations cannot be treated as evidence of opponent folding or checking.
- The guide's constants can change; revalidate with the exact official commit before shipping. No changes to official referee/game/evaluator to favor this agent.

## 14. v0.2 → v0.3 change summary

1. Preserved independently verified preflop score formula and exhaustive counts, exact `get_table()` evaluator, exact draw *per-route* mathematics, legal action/memory framework, and balanced exploitation architecture.
2. Added `FULL_PAIR_v03` and preapproved `REDUCED_64_v03` (subsequently versioned `_v04` in §2.2), mode-specific claims, no silent in-match fallback, and whole-referee runtime gates.
3. Distinguished fragment risk from exact uniform opponent-five-card current-hand risk; added verified `13/325` vs `17,156/65,780` regression and research-only 240-state pilot; tightened provisional strength tiers from `.06/.22` to `.03/.12`.
4. Distinguished `CURRENT_NUTS`, `VULNERABLE_NUTS`, and mathematically sufficient `LOCKED_NUTS`; current vulnerable nuts no longer automatically receive aggressive pot raises/reraises.
5. Replaced mostly unreachable high call ceilings with bands around actual **25% half-pot** and **33⅓% pot** prices; removed low-impact position and SPR call modifiers, retained only explicit effective context modifiers.
6. Clarified missing preflop observations and observed VPIP selection bias; fixed opening-band adaptation to use **preflop**, not postflop, fold evidence; added optional size-specific response diagnostics.
7. Separated development, stress, independent calibration, and **one-time** final match seeds; fixed promotion on development only and reachable-state protocol testing.
8. Added separate `CRITERIA.md`, `EXPERIMENTS.md`, and `CHANGELOG.md` for an independent Goldfish engineer.

**Approval state:** ready for a focused implementation *as a testable experimental baseline* after review; **not** empirically proven competitive, and cannot be tournament-released until mode/calibration/protocol/reliability gates pass. No implementation code is authorized by this document alone.


## 15. v0.3 → v0.4 targeted decision record

1. **[P0]** Explicit pot-first / half-second / check-or-call-last precedence for overlapping STRONG branches in §§8.1 and 9.1; mode clauses are parenthesized. LOCKED branches similarly specify legality-aware priority. Raise aliases normalize to HALF.
2. **[P0]** Fixed E0 runtime PASS/FAIL → E1 accuracy PASS/FAIL/INCONCLUSIVE → freeze FULL-preferred mode → E2–E4; no contradictory early freeze.
3. **[P0]** Explicit terminal-result identity and atomic dedupe against a fixed per-half index tracker, with bounded conflict diagnostics, plus safe hook semantics.
4. **[P1]** Exact offline pair-graph calibration replaces wasteful repeated official-table evaluations per five-card holding. Adds independently scored reference checks, coverage requirements, and binomial uncertainty reporting.
5. **[P1]** REDUCED uses fully specified Python-version-independent affine sampling with a fixed salt; independent-salt and FULL-vs-REDUCED action disagreements are acceptance diagnostics, not nuts proofs.
6. **[P1]** Raw card IDs are now **reconstructed** for the original 240-row pilot, with the source recipe and checksums; official packed-score cross-validation still **NOT EXECUTED** at document creation. The historical pilot remains provisional.
7. **[P2]** One canonical `DESIGN.md`; external `CRITERIA.md`, `EXPERIMENTS.md`, `CHANGELOG.md`, `RESEARCH_README.md`, `READINESS.md`, and `research/` artifacts reference this exact v0.4.

**No poker-agent implementation has been written or benchmarked.** Any research utility is offline-only, not imported by the submission. Implementation readiness means normative rules are complete; **tournament release is blocked** until the actual P0/P1 tests execute. v0.3 preflop score, draw math, core policy tiers, and opponent-profile constants are intentionally unchanged and remain provisional.
