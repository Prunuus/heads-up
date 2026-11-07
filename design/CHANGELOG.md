# CHANGELOG.md — Elephant Poker Bot Specification

## v0.4 (2026-11-07): focused corrections from v0.3 senior engineering review

**Scope:** Final targeted documentation/research-cleanup pass. One canonical `DESIGN.md` replaces the former duplicate. All preflop scoring coefficients, short-deck rules, exact draw formulas, core relative-tier boundaries, bluff policy, and baseline opponent-prior constants are **preserved** as unvalidated v0.3 strategy hypotheses.

| Review issue | Exact v0.4 revision | Status |
|---|---|---|
| Overlapping pot/half branches | §§8.1/9.1 `STRONG` now prioritize fully-qualified legal POT → HALF → CHECK; LOCKED branches have explicit mode-independent first-match order; alias canonicalization remains | **Specification fixed; action tests not run** |
| Early evaluator-mode freeze | Mode-selection state machine now E0 runtime eligibility → E1 accuracy/coverage eligibility → FULL-preferred freeze → E2–E4 tuning; no dynamic switching | **Specification fixed; E0/E1 not run** |
| Missing raw cards in 240-row pilot | Added `research/pilot_raw_reconstructed_v04.csv` with actual card IDs reconstructed via original seed 8037 and state-order recipe; included original CSV and original independent scorer for provenance | **Legacy hero/fragment counts cross-checked (240/240); official verifier not run** |
| Unreproducible rank checks | `research/calibrate_offline.py verify` imports the **pinned** official `evaluator.py`, checks packed scores, graph counts, tie-only counts and separate slow reference; writes per-state machine-checkable report | **Research utility authored; official verification not run** |
| Excessive offline 5-card scoring | Exact BEAT/TIE edge graph over unseen two-card fragments; enumerate each five-card holding using ten edge-membership checks rather than reevaluating 10× board triples | **Algorithm specified; performance unmeasured** |
| REDUCED sampled-tier risk | Added version-independent, one-hash affine sampler (`REDUCED_64_v04`, frozen salt 0) and six deterministic index fixtures; compare FULL/REDUCED tiers/actions and diagnostic salts | **Fixtures locally verified; classification accuracy not measured** |
| Weak calibration strata | Coverage minimums for street/tier/texture/boundaries, Wilson intervals, upward/downward tier errors, action divergence; INCONCLUSIVE state explicit | **Acceptance contract defined; fresh states not evaluated** |
| Terminal-result duplicates | Per-half fixed processed-result indexed bitset, transactional mutations on `on_hand_end`, explicit conflicting duplicates and malformed payload handling | **Contract defined; bot tests not run** |
| Confused status and packaging | `RESEARCH_README.md`, `READINESS.md`, SHA-256 manifest; single `DESIGN.md`; separate verified/preliminary/unexecuted information | **Documentation updated** |

## Files and source of truth

- **Authoritative:** `DESIGN.md` v0.4.
- **Normative acceptance tests:** `CRITERIA.md` v0.4.
- **Research/test sequencing:** `EXPERIMENTS.md` v0.4.
- **Research provenance:** `RESEARCH_README.md`, `research/calibrate_offline.py`, raw CSVs, fixtures, historical CSV/code, `MANIFEST.json`.
- **Status/implementation gate:** `READINESS.md`.
- **Referee/evaluator authority:** [`tamu-datathon-org/heads-up-bundle`](https://github.com/tamu-datathon-org/heads-up-bundle) pinned for this design at `c83ab6eeb7660a2fd1fae7df224315d41fe985a9` (must recheck upstream before release).

## Preserved legacy fixes

v0.1→v0.2 removed constant suitedness bonus, specified current-board 2+3 lookup scoring, introduced relative strength and contextual betting/draws/memory, enumerated all 376,992 possible starting holdings. v0.2→v0.3 corrected fragment versus five-card threat, introduced full/reduced evaluator architecture, vulnerable/locked nuts, the 25% half-pot and 33⅓% pot-call prices, safer adaptation, and separated development/holdout/stress seed families.

## Unvalidated assumptions and deferred work

- No official-table 240-row pilot cross-validation has been run with this version; the legacy reconstruction has **only** been checked using the prior independent research scorer.
- No bot implementation, official-judge match, full time-bank measurement, or empirical strategy win rate exists in this delivery. Offline calibration `fresh_stratified_inputs_v04.csv` contains generated test cards, **not** evaluated outcomes.
- Relative strength boundaries `.03/.12`, preflop cutoffs `20/40/60`, call ceilings, SPR/investment caps, opponent priors, and raise sizing are still `[TUNE]` parameters. Do not claim optimized policy or GTO status.
- Future-runout exact opponent equity, online Monte Carlo, deep opponent-range inference, CFR, neural policies, automatic algorithm switching and adaptive sampler-salt selection are expressly deferred.

**Disposition:** v0.4 resolves the known P0 *documentation ambiguities*. Goldfish implementation handoff remains a **separate user decision**. Tournament deployment remains BLOCKED pending executed P0 and P1 gates.
