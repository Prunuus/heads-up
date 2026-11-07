# RESEARCH_README.md — Research provenance and regeneration

This package is **Elephant-phase documentation**. `research/calibrate_offline.py` is an offline research verification utility, **not agent implementation**. No tournament strategy code, server endpoint, or deployment feature is created here.

## Data and verification status

| Artifact | Meaning | Current status |
|---|---|---|
| `research/legacy_street_calibration_samples_v03.csv` | Original 240-row v0.3 pilot (80 states/street), independent legacy scores, **without** hole/board inputs | Historical / **preliminary** |
| `research/legacy_independent_scorer_v03.py` | Original independent scoring and generation code used for pilot | Historical, not authoritative referee |
| `research/pilot_raw_reconstructed_v04.csv` | Recoverable original 240 card-input rows using documented seed and generator, with copied historical outputs | **Reconstructed**, not yet official-verified |
| `research/fresh_stratified_inputs_v04.csv` | 240 new reproducible raw-card states (20 per street × ordinary/paired/trips/suited), generated with SHA-256 ranked cards | **Inputs only**; not evaluated, no PASS claim |
| `research/sampler_fixtures_v04.json` | Six fixed 64-pair index results, flop/turn/river, diagnostic salts 0 and 1 | **Locally verified** against research sampler algorithm |
| `research/calibrate_offline.py` | Reconstruct, generate raw states, check sampler fixtures, cross-validate via official evaluator and independent slow reference | Script authored; official `verify` **not executed** |

### Raw card format

Card IDs are physical integers 0–35. Rank is `card >> 2` (six=0 through ace=8); suit is `card & 3` (clubs=0, diamonds=1, hearts=2, spades=3). `hole_ids` and `board_ids` are comma-separated decimal byte-card IDs in recorded order. Five hero cards precede 3/4/5 visible board cards; no duplicate cards within a state.

### Historical reconstruction procedure

The v0.3 research source iterated street lengths `(3,4,5)` in order, then row IDs `0..79` for each street, advancing one `random.Random(8037)` instance with `rng.sample(range(36),5+board_length)`. The first five sampled IDs are hero cards, the remainder are board cards in visible order. The reconstructed file *keeps* the reported independent pilot scores/counts in `*_legacy` columns and labels every row `reconstructed_from_seed_8037_not_official_verified`.

Using a Python 3.13.5 research environment, the recovered inputs matched **all 240 legacy hero tuple scores** and **all 240 legacy beating-pair counts** when evaluated with `legacy_independent_scorer_v03.py`. These are **same-method consistency checks**, not cross-validation with the official referee's packed integer scores or exact five-card counts. Old counts cannot be promoted to authoritative until official `verify` succeeds.

### Reproduction commands

Use an actual local clone of the pinned official repository. This utility refuses to label a clone authoritative if `HEAD` is not the exact pinned commit or if `evaluator.py` has uncommitted changes.

```bash
git clone https://github.com/tamu-datathon-org/heads-up-bundle.git
cd heads-up-bundle
git checkout c83ab6eeb7660a2fd1fae7df224315d41fe985a9

# Relative path assumes the v0.4 package is next to the clone.
python ../poker_bot_elephant_v0_4/research/calibrate_offline.py recover \
  --legacy ../poker_bot_elephant_v0_4/research/legacy_street_calibration_samples_v03.csv \
  --output ../poker_bot_elephant_v0_4/research/pilot_raw_reconstructed_v04.csv

python ../poker_bot_elephant_v0_4/research/calibrate_offline.py check-fixtures \
  --path ../poker_bot_elephant_v0_4/research/sampler_fixtures_v04.json

# OFFICIAL checks; have NOT been run in this document creation.
python ../poker_bot_elephant_v0_4/research/calibrate_offline.py verify \
  --repo . \
  --input ../poker_bot_elephant_v0_4/research/pilot_raw_reconstructed_v04.csv \
  --output ../poker_bot_elephant_v0_4/research/pilot_official_verification_v04.csv \
  --slow 3

python ../poker_bot_elephant_v0_4/research/calibrate_offline.py verify \
  --repo . \
  --input ../poker_bot_elephant_v0_4/research/fresh_stratified_inputs_v04.csv \
  --output ../poker_bot_elephant_v0_4/research/fresh_official_verification_v04.csv \
  --slow 3
```

The `verify` action imports *the pinned official `evaluator.py`*, checks full packed hero scores, enumerates full unseen two-card BEAT and TIE fragment graphs, and counts all five-card opponent holdings by edge membership. On one state per street it also computes the **slower independent direct official-table five-card reference**, requiring exact equality. It outputs a report for every state with numerical results, sample-risk tiers and any disagreement. It fails for any mismatched old score/count. It does **not** execute the poker bot or benchmark inference latency.

For new raw states on another seed (inputs only, not matches), use:

```bash
python ../poker_bot_elephant_v0_4/research/calibrate_offline.py generate \
  --seed threat-cal-31 --per-stratum 20 \
  --output ../poker_bot_elephant_v0_4/research/fresh_cal31_inputs.csv
```

Fresh generation ranks cards by `SHA256(b"PB4/CAL\0" + UTF8(seed) + bytes([board_len]) + ASCII(stratum) + trial_index_u32_be + card_id_byte)`, takes the first 5+board_len cards, and rejects trials until the desired board-texture stratum is reached. This is a **version-independent** deterministic raw-card generator. It does not guarantee sufficient STRONG/MEDIUM/WEAK or threshold-boundary coverage; E1 must deliberately supplement strata as needed and mark insufficient coverage INCONCLUSIVE.

## Checksums and official-version control

`MANIFEST.json` contains SHA-256 hashes of all deliverable files other than itself. The expected checksums for the legacy CSV, reconstructed card states, fresh inputs, and sampler fixtures are included. For the **official validation output**, there is deliberately **no expected checksum** until an official pinned checkout has actually run the verifier; record and freeze it after successful execution. If upstream changes, begin a new explicitly versioned design/benchmark cycle rather than silently changing evaluator behavior.

## What the numbers do and do not mean

Fragment threat is the proportion of unseen **two-card** combinations beating hero now. Uniform five-card threat is the proportion of unseen **five-card opponent holdings** containing at least one beating pair now. Neither is showdown equity on an incomplete board or action-conditioned opponent range equity. An approximation that samples zero beating fragments does not prove nuts. The original v0.3 row averages are preliminary and must not certify strategy superiority.
