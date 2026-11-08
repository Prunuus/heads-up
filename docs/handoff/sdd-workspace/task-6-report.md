# Task 6 report: vectorized evaluator

Status: DONE. Commit: `feat(bot): vectorized numpy hand evaluator` (branch kevin, not pushed).

## Implementation
- `bot/fasteval.py`: key = sum(5**rank) over the 5 cards (counts <= 4, so collision-free; no sort). Flush adds OFF = 5**9. TABLE is int32[2*5**9], filled from `evaluator._score_ranks` over 1278 rank multisets. Table is built by `warmup()` (idempotent, also lazy on first call).
- Omaha: key splits additively into a hole-pair part (N,10) and board-triple part (N,T); one broadcast add (N,10,T), flush offset where the pair suit code equals the triple suit code (4 / 5 are never-matching "mixed" codes), `table.take`, max. `made_now_batch` shares the same code (board width 3, 4, 5; 5 equals `omaha_score`).
- Accepts any int dtype; returns int32. Exports `score5_batch`, `omaha_batch`, `made_now_batch`, `PAIR_IDX` (10,2), `TRIPLE_IDX` (10,3), `warmup`. Imports: numpy, stdlib, evaluator only.

## Numbers
- Table build: 5.4 ms. Memory: 15.6 MB (2*5^9 int32; the brief's 7.8 MB figure was one half).
- `omaha_batch` 10,000 evals: 3.2 ms best/median (threshold 10 ms kept, best-of-5 after warm-up).
- Reference `omaha_score` loop: ~458 ms per 10k, so about 140x faster.
- 200k score5 equality test: 0.53 s. Full default suite passes.

## TDD
RED: `ModuleNotFoundError: No module named 'bot.fasteval'` at collection. GREEN: 7 passed in 1.1 s.
Tests: 200k score5 equality; edge cases (wheel, wheel straight flush, flush > full house, trips > straight, quads); 20k omaha; dtype acceptance; made_now vs brute force for widths 3/4/5; index shapes; speed.

## Files
bot/__init__.py, bot/fasteval.py, tests/test_fasteval.py

## Self-review / concerns
No concerns. Inputs are assumed to be distinct valid cards 0..35 (no validation, for speed). Peak temp memory for omaha at N=10k is ~4 MB x few; for large N (1M) chunk the calls (~400 MB of temporaries per 1M x100).
