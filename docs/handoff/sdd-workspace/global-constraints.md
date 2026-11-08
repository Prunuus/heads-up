# Global Constraints (verbatim from plan) + controller rulings that bind all tasks

- Deliverable entry point: `agent.py` in repo root. Edit only between `# ===== YOUR CODE HERE =====` and `# ===== END OF YOUR CODE =====`, plus the `warmup()` area. Never touch SERVER PLUMBING or the last lines that read `PORT`.
- `requirements.txt` in repo root; `fastapi==0.156.1` and `uvicorn==0.54.0` must not change; add `numpy` without a version (already in base image).
- Game constants (from `game.py`): `BIG_BLIND = 2`, `SMALL_BLIND = 1`, `STARTING_STACK = 80`, `RAISE_CAP_PER_STREET = 4`, `NUM_DEALS = 1000`.
- Time: `TIME_BANK_SECONDS = 8.0` per half, `NETWORK_GRACE_SECONDS = 0.001` free per request, `ACTION_TIMEOUT_SECONDS = 1.0` per request, `STARTUP_TIMEOUT_SECONDS = 15.0`.
- `ERROR_LIMIT = 5` errors in a half → forfeit → **match loss**. Every `on_hand_end` exception counts separately.
- Verdict: 95% CI (`CHOP_Z = 1.96`) on paired bb/deal; win 1 / chop 0.5 / loss 0; tie-break = average bb/deal.
- Container: 1 CPU, 4 GB RAM, no network, read-only files except `/tmp` (512 MB), non-root user, ≤ 3 GB added image size, ≤ 30 min build, fresh process every half, `OMP_NUM_THREADS=1`.
- Model/data files committed normally, each < 100 MB (no Git LFS).
- Engineering budgets: median `/act` compute ≤ 0.7 ms, p99 ≤ 10 ms, max single request ≤ 100 ms, total bank use ≤ 3.0 s per half on a dev laptop, startup ≤ 6 s.
- Tournament runs Python 3.12; local `.venv` is Python 3.14 → no 3.13+-only syntax/APIs. Always use `.venv/bin/python`.
- Do NOT modify `game.py`, `evaluator.py`, `judge_engine.py`, `baseline-bots/`, `docker/`, `tools/` (organizer files).
- Runtime code (`bot/`, `agent.py`) may import only stdlib, numpy, and the repo's `game.py`/`evaluator.py`. No numba/torch at runtime.
- Slow tests (≥1 full sim match, tuning, training) are marked `@pytest.mark.slow` (excluded by default via root `pytest.ini`; run with `-m slow`).
- Factories handed to multiprocessing must be picklable (top-level classes or `functools.partial` of top-level functions — no lambdas).
- Never `git push`. Commit on your track branch only (kevin / kevin-cfr / kevin-opp / kevin-lbr).
- Commit messages: ONE simple sentence (no body, no trailers, never a Co-Authored-By line) — user directive 2026-11-07, overrides the plan's commit wording (ruling R36).
