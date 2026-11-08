# Task 1 Report: Repo hygiene and identity

Status: DONE_WITH_CONCERNS (see Concerns)
Commit: `8617b76` chore: repo hygiene, dockerignore, team identity (branch `kevin`, not pushed)

## What was implemented

- `.gitignore` (new): `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `train_data/`, `.superpowers/`, and `/*.json` (root-only match logs, so `data/*.json` stays tracked).
- `.dockerignore` (new): one entry per line: `.git`, `.venv`, `__pycache__`, `.pytest_cache`, `.superpowers`, `context`, `train_data`, `sim`, `tests`, `train`, `docs`. Does not exclude `agent.py`, `bot`, `data`, `game.py`, `evaluator.py`, or `requirements.txt`.
- `requirements.txt`: appended `numpy` (unpinned). The `fastapi==0.156.1` and `uvicorn==0.54.0` pins are unchanged.
- `requirements-train.txt` (new): `-r requirements.txt` plus unpinned `numba`, offline only.
- `pytest.ini` (new, repo root): registers the `slow` marker and sets `addopts = -m "not slow"`. `testpaths` is unset, so both root `test_agent.py` and `tests/` are collected.
- `agent.py`: `PARTICIPANT = "KChen"`, `AGENT_NAME = "ShortDeckHybrid"` (controller decision; the brief's "ask the user" was overridden).
- `tests/test_repo_hygiene.py` (new): the three tests from the brief, verbatim.
- `git rm -r --cached tools/__pycache__`: removed the committed `tools/__pycache__/smoke_test.cpython-312-pytest-9.2.0.pyc` from the index. The organizer source `tools/smoke_test.py` and `tools/check_build.sh` are untouched.

Not touched: `context/`, `docs/`, `game.py`, `evaluator.py`, `judge_engine.py`, `baseline-bots/`, `docker/`, and the `tools/` source files.

## TDD evidence

RED (`.venv/bin/python -m pytest tests/test_repo_hygiene.py -q`, before implementation):
```
FAILED tests/test_repo_hygiene.py::test_dockerignore_excludes_dev_dirs_but_not_bot
FAILED tests/test_repo_hygiene.py::test_names_set - assert ('TotallyAwes...nP...
FAILED tests/test_repo_hygiene.py::test_requirements_pins_untouched - Asserti...
3 failed in 0.02s
```

GREEN (same command, after implementation):
```
3 passed in 0.00s
```

Full default suite (`.venv/bin/python -m pytest -q`):
```
15 passed, 1 warning in 0.87s
```
The 15 are the 3 hygiene tests plus 12 from root `test_agent.py`. The one warning is a pre-existing starlette/httpx deprecation from `fastapi.testclient`, not from this change.

Root test file alone (`.venv/bin/python -m pytest -q test_agent.py`): `12 passed, 1 warning in 0.80s`.

Ignore-rule checks (`git check-ignore -v`): `train_data/x.bin`, `.venv/x`, `__pycache__/a.pyc`, `tools/__pycache__/x.pyc`, `.superpowers/a`, and root `foo.json` are ignored. `data/match.json` and `sub/foo.json` are not ignored.

## Files changed (commit 8617b76)

- `.dockerignore` (new)
- `.gitignore` (new)
- `agent.py` (2 lines: identity)
- `pytest.ini` (new)
- `requirements-train.txt` (new)
- `requirements.txt` (+1 line: `numpy`)
- `tests/test_repo_hygiene.py` (new)
- `tools/__pycache__/smoke_test.cpython-312-pytest-9.2.0.pyc` (deleted from index)

## Self-review findings

- The diff is limited to the files in the brief. The `.dockerignore` entries are exact bare names, matching the `.split()` token check.
- `requirements.txt` keeps the original two pins byte-for-byte and adds one line.
- The `.gitignore` `/*.json` rule is anchored to the root, as specified.
- The untracked root `__pycache__/` and `tools/__pycache__/...cpython-314...pyc` are now ignored by `.gitignore`. `context/` remains untracked and was intentionally not committed.
- No `slow` tests exist yet, so the `-m "not slow"` default has no effect on the current run. It will matter once later tasks add `@pytest.mark.slow` tests.

## Concerns

1. Commit trailer conflict. The global constraints say `Co-Authored-By: Claude Opus 5.5`, but the attribution reminder in this session says `Claude Haiku 5.5`. I used the Opus 5.5 trailer, since the project's global constraints are the user's own written instructions for this plan and take precedence over the default. Amend the trailer if the other value is wanted.
2. Team name. The brief said to ask the user. Per the controller decision I used `KChen`, and the global constraints require the team name to stay identical across resubmissions. Confirm `KChen` is the final name before submission.
3. `context/` is still untracked. It was not part of this task, so I left it alone, but it will show up in `git status` until someone decides whether to commit or ignore it.
