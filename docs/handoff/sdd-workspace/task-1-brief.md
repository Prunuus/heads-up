### Task 1: Repo hygiene and identity

**Files:** Create `.gitignore`, `.dockerignore`, `requirements-train.txt`. Modify `requirements.txt` (add `numpy`) and `agent.py` (`PARTICIPANT`, `AGENT_NAME`). Test: `tests/test_repo_hygiene.py`.

- [ ] **Step 1: Write the failing test**
```python
def test_dockerignore_excludes_dev_dirs_but_not_bot():
    lines = Path(".dockerignore").read_text().split()
    for d in (".venv", "context", "train_data", "sim", "tests", "train", "docs"):
        assert d in lines or f"{d}/" in lines
    for keep in ("agent.py", "bot", "data", "game.py", "evaluator.py", "requirements.txt"):
        assert keep not in lines and f"{keep}/" not in lines

def test_names_set():
    src = Path("agent.py").read_text()
    assert "TotallyAwesomeDatathonParticipant" not in src and "JustAsAwesomeAgent" not in src

def test_requirements_pins_untouched():
    req = Path("requirements.txt").read_text()
    assert "fastapi==0.156.1" in req and "uvicorn==0.54.0" in req and "numpy" in req
```
- [ ] **Step 2:** `.venv/bin/python -m pytest tests/test_repo_hygiene.py -q` → FAIL.
- [ ] **Step 3:** Create the files. `.gitignore` covers `.venv/`, `__pycache__/`, `train_data/`, `*.json` match logs. Set team and bot names; **ask the user for the team name** (it must be identical on every resubmission).
- [ ] **Step 4:** Run the tests → PASS. Also run `.venv/bin/python -m pytest -q test_agent.py` → PASS.
- [ ] **Step 5:** `git rm -r --cached tools/__pycache__`, then commit: `chore: repo hygiene, dockerignore, team identity`.

