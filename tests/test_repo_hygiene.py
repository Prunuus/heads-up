from pathlib import Path


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
