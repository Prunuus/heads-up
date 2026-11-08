"""Regression tests for HTTP match diagnostics."""
from types import SimpleNamespace

import pytest


def test_empty_latency_report_is_valid():
    from research.run_matches import latency

    assert latency([]) == {"n": 0, "max_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0}


def test_healthy_endpoint_cannot_hide_a_failed_child(monkeypatch, tmp_path):
    from research import run_matches as runner

    child = SimpleNamespace(poll=lambda: 1, terminate=lambda: None, wait=lambda timeout: 1)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: child)
    # An existing server can answer health checks even though the newly
    # launched process failed (for example, because its port was occupied).
    monkeypatch.setattr(runner, "MeasuredHandle", lambda *args, **kwargs:
                        SimpleNamespace(wait_healthy=lambda timeout: True, _conn=None))
    monkeypatch.setattr(runner, "play_half", lambda *args, **kwargs:
                        pytest.fail("referee must not use a server from a failed launch"))
    with pytest.raises(RuntimeError, match="exited"):
        runner.run_half(runner.FULL, "baseline_random", "launch-failure", 1,
                        tmp_path, True, False, 1000)


def test_validator_rejects_an_image_with_different_bot_sources(monkeypatch, tmp_path):
    from tools import validate_bot as validator

    source, image = tmp_path / "source", tmp_path / "image"
    source.mkdir()
    image.mkdir()
    for name in validator.FILES:
        (source / name).write_text("current source", encoding="utf8")
        (image / name).write_text("current source", encoding="utf8")
    (image / "poker_memory.py").write_text("old image", encoding="utf8")
    monkeypatch.setattr(validator, "ROOT", source)
    monkeypatch.setattr(validator, "EXECUTION_ROOT", image, raising=False)
    output = tmp_path / "results"
    monkeypatch.setattr(validator.sys, "argv", ["validate_bot.py", "--output", str(output)])
    monkeypatch.setattr(validator, "run_half", lambda *args, **kwargs:
                        pytest.fail("a stale image cannot qualify the current source"))
    with pytest.raises(RuntimeError, match="poker_memory.py"):
        validator.main()
    assert not output.exists()
