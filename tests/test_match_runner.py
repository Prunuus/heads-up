"""Regression tests for HTTP match diagnostics."""
from types import SimpleNamespace
import json

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


@pytest.mark.parametrize("changed", ["poker_memory.py", "baseline-bots/baseline_random.py"])
def test_validator_rejects_an_image_with_different_bot_sources(monkeypatch, tmp_path, changed):
    from tools import validate_bot as validator

    source, image = tmp_path / "source", tmp_path / "image"
    source.mkdir()
    image.mkdir()
    for name in validator.FILES:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (image / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text("current source", encoding="utf8")
        (image / name).write_text("current source", encoding="utf8")
    for name in validator.HARNESS_FILES:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text("current harness", encoding="utf8")
    (image / changed).write_text("old image", encoding="utf8")
    monkeypatch.setattr(validator, "ROOT", source)
    monkeypatch.setattr(validator, "EXECUTION_ROOT", image, raising=False)
    output = tmp_path / "results"
    monkeypatch.setattr(validator.sys, "argv", ["validate_bot.py", "--output", str(output)])
    monkeypatch.setattr(validator, "run_half", lambda *args, **kwargs:
                        pytest.fail("a stale image cannot qualify the current source"))
    with pytest.raises(RuntimeError, match=changed):
        validator.main()
    assert not output.exists()


@pytest.mark.parametrize("fault", ["errors", "forfeited", "timed_out", "skipped_calls"])
def test_failed_opponent_cannot_qualify_a_match(fault):
    from judge_engine import AgentStats
    from research.run_matches import runtime_ok

    own, opponent = AgentStats(calls=2), AgentStats(calls=2)
    setattr(opponent, fault, True if fault in ("forfeited", "timed_out") else 1)
    result = SimpleNamespace(stats_agent1=own, stats_agent2=opponent,
                             num_deals=1000)
    metrics = {"startup_seconds": 1, "candidate_latency": {"n": 2, "max_ms": 2},
               "memory": {"internal_errors": 0, "completed": 1000}}
    assert not runtime_ok(result, metrics, 1000)


@pytest.mark.parametrize("completed", [0, 999, None])
def test_missing_terminal_delivery_cannot_qualify_a_match(completed):
    from judge_engine import AgentStats
    from research.run_matches import runtime_ok

    result = SimpleNamespace(stats_agent1=AgentStats(calls=2),
                             stats_agent2=AgentStats(calls=2), num_deals=1000)
    metrics = {"startup_seconds": 1, "candidate_latency": {"n": 2, "max_ms": 2},
               "memory": {"internal_errors": 0, "completed": completed}}
    assert not runtime_ok(result, metrics, 1000)


def test_clean_complete_match_can_qualify():
    from judge_engine import AgentStats
    from research.run_matches import runtime_ok

    result = SimpleNamespace(stats_agent1=AgentStats(calls=2),
                             stats_agent2=AgentStats(calls=2), num_deals=1000)
    metrics = {"startup_seconds": 1, "candidate_latency": {"n": 2, "max_ms": 2},
               "memory": {"internal_errors": 0, "completed": 1000}}
    assert runtime_ok(result, metrics, 1000)


@pytest.fixture(params=["run_matches", "v05_http"])
def qualification_cli(request, monkeypatch, tmp_path):
    from research import run_matches, v05_http

    module = {"run_matches": run_matches, "v05_http": v05_http}[request.param]
    output = tmp_path / "run"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "FILES", ())
    monkeypatch.setattr(module, "OFFICIAL", ("baseline_random",))
    if module is run_matches:
        arguments = ["--stage", "E0", "--mode", run_matches.FULL,
                     "--opponent", "baseline_random"]
    else:
        (tmp_path / "experiment_plan.json").write_text(
            json.dumps({"development_seeds": ["runner-regression"]}), encoding="utf8")
        arguments = ["--stage", "dev", "--strategy", "v04"]
    monkeypatch.setattr(module.sys, "argv", [module.__name__, "--output", str(output)] + arguments)
    return module, output


def test_failed_launch_still_reserves_the_run(qualification_cli, monkeypatch):
    module, output = qualification_cli

    def failed_launch(*args, **kwargs):
        assert (output / "registry.json").exists()
        raise RuntimeError("simulated startup failure")

    monkeypatch.setattr(module, "run_half", failed_launch)
    with pytest.raises(RuntimeError, match="simulated startup failure"):
        module.main()
    with pytest.raises(RuntimeError, match="overwrite"):
        module.main()


def test_failed_runtime_gate_returns_a_failure_exit_status(qualification_cli, monkeypatch):
    module, _ = qualification_cli
    half = SimpleNamespace(stats_agent1=SimpleNamespace(time_seconds=0))
    metrics = {"candidate_latency": {"max_ms": 2}}
    monkeypatch.setattr(module, "run_half", lambda *a, **kw: (half, metrics, False))
    monkeypatch.setattr(module, "combine_halves", lambda *a: SimpleNamespace(
        to_dict=lambda: {"verdict": "CHOP", "mean_bb_per_deal": 0, "ci": [0, 0]}))
    assert module.main() == 1
