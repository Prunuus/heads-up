"""sim/gate_v3.py: the v3 gate script runs, caches every job and resumes from the cache."""

import functools
import pickle

from sim import gate_v3


def test_gate_bots_resolve_and_pickle():
    for label in ("v3", "v2", "v3_p0.05", "v3_post", "v3_nofeat"):
        fac = pickle.loads(pickle.dumps(gate_v3.bot(label)))
        assert callable(fac)
    v2 = gate_v3.bot("v2")()
    assert v2.blueprint is None and v2.use_exploits
    assert gate_v3.bot("v3_p0.2")().purify == 0.2
    assert isinstance(gate_v3.bot("v3"), functools.partial)     # agent.py, as shipped


def test_gate_runs_resumes_and_summarizes(tmp_path, capsys, monkeypatch):
    argv = ["--out", str(tmp_path), "--sections", "ae", "--h2h-seeds", "1", "--deals", "6",
            "--sweep-seeds", "1", "--sweep-deals", "4", "--purify", "0.1", "--workers", "1"]
    monkeypatch.setattr(gate_v3, "splits", lambda: {
        "holdout": ["threshold_cat"], "field": [], "train": ["calling_station"]})
    assert gate_v3.main(argv) == 0
    first = capsys.readouterr().out
    cached = sorted(p.name for p in tmp_path.glob("[ae]__*.json"))
    assert len(cached) == 3                                     # a: 1 match; e: v2 + 1 purify
    assert "(a) v3 vs v2 head-to-head" in first and "(e) purify sweep" in first
    assert "(c) errors and bank" in first and "v3: 0 errors" in first
    assert gate_v3.main(argv) == 0                              # resumed: nothing re-run
    second = capsys.readouterr().out
    assert "job(s) to run" not in second
    assert sorted(p.name for p in tmp_path.glob("[ae]__*.json")) == cached
    assert (tmp_path / "summary.txt").read_text().startswith("=== v3 gate summary")


def test_gate_lbr_section_is_chunked_and_resumes(tmp_path, capsys):
    argv = ["--out", str(tmp_path), "--sections", "d", "--lbr-seeds", "2", "--lbr-deals", "2",
            "--lbr-particles", "60", "--lbr-workers", "1"]
    code = gate_v3.main(argv)
    first = capsys.readouterr().out
    assert code in (0, 2)                                       # 2: a model check failed
    assert len(list(tmp_path.glob("d__*.json"))) == 4           # 2 targets x 2 seed groups
    assert "LBR(v3) =" in first and "LBR(v2) =" in first and "paired on common deals" in first
    assert gate_v3.main(argv) == code                           # resumed from the cache
    assert "LBR vs v3: seeds" not in capsys.readouterr().out


def test_gate_refuses_a_cache_from_another_blueprint(tmp_path, capsys):
    (tmp_path / "blueprint.json").write_text(
        '{"iterations": -1, "seed": 0, "buckets_sha256": "x"}')
    assert gate_v3.main(["--out", str(tmp_path), "--sections", "a", "--summary-only"]) == 3
    assert "refusing" in capsys.readouterr().out


def test_gate_refuses_without_blueprint(tmp_path, capsys, monkeypatch):
    from types import SimpleNamespace
    from bot import policy
    monkeypatch.setattr(policy, "warmup", lambda: None)
    monkeypatch.setattr(policy, "WARMUP", SimpleNamespace(blueprint=None, failed=0))
    argv = ["--out", str(tmp_path), "--sections", "a"]
    assert gate_v3.main(argv) == 4
    assert "no blueprint loaded" in capsys.readouterr().err
    # the escape gets past the check (summary-only section c runs nothing)
    assert gate_v3.main(["--out", str(tmp_path), "--sections", "c", "--summary-only"]) == 0
    monkeypatch.setattr(gate_v3, "jobs_a", lambda a: (_ for _ in ()).throw(RuntimeError("past"))) 
    try:
        gate_v3.main(argv + ["--allow-no-blueprint"])
    except RuntimeError as e:
        assert str(e) == "past"
    else:
        raise AssertionError("did not proceed")
