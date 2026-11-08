"""Regression tests for HTTP match diagnostics."""


def test_empty_latency_report_is_valid():
    from research.run_matches import latency

    assert latency([]) == {"n": 0, "max_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0}
