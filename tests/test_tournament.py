import pytest

from sim.opponents import HOLDOUT, TRAIN, ZOO
from sim.tournament import (focus_eval, paired_diff, resolve_bot,
                            run_round_robin)


def test_round_robin_points_sum():
    s = run_round_robin({k: ZOO[k] for k in ("random", "calling_station", "nit")}, ["t1"], 100, workers=1)
    assert sum(r.points for r in s.rows) == pytest.approx(3.0)   # 3 matches, 1 point each


def test_paired_diff_of_identical_is_zero():
    lo, mid, hi = paired_diff(ZOO["nit"], ZOO["nit"], {"cs": ZOO["calling_station"]}, ["t1"], 100)
    assert lo == mid == hi == 0


def test_paired_diff_orders_and_signs():
    lo, mid, hi = paired_diff(ZOO["maniac"], ZOO["random"], {"cs": ZOO["calling_station"]}, ["t1"], 100)
    assert lo <= mid <= hi
    lo2, mid2, hi2 = paired_diff(ZOO["random"], ZOO["maniac"], {"cs": ZOO["calling_station"]}, ["t1"], 100)
    assert mid2 == pytest.approx(-mid) and lo2 == pytest.approx(-hi)


def test_focus_eval_report():
    r = focus_eval(ZOO["nit"], {"cs": ZOO["calling_station"]}, ["t1", "t2"], 50, workers=1)
    assert set(r.matches["cs"]) == {"t1", "t2"}
    pts = [m.points_a for m in r.matches["cs"].values()]
    assert r.expected_points == pytest.approx(sum(pts) / 2)


def test_standings_row_fields_and_parallel_matches_serial():
    ents = {k: ZOO[k] for k in ("random", "calling_station", "nit")}
    s1 = run_round_robin(ents, ["t1"], 40, workers=1)
    s2 = run_round_robin(ents, ["t1"], 40, workers=2)
    assert {r.name for r in s1.rows} == set(ents)
    assert [(r.name, r.points, round(r.mean_bb, 9)) for r in s1.rows] == \
           [(r.name, r.points, round(r.mean_bb, 9)) for r in s2.rows]
    r = s1.rows[0]
    assert (r.losses, r.forfeits) == (r.losses, r.forfeits) and r.max_bank >= 0


def test_resolve_bot():
    assert resolve_bot("nit") is ZOO["nit"]
    assert callable(resolve_bot("agent"))
    assert callable(resolve_bot("baseline-bots/baseline_random.py"))
    with pytest.raises(ValueError):
        resolve_bot("nonexistent_thing")
    assert HOLDOUT | TRAIN == set(ZOO)
