import math
import statistics

import pytest

from game import Action
from sim.match import play_match
from sim.opponents import FIELD, HOLDOUT, TRAIN, ZOO
from sim.tournament import (focus_eval, mean_ci, paired_diff, resolve_bot,
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


def test_mean_ci_on_hand_computed_vectors():
    # 1..8: mean 4.5, sample sd sqrt(42/7) = sqrt(6), half-width 1.96 * sqrt(6) / sqrt(8)
    # = 1.96 * sqrt(0.75) = 1.6974097914175
    lo, mid, hi = mean_ci([1, 2, 3, 4, 5, 6, 7, 8])
    assert mid == 4.5
    assert lo == pytest.approx(2.8025902085825, abs=1e-12)
    assert hi == pytest.approx(6.1974097914175, abs=1e-12)
    # -1, 1, -1, 1: mean 0, sd sqrt(4/3), half-width 1.96 * sqrt(4/3) / 2 = 1.1316065276116665
    lo, mid, hi = mean_ci([-1.0, 1.0, -1.0, 1.0])
    assert mid == 0.0
    assert (lo, hi) == (pytest.approx(-1.1316065276116665, abs=1e-12),
                        pytest.approx(1.1316065276116665, abs=1e-12))
    assert mean_ci([3.0]) == (3.0, 3.0, 3.0)                 # one value: no spread
    assert mean_ci([2.0] * 50) == (2.0, 2.0, 2.0)


def test_paired_diff_ci_is_pooled_over_every_deal():
    # The acceptance gates read this interval, so pin it against a recomputation that
    # shares no code with paired_diff: per-deal chips straight from play_match, the
    # stdlib's sample standard deviation, and N = every deal pooled (2 opponents x
    # 2 seeds x 40 deals = 160), not the number of matches.
    opps = {"cs": ZOO["calling_station"], "pot": ZOO["always_pot"]}
    seeds, deals = ["t1", "t2"], 40
    lo, mid, hi = paired_diff(ZOO["nit"], ZOO["threshold_cat"], opps, seeds, deals)

    def per_deal_bb(cand, opp, seed):
        h1, h2 = play_match(cand, opp, seed, deals).halves
        return [(a + b) / 2 for a, b in zip(h1.nets_a, h2.nets_a)]   # chips -> bb

    d = []
    for opp in opps.values():
        for s in seeds:
            d += [x - y for x, y in zip(per_deal_bb(ZOO["nit"], opp, s),
                                        per_deal_bb(ZOO["threshold_cat"], opp, s))]
    assert len(d) == 160
    half = 1.96 * statistics.stdev(d) / math.sqrt(160)
    assert half > 0.05                          # a real spread, so the width check bites
    assert mid == pytest.approx(statistics.fmean(d), abs=1e-12)
    assert hi - lo == pytest.approx(2 * half, rel=1e-9)
    assert (lo, hi) == (pytest.approx(mid - half, abs=1e-12), pytest.approx(mid + half, abs=1e-12))


def test_round_robin_both_forfeit_scores_nobody():
    class Crash:
        def new_half(self, config, memory):
            pass

        def choose_action(self, obs, memory):
            raise RuntimeError("boom")

        def on_hand_end(self, result, memory):
            pass

    st = run_round_robin({"crash_a": Crash, "crash_b": Crash, "cs": ZOO["calling_station"]},
                         ["t1"], 20, workers=1)
    rows = {r.name: r for r in st.rows}
    # crash_a vs crash_b is BOTH_FORFEIT: 0 points each; cs beats both crash bots
    assert {n: r.points for n, r in rows.items()} == {"crash_a": 0.0, "crash_b": 0.0, "cs": 2.0}
    assert st.rows[0].name == "cs"                                   # best first
    assert {n: r.forfeits for n, r in rows.items()} == {"crash_a": 4, "crash_b": 4, "cs": 0}
    assert rows["cs"].losses == 0 and rows["crash_a"].losses >= 1 and rows["crash_b"].losses >= 1
    assert rows["cs"].mean_bb > 0 > rows["crash_a"].mean_bb           # B's side sign-flipped


def test_round_robin_max_bank_is_the_bots_own():
    import time

    class Slow:
        def new_half(self, config, memory):
            pass

        def choose_action(self, obs, memory):
            time.sleep(0.003)                    # 3 ms: 2.25 ms charged after grace
            return Action.CHECK_CALL

        def on_hand_end(self, result, memory):
            pass

    st = run_round_robin({"slow": Slow, "cs": ZOO["calling_station"]}, ["t1"], 6, workers=1)
    rows = {r.name: r for r in st.rows}
    assert rows["slow"].max_bank > 0.01 > rows["cs"].max_bank


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
    assert resolve_bot("agent").args[0] == "agent.py"           # our bot, not a stand-in
    assert callable(resolve_bot("baseline-bots/baseline_random.py"))
    with pytest.raises(ValueError):
        resolve_bot("nonexistent_thing")
    assert HOLDOUT | TRAIN | FIELD == set(ZOO)


_SCALES_SEEN = []


class _ScaleRecorder:
    def new_half(self, config, memory):
        _SCALES_SEEN.append(config.time_scale)

    def choose_action(self, obs, memory):
        return Action.CHECK_CALL

    def on_hand_end(self, result, memory):
        pass


def test_time_scale_reaches_every_match():
    _SCALES_SEEN.clear()
    focus_eval(_ScaleRecorder, {"cs": ZOO["calling_station"]}, ["t1"], 5, workers=1,
               time_scale=2.5)
    paired_diff(_ScaleRecorder, _ScaleRecorder, {"cs": ZOO["calling_station"]}, ["t1"], 5,
                time_scale=2.5)
    run_round_robin({"a": _ScaleRecorder, "b": _ScaleRecorder}, ["t1"], 5, workers=1,
                    time_scale=2.5)
    assert _SCALES_SEEN == [2.5] * 10           # 2 + 4 + 4 halves of our recorder
    _SCALES_SEEN.clear()
    focus_eval(_ScaleRecorder, {"cs": ZOO["calling_station"]}, ["t1"], 5, workers=1)
    assert _SCALES_SEEN == [1.0, 1.0]           # the default is this machine's speed
