import random
import time

import pytest

from bot.timing import TimeManager, grace_for_cpu_factor
from game import Street

LADDER = {0, 16, 32, 64, 128, 256}

# Cost model of one half at dev-laptop speed (measured 2026-11-07).
NOMINAL = 0.2e-3 / 128   # s per rollout: equity_vs_random at 128 iters takes ~0.2 ms per call
BASE = 0.25e-3           # s per decision outside Monte Carlo (features, nut flags, policy); estimate
HAND_END = 20e-6         # s per on_hand_end call; estimate
HTTP = 0.23e-3           # s of server + HTTP overhead per request (measured, plan F3)
GRACE = 0.001            # the judge does not charge the first 1 ms of a request


class FakeClock:
    """A clock that only moves when the simulation advances it."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _pot(rng: random.Random) -> int:
    """Decision-time pot mix measured over 7 zoo halves: 68% <= 8, 20% 9-30, 12% > 30."""
    u = rng.random()
    if u < 0.68:
        return rng.randint(3, 8)
    if u < 0.88:
        return rng.randint(9, 30)
    return rng.randint(31, 160)


def simulate_half(tm, cost_per_iter, rest_slowdown=None, hands=1000, per_hand=4, seed=0):
    """Play one half on tm.clock (a FakeClock). Returns (judge-charged s, [(hand, pot, iters)]).

    A decision costs BASE + iters * cost_per_iter. BASE, HAND_END and the HTTP plumbing
    are slowed down by rest_slowdown, which defaults to the rollout slowdown
    cost_per_iter / NOMINAL (a slower CPU slows everything, request plumbing included).
    Each request is charged as the judge does, max(0, work + http - GRACE), where work is
    the previous hand's on_hand_end (delivered with a hand's first request) plus one
    decision. The judge charge is computed here, independently of tm.used.
    """
    clock = tm.clock
    slow = cost_per_iter / NOMINAL if rest_slowdown is None else rest_slowdown
    base, hand_end, http = BASE * slow, HAND_END * slow, HTTP * slow
    rng = random.Random(seed)
    charged, decisions = 0.0, []
    for h in range(hands):
        pending = 0.0
        if h:
            tm.add(hand_end)
            pending = hand_end
        for d in range(per_hand):
            pot = _pot(rng)
            tm.begin()
            n = tm.iters(pot, Street(d % 4), h)
            work = base + n * cost_per_iter
            clock.advance(work)
            tm.end()
            charged += max(0.0, pending + work + http - GRACE)
            pending = 0.0
            decisions.append((h, pot, n))
    tm.add(hand_end)                                   # the final /hand-end request
    charged += max(0.0, hand_end + http - GRACE)
    return charged, decisions


def _late_big_pot_iters(decisions, hands=1000):
    return [n for h, pot, n in decisions if h >= hands - 100 and pot > 30]


def test_slow_clock_stays_under_bank():
    # Review Focus 3: the tournament CPU is 3x slower than the dev laptop, plumbing included.
    total, decisions = simulate_half(TimeManager(clock=FakeClock()), cost_per_iter=3 * NOMINAL)
    assert total < 8.0
    assert min(_late_big_pot_iters(decisions)) > 0
    # At dev-laptop speed the default grace never under-counts the judge, and big pots
    # late in the half still get rollouts.
    tm = TimeManager(clock=FakeClock())
    total, decisions = simulate_half(tm, cost_per_iter=NOMINAL)
    assert tm.used >= total
    late = _late_big_pot_iters(decisions)
    assert len(late) >= 20 and min(late) > 0


@pytest.mark.parametrize("slow", [2, 3, 5])
def test_cpu_factor_grace_stops_under_counting(slow):
    # Plumbing is CPU-bound too: on a slower CPU the default grace under-counts the judge,
    # while a grace sized for the measured CPU factor does not.
    default = TimeManager(clock=FakeClock())
    total, _ = simulate_half(default, cost_per_iter=slow * NOMINAL)
    assert default.used < total
    tm = TimeManager(clock=FakeClock(), grace_s=grace_for_cpu_factor(slow))
    total, _ = simulate_half(tm, cost_per_iter=slow * NOMINAL)
    assert tm.used >= total
    assert total < 8.0


def test_grace_for_cpu_factor():
    assert grace_for_cpu_factor(1.0) == pytest.approx(0.00075)      # the default grace
    assert grace_for_cpu_factor(3.0) == pytest.approx(0.00025)
    for junk in (float("nan"), None, "x", 0.0, -2.0):               # unmeasured: default
        assert grace_for_cpu_factor(junk) == pytest.approx(0.00075)


def test_pacing_holds_bank_when_rollouts_are_far_slower():
    # Rollouts 50x slower than measured while the rest of a decision runs at laptop speed.
    # A manager with an endless bank and target never throttles: the no-pacing baseline.
    cost = 50 * NOMINAL
    unpaced = TimeManager(bank_s=1e9, target_s=1e9, clock=FakeClock())
    over, _ = simulate_half(unpaced, cost, rest_slowdown=1.0)
    paced, _ = simulate_half(TimeManager(clock=FakeClock()), cost, rest_slowdown=1.0)
    assert over > 8.0
    assert paced < 8.0


def test_big_pot_gets_more_iters_than_small():
    tm = TimeManager(clock=FakeClock())
    for used in (0.0, 2.0, 3.3):                       # behind, on and ahead of a 3 s pace
        tm.used = used
        for street in Street:
            small, medium, big = (tm.iters(p, street, 999) for p in (8, 30, 160))
            assert {small, medium, big} <= LADDER
            assert small < medium < big
    tm.used = 2.0
    assert [tm.iters(p, Street.FLOP, 999) for p in (3, 8, 9, 30, 31, 160)] == [32, 32, 64, 64, 128, 128]


def test_pace_moves_one_level_per_20_percent():
    tm = TimeManager(clock=FakeClock())                # target 3 s over 1000 hands

    def levels(used, hand=999):
        tm.used = used
        return [tm.iters(p, Street.TURN, hand) for p in (5, 20, 100)]

    assert levels(1.0) == [64, 128, 256]               # projected < 1.5 s: one level up
    assert levels(1.5) == [32, 64, 128]                # both boundaries count as on pace
    assert levels(3.0) == [32, 64, 128]
    assert levels(3.3) == [16, 32, 64]                 # 10% over: one level down
    assert levels(3.9) == [0, 16, 32]                  # 30% over: two
    assert levels(4.5) == [0, 0, 16]                   # 50% over: three
    assert levels(5.1) == [0, 0, 0]                    # 70% over: four
    # projected = used / (hand_index + 1) * hands
    assert levels(0.3, hand=999) == [64, 128, 256]     # 0.3 s
    assert levels(0.3, hand=99) == [32, 64, 128]       # 3.0 s
    assert levels(0.3, hand=49) == [0, 0, 0]           # 6.0 s


def test_zero_iters_after_6s():
    clock = FakeClock()
    tm = TimeManager(target_s=20.0, clock=clock)       # generous pace: only the hard rule can stop
    tm.begin()
    clock.advance(5.9)
    tm.end()
    assert tm.iters(160, Street.RIVER, 999) > 0
    tm.begin()
    clock.advance(0.2)
    tm.end()
    assert tm.used > 6.0
    for street in Street:
        for pot in (3, 8, 30, 160):
            for hand in (0, 500, 999, 100_000):
                assert tm.iters(pot, street, hand) == 0


def test_hard_stop_scales_with_retuned_bank():
    tm = TimeManager(bank_s=4.0, target_s=20.0, clock=FakeClock())   # stop at 3.0 s
    tm.used = 2.99
    assert tm.iters(160, Street.RIVER, 999) > 0
    tm.used = 3.01
    assert tm.iters(160, Street.RIVER, 999) == 0


def test_target_scales_with_retuned_bank():
    assert TimeManager().target_s == 3.0
    tm = TimeManager(bank_s=2.0, clock=FakeClock())
    assert tm.target_s == pytest.approx(0.75) and tm.stop_s == pytest.approx(1.5)
    assert tm.target_s < tm.stop_s
    tm.used = 0.5                                      # projected 0.5 s: on a 0.75 s pace
    assert [tm.iters(p, Street.FLOP, 999) for p in (5, 20, 100)] == [32, 64, 128]
    tm.used = 0.85                                     # 13% over 0.75 s: one level down
    assert [tm.iters(p, Street.FLOP, 999) for p in (5, 20, 100)] == [16, 32, 64]


def test_degenerate_config_never_raises():
    # A raise inside new_half forfeits the half, so bad settings fall back instead.
    for bank in (0, -1.0, float("nan"), None, "x"):
        tm = TimeManager(bank_s=bank, clock=FakeClock())
        assert tm.iters(160, Street.RIVER, 0) == 0    # no usable bank: no Monte Carlo
    for hands in (0, -5, float("nan"), None):
        assert TimeManager(hands=hands).hands == 1000
    for target in (0, -1.0, float("nan"), "x"):
        assert TimeManager(target_s=target).target_s == 3.0
    clock = FakeClock()
    tm = TimeManager(grace_s=float("nan"), clock=clock)
    tm.begin()
    clock.advance(0.003)
    tm.end()
    assert tm.used == pytest.approx(0.003 - 0.00075)  # default grace
    tm = TimeManager(bank_s=float("inf"), target_s=1.0)
    tm.used = float("inf")                             # endless overshoot must not overflow
    assert tm.iters(160, Street.RIVER, 999) == 0


def test_begin_end_charges_only_beyond_grace():
    clock = FakeClock()
    tm = TimeManager(clock=clock)
    tm.begin()
    clock.advance(0.0007)
    tm.end()
    assert tm.used == 0.0
    tm.begin()
    clock.advance(0.003)
    tm.end()
    assert tm.used == pytest.approx(0.003 - 0.00075)
    clock.advance(1.0)                                 # time between requests is not ours
    tm.end()                                           # a stray end() charges nothing
    assert tm.used == pytest.approx(0.00225)


def test_on_hand_end_time_counts():
    tm = TimeManager(clock=FakeClock())
    tm.add(0.0004)                                     # below the grace, still charged in full
    assert tm.used == pytest.approx(0.0004)
    tm.add(0.003)
    assert tm.used == pytest.approx(0.0034)
    tm.add(-1.0)
    assert tm.used == pytest.approx(0.0034)


def test_iters_under_5us():
    tm = TimeManager()
    tm.used = 3.3                                      # overshoot branch, the slowest path
    n, best = 20_000, float("inf")
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(n):
            tm.iters(40, Street.RIVER, 999)
        best = min(best, (time.perf_counter() - t) / n)
    assert best < 5e-6
