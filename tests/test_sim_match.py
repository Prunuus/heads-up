import pytest

from game import Action
from sim.match import (HalfStats, MatchStats, bot_file_factory, load_bot_file,
                       play_half, play_match)


class _Crash:
    def new_half(self, config, memory):
        pass

    def choose_action(self, obs, memory):
        raise RuntimeError("boom")

    def on_hand_end(self, result, memory):
        pass


class _Call:
    def new_half(self, config, memory):
        pass

    def choose_action(self, obs, memory):
        return Action.CHECK_CALL

    def on_hand_end(self, result, memory):
        pass


class _HandEndCrash(_Call):
    def on_hand_end(self, result, memory):
        raise RuntimeError("boom")


def crash_factory():
    return _Crash()


def call_factory():
    return _Call()


def handend_crash_factory():
    return _HandEndCrash()


CALLER = "baseline-bots/baseline_calling_station.py"


@pytest.mark.slow
def test_matches_real_judge_value():  # F10
    m = play_match(bot_file_factory("sim/opponents/starter.py"),
                   bot_file_factory(CALLER), seed="1")
    assert m.mean_bb == pytest.approx(-0.516, abs=5e-4) and m.verdict == "B"


def test_identical_bots_chop():
    f = bot_file_factory(CALLER)
    m = play_match(f, f, seed="s1", deals=200)
    assert m.mean_bb == 0 and m.verdict == "CHOP" and m.points_a == 0.5


def test_load_bot_file_fresh_globals():
    a, b = load_bot_file(CALLER), load_bot_file(CALLER)
    assert a is not b and a.__name__ != b.__name__


def test_crashing_bot_forfeits_and_loses():
    m = play_match(crash_factory, call_factory, seed="s1", deals=50)
    assert m.halves[0].forfeit_a and m.points_a == 0 and m.verdict == "B"
    assert isinstance(m, MatchStats) and isinstance(m.halves[0], HalfStats)


def test_both_forfeit():
    m = play_match(crash_factory, crash_factory, seed="s1", deals=50)
    assert m.verdict == "BOTH_FORFEIT" and m.points_a == 0


def test_on_hand_end_errors_count_individually():
    h = play_half(handend_crash_factory, call_factory, "s1", 1, deals=10)
    assert h.errors_a == 5 and h.forfeit_a


def test_slow_bot_error_and_bank():
    import time

    class Slow(_Call):
        def choose_action(self, obs, memory):
            time.sleep(0.02)
            return Action.CHECK_CALL

    h = play_half(Slow, call_factory, "s1", 1, deals=3)
    assert h.bank_a > 0 and h.calls_a > 0 and h.max_call_a >= 0.02
    assert h.errors_a == 0


def test_illegal_or_bad_return_is_error():
    class Bad(_Call):
        def choose_action(self, obs, memory):
            return "nonsense"

    h = play_half(Bad, call_factory, "s1", 1, deals=10)
    assert h.errors_a >= 5 and h.forfeit_a


import time as _time


class _Sleepy(_Call):
    def __init__(self, delay, action=Action.CHECK_CALL):
        self.delay, self.action = delay, action

    def choose_action(self, obs, memory):
        _time.sleep(self.delay)
        return Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else self.action


def test_over_one_second_request_is_error_and_discarded():
    # action timeout shrunk to 20 ms; bot returns RAISE_POT after 40 ms
    h = play_half(lambda: _Sleepy(0.04), call_factory, "s1", 1, deals=2,
                  action_timeout_s=0.02)
    assert h.errors_a >= 1 and h.calls_a >= 1


def test_discarded_action_is_default_action():
    seen = []

    class Rec(_Call):
        def choose_action(self, obs, memory):
            _time.sleep(0.03)
            return Action.RAISE_POT

    class Opp(_Call):
        def choose_action(self, obs, memory):
            seen.append(obs.opp_bet)
            return Action.CHECK_CALL

    play_half(Rec, Opp, "s1", 1, deals=1, action_timeout_s=0.01)
    # A (seat 0) never raised successfully, so B never faced a raise from A on any street
    assert all(x <= 2 for x in seen)


def test_bank_exhaustion_is_not_an_error():
    h = play_half(lambda: _Sleepy(0.02), call_factory, "s1", 1, deals=30,
                  time_bank_s=0.1, action_timeout_s=1.0)
    assert h.errors_a == 0 and not h.forfeit_a
    assert h.bank_a >= 0.1 and h.calls_a < 30 * 3  # bot stopped being called


def test_overrun_of_remaining_bank_is_bankout_not_error():
    # bank 0.05 s; one request of 0.2 s exceeds the remaining bank (< timeout 1.0)
    h = play_half(lambda: _Sleepy(0.2), call_factory, "s1", 1, deals=5,
                  time_bank_s=0.05)
    assert h.errors_a == 0 and not h.forfeit_a
    assert h.calls_a == 2  # new_half + the one overrunning act; never called again


def test_hand_results_delivered_before_next_action_and_flushed():
    log = []

    class Rec(_Call):
        def new_half(self, config, memory):
            memory["n"] = 0

        def choose_action(self, obs, memory):
            log.append(("act", obs.hand_index))
            return Action.CHECK_CALL

        def on_hand_end(self, result, memory):
            log.append(("end", result.hand_index))

    deals = 4
    play_half(Rec, call_factory, "s1", 1, deals=deals)
    ends = [i for e, i in log if e == "end"]
    assert ends == list(range(deals))
    for k in range(deals - 1):
        assert log.index(("end", k)) < min(
            j for j, (e, i) in enumerate(log) if e == "act" and i == k + 1)
    # last hand has no later action: it can only arrive via the final flush
    assert log[-1] == ("end", deals - 1)
    assert not any(e == "act" and i == deals - 1 and j > log.index(("end", deals - 1))
                   for j, (e, i) in enumerate(log))


def test_flush_is_one_request():
    class Rec(_Call):
        def on_hand_end(self, result, memory):
            raise RuntimeError("x")

    # all hands delivered in few requests; calls_a counts requests (new_half + acts + flush)
    h = play_half(Rec, call_factory, "s1", 1, deals=1)
    assert h.errors_a >= 1
