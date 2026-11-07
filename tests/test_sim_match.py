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
