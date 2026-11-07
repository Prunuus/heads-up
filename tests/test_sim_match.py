import sys

import pytest

from game import Action, AgentConfig
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
STARTER = "sim/opponents/starter.py"


def test_matches_real_judge_value():  # F10; 1000 deals but about 0.3 s, so not slow
    m = play_match(bot_file_factory(STARTER), bot_file_factory(CALLER), seed="1")
    assert m.mean_bb == pytest.approx(-0.516, abs=5e-4) and m.verdict == "B"
    # the organizer's judge on these bots and seed: sd 0.985249, CI (-0.57707, -0.45493)
    assert m.sd_bb == pytest.approx(0.98525, abs=1e-4) and m.points_a == 0
    assert (m.ci_low, m.ci_high) == (pytest.approx(-0.57707, abs=1e-5),
                                     pytest.approx(-0.45493, abs=1e-5))


def test_identical_bots_chop():
    f = bot_file_factory(CALLER)
    m = play_match(f, f, seed="s1", deals=200)
    assert m.mean_bb == 0 and m.verdict == "CHOP" and m.points_a == 0.5
    assert m.sd_bb == 0            # deal d of half 2 is paired with deal d of half 1


def test_load_bot_file_fresh_globals():
    a, b = load_bot_file(CALLER), load_bot_file(CALLER)
    assert a is not b and a.__name__ != b.__name__


# ---- every load re-imports our bot package, like a fresh process (audit I3, ruling R22)

LEAKY = "tests/fixtures/leaky_bot.py"     # keeps a counter in an imported bot.* module


def test_bot_package_state_does_not_survive_load_bot_file():
    seen = []
    for _ in range(2):
        m = load_bot_file(LEAKY)
        memory = {}
        m.new_half(AgentConfig(), memory)
        seen.append(memory["halves_started"])
    assert seen == [1, 1]          # two fresh processes; a shared bot.* module gives [1, 2]


def test_bot_package_state_does_not_cross_halves_or_bots():
    loaded = []

    def factory():
        loaded.append(load_bot_file(LEAKY))
        return loaded[-1]

    play_match(factory, factory, seed="s1", deals=2)
    assert len(loaded) == 4                                   # 2 halves x 2 bots
    assert [m.halves_started() for m in loaded] == [1, 1, 1, 1]
    assert len({id(m.shared) for m in loaded}) == 4           # each bot had its own copy


def test_load_bot_file_leaves_the_callers_modules_alone():
    import bot.timing
    mine = sys.modules["bot.timing"]
    m = load_bot_file(LEAKY)
    assert m.shared is not mine and m.shared.TimeManager is not mine.TimeManager
    assert sys.modules["bot.timing"] is mine      # pickling, monkeypatch, isinstance still work
    # The loaded module, and the bot.* copies it holds, live only as long as the caller
    # keeps the module: a long serial run must not keep one copy (and its tables) per half.
    assert m.__name__ not in sys.modules


def test_fresh_packages_is_a_keyword():
    a = load_bot_file(LEAKY, fresh_packages=())
    b = load_bot_file(LEAKY, fresh_packages=())
    assert a.shared is b.shared                    # nothing evicted, so one shared module
    assert load_bot_file(LEAKY).shared is not a.shared


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


# ---- judge rules the default suite must pin (audit I4)

class _HandEndCrashOn(_Call):
    def __init__(self, hands):
        self.hands = hands

    def on_hand_end(self, result, memory):
        if result.hand_index in self.hands:
            raise RuntimeError("boom")


@pytest.mark.parametrize("half", [1, 2])
def test_each_crashing_hand_result_is_an_error_even_in_one_request(half):
    # The starter folds as button, so in those hands our bot gets no request and two
    # results arrive together (hands 0+1, 2+3, ... in half 1; 1+2, 3+4, ... in half 2).
    # The judge counts every crashing result (judge_engine.py:388-392): three, not two.
    h = play_half(lambda: _HandEndCrashOn({0, 1, 2}), bot_file_factory(STARTER), "s1", half,
                  deals=6)
    assert h.errors_a == 3 and not h.forfeit_a


def test_on_hand_end_time_is_charged_to_the_bank():
    class BusyHandEnd(_Call):
        def on_hand_end(self, result, memory):
            t = _time.perf_counter()
            while _time.perf_counter() - t < 0.005:     # 5 ms of work per finished hand
                pass

    h = play_half(BusyHandEnd, call_factory, "s1", 1, deals=10)
    # ten results, each delivered inside a timed request: about 10 x (5 - 0.75) ms
    assert h.bank_a > 0.03


def test_grace_is_free_time_in_every_request():
    class Wait10ms(_Call):
        def choose_action(self, obs, memory):
            _time.sleep(0.01)
            return Action.CHECK_CALL

    h = play_half(Wait10ms, call_factory, "s1", 1, deals=4, grace_s=0.05)
    assert h.calls_a > 4 and h.max_call_a >= 0.01
    assert h.bank_a == 0.0          # every request stayed inside the 50 ms grace


def test_illegal_action_is_an_error_and_the_default_is_played():
    class FoldWhenFree(_Call):
        def choose_action(self, obs, memory):
            if obs.to_call == 0 and not memory.get("folded"):
                memory["folded"] = True
                return Action.FOLD          # a valid Action, but illegal with nothing to call
            return Action.CHECK_CALL

    h = play_half(FoldWhenFree, call_factory, "s1", 1, deals=3)
    assert h.errors_a == 1 and not h.forfeit_a
    assert "FOLD" not in h.actions_a        # the judge plays the default: a free check


class _NewHalfCrash(_Call):
    def new_half(self, config, memory):
        raise RuntimeError("boom")


def test_new_half_failure_forfeits_the_match():
    m = play_match(_NewHalfCrash, call_factory, seed="s1", deals=5)
    assert m.halves[0].forfeit_a and m.halves[1].forfeit_a
    assert m.points_a == 0 and m.verdict == "B"


def test_new_half_past_the_timeout_forfeits():
    class SlowStart(_Call):
        def new_half(self, config, memory):
            _time.sleep(0.05)

    h = play_half(SlowStart, call_factory, "s1", 1, deals=3, action_timeout_s=0.02)
    assert h.forfeit_a and h.errors_a == 1


def test_action_counts_calling_station():
    h = play_half(call_factory, call_factory, "s1", 1, deals=20)
    assert set(h.actions_a) == {"CHECK_CALL"} and set(h.actions_b) == {"CHECK_CALL"}
    assert h.actions_a["CHECK_CALL"] > 0 and not h.bank_exhausted_a
    assert h.ctx_a.get("faced_fold", 0) == 0 and h.ctx_a["pre"] > 0
