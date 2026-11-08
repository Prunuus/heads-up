"""The dev-only heuristic evaluation bot (sim/heuristic_bot.py)."""

import functools
import pickle

import pytest

import sim.heuristic_bot as hb
from bot.heuristic import HeuristicParams
from game import AgentConfig, make_deal, Game
from sim.match import bot_file_factory, play_half, play_match
from sim.opponents import ZOO

BASELINES = ("random", "calling_station", "always_half", "always_pot")


def test_factory_is_picklable_and_carries_params():
    p = HeuristicParams(bluff_freq=0.0)
    fac = pickle.loads(pickle.dumps(functools.partial(hb.HeuristicBot, p)))
    assert fac().params == p
    assert pickle.loads(pickle.dumps(hb.HeuristicBot)) is hb.HeuristicBot
    assert hb.HeuristicBot().params == HeuristicParams()


def test_plays_a_half_without_errors():
    h = play_half(functools.partial(hb.HeuristicBot, HeuristicParams()), ZOO["random"],
                  "hb1", 1, deals=100)
    assert h.errors_a == 0 and not h.forfeit_a and h.bank_a < 1.0


def test_bot_file_interface_plays():
    h = play_half(bot_file_factory("sim/heuristic_bot.py"), ZOO["calling_station"],
                  "hb2", 2, deals=60)
    assert h.errors_a == 0 and not h.forfeit_a


def test_time_manager_charged_even_when_a_decision_raises(monkeypatch):
    bot, memory = hb.HeuristicBot(), {}
    bot.new_half(AgentConfig(), memory)
    g = Game(make_deal("tm", 0), 0)
    obs = g.observation(g.to_act)
    assert bot.choose_action(obs, memory) in obs.legal_actions
    tm = memory["tm"]
    assert tm._t0 is None

    def boom(*a, **k):
        raise RuntimeError("feature failure")

    monkeypatch.setattr(hb, "compute_features", boom)
    with pytest.raises(RuntimeError):
        bot.choose_action(obs, memory)
    assert tm._t0 is None                     # end() ran in the finally block


def test_make_params_overrides():
    p = hb.make_params({"bluff_freq": 0.0, "value_raise": [0.6, 0.7, 0.8]})
    assert p.bluff_freq == 0.0 and p.value_raise == (0.6, 0.7, 0.8)
    assert p.open_min == HeuristicParams().open_min


@pytest.mark.slow
@pytest.mark.parametrize("name", BASELINES)
def test_beats_each_baseline_over_1000_deals(name):
    m = play_match(functools.partial(hb.HeuristicBot, HeuristicParams()), ZOO[name], "v1", 1000)
    assert m.verdict == "A", (name, m.mean_bb, m.ci_low, m.ci_high)
    assert all(h.errors_a == 0 and h.bank_a < 3.0 for h in m.halves)
