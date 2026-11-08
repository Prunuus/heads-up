"""Betting and chip-invariant regressions using official game states."""
from dataclasses import replace

import pytest

from game import Action, AgentConfig, Game, make_deal
from poker_memory import initialize
from poker_policy import choose, raise_valid


def fresh():
    memory = {}
    initialize(AgentConfig(), memory, diagnostics=True)
    return memory


@pytest.mark.parametrize("change", [
    {"pot": 4}, {"my_stack": 100}, {"big_blind": True},
    {"big_blind": 4}, {"is_button": 1}, {"min_raise_to": 2},
])
def test_inconsistent_observation_quantities_use_a_safe_fallback(change):
    observation = Game(make_deal("audit-quantities", 0), 0).observation(0)
    memory = fresh()
    assert choose(replace(observation, **change), memory) == Action.FOLD
    assert memory["internal_errors"] == 1
    assert memory["hand"] == -1


@pytest.mark.parametrize("total", [2, 3, 7, 80])
def test_raise_guard_checks_increment_minimum_and_pot_limit(total):
    observation = Game(make_deal("audit-raises", 0), 0).observation(0)
    # Normal legal opening totals are 4 and 6, not a call, an underraise,
    # or a wager larger than the pot-limit maximum.
    bad = replace(observation, raise_to={Action.RAISE_HALF: total})
    assert not raise_valid(bad, Action.RAISE_HALF)


def test_raise_guard_allows_a_short_all_in_below_the_normal_minimum():
    game = Game(make_deal("audit-short-raise", 0), 0)
    while not game.is_terminal:
        observation = game.observation(game.to_act)
        if (Action.RAISE_POT in observation.legal_actions
                and observation.raise_to[Action.RAISE_POT] < observation.min_raise_to):
            assert raise_valid(observation, Action.RAISE_POT)
            return
        game.apply(Action.RAISE_POT if Action.RAISE_POT in observation.legal_actions
                   else Action.CHECK_CALL)
    pytest.fail("fixture failed to reach a short all-in")
