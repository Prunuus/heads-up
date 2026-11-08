"""Observation and result transaction regressions using official game states."""
from copy import deepcopy
from dataclasses import replace

import pytest

from game import Action, AgentConfig, Game, ObservedAction, Street, make_deal
from poker_memory import finish, ingest, initialize, remember_raise
from poker_policy import choose


def fresh():
    memory = {}
    initialize(AgentConfig(), memory, diagnostics=True)
    return memory


def flop():
    game = Game(make_deal("audit-regressions", 0), 0)
    game.apply(Action.CHECK_CALL)
    game.apply(Action.CHECK_CALL)
    return game


def test_rejected_new_hand_does_not_reset_the_current_hand():
    memory = fresh()
    game = flop()
    ingest(game.observation(game.to_act), memory)
    before = deepcopy(memory)
    next_game = Game(make_deal("audit-regressions", 1), 1, hand_index=1)
    observation = next_game.observation(next_game.to_act)
    bad = replace(observation, action_history=(
        ObservedAction(Street.PREFLOP, False, Action.RAISE_POT, -1, 6),
    ))
    with pytest.raises(ValueError):
        ingest(bad, memory)
    assert memory == before


def test_changed_board_does_not_commit_opponent_statistics():
    game = flop()
    memory = fresh()
    observation = game.observation(game.to_act)
    choose(observation, memory)
    game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_HALF)
    observation = game.observation(game.to_act)
    replacement = next(c for c in range(36)
                       if c not in observation.hole_cards + observation.board)
    bad = replace(observation, board=(replacement,) + observation.board[1:])
    before = deepcopy(memory)
    assert choose(bad, memory) == Action.FOLD
    assert memory.pop("internal_errors") == before.pop("internal_errors") + 1
    assert memory.pop("diagnostics")[-1].startswith("choose_action:")
    before.pop("diagnostics")
    assert memory == before


@pytest.mark.parametrize("change", ["street", "position", "future_history"])
def test_inconsistent_history_and_context_are_rejected_atomically(change):
    game = flop()
    memory = fresh()
    observation = game.observation(game.to_act)
    ingest(observation, memory)
    before = deepcopy(memory)
    if change == "street":
        bad = replace(observation, street=Street.PREFLOP, board=())
    elif change == "position":
        bad = replace(observation, is_button=not observation.is_button)
    else:
        bad = replace(observation, action_history=observation.action_history + (
            ObservedAction(Street.RIVER, False, Action.RAISE_POT, 4, 4),
        ))
    with pytest.raises(ValueError):
        ingest(bad, memory)
    assert memory == before


def test_a_judge_substituted_check_does_not_create_a_raise_opportunity():
    game = flop()
    memory = fresh()
    observation = game.observation(game.to_act)
    ingest(observation, memory)
    remember_raise(observation, Action.RAISE_POT, memory)
    # A timed-out raise can be replaced by the judge's free-check default.
    game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_POT)
    ingest(game.observation(game.to_act), memory)
    assert memory["resolved_post"] == 0
    assert memory["pending"] is None
    assert memory["post_raises"] == 1


def test_retried_decision_replaces_a_speculative_raise_with_a_call():
    game = flop()
    memory = fresh()
    observation = game.observation(game.to_act)
    ingest(observation, memory)
    remember_raise(observation, Action.RAISE_POT, memory)
    remember_raise(observation, Action.CHECK_CALL, memory)
    assert memory["pending"] is None


def test_confirmed_raise_still_records_the_opponent_response():
    game = flop()
    memory = fresh()
    observation = game.observation(game.to_act)
    ingest(observation, memory)
    remember_raise(observation, Action.RAISE_POT, memory)
    game.apply(Action.RAISE_POT)
    game.apply(Action.CHECK_CALL)
    ingest(game.observation(game.to_act), memory)
    assert memory["resolved_post"] == 1
    assert memory["pending"] is None


@pytest.mark.parametrize("field,value", [("pot", 1000), ("my_net", 0)])
def test_impossible_fold_result_cannot_poison_terminal_deduplication(field, value):
    game = Game(make_deal("audit-result", 0), 0)
    game.apply(Action.FOLD)
    memory = fresh()
    result = game.hand_result(1)
    finish(replace(result, **{field: value}), memory)
    assert memory["completed"] == 0
    assert not memory["processed"][0]
    finish(result, memory)
    assert memory["completed"] == 1


def test_impossible_showdown_net_cannot_be_recorded():
    game = flop()
    while not game.is_terminal:
        game.apply(Action.CHECK_CALL)
    result = game.hand_result(0)
    memory = fresh()
    bad = replace(result, my_net=1, winner="me")
    finish(bad, memory)
    assert memory["completed"] == 0
    finish(result, memory)
    assert memory["completed"] == 1


def test_completed_hand_observations_cannot_be_counted_again():
    game = Game(make_deal("audit-replayed-hand", 0), 0)
    observation = game.observation(0)
    game.apply(Action.FOLD)
    memory = fresh()
    finish(game.hand_result(0), memory)
    before = deepcopy(memory)
    with pytest.raises(ValueError):
        ingest(observation, memory)
    assert memory == before
