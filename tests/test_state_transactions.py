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


@pytest.mark.parametrize("button_acts", [False, True])
@pytest.mark.parametrize("executed", [Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT])
def test_terminal_river_response_requires_the_proposed_wager(button_acts, executed):
    game = flop()
    while game.street < Street.RIVER:
        game.apply(Action.CHECK_CALL)
    if button_acts:
        game.apply(Action.CHECK_CALL)
    seat = game.to_act
    observation = game.observation(seat)
    memory = fresh()
    ingest(observation, memory)
    remember_raise(observation, Action.RAISE_POT, memory)
    # A timeout may replace our proposed bet with a check. A changed/retried
    # decision may also execute a smaller wager. Neither confirms this proposal.
    game.apply(executed)
    while not game.is_terminal:
        game.apply(Action.CHECK_CALL)
    finish(game.hand_result(seat), memory)
    assert memory["completed"] == 1
    assert memory["resolved_post"] == int(executed == Action.RAISE_POT)
    assert memory["post_folds"] == 0
    assert memory["pending"] is None
    assert memory["internal_errors"] == 0


def test_proposed_all_in_replaced_by_a_call_is_not_a_raise_response():
    game = flop()
    while not game.is_terminal:
        observation = game.observation(game.to_act)
        if (Action.RAISE_POT in observation.legal_actions
                and observation.raise_to[Action.RAISE_POT] - observation.my_bet
                == observation.my_stack):
            seat = game.to_act
            memory = fresh()
            ingest(observation, memory)
            remember_raise(observation, Action.RAISE_POT, memory)
            game.apply(Action.CHECK_CALL)
            while not game.is_terminal:
                game.apply(Action.CHECK_CALL)
            finish(game.hand_result(seat), memory)
            assert memory["completed"] == 1
            assert memory["resolved_pre"] + memory["resolved_post"] == 0
            assert memory["pending"] is None
            assert memory["internal_errors"] == 0
            return
        game.apply(Action.RAISE_POT if Action.RAISE_POT in observation.legal_actions
                   else Action.CHECK_CALL)
    pytest.fail("fixture did not reach an offered all-in raise")


@pytest.mark.parametrize("change", [
    {"pot": 3, "my_net": 1},
    {"my_net": 1},
])
def test_terminal_payout_cannot_erase_observed_contributions(change):
    game = flop()
    seat = game.to_act
    memory = fresh()
    ingest(game.observation(seat), memory)
    game.apply(Action.RAISE_POT)
    game.apply(Action.FOLD)
    result = game.hand_result(seat)
    # Both players already invested two chips on the observed flop. These
    # otherwise plausible payouts reduce the opponent's investment to one.
    before = deepcopy(memory)
    finish(replace(result, **change), memory)
    assert memory["completed"] == 0
    assert memory["processed"] == before["processed"]
    assert memory["fingerprints"] == before["fingerprints"]
    finish(result, memory)
    assert memory["completed"] == 1
    assert memory["internal_errors"] == 1


def test_showdown_pot_cannot_shrink_below_the_observed_pot():
    # Increase the visible investment using a real raise and call, then
    # submit an otherwise consistent payout with a smaller final pot.
    game = flop()
    game.apply(Action.RAISE_POT)
    game.apply(Action.CHECK_CALL)
    seat = game.to_act
    memory = fresh()
    ingest(game.observation(seat), memory)
    while not game.is_terminal:
        game.apply(Action.CHECK_CALL)
    result = game.hand_result(seat)
    bad = replace(result, pot=4, my_net={"me": 2, "opp": -2, "split": 0}[result.winner])
    finish(bad, memory)
    assert memory["completed"] == 0
    finish(result, memory)
    assert memory["completed"] == 1
