from copy import deepcopy
from dataclasses import replace

import pytest
from game import Action, AgentConfig, Game, HandResult, Street, make_deal
from poker_memory import COUNTERS, diagnostic, finish, ingest, initialize, profile, remember_raise
from poker_policy import choose


def fresh(n=1000):
    m = {}
    initialize(AgentConfig(num_hands=n), m, diagnostics=True)
    return m


def terminal(index=0):
    g = Game(make_deal("terminal", index), index % 2, hand_index=index)
    g.apply(Action.FOLD)
    return g.hand_result(0)


def test_atomic_duplicates_out_of_order_and_reset():
    m = fresh()
    for i in (5, 3, 5, 4):
        finish(terminal(i), m)
    assert m["completed"] == 3
    assert sum(m["processed"]) == 3
    before = deepcopy(m)
    finish(terminal(5), m)
    assert m == before
    conflict = replace(terminal(5), pot=terminal(5).pot + 1)
    finish(conflict, m)
    assert m["completed"] == 3
    assert len(m["diagnostics"]) == 1
    finish(conflict, m)
    assert len(m["diagnostics"]) == 1
    initialize(AgentConfig(), m, diagnostics=True)
    assert not any(m["processed"]) and m["completed"] == 0


@pytest.mark.parametrize("change", [{"hand_index": -1}, {"hand_index": 1000}, {"hand_index": True},
                                    {"winner": "invalid"}, {"pot": -1}, {"pot": "10"},
                                    {"board": (0, 0, 0)}, {"went_to_showdown": True},
                                    {"opp_hole": (1, 2, 3, 4, 5)}, {"my_net": 999}])
def test_malformed_terminal_is_transactional(change):
    m = fresh()
    before = {k: deepcopy(m[k]) for k in (*COUNTERS, "pending", "processed", "fingerprints")}
    finish(replace(terminal(), **change), m)
    assert {k: m[k] for k in before} == before
    assert m["internal_errors"] == 1


def test_unobserved_opponent_fold_is_not_vpip():
    g = Game(make_deal("unseen", 0), 0)
    g.apply(Action.FOLD)
    m = fresh()
    finish(g.hand_result(1), m)
    assert m["completed"] == 1
    assert m["first_observed"] == m["vpip"] == m["resolved_pre"] == 0


def test_history_idempotence_and_nonfold_response():
    g = Game(make_deal("history", 0), 0)
    g.apply(Action.RAISE_HALF)
    m = fresh()
    obs = g.observation(1)
    ingest(obs, m)
    assert m["first_observed"] == m["vpip"] == m["pre_raise_hands"] == 1
    before = deepcopy(m)
    ingest(obs, m)
    assert m == before
    remember_raise(obs, Action.RAISE_HALF, m)
    g.apply(Action.RAISE_HALF)
    g.apply(Action.CHECK_CALL)
    obs2 = g.observation(1)
    ingest(obs2, m)
    assert m["resolved_pre"] == 1 and m["pre_folds"] == 0
    g.apply(Action.RAISE_HALF)
    g.apply(Action.FOLD)
    finish(g.hand_result(1), m)
    assert m["resolved_pre"] == 1 and m["pre_folds"] == 0


def test_direct_terminal_fold_after_raise():
    g = Game(make_deal("pending-fold", 0), 0)
    m = fresh()
    o = g.observation(0)
    ingest(o, m)
    remember_raise(o, Action.RAISE_POT, m)
    g.apply(Action.RAISE_POT)
    g.apply(Action.FOLD)
    finish(g.hand_result(0), m)
    assert m["resolved_pre"] == m["pre_folds"] == 1
    finish(g.hand_result(0), m)
    assert m["resolved_pre"] == m["pre_folds"] == 1


def test_showdown_response_no_fold_or_invented_vpip():
    g = Game(make_deal("pending-allin", 0), 0)
    while not g.is_terminal:
        o = g.observation(g.to_act)
        if Action.RAISE_POT in o.legal_actions and o.raise_to[Action.RAISE_POT] - o.my_bet == o.my_stack:
            seat = g.to_act
            m = fresh()
            ingest(o, m)
            remember_raise(o, Action.RAISE_POT, m)
            old = m["first_observed"]
            g.apply(Action.RAISE_POT)
            g.apply(Action.CHECK_CALL)
            assert g.is_terminal
            finish(g.hand_result(seat), m)
            assert m["resolved_pre"] + m["resolved_post"] == 1
            assert m["pre_folds"] + m["post_folds"] == 0
            assert m["first_observed"] == old
            return
        g.apply(Action.RAISE_POT if Action.RAISE_POT in o.legal_actions else Action.CHECK_CALL)
    pytest.fail("all-in fixture did not reach all-in raise")


def test_history_shrink_and_backwards_are_safe():
    g = Game(make_deal("shrink", 0), 0)
    g.apply(Action.CHECK_CALL)
    o = g.observation(1)
    m = fresh()
    ingest(o, m)
    before = {k: m[k] for k in COUNTERS}
    with pytest.raises(ValueError, match="history"):
        ingest(replace(o, action_history=()), m)
    assert {k: m[k] for k in COUNTERS} == before
    assert choose(replace(o, action_history=()), m) == Action.CHECK_CALL


def test_diagnostics_and_invalid_config():
    m = fresh()
    for _ in range(100):
        diagnostic(m, "x" * 1000)
    assert len(m["diagnostics"]) == 32
    assert all(len(x) == 160 for x in m["diagnostics"])
    initialize(AgentConfig(num_hands=-1), m, diagnostics=True)
    assert not m["config_valid"] and m["processed"] == bytearray(1000)
    initialize(AgentConfig(big_blind=4, small_blind=2, starting_stack=160, num_hands=7), m)
    assert m["config_valid"] and len(m["processed"]) == 7


def test_smoothed_profile_order_and_observed_denominator():
    m = fresh()
    m.update(completed=1000)
    assert profile(m)[0] == "UNKNOWN"
    m.update(first_observed=40, vpip=0, pre_raise_hands=0)
    assert profile(m)[0] == "TIGHT"
    m.update(post_nonfold=30, post_raises=30)
    assert profile(m)[0] == "AGGRESSIVE"
    m.update(resolved_post=30, post_folds=30)
    assert profile(m)[0] == "OVERFOLDER"
