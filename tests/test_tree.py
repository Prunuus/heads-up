"""Exact betting tree (Task 15): F1 counts, agreement with game.Game, terminal payoffs,
history lookup that never raises."""

import dataclasses
import importlib.util
import pickle
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import game
from bot.tree import BettingTree, build_tree, get_tree, node_from_actions, node_of
from game import Action, Game, Observation, ObservedAction, Street, make_deal

F, C, H, P = Action          # FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT
ALL = (F, C, H, P)
ARRAYS = ("street", "actor_is_button", "child", "pot_at", "to_call_at", "street_contrib",
          "term_fold_by_button", "term_contrib")


@pytest.fixture(scope="module")
def tree() -> BettingTree:
    return get_tree()


def n_terminals(t: BettingTree) -> int:
    n = t.term_contrib.shape[0]
    assert t.term_fold_by_button.shape == (n,) and t.n_term == n
    return n


def _policy(rng: random.Random):
    """Random legal actions with per-hand weights, so hands reach rivers, raise wars and all-ins."""
    w = {F: rng.choice((0.05, 0.3, 1.0)), C: 1.0,
         H: rng.choice((0.3, 1.0, 2.0)), P: rng.choice((0.3, 1.0, 2.0))}
    return lambda legal: rng.choices(legal, [w[a] for a in legal])[0]


def _walk(t: BettingTree, key: tuple) -> tuple[str, int]:
    """Follow `key` from the root along child edges: ("node", id) or ("term", tid)."""
    node = 0
    for i, a in enumerate(key):
        code = int(t.child[node, a])
        assert code != -1, f"{key[:i + 1]} is illegal"
        if code <= -2:
            assert i == len(key) - 1, "play continued after a terminal"
            return "term", -2 - code
        node = code
    return "node", node


def _assert_node_matches(t: BettingTree, node: int, obs) -> None:
    """The node's stored state equals what the actor's observation shows."""
    assert t.legal[node] == obs.legal_actions                  # same actions, same order
    assert t.street[node] == obs.street
    assert bool(t.actor_is_button[node]) == obs.is_button
    assert t.pot_at[node] == obs.pot and t.to_call_at[node] == obs.to_call
    me = 0 if obs.is_button else 1                              # column 0 = button
    assert t.street_contrib[node, me] == obs.my_bet
    assert t.street_contrib[node, 1 - me] == obs.opp_bet


def _button_net(flag: int, contrib: tuple, winner_seat, button: int) -> int:
    """The button's chips won, read off a terminal the way the trainer will (Task 18)."""
    bc, bbc = contrib
    if flag == 1:
        return -bc
    if flag == 0:
        return bbc
    if winner_seat is None:
        return 0                   # split: showdown contributions are equal
    return bbc if winner_seat == button else -bc


def _assert_terminal_matches(t: BettingTree, tid: int, g: Game) -> str:
    """Terminal tid holds the finished game's contributions and outcome; returns its kind."""
    btn, res = g.button_seat, g.result()
    contrib = (g.contributed[btn], g.contributed[1 - btn])
    assert tuple(t.term_contrib[tid].tolist()) == contrib
    flag = int(t.term_fold_by_button[tid])
    if res.went_to_showdown:
        assert flag == -1
    else:
        assert flag == int(res.winner_seat != btn)       # the folder is the seat that lost
    assert _button_net(flag, contrib, res.winner_seat, btn) == res.nets[btn]
    if flag == -1:
        return "run-out" if g.history[-1].street < Street.RIVER else "showdown"
    return "button folds" if flag == 1 else "bb folds"


def test_tree_size(tree):
    assert tree.n_dec == 15952 and n_terminals(tree) == 25538              # F1
    assert np.bincount(tree.street, minlength=4).tolist() == [62, 810, 3678, 11402]
    n, m = tree.n_dec, tree.n_term
    assert tree.street.dtype == np.int8 and tree.street.shape == (n,)
    assert tree.actor_is_button.dtype == np.bool_ and tree.actor_is_button.shape == (n,)
    assert tree.child.dtype == np.int32 and tree.child.shape == (n, 4)
    assert tree.term_fold_by_button.dtype == np.int8
    assert tree.term_contrib.dtype == np.int16 and tree.term_contrib.shape == (m, 2)
    assert tree.pot_at.shape == tree.to_call_at.shape == (n,)
    assert tree.street_contrib.shape == (n, 2)
    assert len(tree.legal) == n and len(tree.key_to_node) == n


def test_preflop_nodes(tree):
    assert (tree.street == 0).sum() == 62


# key -> (street, actor is button, legal, pot, to_call, (button, bb) street bets),
# derived by hand from the rules: blinds 1/2, pot raise ladder 6 -> 18 -> 54 -> 80 (F2).
HAND_NODES = [
    ((), (0, True, ALL, 3, 1, (1, 2))),
    ((C,), (0, False, (C, H, P), 4, 0, (2, 2))),            # limp: the BB may check
    ((H,), (0, False, ALL, 6, 2, (4, 2))),                  # half raise = min raise, to 4
    ((P,), (0, False, ALL, 8, 4, (6, 2))),
    ((P, P), (0, True, ALL, 24, 12, (6, 18))),
    ((P, P, P), (0, False, ALL, 72, 36, (54, 18))),
    ((P, P, P, P), (0, True, (F, C), 134, 26, (54, 80))),   # BB all-in: no raise left
    ((C, C), (1, False, (C, H, P), 4, 0, (0, 0))),          # BB acts first postflop
    ((C, C, C), (1, True, (C, H, P), 4, 0, (0, 0))),
    ((C, C, C, C), (2, False, (C, H, P), 4, 0, (0, 0))),
    ((C,) * 6, (3, False, (C, H, P), 4, 0, (0, 0))),
]
# key -> (term_fold_by_button, (button, bb) contributions)
HAND_TERMINALS = [
    ((F,), (1, (1, 2))),
    ((P, F), (0, (6, 2))),
    ((C, P, F), (1, (2, 6))),
    ((P, P, P, P, F), (1, (54, 80))),
    ((P, P, P, P, C), (-1, (80, 80))),       # all-in called preflop: run-out showdown
    ((C,) * 8, (-1, (2, 2))),                # checked down
]


@pytest.mark.parametrize("key,want", HAND_NODES)
def test_hand_checked_nodes(tree, key, want):
    kind, node = _walk(tree, key)
    assert kind == "node"
    street, is_button, legal, pot, to_call, bets = want
    assert (int(tree.street[node]), bool(tree.actor_is_button[node]), tree.legal[node],
            int(tree.pot_at[node]), int(tree.to_call_at[node]),
            tuple(tree.street_contrib[node].tolist())) == (street, is_button, legal, pot,
                                                           to_call, bets)
    assert tree.key_to_node[tuple(int(a) for a in key)] == node


@pytest.mark.parametrize("key,want", HAND_TERMINALS)
def test_hand_checked_terminals(tree, key, want):
    kind, tid = _walk(tree, key)
    assert kind == "term"
    assert (int(tree.term_fold_by_button[tid]), tuple(tree.term_contrib[tid].tolist())) == want


def test_tree_structure_is_consistent(tree):
    child = tree.child
    for node in range(tree.n_dec):                 # child == -1 exactly where illegal
        legal = {int(a) for a in tree.legal[node]}
        assert [int(c) != -1 for c in child[node]] == [a in legal for a in range(4)]
    assert np.array_equal(child[:, F] != -1, tree.to_call_at > 0)     # fold iff facing a bet
    assert (child[child[:, F] != -1, F] <= -2).all()                    # a fold ends the hand
    # Every node but the root, and every terminal, has exactly one parent edge.
    assert np.array_equal(np.sort(child[child >= 0]), np.arange(1, tree.n_dec))
    assert np.array_equal(np.sort(-2 - child[child <= -2]), np.arange(tree.n_term))
    assert tree.key_to_node[()] == 0
    for key, node in tree.key_to_node.items():    # keys agree with edges
        if key:
            assert child[tree.key_to_node[key[:-1]], key[-1]] == node
    # Within a street the actor alternates; a new street starts with the BB.
    par, act = np.nonzero(child >= 0)
    kid = child[par, act]
    same = tree.street[kid] == tree.street[par]
    assert (tree.actor_is_button[kid][same] != tree.actor_is_button[par][same]).all()
    assert (tree.street[kid][~same] == tree.street[par][~same] + 1).all()
    assert not tree.actor_is_button[kid][~same].any()
    # Showdowns put equal chips in; a fold leaves the folder with less in.
    flag, contrib = tree.term_fold_by_button, tree.term_contrib
    assert set(np.unique(flag).tolist()) == {-1, 0, 1}
    assert (contrib[flag == -1, 0] == contrib[flag == -1, 1]).all()
    assert (contrib[flag == 1, 0] < contrib[flag == 1, 1]).all()
    assert (contrib[flag == 0, 0] > contrib[flag == 0, 1]).all()


def test_every_node_and_terminal_matches_game_exhaustively(tree):
    """Walk the whole game with the button in seat 1 (the tree is built with seat 0)."""
    streets, kinds = Counter(), Counter()
    stack = [(Game(make_deal("exhaustive", 0), button_seat=1), 0)]
    while stack:
        g, node = stack.pop()
        obs = g.observation(g.to_act)
        assert node_of(obs, tree) == node
        _assert_node_matches(tree, node, obs)
        streets[int(obs.street)] += 1
        for a in obs.legal_actions:
            c = g.clone()
            c.apply(a)
            code = int(tree.child[node, a])
            if c.is_terminal:
                assert code <= -2
                kinds[_assert_terminal_matches(tree, -2 - code, c)] += 1
            else:
                assert code >= 0
                stack.append((c, code))
    assert [streets[s] for s in range(4)] == [62, 810, 3678, 11402]
    assert sum(kinds.values()) == 25538 and min(kinds.values()) > 0 and len(kinds) == 4


def test_terminal_contributions_match_game(tree):
    """500 random hands with random button seats, replayed down the child edges."""
    rng = random.Random(1515)
    kinds = Counter()
    for i in range(500):
        pick, btn = _policy(rng), rng.randrange(2)
        g = Game(make_deal("terminal", i), button_seat=btn, hand_index=i)
        code = 0
        while not g.is_terminal:
            assert code >= 0
            node = code
            assert bool(tree.actor_is_button[node]) == (g.to_act == btn)
            a = pick(g.legal_actions())
            code = int(tree.child[node, a])
            g.apply(a)
        assert code <= -2
        kinds[_assert_terminal_matches(tree, -2 - code, g)] += 1
    assert min(kinds[k] for k in ("showdown", "run-out", "button folds", "bb folds")) > 0


def test_build_under_2s_pure_and_cached(tree):
    t0 = time.perf_counter()
    fresh = build_tree()
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"build took {elapsed:.2f} s"
    assert fresh is not tree and get_tree() is tree       # build_tree() pure, get_tree() cached
    for name in ARRAYS:
        assert np.array_equal(getattr(fresh, name), getattr(tree, name)), name
    assert fresh.legal == tree.legal and dict(fresh.key_to_node) == dict(tree.key_to_node)


def test_tree_is_read_only_and_picklable(tree):
    assert not any(getattr(tree, name).flags.writeable for name in ARRAYS)
    assert isinstance(tree.legal, tuple)
    with pytest.raises(TypeError):
        tree.key_to_node[("read-only probe",)] = 0
    copy = pickle.loads(pickle.dumps(tree))
    for name in ARRAYS:
        assert np.array_equal(getattr(copy, name), getattr(tree, name)), name
        with pytest.raises(ValueError):
            getattr(copy, name)[0] = 0
    assert copy.legal == tree.legal and dict(copy.key_to_node) == dict(tree.key_to_node)
    with pytest.raises(TypeError):
        copy.key_to_node[("read-only probe",)] = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        copy.n_dec = 0


def test_every_live_obs_maps_to_node_with_same_legal_actions(tree):
    """~2000 decisions of real hands: random legal actions, random button seats."""
    rng = random.Random(15)
    decisions, hand, seen = 0, 0, set()
    while decisions < 2000:
        pick = _policy(rng)
        g = Game(make_deal("live", hand), button_seat=rng.randrange(2), hand_index=hand)
        hand += 1
        while not g.is_terminal:
            obs = g.observation(g.to_act)
            node = node_of(obs, tree)
            assert node is not None, [a.to_dict() for a in obs.action_history]
            assert tree.legal[node] == obs.legal_actions          # same actions, same order
            _assert_node_matches(tree, node, obs)
            key = tuple(int(a.action) for a in obs.action_history)
            assert node_from_actions(key, tree) == node
            assert node_of(Observation.from_dict(obs.to_dict()), tree) == node   # via JSON
            seen.add(node)
            g.apply(pick(obs.legal_actions))
            decisions += 1
    assert np.bincount(tree.street[sorted(seen)], minlength=4).min() > 0    # every street


def _obs_after(actions) -> Observation:
    g = Game(make_deal("unknown", 0), button_seat=0)
    for a in actions:
        g.apply(a)
    return g.observation(g.to_act)


def _with_history(obs: Observation, history) -> Observation:
    return dataclasses.replace(obs, action_history=tuple(history))


def _oa(action, street=Street.PREFLOP) -> ObservedAction:
    return ObservedAction(street, True, action, 0, 0)


def test_unknown_history_returns_none(tree):
    """Review Focus 5: a history outside the tree or malformed input gives None, never raises."""
    flop = _obs_after([C, C])                     # BB to act first on the flop
    assert node_of(flop, tree) == tree.key_to_node[(1, 1)]
    hist = flop.action_history
    outside = [
        hist + (_oa(F, Street.FLOP),),           # folding when nothing is owed
        (_oa(F),),                                # the hand is over: not a decision
        (_oa(F), _oa(C)),                         # play after a fold
        (_oa(H),) * 5,                            # a fifth raise on one street (cap 4)
        (_oa(C),) * 40,                           # longer than any hand
        (_oa(7),), (_oa(-1),), hist + (_oa(4),),  # codes that are not actions
    ]
    for h in outside:
        assert node_of(_with_history(flop, h), tree) is None, h
    for el in (None, 3, "x", SimpleNamespace(), SimpleNamespace(action=None),
               SimpleNamespace(action="CHECK_CALL"), SimpleNamespace(action=[1])):
        assert node_of(_with_history(flop, hist + (el,)), tree) is None, el
        assert node_of(_with_history(flop, (el,)), tree) is None, el
    for bad in (None, object(), flop.to_dict(), SimpleNamespace(action_history=None),
                SimpleNamespace(action_history=5), SimpleNamespace(action_history="11"),
                SimpleNamespace(action_history=())):
        assert node_of(bad, tree) is None, bad
    assert node_of(flop, None) is None


def test_lookalike_action_codes_are_not_actions(tree):
    """Values that int() would turn into CHECK_CALL are malformed, not CHECK_CALL."""
    obs = _obs_after([C, C, C])                   # button to act on the flop
    node = tree.key_to_node[(1, 1, 1)]
    assert node_of(obs, tree) == node
    *head, last = obs.action_history
    for fake in (1.0, True, "1", np.int64(1), "CHECK_CALL"):
        h = (*head, dataclasses.replace(last, action=fake))
        assert node_of(_with_history(obs, h), tree) is None, fake
    for code in (1, Action.CHECK_CALL):          # a plain int code is an action too
        h = (*head, dataclasses.replace(last, action=code))
        assert node_of(_with_history(obs, h), tree) == node


def test_actions_from_another_copy_of_game_py_still_map(tree, monkeypatch):
    """A loader that imports game.py afresh hands us Actions of another class."""
    spec = importlib.util.spec_from_file_location("game_copy", Path(game.__file__))
    other = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "game_copy", other)   # dataclasses look it up
    spec.loader.exec_module(other)
    g = other.Game(other.make_deal("copy", 0), button_seat=1)
    for a in (other.Action.RAISE_POT, other.Action.CHECK_CALL, other.Action.CHECK_CALL):
        g.apply(a)
    obs = g.observation(g.to_act)                 # button to act on the flop
    assert type(obs.action_history[0].action) is not Action
    assert node_of(obs, tree) == tree.key_to_node[(3, 1, 1)]


def test_obs_that_disagrees_with_its_node_returns_none(tree):
    """A known history whose observed state differs from the node's is not that node."""
    flop = _obs_after([C, C])
    for change in (dict(pot=flop.pot + 2), dict(to_call=2), dict(is_button=True),
                   dict(street=Street.TURN), dict(legal_actions=(C, P)),
                   dict(legal_actions=(P, H, C)), dict(my_stack=flop.my_stack + 20)):
        assert node_of(dataclasses.replace(flop, **change), tree) is None, change


RETUNES = [("STARTING_STACK", 60), ("STARTING_STACK", 100), ("BIG_BLIND", 4),
           ("SMALL_BLIND", 2), ("RAISE_CAP_PER_STREET", 3), ("RAISE_CAP_PER_STREET", 5)]


@pytest.mark.parametrize("const,value", RETUNES)
def test_retuned_rules_never_map_to_a_mismatched_node(tree, monkeypatch, const, value):
    """Review Focus 1/5: hands under retuned constants map to None, or to a node whose
    state equals the observation; a retuned stack maps nothing."""
    monkeypatch.setattr(game, const, value)
    rng = random.Random(f"{const}={value}")
    hits = misses = 0
    for i in range(200):
        pick = _policy(rng)
        g = Game(make_deal("retuned", i), button_seat=rng.randrange(2))
        while not g.is_terminal:
            obs = g.observation(g.to_act)
            node = node_of(obs, tree)
            if node is None:
                misses += 1
            else:
                hits += 1
                _assert_node_matches(tree, node, obs)
            g.apply(pick(obs.legal_actions))
    assert misses > 0
    if const == "STARTING_STACK":
        assert hits == 0


def test_node_from_actions(tree):
    flop = tree.key_to_node[(1, 1)]
    assert node_from_actions((), tree) == 0
    assert node_from_actions((1, 1), tree) == flop
    assert node_from_actions((C, C), tree) == flop
    assert node_from_actions([1, 1], tree) == flop          # any sequence of codes
    for bad in ((0,), (1, 1, 0), (9,), (1.0, 1), (True, True), ("1", "1"), None, 5, "11",
                [[1]]):
        assert node_from_actions(bad, tree) is None, bad
    assert node_from_actions((1, 1), None) is None


def test_node_from_actions_accepts_numpy_integers_but_not_bools(tree):
    """Trainer input (Task 18): action codes straight from numpy arrays."""
    flop = tree.key_to_node[(1, 1)]
    assert node_from_actions(np.array([1, 1]), tree) == flop
    assert node_from_actions((np.int8(1), np.int64(1)), tree) == flop
    assert node_from_actions(np.array([1, 1], dtype=np.uint8), tree) == flop
    assert node_from_actions(np.array([], dtype=np.int64), tree) == 0
    for bad in (np.array([True, True]), (np.True_, np.True_), np.array([1.0, 1.0]), (np.float32(1), 1)):
        assert node_from_actions(bad, tree) is None, bad


def test_node_of_median_under_20us(tree):
    rng = random.Random(20)
    observations = []
    for i in range(300):
        pick = _policy(rng)
        g = Game(make_deal("speed", i), button_seat=i % 2)
        while not g.is_terminal:
            observations.append(g.observation(g.to_act))
            g.apply(pick(g.legal_actions()))
    times = []
    for obs in observations:
        t0 = time.perf_counter()
        node_of(obs, tree)
        times.append(time.perf_counter() - t0)
    assert statistics.median(times) < 20e-6
