"""Task 10: decision features and the parameterized heuristic policy."""

import dataclasses
import math
import pickle
import random
import time

import numpy as np
import pytest

from bot.features import Features, compute_features
from bot.heuristic import HeuristicParams, heuristic_action
from evaluator import cards_from_str
from game import (Action, Deal, Game, Observation, ObservedAction, Street,
                  button_for_deal, make_deal)

A = Action
RAISES = (A.RAISE_HALF, A.RAISE_POT)

# Parameter sets that push every branch: the defaults, one that wants to raise and
# bluff with anything, and one that wants to fold anything that costs chips.
AGGRESSIVE = HeuristicParams(open_min=0.0, threebet_min=0.0, call_margin=-1.0,
                             range_discount=(0.0, 0.0, 0.0), value_raise=(0.0, 0.0, 0.0),
                             pot_vs_half_cut=0.5, bluff_freq=1.0, nonnut_penalty=0.0)
PASSIVE = HeuristicParams(open_min=2.0, threebet_min=2.0, call_margin=1.0,
                          range_discount=(0.5, 0.5, 0.5), value_raise=(2.0, 2.0, 2.0),
                          pot_vs_half_cut=2.0, bluff_freq=0.0, nonnut_penalty=0.5)
PARAM_SETS = (HeuristicParams(), AGGRESSIVE, PASSIVE)


def _drive(deal: Deal, button: int, actions) -> Game:
    """A game advanced by a fixed action list (each must be legal)."""
    g = Game(deal, button, hand_index=0)
    for a in actions:
        g.apply(a)
    return g


def _deal(hero: str, villain: str, board: str) -> Deal:
    return Deal(hole_cards=(tuple(sorted(cards_from_str(hero))),
                            tuple(sorted(cards_from_str(villain)))),
                board=tuple(cards_from_str(board)))


def _observations(n: int, seed: int):
    """n observations from games of the heuristic (default params) vs a random bot.

    Both seats' observations are collected; the hero seat alternates every deal.
    """
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    params = HeuristicParams()
    out = []
    d = 0
    while len(out) < n:
        g = Game(make_deal(f"t10:{seed}", d), button_for_deal(d), hand_index=d)
        hero = d % 2
        while not g.is_terminal:
            seat = g.to_act
            obs = g.observation(seat)
            out.append(obs)
            if seat == hero:
                a = heuristic_action(obs, compute_features(obs, 0, rng), params, rng)
                assert a in obs.legal_actions
            else:
                a = rnd.choice(obs.legal_actions)
            g.apply(a)
        d += 1
    return out[:n]


@pytest.fixture(scope="module")
def obs10k():
    return _observations(10_000, seed=7)


def test_heuristic_params_defaults_are_the_briefs():
    p = HeuristicParams()
    assert (p.open_min, p.threebet_min, p.call_margin) == (0.44, 0.53, 0.02)
    assert p.range_discount == (0.10, 0.08, 0.06)
    assert p.value_raise == (0.70, 0.75, 0.80)
    assert (p.pot_vs_half_cut, p.bluff_freq, p.nonnut_penalty) == (0.85, 0.10, 0.08)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.open_min = 0.5
    assert pickle.loads(pickle.dumps(p)) == p


def test_always_legal_on_10k_random_observations(obs10k):
    assert len(obs10k) == 10_000
    streets = {o.street for o in obs10k}
    assert streets == set(Street)
    rng = np.random.default_rng(3)
    for i, obs in enumerate(obs10k):
        feats = compute_features(obs, (0, 16)[i % 2], rng)
        for params in PARAM_SETS:
            a = heuristic_action(obs, feats, params, rng)
            assert a in obs.legal_actions, (obs, params, a)


def test_never_folds_when_free(obs10k):
    rng = np.random.default_rng(4)
    free = [o for o in obs10k if o.to_call == 0]
    assert len(free) > 1000
    for obs in free:
        feats = compute_features(obs, 0, rng)
        for params in PARAM_SETS:
            assert heuristic_action(obs, feats, params, rng) != A.FOLD
    # A malformed observation that lists FOLD although nothing is owed: still no fold.
    g = _drive(make_deal("free", 0), 0, [A.CHECK_CALL])           # SB limps, BB to act
    obs = g.observation(g.to_act)
    assert obs.to_call == 0
    bad = dataclasses.replace(obs, legal_actions=(A.FOLD,) + obs.legal_actions)
    for params in PARAM_SETS:
        for k in range(20):
            feats = compute_features(bad, 0, rng)
            assert heuristic_action(bad, feats, params, np.random.default_rng(k)) != A.FOLD


def _corner_observations():
    """Review Focus 2 corners, as (label, observation)."""
    out = []
    deal = make_deal("corners", 3)
    # Preflop pot-raise war: 6 -> 18 -> 54 -> 80 (all-in, below min_raise_to 90).
    g = _drive(deal, 0, [A.RAISE_POT, A.RAISE_POT, A.RAISE_POT])
    o = g.observation(g.to_act)
    assert o.raise_to[A.RAISE_POT] == 80 < o.min_raise_to
    out.append(("all-in raise_to below min_raise_to (preflop)", o))
    g.apply(A.RAISE_POT)
    o = g.observation(g.to_act)
    assert o.to_call >= o.my_stack and o.opp_stack == 0 and o.legal_actions == (A.FOLD, A.CHECK_CALL)
    out.append(("to_call >= my_stack, opponent all-in, raises capped", o))
    # Flop: bet, raise, all-in re-raise below min_raise_to, then the caller's spot.
    g = _drive(deal, 0, [A.RAISE_POT, A.CHECK_CALL, A.RAISE_POT, A.RAISE_POT])
    o = g.observation(g.to_act)
    assert o.street == Street.FLOP and o.raise_to[A.RAISE_HALF] < o.min_raise_to
    out.append(("all-in raise_to below min_raise_to (flop)", o))
    g.apply(A.RAISE_HALF)
    o = g.observation(g.to_act)
    assert o.to_call >= o.my_stack and o.opp_stack == 0 and A.RAISE_HALF not in o.legal_actions
    out.append(("to_call >= my_stack on the flop, opponent all-in", o))
    # Opponent all-in for less than our bet: only CHECK_CALL is legal (not reachable with
    # equal stacks, but a retuned config or a corrupt observation could send it).
    base = g.observation(g.to_act)
    only = dataclasses.replace(base, to_call=0, my_bet=base.opp_bet, opp_stack=0,
                               legal_actions=(A.CHECK_CALL,), raise_to={})
    out.append(("legal_actions == (CHECK_CALL,)", only))
    river = _drive(deal, 1, [A.CHECK_CALL, A.CHECK_CALL] + [A.CHECK_CALL] * 4)
    o = river.observation(river.to_act)
    out.append(("legal_actions == (CHECK_CALL,) on the river",
                dataclasses.replace(o, legal_actions=(A.CHECK_CALL,), raise_to={})))
    # A short stack facing a bet bigger than itself, with only FOLD/CHECK_CALL legal.
    out.append(("short stack, to_call > my_stack",
                dataclasses.replace(o, to_call=30, opp_bet=30, my_stack=12, opp_stack=50,
                                    legal_actions=(A.FOLD, A.CHECK_CALL), raise_to={})))
    return out


def test_all_in_corners_are_legal():
    corners = _corner_observations()
    assert len(corners) == 7
    for label, obs in corners:
        for iters in (0, 32):
            for seed in range(10):
                rng = np.random.default_rng(seed)
                feats = compute_features(obs, iters, rng)
                assert all(math.isfinite(x) for x in (feats.eq, feats.now, feats.spr, feats.pot_odds))
                for params in PARAM_SETS:
                    a = heuristic_action(obs, feats, params, rng)
                    assert a in obs.legal_actions, (label, params, a)


def _river_nuts_facing_bet(raises_by_villain: int):
    """Hero (BB, out of position) holds the nuts on the river and faces a bet or raise."""
    deal = _deal("Th 9h 7s 7d 6s", "Ac Kc 8d 8s 9c", "Ah Kh Qh Jh 6c")
    acts = [A.CHECK_CALL, A.CHECK_CALL,          # preflop: SB limps, BB checks
            A.CHECK_CALL, A.CHECK_CALL,          # flop
            A.CHECK_CALL, A.CHECK_CALL]          # turn
    if raises_by_villain == 1:
        acts += [A.CHECK_CALL, A.RAISE_POT]      # river: hero checks, villain bets pot
    else:
        acts += [A.RAISE_HALF, A.RAISE_POT, A.RAISE_HALF, A.RAISE_POT]   # bet, raise, 3-bet, 4-bet
    g = _drive(deal, 1, acts)                    # seat 1 is the button, hero is seat 0
    obs = g.observation(g.to_act)
    assert g.to_act == 0 and obs.street == Street.RIVER and obs.to_call > 0
    return obs


@pytest.mark.parametrize("raises_by_villain", [1, 2])
def test_calls_with_nuts_on_river(raises_by_villain):
    obs = _river_nuts_facing_bet(raises_by_villain)
    for iters in (0, 16, 128):
        feats = compute_features(obs, iters, np.random.default_rng(iters))
        assert feats.nut & 1
        assert feats.opp_raised_this_street == raises_by_villain
        for params in PARAM_SETS:
            assert heuristic_action(obs, feats, params, np.random.default_rng(1)) != A.FOLD


def test_raises_with_nuts_on_river_when_allowed():
    obs = _river_nuts_facing_bet(1)
    assert A.RAISE_POT in obs.legal_actions
    feats = compute_features(obs, 64, np.random.default_rng(0))
    assert heuristic_action(obs, feats, HeuristicParams(), np.random.default_rng(0)) in RAISES


def _street_observation(street: Street) -> Observation:
    deal = make_deal("streets", 11)
    acts = {Street.PREFLOP: [], Street.FLOP: [A.CHECK_CALL, A.CHECK_CALL],
            Street.TURN: [A.CHECK_CALL] * 4, Street.RIVER: [A.CHECK_CALL] * 6}[street]
    g = _drive(deal, 0, acts)
    obs = g.observation(g.to_act)
    assert obs.street == street
    return obs


@pytest.mark.parametrize("street", list(Street))
def test_compute_features_iters_zero(street):
    obs = _street_observation(street)
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    feats = compute_features(obs, 0, rng)
    assert rng.bit_generator.state == state            # iters == 0 draws nothing
    assert isinstance(feats, Features) and feats.street == street
    for x in (feats.eq, feats.now, feats.spr, feats.pot_odds):
        assert math.isfinite(x)
    assert 0.0 <= feats.eq <= 1.0 and 0.0 <= feats.now <= 1.0
    assert feats.pre is not None and 0.0 < feats.pre < 1.0
    if street == Street.PREFLOP:
        assert feats.eq == feats.pre and feats.nut == 0
    a = heuristic_action(obs, feats, HeuristicParams(), rng)
    assert a in obs.legal_actions
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        heuristic_action(obs, compute_features(obs, 0, rng), HeuristicParams(), rng)
        ts.append(time.perf_counter() - t0)
    assert float(np.median(ts)) < 0.0005                # generous: ~0.15 ms measured


def test_feature_values_from_history():
    deal = make_deal("vals", 2)
    # Button seat 0 raises pot (to 6), BB 3-bets pot (to 18): button faces 12 into 24.
    g = _drive(deal, 0, [A.RAISE_POT, A.RAISE_POT])
    obs = g.observation(0)
    f = compute_features(obs, 0, np.random.default_rng(0))
    assert f.street == Street.PREFLOP and f.in_position is True
    assert f.opp_raised_this_street == 1
    assert f.pot_odds == pytest.approx(12 / (24 + 12))
    assert f.spr == pytest.approx(62 / 36)               # stack behind after calling / pot after calling
    # Flop after a called 3-bet: BB acts first, nothing owed, no raises yet.
    g.apply(A.CHECK_CALL)
    obs = g.observation(1)
    f = compute_features(obs, 32, np.random.default_rng(0))
    assert f.street == Street.FLOP and f.in_position is False
    assert f.opp_raised_this_street == 0 and f.pot_odds == 0.0
    assert f.spr == pytest.approx(62 / 36)
    assert 0.0 <= f.eq <= 1.0 and 0.0 <= f.now <= 1.0


def test_features_are_frozen():
    f = compute_features(_street_observation(Street.FLOP), 16, np.random.default_rng(0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.eq = 0.0


def test_heuristic_is_deterministic_given_rng_and_ignores_opp(obs10k):
    class Whatever:
        pass

    sample = obs10k[:500]
    for opp in (None, Whatever(), object()):
        acts = []
        for i, obs in enumerate(sample):
            rng = np.random.default_rng(i)
            feats = compute_features(obs, 16, rng)
            acts.append(heuristic_action(obs, feats, HeuristicParams(), rng, opp))
        if opp is None:
            ref = acts
        assert acts == ref


def test_preflop_open_and_fold_behaviour():
    """Strong hands raise first in, very weak hands fold to a big raise."""
    strong = _deal("Ah As Kh Kd Qs", "6c 7d 8h 9s Tc", "Jc Jd Td 9d 6d")
    g = Game(strong, 0)                                   # hero (seat 0) is the button
    obs = g.observation(0)
    f = compute_features(obs, 0, np.random.default_rng(0))
    assert f.pre > 0.6
    assert heuristic_action(obs, f, HeuristicParams(), np.random.default_rng(0)) in RAISES
    weak = _deal("6c 6d 6h 6s 7c", "Ah As Kh Kd Qs", "Jc Jd Td 9d 8c")
    g = _drive(weak, 1, [A.RAISE_POT])                    # villain (button) raises to 6
    obs = g.observation(0)
    assert obs.to_call == 4
    f = compute_features(obs, 0, np.random.default_rng(0))
    assert f.pre < 0.2
    assert heuristic_action(obs, f, HeuristicParams(), np.random.default_rng(0)) == A.FOLD
