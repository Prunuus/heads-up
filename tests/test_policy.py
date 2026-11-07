"""Reachable betting contexts with explicitly injected assessment predicates.

Rare A02/A05/A06/A07 action/mapping intersections are synthetic overlays on
Game observations; they do not claim those combinations are poker outcomes.
"""
from dataclasses import replace
from fractions import Fraction

import pytest
from game import Action, AgentConfig, Game, Street, make_deal
from poker_eval import FULL, REDUCED, Assessment
from poker_memory import initialize
from poker_policy import (CC, FOLD, HALF, POT, call_ceiling, canonical, postflop,
                          preflop, price, raise_valid)


def context(street=Street.FLOP):
    g = Game(make_deal("policy-context", 0), 0)
    for a in (HALF, HALF, HALF, CC):
        g.apply(a)
    while g.street < street:
        g.apply(CC)
        g.apply(CC)
    return g, g.observation(g.to_act)


def memory():
    m = {}
    initialize(AgentConfig(), m)
    return m


def assessment(**changes):
    a = Assessment(5 << 20, 325, 1, 0, .01, 0., "STRONG", False, False,
                   False, False, 0., FULL)
    return replace(a, **changes)


@pytest.mark.parametrize("street", [Street.FLOP, Street.TURN, Street.RIVER])
@pytest.mark.parametrize("mode", [FULL, REDUCED])
@pytest.mark.parametrize("case,expected", [(1, POT), (2, POT), (3, HALF), (4, CC), (5, HALF), (6, HALF), (7, CC)])
def test_strong_binding_matrix(street, mode, case, expected):
    _, o = context(street)
    a = assessment(mode=mode, risk=0 if mode == REDUCED else .01)
    label = "STATION"
    if case == 2:
        o = replace(o, legal_actions=tuple(x for x in o.legal_actions if x != HALF))
    elif case == 3:
        label = "UNKNOWN"
    elif case == 4:
        o = replace(o, legal_actions=(CC,), raise_to={})
    elif case == 5:
        o = replace(o, raise_to={HALF: o.raise_to[HALF], POT: o.raise_to[HALF]})
    elif case == 6:
        o = replace(o, my_stack=40, opp_stack=40)
    elif case == 7:
        o = replace(o, raise_to={HALF: "broken", POT: -1})
    result = canonical(o, postflop(o, memory(), a, label, .32))
    assert result == expected


@pytest.mark.parametrize("street", [Street.FLOP, Street.TURN, Street.RIVER])
@pytest.mark.parametrize("facing", [False, True])
@pytest.mark.parametrize("case,expected", [(1, POT), (2, POT), (3, HALF), (4, CC), (5, HALF), (6, HALF), (7, CC)])
def test_locked_binding_matrix(street, facing, case, expected):
    g, o = context(street)
    if facing:
        g.apply(HALF)
        o = g.observation(g.to_act)
    label = "STATION"
    a = assessment(tier="NUTS", current=True, locked=True)
    if case == 2:
        o = replace(o, legal_actions=tuple(x for x in o.legal_actions if x != HALF))
    elif case == 3:
        a = replace(a, tie=.5)
    elif case == 4:
        o = replace(o, legal_actions=(CC,), raise_to={})
    elif case == 5:
        o = replace(o, raise_to={HALF: o.raise_to[HALF], POT: o.raise_to[HALF]})
    elif case == 6:
        o = replace(o, legal_actions=tuple(x for x in o.legal_actions if x != POT))
    elif case == 7:
        o = replace(o, raise_to={HALF: None, POT: -1})
    assert canonical(o, postflop(o, memory(), a, label, .32)) == expected


@pytest.mark.parametrize("street", [Street.FLOP, Street.TURN])
def test_vulnerable_nuts_never_automatic_pot_or_reraise(street):
    g, o = context(street)
    a = assessment(current=True, vulnerable=True)
    assert postflop(o, memory(), a, "STATION", .32) == HALF
    assert postflop(o, memory(), replace(a, danger=True), "STATION", .32) == CC
    g.apply(HALF)
    o = g.observation(g.to_act)
    assert postflop(o, memory(), a, "STATION", .32) == CC
    assert postflop(o, memory(), replace(a, tier="WEAK", current=False, vulnerable=False), "STATION", .32) == FOLD


@pytest.mark.parametrize("street", [Street.FLOP, Street.TURN, Street.RIVER])
@pytest.mark.parametrize("tier", ["WEAK", "MEDIUM", "STRONG", "NUTS"])
@pytest.mark.parametrize("label,delta", [("UNKNOWN", 0), ("AGGRESSIVE", .02), ("TIGHT", -.04)])
def test_all_call_ceilings_and_boundaries(street, tier, label, delta):
    _, o = context(street)
    a = assessment(tier=tier, locked=tier == "NUTS", risk=.08 if tier == "MEDIUM" else .2)
    base = {"WEAK": (.12, .10, .08), "MEDIUM": (.27, .26, .26), "STRONG": (.34,) * 3, "NUTS": (.34,) * 3}[tier][street - 1]
    expected = .34 if a.locked else round(max(.03, min(.34, base + delta)), 12)
    assert call_ceiling(o, a, label) == pytest.approx(expected)
    # Exact integer chips at and either side of every price threshold.
    frac = Fraction(str(expected))
    c = frac.numerator * 100
    p = (frac.denominator - frac.numerator) * 100
    for adjustment, result in ((-1, CC), (0, CC), (1, FOLD)):
        trial = replace(o, to_call=c + adjustment, my_stack=100000, opp_bet=c + adjustment,
                        pot=p, legal_actions=(FOLD, CC), raise_to={})
        assert postflop(trial, memory(), a, label, .32) == result


@pytest.mark.parametrize("street", [Street.FLOP, Street.TURN, Street.RIVER])
def test_reachable_half_and_pot_prices(street):
    for action, expected in ((HALF, .25), (POT, 1 / 3)):
        g, _ = context(street)
        g.apply(action)
        o = g.observation(g.to_act)
        assert price(o) == expected
        medium = assessment(tier="MEDIUM", risk=.08)
        assert postflop(o, memory(), medium, "UNKNOWN", .32) == (CC if action == HALF else FOLD)
        strong = assessment()
        assert postflop(o, memory(), strong, "UNKNOWN", .32) == (CC if action == HALF else FOLD)
        locked = assessment(tier="NUTS", locked=True)
        assert postflop(o, memory(), locked, "UNKNOWN", .32) in (CC, HALF)


def test_draw_danger_and_raise_context():
    g, _ = context(Street.TURN)
    g.apply(HALF)
    o = g.observation(g.to_act)
    assert call_ceiling(o, assessment(tier="MEDIUM", danger=True), "UNKNOWN") == pytest.approx(.23)
    assert call_ceiling(o, assessment(tier="WEAK", draw=.33), "UNKNOWN") == pytest.approx(.311)
    assert call_ceiling(replace(o, raises_this_street=2), assessment(), "UNKNOWN") == pytest.approx(.30)
    assert price(replace(o, to_call=100, my_stack=10, pot=30)) == .25


@pytest.mark.parametrize("score,expected", [(19, FOLD), (20, CC), (39, CC), (40, HALF), (59, HALF), (60, POT)])
def test_opening_boundaries(score, expected, monkeypatch):
    import poker_policy
    monkeypatch.setattr(poker_policy, "preflop_score", lambda h: score)
    o = Game(make_deal("open", 0), 0).observation(0)
    assert preflop(o, memory(), "UNKNOWN", .32) == expected
    assert preflop(replace(o, raise_to={}), memory(), "UNKNOWN", .32) == (FOLD if score < 20 else CC)


@pytest.mark.parametrize("score,ceiling", [(10, .12), (25, .21), (40, .27), (55, .31), (70, .34)])
def test_preflop_willingness_boundaries(score, ceiling, monkeypatch):
    import poker_policy
    monkeypatch.setattr(poker_policy, "preflop_score", lambda h: score)
    g = Game(make_deal("pre-price", 0), 0)
    g.apply(HALF)
    o = g.observation(1)
    frac = Fraction(str(ceiling))
    c, p = frac.numerator * 100, (frac.denominator - frac.numerator) * 100
    for adj, expected in ((-1, CC), (0, CC), (1, FOLD)):
        t = replace(o, to_call=c + adj, pot=p, my_stack=100000, opp_bet=o.my_bet+c+adj,
                    legal_actions=(FOLD, CC), raise_to={})
        assert preflop(t, memory(), "UNKNOWN", .32) == expected


def test_medium_semibluff_purebluff_and_river_blocker():
    _, o = context(Street.FLOP)
    m = memory()
    medium = assessment(tier="MEDIUM", score=1 << 20, risk=.08)
    assert postflop(o, m, medium, "STATION", .32) == HALF
    assert postflop(o, m, replace(medium, danger=True), "STATION", .32) == CC
    m["aggressor"] = True
    assert postflop(o, m, medium, "UNKNOWN", .32) == HALF
    weak = assessment(tier="WEAK", score=0, risk=.3)
    assert postflop(o, m, replace(weak, draw=.20), "UNKNOWN", .32) == HALF
    assert postflop(o, m, replace(weak, draw=.199), "UNKNOWN", .32) == CC
    assert postflop(o, m, weak, "OVERFOLDER", .52) == HALF
    m["pure_bluffs"] = False
    assert postflop(o, m, weak, "OVERFOLDER", .52) == CC


def test_big_blind_option_and_opening_adaptation(monkeypatch):
    import poker_policy
    g = Game(make_deal("bb-option", 0), 0)
    g.apply(CC)
    o = g.observation(1)
    for s, expected in ((44, CC), (45, HALF), (64, HALF), (65, POT)):
        monkeypatch.setattr(poker_policy, "preflop_score", lambda h: s)
        assert preflop(o, memory(), "UNKNOWN", .32) == expected
    opener = Game(make_deal("bb-option", 0), 0).observation(0)
    monkeypatch.setattr(poker_policy, "preflop_score", lambda h: 40)
    assert preflop(opener, memory(), "STATION", .8) == CC
    assert preflop(opener, memory(), "OVERFOLDER", .32) == HALF
    m = memory()
    m["resolved_pre"] = 12
    monkeypatch.setattr(poker_policy, "preflop_score", lambda h: 35)
    assert preflop(opener, m, "UNKNOWN", .52) == HALF
    assert preflop(opener, memory(), "OVERFOLDER", .52) == CC


@pytest.mark.parametrize("street", [Street.FLOP, Street.TURN, Street.RIVER])
@pytest.mark.parametrize("kind", ["thin", "strong", "locked", "bluff"])
def test_matching_failing_and_unavailable_free_rules(street, kind):
    _, o = context(street)
    m = memory()
    label = "STATION"
    a = assessment()
    expected = HALF
    if kind == "thin":
        a = assessment(tier="MEDIUM", score=1 << 20, risk=.10)
        failing = replace(a, risk=.101)
    elif kind == "strong":
        label = "UNKNOWN"
        failing = replace(a, danger=True, score=1 << 20)
    elif kind == "locked":
        a = assessment(tier="NUTS", locked=True, current=True)
        expected = POT
        failing = replace(a, tier="WEAK", locked=False, current=False, score=0, risk=.5)
    else:
        label = "OVERFOLDER"
        m.update(aggressor=True, resolved_post=12)
        a = assessment(tier="WEAK", score=0, risk=.5)
        if street == Street.RIVER:
            # Synthetic blocker cards on a reachable free river context.
            # Exactly one hero club including Ac; three clubs on board.
            o = replace(o, hole_cards=(32, 29, 22, 15, 6), board=(0, 8, 16, 25, 30),
                        is_button=True, action_history=o.action_history + (
                            __import__('game').ObservedAction(Street.RIVER, False, CC, 0, 0),))
            failing = replace(a, score=2 << 20)
        else:
            o = replace(o, is_button=True)
            failing = replace(a, danger=True)
    assert postflop(o, m, a, label, .52) == expected
    assert postflop(o, m, failing, label, .52) == CC
    no_raise = replace(o, legal_actions=(CC,), raise_to={})
    assert postflop(no_raise, m, a, label, .52) == CC


@pytest.mark.parametrize("street", [Street.FLOP, Street.TURN, Street.RIVER])
def test_strong_reraise_qualification(street):
    g, _ = context(street)
    # Low-price short wager synthetic overlay isolates the willingness margin;
    # the action history itself was generated by the official Game.
    g.apply(HALF)
    o = g.observation(g.to_act)
    a = assessment()
    # Synthetic stack overlay isolates the .55/.60 investment predicates.
    # The offered raise adds 48, so a 100 stack satisfies either cap.
    o = replace(o, my_stack=100)
    assert postflop(o, memory(), a, "STATION", .32) == HALF
    assert postflop(o, memory(), replace(a, risk=.025), "STATION", .32) == CC
    assert postflop(replace(o, legal_actions=(FOLD, CC), raise_to={}), memory(), a, "STATION", .32) == CC


@pytest.mark.parametrize("score,expected", [(69, CC), (70, HALF), (79, HALF), (80, POT)])
def test_preflop_premium_reraise_guards(score, expected, monkeypatch):
    import poker_policy
    monkeypatch.setattr(poker_policy, "preflop_score", lambda h: score)
    g = Game(make_deal("premium", 0), 0)
    g.apply(HALF)
    o = g.observation(1)
    assert preflop(o, memory(), "UNKNOWN", .32) == expected
    assert preflop(replace(o, raise_to={}), memory(), "UNKNOWN", .32) == CC


def test_configuration_driven_blinds_and_half_reset():
    o = Game(make_deal("dynamic", 0), 0).observation(0)
    doubled = replace(o, my_bet=2, opp_bet=4, to_call=2, pot=6, big_blind=4,
                      my_stack=158, opp_stack=156, raise_to={HALF:8, POT:12})
    m = {}
    initialize(AgentConfig(starting_stack=160, small_blind=2, big_blind=4), m)
    # Score, position and price stay identical under scale changes.
    assert preflop(doubled, m, "UNKNOWN", .32) == preflop(o, memory(), "UNKNOWN", .32)
