"""Independent official-score references and realistic Game-driven acceptance."""
from collections import Counter
from dataclasses import replace
from itertools import combinations
from math import comb
from pathlib import Path
import json
import random

import pytest
import agent
import evaluator as ev
from game import Action, AgentConfig, Game, Street, make_deal
from poker_eval import (FULL, REDUCED, SEQUENCES, assess, draw_routes, preflop_score,
                        sample_indices, visible_score)
from poker_memory import initialize, finish
from poker_policy import choose, fallback


def reference(hole, board):
    return max(ev.score_five(p + t) for p in combinations(hole, 2) for t in combinations(board, 3))


def test_categories_wheel_kickers_and_allocation():
    examples = ("Ac Kd Qh Js 9c", "Ac Ad Qh Js 9c", "Ac Ad Kh Ks 9c",
                "6c 7d 8h 9s Tc", "6c 6d 6h Ks Ac", "Kc Kd Kh As Ac",
                "6c 8c Tc Qc Ac", "Kc Kd Kh Ks Ac", "Tc Jc Qc Kc Ac")
    scores = [ev.score_five(ev.cards_from_str(s)) for s in examples]
    assert scores == sorted(scores)
    assert [s >> 20 for s in scores] == list(range(9))
    assert ev.score_five(ev.cards_from_str("Ac 6d 7h 8s 9c")) < scores[3]
    assert ev.score_five(ev.cards_from_str("Ac Ad Qh Js Tc")) > scores[1]
    hole = tuple(ev.cards_from_str("Ac Kc Qc Jc 6d"))
    board = tuple(ev.cards_from_str("Tc 7c 8d 9h As"))
    assert visible_score(hole, board) == reference(hole, board)
    assert visible_score(hole, board) >> 20 != ev.STRAIGHT_FLUSH
    assert visible_score(hole, board) >> 20 != ev.FLUSH


def test_ten_thousand_distinct_rivers():
    rng = random.Random("correctness-river-10000")
    seen = set()
    while len(seen) < 10000:
        cards = tuple(rng.sample(range(36), 10))
        if cards in seen:
            continue
        seen.add(cards)
        h, b = cards[:5], cards[5:]
        assert visible_score(h, b) == ev.omaha_score(h, b)


@pytest.mark.parametrize("b,n,probes", [(3, 378, 378), (4, 351, 1404), (5, 325, 3250)])
def test_probe_counts(b, n, probes, monkeypatch):
    import poker_eval
    class Counting(dict):
        def __getitem__(self, key):
            self.calls += 1
            return super().__getitem__(key)
    t = Counting(ev.get_table())
    t.calls = 0
    monkeypatch.setattr(poker_eval, "get_table", lambda: t)
    h, board = tuple(range(5)), tuple(range(10, 10 + b))
    a = assess(h, board, FULL)
    assert a.n == n
    assert t.calls == probes + 10 * comb(b, 3)
    t.calls = 0
    assess(h, board, REDUCED)
    assert t.calls == 74 * comb(b, 3)


def test_one_thousand_full_pair_references():
    rng = random.Random("pair-reference-1000")
    textures = Counter()
    for i in range(1000):
        b = (3, 4, 5)[i % 3]
        cards = tuple(rng.sample(range(36), 5 + b))
        hole, board = cards[:5], cards[5:]
        a = assess(hole, board)
        score = reference(hole, board)
        unseen = set(range(36)) - set(cards)
        scores = [reference(p, board) for p in combinations(sorted(unseen), 2)]
        assert (a.score, a.beating, a.ties, a.n) == (score, sum(s > score for s in scores), sum(s == score for s in scores), len(scores))
        textures["paired"] += len({c >> 2 for c in board}) < b
        textures["trips"] += max(Counter(c >> 2 for c in board).values()) >= 3
        textures["suited"] += max(Counter(c & 3 for c in board).values()) >= 3
    assert min(textures.values()) >= 10


def test_kings_full_regression_and_graph():
    from design.research.calibrate_offline import fragment_graph, exact_opp5_graph
    h = tuple(ev.cards_from_str("Kh Kd 8c 7s 6h"))
    b = tuple(ev.cards_from_str("Ah Ac Kc Qd 6s"))
    a = assess(h, b)
    assert (a.beating, a.n, a.risk, a.tier, a.category) == (13, 325, .04, "MEDIUM", 5)
    _, u, beat, tie = fragment_graph(h, b, ev.get_table())
    assert exact_opp5_graph(u, beat, tie)[0] == 17156
    assert comb(len(u), 5) == 65780


def test_sampler_all_fixtures():
    fixtures = json.loads(Path("design/research/sampler_fixtures_v04.json").read_text())
    for row in fixtures:
        actual = sample_indices(row["hole_ids"], row["board_ids"], row["salt"])
        assert actual == tuple(row["sample_indices"])
        assert len(set(actual)) == 64


def test_current_locked_and_reduced_zero():
    h = tuple(ev.cards_from_str("Ac Kc 6d 7h 8s"))
    b = tuple(ev.cards_from_str("Qc Jc Tc"))
    for mode in (FULL, REDUCED):
        a = assess(h, b, mode)
        assert a.locked and a.current and not a.vulnerable and a.tier == "NUTS"
    # Top set on a rainbow low flop: currently unbeatable, exposed to redraws.
    h = tuple(ev.cards_from_str("Ac Ad Kd Qh Js"))
    b = tuple(ev.cards_from_str("Ah 6c 8s"))
    full, reduced = assess(h, b), assess(h, b, REDUCED)
    assert full.current and full.vulnerable and not full.locked
    assert reduced.beating == 0 and not reduced.current and not reduced.locked
    assert reduced.tier == "STRONG"


def runout_route(h, b, route):
    u = sorted(set(range(36)) - set(h + b))
    kind, target, _ = route
    if kind == "flush":
        good = lambda run: sum(c & 3 == target for c in b + run) >= 3
    else:
        good = lambda run: set(target) <= {c >> 2 for c in b + run}
    return sum(good(r) for r in combinations(u, 5 - len(b))) / comb(len(u), 5 - len(b))


def test_one_hundred_draw_states_by_runout_enumeration():
    rng = random.Random("draw-reference-100")
    for i in range(100):
        count = 3 + i % 2
        cards = tuple(rng.sample(range(36), 5 + count))
        h, b = cards[:5], cards[5:]
        # Category 0 exposes every permitted route to the independent oracle.
        for route in draw_routes(h, b, 0):
            assert route[2] == pytest.approx(runout_route(h, b, route), abs=1e-12)
        assert not draw_routes(h, b, 6)
        assert all(r[0] != "straight" for r in draw_routes(h, b, 3))


@pytest.mark.parametrize("board,expected", [("8c 9c Td", 1 - comb(23, 2) / comb(28, 2)),
                                           ("8c 9c Td Jh", 5 / 27),
                                           ("8c 9d Th", comb(6, 2) / comb(28, 2))])
def test_flush_examples(board, expected):
    h = tuple(ev.cards_from_str("Ac Kc Qd Js 6h"))
    b = tuple(ev.cards_from_str(board))
    p = [r[2] for r in draw_routes(h, b, 0) if r[:2] == ("flush", 0)]
    assert p == pytest.approx([expected])


def test_preflop_exhaustive_distribution():
    counts, scores = Counter(), Counter()
    for h in combinations(range(36), 5):
        counts[tuple(sorted(Counter(c & 3 for c in h).values(), reverse=True))] += 1
        scores[preflop_score(h)] += 1
    assert counts == {(2, 2, 1): 139968, (2, 1, 1, 1): 104976, (3, 2): 36288,
                      (3, 1, 1): 81648, (4, 1): 13608, (5,): 504}
    n = sum(scores.values())
    assert (n, min(scores), max(scores)) == (376992, 0, 93)
    assert sum(s * c for s, c in scores.items()) / n == pytest.approx(37.9097, abs=.0001)
    for threshold, target in ((40, 40.129), (60, 9.211), (70, 2.902), (80, .439)):
        assert 100 * sum(c for s, c in scores.items() if s >= threshold) / n == pytest.approx(target, abs=.01)
    for lo, hi, target in ((0, 20, 6.82), (20, 40, 53.05), (40, 60, 30.92), (60, 101, 9.21)):
        assert 100 * sum(c for s, c in scores.items() if lo <= s < hi) / n == pytest.approx(target, abs=.01)


def test_ten_thousand_reachable_decisions():
    rng = random.Random("reachable-10000")
    coverage = Counter()
    decisions = 0
    for mode in (FULL, REDUCED):
        memories = [{}, {}]
        for m in memories:
            initialize(AgentConfig(num_hands=10000), m, mode, diagnostics=True)
        hand = 0
        while decisions < (5000 if mode == FULL else 10000):
            g = Game(make_deal("reachable-calibration", hand), hand % 2, hand_index=hand)
            while not g.is_terminal:
                memory = memories[g.to_act]
                obs = g.observation(g.to_act)
                proposed = choose(obs, memory)
                assert proposed in obs.legal_actions
                assert memory["internal_errors"] == 0, memory["diagnostics"]
                coverage[f"street_{obs.street.name}"] += 1
                coverage[f"chosen_{proposed.name}"] += 1
                coverage[f"seat_{g.to_act}"] += 1
                coverage["raise_cap"] += obs.raises_this_street == 4
                decisions += 1
                # Drive independently of our proposed action to cover weak calls,
                # late streets and cap states that our policy would often fold.
                choices = [a for a in obs.legal_actions if a != Action.FOLD]
                executed = rng.choice(choices)
                if executed != proposed:
                    # A proposal deliberately not executed cannot create a real
                    # fold-to-raise opportunity in this coverage-only harness.
                    memory["pending"] = None
                g.apply(executed)
            for seat in (0, 1):
                finish(g.hand_result(seat), memories[seat])
            hand += 1
        assert hand > 0
    assert all(coverage[f"street_{s.name}"] > 100 for s in Street)
    assert all(coverage[f"chosen_{a.name}"] > 0 for a in Action)
    assert coverage["raise_cap"] > 10
    Path("reports/v05/reachability_control.json").write_text(json.dumps(dict(coverage), indent=2))


@pytest.mark.parametrize("i", range(100))
def test_malformed_observation_fallback(i):
    g = Game(make_deal("malformed", i), i % 2, hand_index=i)
    o = g.observation(g.to_act)
    memory = {}
    initialize(AgentConfig(), memory, diagnostics=True)
    changes = ({"hole_cards": o.hole_cards[:-1]}, {"hole_cards": (0,) * 5},
               {"board": (35,)}, {"pot": -1}, {"my_stack": -1}, {"to_call": -1},
               {"hand_index": -1}, {"raises_this_street": 99}, {"opp_bet": -1},
               {"hole_cards": (40, 1, 2, 3, 4)})
    bad = replace(o, **changes[i % len(changes)])
    assert choose(bad, memory) == fallback(bad)
    assert memory["internal_errors"] == 1


def test_unrecoverable_legal_set_is_protocol_failure():
    o = Game(make_deal("protocol", 0), 0).observation(0)
    with pytest.raises(ValueError, match="protocol failure"):
        choose(replace(o, legal_actions=()), {})
