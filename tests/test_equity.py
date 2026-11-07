import time
from itertools import combinations

import numpy as np
import pytest

import evaluator as ev
from bot.equity import equity_vs_random, nut_flags, strength_now
from bot.fasteval import warmup


def _spot(rng, nb):
    cards = rng.permutation(36)
    return [int(c) for c in cards[:5]], [int(c) for c in cards[5:5 + nb]]


def _slow_equity(hole, board, iters, rnd):
    unseen = [c for c in range(36) if c not in hole and c not in board]
    score = 0.0
    for _ in range(iters):
        s = rnd.sample(unseen, 5 + 5 - len(board))
        opp, full = s[:5], list(board) + s[5:]
        a, b = ev.omaha_score(hole, full), ev.omaha_score(opp, full)
        score += 1.0 if a > b else 0.5 if a == b else 0.0
    return score / iters


@pytest.mark.slow
def test_equity_agrees_with_slow_reference():
    import random
    rnd = random.Random(11)
    rng = np.random.default_rng(12)
    for i in range(50):
        hole, board = _spot(np.random.default_rng(100 + i), (0, 3, 4, 5)[i % 4])
        fast = equity_vs_random(hole, board, 2000, rng)
        slow = _slow_equity(hole, board, 4000, rnd)
        assert abs(fast - slow) < 0.03, (hole, board, fast, slow)


def test_equity_quick_sanity():
    rng = np.random.default_rng(0)
    H = ev.cards_from_str("Kc Ks As Ah Ad")
    B = ev.cards_from_str("Kh Kd 7c 8c 9d")
    assert equity_vs_random(H, B, 500, rng) > 0.95
    e = equity_vs_random(H, [], 500, rng)
    assert 0.5 < e < 1.0


def test_river_equity_is_deterministic_given_rng_seed():
    hole, board = _spot(np.random.default_rng(5), 5)
    a = equity_vs_random(hole, board, 256, np.random.default_rng(9))
    b = equity_vs_random(hole, board, 256, np.random.default_rng(9))
    assert a == b


def test_strength_now_basic():
    rng = np.random.default_rng(0)
    H = ev.cards_from_str("Kc Ks As Ah Ad")
    B = ev.cards_from_str("Kh Kd 7c")
    assert strength_now(H, B, 300, rng) > 0.9
    with pytest.raises(ValueError):
        strength_now(H, [], 10, rng)


def test_strength_equals_equity_on_river_statistically():
    hole, board = _spot(np.random.default_rng(11), 5)
    s = strength_now(hole, board, 20000, np.random.default_rng(1))
    e = equity_vs_random(hole, board, 20000, np.random.default_rng(2))
    assert abs(s - e) < 0.03


def test_nut_flag_royal_example():
    H = ev.cards_from_str("Th 7s 7d 8c 9c")
    B = ev.cards_from_str("Ah Kh Qh Jh 6c")
    f = nut_flags(H, B)
    assert not (f & 1)
    assert f & 2 == 0
    assert f & 4


def test_nut_flags_bits():
    assert nut_flags(ev.cards_from_str("As Ah Ad Ac Ks"), []) == 0
    # royal-ish: we hold Ah Th, board Kh Qh Jh -> straight flush nuts
    f = nut_flags(ev.cards_from_str("Ah Th 6c 6d 7c"), ev.cards_from_str("Kh Qh Jh"))
    assert f & 1 and f & 4 and f & 8 and not f & 2
    # board paired
    f = nut_flags(ev.cards_from_str("6c 7d 8c 9d Ts"), ev.cards_from_str("Kh Kd 2h".replace("2h", "6h")))
    assert f & 2


def test_nut_flags_matches_bruteforce():
    rnd = np.random.default_rng(3)
    for i in range(30):
        hole, board = _spot(rnd, (3, 4, 5)[i % 3])
        mine = max(ev.score_five([hole[a], hole[b], *[board[x] for x in t]])
                   for a, b in combinations(range(5), 2) for t in combinations(range(len(board)), 3))
        unseen = [c for c in range(36) if c not in hole and c not in board]
        best = max(ev.score_five([a, b, *[board[x] for x in t]])
                   for a, b in combinations(unseen, 2) for t in combinations(range(len(board)), 3))
        assert bool(nut_flags(hole, board) & 1) == (mine >= best)


def _timeit(fn, n=300):
    warmup()
    for _ in range(30):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 99))


def test_timing_p99_under_1_5ms_at_128_iters_flop():
    hole, board = _spot(np.random.default_rng(7), 3)
    rng = np.random.default_rng(0)
    med, p99 = _timeit(lambda: equity_vs_random(hole, board, 128, rng))
    print("equity flop", med, p99)
    assert p99 < 1.5
