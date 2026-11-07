from itertools import combinations
from time import perf_counter

import numpy as np

import evaluator as ev
from bot.fasteval import (PAIR_IDX, TRIPLE_IDX, made_now_batch, omaha_batch,
                          score5_batch, warmup)

warmup()


def _deal(rng, n, k):
    """n rows of k distinct cards from 0..35."""
    return np.argsort(rng.random((n, 36)), axis=1)[:, :k].astype(np.int64)


def test_score5_matches_reference_on_200k_random():
    rng = np.random.default_rng(1)
    cards = _deal(rng, 200_000, 5)
    got = score5_batch(cards)
    want = np.array([ev.score_five(r) for r in cards.tolist()])
    assert np.array_equal(got, want)


def test_score5_edge_cases():
    c = ev.cards_from_str
    hands = [
        "Ah6c7d8h9s",  # wheel
        "Ah6h7h8h9h",  # wheel straight flush
        "6h7h8h9hTh",  # straight flush
        "6c7c8c9cTd",  # straight
        "Ah6c7d8h9s", "KhKdKcKs6c",  # quads
        "KhKdKc7s7c",  # full house
        "Ah9h7h6h8s",  # wheel-ish non-flush
        "Ah9h7h6hJh",  # flush
        "QhQdQc7s6c",  # trips
        "AhAdKhKdQs", "AhAd9c7s6c", "AhKd9c7s6c",
    ]
    arr = np.array([c(h) for h in hands])
    want = [ev.score_five(r) for r in arr.tolist()]
    got = score5_batch(arr)
    assert got.tolist() == want
    cat = lambda h: ev.category_of(ev.score_five(c(h)))
    assert cat("Ah6c7d8h9s") == ev.STRAIGHT
    assert cat("Ah6h7h8h9h") == ev.STRAIGHT_FLUSH
    assert ev.score_five(c("Ah6c7d8h9s")) < ev.score_five(c("6c7d8h9sTc"))
    assert got[8] > got[6]      # flush beats full house
    assert got[9] > got[3]      # trips beat straight
    assert got[5] > got[8]      # quads beat flush


def test_omaha_batch_matches_reference_on_20k_random():
    rng = np.random.default_rng(2)
    d = _deal(rng, 20_000, 10)
    H, B = d[:, :5], d[:, 5:]
    got = omaha_batch(H, B)
    want = np.array([ev.omaha_score(h, b) for h, b in zip(H.tolist(), B.tolist())])
    assert np.array_equal(got, want)


def test_accepts_other_int_dtypes():
    rng = np.random.default_rng(3)
    d = _deal(rng, 500, 10)
    ref = omaha_batch(d[:, :5], d[:, 5:])
    for dt in (np.int8, np.uint8, np.int16, np.int32):
        x = d.astype(dt)
        assert np.array_equal(omaha_batch(x[:, :5], x[:, 5:]), ref)
        assert np.array_equal(score5_batch(x[:, :5]), score5_batch(d[:, :5]))


def test_made_now_on_flop_equals_bruteforce():
    rng = np.random.default_rng(4)
    for w in (3, 4, 5):
        d = _deal(rng, 600, 5 + w)
        H, B = d[:, :5], d[:, 5:]
        got = made_now_batch(H, B)
        want = []
        for h, b in zip(H.tolist(), B.tolist()):
            want.append(max(
                ev.score_five([h[i] for i in p] + [b[j] for j in t])
                for p in combinations(range(5), 2)
                for t in combinations(range(w), 3)))
        assert np.array_equal(got, np.array(want)), w
        if w == 5:
            ref = [ev.omaha_score(h, b) for h, b in zip(H.tolist(), B.tolist())]
            assert np.array_equal(got, np.array(ref))


def test_index_arrays():
    assert PAIR_IDX.shape == (10, 2) and TRIPLE_IDX.shape == (10, 3)


def test_speed():
    rng = np.random.default_rng(5)
    d = _deal(rng, 10_000, 10)
    H, B = d[:, :5].copy(), d[:, 5:].copy()
    omaha_batch(H, B)
    best = 1e9
    for _ in range(5):
        t = perf_counter()
        omaha_batch(H, B)
        best = min(best, perf_counter() - t)
    assert best < 0.010, best
