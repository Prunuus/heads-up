from itertools import combinations
from time import perf_counter

import numpy as np
import pytest

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


# ---- shapes: one hole vs many boards, many holes vs one board, equal rows (audit I2)

def _one_and_many(rng, n, k_one, k_many):
    """One row of k_one cards and n rows of k_many cards, none sharing a card with the one row."""
    perm = rng.permutation(36)
    one, rest = perm[:k_one], perm[k_one:]
    many = rest[np.argsort(rng.random((n, len(rest))), axis=1)[:, :k_many]]
    return one[None, :].astype(np.int64), many.astype(np.int64)


def _made_ref(hole, board):
    return max(ev.score_five([hole[i] for i in p] + [board[j] for j in t])
               for p in combinations(range(5), 2) for t in combinations(range(len(board)), 3))


def _pairs(holes, boards):
    """Row pairs the way a (1,k) operand broadcasts against an (n,k) one."""
    n = max(len(holes), len(boards))
    H = holes.tolist() * n if len(holes) == 1 else holes.tolist()
    B = boards.tolist() * n if len(boards) == 1 else boards.tolist()
    return list(zip(H, B))


@pytest.mark.parametrize("n_holes,n_boards", [(1, 60), (60, 1), (60, 60), (1, 1)])
def test_omaha_batch_every_row_combination_matches_omaha_score(n_holes, n_boards):
    rng = np.random.default_rng(1000 + 7 * n_holes + n_boards)
    if n_holes == n_boards:
        d = _deal(rng, n_holes, 10)
        holes, boards = d[:, :5], d[:, 5:]
    elif n_holes == 1:
        holes, boards = _one_and_many(rng, n_boards, 5, 5)
    else:
        boards, holes = _one_and_many(rng, n_holes, 5, 5)
    got = omaha_batch(holes, boards)
    want = [ev.omaha_score(h, b) for h, b in _pairs(holes, boards)]
    assert got.shape == (max(n_holes, n_boards),)
    assert got.tolist() == want


@pytest.mark.parametrize("width", [3, 4, 5])
@pytest.mark.parametrize("n_holes,n_boards", [(1, 40), (40, 1), (40, 40), (1, 1)])
def test_made_now_batch_every_row_combination_matches_bruteforce(width, n_holes, n_boards):
    rng = np.random.default_rng(2000 + 100 * width + 7 * n_holes + n_boards)
    if n_holes == n_boards:
        d = _deal(rng, n_holes, 5 + width)
        holes, boards = d[:, :5], d[:, 5:]
    elif n_holes == 1:
        holes, boards = _one_and_many(rng, n_boards, 5, width)
    else:
        boards, holes = _one_and_many(rng, n_holes, width, 5)
    got = made_now_batch(holes, boards)
    want = [_made_ref(h, b) for h, b in _pairs(holes, boards)]
    assert got.shape == (max(n_holes, n_boards),)
    assert got.tolist() == want


def test_one_hole_against_many_boards_is_per_board_not_the_max():
    # Kc Ks As Ah Ad: quad kings on the first board, only a pair of aces on the second.
    hole = np.array([ev.cards_from_str("Kc Ks As Ah Ad")])
    boards = np.array([ev.cards_from_str("Kh Kd 7c 8c 9d"), ev.cards_from_str("6c 7d 8h Ts Jd")])
    got = omaha_batch(hole, boards)
    assert got.tolist() == [ev.omaha_score(hole[0].tolist(), b) for b in boards.tolist()]
    assert got[0] > got[1]


def test_zero_rows_give_an_empty_result():
    for holes, boards in (((0, 5), (0, 5)), ((1, 5), (0, 5)), ((0, 5), (1, 5))):
        got = omaha_batch(np.zeros(holes, dtype=np.int64), np.zeros(boards, dtype=np.int64))
        assert got.shape == (0,) and got.dtype == np.int32
    got = made_now_batch(np.zeros((1, 5), dtype=np.int64), np.zeros((0, 3), dtype=np.int64))
    assert got.shape == (0,)
    assert score5_batch(np.zeros((0, 5), dtype=np.int64)).shape == (0,)


# A clear error names the argument at fault. numpy's own errors for these shapes
# ("operands could not be broadcast together with shapes (3,10,1) (5,1,10)", "axis 1 is
# out of bounds", ...) talk about internal arrays instead.
CLEAR = "holes|boards|cards"


@pytest.mark.parametrize("holes,boards", [
    ((3, 5), (5, 5)),       # row counts differ and neither is 1
    ((5,), (4, 5)),         # 1-D hole
    ((4, 5), (5,)),         # 1-D board
    ((4, 6), (4, 5)),       # six hole cards: the sixth would be ignored
    ((4, 4), (4, 5)),       # four hole cards
    ((4, 5), (4, 3)),       # omaha_batch needs the full 5-card board
    ((4, 5, 1), (4, 5)),    # 3-D
])
def test_omaha_batch_rejects_unsupported_shapes(holes, boards):
    with pytest.raises(ValueError, match=CLEAR):
        omaha_batch(np.zeros(holes, dtype=np.int64), np.zeros(boards, dtype=np.int64))


@pytest.mark.parametrize("holes,boards", [
    ((3, 5), (5, 3)), ((5,), (4, 3)), ((4, 6), (4, 4)), ((4, 5), (4, 2)), ((4, 5), (4, 6)),
])
def test_made_now_batch_rejects_unsupported_shapes(holes, boards):
    with pytest.raises(ValueError, match=CLEAR):
        made_now_batch(np.zeros(holes, dtype=np.int64), np.zeros(boards, dtype=np.int64))


@pytest.mark.parametrize("shape", [(5,), (4, 4), (4, 6), (2, 5, 1)])
def test_score5_batch_rejects_unsupported_shapes(shape):
    with pytest.raises(ValueError, match=CLEAR):
        score5_batch(np.zeros(shape, dtype=np.int64))


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
