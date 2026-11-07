"""Vectorized numpy Short Deck hand evaluator, exactly equal to evaluator.py.

Encoding: a five-card hand's rank multiset is keyed by key = sum(5**rank) over
its cards (per-rank counts are at most 4, so this is a collision-free base-5
number, no sorting needed). A flush adds OFF = 5**9 to the index. TABLE is an
int32[2 * 5**9] filled from evaluator._score_ranks (unreachable slots are 0).

For Omaha the key splits additively into a hole-pair part and a board-triple
part, so the 100 combinations are one broadcast add of (N,10,1) + (N,1,10).
"""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement

import numpy as np

import evaluator as _ev

OFF = 5 ** 9
PAIR_IDX = np.array(list(combinations(range(5), 2)), dtype=np.intp)      # (10, 2)
TRIPLE_IDX = np.array(list(combinations(range(5), 3)), dtype=np.intp)    # (10, 3)

_POW5 = (5 ** np.arange(9)).astype(np.int32)
# Per-card lookups (cards 0..35): rank power, suit.
_CARD_POW = np.repeat(_POW5, 4)
_CARD_SUIT = np.tile(np.arange(4, dtype=np.int8), 9)

_TABLE: np.ndarray | None = None


def _build_table() -> np.ndarray:
    table = np.zeros(2 * OFF, dtype=np.int32)
    for ranks in combinations_with_replacement(range(8, -1, -1), 5):  # descending
        if ranks.count(ranks[0]) == 5:
            continue
        key = sum(5 ** r for r in ranks)
        table[key] = _ev._score_ranks(ranks, False)
        if len(set(ranks)) == 5:
            table[key + OFF] = _ev._score_ranks(ranks, True)
    return table


def warmup() -> None:
    """Build the lookup table (idempotent, ~10 ms)."""
    global _TABLE
    if _TABLE is None:
        _TABLE = _build_table()


def _table() -> np.ndarray:
    if _TABLE is None:
        warmup()
    return _TABLE


def _prep(a) -> np.ndarray:
    a = np.asarray(a)
    return a if a.dtype == np.intp else a.astype(np.intp)


def score5_batch(cards: np.ndarray) -> np.ndarray:
    """(N,5) distinct cards 0..35 -> (N,) int32 scores equal to score_five.

    Any other shape raises ValueError.
    """
    c = _prep(cards)
    if c.ndim != 2 or c.shape[1] != 5:
        raise ValueError(f"score5_batch needs cards of shape (N, 5), got {c.shape}")
    key = _CARD_POW[c].sum(axis=1, dtype=np.int32)
    s = _CARD_SUIT[c]
    flush = (s == s[:, :1]).all(axis=1)
    return _table()[key + flush * np.int32(OFF)]


_TRIPLES = {w: np.array(list(combinations(range(w), 3)), dtype=np.intp) for w in (3, 4, 5)}


def _best_of(holes: np.ndarray, boards: np.ndarray, widths: tuple) -> np.ndarray:
    """Best 2-hole + 3-board score per row.

    holes (Nh,5) and boards (Nb,w) with w in widths. Nh == Nb pairs row i with row i.
    A single row on either side (Nh == 1 or Nb == 1) is paired with every row of the
    other side, so the result has max(Nh, Nb) rows. Anything else raises ValueError:
    a wrong shape must never collapse into a single, silently wrong, score.
    """
    h = _prep(holes)
    b = _prep(boards)
    if h.ndim != 2 or h.shape[1] != 5:
        raise ValueError(f"holes must have shape (N, 5), got {h.shape}")
    if b.ndim != 2 or b.shape[1] not in widths:
        raise ValueError(f"boards must have shape (N, w) with w in {widths}, got {b.shape}")
    nh, nb = h.shape[0], b.shape[0]
    if nh != nb and nh != 1 and nb != 1:
        raise ValueError(f"{nh} holes vs {nb} boards: row counts must match, "
                         "or one side must be a single row")
    n = nb if nh == 1 else nh
    tri = _TRIPLES[b.shape[1]]
    hp, bp, hs, bs = _CARD_POW[h], _CARD_POW[b], _CARD_SUIT[h], _CARD_SUIT[b]
    pk = hp[:, PAIR_IDX[:, 0]] + hp[:, PAIR_IDX[:, 1]]                   # (Nh,10)
    tk = bp[:, tri[:, 0]] + bp[:, tri[:, 1]] + bp[:, tri[:, 2]]          # (Nb,T)
    # suit code of a group if monochrome, else a code that never matches
    a, c = hs[:, PAIR_IDX[:, 0]], hs[:, PAIR_IDX[:, 1]]
    ps = np.where(a == c, a, np.int8(4))
    x, y, z = bs[:, tri[:, 0]], bs[:, tri[:, 1]], bs[:, tri[:, 2]]
    ts = np.where((x == y) & (y == z), x, np.int8(5))
    idx = pk[:, :, None] + tk[:, None, :]                                # (n,10,T)
    idx += (ps[:, :, None] == ts[:, None, :]) * np.int32(OFF)
    return _table().take(idx).reshape(n, idx.shape[1] * idx.shape[2]).max(axis=1)


def omaha_batch(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """(N,5) holes, (N,5) boards -> (N,) int32, equal to omaha_score row by row.

    Either side may be a single row, (1,5), paired with every row of the other side.
    Other shapes, including boards that are not 5 cards wide, raise ValueError.
    """
    return _best_of(holes, boards, (5,))


def made_now_batch(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Best hand from exactly 2 hole + 3 board cards; board width 3, 4 or 5.

    Same row rules as omaha_batch: equal row counts, or a single row on either side.
    """
    return _best_of(holes, boards, (3, 4, 5))
