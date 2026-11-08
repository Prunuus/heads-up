import time
from itertools import combinations, permutations

import numpy as np
import pytest

from bot import preflop
from bot.preflop import canonical_key, preflop_strength


def test_canonical_invariant_under_suit_permutation():
    rng = np.random.default_rng(1)
    perms = list(permutations(range(4)))
    for _ in range(1000):
        hand = [int(c) for c in rng.permutation(36)[:5]]
        k = canonical_key(hand)
        for p in perms:
            assert canonical_key([(c & ~3) | p[c & 3] for c in hand]) == k


def _table():
    preflop.load_preflop()
    return preflop._MASKS, preflop._STRENGTH


def test_table_covers_all_hands():
    masks, _ = _table()
    combos = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    allm = (np.int64(1) << combos).sum(axis=1)
    assert len(masks) == len(allm) == 376992
    assert np.array_equal(np.sort(allm), masks)


def test_strength_range():
    masks, s = _table()
    # Hands holding four of a kind play like a single pair in Omaha (only two
    # hole cards are used), so they are legitimately far below the pack
    # (four sixes ~0.10). Trips of low ranks are also weak (min 0.222); the rest is >= 0.20.
    ranks = np.stack([(masks >> (4 * r)) & 15 for r in range(9)])
    quads = np.isin(ranks, (15,)).any(axis=0)
    assert quads.sum() == 9 * 32
    assert s[~quads].min() >= 0.20 and s.max() <= 0.75
    assert s.min() >= 0.05
    assert 0.45 <= float(np.median(s)) <= 0.53


def test_lookup_under_50us():
    preflop.load_preflop()
    rng = np.random.default_rng(2)
    hands = [[int(c) for c in rng.permutation(36)[:5]] for _ in range(1000)]
    preflop_strength(hands[0])
    ts = []
    for h in hands:
        t = time.perf_counter()
        preflop_strength(h)
        ts.append(time.perf_counter() - t)
    assert float(np.median(ts)) < 50e-6


def test_missing_file_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(preflop, "_MASKS", None)
    monkeypatch.setattr(preflop, "_STRENGTH", None)
    monkeypatch.setattr(preflop, "_DEFAULT_PATH", tmp_path / "nope.npz")
    with pytest.raises(FileNotFoundError):
        preflop_strength([0, 1, 2, 3, 4])
