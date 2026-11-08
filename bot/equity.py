"""Vectorized Monte Carlo equity, current-board strength, and exact nut flags.

Zero budget: bot.timing.TimeManager.iters() returns 0 when time is short, and callers
pass that straight in. equity_vs_random and strength_now treat any iters <= 0 as "no
rollouts": they sample nothing, leave rng untouched and return a cheap deterministic
estimate in [0, 1] (about a quarter of the cost of 16 rollouts), never NaN and never an
exception:
- equity_vs_random preflop: the offline preflop table (bot.preflop), the same
  quantity from 3,000 rollouts per hand class; 0.5 if the table cannot be loaded.
- otherwise (both functions): where our current made hand ranks among the made hands
  of random holes on random boards of the same size (ties count half). It ignores this
  board's texture, so it is rough (mean error about 0.1 against the exact strength),
  and equity gets no credit for draws.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from bot import fasteval, preflop

_ALL = np.arange(36, dtype=np.intp)
_ACE = 8

_REF_ROWS = 2048           # random (hole, board) spots per board width in the reference
_REF_SEED = 20261107
_REF: dict[int, np.ndarray] | None = None   # board width -> sorted made-hand scores


def warmup() -> None:
    """Build fasteval's table and the zero-budget reference (idempotent, about 10 ms)."""
    global _REF
    fasteval.warmup()
    if _REF is None:
        rng = np.random.default_rng(_REF_SEED)
        ref = {}
        for w in (3, 4, 5):
            cards = rng.random((_REF_ROWS, 36)).argsort(axis=1)[:, :5 + w]
            ref[w] = np.sort(fasteval.made_now_batch(cards[:, :5], cards[:, 5:]))
        _REF = ref


def _made_hand_estimate(hole: Sequence[int], board: Sequence[int]) -> float:
    """Zero-budget strength: our made hand's rank among the reference made hands."""
    if _REF is None:
        warmup()
    ours = fasteval.made_now_batch(np.asarray(hole, dtype=np.intp)[None, :],
                                   np.asarray(board, dtype=np.intp)[None, :])[0]
    ref = _REF[len(board)]
    return float(ref.searchsorted(ours, "left") + ref.searchsorted(ours, "right")) / (2 * len(ref))


def _zero_budget_equity(hole: Sequence[int], board: Sequence[int]) -> float:
    if len(board) == 0:
        try:
            return preflop.preflop_strength(hole)
        except (OSError, KeyError, ValueError):     # missing or unreadable table, bad hole
            return 0.5
    if len(board) not in (3, 4, 5):
        raise ValueError("equity_vs_random with iters <= 0 needs a board of 0, 3, 4 or 5 cards")
    return _made_hand_estimate(hole, board)


def _unseen(hole: Sequence[int], board: Sequence[int]) -> np.ndarray:
    mask = np.ones(36, dtype=bool)
    mask[list(hole)] = False
    mask[list(board)] = False
    return _ALL[mask]


def _sample(unseen: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    """(iters, k) cards drawn without replacement from unseen."""
    order = rng.random((iters, len(unseen))).argsort(axis=1)[:, :k]
    return unseen[order]


def _score(ours: np.ndarray, theirs: np.ndarray) -> float:
    return float(((ours > theirs).sum() + 0.5 * (ours == theirs).sum()) / len(ours))


def equity_vs_random(hole: Sequence[int], board: Sequence[int], iters: int,
                     rng: np.random.Generator) -> float:
    """P(win) + 0.5 P(tie) vs a random 5-card hand, rolling out the board.

    iters <= 0: no sampling; the deterministic zero-budget estimate (module docstring).
    """
    if iters <= 0:
        return _zero_budget_equity(hole, board)
    need = 5 - len(board)
    draw = _sample(_unseen(hole, board), 5 + need, iters, rng)
    opp = draw[:, :5]
    boards = np.empty((iters, 5), dtype=np.intp)
    boards[:, :len(board)] = np.asarray(board, dtype=np.intp)
    boards[:, len(board):] = draw[:, 5:]
    holes = np.broadcast_to(np.asarray(hole, dtype=np.intp), (iters, 5))
    return _score(fasteval.omaha_batch(holes, boards), fasteval.omaha_batch(opp, boards))


def strength_now(hole: Sequence[int], board: Sequence[int], iters: int,
                 rng: np.random.Generator) -> float:
    """Win + half-tie rate of our made hand vs random hands on the current board.

    iters <= 0: no sampling; the deterministic zero-budget estimate (module docstring).
    """
    if len(board) not in (3, 4, 5):
        raise ValueError("strength_now needs a board of 3, 4 or 5 cards")
    if iters <= 0:
        return _made_hand_estimate(hole, board)
    opp = _sample(_unseen(hole, board), 5, iters, rng)
    boards = np.broadcast_to(np.asarray(board, dtype=np.intp), (iters, len(board)))
    holes = np.broadcast_to(np.asarray(hole, dtype=np.intp), (iters, 5))
    return _score(fasteval.made_now_batch(holes, boards), fasteval.made_now_batch(opp, boards))


_PAIR_CACHE: dict[int, np.ndarray] = {}
_TRI_CACHE: dict[int, np.ndarray] = {}


def _pairs(n: int) -> np.ndarray:
    p = _PAIR_CACHE.get(n)
    if p is None:
        p = _PAIR_CACHE[n] = np.array(list(combinations(range(n), 2)), dtype=np.intp)
    return p


def _tris(w: int) -> np.ndarray:
    t = _TRI_CACHE.get(w)
    if t is None:
        t = _TRI_CACHE[w] = np.array(list(combinations(range(w), 3)), dtype=np.intp)
    return t


def nut_flags(hole: Sequence[int], board: Sequence[int]) -> int:
    """Bitmask: 1 nuts-now, 2 board paired, 4 flush possible, 8 ace-flush holding."""
    w = len(board)
    if w == 0:
        return 0
    b = np.asarray(board, dtype=np.intp)
    h = np.asarray(hole, dtype=np.intp)
    flags = 0

    ours = int(fasteval.made_now_batch(h[None, :], b[None, :])[0])
    unseen = _unseen(hole, board)
    pr = unseen[_pairs(len(unseen))]                    # (M,2)
    tri = b[_tris(w)]                                   # (T,3)
    M, T = len(pr), len(tri)
    cards = np.concatenate([np.repeat(pr, T, axis=0), np.tile(tri, (M, 1))], axis=1)
    best = int(fasteval.score5_batch(cards).max())
    if ours >= best:
        flags |= 1

    br = b >> 2
    if len(np.unique(br)) < w:
        flags |= 2
    bsuit = np.bincount(b & 3, minlength=4)
    if bsuit.max() >= 3:
        flags |= 4
    hsuit = np.bincount(h & 3, minlength=4)
    for s in range(4):
        if hsuit[s] >= 2 and bsuit[s] >= 2 and (_ACE * 4 + s) in set(map(int, h)):
            flags |= 8
            break
    return flags
