"""Preflop strength table: equity vs a random hand for every 5-card hole."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
from typing import Sequence

import numpy as np

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "preflop.npz"
_PERMS = tuple(permutations(range(4)))
_MASKS: np.ndarray | None = None
_STRENGTH: np.ndarray | None = None


def canonical_key(hole: Sequence[int]) -> int:
    """Suit-permutation-invariant key: lexicographic-min relabeling, packed base 36."""
    best = None
    for p in _PERMS:
        cards = sorted((c & ~3) | p[c & 3] for c in hole)
        if best is None or cards < best:
            best = cards
    key = 0
    for c in best:
        key = key * 36 + c
    return key


def load_preflop(path: str | Path | None = None) -> None:
    """Load the table (raises FileNotFoundError with a clear message if absent)."""
    global _MASKS, _STRENGTH
    p = Path(path) if path is not None else _DEFAULT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"preflop table not found: {p}")
    with np.load(p) as z:
        masks, strength = z["masks"], z["strength"]
    _MASKS, _STRENGTH = masks, strength


def preflop_strength(hole: Sequence[int]) -> float:
    """Equity vs a random hand, by O(log n) lookup on the card bitmask."""
    if _MASKS is None:
        load_preflop()
    m = 0
    for c in hole:
        m |= 1 << c
    i = int(_MASKS.searchsorted(m))
    if i >= len(_MASKS) or _MASKS[i] != m:
        raise KeyError(f"hand not in preflop table: {list(hole)}")
    return float(_STRENGTH[i])
