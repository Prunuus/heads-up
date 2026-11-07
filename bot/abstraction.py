"""Card abstraction: (street, hole, board) -> a small bucket id for the CFR blueprint.

The blueprint keeps one strategy per (betting node, bucket) pair (nodes: bot/tree.py),
so the bucket is all it knows about the cards. Two rules shape this module: the bucket
must be cheap at the table, and it must be the very bucket the trainer saw for the same
cards. Both hold by construction. Every feature below is a deterministic function of
the cards (no rollouts, no rng), computed by this code offline (train/fit_buckets.py,
the Task 17 deal dataset) and at runtime. So the runtime bucket equals the offline
bucket exactly, whatever rollout budget the time manager allows, iters=0 included.

Buckets (defaults: preflop 200, flop 100, turn 100, river 100 = DEFAULT_NUM_BUCKETS)
- preflop: quantile bucket of preflop_strength(hole) (bot/preflop.py: equity vs a
  random hand), edges fitted on all 376,992 holes.
- flop, turn: a rows x cols grid (10 x 10), bucket = row * cols + col.
  - row: quantile bucket of strength, our made hand now vs a random hand (ties half),
    the quantity bot.equity.strength_now estimates.
  - col: quantile bucket of equity, P(win) + P(tie)/2 vs a random hand with the board
    run out (bot.equity.equity_vs_random), among the hands of that row only: every row
    has its own equity edges, so each cell holds about 1% of hands, and a big draw (low
    row, high column) never shares a bucket with a made hand of the same equity.
- river: hands that are not the nuts get quantile buckets 0 .. B-6 of equity (equal to
  strength on the river). The nuts (nut_flags bit 0, ties count) take the top 5, by the
  category of our hand: B-5 trips, B-4 full house, B-3 flush, B-2 four of a kind, B-1
  straight flush. Trips beat a straight in this game and the opponent can always make
  some set, so the river nuts are never below trips (lower categories would map to B-5).
- Edges: a value equal to an edge goes to the upper bucket (searchsorted side "right"),
  and the result always lies in range(NUM_BUCKETS[street]).

Features (exact up to fixed designs; rms errors measured on 100 random spots per street)
- Pair values: every unseen pair's best 2+3 hand with the board is scored once. An
  opponent hole's made hand is then the max over its 10 pairs: 10 gathers, no scoring.
  Scores come from a 0.7 MB pair-class x triple-class table built from bot.fasteval.
- Opponent holes come from fixed designs over positions 0..n-1, mapped to this spot's n
  unseen cards in ascending order. Rows are blocks cut from shuffles, so each position is
  used about equally often.
- strength (flop, turn, river): our made hand vs 4,096 design holes (16,384 on the
  river): 0.004-0.005 vs exact enumeration of every opponent hole (0.003 river).
- equity, turn: exact over the river card, 304 design holes for each of the 27 river
  cards: 0.004 vs 40k-rollout Monte Carlo (whose own sd is 0.0025).
- equity, flop: exact over all 378 (turn, river) runouts, 8 design holes each. A pair's
  full-board hand is max(pair + flop, pair + one runout card + two flop cards, pair +
  both runout cards + one flop card); the first two are shared by many cells, so a cell
  costs 3 lookups: 0.007 vs 40k-rollout Monte Carlo.
- nut flag (exact): our hand >= every unseen pair's best hand, as nut_flags bit 0.

Cost and the rollout budget
- One call costs about 3 us preflop, 0.25 ms flop, 0.15 ms turn, 0.09 ms river on the
  dev laptop (p99 0.3 ms). That fits in the judge's free 1 ms per request. A policy that
  acts several times on one street can cache the bucket per (street, hole, board).
- bucket() keeps the plan's `iters` and `rng` arguments but ignores them: there is no
  minimum budget, and the bucket at iters=0 is the offline bucket.
- Monte Carlo features (the plan's design) would disagree with the offline bucket at the
  table. On this 10 x 10 grid, runtime at 128 rollouts vs offline at 2,000 lands in the
  same or an adjacent cell only 71% of the time on the flop and 64% on the turn; at 0
  rollouts, 15% and 19% (Task 16 report).

Data file (data/buckets.npz from train/fit_buckets.py; arrays)
- format: 1. num_buckets: int[4] (preflop, flop, turn, river). flop_grid, turn_grid: int[2].
- preflop_edges [B0-1]; flop/turn_strength_edges [rows-1]; flop/turn_equity_edges
  [rows, cols-1]; river_edges [B3-6]. All non-decreasing floats.
- design_strength_28, design_strength_27, design_strength_26: int16 [R, 10] indices into
  combinations(range(n), 2), the 10 pairs of one design hole (flop, turn, river).
- design_turn_pairs int16 [Rt, 10] (pairs of range(27)), design_turn_river int16 [Rt]
  (the river position, never in the hole).
- design_flop_pairs int16 [Rf, 10] (pairs of range(28)), design_flop_runout int16 [Rf]
  (an index into combinations(range(28), 2): the runout's two positions, never in the hole).
- Other arrays are metadata from the fit (seed, sample sizes, validation errors).

Module state is read-only (R22).
- NUM_BUCKETS is a read-only mapping, set at import from the file's num_buckets, or
  DEFAULT_NUM_BUCKETS if there is no readable file.
- Tables load lazily on first use (bucket / load_buckets / warmup) into one cached
  BucketTables with non-writeable arrays.
- A missing file raises FileNotFoundError at call time, never at import.
"""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations, combinations_with_replacement, permutations
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import numpy as np

from bot import fasteval, preflop
from game import BOARD_CARDS_VISIBLE, Street

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "buckets.npz"
DEFAULT_NUM_BUCKETS: Mapping[Street, int] = MappingProxyType(
    {Street.PREFLOP: 200, Street.FLOP: 100, Street.TURN: 100, Street.RIVER: 100})
NUT_BUCKETS = 5            # river: trips, full house, flush, four of a kind, straight flush
FORMAT = 1
_LOWEST_NUT_CATEGORY = 4   # three of a kind (evaluator category numbers)


class Features(NamedTuple):
    """What a bucket is cut from. Preflop: strength == equity == the preflop table."""

    strength: float        # made hand now vs a random hand, ties half
    equity: float          # showdown equity vs a random hand, board run out
    nut: bool              # we hold the nuts now (ties count); False preflop
    category: int          # evaluator category of our made hand; 0 preflop


# ---- pair-class x triple-class score table (built once, read-only)

_RANK_PAIRS = tuple(combinations_with_replacement(range(9), 2))      # 45 rank classes
_RANK_TRIPLES = tuple(combinations_with_replacement(range(9), 3))    # 165 rank classes
_NTC = len(_RANK_TRIPLES) * 5     # triple codes: rank class + 165 * suit code (4 = mixed)
_SCORE: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None     # pair codes, triple codes, table


def _build_score_table() -> np.ndarray:
    """int32 [45 * 5 * 165 * 5]: score of (pair class, pair suit code, triple class,
    triple suit code). Suit code s < 4: every card of suit s; 4: mixed suits. A flush is
    a suited pair and a monotone triple of the same suit; any other code pair scores the
    ranks without a flush."""
    rows, where = [], []
    for i, pair in enumerate(_RANK_PAIRS):
        for j, triple in enumerate(_RANK_TRIPLES):
            ranks = pair + triple
            counts = Counter(ranks)
            if max(counts.values()) > 4:
                continue                       # five of a kind: no such hand
            seen: Counter = Counter()
            cards = []
            for r in ranks:                    # copies of a rank get suits 0, 1, ...
                cards.append(4 * r + seen[r])
                seen[r] += 1
            if len(counts) == 5:
                cards[0] += 1                  # five distinct ranks: break the flush
                rows.append([4 * r for r in ranks])
                where.append((1, i, j))
            rows.append(cards)
            where.append((0, i, j))
    scores = fasteval.score5_batch(np.array(rows, dtype=np.intp))
    plain = np.zeros((len(_RANK_PAIRS), len(_RANK_TRIPLES)), dtype=np.int32)
    flush = np.zeros_like(plain)
    for (f, i, j), v in zip(where, scores):
        (flush if f else plain)[i, j] = v
    table = np.empty((5, len(_RANK_PAIRS), 5, len(_RANK_TRIPLES)), dtype=np.int32)
    for ps in range(5):
        for ts in range(5):
            table[ps, :, ts, :] = flush if ps == ts < 4 else plain
    return table.reshape(-1)


def _score_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pair_code [36, 36], triple_code [36, 36, 36], table): a 5-card hand made of
    cards (a, b) and (x, y, z) scores table[pair_code[a, b] + triple_code[x, y, z]],
    equal to fasteval.score5_batch. Entries for repeated cards are meaningless."""
    global _SCORE
    if _SCORE is None:
        rank, suit = np.arange(36) >> 2, np.arange(36) & 3
        pair_class = np.zeros((9, 9), dtype=np.intp)
        for i, (a, b) in enumerate(_RANK_PAIRS):
            pair_class[a, b] = pair_class[b, a] = i
        triple_class = np.zeros((9, 9, 9), dtype=np.intp)
        for j, t in enumerate(_RANK_TRIPLES):
            for p in set(permutations(t)):
                triple_class[p] = j
        ps = np.where(suit[:, None] == suit[None, :], suit[:, None], 4)
        pair_code = (ps * len(_RANK_PAIRS) + pair_class[rank[:, None], rank[None, :]]) * _NTC
        s0, s1, s2 = suit[:, None, None], suit[None, :, None], suit[None, None, :]
        ts = np.where((s0 == s1) & (s1 == s2), s0, 4)
        triple_code = ts * len(_RANK_TRIPLES) + triple_class[rank[:, None, None], rank[None, :, None],
                                                             rank[None, None, :]]
        arrays = (pair_code.astype(np.intp), triple_code.astype(np.intp), _build_score_table())
        for a in arrays:
            a.flags.writeable = False
        _SCORE = arrays
    return _SCORE


# ---- position tables shared by every spot

def _combos(n: int, k: int) -> np.ndarray:
    a = np.array(list(combinations(range(n), k)), dtype=np.intp)
    a.flags.writeable = False
    return a


_ALL = np.arange(36, dtype=np.intp)
_ALL.flags.writeable = False
_PAIRS = MappingProxyType({n: _combos(n, 2) for n in (26, 27, 28)})        # unseen-card pairs
_TRIPLES = MappingProxyType({w: _combos(w, 3) for w in (3, 4, 5)})         # board triples
_BOARD_PAIRS = MappingProxyType({w: _combos(w, 2) for w in (3, 4)})        # join a new card
_HOLE_A, _HOLE_B = fasteval.PAIR_IDX[:, 0], fasteval.PAIR_IDX[:, 1]       # the 10 pairs of a hole


def _rate(code: np.ndarray) -> float:
    """code [10, R] int8 per (pair, design hole): 0 we beat that pair, 1 tie, 2 it beats
    us. A hole's result is its worst pair; returns P(win) + P(tie) / 2 over the R holes."""
    worst = code.max(axis=0)
    return (np.count_nonzero(worst == 0) + 0.5 * np.count_nonzero(worst == 1)) / worst.size


def _versus(theirs: np.ndarray, ours) -> np.ndarray:
    """int8 code: 0 where theirs < ours, 1 where equal, 2 where theirs > ours."""
    code = (theirs >= ours).astype(np.int8)
    code += theirs > ours
    return code


def _read_only(a: np.ndarray, dtype=None) -> np.ndarray:
    out = np.array(a, dtype=dtype)       # a private copy
    out.flags.writeable = False
    return out


def _check(street, hole, board) -> tuple[Street, np.ndarray, np.ndarray]:
    """Validated (street, hole, board) as Street and intp arrays; ValueError otherwise."""
    try:
        st = Street(int(street))
        h = [int(c) for c in hole]
        b = [int(c) for c in board]
    except (TypeError, ValueError) as e:
        raise ValueError(f"bad street or cards: {street!r}, {hole!r}, {board!r}") from e
    if len(h) != 5:
        raise ValueError(f"a hole has 5 cards, got {h}")
    if len(b) != BOARD_CARDS_VISIBLE[st]:
        raise ValueError(f"{st.name} has {BOARD_CARDS_VISIBLE[st]} board cards, got {b}")
    cards = h + b
    if min(cards) < 0 or max(cards) > 35 or len(set(cards)) != len(cards):
        raise ValueError(f"cards must be distinct ints 0..35, got hole {h} board {b}")
    return st, np.array(h, dtype=np.intp), np.array(b, dtype=np.intp)


class Featurizer:
    """Deterministic features from the fixed designs (module docstring)."""

    def __init__(self, designs: Mapping[str, np.ndarray]) -> None:
        self._pair_code, self._triple_code, self._table = _score_tables()
        strength_rows = {}
        for n in (28, 27, 26):
            rows, _ = self._holes_of(designs, f"design_strength_{n}", n)
            strength_rows[n] = _read_only(rows.T, np.intp)                     # [10, R]
        self._strength_rows = MappingProxyType(strength_rows)
        tp, holes = self._holes_of(designs, "design_turn_pairs", 27)
        river = self._positions(designs, "design_turn_river", len(tp), 27)
        if (holes == river[:, None]).any():
            raise ValueError("design_turn_river: a river card is also in its hole")
        self._turn_cells = _read_only((tp * 27 + river[:, None]).T, np.intp)   # into [P, 27]
        fp, holes = self._holes_of(designs, "design_flop_pairs", 28)
        runout = self._positions(designs, "design_flop_runout", len(fp), len(_PAIRS[28]))
        turn_pos, river_pos = _PAIRS[28][runout, 0], _PAIRS[28][runout, 1]
        if (holes == turn_pos[:, None]).any() or (holes == river_pos[:, None]).any():
            raise ValueError("design_flop_runout: a runout card is also in its hole")
        self._flop_pairs = _read_only(fp.T, np.intp)                              # [10, Rf]
        self._flop_runout = _read_only(runout, np.intp)                           # [Rf]
        self._flop_turn_cell = _read_only((fp * 28 + turn_pos[:, None]).T, np.intp)
        self._flop_river_cell = _read_only((fp * 28 + river_pos[:, None]).T, np.intp)

    @staticmethod
    def _holes_of(designs, key: str, n: int) -> tuple[np.ndarray, np.ndarray]:
        """([R, 10] pair indices of range(n), [R, 5] the positions of each hole). Every row
        must be the 10 pairs of 5 distinct positions; anything else raises ValueError."""
        a = np.asarray(designs[key]).astype(np.intp)
        if a.ndim != 2 or a.shape[1] != 10 or len(a) == 0 or a.min() < 0 or a.max() >= len(_PAIRS[n]):
            raise ValueError(f"{key}: need [R, 10] pair indices of range({n}), got {a.shape}")
        ends = np.sort(_PAIRS[n][a].reshape(len(a), 20), axis=1)
        holes = ends[:, ::4]                     # a hole's 10 pairs name each position 4 times
        pair_id = np.zeros((n, n), dtype=np.intp)
        pair_id[_PAIRS[n][:, 0], _PAIRS[n][:, 1]] = np.arange(len(_PAIRS[n]))
        want = pair_id[holes[:, _HOLE_A], holes[:, _HOLE_B]]
        if (np.diff(holes, axis=1) <= 0).any() or (np.sort(want, axis=1) != np.sort(a, axis=1)).any():
            raise ValueError(f"{key}: a row is not the 10 pairs of one 5-card hole")
        return a, holes

    @staticmethod
    def _positions(designs, key: str, rows: int, limit: int) -> np.ndarray:
        a = np.asarray(designs[key]).astype(np.intp)
        if a.shape != (rows,) or a.min() < 0 or a.max() >= limit:
            raise ValueError(f"{key}: need [{rows}] indices below {limit}, got {a.shape}")
        return a

    def features(self, street, hole, board) -> Features:
        st, h, b = _check(street, hole, board)
        if st == Street.PREFLOP:
            s = preflop.preflop_strength([int(c) for c in h])
            return Features(s, s, False, 0)
        pc_of, tc_of, table = self._pair_code, self._triple_code, self._table
        mask = np.ones(36, dtype=bool)
        mask[h] = False
        mask[b] = False
        unseen = _ALL[mask]                                   # ascending: design positions
        n = len(unseen)
        pairs = _PAIRS[n]
        pc = pc_of[unseen[pairs[:, 0]], unseen[pairs[:, 1]]]  # [P] every opponent pair
        hc = pc_of[h[_HOLE_A], h[_HOLE_B]]                    # [10] our pairs
        tri = b[_TRIPLES[len(b)]]
        tc = tc_of[tri[:, 0], tri[:, 1], tri[:, 2]]           # [T] board triples
        pv = table[tc[:, None] + pc[None, :]].max(axis=0)     # [P] each pair's hand now
        ours = int(table[tc[:, None] + hc[None, :]].max())
        strength = _rate(_versus(pv, ours)[self._strength_rows[n]])
        nut = ours >= int(pv.max())
        category = ours >> 20
        if st == Street.RIVER:
            return Features(strength, strength, nut, category)
        if st == Street.TURN:
            equity = self._turn_equity(unseen, b, pc, hc, pv, ours)
        else:
            equity = self._flop_equity(unseen, b, pc, hc, pv, ours)
        return Features(strength, equity, nut, category)

    def _turn_equity(self, unseen, b, pc, hc, pv, ours) -> float:
        """Exact over the 27 river cards; design holes for the opponent."""
        table, tc_of = self._table, self._triple_code
        bp = b[_BOARD_PAIRS[4]]                                           # [6, 2]
        new = tc_of[unseen[None, :], bp[:, :1], bp[:, 1:]]                # [6, 27]: river + 2 board
        v = table[new[:, None, :] + pc[None, :, None]].max(axis=0)        # [P, 27]
        np.maximum(v, pv[:, None], out=v)
        o = table[new[:, None, :] + hc[None, :, None]].max(axis=(0, 1))   # [27]
        np.maximum(o, ours, out=o)
        return _rate(_versus(v, o[None, :]).ravel()[self._turn_cells])

    def _flop_equity(self, unseen, b, pc, hc, pv, ours) -> float:
        """Exact over the 378 runouts; design holes for the opponent, evaluated per cell."""
        table, tc_of = self._table, self._triple_code
        bp = b[_BOARD_PAIRS[3]]                                           # [3, 2]
        one = tc_of[unseen[None, :], bp[:, :1], bp[:, 1:]]                # [3, 28]: card + 2 flop
        run = _PAIRS[28]
        two = tc_of[unseen[run[:, 0]][None, :], unseen[run[:, 1]][None, :], b[:, None]]  # [3, 378]
        one_card = table[one[:, None, :] + pc[None, :, None]].max(axis=0).ravel()        # [P * 28]
        cells, k = self._flop_pairs, self._flop_runout
        v = pv[cells]                                                     # [10, Rf]
        np.maximum(v, one_card[self._flop_turn_cell], out=v)
        np.maximum(v, one_card[self._flop_river_cell], out=v)
        pcc = pc[cells]
        for i in range(3):
            np.maximum(v, table[pcc + two[i][k]], out=v)
        o1 = table[one[:, None, :] + hc[None, :, None]].max(axis=(0, 1))  # [28]
        o2 = table[two[:, None, :] + hc[None, :, None]].max(axis=(0, 1))  # [378]
        o = np.maximum(np.maximum(o1[run[:, 0]], o1[run[:, 1]]), o2)
        np.maximum(o, ours, out=o)
        return _rate(_versus(v, o[k][None, :]))


class BucketTables:
    """A loaded abstraction: bucket counts, edges and the feature designs (read-only)."""

    def __init__(self, arrays: Mapping[str, np.ndarray], path: Path | None = None) -> None:
        self.path = path
        try:
            fmt = int(np.asarray(arrays["format"]))
            nb = [int(x) for x in np.asarray(arrays["num_buckets"]).reshape(-1)]
            grids = {s: tuple(int(x) for x in np.asarray(arrays[f"{s.name.lower()}_grid"]).reshape(-1))
                     for s in (Street.FLOP, Street.TURN)}
        except KeyError as e:
            raise ValueError(f"{path}: not a bucket table, missing {e}") from None
        if fmt != FORMAT:
            raise ValueError(f"{path}: bucket table format {fmt}, this code reads {FORMAT}")
        if len(nb) != 4 or min(nb) < 1 or nb[Street.RIVER] <= NUT_BUCKETS:
            raise ValueError(f"{path}: bad num_buckets {nb}")
        for s, g in grids.items():
            if len(g) != 2 or min(g) < 1 or g[0] * g[1] != nb[s]:
                raise ValueError(f"{path}: {s.name} grid {g} does not make {nb[s]} buckets")
        self.num_buckets: Mapping[Street, int] = MappingProxyType(dict(zip(Street, nb)))
        self.grid: Mapping[Street, tuple[int, int]] = MappingProxyType(grids)
        self.preflop_edges = self._edges(arrays, "preflop_edges", (nb[Street.PREFLOP] - 1,))
        self.river_edges = self._edges(arrays, "river_edges", (nb[Street.RIVER] - NUT_BUCKETS - 1,))
        strength, equity = {}, {}
        for s, (rows, cols) in grids.items():
            name = s.name.lower()
            strength[s] = self._edges(arrays, f"{name}_strength_edges", (rows - 1,))
            equity[s] = self._edges(arrays, f"{name}_equity_edges", (rows, cols - 1))
        self.strength_edges: Mapping[Street, np.ndarray] = MappingProxyType(strength)
        self.equity_edges: Mapping[Street, np.ndarray] = MappingProxyType(equity)
        # bisect on tuples: the same cut as searchsorted(side="right"), faster for one value
        self._cuts = MappingProxyType({
            Street.PREFLOP: tuple(self.preflop_edges.tolist()),
            Street.RIVER: tuple(self.river_edges.tolist()),
            **{s: (tuple(strength[s].tolist()), tuple(tuple(r) for r in equity[s].tolist()))
               for s in grids},
        })
        self.featurizer = Featurizer(arrays)

    def _edges(self, arrays, key: str, shape: tuple[int, ...]) -> np.ndarray:
        if key not in arrays:
            raise ValueError(f"{self.path}: missing {key}")
        a = _read_only(arrays[key], np.float64)
        if a.shape != shape:
            raise ValueError(f"{self.path}: {key} has shape {a.shape}, expected {shape}")
        if not np.isfinite(a).all() or (a.size and (np.diff(a, axis=-1) < 0).any()):
            raise ValueError(f"{self.path}: {key} must be finite and non-decreasing")
        return a

    def features(self, street, hole, board) -> Features:
        return self.featurizer.features(street, hole, board)

    def bucket_from_features(self, street, f: Features) -> int:
        """The bucket of features f on this street (the cut rules in the module docstring)."""
        st = Street(int(street))
        cut = self._cuts[st]
        if st == Street.PREFLOP:
            return bisect.bisect_right(cut, f.strength)
        if st == Street.RIVER:
            if f.nut:
                top = self.num_buckets[st] - NUT_BUCKETS
                return top + min(max(int(f.category) - _LOWEST_NUT_CATEGORY, 0), NUT_BUCKETS - 1)
            return bisect.bisect_right(cut, f.equity)
        row = bisect.bisect_right(cut[0], f.strength)
        return row * self.grid[st][1] + bisect.bisect_right(cut[1][row], f.equity)

    def bucket(self, street, hole, board) -> int:
        return self.bucket_from_features(street, self.featurizer.features(street, hole, board))


def _num_buckets_in(path: Path) -> Mapping[Street, int]:
    """The file's bucket counts, or DEFAULT_NUM_BUCKETS if it can't be read. Never raises."""
    try:
        with np.load(path) as z:
            nb = [int(x) for x in z["num_buckets"].reshape(-1)]
        if len(nb) == 4 and min(nb) > 0:
            return MappingProxyType(dict(zip(Street, nb)))
    except Exception:          # missing or unreadable: bucket() reports it at call time
        pass
    return DEFAULT_NUM_BUCKETS


NUM_BUCKETS: Mapping[Street, int] = _num_buckets_in(DEFAULT_PATH)
_TABLES: BucketTables | None = None


def _load(path: Path) -> BucketTables:
    if not path.is_file():
        raise FileNotFoundError(f"card abstraction table not found: {path} "
                                "(build it with: python -m train.fit_buckets)")
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    return BucketTables(arrays, path)


def load_buckets(path: str | Path | None = None) -> BucketTables:
    """The tables at `path`; with no path, the default file, loaded once and cached."""
    global _TABLES
    if path is not None:
        return _load(Path(path))
    if _TABLES is None:
        tables = _load(DEFAULT_PATH)
        if dict(tables.num_buckets) != dict(NUM_BUCKETS):
            raise RuntimeError(f"{DEFAULT_PATH} changed after import: it has "
                               f"{dict(tables.num_buckets)} buckets, NUM_BUCKETS says {dict(NUM_BUCKETS)}")
        _TABLES = tables
    return _TABLES


def features(street, hole: Sequence[int], board: Sequence[int]) -> Features:
    """Features of the default tables (load_buckets())."""
    return load_buckets().features(street, hole, board)


def bucket(street, hole: Sequence[int], board: Sequence[int], iters: int = 0, rng=None) -> int:
    """Bucket id in range(NUM_BUCKETS[street]) for our hole on this board.

    Deterministic: `iters` and `rng` are accepted for the plan's interface and ignored
    (no rollouts, rng untouched), so any budget, iters <= 0 included, gives the bucket
    the trainer used. There is no minimum iters to recommend; the cost is fixed (module
    docstring). Raises ValueError for a bad street or bad cards, FileNotFoundError if
    data/buckets.npz is missing.
    """
    return load_buckets().bucket(street, hole, board)


def warmup() -> None:
    """Build the evaluator and score tables and load every table (about 50 ms).

    Raises FileNotFoundError if data/buckets.npz or data/preflop.npz is missing.
    """
    fasteval.warmup()
    preflop.load_preflop()
    load_buckets()
