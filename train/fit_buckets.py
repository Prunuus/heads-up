"""Offline fit of the card abstraction (bot/abstraction.py) into data/buckets.npz.

    python -m train.fit_buckets --per-street 200000 --iters 2000 --workers 4 --seed 0 \\
        --out data/buckets.npz

Steps
1. Designs: the fixed opponent-hole designs of the deterministic features, from --seed.
2. Preflop edges: exact quantiles of data/preflop.npz over all 376,992 holes, no sampling.
3. Flop, turn, river: --per-street uniform random spots each, features computed in a
   spawn Pool, then the edges.
   - flop and turn: strength rows first, then equity columns fitted within each row;
   - river: equity edges of the hands that are not the nuts (the 5 nut buckets go by
     category and need no edges).
4. Validation: --validate spots per postflop street, compared with bot.equity's Monte
   Carlo at --iters rollouts (strength_now, equity_vs_random). The rms errors are stored.

Every chunk of spots draws from its own seed (seed, street, chunk), so the file does not
depend on --workers. Runs on Linux and macOS with requirements-train.txt.
train/README.md has the runbook.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from bot import abstraction as ab
from bot.fasteval import PAIR_IDX
from game import Street

REPO = Path(__file__).resolve().parent.parent
POSTFLOP = (Street.FLOP, Street.TURN, Street.RIVER)
WIDTH = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
VALIDATION_CHUNKS = 1_000_000          # validation spots use chunk ids from here up

STRENGTH_ROWS = 4096                   # design holes for strength, flop and turn
RIVER_ROWS = 16384                     # river strength is also the river equity
TURN_PER_RIVER = 304                   # design holes per river card (27 cards: 8,208)
FLOP_PER_RUNOUT = 8                    # design holes per runout (378 runouts: 3,024)


# ---- designs

def _pair_index(n: int) -> np.ndarray:
    """[n, n]: index of pair (a, b), a < b, in combinations(range(n), 2) order."""
    idx = np.full((n, n), -1, dtype=np.intp)
    p = np.array(list(combinations(range(n), 2)), dtype=np.intp)
    idx[p[:, 0], p[:, 1]] = np.arange(len(p))
    return idx


def _holes(rng: np.random.Generator, n: int, count: int) -> np.ndarray:
    """count sorted 5-position holes of range(n): shuffles cut into blocks of 5, so every
    position appears about equally often."""
    per = n // 5
    shuffles = rng.random((-(-count // per), n)).argsort(axis=1)[:, :5 * per]
    return np.sort(shuffles.reshape(-1, 5)[:count], axis=1)


def _pairs_of(holes: np.ndarray, n: int) -> np.ndarray:
    """[count, 10] pair indices (combinations(range(n), 2) order) of each hole."""
    return _pair_index(n)[holes[:, PAIR_IDX[:, 0]], holes[:, PAIR_IDX[:, 1]]].astype(np.int16)


def make_designs(seed: int, strength_rows: int = STRENGTH_ROWS, river_rows: int = RIVER_ROWS,
                 turn_per_river: int = TURN_PER_RIVER,
                 flop_per_runout: int = FLOP_PER_RUNOUT) -> dict[str, np.ndarray]:
    """The design arrays bot.abstraction.Featurizer reads (format: its module docstring)."""
    rng = np.random.default_rng([seed, 1])        # the spots use streams [seed, street, chunk]
    d = {}
    for n, rows in ((28, strength_rows), (27, strength_rows), (26, river_rows)):
        d[f"design_strength_{n}"] = _pairs_of(_holes(rng, n, rows), n)
    pairs, river = [], []
    for j in range(27):                           # every river card, the same number of holes
        others = np.delete(np.arange(27), j)
        pairs.append(_pairs_of(others[_holes(rng, 26, turn_per_river)], 27))
        river.append(np.full(turn_per_river, j, dtype=np.int16))
    d["design_turn_pairs"], d["design_turn_river"] = np.concatenate(pairs), np.concatenate(river)
    pairs, runout = [], []
    for k, (t, r) in enumerate(combinations(range(28), 2)):    # every runout, the same number
        others = np.delete(np.arange(28), [t, r])
        pairs.append(_pairs_of(others[_holes(rng, 26, flop_per_runout)], 28))
        runout.append(np.full(flop_per_runout, k, dtype=np.int16))
    d["design_flop_pairs"], d["design_flop_runout"] = np.concatenate(pairs), np.concatenate(runout)
    return d


# ---- edges

def fit_edges(x, nb: int) -> np.ndarray:
    """nb - 1 non-decreasing edges cutting x into nb buckets of about equal mass, for
    bucket = searchsorted(edges, v, side="right") (an edge opens its bucket).

    Each cut goes to the boundary between distinct values nearest its quantile. A run of
    equal values is never split: it goes whole to the side nearer the quantile.
    """
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = len(xs)
    if n < nb:
        raise ValueError(f"{n} values cannot fill {nb} buckets")
    edges = np.empty(nb - 1)
    for k in range(1, nb):
        i = min(max(round(k * n / nb), 1), n - 1)   # the cut: xs[:i] below, xs[i:] above
        v = xs[i]
        if xs[i - 1] < v:
            edges[k - 1] = v
            continue
        lo = int(np.searchsorted(xs, v, "left"))    # equal values xs[lo:hi] straddle the cut
        hi = int(np.searchsorted(xs, v, "right"))
        edges[k - 1] = v if (i - lo <= hi - i or hi == n) else xs[hi]
    return edges


def _grid(text: str) -> tuple[int, int]:
    rows, cols = (int(x) for x in text.lower().split("x"))
    if rows < 1 or cols < 1:
        raise argparse.ArgumentTypeError(f"bad grid {text!r}")
    return rows, cols


# ---- workers (spawn: everything they need arrives through _init)

_FEAT: ab.Featurizer | None = None


def _init(designs: dict[str, np.ndarray]) -> None:
    global _FEAT
    from bot import equity, preflop
    equity.warmup()
    preflop.load_preflop()
    _FEAT = ab.Featurizer(designs)


def _spot_cards(seed: int, street: Street, chunk: int, count: int) -> np.ndarray:
    rng = np.random.default_rng([seed, int(street), chunk])
    return rng.random((count, 36)).argsort(axis=1)[:, :5 + WIDTH[street]]


def _features(job) -> tuple[int, int, np.ndarray]:
    """[count, 4]: strength, equity, nut, category of one chunk of random spots."""
    street, seed, chunk, count = job
    out = np.empty((count, 4))
    for i, c in enumerate(_spot_cards(seed, street, chunk, count)):
        f = _FEAT.features(street, c[:5], c[5:])
        out[i] = f.strength, f.equity, f.nut, f.category
    return int(street), chunk, out


def _validation(job) -> tuple[int, int, np.ndarray]:
    """[count, 4]: our strength and equity, then Monte Carlo strength_now and equity."""
    from bot.equity import equity_vs_random, strength_now
    street, seed, chunk, count, iters = job
    rng = np.random.default_rng([seed, int(street), chunk, 1])
    out = np.empty((count, 4))
    for i, c in enumerate(_spot_cards(seed, street, chunk, count)):
        h, b = [int(x) for x in c[:5]], [int(x) for x in c[5:]]
        f = _FEAT.features(street, h, b)
        out[i] = f.strength, f.equity, strength_now(h, b, iters, rng), equity_vs_random(h, b, iters, rng)
    return int(street), chunk, out


def _run(fn, jobs: list, workers: int, designs: dict) -> list:
    if workers <= 1:
        _init(designs)
        return [fn(j) for j in jobs]
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(designs,)) as pool:
        return list(pool.imap_unordered(fn, jobs))


def _chunks(street: Street, seed: int, total: int, size: int, first: int = 0) -> list:
    return [(street, seed, first + c, min(size, total - c * size)) for c in range(-(-total // size))]


def _gather(results: list, street: Street) -> np.ndarray:
    parts = sorted((chunk, out) for s, chunk, out in results if s == int(street))
    return np.concatenate([out for _, out in parts])


# ---- the fit

def fit(features: dict[Street, np.ndarray], preflop_strength: np.ndarray, nb_preflop: int,
        grids: dict[Street, tuple[int, int]], nb_river: int) -> dict[str, np.ndarray]:
    """Edges from the features ([N, 4] per postflop street) and every hole's preflop strength."""
    out = {"preflop_edges": fit_edges(preflop_strength, nb_preflop)}
    for street, (rows, cols) in grids.items():
        s, e = features[street][:, 0], features[street][:, 1]
        se = fit_edges(s, rows)
        row = np.searchsorted(se, s, side="right")
        name = street.name.lower()
        out[f"{name}_strength_edges"] = se
        out[f"{name}_equity_edges"] = np.array(
            [fit_edges(e[row == r], cols) for r in range(rows)]).reshape(rows, cols - 1)
    river = features[Street.RIVER]
    out["river_edges"] = fit_edges(river[river[:, 2] == 0, 1], nb_river - ab.NUT_BUCKETS)
    return out


def _occupancy(tables: ab.BucketTables, street: Street, f: np.ndarray) -> np.ndarray:
    counts = np.zeros(tables.num_buckets[street], dtype=np.int64)
    for s, e, nut, cat in f:
        counts[tables.bucket_from_features(street, ab.Features(s, e, bool(nut), int(cat)))] += 1
    return counts / (len(f) / len(counts))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-street", type=int, default=200_000, help="random spots per postflop street")
    ap.add_argument("--iters", type=int, default=2000, help="Monte Carlo rollouts per validation spot")
    ap.add_argument("--validate", type=int, default=2000, help="validation spots per postflop street (0: none)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "data" / "buckets.npz"))
    ap.add_argument("--preflop", type=int, default=ab.DEFAULT_NUM_BUCKETS[Street.PREFLOP],
                    help="preflop buckets")
    ap.add_argument("--flop-grid", type=_grid, default=(10, 10), help="strength rows x equity columns")
    ap.add_argument("--turn-grid", type=_grid, default=(10, 10), help="strength rows x equity columns")
    ap.add_argument("--river", type=int, default=ab.DEFAULT_NUM_BUCKETS[Street.RIVER],
                    help=f"river buckets, the top {ab.NUT_BUCKETS} for the nuts")
    ap.add_argument("--preflop-table", default=str(REPO / "data" / "preflop.npz"))
    ap.add_argument("--chunk", type=int, default=2000, help="spots per job")
    a = ap.parse_args(argv)
    if a.per_street < 1 or a.river <= ab.NUT_BUCKETS or a.preflop < 1:
        ap.error("need --per-street >= 1, --preflop >= 1 and --river > 5")
    t0 = time.time()
    grids = {Street.FLOP: a.flop_grid, Street.TURN: a.turn_grid}

    designs = make_designs(a.seed)
    with np.load(a.preflop_table) as z:
        preflop_strength = z["strength"].astype(np.float64)

    jobs = [j for st in POSTFLOP for j in _chunks(st, a.seed, a.per_street, a.chunk)]
    results = _run(_features, jobs, a.workers, designs)
    feats = {st: _gather(results, st) for st in POSTFLOP}
    t_feat = time.time() - t0
    print(f"features: {3 * a.per_street} spots in {t_feat:.1f} s "
          f"({3 * a.per_street / t_feat:.0f} spots/s on {a.workers} workers)", flush=True)

    edges = fit(feats, preflop_strength, a.preflop, grids, a.river)
    arrays = {
        "format": np.int64(ab.FORMAT),
        "num_buckets": np.array([a.preflop, a.flop_grid[0] * a.flop_grid[1],
                                 a.turn_grid[0] * a.turn_grid[1], a.river], dtype=np.int64),
        "flop_grid": np.array(a.flop_grid, dtype=np.int64),
        "turn_grid": np.array(a.turn_grid, dtype=np.int64),
        **edges, **designs,
        "seed": np.int64(a.seed),
        "fit_spots": np.array([len(preflop_strength)] + [len(feats[st]) for st in POSTFLOP], dtype=np.int64),
        "validate_iters": np.int64(a.iters),
        "validate_spots": np.int64(a.validate),
    }

    # rms error and mean signed error vs Monte Carlo, per postflop street:
    # [strength rms, strength mean, equity rms, equity mean]
    validation = np.full((3, 4), np.nan)
    if a.validate > 0:
        t1 = time.time()
        jobs = [j + (a.iters,) for st in POSTFLOP
                for j in _chunks(st, a.seed, a.validate, max(1, a.chunk // 10), VALIDATION_CHUNKS)]
        results = _run(_validation, jobs, a.workers, designs)
        for k, st in enumerate(POSTFLOP):
            v = _gather(results, st)
            ds, de = v[:, 0] - v[:, 2], v[:, 1] - v[:, 3]
            validation[k] = np.sqrt(np.mean(ds ** 2)), ds.mean(), np.sqrt(np.mean(de ** 2)), de.mean()
            print(f"validation {st.name}: strength rms {validation[k, 0]:.4f} mean {validation[k, 1]:+.4f}, "
                  f"equity rms {validation[k, 2]:.4f} mean {validation[k, 3]:+.4f} "
                  f"(Monte Carlo sd at {a.iters} rollouts <= {0.5 / np.sqrt(a.iters):.4f})", flush=True)
        print(f"validation: {3 * a.validate} spots in {time.time() - t1:.1f} s", flush=True)
    arrays["validation"] = validation

    tables = ab.BucketTables(arrays)       # the same checks the runtime loader makes
    for st in POSTFLOP:
        occ = _occupancy(tables, st, feats[st])
        print(f"fit-sample occupancy {st.name}: min {occ.min():.2f} max {occ.max():.2f} x expected")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), wall {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
