"""Offline fit of bot.features' range model: python -m train.fit_range_model --spots 100000.

For random spots on each postflop street it records our equity vs random and nut flags,
the opponent's equity vs random (both 128 rollouts) and who wins at showdown. For each
threshold t = 0.0, 0.1, ..., 0.9 it fits, on the spots where the opponent's equity is
>= t, a logistic model (Newton's method, small L2):

    logit P(win) = a + b * logit(eq) + c * [nuts now] + d * [flush possible, not the nuts]

and prints the rows to paste into bot/features.py:_RANGE_COEF (ties count as half a win).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

TS = tuple(round(0.1 * k, 1) for k in range(10))
ITERS = 128
CHUNK = 2500


def _spots(args):
    seed, idx, n, width = args
    from bot import equity, fasteval
    equity.warmup()
    rng = np.random.default_rng([seed, width, idx])
    out = np.empty((n, 4))
    for i in range(n):
        c = rng.permutation(36)
        hole, opp, board = c[:5], c[5:10], c[10:15]
        seen = board[:width]
        e = equity.equity_vs_random(hole, seen, ITERS, rng)
        o = equity.equity_vs_random(opp, seen, ITERS, rng)
        a = fasteval.omaha_batch(hole[None], board[None])[0]
        b = fasteval.omaha_batch(opp[None], board[None])[0]
        out[i] = (e, o, 1.0 if a > b else 0.5 if a == b else 0.0, equity.nut_flags(hole, seen))
    return out


def _logit(x):
    x = np.clip(x, 0.005, 0.995)
    return np.log(x / (1.0 - x))


def fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 40, lam: float = 1e-2) -> np.ndarray:
    """Weights minimizing cross-entropy (soft labels allowed) + lam/2 |w|^2."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - y) + lam * w
        H = (X * (p * (1.0 - p))[:, None]).T @ X + lam * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def fit_street(data: np.ndarray) -> np.ndarray:
    """(len(TS), 4) rows of (a, b, c, d) for one street's (eq, opp_eq, win, nut) rows."""
    e, o, w, flags = data.T
    flags = flags.astype(int)
    nut = (flags & 1) > 0
    flush_nonnut = ((flags & 4) > 0) & ~nut
    X = np.stack([np.ones_like(e), _logit(e), nut, flush_nonnut], axis=1).astype(float)
    return np.array([fit_logistic(X[o >= t], w[o >= t]) for t in TS])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", type=int, default=100_000, help="spots per street")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--out", default="train_data/range_model.npz")
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(a.seed, k, min(CHUNK, a.spots - s), width)
            for width in (3, 4, 5) for k, s in enumerate(range(0, a.spots, CHUNK))]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        parts = pool.map(_spots, jobs, chunksize=1)
    per = {3: [], 4: [], 5: []}
    for (_, _, _, width), part in zip(jobs, parts):
        per[width].append(part)
    tables = {}
    for width, name in ((3, "FLOP"), (4, "TURN"), (5, "RIVER")):
        data = np.concatenate(per[width])
        tables[name] = fit_street(data)
        rows = [f"({r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}, {r[3]:.2f})" for r in tables[name]]
        lines = [", ".join(rows[k:k + 3]) for k in range(0, len(rows), 3)]
        print(f"    # {name}\n    (" + ",\n     ".join(lines) + "),")
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **tables, ts=np.array(TS), spots=a.spots, seed=a.seed)
    print(f"wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
