"""Offline builder: python -m train.build_preflop --rollouts 3000 --workers 4 --seed 0."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from bot.preflop import canonical_key

CHUNK = 20


def _work(args):
    seed, idx, hands, rollouts = args
    from bot import fasteval
    from bot.equity import equity_vs_random
    fasteval.warmup()
    rng = np.random.default_rng([seed, idx])
    return idx, [equity_vs_random(h, [], rollouts, rng) for h in hands]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/preflop.npz")
    a = ap.parse_args()
    t0 = time.time()

    combos = list(combinations(range(36), 5))
    masks = np.array([sum(1 << c for c in h) for h in combos], dtype=np.int64)
    keys = np.array([canonical_key(h) for h in combos], dtype=np.int64)
    uniq, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    n = len(uniq)
    print(f"classes: {n}", flush=True)
    reps = [list(combos[i]) for i in first]
    jobs = [(a.seed, i // CHUNK, reps[i:i + CHUNK], a.rollouts) for i in range(0, n, CHUNK)]
    eq = np.zeros(n, dtype=np.float32)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, (idx, vals) in enumerate(pool.imap_unordered(_work, jobs)):
            eq[idx * CHUNK:idx * CHUNK + len(vals)] = vals
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} chunks", flush=True)
    order = np.argsort(masks)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, masks=masks[order], strength=eq[inv.ravel()][order],
                        rollouts=a.rollouts, n_classes=n, seed=a.seed)
    print(f"wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
