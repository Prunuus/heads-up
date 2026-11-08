"""Export a verified MCCFR checkpoint (or final output) to the runtime blueprint (Task 19).

    python -m train.export train_data/mccfr-b400/blueprint_raw.npz data/blueprint.npz

Reads through train.mccfr.load_checkpoint (tree, constants, bucket table, shapes, dtypes all
verified), takes the average strategy normalized over the legal actions, stores it as float16
[n_info, 4] in the trainer's own row order (the format is in bot/strategy.py), and writes
atomically. The result is re-loaded with bot.strategy.Blueprint.load as a check before the
rename. About 8 bytes per infoset: 51 MB at 400 buckets a street.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import game
from bot import abstraction as ab
from bot import strategy
from bot.tree import BettingTree, get_tree
from train import mccfr


def export(src, out, *, tree: BettingTree | None = None, buckets_table=ab.DEFAULT_PATH) -> Path:
    """Write the blueprint for checkpoint `src` to `out`. buckets_table None: synthetic data."""
    tree = get_tree() if tree is None else tree
    ck = mccfr.load_checkpoint(src, tree, buckets_table)       # raises ValueError on any mismatch
    m = ck.meta
    probs = mccfr.average_strategy(ck.avg, tree, ck.offset)
    legal = np.repeat(tree.child != -1, np.diff(ck.offset), axis=0)
    table = probs.astype(np.float16)
    table[~legal] = 0
    arrays = {
        "format": np.int64(strategy.FORMAT), "probs": table,
        "stack": np.int64(m["starting_stack"]), "sb": np.int64(m["small_blind"]),
        "bb": np.int64(m["big_blind"]), "cap": np.int64(m["raise_cap"]),
        "iterations": np.int64(m["iterations"]), "seed": np.int64(m["seed"]),
        "linear": np.bool_(m["linear"]), "n_dec": np.int64(m["n_dec"]), "n_term": np.int64(m["n_term"]),
        "tree_child_sha256": np.array(m["tree_child_sha256"]),
        "buckets_sha256": np.array(m["buckets_sha256"]),
        "features_sha256": np.array(m["features_sha256"]),
        "buckets": np.array(m["num_buckets"], dtype=np.int64),
        "dataset_path": np.array(m["dataset_path"]), "dataset_n": np.int64(m["dataset_n"]),
        "dataset_seed": np.int64(m["dataset_seed"]),
        "dataset_shard_size": np.int64(m["dataset_shard_size"]),
        "dataset_format": np.int64(m["dataset_format"]),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    try:
        np.savez(tmp, **arrays)
        bp = strategy.Blueprint.load(tmp, tree=tree,
                                     buckets_path=None if buckets_table is None else buckets_table)
        assert bp.meta["iterations"] == m["iterations"]
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="a checkpoint or final output of train.mccfr")
    ap.add_argument("out", nargs="?", default=str(strategy.DEFAULT_PATH))
    a = ap.parse_args(argv)
    out = export(a.src, a.out)
    print(f"exported {a.src} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
