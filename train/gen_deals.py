"""Training deal dataset for the CFR blueprint (Task 17): shards train_data/deals_<k>.npz.

    python -m train.gen_deals --n 2_000_000 --workers 8 --iters 1000
    python -m train.gen_deals --rebucket --buckets data/buckets.npz     # after a refit

The MCCFR trainer (Task 18) samples one row per iteration, so these rows are the game the
blueprint is solved for. A row is one uniform random deal of the 36-card deck:
- cards int8 [15]: the button's hole (5, sorted), the big blind's hole (5, sorted), then
  the board as dealt (flop 3, turn, river).
- buckets int16 [2, 4]: seat (0 = button, 1 = big blind) x street. Each one is
  bot.abstraction's bucket(street, hole, board) under the table the shard names.
- showdown int8: +1 the button wins, 0 split, -1 the big blind wins (evaluator.omaha_score,
  the engine's showdown).
- features float64 [2, 6] (FEATURE_COLUMNS), river_nut bool [2], river_category int8 [2]:
  what BucketTables.bucket_from_features reads. A refit with other bucket counts (same
  designs) is then a re-bucketing, seconds per million deals, not a regeneration.
  - float64, because float32 is not exact here. Flop and turn equity are multiples of
    1/6048 and 1/16416, not dyadic. A bucket edge is a fit-sample value, so many deals sit
    exactly on one, and a rounded copy would fall into the bucket below it.

Shards (format FORMAT)
- Shard k holds deals [k * shard_size, k * shard_size + count) of the stream for --seed.
  Its cards come only from the rng stream [seed, STREAM, k], so the dataset depends on
  --seed, --shard-size and the numpy version, never on --workers. A shorter shard is a
  prefix of the full one.
- Each shard also stores: format, seed, shard, shard_size, numpy_version; buckets_sha256
  (of the bucket table file) and the num_buckets, flop_grid and turn_grid it implies;
  features_sha256 (of the table's feature designs and data/preflop.npz: what the features
  depend on). A change to the feature code itself is not hashed: regenerate after one.
- Writes are atomic (a temp file, then a rename). A rerun skips finished shards and makes
  the missing ones. A larger --n adds shards on top, and a short last shard is redone at
  full size.

Reading: load_deals() concatenates the shards of a directory. It checks that they form one
dataset, and that its bucket table is the one the trainer will use. rebucket() recomputes
the buckets against another table. train/README.md has the runbook.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import multiprocessing as mp
import os
import re
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

import evaluator as ev
from bot import abstraction as ab
from game import BOARD_CARDS_VISIBLE, Street

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "train_data"
PREFLOP_TABLE = REPO / "data" / "preflop.npz"       # the preflop strengths bot.preflop reads
FORMAT = 1
SHARD_SIZE = 50_000
STREAM = 17             # rng [seed, STREAM, shard]; train/fit_buckets.py uses [seed, 1], [seed, street, ...]
FEATURE_COLUMNS = ("preflop strength", "flop strength", "flop equity",
                   "turn strength", "turn equity", "river equity")
SHARD_NAME = re.compile(r"deals_(\d+)\.npz")
_IDENTITY = {"format": "format", "seed": "seed", "shard_size": "shard size",
             "features_sha256": "features sha256", "buckets_sha256": "bucket table sha256",
             "num_buckets": "num_buckets", "flop_grid": "flop grid", "turn_grid": "turn grid"}


def _say(msg: str) -> None:
    print(msg, flush=True)


# ---- the bucket table and what identifies it

class Table(NamedTuple):
    tables: ab.BucketTables
    buckets_sha256: str     # sha256 of the table file's bytes
    features_sha256: str    # sha256 of its design arrays and of data/preflop.npz


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def open_table(path) -> Table:
    """The bucket table at path, its hashes taken from the very bytes it was loaded from."""
    path = Path(path)
    data = path.read_bytes()
    with np.load(io.BytesIO(data)) as z:
        arrays = {k: z[k] for k in z.files}
    tables = ab.BucketTables(arrays, path)
    if max(tables.num_buckets.values()) > np.iinfo(np.int16).max:
        raise ValueError(f"{path}: bucket ids beyond int16")
    h = hashlib.sha256()
    for k in sorted(k for k in arrays if k.startswith("design_")):
        a = np.ascontiguousarray(arrays[k])
        h.update(f"{k} {a.dtype.str} {a.shape}\n".encode())
        h.update(a.tobytes())
    h.update(hashlib.sha256(PREFLOP_TABLE.read_bytes()).digest())
    return Table(tables, hashlib.sha256(data).hexdigest(), h.hexdigest())


def _table_meta(t: Table) -> dict[str, np.ndarray]:
    nb, grid = t.tables.num_buckets, t.tables.grid
    return {"buckets_sha256": np.array(t.buckets_sha256), "features_sha256": np.array(t.features_sha256),
            "num_buckets": np.array([nb[s] for s in Street], dtype=np.int64),
            "flop_grid": np.array(grid[Street.FLOP], dtype=np.int64),
            "turn_grid": np.array(grid[Street.TURN], dtype=np.int64)}


# ---- one shard

def deal_cards(seed: int, shard: int, count: int) -> np.ndarray:
    """int8 [count, 15]: the deals of one shard (a smaller count gives a prefix)."""
    rng = np.random.default_rng([seed, STREAM, shard])
    c = rng.random((count, 36)).argsort(axis=1)[:, :15]
    c[:, 0:5].sort(axis=1)
    c[:, 5:10].sort(axis=1)
    return c.astype(np.int8)


def _deal_features(tables: ab.BucketTables, c: list[int]):
    """Features [2, 6], river nut [2], river category [2] and buckets [2, 4] of one deal
    (15 Python ints), through the runtime's own features() and bucket_from_features()."""
    feats, nut = np.empty((2, 6)), np.zeros(2, bool)
    cat, buckets = np.zeros(2, np.int8), np.empty((2, 4), np.int16)
    board = c[10:15]
    for seat in (0, 1):
        hole = c[5 * seat:5 * seat + 5]
        f = [tables.features(st, hole, board[:BOARD_CARDS_VISIBLE[st]]) for st in Street]
        buckets[seat] = [tables.bucket_from_features(st, f[st]) for st in Street]
        pre, flop, turn, river = f
        feats[seat] = pre.strength, flop.strength, flop.equity, turn.strength, turn.equity, river.equity
        nut[seat], cat[seat] = river.nut, river.category
    return feats, nut, cat, buckets


def make_shard(t: Table, seed: int, shard: int, count: int, shard_size: int) -> dict[str, np.ndarray]:
    """Every array of shard `shard` (count rows), as written to its file."""
    cards = deal_cards(seed, shard, count)
    feats, nut = np.empty((count, 2, 6)), np.zeros((count, 2), bool)
    cat, buckets = np.zeros((count, 2), np.int8), np.empty((count, 2, 4), np.int16)
    showdown = np.empty(count, np.int8)
    for i, c in enumerate(cards.tolist()):                # Python ints: evaluator shifts 1 << card
        feats[i], nut[i], cat[i], buckets[i] = _deal_features(t.tables, c)
        button, big_blind = ev.omaha_score(c[0:5], c[10:15]), ev.omaha_score(c[5:10], c[10:15])
        showdown[i] = (button > big_blind) - (button < big_blind)
    if not np.array_equal(buckets_from_features(t.tables, feats, nut, cat), buckets):
        raise RuntimeError(f"shard {shard}: buckets_from_features disagrees with bucket_from_features")
    return {"format": np.int64(FORMAT), "seed": np.int64(seed), "shard": np.int64(shard),
            "shard_size": np.int64(shard_size), "numpy_version": np.array(np.__version__),
            **_table_meta(t), "cards": cards, "buckets": buckets, "showdown": showdown,
            "features": feats, "river_nut": nut, "river_category": cat}


def buckets_from_features(tables: ab.BucketTables, features, river_nut, river_category) -> np.ndarray:
    """int16 [..., 4]: BucketTables.bucket_from_features on every street, vectorized.

    features [..., 6] in FEATURE_COLUMNS order, river_nut and river_category [...]. A value
    on an edge goes to the upper bucket: an id counts the edges <= the value, as bisect_right.
    """
    f = np.asarray(features, dtype=np.float64)
    out = np.empty(f.shape[:-1] + (4,), dtype=np.int16)
    out[..., Street.PREFLOP] = np.searchsorted(tables.preflop_edges, f[..., 0], side="right")
    for st, col in ((Street.FLOP, 1), (Street.TURN, 3)):
        row = np.searchsorted(tables.strength_edges[st], f[..., col], side="right")
        cut = tables.equity_edges[st][row]                     # [..., cols - 1]: this row's edges
        out[..., st] = row * tables.grid[st][1] + (cut <= f[..., col + 1, None]).sum(axis=-1)
    top = tables.num_buckets[Street.RIVER] - ab.NUT_BUCKETS     # the nuts: top 5, trips and up
    nut = top + np.clip(np.asarray(river_category, np.int64) - ev.THREE_OF_A_KIND, 0, ab.NUT_BUCKETS - 1)
    out[..., Street.RIVER] = np.where(river_nut, nut,
                                      np.searchsorted(tables.river_edges, f[..., 5], side="right"))
    return out


def shard_path(out, shard: int) -> Path:
    return Path(out) / f"deals_{shard:05d}.npz"


def _write(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Atomic: a reader sees the old file or the whole new one, never a partial one."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)                 # uncompressed: loading is a plain read
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ---- reading shards

def _meta(path: Path) -> dict:
    with np.load(path) as z:
        try:
            m = {"format": int(z["format"]), "seed": int(z["seed"]), "shard": int(z["shard"]),
                 "shard_size": int(z["shard_size"]), "numpy_version": str(z["numpy_version"]),
                 "buckets_sha256": str(z["buckets_sha256"]), "features_sha256": str(z["features_sha256"]),
                 "num_buckets": tuple(int(x) for x in z["num_buckets"]),
                 "flop_grid": tuple(int(x) for x in z["flop_grid"]),
                 "turn_grid": tuple(int(x) for x in z["turn_grid"]), "count": len(z["showdown"])}
        except KeyError as e:
            raise ValueError(f"{path}: not a deal shard, missing {e}") from None
    if m["format"] != FORMAT:
        raise ValueError(f"{path}: deal shard format {m['format']}, this code reads {FORMAT}")
    match = SHARD_NAME.fullmatch(path.name)
    if match and int(match.group(1)) != m["shard"]:
        raise ValueError(f"{path}: holds shard {m['shard']}")
    return m


def _scan(path) -> list[tuple[Path, dict]]:
    """[(file, meta)] by shard number: every deals_<k>.npz of a directory, or one file."""
    p = Path(path)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = [f for f in p.iterdir() if SHARD_NAME.fullmatch(f.name)]
    else:
        files = []
    return sorted(((f, _meta(f)) for f in files), key=lambda fm: fm[1]["shard"])


def _agree(shards) -> None:
    """ValueError unless every shard has the first shard's identity (_IDENTITY)."""
    ref_name, ref = shards[0][0].name, shards[0][1]
    for f, m in shards:
        for k in _IDENTITY:
            if m[k] != ref[k]:
                raise ValueError(f"{f}: {_IDENTITY[k]} {m[k]!r}, but {ref_name} has {ref[k]!r}")


def _dataset(path) -> list[tuple[Path, dict]]:
    """The shards of one complete dataset: they agree, and form shards 0..K-1 of full size
    (the last may be short). Anything else raises ValueError."""
    shards = _scan(path)
    if not shards:
        raise ValueError(f"no deal shards (deals_<k>.npz) at {path}")
    _agree(shards)
    if Path(path).is_dir():
        have = [m["shard"] for _, m in shards]
        if have != list(range(len(have))):
            missing = sorted(set(range(max(have) + 1)) - set(have))
            raise ValueError(f"{path}: shards {missing} missing (an interrupted run? rerun gen_deals "
                             "with the same arguments)")
        for f, m in shards[:-1]:
            if m["count"] != m["shard_size"]:
                raise ValueError(f"{f}: holds {m['count']} of {m['shard_size']} deals before the last "
                                 "shard (an interrupted run? rerun gen_deals with the same arguments)")
    return shards


class Deals(NamedTuple):
    buckets: np.ndarray                   # int16 [N, 2, 4]: seat (0 = button) x street
    showdown: np.ndarray                  # int8 [N]: +1 button wins, 0 split, -1 big blind wins
    meta: dict
    cards: np.ndarray | None = None       # int8 [N, 15], with cards=True
    features: dict | None = None          # features, river_nut, river_category, with features=True


def load_deals(path, buckets_table=ab.DEFAULT_PATH, n: int | None = None, cards: bool = False,
               features: bool = False) -> Deals:
    """The deals of a dataset directory (or of one shard file), the first n if given.

    Raises ValueError if the shards do not form one complete dataset, or if they were
    bucketed with a table other than buckets_table (its sha256 must match; None skips this
    check, for synthetic data). Reads only the arrays asked for, into contiguous arrays.
    """
    shards = _dataset(path)
    first = shards[0][1]
    if buckets_table is not None and first["buckets_sha256"] != file_sha256(buckets_table):
        raise ValueError(f"{path}: bucketed with the table of sha256 {first['buckets_sha256'][:16]}..., "
                         f"not {buckets_table} (sha256 {file_sha256(buckets_table)[:16]}...). Rebucket it: "
                         f"python -m train.gen_deals --rebucket --out {path} --buckets {buckets_table}")
    total = sum(m["count"] for _, m in shards)
    n = total if n is None else n
    if not 0 < n <= total:
        raise ValueError(f"{path}: holds {total} deals, asked for {n}")
    names = {"buckets": ((2, 4), np.int16), "showdown": ((), np.int8)}
    if cards:
        names["cards"] = ((15,), np.int8)
    if features:
        names |= {"features": ((2, 6), np.float64), "river_nut": ((2,), bool),
                  "river_category": ((2,), np.int8)}
    out = {k: np.empty((n,) + shape, dtype) for k, (shape, dtype) in names.items()}
    done = used = 0
    for f, m in shards:
        take = min(m["count"], n - done)
        if take <= 0:
            break
        with np.load(f) as z:
            for k, (shape, dtype) in names.items():
                a = z[k]
                if a.shape != (m["count"],) + shape or a.dtype != dtype:
                    raise ValueError(f"{f}: {k} is {a.dtype} {a.shape}, expected {np.dtype(dtype)} "
                                     f"{(m['count'],) + shape}")
                out[k][done:done + take] = a[:take]
        done += take
        used += 1
    meta = {k: first[k] for k in _IDENTITY} | {
        "n": n, "shards": used, "path": str(path),
        "numpy_versions": sorted({m["numpy_version"] for _, m in shards})}
    return Deals(out["buckets"], out["showdown"], meta, out.get("cards"),
                 {k: out[k] for k in ("features", "river_nut", "river_category")} if features else None)


# ---- re-bucketing

def _spot_check(tables: ab.BucketTables, arrays: dict, buckets: np.ndarray, rows: int, f: Path) -> None:
    """Recompute `rows` evenly spaced deals from their cards: features and buckets must match."""
    count = len(buckets)
    for i in np.unique(np.linspace(0, count - 1, min(rows, count)).astype(np.int64)):
        feats, nut, cat, b = _deal_features(tables, arrays["cards"][i].tolist())
        if not (np.array_equal(feats, arrays["features"][i]) and np.array_equal(nut, arrays["river_nut"][i])
                and np.array_equal(cat, arrays["river_category"][i]) and np.array_equal(b, buckets[i])):
            raise RuntimeError(f"{f}: row {i}: stored features or new buckets differ from bot.abstraction "
                               "on the cards (feature code changed? regenerate the dataset)")


def rebucket(path, buckets_table=ab.DEFAULT_PATH, verify: int = 4) -> int:
    """Recompute every shard's buckets from its stored features against buckets_table and
    rewrite it (atomically) with the new hash and counts. Returns the shards rewritten.

    Shards already on this table are skipped, so an interrupted run resumes. Every shard's
    features must come from this table's designs (features_sha256), else ValueError and
    nothing is written. `verify` rows per shard are recomputed from the cards as a check.
    """
    t = open_table(buckets_table)
    shards = _scan(path)
    if not shards:
        raise ValueError(f"no deal shards (deals_<k>.npz) at {path}")
    for f, m in shards:
        if m["features_sha256"] != t.features_sha256:
            raise ValueError(f"{f}: its features come from other designs or another data/preflop.npz "
                             f"than {buckets_table}. Re-bucketing cannot fix that: regenerate the dataset")
    t0, done = time.time(), 0
    for f, m in shards:
        if m["buckets_sha256"] == t.buckets_sha256:
            continue
        with np.load(f) as z:
            arrays = {k: z[k] for k in z.files}
        new = buckets_from_features(t.tables, arrays["features"], arrays["river_nut"],
                                    arrays["river_category"])
        _spot_check(t.tables, arrays, new, verify, f)
        arrays.update(buckets=new, **_table_meta(t))
        _write(f, arrays)
        done += 1
    deals = sum(m["count"] for _, m in shards)
    _say(f"rebucket: {done} of {len(shards)} shards ({deals} deals) rewritten for {buckets_table} "
         f"in {time.time() - t0:.1f} s")
    return done


# ---- generation (spawn workers: everything they need arrives through _init and the job)

_TABLE: Table | None = None
_INIT_ERROR: Exception | None = None


def _init(buckets_table: str) -> None:
    # An exception escaping a Pool initializer makes the Pool respawn workers forever, so
    # it is kept and raised by the first job instead.
    global _TABLE, _INIT_ERROR
    _TABLE = _INIT_ERROR = None
    try:
        from bot import fasteval, preflop
        fasteval.warmup()
        preflop.load_preflop()
        ev.get_table()
        _TABLE = open_table(buckets_table)
    except Exception as e:
        _INIT_ERROR = e


def _shard_job(job) -> tuple[int, int]:
    seed, shard, count, shard_size, out, buckets_sha256 = job
    if _TABLE is None:
        raise RuntimeError(f"worker set-up failed: {_INIT_ERROR!r}")
    if _TABLE.buckets_sha256 != buckets_sha256:
        raise RuntimeError("the bucket table changed while the dataset was being generated")
    _write(shard_path(out, shard), make_shard(_TABLE, seed, shard, count, shard_size))
    return shard, count


def _results(jobs: list, workers: int, buckets_table: str):
    if not jobs:
        return
    if workers <= 1:
        _init(buckets_table)
        if _INIT_ERROR is not None:
            raise _INIT_ERROR
        yield from map(_shard_job, jobs)
        return
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(buckets_table,)) as pool:
        yield from pool.imap_unordered(_shard_job, jobs)


def default_workers() -> int:
    """The CPUs this process may run on (taskset is respected on Linux)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def generate(n: int, out, seed: int = 0, shard_size: int = SHARD_SIZE, workers: int = 1,
             buckets_table=ab.DEFAULT_PATH) -> int:
    """Make sure `out` holds the first n deals of the stream for seed; returns the deals made.

    Existing shards must agree with this run (seed, shard size, table), else ValueError:
    a table refit is a rebucket() first. Leftover temp files of an interrupted run go.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t = open_table(buckets_table)
    for tmp in out.glob("deals_*.tmp"):
        tmp.unlink()
    existing = _scan(out)                     # _meta has refused other formats
    run = {"seed": seed, "shard_size": shard_size, "features_sha256": t.features_sha256,
           "buckets_sha256": t.buckets_sha256}
    hints = {"seed": "use another --out", "shard_size": "use another --out",
             "features_sha256": "its features are stale: regenerate into another --out",
             "buckets_sha256": f"rebucket it first: python -m train.gen_deals --rebucket --out {out} "
                               f"--buckets {buckets_table}"}
    for f, m in existing:
        for k, hint in hints.items():
            if m[k] != run[k]:
                raise ValueError(f"{f}: {_IDENTITY[k]} {m[k]!r}, but this run has {run[k]!r}: {hint}")
    have = {m["shard"]: m["count"] for _, m in existing}
    jobs = []
    for k in range(-(-n // shard_size)):
        count = min(shard_size, n - k * shard_size)
        if have.get(k, 0) < count:
            jobs.append((seed, k, count, shard_size, str(out), t.buckets_sha256))
    total = sum(j[2] for j in jobs)
    workers = max(1, min(workers, len(jobs)))         # no idle workers
    _say(f"gen_deals: {n} deals in {out}, seed {seed}, shard size {shard_size}: "
         f"{len(jobs)} shards ({total} deals) to make on {workers} workers")
    t0, done = time.time(), 0
    for k, count in _results(jobs, workers, str(buckets_table)):
        done += count
        el = time.time() - t0
        _say(f"  shard {k}: {done}/{total} deals, {done / el:.0f} deals/s, "
             f"eta {(total - done) / max(done / el, 1e-9):.0f} s")
    wall = time.time() - t0
    if total:
        _say(f"generated {total} deals in {wall:.1f} s ({total / wall:.0f} deals/s on {workers} workers)")
    return total


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=2_000_000, help="deals the dataset must hold")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="worker processes (default: the CPUs this process may run on)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE,
                    help="deals per shard file (part of the dataset's identity)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="dataset directory")
    ap.add_argument("--buckets", default=str(ab.DEFAULT_PATH), help="bucket table (data/buckets.npz)")
    ap.add_argument("--iters", type=int, default=None,
                    help="ignored: the bucket features are deterministic (Task 16); "
                         "kept for the plan's command")
    ap.add_argument("--rebucket", action="store_true",
                    help="recompute the buckets of the shards in --out against --buckets, then exit")
    ap.add_argument("--verify", type=int, default=4,
                    help="with --rebucket: deals per shard recomputed from the cards as a check")
    a = ap.parse_args(argv)
    if a.rebucket:
        rebucket(a.out, a.buckets, a.verify)
        return
    if a.n < 1 or a.shard_size < 1 or a.workers < 1:
        ap.error("need --n, --shard-size and --workers >= 1")
    if a.iters is not None:
        _say(f"--iters {a.iters} ignored: bot.abstraction.bucket() uses deterministic features")
    generate(a.n, a.out, a.seed, a.shard_size, a.workers, a.buckets)


if __name__ == "__main__":
    main()
