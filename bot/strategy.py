"""Runtime CFR blueprint: loads data/blueprint.npz and samples actions from it (Task 19).

File format (written by train/export.py; FORMAT 1). The arrays are those of the trainer
(train/mccfr.py, Layout), not regrouped per street:
- probs: float16 [n_info, 4], the trainer's average strategy normalized over the legal
  actions, columns in game.Action order, illegal actions exactly 0. Rows are node-major in
  decision-node id order, so the infoset (node, bucket) is row offset[node] + bucket, and
  offset = concatenate([0], cumsum(buckets[tree.street])) is recomputed at load from the
  per-street bucket counts and the tree (the trainer's streets interleave along node ids,
  hence no per-street slicing). An infoset the trainer never updated holds the uniform
  distribution over the legal actions.
- scalars: format, stack, sb, bb, cap, iterations, seed, linear, n_dec, n_term,
  tree_child_sha256, buckets_sha256, features_sha256, buckets (the four per-street counts),
  dataset_* (the T17 dataset's identity).

Blueprint.load() verifies the file against the runtime: format, game constants, the tree
(n_dec, n_term, sha256 of child.tobytes()), the sha256 of data/buckets.npz and its bucket
counts, and the array shape. Any mismatch is a ValueError (a refit or a tree change
invalidates the blueprint). Blueprint.try_load() / load_default() never raise: they return
None, which the caller treats as "no blueprint" (heuristic only).

Everything here is read-only: the table is non-writeable, meta is a mappingproxy, and there
is no module-level mutable state. stdlib + numpy only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np

import game
from bot import abstraction as ab
from bot.tree import BettingTree, get_tree

FORMAT = 1
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "blueprint.npz"
_DEFAULT_BUCKETS = object()        # sentinel: verify against data/buckets.npz
_INT = ("format", "stack", "sb", "bb", "cap", "iterations", "seed", "n_dec", "n_term")
_STR = ("tree_child_sha256", "buckets_sha256", "features_sha256")


def tree_sha256(tree: BettingTree) -> str:
    return hashlib.sha256(np.ascontiguousarray(tree.child).tobytes()).hexdigest()


def _constants() -> dict:
    return {"stack": game.STARTING_STACK, "sb": game.SMALL_BLIND, "bb": game.BIG_BLIND,
            "cap": game.RAISE_CAP_PER_STREET}


class Blueprint:
    """A loaded blueprint. `meta` (read-only mapping) has stack, sb, bb, cap, buckets (counts
    per street), iterations, and the identity fields listed in the module docstring."""

    __slots__ = ("meta", "table", "offset", "_child")

    def __init__(self, meta, table: np.ndarray, offset: np.ndarray, child: np.ndarray):
        self.meta = MappingProxyType(dict(meta))
        self.table, self.offset, self._child = table, offset, child
        for a in (table, offset):
            a.flags.writeable = False

    # ---- loading

    @classmethod
    def load(cls, path, *, tree: BettingTree | None = None, buckets_path=_DEFAULT_BUCKETS) -> Blueprint:
        """Load and verify. ValueError on a missing, corrupt or mismatching file.
        buckets_path: the bucket table to check against (default data/buckets.npz;
        None skips the table check, for synthetic test blueprints)."""
        try:
            return cls._load(path, tree, buckets_path)
        except ValueError:
            raise
        except Exception as e:                      # OSError, BadZipFile, KeyError, EOFError, ...
            raise ValueError(f"{path}: cannot read blueprint ({type(e).__name__}: {e})") from None

    @classmethod
    def try_load(cls, path=DEFAULT_PATH, **kw) -> Blueprint | None:
        """load(), but None instead of any exception."""
        try:
            return cls.load(path, **kw)
        except Exception:
            return None

    @classmethod
    def _load(cls, path, tree, buckets_path) -> Blueprint:
        with np.load(path, allow_pickle=False) as z:
            meta = {k: int(z[k]) for k in _INT} | {k: str(z[k]) for k in _STR}
            meta["linear"] = bool(z["linear"])
            meta["buckets"] = tuple(int(x) for x in z["buckets"])
            for k in z.files:
                if k.startswith("dataset_"):
                    meta[k] = z[k].item()
            table = z["probs"]
        if meta["format"] != FORMAT:
            raise ValueError(f"{path}: blueprint format {meta['format']}, this code reads {FORMAT}")
        tree = get_tree() if tree is None else tree
        for k, want in (*_constants().items(), ("n_dec", tree.n_dec), ("n_term", tree.n_term),
                        ("tree_child_sha256", tree_sha256(tree))):
            if meta[k] != want:
                raise ValueError(f"{path}: built for another tree or rules: {k} {meta[k]!r}, "
                                 f"the runtime has {want!r}")
        if buckets_path is not None:
            p = ab.DEFAULT_PATH if buckets_path is _DEFAULT_BUCKETS else Path(buckets_path)
            have = tuple(ab.NUM_BUCKETS[s] for s in game.Street) if p == ab.DEFAULT_PATH else None
            sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
            if meta["buckets_sha256"] != sha:
                raise ValueError(f"{path}: built with another bucket table (sha256 "
                                 f"{meta['buckets_sha256'][:12]}, {p} is {sha[:12]})")
            if have is not None and meta["buckets"] != have:
                raise ValueError(f"{path}: bucket counts {meta['buckets']}, the runtime has {have}")
        counts = np.asarray(meta["buckets"], dtype=np.int64)
        if counts.shape != (4,) or counts.min() < 1:
            raise ValueError(f"{path}: bad bucket counts {meta['buckets']}")
        offset = np.concatenate([[0], np.cumsum(counts[tree.street.astype(np.int64)])]).astype(np.int64)
        if table.dtype != np.float16 or table.shape != (int(offset[-1]), 4):
            raise ValueError(f"{path}: probs is {table.dtype} {table.shape}, "
                             f"expected float16 {(int(offset[-1]), 4)}")
        return cls(meta, table, offset, tree.child)

    # ---- queries

    def probs(self, node: int, bucket: int) -> np.ndarray:
        """float64 [4] over game.Action: the blueprint's distribution at (decision node, the
        actor's bucket on that street), summing to 1, exactly 0 on illegal actions.
        IndexError on a node or bucket outside the tree."""
        node, bucket = int(node), int(bucket)
        lo = self.offset[node]
        if node < 0 or bucket < 0 or lo + bucket >= self.offset[node + 1]:
            raise IndexError(f"no infoset for node {node}, bucket {bucket}")
        p = self.table[lo + bucket].astype(np.float64)
        s = p.sum()
        if s <= 0.0:                       # cannot happen in an exported file; stay legal anyway
            p = (self._child[node] != -1).astype(np.float64)
            s = p.sum()
        return p / s

    def act(self, node: int, bucket: int, legal, rng, purify: float = 0.1) -> game.Action:
        """Sample a legal action. The distribution is masked to `legal` and renormalized;
        probabilities below `purify` (I9) are zeroed and it is renormalized again. If nothing
        is left, the most probable legal action. rng: anything with .random() (numpy
        Generator, random.Random). Always returns a member of `legal`."""
        legal = tuple(legal)
        if not legal:
            raise ValueError("no legal actions")
        p = self.probs(node, bucket)
        q = np.zeros(4)
        idx = [int(a) for a in legal]
        q[idx] = p[idx]
        best = max(legal, key=lambda a: q[int(a)])          # most probable legal (first on ties)
        total = q.sum()
        if total <= 0.0:
            return best
        q /= total
        if purify > 0.0:
            q[q < purify] = 0.0
            total = q.sum()
            if total <= 0.0:
                return best
            q /= total
        u = rng.random()
        acc = 0.0
        pick = None
        for a in legal:
            w = q[int(a)]
            if w > 0.0:
                acc += w
                pick = a
                if u < acc:
                    return a
        return pick if pick is not None else best        # u rounded past the last cumulative


def load_default(path=DEFAULT_PATH, **kw) -> Blueprint | None:
    """The shipped blueprint, or None if it is missing, corrupt or stale. Never raises."""
    return Blueprint.try_load(path, **kw)
