"""External-sampling MCCFR for the CFR blueprint (Task 18).

    python -m train.mccfr --deals train_data/d10m --iterations 24_000_000_000 --max-hours 6 \\
        --out train_data/mccfr/blueprint_raw.npz --seed 0 --threads 16 --count-visits

The game: chance picks one row of the Task 17 deal dataset uniformly, then both players bet
on the exact betting tree (bot/tree.py), each knowing only its own bucket on the current
street. Info set = (decision node, buckets[row, actor seat, node street]): imperfect recall,
since a player forgets its earlier streets' buckets.

Seats. The dataset's seat 0 is the button (showdown +1 = the button wins) and the tree's
arrays are relative to the button, so player 0 = the button everywhere. terminal_value()
reads a terminal as the engine pays it (tests/test_mccfr.py checks it against game.Game).

Layout (R26; the contract with Task 19). regret is float32 [n_info, 4], avg is float64
[n_info, 4], columns in game.Action order.
- Rows are node-major in decision-node id order: node n owns rows offset[n] .. offset[n + 1]
  - 1, one per bucket of its street, so the infoset of (n, b) is row offset[n] + b.
- Node ids are DFS pre-order, so the streets interleave: a preflop node's flop, turn and
  river descendants come before the next preflop node (3,182 street changes along the ids).
  A street's array [nodes of the street, B_street, 4] is therefore a gather, never a slice:
  array[street_rows(tree, offset, street)].reshape(-1, B_street, 4).
- The per-street counts B are the dataset's num_buckets (the table's,
  bot.abstraction.NUM_BUCKETS for data/buckets.npz), so no row is unused.
- Illegal actions are never written: their regret and avg stay exactly 0.
- avg is float64 because an increment is about 2/k of its row's sum after k updates with
  linear weights. Around k = 1e7, which the hottest preflop rows reach in a run of billions
  of iterations, that is about one float32 ulp: increments get rounded to whole ulps or
  dropped. The result is a systematic bias of up to several percentage points in the
  average strategy (the Task 18 review simulated 2.2 pp at 1.5e7 updates and 7.9 pp at 6e7).
  Regret increments stay about 1/sqrt(k) of their row, so float32 is enough there.

Algorithm (Lanctot et al. 2009, external sampling). Iteration t (0-based) samples a row
and the traverser t % 2 (0 = button) from its own rng stream, then walks the tree once:
- at a traverser node: strategy sigma by regret matching (positive regrets, uniform over the
  legal actions if none is positive); every legal action is explored; then
  regret[I, a] += w * (v(a) - sum_a sigma(a) v(a));
- at an opponent node: avg[I, a] += w * sigma(a) (the opponent's reach is what sampled
  this node, so this is the reach-weighted average), then one action is sampled from sigma.
- terminals pay the traverser's chips (terminal_value).
w = t + 1 with linear=True (Linear CFR, Brown & Sandholm 2019: both the regret and the
average increments weigh iteration t by t + 1), else 1. The blueprint is the average
strategy avg normalized over the legal actions (average_strategy()).

Threads (R23). One parallel numba kernel runs blocks of iterations on `threads` worker
threads (prange). Thread k takes the iteration pairs (t0 + 2p, t0 + 2p + 1) for p = k,
k + threads, ..., so every thread alternates the traverser (_run_block says why that matters).
They share regret and avg without locks (hogwild, as parallel MCCFR usually does):
- a lost update (two threads read-modify-write the same float at once) drops one sampled
  increment. Each iteration touches a few hundred of the millions of rows, so collisions
  are rare, and a dropped increment is a missing sample, not a biased one;
- a thread may read a row while another writes it. Each float load is atomic on x86-64 and
  ARM64, each row is read once and normalized from that copy, so sigma is always a valid
  distribution, at worst one update stale: the bounded staleness asynchronous CFR tolerates.
With threads=1 a run is deterministic: the rng is keyed by (seed, iteration) and the linear
weight is the iteration number. That is what makes resume exact (below).

Checkpoints and resume (R23). The state is (regret, avg, iterations done), plus the
diagnostic history and optional visit counts. It is written atomically (format FORMAT = 2:
float64 avg; a format-1 file, float32 avg from before that change, is refused) every 10% of
the iterations (<out stem>.ckpt-<iterations>.npz), on SIGINT/SIGTERM, and when --max-hours of
training time runs out; the final state goes to `out`. A rerun with the same arguments
resumes from the latest of those files (refusing one from another run: tree, table, dataset,
seed, linear). It continues the iteration count, so the rng streams and the linear weights
continue exactly: with one thread an interrupted-and-resumed run is bit-identical to an
uninterrupted one.

Metadata (T19). Every checkpoint and the final output carry n_dec, n_term, the sha256 of the
tree's child.tobytes(), the sha256 of the bucket table the dataset was bucketed with and of
its features' inputs, the per-street bucket counts, the game constants, the iterations,
seed, linear flag and the dataset's identity. load_checkpoint() refuses a mismatch with the
current tree, game constants or bucket table.

Evaluation. best_response() computes a best response to a strategy by backward induction
over the tree, for a weighted set of rows. On a dataset with perfect recall (each player's
bucket the same on every street) and every row, it is the exact best response of the
abstract game, and evaluate()'s exploitability is exact (tests/test_mccfr.py). On the real
dataset it is approximate, and labelled so: the greedy backward induction is not an exact
best response under imperfect recall, and the rows are a fixed sample (EVAL_ROWS, the same
at every checkpoint). It is the exact value of a valid pure strategy in the game of those
rows, but with few rows per infoset that strategy mostly exploits the sample's own noise.
Measured on the 2M dataset between 30M and 300M iterations: 3.3-3.5 bb/hand on 8,192 rows,
0.84-0.90 on 65,536, 0.29 on 262,144 (76 s), flat throughout. So on real data it is a
sanity monitor (it and the self-play value stay finite and steady), not a convergence
measure. Fitting on half the sample and valuing the other half gives a negative "bound"
(-1.30 bb). train/README.md has the runbook.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import signal
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numba
import numpy as np
from numba import njit, prange

import game
from bot import abstraction as ab
from bot.tree import BettingTree, get_tree
from game import Street
from train import gen_deals as gd

FORMAT = 2                        # 2: avg float64 (1: float32, refused)
CHECKPOINTS = 10                  # one every 10% of the iterations
EVAL_ROWS = 8192                  # rows the checkpoint diagnostic samples from a large dataset
EVAL_STREAM = 18                  # rng (seed, EVAL_STREAM) picks those rows
DIAG_COLUMNS = ("iterations", "value", "br_button", "br_bb", "exploitability", "exact")
BLOCK_SECONDS = 0.5               # target length of one kernel call (signals wait for it)
_CKPT = re.compile(r"\.ckpt-(\d+)\.npz")

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_SALT = np.uint64(0x6D63636672313800)      # seed -> key
_UNIT = 1.0 / 9007199254740992.0           # 2**-53


def _say(msg: str) -> None:
    print(msg, flush=True)


# ---- layout and identity

def infoset_offsets(tree: BettingTree, num_buckets) -> np.ndarray:
    """int64 [n_dec + 1]: node n owns rows offset[n] .. offset[n + 1] - 1 (one per bucket)."""
    nb = np.asarray(tuple(num_buckets), dtype=np.int64)
    if nb.shape != (4,) or nb.min() < 1:
        raise ValueError(f"bad bucket counts {tuple(num_buckets)}")
    return np.concatenate([[0], np.cumsum(nb[tree.street.astype(np.int64)])]).astype(np.int64)


def street_rows(tree: BettingTree, offset: np.ndarray, street) -> np.ndarray:
    """int64 rows of one street's infosets, by node (ascending id) then bucket, so that
    array[street_rows(tree, offset, s)].reshape(-1, B_s, 4) is street s's per-street array.
    A gather, because the streets' rows interleave (module docstring, Layout)."""
    nodes = np.flatnonzero(tree.street == int(street))
    if not len(nodes):
        return np.zeros(0, dtype=np.int64)
    width = np.diff(offset)[nodes]
    if (width != width[0]).any():
        raise ValueError("offset gives one street's nodes different bucket counts")
    return (offset[nodes][:, None] + np.arange(width[0])).reshape(-1).astype(np.int64)


def tree_sha256(tree: BettingTree) -> str:
    return hashlib.sha256(np.ascontiguousarray(tree.child).tobytes()).hexdigest()


def game_constants() -> dict:
    return {"starting_stack": game.STARTING_STACK, "small_blind": game.SMALL_BLIND,
            "big_blind": game.BIG_BLIND, "raise_cap": game.RAISE_CAP_PER_STREET}


def _max_depth(tree: BettingTree) -> int:
    """Decision nodes on the longest root-to-terminal path (ids are pre-order: parents first)."""
    depth = np.ones(tree.n_dec, dtype=np.int64)
    for n in range(tree.n_dec):
        for c in tree.child[n]:
            if c >= 0:
                depth[c] = depth[n] + 1
    return int(depth.max())


# ---- the kernels (numba)

@njit(cache=True)
def _mix(z):
    """splitmix64's finalizer: a uint64 -> uint64 bijection that scrambles every bit."""
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def terminal_value(tid, player, showdown, term_fold_by_button, term_contrib):
    """Chips won by player (0 = button, 1 = big blind) at terminal tid: got back - put in."""
    fold = term_fold_by_button[tid]
    button_in = float(term_contrib[tid, 0])
    bb_in = float(term_contrib[tid, 1])
    if fold == 1:
        u = -button_in                      # the button folded
    elif fold == 0:
        u = bb_in                           # the big blind folded
    elif showdown > 0:
        u = bb_in
    elif showdown < 0:
        u = -button_in
    else:
        u = 0.0                             # split: showdown contributions are equal
    return u if player == 0 else -u


@njit(cache=True)
def _traverse(i, r, w, state, regret, avg, visits, offset, child, street, actor_is_button,
              term_fold, term_contrib, buckets, showdown, f_node, f_idx, f_act, f_sig, f_val):
    """One external-sampling walk for traverser i on row r. Iterative: a frame per traverser
    node on the path (f_*), opponent nodes pass straight through. Returns (value, rng state)."""
    sd = showdown[r]
    depth = 0
    code = 0
    while True:
        while code >= 0:                                    # down to a terminal
            n = code
            actor = 0 if actor_is_button[n] else 1
            idx = offset[n] + buckets[r, actor, street[n]]
            total = 0.0
            nlegal = 0
            for a in range(4):                              # regret matching, one read per float
                p = 0.0
                if child[n, a] != -1:
                    nlegal += 1
                    g = regret[idx, a]
                    if g > 0.0:
                        p = g
                f_sig[depth, a] = p
                total += p
            for a in range(4):
                if total > 0.0:
                    f_sig[depth, a] /= total
                elif child[n, a] != -1:
                    f_sig[depth, a] = 1.0 / nlegal
            if actor == i:
                if visits.size > 0:
                    visits[idx] += 1
                f_node[depth] = n
                f_idx[depth] = idx
                a = 0
                while child[n, a] == -1:
                    a += 1
                f_act[depth] = a
                depth += 1
                code = child[n, a]
            else:
                state += _GOLDEN
                u = np.float64(_mix(state) >> np.uint64(11)) * _UNIT
                pick = -1
                last = 0
                cum = 0.0
                for a in range(4):
                    if child[n, a] != -1:
                        s = f_sig[depth, a]
                        avg[idx, a] += w * s
                        cum += s
                        last = a
                        if pick < 0 and u < cum:
                            pick = a
                if pick < 0:                                # u beyond a rounded-down total
                    pick = last
                code = child[n, pick]
        v = terminal_value(-2 - code, i, sd, term_fold, term_contrib)
        descend = False
        while depth > 0:                                    # back up to the next unexplored action
            f = depth - 1
            n = f_node[f]
            a = f_act[f]
            f_val[f, a] = v
            a += 1
            while a < 4 and child[n, a] == -1:
                a += 1
            if a < 4:
                f_act[f] = a
                code = child[n, a]
                descend = True
                break
            v = 0.0
            for a in range(4):
                if child[n, a] != -1:
                    v += f_sig[f, a] * f_val[f, a]
            idx = f_idx[f]
            for a in range(4):
                if child[n, a] != -1:
                    regret[idx, a] += w * (f_val[f, a] - v)
            depth -= 1
        if not descend:
            return v, state


@njit(cache=True, parallel=True)
def _run_block(t0, count, threads, key, linear, regret, avg, visits, offset, child, street,
               actor_is_button, term_fold, term_contrib, buckets, showdown, depth):
    """Iterations t0 .. t0 + count - 1, handed out in pairs (t0 + 2p, t0 + 2p + 1): thread k
    takes the pairs p = k, k + threads, ..., so every thread alternates the traverser.

    - The pairs matter. A thread that only ever traverses for one player (k = t mod threads
      with an even thread count) lets that player run ahead whenever its walks are cheaper,
      and long runs of one player's updates against a frozen opponent oscillate instead of
      converging. Measured on the perfect-recall test game: one thread reaches exploitability
      0.055 bb in 1M iterations, the same iterations in two one-player runs of 2048 give 1.02.
    - The walk's frame buffers are allocated here, one slice per thread, padded so that no
      two threads share a cache line. An np.empty inside the prange body would be hoisted
      out of the loop by numba (allocation hoisting) and shared by every thread."""
    rows = np.uint64(buckets.shape[0])
    pad = depth + 9                             # >= 64 bytes between threads' frames
    node_buf = np.empty((threads, pad), np.int64)
    idx_buf = np.empty((threads, pad), np.int64)
    act_buf = np.empty((threads, pad), np.int64)
    sig_buf = np.empty((threads, pad, 4))
    val_buf = np.empty((threads, pad, 4))
    for k in prange(threads):
        for p in range(k, (count + 1) // 2, threads):
            for j in range(2 * p, min(2 * p + 2, count)):
                t = t0 + j
                state = _mix(key + np.uint64(t) * _GOLDEN)
                state += _GOLDEN
                r = np.int64(_mix(state) % rows)
                w = np.float64(t + 1) if linear else 1.0
                _traverse(t & 1, r, w, state, regret, avg, visits, offset, child, street,
                          actor_is_button, term_fold, term_contrib, buckets, showdown,
                          node_buf[k], idx_buf[k], act_buf[k], sig_buf[k], val_buf[k])


@njit(cache=True)
def _values(responder, strategy, offset, child, street, actor_is_button, term_fold, term_contrib,
            rb, sd, w_fit, w_val, depth, b_max, policy):
    """sum_r w_val[r] * value of row r when the responder (0 button, 1 big blind) best-responds
    to `strategy` and the other player plays it; responder -1: both play it, button's value.

    Depth-first over the tree with a stack of per-row vectors: reach[L] (chance weight w_fit
    times the opponent's strategy above the level-L node) and vals[L, a] (child a's values).
    At a responder node the action of bucket b maximizes sum over rows with bucket b of
    reach * value (first maximum on ties) and is written to policy (if non-empty)."""
    R = rb.shape[0]
    reach = np.empty((depth + 1, R))
    vals = np.empty((depth + 1, 4, R))
    v = np.empty(R)
    q = np.empty((b_max, 4))
    best = np.empty(b_max, np.int64)
    node_at = np.empty(depth + 1, np.int64)
    act = np.empty(depth + 1, np.int64)
    player = 0 if responder < 0 else responder
    for r in range(R):
        reach[0, r] = w_fit[r]
    L = 0
    node_at[0] = 0
    act[0] = -1
    while True:
        n = node_at[L]
        actor = 0 if actor_is_button[n] else 1
        st = street[n]
        base = offset[n]
        a = act[L] + 1
        while a < 4 and child[n, a] == -1:
            a += 1
        if a < 4:                                           # the next child of n
            act[L] = a
            c = child[n, a]
            if c <= -2:
                for r in range(R):
                    vals[L, a, r] = terminal_value(-2 - c, player, sd[r], term_fold, term_contrib)
            else:
                if actor == responder or responder < 0:
                    for r in range(R):
                        reach[L + 1, r] = reach[L, r]
                else:
                    for r in range(R):
                        reach[L + 1, r] = reach[L, r] * strategy[base + rb[r, actor, st], a]
                L += 1
                node_at[L] = c
                act[L] = -1
            continue
        if actor == responder:                              # every child done: n's values
            nb = offset[n + 1] - base
            for b in range(nb):
                for a in range(4):
                    q[b, a] = 0.0
            for r in range(R):
                b = rb[r, actor, st]
                x = reach[L, r]
                for a in range(4):
                    if child[n, a] != -1:
                        q[b, a] += x * vals[L, a, r]
            for b in range(nb):
                ba = -1
                for a in range(4):
                    if child[n, a] != -1 and (ba < 0 or q[b, a] > q[b, ba]):
                        ba = a
                best[b] = ba
                if policy.size > 0:
                    policy[base + b] = ba
            for r in range(R):
                v[r] = vals[L, best[rb[r, actor, st]], r]
        else:
            for r in range(R):
                s = base + rb[r, actor, st]
                x = 0.0
                for a in range(4):
                    if child[n, a] != -1:
                        x += strategy[s, a] * vals[L, a, r]
                v[r] = x
        if L == 0:
            break
        L -= 1
        pa = act[L]
        for r in range(R):
            vals[L, pa, r] = v[r]
    total = 0.0
    for r in range(R):
        total += w_val[r] * v[r]
    return total


# ---- strategies and evaluation

def average_strategy(avg: np.ndarray, tree: BettingTree, offset: np.ndarray) -> np.ndarray:
    """float64 [n_info, 4]: avg normalized over the legal actions (uniform where avg is 0)."""
    legal = np.repeat(tree.child != -1, np.diff(offset), axis=0)
    s = np.where(legal, avg.astype(np.float64), 0.0)
    total = s.sum(axis=1, keepdims=True)
    uniform = legal / legal.sum(axis=1, keepdims=True)
    return np.where(total > 0, s / np.where(total > 0, total, 1.0), uniform)


class Evaluation(NamedTuple):
    value: float            # the button's bb per hand when both players play the strategy
    br_button: float        # bb per hand a best-responding button wins against it
    br_bb: float            # bb per hand a best-responding big blind wins against it
    exploitability: float   # (br_button + br_bb) / 2: what a best response wins per hand
    exact: bool             # perfect-recall rows, fitted and valued alike: the exact best
                            # response in the game these rows define


def _prepare(tree, strategy, offset, buckets, showdown, w_fit, w_val):
    rb = np.ascontiguousarray(buckets, dtype=np.int16)
    sd = np.ascontiguousarray(showdown, dtype=np.int8)
    if rb.ndim != 3 or rb.shape[1:] != (2, 4) or sd.shape != (rb.shape[0],):
        raise ValueError("buckets must be [R, 2, 4] and showdown [R]")
    nb = np.diff(offset)
    for s in Street:
        rows = nb[tree.street == s]
        if len(rows) and rb[:, :, s].size and not (0 <= rb[:, :, s].min() and rb[:, :, s].max() < rows[0]):
            raise ValueError(f"{s.name} buckets outside 0..{rows[0] - 1}")
    ones = np.ones(len(sd))
    w_fit = ones if w_fit is None else np.ascontiguousarray(w_fit, dtype=np.float64)
    w_val = w_fit if w_val is None else np.ascontiguousarray(w_val, dtype=np.float64)
    strategy = np.ascontiguousarray(strategy, dtype=np.float64)
    if strategy.shape != (offset[-1], 4):
        raise ValueError(f"strategy is {strategy.shape}, the layout has {offset[-1]} infosets")
    tree_arrays = (tree.child, tree.street, tree.actor_is_button, tree.term_fold_by_button,
                   tree.term_contrib)
    return strategy, rb, sd, w_fit, w_val, tree_arrays, _max_depth_cached(tree), int(nb.max())


_DEPTHS: dict = {}


def _max_depth_cached(tree: BettingTree) -> int:
    key = tree_sha256(tree)
    if key not in _DEPTHS:
        _DEPTHS[key] = _max_depth(tree)
    return _DEPTHS[key]


def best_response(tree, strategy, offset, buckets, showdown, responder: int, w_fit=None, w_val=None):
    """(bb per hand, policy): the responder's (0 button, 1 big blind) best response to
    `strategy` on rows (buckets [R, 2, 4], showdown [R]) chosen with chance weights w_fit and
    valued with w_val (both default to 1 per row). policy int8 [n_info] is its action at its
    own infosets, -1 at the other player's. Exact for perfect-recall rows (module docstring)."""
    strategy, rb, sd, w_fit, w_val, ta, depth, b_max = _prepare(tree, strategy, offset, buckets,
                                                                 showdown, w_fit, w_val)
    policy = np.full(offset[-1], -1, dtype=np.int8)
    total = _values(responder, strategy, offset, *ta, rb, sd, w_fit, w_val, depth, b_max, policy)
    return total / w_val.sum() / game.BIG_BLIND, policy


def profile_value(tree, strategy, offset, buckets, showdown, weights=None) -> float:
    """The button's bb per hand when both players play `strategy` (rows weighted by weights)."""
    strategy, rb, sd, w, _, ta, depth, b_max = _prepare(tree, strategy, offset, buckets, showdown,
                                                        weights, None)
    total = _values(-1, strategy, offset, *ta, rb, sd, w, w, depth, b_max, np.empty(0, np.int8))
    return total / w.sum() / game.BIG_BLIND


def has_perfect_recall(buckets: np.ndarray) -> bool:
    """Every player's bucket is the same on every street: (node, bucket) then recalls everything."""
    b = np.asarray(buckets)
    return bool((b == b[:, :, :1]).all())


def evaluate(tree, strategy, offset, buckets, showdown, w_fit=None, w_val=None) -> Evaluation:
    """Self-play value, both best responses and exploitability, in bb per hand."""
    br_button, _ = best_response(tree, strategy, offset, buckets, showdown, 0, w_fit, w_val)
    br_bb, _ = best_response(tree, strategy, offset, buckets, showdown, 1, w_fit, w_val)
    split = w_val is not None and (w_fit is None or not np.array_equal(w_fit, w_val))
    value = profile_value(tree, strategy, offset, buckets, showdown, w_val if split else w_fit)
    exact = has_perfect_recall(buckets) and not split
    return Evaluation(value, br_button, br_bb, (br_button + br_bb) / 2, exact)


def _diagnostic_rows(buckets, showdown, eval_rows: int, seed: int):
    """(buckets, showdown, w_fit, w_val, exact) for the checkpoint diagnostic. A dataset of at
    most eval_rows rows is used whole, its identical rows merged (exact if perfect recall);
    a larger one gives a fixed sample (approximate: module docstring)."""
    n = len(showdown)
    if eval_rows >= n:
        keys = np.concatenate([buckets.reshape(n, 8).astype(np.int64), showdown[:, None]], axis=1)
        u, counts = np.unique(keys, axis=0, return_counts=True)
        w = counts.astype(np.float64)
        return u[:, :8].reshape(-1, 2, 4), u[:, 8], w, w, has_perfect_recall(buckets)
    pick = np.sort(np.random.default_rng([seed, EVAL_STREAM]).choice(n, size=eval_rows, replace=False))
    w = np.ones(eval_rows)
    return buckets[pick], showdown[pick], w, w, False


# ---- checkpoints

class Checkpoint(NamedTuple):
    regret: np.ndarray      # float32 [n_info, 4]
    avg: np.ndarray         # float64 [n_info, 4]
    offset: np.ndarray      # int64 [n_dec + 1]
    visits: np.ndarray      # int64 [n_info] traverser visits (empty unless counted)
    diag: np.ndarray        # float64 [k, len(DIAG_COLUMNS)]: the checkpoint diagnostics so far
    meta: dict


_META_INT = ("iterations", "target", "seed", "n_dec", "n_term", "starting_stack", "small_blind",
             "big_blind", "raise_cap", "dataset_n", "dataset_seed", "dataset_shard_size",
             "dataset_format")
_META_STR = ("tree_child_sha256", "buckets_sha256", "features_sha256", "dataset_path",
             "numpy_version", "numba_version")
_META_BOOL = ("linear", "count_visits")
_META_TUPLE = ("num_buckets", "flop_grid", "turn_grid")
_RUN_IDENTITY = ("seed", "linear", "count_visits", "dataset_n", "dataset_seed", "dataset_shard_size",
                 "dataset_format", "buckets_sha256", "features_sha256", "num_buckets")


def checkpoint_path(out, iterations: int) -> Path:
    out = Path(out)
    return out.with_name(f"{out.stem}.ckpt-{iterations:012d}.npz")


def checkpoints(out) -> list[Path]:
    """The checkpoint files of `out`, oldest first."""
    out = Path(out)
    if not out.parent.is_dir():
        return []
    found = [(int(m.group(1)), f) for f in out.parent.iterdir()
             if f.name.startswith(out.stem) and (m := _CKPT.fullmatch(f.name[len(out.stem):]))]
    return [f for _, f in sorted(found)]


def save_checkpoint(path, ck: Checkpoint) -> None:
    """Atomic (a temp file, then a rename), uncompressed."""
    m = ck.meta
    arrays = {"format": np.int64(FORMAT), "regret": ck.regret, "avg": ck.avg, "offset": ck.offset,
              "visits": ck.visits, "diag": ck.diag, "train_seconds": np.float64(m["train_seconds"]),
              **{k: np.int64(m[k]) for k in _META_INT}, **{k: np.array(m[k]) for k in _META_STR},
              **{k: np.bool_(m[k]) for k in _META_BOOL},
              **{k: np.array(m[k], dtype=np.int64) for k in _META_TUPLE}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gd._write(path, arrays)


def _read(path) -> Checkpoint:
    with np.load(path) as z:
        try:
            fmt = int(z["format"])
            if fmt == 1:
                raise ValueError(f"{path}: checkpoint format 1 keeps avg in float32, which biases the "
                                 "average strategy of heavily visited infosets; this code reads format "
                                 f"{FORMAT} (float64 avg). Retrain: format-1 runs cannot be resumed")
            if fmt != FORMAT:
                raise ValueError(f"{path}: checkpoint format {fmt}, this code reads {FORMAT}")
            meta = {k: int(z[k]) for k in _META_INT} | {k: str(z[k]) for k in _META_STR} | {
                k: bool(z[k]) for k in _META_BOOL} | {k: tuple(int(x) for x in z[k]) for k in _META_TUPLE}
            meta["train_seconds"] = float(z["train_seconds"])
            return Checkpoint(z["regret"], z["avg"], z["offset"], z["visits"], z["diag"], meta)
        except KeyError as e:
            raise ValueError(f"{path}: not an MCCFR checkpoint, missing {e}") from None


def load_checkpoint(path, tree: BettingTree | None = None, buckets_table=ab.DEFAULT_PATH) -> Checkpoint:
    """A checkpoint or final output, verified against the current tree (n_dec, n_term, child
    hash), game constants and, unless buckets_table is None (synthetic data), the bucket table:
    its sha256, the sha256 of its features' inputs and its bucket counts. ValueError on a mismatch."""
    tree = get_tree() if tree is None else tree
    ck = _read(path)
    m = ck.meta
    for k, want in (("n_dec", tree.n_dec), ("n_term", tree.n_term),
                    ("tree_child_sha256", tree_sha256(tree)), *game_constants().items()):
        if m[k] != want:
            raise ValueError(f"{path}: trained on another betting tree or rules: {k} {m[k]!r}, "
                             f"the current tree has {want!r}")
    if buckets_table is not None:
        t = gd.open_table(buckets_table)
        have = tuple(t.tables.num_buckets[s] for s in Street)
        for k, want in (("buckets_sha256", t.buckets_sha256), ("features_sha256", t.features_sha256),
                        ("num_buckets", have)):
            if m[k] != want:
                raise ValueError(f"{path}: {k} {m[k]!r}, but {buckets_table} has {want!r}")
    offset = infoset_offsets(tree, m["num_buckets"])
    if not np.array_equal(ck.offset, offset):
        raise ValueError(f"{path}: its offset table does not match num_buckets {m['num_buckets']}")
    n_info = int(offset[-1])
    for name, dtype in (("regret", np.float32), ("avg", np.float64)):
        a = getattr(ck, name)
        if a.shape != (n_info, 4) or a.dtype != dtype:
            raise ValueError(f"{path}: {name} is {a.dtype} {a.shape}, expected {np.dtype(dtype)} {(n_info, 4)}")
    if ck.visits.shape != ((n_info,) if m["count_visits"] else (0,)):
        raise ValueError(f"{path}: visits has shape {ck.visits.shape}")
    return ck


def _iterations_of(path) -> int:
    with np.load(path) as z:
        try:
            return int(z["iterations"])
        except KeyError:
            raise ValueError(f"{path}: not an MCCFR checkpoint") from None


def _latest(out) -> Path | None:
    """The saved state with the most iterations among out's checkpoints and out itself."""
    best, best_it = None, -1
    for p in checkpoints(out) + ([Path(out)] if Path(out).is_file() else []):
        it = _iterations_of(p)
        if it > best_it:
            best, best_it = p, it
    return best


# ---- training

class _Stop:
    """SIGINT/SIGTERM set a flag; the loop checkpoints at the end of the current block."""

    def __init__(self):
        self.signum = None
        self.previous = {}

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for s in (signal.SIGINT, signal.SIGTERM):
                self.previous[s] = signal.signal(s, self._handle)
        return self

    def _handle(self, signum, frame):
        self.signum = signum

    def __exit__(self, *exc):
        for s, h in self.previous.items():
            signal.signal(s, h)


def _check_rows(deals, nb, path) -> None:
    for s in Street:
        b = deals.buckets[:, :, s]
        if b.min() < 0 or b.max() >= nb[s]:
            raise ValueError(f"{path}: {s.name} buckets span {b.min()}..{b.max()}, outside 0..{nb[s] - 1}")
    if not np.isin(deals.showdown, (-1, 0, 1)).all():
        raise ValueError(f"{path}: showdown values other than -1, 0, 1")


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def train(tree: BettingTree, deals_path, iterations: int, out: str, seed: int, linear: bool = True, *,
          threads: int | None = None, buckets_table=ab.DEFAULT_PATH, eval_rows: int = EVAL_ROWS,
          max_hours: float | None = None, count_visits: bool = False, log_every: float = 60.0,
          checkpoints_per_run: int = CHECKPOINTS) -> None:
    """Run external-sampling MCCFR until `iterations` are done, writing the final state to out.

    Resumes from the latest saved state of out. Checkpoints every 1/checkpoints_per_run of the
    iterations, each with the exploitability diagnostic on eval_rows rows (0: none). Stops
    early, with a checkpoint, on SIGINT (then raises KeyboardInterrupt) or SIGTERM (SystemExit
    143), and when the run's training time reaches max_hours (then also writes out).
    buckets_table None skips the table checks (synthetic data). Module docstring: the details.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if tree.starting_stack != game.STARTING_STACK:
        raise ValueError(f"the tree has stack {tree.starting_stack}, game.py {game.STARTING_STACK}")
    out = Path(out)
    threads = gd.default_workers() if threads is None else int(threads)
    if not 1 <= threads <= numba.config.NUMBA_NUM_THREADS:
        raise ValueError(f"threads must be in 1..{numba.config.NUMBA_NUM_THREADS}")
    deals = gd.load_deals(deals_path, buckets_table)       # checks the table and features hashes
    nb = tuple(deals.meta["num_buckets"])
    if (buckets_table is not None and Path(buckets_table).resolve() == ab.DEFAULT_PATH.resolve()
            and nb != tuple(ab.NUM_BUCKETS[s] for s in Street)):
        raise ValueError(f"{ab.DEFAULT_PATH} has {nb} buckets, bot.abstraction.NUM_BUCKETS says "
                         f"{dict(ab.NUM_BUCKETS)} (replaced after import?)")
    _check_rows(deals, nb, deals_path)
    offset = infoset_offsets(tree, nb)
    n_info = int(offset[-1])
    dm = deals.meta
    run_meta = {"seed": int(seed), "linear": bool(linear), "count_visits": bool(count_visits),
                "n_dec": tree.n_dec, "n_term": tree.n_term, "tree_child_sha256": tree_sha256(tree),
                **game_constants(), "buckets_sha256": dm["buckets_sha256"],
                "features_sha256": dm["features_sha256"], "num_buckets": nb,
                "flop_grid": tuple(dm["flop_grid"]), "turn_grid": tuple(dm["turn_grid"]),
                "dataset_path": str(dm["path"]), "dataset_n": dm["n"], "dataset_seed": dm["seed"],
                "dataset_shard_size": dm["shard_size"], "dataset_format": dm["format"],
                "numpy_version": np.__version__, "numba_version": numba.__version__}

    latest = _latest(out)
    if latest is None:
        ck = Checkpoint(np.zeros((n_info, 4), np.float32), np.zeros((n_info, 4), np.float64), offset,
                        np.zeros(n_info if count_visits else 0, np.int64),
                        np.zeros((0, len(DIAG_COLUMNS))), run_meta | {"iterations": 0, "train_seconds": 0.0})
    else:
        ck = load_checkpoint(latest, tree, buckets_table)
        for k in _RUN_IDENTITY:
            if ck.meta[k] != run_meta[k]:
                what = "dataset" if k.startswith("dataset") or k.endswith("sha256") or k == "num_buckets" else k
                raise ValueError(f"{latest} is from another run ({what}: {k} {ck.meta[k]!r}, this run "
                                 f"has {run_meta[k]!r}). Use another out, or delete its files to restart")
        _say(f"mccfr: resuming {latest} at {ck.meta['iterations']:,} iterations")
    regret, avg, visits = ck.regret, ck.avg, ck.visits
    done, diag, spent = ck.meta["iterations"], ck.diag, ck.meta["train_seconds"]
    ev_rows = _diagnostic_rows(deals.buckets, deals.showdown, eval_rows, seed) if eval_rows > 0 else None

    def save(path, final=False):
        nonlocal diag
        if ev_rows is not None and (not len(diag) or diag[-1, 0] != done):
            t1 = time.perf_counter()
            rb, sd, w_fit, w_val, exact = ev_rows
            e = evaluate(tree, average_strategy(avg, tree, offset), offset, rb, sd, w_fit, w_val)
            diag = np.vstack([diag, [[done, e.value, e.br_button, e.br_bb, e.exploitability, exact]]])
            _say(f"  diagnostic at {done:,}: value {e.value:+.4f} bb, best responses: button "
                 f"{e.br_button:+.4f} bb, big blind {e.br_bb:+.4f} bb, exploitability "
                 f"{e.exploitability:.4f} bb/hand "
                 f"({'exact' if exact else f'approximate: greedy response on {len(sd)} sampled rows'}, "
                 f"{time.perf_counter() - t1:.1f} s)")
        meta = run_meta | {"iterations": done, "target": iterations, "train_seconds": spent}
        t1 = time.perf_counter()
        save_checkpoint(path, Checkpoint(regret, avg, offset, visits, diag, meta))
        _say(f"  {'final output' if final else 'checkpoint'} {path} at {done:,} iterations "
             f"({time.perf_counter() - t1:.1f} s)")

    if done >= iterations:
        _say(f"mccfr: {latest} already has {done:,} >= {iterations:,} iterations")
        if latest != out:
            save(out, final=True)
        return
    step = -(-iterations // checkpoints_per_run)
    marks = sorted({min(iterations, k * step) for k in range(1, checkpoints_per_run + 1)})
    numba.set_num_threads(threads)
    key = np.uint64(_mix(np.uint64(seed & 0xFFFFFFFFFFFFFFFF) ^ _SALT))
    depth = _max_depth_cached(tree)
    args = (regret, avg, visits, offset, tree.child, tree.street, tree.actor_is_button,
            tree.term_fold_by_button, tree.term_contrib, deals.buckets, deals.showdown, depth)
    _say(f"mccfr: {iterations:,} iterations on {deals.meta['n']:,} deals of {deals_path}, "
         f"{n_info:,} infosets (buckets {nb}), {threads} threads, seed {seed}, "
         f"{'linear' if linear else 'uniform'} weights, from {done:,}")
    block, start = min(4096, iterations - done), done
    t_start = t_log = time.perf_counter()
    spent0, log_done = spent, done
    with _Stop() as stop:
        while done < iterations:
            nxt = next(m for m in marks if m > done)
            count = min(block, nxt - done)
            t1 = time.perf_counter()
            _run_block(np.int64(done), np.int64(count), np.int64(threads), key, bool(linear), *args)
            dt = time.perf_counter() - t1
            done += count
            spent = spent0 + time.perf_counter() - t_start
            block = int(min(max(count * BLOCK_SECONDS / max(dt, 1e-6), 1000), 2 * max(block, count),
                            50_000_000))
            now = time.perf_counter()
            if now - t_log >= log_every or done == iterations:
                rate = (done - log_done) / max(now - t_log, 1e-9)
                overall = (done - start) / max(now - t_start, 1e-9)
                _say(f"mccfr: {done:,}/{iterations:,} iterations ({100 * done / iterations:.1f}%), "
                     f"{rate:,.0f} it/s now, {overall:,.0f} it/s since start, elapsed "
                     f"{_fmt_time(spent)}, eta {_fmt_time((iterations - done) / max(overall, 1e-9))}")
                t_log, log_done = now, done
            out_of_time = max_hours is not None and spent >= max_hours * 3600
            if done == iterations or (out_of_time and stop.signum is None):
                save(out, final=True)
                if out_of_time and done < iterations:
                    _say(f"mccfr: --max-hours {max_hours} reached at {done:,} iterations")
                break
            if done == nxt or stop.signum is not None:
                save(checkpoint_path(out, done))
            if stop.signum is not None:
                _say(f"mccfr: stopped by signal {stop.signum} at {done:,} iterations; "
                     "rerun the same command to resume")
                break
    if done < iterations and stop.signum == signal.SIGINT:
        raise KeyboardInterrupt
    if done < iterations and stop.signum is not None:
        raise SystemExit(128 + stop.signum)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deals", default=str(gd.DEFAULT_OUT), help="deal dataset directory (Task 17)")
    ap.add_argument("--iterations", type=int, required=True, help="sampled iterations in all")
    ap.add_argument("--out", default=str(gd.DEFAULT_OUT / "mccfr" / "blueprint_raw.npz"),
                    help="final output; checkpoints go next to it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-linear", action="store_true", help="uniform instead of linear weights")
    ap.add_argument("--threads", type=int, default=gd.default_workers(),
                    help="worker threads (default: the CPUs this process may run on)")
    ap.add_argument("--buckets", default=str(ab.DEFAULT_PATH),
                    help="bucket table the dataset must match ('none': synthetic data, no checks)")
    ap.add_argument("--eval-rows", type=int, default=EVAL_ROWS,
                    help="rows of the exploitability diagnostic at each checkpoint (0: none)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop (checkpoint and final output) after this much training time")
    ap.add_argument("--log-every", type=float, default=60.0, help="seconds between progress lines")
    ap.add_argument("--count-visits", action="store_true", help="also count traverser visits per infoset")
    a = ap.parse_args(argv)
    try:
        train(get_tree(), a.deals, a.iterations, a.out, a.seed, not a.no_linear, threads=a.threads,
              buckets_table=None if a.buckets.lower() == "none" else a.buckets, eval_rows=a.eval_rows,
              max_hours=a.max_hours, count_visits=a.count_visits, log_every=a.log_every)
    except KeyboardInterrupt:
        raise SystemExit(128 + signal.SIGINT) from None


if __name__ == "__main__":
    main()
