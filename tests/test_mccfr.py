"""MCCFR trainer (Task 18): terminal values against the engine, the exact best response, convergence
on a perfect-recall game, checkpoints, resume and the load checks."""

import os
import random
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import game
from bot import abstraction as ab
from bot.tree import get_tree, node_from_actions
from game import Action, Deal, Game, Street
from train import gen_deals as gd
from train import mccfr

REPO = Path(__file__).resolve().parent.parent
F, C, H, P = Action


@pytest.fixture(scope="module")
def tree():
    return get_tree()


def write_dataset(path, buckets, showdown, num_buckets, seed=0):
    """A synthetic deal dataset in train.gen_deals' shard format (one shard, no cards)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    n = len(showdown)
    arrays = {"format": np.int64(gd.FORMAT), "seed": np.int64(seed), "shard": np.int64(0),
              "shard_size": np.int64(n), "numpy_version": np.array(np.__version__),
              "buckets_sha256": np.array("synthetic"), "features_sha256": np.array("synthetic"),
              "num_buckets": np.array(num_buckets, np.int64),
              "flop_grid": np.array([1, num_buckets[1]], np.int64),
              "turn_grid": np.array([1, num_buckets[2]], np.int64),
              "buckets": np.asarray(buckets, np.int16), "showdown": np.asarray(showdown, np.int8)}
    gd._write(gd.shard_path(path, 0), arrays)
    return path


def class_game(path, n=3000, k=3, seed=0):
    """Perfect recall: each player holds one class for the whole hand (the same bucket on every
    street). The higher class wins the showdown 80% of the time, equal classes are a coin flip."""
    rng = np.random.default_rng(seed)
    cls = rng.integers(0, k, size=(n, 2))
    diff = cls[:, 0] - cls[:, 1]
    win = np.where(rng.random(n) < 0.8, 1, -1)
    showdown = np.where(diff > 0, win, np.where(diff < 0, -win, rng.integers(-1, 2, n)))
    return write_dataset(path, np.repeat(cls[:, :, None], 4, axis=2), showdown, (k, k, k, k), seed)


def run(tree, deals, iterations, out, seed=0, **kw):
    kw = {"threads": 1, "buckets_table": None, "log_every": 1e9} | kw
    mccfr.train(tree, deals, iterations, str(out), seed, **kw)
    return mccfr.load_checkpoint(out, tree, kw["buckets_table"])


def _rows(path):
    d = gd.load_deals(path, buckets_table=None)
    return d.buckets, d.showdown


# ---- the game the trainer solves

def _policy(rng):
    """Random legal actions with per-hand weights, so hands reach showdowns as well as folds."""
    w = {F: rng.choice((0.05, 0.3, 1.0)), C: 1.0, H: rng.choice((0.3, 1.0)), P: rng.choice((0.3, 1.0))}
    return lambda legal: rng.choices(legal, [w[a] for a in legal])[0]


def test_terminal_values_match_the_engine_for_both_seats(tree, tmp_path):
    """Dataset seat 0 is the button. Replay random hands on real dataset deals with the button in
    either engine seat: the trainer's value for each player equals the engine's net."""
    out = tmp_path / "deals"
    gd.main(["--n", "24", "--seed", "18", "--shard-size", "24", "--workers", "1", "--out", str(out)])
    d = gd.load_deals(out, cards=True)
    rng, kinds = random.Random(18), set()
    for row in range(24):
        c = d.cards[row].tolist()
        holes = (tuple(c[0:5]), tuple(c[5:10]))                  # the button's hole first
        for button_seat in (0, 1):
            deal = Deal(holes if button_seat == 0 else holes[::-1], tuple(c[10:15]))
            for _ in range(6):
                pick, g, code = _policy(rng), Game(deal, button_seat=button_seat), 0
                while not g.is_terminal:
                    a = pick(g.legal_actions())
                    code = int(tree.child[code, a])
                    g.apply(a)
                tid, res = -2 - code, g.result()
                sd = int(d.showdown[row])
                for player, seat in ((0, button_seat), (1, 1 - button_seat)):
                    got = mccfr.terminal_value(tid, player, sd, tree.term_fold_by_button,
                                               tree.term_contrib)
                    assert got == res.nets[seat], (row, button_seat, player)
                flag = int(tree.term_fold_by_button[tid])
                kinds.add(("fold", flag) if flag >= 0 else ("showdown", sd))
    assert {("fold", 0), ("fold", 1), ("showdown", 1), ("showdown", -1)} <= kinds


def test_offsets_give_each_node_its_streets_bucket_count(tree):
    nb = (7, 5, 4, 3)
    off = mccfr.infoset_offsets(tree, nb)
    assert off.shape == (tree.n_dec + 1,) and off[0] == 0
    assert np.array_equal(np.diff(off), np.array(nb)[tree.street])
    assert off[-1] == sum(nb[s] * int((tree.street == s).sum()) for s in Street)


# ---- the evaluator: exact best response on perfect recall

def _fold_to_any_bet(tree, off):
    """Fold whenever facing a bet, check otherwise."""
    s = np.zeros((off[-1], 4))
    for n in range(tree.n_dec):
        s[off[n]:off[n + 1], F if tree.child[n, F] != -1 else C] = 1.0
    return s


def test_best_response_known_answers(tree, tmp_path):
    """Against "fold to any bet" the button raises and takes the big blind (2 chips) and the big
    blind collects the small blind (1 chip): 1 bb and 0.5 bb, exploitability 0.75 bb."""
    buckets, showdown = np.zeros((3, 2, 4), np.int16), np.array([1, 0, -1], np.int8)
    off = mccfr.infoset_offsets(tree, (1, 1, 1, 1))
    e = mccfr.evaluate(tree, _fold_to_any_bet(tree, off), off, buckets, showdown)
    assert e.br_button == pytest.approx(1.0) and e.br_bb == pytest.approx(0.5)
    assert e.exploitability == pytest.approx(0.75) and e.value == pytest.approx(-0.5)
    assert e.exact


def test_best_response_beats_random_strategies_and_matches_its_own_policy(tree, tmp_path):
    buckets, showdown = _rows(class_game(tmp_path / "g", n=400, k=2, seed=1))
    off = mccfr.infoset_offsets(tree, (2, 2, 2, 2))
    rng = np.random.default_rng(5)
    legal = np.repeat(tree.child != -1, np.diff(off), axis=0)

    def random_strategy():
        s = rng.random((off[-1], 4)) ** 3 * legal
        return s / s.sum(axis=1, keepdims=True)

    sigma = random_strategy()
    for responder in (0, 1):
        sign = 1 if responder == 0 else -1
        br, policy = mccfr.best_response(tree, sigma, off, buckets, showdown, responder)
        mine = np.asarray(tree.actor_is_button != bool(responder))[np.repeat(np.arange(tree.n_dec), np.diff(off))]
        assert (policy[mine] >= 0).all() and (policy[~mine] == -1).all()
        assert legal[np.flatnonzero(mine), policy[mine]].all()          # a legal action everywhere
        # playing the policy against sigma is worth exactly the best-response value
        pure = sigma.copy()
        pure[mine] = np.eye(4)[policy[mine]]
        assert sign * mccfr.profile_value(tree, pure, off, buckets, showdown) == pytest.approx(br, abs=1e-9)
        # and no random strategy of the responder does better
        for _ in range(10):
            other = sigma.copy()
            other[mine] = random_strategy()[mine]
            assert sign * mccfr.profile_value(tree, other, off, buckets, showdown) <= br + 1e-9


# ---- training

def test_regret_matching_never_picks_illegal(tree, tmp_path):
    ck = run(tree, class_game(tmp_path / "g", k=3), 20_000, tmp_path / "out.npz")
    illegal = np.repeat(tree.child == -1, np.diff(ck.offset), axis=0)
    assert (ck.avg[illegal] == 0).all() and (ck.regret[illegal] == 0).all()
    assert (ck.avg[~illegal] > 0).any() and (ck.regret[~illegal] != 0).any()
    s = mccfr.average_strategy(ck.avg, tree, ck.offset)
    assert (s[illegal] == 0).all() and np.allclose(s.sum(axis=1), 1.0)


def test_checkpoint_roundtrip(tree, tmp_path):
    deals = class_game(tmp_path / "g", k=3)
    out = tmp_path / "run" / "out.npz"
    ck = run(tree, deals, 10_000, out, seed=7, eval_rows=10**9)
    m = ck.meta
    assert m["iterations"] == 10_000 and m["seed"] == 7 and m["linear"] is True
    assert m["n_dec"] == tree.n_dec and m["n_term"] == tree.n_term
    assert m["tree_child_sha256"] == mccfr.tree_sha256(tree)
    assert m["num_buckets"] == (3, 3, 3, 3) and m["buckets_sha256"] == "synthetic"
    assert (m["starting_stack"], m["small_blind"], m["big_blind"], m["raise_cap"]) == (80, 1, 2, 4)
    assert m["dataset_n"] == 3000 and m["dataset_seed"] == 0
    assert ck.regret.dtype == ck.avg.dtype == np.float32 and ck.regret.shape == (ck.offset[-1], 4)
    # every 10%: nine checkpoints and the final output, each with its diagnostic
    marks = [mccfr.load_checkpoint(p, tree, None).meta["iterations"] for p in mccfr.checkpoints(out)]
    assert marks == list(range(1000, 10_000, 1000))
    assert ck.diag[:, 0].tolist() == list(range(1000, 10_001, 1000)) and ck.diag[:, -1].all()
    # save -> load gives back every array and field
    again = tmp_path / "again.npz"
    mccfr.save_checkpoint(again, ck)
    ck2 = mccfr.load_checkpoint(again, tree, None)
    for name in ("regret", "avg", "offset", "visits", "diag"):
        assert np.array_equal(getattr(ck, name), getattr(ck2, name)), name
    assert ck2.meta == ck.meta


def test_resume_from_a_checkpoint_equals_an_uninterrupted_run(tree, tmp_path):
    """One thread is deterministic: the RNG is keyed by (seed, iteration) and the linear weight is
    the iteration number, so stopping at a checkpoint and resuming gives the same bits."""
    deals = class_game(tmp_path / "g", k=3)
    whole = run(tree, deals, 8000, tmp_path / "a" / "out.npz", count_visits=True)
    out = tmp_path / "b" / "out.npz"
    run(tree, deals, 8000, out, count_visits=True)
    for p in mccfr.checkpoints(out):                 # keep only the 40% checkpoint
        if mccfr.load_checkpoint(p, tree, None).meta["iterations"] != 3200:
            p.unlink()
    out.unlink()
    resumed = run(tree, deals, 8000, out, count_visits=True)
    assert resumed.meta["iterations"] == 8000
    for name in ("regret", "avg", "visits", "diag"):
        assert np.array_equal(getattr(whole, name), getattr(resumed, name)), name
    assert whole.visits.sum() > 0
    # a finished run is not redone, and a larger target continues it
    run(tree, deals, 8000, out, count_visits=True)
    assert mccfr.load_checkpoint(out, tree, None).meta["iterations"] == 8000
    longer = run(tree, deals, 9000, out, count_visits=True)
    assert longer.meta["iterations"] == 9000


def test_loading_refuses_another_tree_table_or_run(tree, tmp_path, monkeypatch):
    deals = class_game(tmp_path / "g", k=3)
    out = tmp_path / "out.npz"
    ck = run(tree, deals, 2000, out)
    with pytest.raises(ValueError, match="tree"):
        mccfr.load_checkpoint(out, _other_tree(tree), None)
    with pytest.raises(ValueError, match="sha256"):
        mccfr.load_checkpoint(out, tree, ab.DEFAULT_PATH)       # bucketed by "synthetic", not the table
    bad = dict(ck.meta, num_buckets=(3, 3, 3, 4))
    mccfr.save_checkpoint(tmp_path / "bad.npz", ck._replace(meta=bad))
    with pytest.raises(ValueError, match="num_buckets|offset"):
        mccfr.load_checkpoint(tmp_path / "bad.npz", tree, None)
    monkeypatch.setattr(game, "RAISE_CAP_PER_STREET", 5)
    with pytest.raises(ValueError, match="raise_cap"):
        mccfr.load_checkpoint(out, tree, None)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="seed"):               # resuming another run's output
        mccfr.train(tree, deals, 4000, str(out), 1, threads=1, buckets_table=None, log_every=1e9)
    other = class_game(tmp_path / "g2", k=3, n=2000)
    with pytest.raises(ValueError, match="dataset"):
        mccfr.train(tree, other, 4000, str(out), 0, threads=1, buckets_table=None, log_every=1e9)


def _other_tree(tree):
    import dataclasses
    child = tree.child.copy()
    child[[0, 1]] = child[[1, 0]]
    return dataclasses.replace(tree, child=child)


def test_training_refuses_buckets_outside_the_counts(tree, tmp_path):
    b = np.zeros((10, 2, 4), np.int16)
    b[3, 1, Street.RIVER] = 2
    with pytest.raises(ValueError, match="bucket"):
        mccfr.train(tree, write_dataset(tmp_path / "g", b, np.zeros(10), (2, 2, 2, 2)), 100,
                    str(tmp_path / "o.npz"), 0, threads=1, buckets_table=None)


def test_real_dataset_run_with_the_approximate_diagnostic(tree, tmp_path):
    out = tmp_path / "deals"
    gd.main(["--n", "64", "--seed", "2", "--shard-size", "64", "--workers", "1", "--out", str(out)])
    ck = run(tree, out, 3000, tmp_path / "o.npz", buckets_table=ab.DEFAULT_PATH, eval_rows=32,
             checkpoints_per_run=2)                  # 51 MB per checkpoint at the real counts
    assert ck.meta["num_buckets"] == tuple(ab.NUM_BUCKETS[s] for s in Street)
    assert ck.meta["buckets_sha256"] == gd.file_sha256(ab.DEFAULT_PATH)
    assert ck.offset[-1] == sum(ab.NUM_BUCKETS[s] * int((tree.street == s).sum()) for s in Street)
    assert np.isfinite(ck.diag).all() and not ck.diag[:, -1].any()          # labelled approximate


# ---- convergence (the known-answer test) and the plan's behaviour tests

def _exploitability_curve(tree, tmp_path, iterations, threads):
    ck = run(tree, class_game(tmp_path / "g", k=3), iterations, tmp_path / "o.npz", threads=threads,
             eval_rows=10**9)
    assert ck.diag[:, -1].all()                       # perfect recall, every row: exact
    return ck.diag[:, 4]


# bb/hand, from measurement (Task 18 report). The exact exploitability of the average strategy
# on this game after 200k iterations: 0.194 on one thread (deterministic), 0.193 on two and
# 0.209 on eight threads; from 1.36 at 20k. It keeps falling about as 1/sqrt(T): 0.032 at 2M,
# 0.010 at 20M. A wrong sign, regret update, traverser alternation or averaging weight stalls
# or diverges instead (two one-player runs per block of 4096 reach 1.02 at 1M).
EXPLOITABILITY_AT_200K = 0.25
EXPLOITABILITY_AT_4M = 0.035       # measured 0.026 on eight threads


@pytest.mark.parametrize("threads", [1, 2])
def test_exploitability_falls_on_a_perfect_recall_game(tree, tmp_path, threads):
    e = _exploitability_curve(tree, tmp_path, 200_000, threads)
    assert e[-1] < EXPLOITABILITY_AT_200K, e
    assert e[-1] < e[0] / 3, e
    if threads == 1:                                  # deterministic: falls at every checkpoint
        assert all(later <= 1.1 * earlier for earlier, later in zip(e, e[1:])), e


@pytest.mark.slow
def test_exploitability_keeps_falling_with_hogwild_threads(tree, tmp_path):
    e = _exploitability_curve(tree, tmp_path, 4_000_000, threads=8)
    assert e[-1] < EXPLOITABILITY_AT_4M, e
    assert e[-1] < e[0] / 3 and e[-1] < e[len(e) // 2], e


@pytest.mark.slow
def test_one_bucket_game_is_seed_stable(tree, tmp_path):
    """Measured on 4 threads, seeds 1 and 2: both converge to +0.345 bb for the button (these
    4,000 random showdowns do not average exactly 0, so the big blind folds to avoid big
    pots). The two values differ by 0.024 at 500k iterations, by <= 0.003 from 1M on."""
    rng = np.random.default_rng(0)
    deals = write_dataset(tmp_path / "g", np.zeros((4000, 2, 4)), rng.integers(-1, 2, 4000), (1, 1, 1, 1))
    buckets, showdown = _rows(deals)
    values = []
    for seed in (1, 2):
        ck = run(tree, deals, 2_000_000, tmp_path / f"o{seed}.npz", seed=seed, threads=4, eval_rows=0)
        s = mccfr.average_strategy(ck.avg, tree, ck.offset)
        values.append(mccfr.profile_value(tree, s, ck.offset, buckets, showdown))
    assert abs(values[0] - values[1]) < 0.02, values


@pytest.mark.slow
def test_strong_bucket_raises_more_than_weak(tree, tmp_path):
    """100 buckets a street; each player holds bucket 0, 33, 66 or 99 for the whole hand and the
    higher one wins, so bucket 99 always wins (splits against 99).

    Measured after 3M iterations on 4 threads, 8 runs. The hand is checked down to the river.
    - The big blind acts first there. Bucket 99 raises 0.80-0.85: it checks some to check-raise,
      so 0.8 is no margin. The bluff-catchers 33 and 66 raise 0.00 and 0.15-0.19, and bucket 0
      bluffs 0.21-0.23.
    - When the big blind checks, the button acts: checking back ends the hand, so trapping is
      worthless and bucket 99 raises 1.000 in every run. Bucket 0 also bets 1.000 there (it
      cannot win a showdown, so a bluff is free); 33 and 66 bet 0.15-0.37.
    With information only on the river instead, the check-down line is almost never played, so
    these nodes' averages stay near the prior (2,400 visits in 10M iterations)."""
    rng = np.random.default_rng(3)
    n = 20_000
    cls = rng.choice(np.array([0, 33, 66, 99]), size=(n, 2))
    deals = write_dataset(tmp_path / "g", np.repeat(cls[:, :, None], 4, axis=2),
                          np.sign(cls[:, 0] - cls[:, 1]), (100, 100, 100, 100))
    ck = run(tree, deals, 3_000_000, tmp_path / "o.npz", threads=4, eval_rows=0)
    s = mccfr.average_strategy(ck.avg, tree, ck.offset)

    def raise_p(node):
        rows = s[ck.offset[node]:ck.offset[node + 1]]
        return rows[:, H] + rows[:, P]

    bb_first = node_from_actions((C,) * 6, tree)        # the first river node: checked down, BB to act
    button_first = node_from_actions((C,) * 7, tree)    # the button's first river node: checked to
    assert tree.street[bb_first] == tree.street[button_first] == Street.RIVER
    assert not tree.actor_is_button[bb_first] and tree.actor_is_button[button_first]
    p = raise_p(button_first)
    assert p[99] > 0.8, p[[0, 33, 66, 99]]
    assert p[99] > p[[33, 66]].max() + 0.4, p[[0, 33, 66, 99]]
    p = raise_p(bb_first)
    assert p[99] > 0.7 and p[99] > p[[0, 33, 66]].max() + 0.4, p[[0, 33, 66, 99]]


@pytest.mark.slow
def test_sigterm_checkpoints_and_resume_is_exact(tree, tmp_path):
    deals = class_game(tmp_path / "g", k=3)
    out = tmp_path / "run" / "out.npz"
    cmd = [sys.executable, "-m", "train.mccfr", "--deals", str(deals), "--buckets", "none",
           "--iterations", str(10**9), "--out", str(out), "--threads", "1", "--log-every", "0",
           "--eval-rows", "0"]
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, PYTHONUNBUFFERED="1"))
    lines = []
    for line in proc.stdout:                          # wait for training to be under way
        lines.append(line)
        if "it/s" in line:
            break
    proc.send_signal(signal.SIGTERM)
    rest, _ = proc.communicate(timeout=60)
    assert proc.returncode == 128 + signal.SIGTERM, "".join(lines) + rest
    (ckpt,) = mccfr.checkpoints(out)
    k = mccfr.load_checkpoint(ckpt, tree, None).meta["iterations"]
    assert 0 < k < 10**9
    resumed = run(tree, deals, k + 3000, out, eval_rows=0)
    whole = run(tree, deals, k + 3000, tmp_path / "whole" / "out.npz", eval_rows=0)
    assert np.array_equal(resumed.regret, whole.regret) and np.array_equal(resumed.avg, whole.avg)
