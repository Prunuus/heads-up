"""Blueprint export (train/export.py) and the runtime sampler (bot/strategy.py)."""

import random
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from bot import strategy
from bot.strategy import Blueprint
from bot.tree import get_tree
from game import Action
from tests.test_mccfr import class_game
from train import export, mccfr

F, C, H, P = Action


@pytest.fixture(scope="module")
def tree():
    return get_tree()


@pytest.fixture(scope="module")
def ckpt(tree, tmp_path_factory):
    d = tmp_path_factory.mktemp("bp")
    deals = class_game(d / "deals", n=2000)
    out = d / "raw.npz"
    mccfr.train(tree, deals, 20000, str(out), 0, threads=1, buckets_table=None, log_every=1e9,
                count_visits=True, eval_rows=0)
    return out


@pytest.fixture(scope="module")
def bp_path(tree, ckpt, tmp_path_factory):
    out = tmp_path_factory.mktemp("exp") / "blueprint.npz"
    export.export(ckpt, out, tree=tree, buckets_table=None)
    return out


@pytest.fixture(scope="module")
def bp(tree, bp_path):
    return Blueprint.load(bp_path, tree=tree, buckets_path=None)


def test_load_under_1s_and_meta_present(tree, bp_path):
    t0 = time.perf_counter()
    bp = Blueprint.load(bp_path, tree=tree, buckets_path=None)
    assert time.perf_counter() - t0 < 1.0
    m = bp.meta
    assert (m["stack"], m["sb"], m["bb"], m["cap"]) == (80, 1, 2, 4)
    assert m["buckets"] == (3, 3, 3, 3) and m["iterations"] == 20000
    assert m["n_dec"] == tree.n_dec and m["n_term"] == tree.n_term
    assert len(m["tree_child_sha256"]) == 64 and "seed" in m and "linear" in m


def test_export_matches_the_average_strategy_per_street(tree, ckpt, bp):
    ck = mccfr.load_checkpoint(ckpt, tree, None)
    strat = mccfr.average_strategy(ck.avg, tree, ck.offset)
    for s in range(4):
        rows = mccfr.street_rows(tree, ck.offset, s)
        got = np.stack([bp.probs(int(n), b) for n in np.flatnonzero(tree.street == s) for b in range(3)])
        assert np.abs(got - strat[rows]).max() < 2e-3          # float16 rounding
    for n in (0, 5, 100, 5000):
        for b in range(3):
            p = bp.probs(n, b)
            assert p.shape == (4,) and abs(p.sum() - 1) < 1e-9
            assert (p[tree.child[n] == -1] == 0).all()


def test_arrays_are_read_only(bp):
    assert not bp.table.flags.writeable
    with pytest.raises(TypeError):
        bp.meta["stack"] = 1


def test_act_always_legal_even_if_all_mass_illegal(tree, bp):
    rng = random.Random(1)
    for n in range(0, tree.n_dec, 97):
        legal = tree.legal[n]
        for b in range(3):
            assert bp.act(n, b, legal, rng) in legal
    # a node where the blueprint is certain of FOLD/raise but only CHECK_CALL is offered
    n = next(n for n in range(tree.n_dec) if bp.probs(n, 0)[C] == 0.0 and tree.child[n, C] != -1)
    assert bp.act(n, 0, (C,), rng) == C
    # no legal action carries any mass: still a legal action
    n = next(n for n in range(tree.n_dec) if tree.child[n, F] == -1)
    assert bp.act(n, 0, (F,), rng, purify=0.0) == F
    assert bp.act(n, 0, (F, H), rng, purify=0.9) in (F, H)
    with pytest.raises(ValueError):
        bp.act(0, 0, (), rng)


def test_purify_zero_equals_raw_sampling_distribution(tree, bp):
    # a node/bucket with a mixed strategy
    n, b = next((n, b) for n in range(tree.n_dec) for b in range(3)
                if (bp.probs(n, b) > 0.15).sum() >= 3)
    legal = tree.legal[n]
    p = bp.probs(n, b)[list(legal)]
    p = p / p.sum()
    rng = np.random.default_rng(7)
    draws = np.bincount([int(bp.act(n, b, legal, rng, purify=0.0)) for _ in range(10000)], minlength=4)
    exp = np.zeros(4)
    exp[list(legal)] = p * 10000
    keep = exp > 0
    chi2 = (((draws - exp)[keep]) ** 2 / exp[keep]).sum()
    assert chi2 < {1: 10.8, 2: 13.8, 3: 16.3}.get(int(keep.sum()) - 1, 20.0)   # p = 0.001 critical values
    assert (draws[~keep] == 0).all()


def test_purify_drops_small_probabilities(tree, bp):
    n, b = next((n, b) for n in range(tree.n_dec) for b in range(3)
                if 0 < bp.probs(n, b)[list(tree.legal[n])].min() < 0.05 and len(tree.legal[n]) > 2)
    legal = tree.legal[n]
    small = [a for a in legal if bp.probs(n, b)[a] < 0.1]
    rng = random.Random(3)
    got = {bp.act(n, b, legal, rng, purify=0.1) for _ in range(2000)}
    assert got and not got & set(small)


def test_load_refuses_a_changed_tree_or_table(tree, ckpt, bp_path, tmp_path):
    with pytest.raises(ValueError):
        Blueprint.load(bp_path, tree=tree)                      # data/buckets.npz is not the synthetic table
    assert Blueprint.try_load(bp_path, tree=tree) is None
    z = dict(np.load(bp_path))
    z["tree_child_sha256"] = np.array("0" * 64)
    bad = tmp_path / "bad.npz"
    np.savez(bad, **z)
    with pytest.raises(ValueError):
        Blueprint.load(bad, tree=tree, buckets_path=None)
    z = dict(np.load(bp_path))
    z["stack"] = np.int64(100)
    np.savez(bad, **z)
    with pytest.raises(ValueError):
        Blueprint.load(bad, tree=tree, buckets_path=None)


def test_missing_or_corrupt_file_never_raises_from_try_load(tree, bp_path, tmp_path):
    assert Blueprint.try_load(tmp_path / "nope.npz", tree=tree, buckets_path=None) is None
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"not a zip")
    assert Blueprint.try_load(junk, tree=tree, buckets_path=None) is None
    trunc = tmp_path / "trunc.npz"
    trunc.write_bytes(Path(bp_path).read_bytes()[:5000])
    assert Blueprint.try_load(trunc, tree=tree, buckets_path=None) is None
    with pytest.raises(ValueError):
        Blueprint.load(junk, tree=tree, buckets_path=None)
    assert strategy.load_default(tmp_path / "nope.npz") is None   # never raises


def test_probs_is_cheap(tree, bp):
    rng = np.random.default_rng(0)
    nodes = rng.integers(0, tree.n_dec, 2000)
    ts = []
    for n in nodes:
        t0 = time.perf_counter()
        bp.probs(int(n), 1)
        ts.append(time.perf_counter() - t0)
    assert np.percentile(ts, 99) < 50e-6


def test_export_refuses_a_checkpoint_from_another_table(tree, ckpt, tmp_path):
    with pytest.raises(ValueError):
        export.export(ckpt, tmp_path / "x.npz", tree=tree)      # real table; checkpoint says "synthetic"
    assert not (tmp_path / "x.npz").exists()
