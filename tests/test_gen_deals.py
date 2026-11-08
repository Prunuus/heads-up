"""Training deal dataset (Task 17): train/gen_deals.py shards, loader and re-bucketing."""

import shutil
from pathlib import Path

import numpy as np
import pytest

import evaluator as ev
from bot import abstraction as ab
from game import Action, Deal, Game, Street
from train import gen_deals as gd

WIDTH = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
SMALL = ["--n", "200", "--seed", "3", "--shard-size", "64"]      # shards of 64, 64, 64, 8


def _gen(out, *extra):
    gd.main(SMALL + ["--workers", "1", "--out", str(out)] + list(extra))


def _shards(path):
    """{file name: {array name: array}} for every finished shard in a directory."""
    out = {}
    for f in sorted(Path(path).glob("deals_*.npz")):
        with np.load(f) as z:
            out[f.name] = {k: z[k] for k in z.files}
    return out


def _assert_same(a, b):
    assert sorted(a) == sorted(b)
    for name in a:
        assert sorted(a[name]) == sorted(b[name]), name
        for k in a[name]:
            assert a[name][k].dtype == b[name][k].dtype, (name, k)
            assert np.array_equal(a[name][k], b[name][k]), (name, k)


def _copy(src, dst):
    shutil.copytree(src, dst)
    return dst


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    out = tmp_path_factory.mktemp("small")
    _gen(out)
    return out


@pytest.fixture(scope="module")
def alt_table(tmp_path_factory):
    """A refit with other bucket counts and the committed table's designs (seed 0)."""
    from train import fit_buckets
    out = tmp_path_factory.mktemp("alt") / "buckets.npz"
    fit_buckets.main(["--per-street", "300", "--validate", "0", "--seed", "0", "--workers", "1",
                      "--preflop", "40", "--flop-grid", "4x5", "--turn-grid", "5x4", "--river", "30",
                      "--out", str(out)])
    return out


# ---- the plan's test and the brief's content checks

def test_small_dataset_shapes_and_outcomes(small):
    d = gd.load_deals(small, cards=True, features=True)
    n = 200
    assert d.buckets.shape == (n, 2, 4) and d.buckets.dtype == np.int16
    assert d.showdown.shape == (n,) and d.showdown.dtype == np.int8
    assert d.cards.shape == (n, 15) and d.cards.dtype == np.int8
    assert d.features["features"].shape == (n, 2, 6) and d.features["features"].dtype == np.float64
    assert d.features["river_nut"].shape == (n, 2) and d.features["river_nut"].dtype == bool
    assert d.features["river_category"].shape == (n, 2) and d.features["river_category"].dtype == np.int8
    assert d.buckets.flags.c_contiguous and d.showdown.flags.c_contiguous
    assert d.meta["n"] == n and d.meta["shards"] == 4 and d.meta["seed"] == 3
    assert d.meta["buckets_sha256"] == gd.file_sha256(ab.DEFAULT_PATH)
    assert d.meta["num_buckets"] == tuple(ab.NUM_BUCKETS[s] for s in Street)

    # the cards are the deals regenerated from (seed, shard)
    regenerated = np.concatenate([gd.deal_cards(3, k, c) for k, c in enumerate((64, 64, 64, 8))])
    assert np.array_equal(d.cards, regenerated)

    for i, c in enumerate(d.cards.tolist()):
        button, big_blind = ev.omaha_score(c[0:5], c[10:15]), ev.omaha_score(c[5:10], c[10:15])
        assert d.showdown[i] == (button > big_blind) - (button < big_blind), i
    assert {-1, 1} <= set(d.showdown.tolist()) <= {-1, 0, 1}

    for st in Street:          # every street's ids in range, and not all equal
        b = d.buckets[:, :, st]
        assert b.min() >= 0 and b.max() < ab.NUM_BUCKETS[st] and len(np.unique(b)) > 10


def test_showdown_is_the_engines_check_down_winner(small):
    """Seat 0 of a row is the button: play the deal through game.Game to showdown."""
    d = gd.load_deals(small, cards=True)
    for i, c in enumerate(d.cards[:40].tolist()):
        button_seat = i % 2
        holes = [None, None]
        holes[button_seat], holes[1 - button_seat] = tuple(sorted(c[0:5])), tuple(sorted(c[5:10]))
        game = Game(Deal(hole_cards=tuple(holes), board=tuple(c[10:15])), button_seat=button_seat)
        while not game.is_terminal:
            game.apply(Action.CHECK_CALL)
        r = game.result()
        assert r.went_to_showdown
        want = 0 if r.winner_seat is None else (1 if r.winner_seat == button_seat else -1)
        assert d.showdown[i] == want, i


def test_every_row_has_15_distinct_cards_in_range(small):
    cards = gd.load_deals(small, cards=True).cards.astype(np.int64)
    assert cards.min() >= 0 and cards.max() <= 35
    assert all(len(set(row)) == 15 for row in cards.tolist())
    assert len(np.unique(cards)) == 36                   # the whole short deck gets dealt
    holes = cards[:, :10].reshape(-1, 2, 5)
    assert (np.diff(holes, axis=2) > 0).all()            # each hole sorted, as game.Deal has it


def test_buckets_equal_runtime_bucket_on_the_stored_cards(small):
    d = gd.load_deals(small, cards=True)
    for i, c in enumerate(d.cards.tolist()):
        for seat in (0, 1):
            hole = c[5 * seat:5 * seat + 5]
            for st in Street:
                assert d.buckets[i, seat, st] == ab.bucket(st, hole, c[10:10 + WIDTH[st]]), (i, seat, st)


def test_stored_features_are_the_runtime_features_and_cut_to_the_stored_buckets(small):
    """The re-bucketing precondition: buckets from the stored features == bucket() on the cards."""
    d = gd.load_deals(small, cards=True, features=True)
    t = ab.load_buckets()
    f, nut, cat = (d.features[k] for k in ("features", "river_nut", "river_category"))
    for i, c in enumerate(d.cards[:50].tolist()):
        for seat in (0, 1):
            hole = c[5 * seat:5 * seat + 5]
            got = [t.features(st, hole, c[10:10 + WIDTH[st]]) for st in Street]
            assert tuple(f[i, seat]) == (got[0].strength, got[1].strength, got[1].equity,
                                         got[2].strength, got[2].equity, got[3].equity)
            assert nut[i, seat] == got[3].nut and cat[i, seat] == got[3].category
    assert np.array_equal(gd.buckets_from_features(t, f, nut, cat), d.buckets)


def test_vectorized_cut_equals_bucket_from_features_on_and_beside_the_edges():
    t = ab.load_buckets()
    rng = np.random.default_rng(0)
    n = 4000

    def beside(e):
        """Each value: the edge itself, one ulp either side, uniform, or an extreme."""
        return np.choose(rng.integers(0, 5, len(e)), [e, np.nextafter(e, -1), np.nextafter(e, 2),
                                                      rng.random(len(e)), rng.choice([0.0, 1.0], len(e))])

    f = np.empty((n, 2, 6))
    for seat in (0, 1):
        f[:, seat, 0] = beside(rng.choice(t.preflop_edges, n))
        for col, st in ((1, Street.FLOP), (3, Street.TURN)):
            s = beside(rng.choice(t.strength_edges[st], n))
            row = np.searchsorted(t.strength_edges[st], s, side="right")     # equity edges of s's row
            f[:, seat, col] = s
            f[:, seat, col + 1] = beside(t.equity_edges[st][row, rng.integers(0, t.grid[st][1] - 1, n)])
        f[:, seat, 5] = beside(rng.choice(t.river_edges, n))
    nut = rng.random((n, 2)) < 0.3
    cat = rng.integers(0, 9, (n, 2)).astype(np.int8)
    got = gd.buckets_from_features(t, f, nut, cat)
    assert got.dtype == np.int16 and got.shape == (n, 2, 4)
    for i in range(n):
        for seat in (0, 1):
            x = f[i, seat]
            want = [t.bucket_from_features(Street.PREFLOP, ab.Features(x[0], x[0], False, 0)),
                    t.bucket_from_features(Street.FLOP, ab.Features(x[1], x[2], False, 0)),
                    t.bucket_from_features(Street.TURN, ab.Features(x[3], x[4], False, 0)),
                    t.bucket_from_features(Street.RIVER, ab.Features(x[5], x[5], bool(nut[i, seat]),
                                                                     int(cat[i, seat])))]
            assert list(got[i, seat]) == want, (i, seat)


# ---- determinism and resume

def test_output_depends_on_the_seed_not_on_the_worker_count(tmp_path, small):
    again, two = tmp_path / "again", tmp_path / "two"
    _gen(again)
    gd.main(SMALL + ["--workers", "2", "--out", str(two)])
    _assert_same(_shards(small), _shards(again))
    _assert_same(_shards(small), _shards(two))
    other = tmp_path / "other"
    gd.main(["--n", "64", "--seed", "4", "--shard-size", "64", "--workers", "1", "--out", str(other)])
    assert not np.array_equal(_shards(other)["deals_00000.npz"]["cards"],
                              _shards(small)["deals_00000.npz"]["cards"])


def test_resume_ignores_a_partial_temp_file_and_finishes_identically(tmp_path, small):
    out = _copy(small, tmp_path / "resumed")
    kept = (out / "deals_00000.npz").stat()
    (out / "deals_00001.npz").unlink()                  # interrupted before shards 1 and 3 landed
    (out / "deals_00003.npz").unlink()
    partial = out / "deals_00001.npz.12345.tmp"         # a worker died mid-write
    partial.write_bytes((small / "deals_00001.npz").read_bytes()[:1000])
    with pytest.raises(ValueError, match="missing"):
        gd.load_deals(out)                              # a gap is never loaded as a dataset
    _gen(out)
    _assert_same(_shards(small), _shards(out))
    assert not list(out.glob("*.tmp"))
    assert (out / "deals_00000.npz").stat().st_mtime_ns == kept.st_mtime_ns    # skipped, not redone


def test_raising_n_adds_shards_on_top_of_the_existing_ones(tmp_path, small):
    out = tmp_path / "grown"
    gd.main(["--n", "100", "--seed", "3", "--shard-size", "64", "--workers", "1", "--out", str(out)])
    assert gd.load_deals(out).meta["n"] == 100           # shards of 64 and 36
    first = (out / "deals_00000.npz").stat()
    _gen(out)                                            # --n 200
    _assert_same(_shards(small), _shards(out))
    assert (out / "deals_00000.npz").stat().st_mtime_ns == first.st_mtime_ns
    assert gd.load_deals(out, n=150).buckets.shape == (150, 2, 4)
    assert np.array_equal(gd.load_deals(out, n=150).showdown, gd.load_deals(small).showdown[:150])


def test_generation_refuses_to_mix_seeds_or_shard_sizes(tmp_path, small):
    out = _copy(small, tmp_path / "mixed")
    with pytest.raises(ValueError, match="seed"):
        gd.main(["--n", "200", "--seed", "4", "--shard-size", "64", "--workers", "1", "--out", str(out)])
    with pytest.raises(ValueError, match="shard size"):
        gd.main(["--n", "200", "--seed", "3", "--shard-size", "50", "--workers", "1", "--out", str(out)])
    _assert_same(_shards(small), _shards(out))


# ---- the loader's checks

def test_load_deals_rejects_a_bucket_table_mismatch(tmp_path, small, alt_table):
    with pytest.raises(ValueError, match="sha256"):
        gd.load_deals(small, buckets_table=alt_table)
    # any byte change counts, even with the same bucket counts
    same_counts = tmp_path / "buckets.npz"
    with np.load(ab.DEFAULT_PATH) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["validation"] = arrays["validation"] + 1.0
    np.savez(same_counts, **arrays)
    with pytest.raises(ValueError, match="sha256"):
        gd.load_deals(small, buckets_table=same_counts)
    assert gd.load_deals(small).meta["n"] == 200         # the default table is the dataset's


def test_load_deals_rejects_features_from_another_preflop_table(tmp_path, small, monkeypatch):
    """T17 M1: the same bucket table with a changed data/preflop.npz changes every preflop
    feature, so the dataset no longer matches the runtime: load_deals refuses it."""
    changed = tmp_path / "preflop.npz"
    with np.load(gd.PREFLOP_TABLE) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["strength"] = arrays["strength"] * 0.5
    np.savez(changed, **arrays)
    monkeypatch.setattr(gd, "PREFLOP_TABLE", changed)
    with pytest.raises(ValueError, match="features"):
        gd.load_deals(small)
    assert gd.load_deals(small, buckets_table=None).meta["n"] == 200     # synthetic data: no checks
    monkeypatch.undo()
    assert gd.load_deals(small).meta["n"] == 200


def test_preflop_table_is_the_one_the_runtime_reads():
    """T17 M4: the features hash covers the file bot.preflop loads."""
    from bot import preflop
    assert gd.PREFLOP_TABLE == preflop._DEFAULT_PATH


def test_load_deals_rejects_mixed_shards(tmp_path, small):
    out = _copy(small, tmp_path / "foreign")
    other = tmp_path / "other"
    gd.main(["--n", "128", "--seed", "4", "--shard-size", "64", "--workers", "1", "--out", str(other)])
    shutil.copy(other / "deals_00001.npz", out / "deals_00001.npz")
    with pytest.raises(ValueError, match="seed"):
        gd.load_deals(out)
    with pytest.raises(ValueError, match="no deal shards"):
        gd.load_deals(tmp_path / "empty_dir_that_does_not_exist")


# ---- re-bucketing

def test_rebucket_equals_fresh_bucketing(tmp_path, small, alt_table):
    out = _copy(small, tmp_path / "rebucketed")
    assert gd.rebucket(out, alt_table) == 4
    with pytest.raises(ValueError, match="sha256"):
        gd.load_deals(out)                               # the dataset now belongs to the alt table
    d = gd.load_deals(out, buckets_table=alt_table, cards=True)
    assert d.meta["num_buckets"] == (40, 20, 20, 30)
    assert d.meta["buckets_sha256"] == gd.file_sha256(alt_table)
    t = ab.load_buckets(alt_table)
    for i, c in enumerate(d.cards.tolist()):
        for seat in (0, 1):
            hole = c[5 * seat:5 * seat + 5]
            for st in Street:
                assert d.buckets[i, seat, st] == t.bucket(st, hole, c[10:10 + WIDTH[st]]), (i, seat, st)

    fresh = tmp_path / "fresh"
    _gen(fresh, "--buckets", str(alt_table))
    _assert_same(_shards(fresh), _shards(out))           # every array, hashes and meta included

    assert gd.rebucket(out, alt_table) == 0              # already done: nothing rewritten
    assert gd.rebucket(out, ab.DEFAULT_PATH) == 4        # and back: the original dataset
    _assert_same(_shards(small), _shards(out))


def test_rebucket_refuses_a_table_with_other_feature_designs(tmp_path, small):
    from train import fit_buckets
    other = tmp_path / "seed1.npz"
    fit_buckets.main(["--per-street", "300", "--validate", "0", "--seed", "1", "--workers", "1",
                      "--out", str(other)])
    out = _copy(small, tmp_path / "kept")
    with pytest.raises(ValueError, match="features"):
        gd.rebucket(out, other)
    _assert_same(_shards(small), _shards(out))
