"""Card abstraction (Task 16): street buckets for the CFR blueprint."""

import os
import shutil
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

import evaluator as ev
from bot import abstraction as ab
from bot.equity import equity_vs_random, nut_flags, strength_now
from bot.fasteval import made_now_batch
from bot.preflop import preflop_strength
from bot.timing import TimeManager
from game import Street

REPO = Path(__file__).resolve().parent.parent
WIDTH = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
POSTFLOP = (Street.FLOP, Street.TURN, Street.RIVER)


def _spot(rng, width):
    cards = rng.permutation(36)
    return [int(c) for c in cards[:5]], [int(c) for c in cards[5:5 + width]]


def _spots(seed, street, n):
    rng = np.random.default_rng(seed)
    return [_spot(rng, WIDTH[street]) for _ in range(n)]


def _grid(street):
    return ab.load_buckets().grid[street]


def _adjacent(street, a, b):
    """Same or neighbouring bucket. Flop/turn ids are row * cols + col on a grid, so a
    neighbour is a king's move away; preflop and river ids are 1-D (|a - b| <= 1)."""
    if street in (Street.FLOP, Street.TURN):
        cols = _grid(street)[1]
        return abs(a // cols - b // cols) <= 1 and abs(a % cols - b % cols) <= 1
    return abs(a - b) <= 1


def _occupancy(street, n, seed):
    counts = np.zeros(ab.NUM_BUCKETS[street], dtype=np.int64)
    for hole, board in _spots(seed, street, n):
        counts[ab.bucket(street, hole, board, 0, None)] += 1
    return counts / (n / len(counts))


# ---- the brief's four tests

@pytest.mark.parametrize("street", list(Street))
def test_bucket_range_and_determinism(street):
    nb = ab.NUM_BUCKETS[street]
    for i, (hole, board) in enumerate(_spots(10 + street, street, 150)):
        b = ab.bucket(street, hole, board, 128, np.random.default_rng(i))
        assert type(b) is int and 0 <= b < nb
        assert ab.bucket(street, hole, board, 128, np.random.default_rng(i + 1)) == b
        # card order and container type do not matter
        assert ab.bucket(int(street), tuple(reversed(hole)), np.array(board[::-1], dtype=np.int8),
                         64, None) == b


# A fresh sample (seeds the fit never uses). 20k per street is the slow test below; the
# default suite checks 20k preflop hands and 6k spots per postflop street. At 6k a
# bucket of exactly 1% mass sits 4.7 sd above the 0.4x floor.
@pytest.mark.parametrize("street,n", [(Street.PREFLOP, 20_000), (Street.FLOP, 6_000),
                                      (Street.TURN, 6_000), (Street.RIVER, 6_000)])
def test_bucket_occupancy_roughly_uniform(street, n):
    occ = _occupancy(street, n, 900 + street)
    print(street.name, "occupancy min", occ.min(), "max", occ.max())
    assert occ.min() >= 0.4 and occ.max() <= 2.5


@pytest.mark.slow
@pytest.mark.parametrize("street", list(Street))
def test_bucket_occupancy_roughly_uniform_20k(street):
    occ = _occupancy(street, 20_000, 950 + street)
    print(street.name, "occupancy min", occ.min(), "max", occ.max())
    assert occ.min() >= 0.4 and occ.max() <= 2.5


@pytest.mark.parametrize("street", [Street.FLOP, Street.TURN])
def test_runtime_bucket_agrees_with_offline(street):
    spots = _spots(30 + street, street, 200)
    runtime = [ab.bucket(street, h, b, 128, np.random.default_rng(i)) for i, (h, b) in enumerate(spots)]
    offline = [ab.bucket(street, h, b, 2000, np.random.default_rng(10_000 + i))
               for i, (h, b) in enumerate(spots)]
    near = np.mean([_adjacent(street, x, y) for x, y in zip(runtime, offline)])
    assert near >= 0.80
    # The features are deterministic, so the agreement is in fact exact.
    assert runtime == offline


@pytest.mark.parametrize("street", list(Street))
def test_bucket_time_p99_under_1_5ms(street):
    # p99 over 400 spots of each spot's cost. A spot's cost is the best of 3 calls, so a
    # busy machine (other agents, other test workers) preempting one call is not counted,
    # while a spot that is slow every time still is.
    ab.warmup()
    spots = _spots(50 + street, street, 400)
    rng = np.random.default_rng(0)
    for hole, board in spots[:40]:
        ab.bucket(street, hole, board, 128, rng)
    ts = []
    for hole, board in spots:
        best = np.inf
        for _ in range(3):
            t = time.perf_counter()
            ab.bucket(street, hole, board, 128, rng)
            best = min(best, time.perf_counter() - t)
        ts.append(best * 1e3)
    print(street.name, "p50 %.3f ms p99 %.3f ms" % (np.median(ts), np.percentile(ts, 99)))
    assert np.percentile(ts, 99) < 1.5


# ---- the rollout budget (audit note: TimeManager.iters() returns 0 by design)

@pytest.mark.parametrize("street", list(Street))
def test_every_budget_gives_the_offline_bucket_and_samples_nothing(street):
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    for hole, board in _spots(60 + street, street, 30):
        offline = ab.bucket(street, hole, board, 2000, np.random.default_rng(1))
        for iters in (0, -1, 16, 32, 64, 128, 256):
            assert ab.bucket(street, hole, board, iters, rng) == offline
    assert rng.bit_generator.state == state


def test_exhausted_time_manager_budget_still_buckets():
    tm = TimeManager()
    tm.add(7.0)                                       # past the hard stop
    n = tm.iters(pot=60, street=Street.RIVER, hand_index=900)
    assert n == 0
    for street in Street:
        hole, board = _spot(np.random.default_rng(int(street)), WIDTH[street])
        assert 0 <= ab.bucket(street, hole, board, n, None) < ab.NUM_BUCKETS[street]


# ---- the features behind the buckets, checked against independent references

_POS5 = {}


def _exact_strength(hole, board):
    """Made hand now vs every opponent hole, ties half (as in tests/test_equity.py)."""
    unseen = np.array([c for c in range(36) if c not in hole and c not in board], dtype=np.int64)
    pos = _POS5.get(len(unseen))
    if pos is None:
        pos = _POS5[len(unseen)] = np.array(list(combinations(range(len(unseen)), 5)), dtype=np.int64)
    b = np.array([board], dtype=np.int64)
    theirs = made_now_batch(unseen[pos], b)
    ours = int(made_now_batch(np.array([hole], dtype=np.int64), b)[0])
    return float(((ours > theirs).sum() + 0.5 * (ours == theirs).sum()) / len(theirs))


@pytest.mark.parametrize("street,tol", [(Street.FLOP, 0.03), (Street.TURN, 0.03), (Street.RIVER, 0.015)])
def test_strength_matches_exact_enumeration(street, tol):
    errs = np.array([ab.features(street, h, b).strength - _exact_strength(h, b)
                     for h, b in _spots(70 + street, street, 8)])
    print(street.name, "max", np.abs(errs).max(), "rms", np.sqrt(np.mean(errs ** 2)))
    assert np.abs(errs).max() < tol
    assert np.sqrt(np.mean(errs ** 2)) < tol / 3


@pytest.mark.parametrize("street", [Street.FLOP, Street.TURN])
def test_equity_matches_monte_carlo(street):
    d = np.array([ab.features(street, h, b).equity - equity_vs_random(h, b, 20_000, np.random.default_rng(i))
                  for i, (h, b) in enumerate(_spots(80 + street, street, 14))])
    print(street.name, "max", np.abs(d).max(), "mean", d.mean())
    assert np.abs(d).max() < 0.03
    assert abs(d.mean()) < 0.01


def test_river_equity_is_the_strength():
    for hole, board in _spots(85, Street.RIVER, 20):
        f = ab.features(Street.RIVER, hole, board)
        assert f.equity == f.strength


def test_preflop_features_are_the_preflop_table():
    for hole, _ in _spots(86, Street.PREFLOP, 20):
        f = ab.features(Street.PREFLOP, hole, [])
        assert f.strength == f.equity == preflop_strength(hole)


@pytest.mark.parametrize("street", POSTFLOP)
def test_nut_flag_matches_nut_flags(street):
    nuts = 0
    for hole, board in _spots(87 + street, street, 250):
        f = ab.features(street, hole, board)
        assert f.nut == bool(nut_flags(hole, board) & 1)
        assert f.category == ev.category_of(max(
            ev.score_five([hole[a], hole[c], *[board[x] for x in t]])
            for a, c in combinations(range(5), 2) for t in combinations(range(len(board)), 3)))
        nuts += f.nut
    assert nuts >= 5          # the sample exercises the flag both ways


def test_strength_and_equity_separate_a_big_draw_from_a_made_hand():
    # Ace high with a wrap and the nut flush draw: almost no made strength now, lots of
    # equity (tests/test_equity.py). Bottom set on the same wet board: about the same
    # equity, but it is ahead of nearly every hand right now. A 1-D equity bucket would
    # put them side by side; the strength rows must not.
    draw = (ev.cards_from_str("Ah Jh Td 7c 6s"), ev.cards_from_str("9h 8h Kc"))
    made = (ev.cards_from_str("8c 8d Kd 6c 7d"), ev.cards_from_str("9h 8h Kc"))
    s_draw = strength_now(*draw, 6000, np.random.default_rng(1))
    s_made = strength_now(*made, 6000, np.random.default_rng(2))
    e_draw = equity_vs_random(*draw, 6000, np.random.default_rng(3))
    e_made = equity_vs_random(*made, 6000, np.random.default_rng(4))
    print("draw", s_draw, e_draw, "made", s_made, e_made)
    assert s_made > s_draw + 0.6 and abs(e_made - e_draw) < 0.15   # premise, by independent MC
    cols = _grid(Street.FLOP)[1]
    b_draw = ab.bucket(Street.FLOP, *draw, 0, None)
    b_made = ab.bucket(Street.FLOP, *made, 0, None)
    assert b_draw // cols < b_made // cols       # different strength rows
    assert b_draw % cols >= cols // 2            # the draw sits high in its row


# ---- bucket definitions

@pytest.mark.parametrize("hole,board,offset", [
    ("Ad Ah 9s Tc Jd", "Ac Kd 8h 7s 6c", 0),   # top set on a dry unpaired board: trips are the nuts
    ("Kh 9d 8s 8d Th", "Kc Kd 7h 6s 9c", 1),   # kings full of nines (Ks9x only ties): full house
    ("Qh Jh 8c 8s Tc", "Ah Kh 7h 9c 6d", 2),   # queen-jack of hearts: the nut flush, no straight flush
    ("9h 9s 8d Tc Jd", "9c 9d Kh 7s 6c", 3),   # quad nines
    ("Ah Th 6c 7d 8s", "Kh Qh Jh 7c 6d", 4),   # royal flush
])
def test_river_nuts_take_the_top_five_buckets_by_category(hole, board, offset):
    H, B = ev.cards_from_str(hole), ev.cards_from_str(board)
    assert nut_flags(H, B) & 1
    assert ab.bucket(Street.RIVER, H, B, 0, None) == ab.NUM_BUCKETS[Street.RIVER] - 5 + offset


def test_river_non_nut_hands_stay_below_the_nut_buckets():
    H, B = ev.cards_from_str("Kc Kd 9s 8d 6s"), ev.cards_from_str("Kh Qh Jh 7c 6d")  # trips, royal possible
    assert not nut_flags(H, B) & 1
    assert ab.bucket(Street.RIVER, H, B, 0, None) < ab.NUM_BUCKETS[Street.RIVER] - 5
    for hole, board in _spots(88, Street.RIVER, 300):
        if not nut_flags(hole, board) & 1:
            assert ab.bucket(Street.RIVER, hole, board, 0, None) < ab.NUM_BUCKETS[Street.RIVER] - 5


def test_preflop_bucket_is_monotone_in_preflop_strength():
    hands = [h for h, _ in _spots(89, Street.PREFLOP, 3000)]
    s = np.array([preflop_strength(h) for h in hands])
    b = np.array([ab.bucket(Street.PREFLOP, h, [], 0, None) for h in hands])
    order = np.argsort(s, kind="stable")
    assert (np.diff(b[order]) >= 0).all()
    assert b.min() <= 2 and b.max() >= ab.NUM_BUCKETS[Street.PREFLOP] - 3


@pytest.mark.parametrize("street", [Street.FLOP, Street.TURN])
def test_grid_rows_follow_strength_and_columns_follow_equity_within_a_row(street):
    rows, cols = _grid(street)
    assert rows * cols == ab.NUM_BUCKETS[street]
    f, b = [], []
    for hole, board in _spots(91 + street, street, 600):
        f.append(ab.features(street, hole, board))
        b.append(ab.bucket(street, hole, board, 0, None))
    s = np.array([x.strength for x in f])
    e = np.array([x.equity for x in f])
    r, c = np.array(b) // cols, np.array(b) % cols
    o = np.argsort(s, kind="stable")
    assert (np.diff(r[o]) >= 0).all()
    for row in range(rows):
        m = r == row
        o = np.argsort(e[m], kind="stable")
        assert (np.diff(c[m][o]) >= 0).all()
    assert len(set(r)) == rows                   # every row is used


def test_river_non_nut_bucket_is_monotone_in_equity():
    f, b = [], []
    for hole, board in _spots(93, Street.RIVER, 600):
        x = ab.features(Street.RIVER, hole, board)
        if not x.nut:
            f.append(x.equity)
            b.append(ab.bucket(Street.RIVER, hole, board, 0, None))
    o = np.argsort(f, kind="stable")
    assert (np.diff(np.array(b)[o]) >= 0).all()


def test_values_on_an_edge_go_to_the_upper_bucket_and_extremes_stay_in_range():
    t = ab.load_buckets()
    F = ab.Features
    e = t.preflop_edges
    k = len(e) // 2
    assert t.bucket_from_features(Street.PREFLOP, F(e[k], e[k], False, 0)) == k + 1
    below = np.nextafter(e[k], -np.inf)
    assert t.bucket_from_features(Street.PREFLOP, F(below, below, False, 0)) == k
    assert t.bucket_from_features(Street.PREFLOP, F(0.0, 0.0, False, 0)) == 0
    assert t.bucket_from_features(Street.PREFLOP, F(1.0, 1.0, False, 0)) == ab.NUM_BUCKETS[Street.PREFLOP] - 1

    rows, cols = _grid(Street.TURN)
    se, ee = t.strength_edges[Street.TURN], t.equity_edges[Street.TURN]
    assert se.shape == (rows - 1,) and ee.shape == (rows, cols - 1)
    s = se[2]                                    # exactly on the 3rd strength edge: row 3
    q = ee[3][4]                                 # exactly on row 3's 5th equity edge: column 5
    assert t.bucket_from_features(Street.TURN, F(s, q, False, 3)) == 3 * cols + 5
    assert t.bucket_from_features(Street.TURN, F(s, np.nextafter(q, -np.inf), False, 3)) == 3 * cols + 4
    assert t.bucket_from_features(Street.TURN, F(0.0, 0.0, False, 0)) == 0
    assert t.bucket_from_features(Street.TURN, F(1.0, 1.0, True, 8)) == rows * cols - 1

    nb = ab.NUM_BUCKETS[Street.RIVER]
    re = t.river_edges
    assert re.shape == (nb - 5 - 1,)
    assert t.bucket_from_features(Street.RIVER, F(re[10], re[10], False, 2)) == 11
    assert t.bucket_from_features(Street.RIVER, F(1.0, 1.0, False, 2)) == nb - 6   # top non-nut
    assert t.bucket_from_features(Street.RIVER, F(0.0, 0.0, False, 1)) == 0
    assert t.bucket_from_features(Street.RIVER, F(1.0, 1.0, True, 6)) == nb - 3


def test_loaded_tables_are_read_only():
    t = ab.load_buckets()
    arrays = [t.preflop_edges, t.river_edges, *t.strength_edges.values(), *t.equity_edges.values()]
    for a in arrays:
        assert not a.flags.writeable
        with pytest.raises(ValueError):
            a[0] = 0.5
    assert ab.load_buckets() is t                # one cached copy per process


def _table_arrays():
    with np.load(ab.DEFAULT_PATH) as z:
        return {k: z[k].copy() for k in z.files}


def _turn_river_in_hole(a):
    a["design_turn_river"][0] = ab._PAIRS[27][a["design_turn_pairs"][0, 0], 0]


def _flop_runout_in_hole(a):
    hole_card = ab._PAIRS[28][a["design_flop_pairs"][0, 0], 0]
    a["design_flop_runout"][0] = next(i for i, p in enumerate(ab._PAIRS[28]) if hole_card in p)


@pytest.mark.parametrize("corrupt", [
    lambda a: a.__setitem__("format", np.int64(2)),                           # newer format
    lambda a: a["num_buckets"].__setitem__(1, 50),                            # count 50, 10 x 10 grid
    lambda a: a.__setitem__("river_edges", a["river_edges"][:-1]),            # one edge short
    lambda a: a["turn_equity_edges"].__setitem__((2, 3), 2.0),                # edges out of order
    lambda a: a.pop("preflop_edges"),                                         # missing edges
    lambda a: a["design_strength_26"].__setitem__(0, 0),                      # pair 0 ten times
    lambda a: a["design_flop_pairs"].__setitem__((5, 1), a["design_flop_pairs"][5, 0]),  # a pair twice
    _turn_river_in_hole,                                                      # river card in the hole
    _flop_runout_in_hole,                                                     # runout card in the hole
])
def test_corrupt_table_is_rejected_at_load(corrupt):
    arrays = _table_arrays()
    ab.BucketTables(arrays)                      # the committed table loads
    corrupt(arrays)
    with pytest.raises(ValueError):
        ab.BucketTables(arrays)


@pytest.mark.parametrize("street,hole,board", [
    (Street.FLOP, [0, 1, 2, 3, 4], [5, 6, 7, 8]),     # a turn board on the flop
    (Street.RIVER, [0, 1, 2, 3, 4], [4, 5, 6, 7, 8]),  # card 4 twice
    (4, [0, 1, 2, 3, 4], []),                          # no such street
    (Street.PREFLOP, [0, 1, 2, 3], []),                # four hole cards
    (Street.FLOP, [0, 1, 2, 3, 36], [5, 6, 7]),        # not a card
    (Street.TURN, [0, 1, 2, 3, -1], [5, 6, 7, 8]),     # not a card
])
def test_bad_input_raises_value_error(street, hole, board):
    with pytest.raises(ValueError):
        ab.bucket(street, hole, board, 0, None)


# ---- loading, the module constant, and the offline fit

def _copy_code(dst):
    """A minimal copy of the runtime code (bot/, game.py, evaluator.py, the preflop table)."""
    for name in ("game.py", "evaluator.py"):
        shutil.copy(REPO / name, dst / name)
    shutil.copytree(REPO / "bot", dst / "bot", ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "data").mkdir()
    shutil.copy(REPO / "data" / "preflop.npz", dst / "data" / "preflop.npz")


def _run(cwd, code):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=120)


_PROBE = (
    "import bot.abstraction as a\n"
    "print(sorted((int(k), v) for k, v in a.NUM_BUCKETS.items()))\n"
    "try:\n"
    "    print('bucket', a.bucket(3, [0, 1, 2, 3, 4], [5, 6, 7, 8, 9], 0, None))\n"
    "except FileNotFoundError as e:\n"
    "    print('missing', e)\n"
)


def test_missing_table_raises_at_call_time_not_at_import(tmp_path):
    _copy_code(tmp_path)
    r = _run(tmp_path, _PROBE)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "[(0, 200), (1, 100), (2, 100), (3, 100)]"     # the plan's defaults
    assert lines[1].startswith("missing") and "buckets.npz" in lines[1]


def test_fit_script_is_deterministic_and_num_buckets_reflects_the_file(tmp_path):
    from train import fit_buckets
    args = ["--per-street", "300", "--validate", "0", "--seed", "5",
            "--preflop", "40", "--flop-grid", "4x5", "--turn-grid", "5x4", "--river", "30"]
    one, two = tmp_path / "one.npz", tmp_path / "two.npz"
    fit_buckets.main(args + ["--workers", "1", "--out", str(one)])
    fit_buckets.main(args + ["--workers", "2", "--out", str(two)])   # worker count changes nothing
    with np.load(one) as a, np.load(two) as b:
        assert sorted(a.files) == sorted(b.files)
        for k in a.files:
            assert np.array_equal(a[k], b[k], equal_nan=True), k     # validation is NaN: skipped
    t = ab.load_buckets(one)
    assert dict(t.num_buckets) == {Street.PREFLOP: 40, Street.FLOP: 20, Street.TURN: 20, Street.RIVER: 30}
    assert t.grid[Street.FLOP] == (4, 5) and t.grid[Street.TURN] == (5, 4)
    for street in Street:
        for hole, board in _spots(95 + street, street, 40):
            assert 0 <= t.bucket(street, hole, board) < t.num_buckets[street]

    # the module constant reads the table it will use
    code = tmp_path / "code"
    code.mkdir()
    _copy_code(code)
    shutil.copy(one, code / "data" / "buckets.npz")
    r = _run(code, _PROBE)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "[(0, 40), (1, 20), (2, 20), (3, 30)]"
    assert lines[1].startswith("bucket") and 0 <= int(lines[1].split()[1]) < 30


def test_fit_cuts_rows_by_strength_then_equity_within_each_row():
    from train.fit_buckets import fit, make_designs

    def synthetic(seed, n):
        # strength and equity strongly correlated, as on the turn: cutting equity on its
        # marginal deciles would leave the off-diagonal cells nearly empty
        rng = np.random.default_rng(seed)
        s = rng.random(n)
        e = np.clip(s + rng.normal(0.0, 0.08, n), 0.0, 1.0)
        nut = rng.random(n) < 0.05
        e[nut] = 1.0                           # the nuts have (almost) all the equity
        return np.column_stack([s, e, nut, rng.integers(4, 9, n)])

    grids = {Street.FLOP: (4, 5), Street.TURN: (5, 4)}
    feats = {st: synthetic(st, 20_000) for st in POSTFLOP}
    arrays = {"format": np.int64(ab.FORMAT), "num_buckets": np.array([10, 20, 20, 30]),
              "flop_grid": np.array(grids[Street.FLOP]), "turn_grid": np.array(grids[Street.TURN]),
              **fit(feats, np.random.default_rng(9).random(5000), 10, grids, 30), **make_designs(0)}
    t = ab.BucketTables(arrays)
    for st in POSTFLOP:
        f = synthetic(100 + st, 20_000)        # fresh points from the same distribution
        b = [t.bucket_from_features(st, ab.Features(x[0], x[1], bool(x[2]), int(x[3]))) for x in f]
        counts = np.bincount(b, minlength=t.num_buckets[st])
        if st == Street.RIVER:                 # the 25 equity buckets share the 95% non-nut mass
            counts = counts[:-ab.NUT_BUCKETS]
        occ = counts / counts.mean()
        assert occ.min() > 0.85 and occ.max() < 1.15, (st, occ.min(), occ.max())


def test_fit_edges_split_ties_toward_equal_mass():
    from train.fit_buckets import fit_edges
    # 40% of the mass on one value: a plain quantile cut would leave a bucket empty.
    x = np.concatenate([np.full(400, 0.5), np.linspace(0.0, 0.49, 300), np.linspace(0.51, 1.0, 300)])
    e = fit_edges(x, 5)
    assert e.shape == (4,) and (np.diff(e) > 0).all()
    b = np.searchsorted(e, x, side="right")
    counts = np.bincount(b, minlength=5)
    # the atom stays whole and every bucket gets mass: 200, 100, 400 (the atom), 100, 200
    assert len(set(b[:400])) == 1 and counts[b[0]] == 400          # x[:400] is the atom
    assert list(counts) == [200, 100, 400, 100, 200]
    y = np.linspace(0.0, 1.0, 1001)                                 # no ties: plain quantiles
    counts = np.bincount(np.searchsorted(fit_edges(y, 10), y, side="right"), minlength=10)
    assert set(counts) <= {100, 101} and counts.sum() == 1001
