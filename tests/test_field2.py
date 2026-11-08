"""FIELD2 split: the second-wave competitor bots, copied unmodified (sim/opponents/field2/README.md)."""
import hashlib
import re

import numpy as np
import pytest

import sim.tournament
from sim import heuristic_bot
from sim.match import REPO_ROOT, play_half
from sim.opponents import FIELD, FIELD2, HOLDOUT, TRAIN, ZOO
from sim.tournament import select_opponents
from sim.tune import train_opponents
from tests.test_field import _assert_played_its_strategy, swallowed_exceptions

FIELD2_DIR = REPO_ROOT / "sim" / "opponents" / "field2"
SMOKE_DEALS = 10
NAMES = {"field2_solver", "field2_bayes", "field2_adaptive"}


@pytest.fixture(scope="module")
def loaded():
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = ZOO[name]()
        return cache[name]
    return get


def test_field2_names():
    assert FIELD2 == NAMES
    assert FIELD2 <= set(ZOO)


def test_splits_partition_the_zoo():
    assert HOLDOUT | TRAIN | FIELD | FIELD2 == set(ZOO)
    for a, b in [(HOLDOUT, TRAIN), (HOLDOUT, FIELD), (HOLDOUT, FIELD2), (TRAIN, FIELD),
                 (TRAIN, FIELD2), (FIELD, FIELD2)]:
        assert not a & b


def test_select_opponents_field2():
    assert set(select_opponents("field2")) == FIELD2
    assert not set(select_opponents("train")) & FIELD2
    assert set(select_opponents("all")) == set(ZOO)


def test_the_tuner_refuses_field2():
    for bad in ["field2", ["field2_solver"], ["nit", "field2_bayes"]]:
        with pytest.raises(ValueError):
            train_opponents(bad)


class _Selected(Exception):
    pass


def _capture_opponents(candidate, opponents, *args, **kwargs):
    raise _Selected(sorted(opponents))


@pytest.mark.parametrize("main,argv", [
    (sim.tournament.main, ["--candidate", "nit", "--opponents", "field2", "--deals", "1"]),
    (heuristic_bot.main, ["--opponents", "field2", "--deals", "1"]),
], ids=["tournament", "heuristic_bot"])
def test_clis_accept_opponents_field2(monkeypatch, main, argv):
    monkeypatch.setattr(sim.tournament, "focus_eval", _capture_opponents)
    with pytest.raises(_Selected) as sel:
        main(argv)
    assert sel.value.args[0] == sorted(FIELD2)


def _readme_hashes() -> dict:
    rows = re.findall(r"^\| `(field2_\w+/[\w.]+)` \| `([0-9a-f]{64})` \|",
                      (FIELD2_DIR / "README.md").read_text(), flags=re.M)
    return dict(rows)


def test_copies_are_unmodified_and_listed():
    listed = _readme_hashes()
    on_disk = {p.relative_to(FIELD2_DIR).as_posix() for p in FIELD2_DIR.glob("field2_*/*")
               if p.is_file()}
    assert set(listed) == on_disk
    for rel, digest in listed.items():
        assert hashlib.sha256((FIELD2_DIR / rel).read_bytes()).hexdigest() == digest, rel


def test_field2_bots_load_their_data_files(loaded):
    """Each bot silently falls back (weaker play or rebuilt tables) if a data file is missing."""
    s = loaded("field2_solver")
    assert s.PF_PCT and s.RIVER and s.PF_E is not None
    b = loaded("field2_bayes")
    assert np.array_equal(b._TP, np.load(FIELD2_DIR / "field2_bayes" / "tp_rank.npy"))
    a = loaded("field2_adaptive")
    assert np.array_equal(a._T, np.load(FIELD2_DIR / "field2_adaptive" / "ptab.npy"))
    assert np.array_equal(a._PF, np.load(FIELD2_DIR / "field2_adaptive" / "pfeq.npy")
                          .astype(np.float32))


@pytest.mark.parametrize("name", sorted(NAMES))
def test_field2_bot_plays_a_short_half_errorless(loaded, name):
    bot = loaded(name)
    with swallowed_exceptions(FIELD2_DIR) as swallowed:
        h = play_half(lambda: bot, ZOO["calling_station"], "g1", 1, deals=SMOKE_DEALS)
    _assert_played_its_strategy(h, swallowed)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(NAMES))
def test_field2_bot_plays_200_deals_errorless(name):
    with swallowed_exceptions(FIELD2_DIR) as swallowed:
        h = play_half(ZOO[name], ZOO["calling_station"], "g200", 1, deals=200)
    _assert_played_its_strategy(h, swallowed)
