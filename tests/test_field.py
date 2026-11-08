"""FIELD split: the agentic competitor bots, copied unmodified (sim/opponents/field/README.md)."""
import hashlib
import re

import pytest

import sim.tournament
from sim import heuristic_bot
from sim.match import REPO_ROOT, play_half
from sim.opponents import FIELD, HOLDOUT, TRAIN, ZOO
from sim.tournament import select_opponents

FIELD_DIR = REPO_ROOT / "sim" / "opponents" / "field"
SMOKE_DEALS = 10

# Each bot finds its data file next to its agent.py through __file__. If the file is missing,
# the bot silently plays a weaker fallback, so check that the data really loaded.
DATA_ATTRS = [("field_equity", "_PRE_EQ"), ("field_gto", "_CFR")]


@pytest.fixture(scope="module")
def loaded():
    """Loads each FIELD bot at most once per test module, through its ZOO factory."""
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = ZOO[name]()
        return cache[name]
    return get


def test_field_names():
    assert FIELD == {"field_equity", "field_gto", "field_exploit"}
    assert FIELD <= set(ZOO)


def test_field_is_disjoint_from_train_and_holdout():
    assert not FIELD & TRAIN
    assert not FIELD & HOLDOUT


def test_select_opponents_field_and_all():
    assert set(select_opponents("field")) == FIELD
    assert not set(select_opponents("train")) & FIELD
    assert set(select_opponents("all")) == set(ZOO)


class _Selected(Exception):
    """Raised by the stand-in focus_eval with the opponent names a CLI selected."""


def _capture_opponents(candidate, opponents, *args, **kwargs):
    raise _Selected(sorted(opponents))


@pytest.mark.parametrize("main,argv", [
    (sim.tournament.main, ["--candidate", "nit", "--opponents", "field", "--deals", "1"]),
    (heuristic_bot.main, ["--opponents", "field", "--deals", "1"]),
], ids=["tournament", "heuristic_bot"])
def test_clis_accept_opponents_field(monkeypatch, main, argv):
    monkeypatch.setattr(sim.tournament, "focus_eval", _capture_opponents)
    with pytest.raises(_Selected) as sel:
        main(argv)
    assert sel.value.args[0] == sorted(FIELD)


def _readme_hashes() -> dict:
    """{path relative to FIELD_DIR: sha256} from the README's file table."""
    rows = re.findall(r"^\| `(field_\w+/[\w.]+)` \| `([0-9a-f]{64})` \|",
                      (FIELD_DIR / "README.md").read_text(), flags=re.M)
    return dict(rows)


def test_copies_are_unmodified_and_listed():
    listed = _readme_hashes()
    on_disk = {p.relative_to(FIELD_DIR).as_posix() for p in FIELD_DIR.glob("field_*/*")
               if p.is_file()}
    assert set(listed) == on_disk
    for rel, digest in listed.items():
        assert hashlib.sha256((FIELD_DIR / rel).read_bytes()).hexdigest() == digest, rel


@pytest.mark.parametrize("name,attr", DATA_ATTRS)
def test_field_bot_loads_its_data_file(loaded, name, attr):
    assert getattr(loaded(name), attr) is not None


@pytest.mark.parametrize("name", sorted(FIELD))
def test_field_bot_plays_a_short_half_errorless(loaded, name):
    bot = loaded(name)
    h = play_half(lambda: bot, ZOO["calling_station"], "f1", 1, deals=SMOKE_DEALS)
    assert h.errors_a == 0 and not h.forfeit_a


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(FIELD))
def test_field_bot_plays_200_deals_errorless(name):
    h = play_half(ZOO[name], ZOO["calling_station"], "f200", 1, deals=200)
    assert h.errors_a == 0 and not h.forfeit_a
