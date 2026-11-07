import pickle

import pytest

from sim.match import play_half
from sim.opponents.archetypes import ZOO, checkfold_after


@pytest.mark.parametrize("name", list(ZOO))
def test_zoo_bot_is_legal_and_errorless(name):
    h = play_half(ZOO[name], ZOO["calling_station"], "z1", 1, deals=100)
    assert h.errors_a == 0 and not h.forfeit_a


def test_zoo_factories_are_picklable():
    for name, fac in ZOO.items():
        pickle.dumps(fac)
    pickle.dumps(checkfold_after(50))


def test_zoo_expected_names():
    expected = {"random", "calling_station", "always_half", "always_pot", "guide_example",
                "nit", "maniac", "mc_equity_fast", "mc_equity_slow", "threshold_cat",
                "checkfold_after_300"}
    assert expected <= set(ZOO)


def test_checkfold_after_stops_playing():
    h = play_half(checkfold_after(10), ZOO["calling_station"], "z2", 1, deals=60)
    assert h.errors_a == 0 and not h.forfeit_a


def test_nit_preflop_pass_rate_about_20_percent():
    import random
    from sim.opponents.archetypes import nit_preflop_score, NIT_THRESHOLD
    rng = random.Random(1)
    n = 20000
    ok = sum(nit_preflop_score(rng.sample(range(36), 5)) >= NIT_THRESHOLD for _ in range(n))
    assert 0.15 < ok / n < 0.25
