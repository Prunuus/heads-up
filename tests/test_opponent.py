"""Task 13: opponent model (bot/opponent.py).

Realistic traces come from the in-process simulator: our side is a bettor that feeds an
OpponentModel exactly as the real bot does (observe() in choose_action, finish() in
on_hand_end, model in memory["opp"]), and the opponent is wrapped so every decision it
really made is logged as ground truth.
"""

import ast
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot.opponent import (CF_STREAK, MANIAC_MIN_N, SHOWDOWN_MIN_N, STATION_MIN_N,
                          OpponentModel)
from evaluator import omaha_score, warmup
from game import Action, Game, HandResult, Street, make_deal
from sim.match import play_half
from sim.opponents import ZOO
from sim.opponents.archetypes import Maniac, checkfold_after

BIG_BANK = 1e9          # no bank-outs: a timed-out opponent would become a real check-folder
RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


@pytest.fixture(scope="module", autouse=True)
def _table():
    warmup()            # agent.py builds omaha_score's table at startup; so do the tests


def flags(m):
    return (m.is_check_folder(), m.is_station(), m.is_maniac())


class Watcher:
    """Our side: the zoo maniac (bets or raises 70% of the time) feeding an OpponentModel."""

    def __init__(self):
        self.inner = Maniac()
        self.model = None
        self.after_hand = {}    # hand_index -> detector flags right after finish(result)
        self.at_decision = []   # (hand_index, flags) right after observe(obs)
        self.trace = []         # ("obs", obs) / ("res", result) in call order

    def new_half(self, config, memory):
        self.inner.new_half(config, memory)
        memory["opp"] = self.model = OpponentModel()

    def choose_action(self, obs, memory):
        opp = memory["opp"]
        opp.observe(obs)
        self.trace.append(("obs", obs))
        self.at_decision.append((obs.hand_index, flags(opp)))
        return self.inner.choose_action(obs, memory)

    def on_hand_end(self, result, memory):
        opp = memory["opp"]
        opp.finish(result)
        self.trace.append(("res", result))
        self.after_hand[result.hand_index] = flags(opp)


class Truth:
    """Wraps an opponent and logs each decision it made: (hand, street, facing, could_raise, action)."""

    def __init__(self, factory):
        self.inner = factory()
        self.log = []

    def new_half(self, config, memory):
        self.inner.new_half(config, memory)

    def choose_action(self, obs, memory):
        a = Action(self.inner.choose_action(obs, memory))
        could = any(r in obs.legal_actions for r in RAISES)
        self.log.append((obs.hand_index, int(obs.street), obs.to_call > 0, could, a))
        return a

    def on_hand_end(self, result, memory):
        self.inner.on_hand_end(result, memory)


def simulate(opp_factory, seed, deals, half=1):
    me, opp = Watcher(), Truth(opp_factory)
    h = play_half(lambda: me, lambda: opp, seed, half, deals=deals, time_bank_s=BIG_BANK)
    assert h.errors_a == 0 and h.errors_b == 0
    assert me.model.errors == 0
    return me, opp


def truth_stats(log):
    """Per-street fold-to-bet and raise-rate counts, and vpip, from the opponent's own view."""
    face = [[0, 0] for _ in range(4)]     # [folds, n]
    rais = [[0, 0] for _ in range(4)]     # [raises, n] over decisions where a raise was legal
    pre_hands, vpip_hands = set(), set()
    for hand, s, facing, could, a in log:
        if facing:
            face[s][0] += a == Action.FOLD
            face[s][1] += 1
        if could:
            rais[s][0] += a in RAISES
            rais[s][1] += 1
        if s == 0:
            pre_hands.add(hand)
            if a in RAISES or (facing and a == Action.CHECK_CALL):
                vpip_hands.add(hand)
    return face, rais, (len(vpip_hands), len(pre_hands))


def assert_rate(pair, k, n):
    rate, got_n = pair
    assert got_n == n
    assert rate == pytest.approx(k / n if n else 0.0)


# ---------------------------------------------------------------------------
# Scripted hands on the real engine (we are seat 0)
# ---------------------------------------------------------------------------

def play_hand(model, hand_index, we_are_button, ours, theirs, after=None, seed="t13"):
    """Play one hand; feed our observations and our HandResult to the model.
    after(model) runs after every model call. Returns (game, opponent decisions)."""
    game = Game(make_deal(seed, hand_index), 0 if we_are_button else 1, hand_index=hand_index)
    truth = []
    while not game.is_terminal:
        seat = game.to_act
        obs = game.observation(seat)
        if seat == 0:
            model.observe(obs)
            if after:
                after(model)
            game.apply(ours(obs))
        else:
            a = theirs(obs)
            truth.append((int(obs.street), obs.to_call > 0, any(r in obs.legal_actions for r in RAISES), a))
            game.apply(a)
    model.finish(game.hand_result(0))
    if after:
        after(model)
    return game, truth


def call_any(obs):
    return Action.CHECK_CALL


def fold_to_bets(obs):
    return Action.FOLD if obs.to_call > 0 else Action.CHECK_CALL


def raise_when_legal(obs):
    return Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL


def bet_flop_once(obs):
    """Limp preflop, bet the flop when checked to, then check/call."""
    if obs.street == Street.FLOP and obs.to_call == 0 and Action.RAISE_POT in obs.legal_actions:
        return Action.RAISE_POT
    return Action.CHECK_CALL


def bet_street(street, size=Action.RAISE_POT):
    def policy(obs):
        if obs.street == street and obs.to_call == 0 and size in obs.legal_actions:
            return size
        return Action.CHECK_CALL
    return policy


# ---------------------------------------------------------------------------
# The four tests from the brief
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", ["cf1", "cf2", "cf3", "cf4"])
def test_detects_checkfolder_after_switch(seed):
    """checkfold_after(50) plays mc_equity_fast for hands 0-49, then check/fold forever."""
    me, _ = simulate(checkfold_after(50), seed, deals=200)
    first = min(h for h, f in me.after_hand.items() if f[0])
    assert 50 <= first <= 70, first
    assert not any(f[0] for h, f in me.at_decision if h <= 50)
    # Nothing ever brings the evidence back: once on, it stays on.
    assert all(f[0] for h, f in me.after_hand.items() if h >= first)
    assert not any(f[1] or f[2] for h, f in me.after_hand.items() if h >= first)


def test_station_and_maniac_detection_thresholds():
    # Station: one postflop bet faced per hand. Off at n = 59, on at n = 60.
    m = OpponentModel()
    for i in range(STATION_MIN_N):
        assert not m.is_station()
        play_hand(m, i, True, bet_flop_once, call_any)
        assert m.recent_postflop_fold_to_bet == (0.0, i + 1)
    assert m.is_station() and not m.is_maniac() and not m.is_check_folder()
    # Reverts once the recent fold rate reaches 0.05: 3/63 < 0.05 <= 4/64.
    for i in range(3):
        play_hand(m, 100 + i, True, bet_flop_once, fold_to_bets)
        assert m.is_station()
    play_hand(m, 103, True, bet_flop_once, fold_to_bets)
    assert m.fold_to_bet(Street.FLOP) == (4 / 64, 64)
    assert not m.is_station()

    # Maniac: the opponent (button) bets or raises every street, 4 decisions per hand.
    m = OpponentModel()
    seen = []
    play = lambda i, theirs: play_hand(m, i, False, call_any, theirs,
                                       after=lambda mm: seen.append((mm.recent_raise_rate, mm.is_maniac())))
    for i in range(MANIAC_MIN_N // 4):
        play(i, raise_when_legal)
    assert m.recent_raise_rate == (1.0, MANIAC_MIN_N)
    for (rate, n), on in seen:
        assert on == (n >= MANIAC_MIN_N), (n, on)
    assert not m.is_station()       # we never bet into it here, so there is no evidence
    # Reverts once the recent raise rate is <= 0.55: 80/145 > 0.55 >= 80/146.
    seen.clear()
    for i in range(20):
        play(200 + i, call_any)
    for (rate, n), on in seen:
        assert on == (rate > 0.55), (rate, n, on)
        if n <= 145:
            assert on
        if n >= 146:
            assert not on
    assert not m.is_maniac()


def test_infers_fold_to_our_bet_from_result():
    m = OpponentModel()
    # We bet the flop, the opponent folds: the result is the only evidence (F11).
    game, truth = play_hand(m, 0, True, bet_flop_once, fold_to_bets)
    assert truth[-1] == (int(Street.FLOP), True, True, Action.FOLD)
    assert not game.result().went_to_showdown
    assert m.fold_to_bet(Street.FLOP) == (1.0, 1)
    assert m.raise_rate(Street.FLOP) == (0.0, 2)          # its check, then its fold
    # We are the big blind and the opponent folds its button: we never act at all.
    game = Game(make_deal("t13", 1), 1, hand_index=1)
    game.apply(Action.FOLD)
    res = game.hand_result(0)
    assert res.pot == 3 and res.my_net == 1 and res.winner == "me"
    m.finish(res)
    assert m.fold_to_bet(Street.PREFLOP) == (1.0, 1)
    assert m.vpip == (0.0, 2)
    assert m.fold_streak == 2
    # We fold: nothing about the opponent to infer.
    game, _ = play_hand(m, 2, True, fold_to_bets, raise_when_legal)
    assert m.fold_to_bet(Street.PREFLOP) == (1.0, 1)
    # We bet the river and it calls: the call is seen only at showdown, and it ends the streak.
    game, truth = play_hand(m, 3, True, bet_street(Street.RIVER), call_any)
    assert truth[-1] == (int(Street.RIVER), True, True, Action.CHECK_CALL)
    assert m.fold_to_bet(Street.RIVER) == (0.0, 1)
    assert m.fold_streak == 0
    # We check the river first and it checks behind: a decision, but not facing a bet.
    game, truth = play_hand(m, 4, False, call_any, call_any)
    assert truth[-1] == (int(Street.RIVER), False, True, Action.CHECK_CALL)
    assert m.raise_rate(Street.RIVER) == (0.0, 3)
    assert m.fold_to_bet(Street.RIVER) == (0.0, 1)
    # Same hand index again (a duplicate result) changes nothing.
    before = (m.fold_to_bet(Street.RIVER), m.raise_rate(Street.RIVER), m.vpip, m.decisions)
    m.finish(game.hand_result(0))
    assert (m.fold_to_bet(Street.RIVER), m.raise_rate(Street.RIVER), m.vpip, m.decisions) == before
    assert m.errors == 0


def _results_batch():
    """20 results in one delivery, as the judge sends them at the end of a half."""
    good = []
    m = OpponentModel()
    for i in range(6):                              # real results of hands we played
        game, _ = play_hand(m, i, i % 2 == 0, bet_flop_once, call_any)
        good.append(game.hand_result(0))
    bb_walk = HandResult(6, 1, 3, False, (), None, "me")           # never acted: button folded
    split = HandResult(7, 0, 40, True, tuple(range(10, 15)), (0, 1, 2, 3, 4), "split")
    no_hole = HandResult(8, 20, 40, True, tuple(range(10, 15)), None, "me")
    bad_cards = HandResult(9, -20, 40, True, (1, 2, 3), (99, 98, 97, 96, 95), "opp")
    dupe_cards = HandResult(10, 20, 40, True, (5, 5, 5, 5, 5), (5, 5, 5, 5, 5), "me")
    weird = HandResult("11", "x", None, "yes", None, "nope", "nobody")
    missing = SimpleNamespace(hand_index=12)                       # no other fields
    return [good[3], bb_walk, good[0], split, None, good[0],        # out of order, duplicate
            no_hole, "garbage", bad_cards, 42, good[5], dupe_cards, weird, missing,
            good[1], HandResult(-1, 0, 0, False, (), None, "split"), object(),
            HandResult(10**12, 10**9, -5, True, (), (), "me"), good[2], good[4]]


def test_hand_end_batch_never_raises():
    """Review Focus 4: a batch of 20 results, odd and malformed ones included."""
    batch = _results_batch()
    assert len(batch) == 20
    m = OpponentModel()
    memory = {"opp": m}
    for r in batch:                                 # what on_hand_end does with each result
        memory["opp"].finish(r)
    # Odd observations never raise either.
    for junk in (None, "obs", 7, SimpleNamespace(hand_index=3, action_history=None),
                 SimpleNamespace(hand_index=[], action_history=())):
        m.observe(junk)
    # ... and the model still works afterwards.
    game, _ = play_hand(m, 500, True, bet_flop_once, fold_to_bets)
    assert m.fold_to_bet(Street.FLOP)[1] >= 1
    for f in (m.is_check_folder, m.is_station, m.is_maniac):
        assert f() in (True, False)
    for s in Street:
        m.fold_to_bet(s), m.raise_rate(s)
    for junk in (None, "river", 9, (Street.RIVER,), (1, 2, 3)):
        assert m.fold_to_bet(junk) == (0.0, 0)
        assert m.raise_rate(junk) == (0.0, 0)
        assert m.showdown_min_category(junk) is None


# ---------------------------------------------------------------------------
# Accuracy against the engine's ground truth, and reversibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["calling_station", "maniac", "nit", "mc_equity_fast",
                                  "random", "always_pot", "guide_example"])
def test_stats_match_engine_ground_truth(name):
    """Every opponent decision is recovered, from the history or inferred from the result."""
    me, opp = simulate(ZOO[name], "gt-" + name, deals=150)
    face, rais, (vk, vn) = truth_stats(opp.log)
    for s in Street:
        assert_rate(me.model.fold_to_bet(s), *face[s])
        assert_rate(me.model.raise_rate(s), *rais[s])
    assert_rate(me.model.vpip, vk, vn)
    assert me.model.decisions == len(opp.log)


def test_checkfolder_reverts_when_it_calls_or_bets():
    m = OpponentModel()
    for i in range(CF_STREAK):
        assert not m.is_check_folder()
        play_hand(m, i, True, bet_flop_once, fold_to_bets)
    assert m.is_check_folder()
    play_hand(m, 50, True, bet_flop_once, call_any)       # calls once: evidence gone
    assert not m.is_check_folder() and m.fold_streak == 0
    for i in range(CF_STREAK):
        play_hand(m, 60 + i, True, bet_flop_once, fold_to_bets)
    assert m.is_check_folder()
    # Its only decision is a bet without facing one (we limp, it raises its big blind
    # option, we fold): a check-folder never bets.
    limp_then_fold = lambda o: Action.CHECK_CALL if o.to_call <= 1 else Action.FOLD
    _, truth = play_hand(m, 80, True, limp_then_fold, raise_when_legal)
    assert truth == [(int(Street.PREFLOP), False, True, Action.RAISE_POT)]
    assert not m.is_check_folder() and m.fold_streak == 0


def _assert_no_station_or_maniac(name, seed):
    me, _ = simulate(ZOO[name], seed, deals=1000)
    assert len(me.after_hand) == 1000
    assert not any(f[1] or f[2] for f in me.after_hand.values()), name
    assert not any(f[1] or f[2] for _, f in me.at_decision), name
    return me


def test_no_false_positives_vs_nit_over_a_full_half():
    """Station and maniac never fire against the nit in a 1000-deal half."""
    _assert_no_station_or_maniac("nit", "fp-nit")


@pytest.mark.slow
@pytest.mark.parametrize("seed", ["fp-mc1", "fp-mc2"])
def test_no_false_positives_vs_mc_equity_over_a_full_half(seed):
    """mc_equity_fast (~15 s per half): no station, no maniac, and no check-folder either."""
    me = _assert_no_station_or_maniac("mc_equity_fast", seed)
    assert not any(f[0] for f in me.after_hand.values())


def test_station_and_maniac_found_in_real_traces():
    me, _ = simulate(ZOO["calling_station"], "st", deals=300)
    first = min(h for h, f in me.after_hand.items() if f[1])
    assert first <= 60
    assert all(f[1] for h, f in me.after_hand.items() if h >= first)
    assert not any(f[0] or f[2] for f in me.after_hand.values())
    me, _ = simulate(ZOO["maniac"], "mn", deals=300)
    first = min(h for h, f in me.after_hand.items() if f[2])
    assert first <= 60
    assert all(f[2] for h, f in me.after_hand.items() if h >= first)
    assert not any(f[0] for f in me.after_hand.values())


def test_showdown_min_category_by_action_class():
    m = OpponentModel()
    cats = {}
    i = 0
    while min((len(v) for v in cats.values()), default=0) < SHOWDOWN_MIN_N or len(cats) < 2:
        # Opponent bets the river when checked to on even hands, checks behind on odd ones.
        theirs = raise_when_legal if i % 2 == 0 else call_any
        game, _ = play_hand(m, i, False, call_any, lambda o, t=theirs: t(o) if o.street == Street.RIVER else Action.CHECK_CALL)
        res = game.hand_result(0)
        assert res.went_to_showdown
        cats.setdefault(i % 2 == 0, []).append(omaha_score(res.opp_hole, res.board) >> 20)
        i += 1
    for aggressive, got in cats.items():
        cls = (Street.RIVER, aggressive)
        assert m.showdown_count(cls) == len(got)
        assert m.showdown_min_category(cls) == min(got)
        assert m.showdown_min_category(cls, min_n=len(got) + 1) is None
    assert m.showdown_min_category((Street.RIVER, True), min_n=1) == min(cats[True])
    assert m.showdown_count((Street.TURN, True)) == 0
    assert m.showdown_min_category((Street.TURN, True)) is None
    # The class of the opponent's play on the current street, as Task 14 will look it up.
    game = Game(make_deal("cls", 0), 1, hand_index=0)
    game.apply(Action.RAISE_POT)                       # opponent (button) raises preflop
    assert OpponentModel.current_class(game.observation(0)) == (int(Street.PREFLOP), True)
    assert OpponentModel.current_class(None) is None


def test_cost_per_call_is_small():
    """observe() and finish() each cost < 50 us typical (median over a real trace)."""
    me, _ = simulate(ZOO["random"], "cost", deals=400)
    obs_t, res_t = [], []
    for _ in range(3):
        m = OpponentModel()
        clock = time.perf_counter
        for kind, x in me.trace:
            if kind == "obs":
                t0 = clock(); m.observe(x); obs_t.append(clock() - t0)
            else:
                t0 = clock(); m.finish(x); res_t.append(clock() - t0)
        assert m.errors == 0
    assert statistics.median(obs_t) < 50e-6
    assert statistics.median(res_t) < 50e-6


def test_runtime_imports_are_pure():
    tree = ast.parse(Path("bot/opponent.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "collections", "game", "evaluator", "math", "typing"}
