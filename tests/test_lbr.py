"""Local Best Response probe (sim/opponents/lbr.py)."""

import dataclasses
import functools
import pickle

import numpy as np
import pytest

import sim.opponents.lbr as lbr
from evaluator import omaha_score
from game import Action, AgentConfig, Game, HandResult, make_deal
from sim.heuristic_bot import HeuristicBot
from sim.match import play_half
from sim.opponents import ZOO
from sim.opponents.lbr import (DistributionTarget, LBRBot, ModelCheckError, action_values,
                               lbr_eval, make_lbr, model_problems, particle_wins,
                               target_result, win_prob)

FOLD, CALL, HALF, POT = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT


# --------------------------------------------------------------------------- targets

class FoldToBets:
    """Checks when free, folds to any bet or raise."""

    def new_half(self, config, memory):
        pass

    def choose_action(self, obs, memory):
        return CALL if obs.to_call == 0 else FOLD

    def on_hand_end(self, result, memory):
        pass


def _has_ace(hole) -> bool:
    return any(c >> 2 == 8 for c in hole)


class AceRaiser:
    """Preflop: pot-raises with an ace, else checks/calls. Postflop: checks/calls."""

    def new_half(self, config, memory):
        pass

    def choose_action(self, obs, memory):
        if not obs.board and _has_ace(obs.hole_cards) and POT in obs.legal_actions:
            return POT
        return CALL

    def on_hand_end(self, result, memory):
        pass


class CoinFolder:
    """Facing a bet: folds with probability 1/2 (rng in memory), else calls."""

    def new_half(self, config, memory):
        memory["rng"] = np.random.default_rng(7)

    def choose_action(self, obs, memory):
        if obs.to_call > 0 and memory["rng"].random() < 0.5:
            return FOLD
        return CALL

    def on_hand_end(self, result, memory):
        pass


def ace_probs(obs, hole):
    """Pot-raise with probability 0.9 holding an ace, 0.1 without; else check/call."""
    p = (0.9 if _has_ace(hole) else 0.1) if POT in obs.legal_actions else 0.0
    return [p if a == POT else 1.0 - p if a == CALL else 0.0 for a in obs.legal_actions]


def _lbr_facing(target_cls, deal_seed="lbr-t", particles=400, **kw):
    """LBR in the big blind (seat 1) after the target (button, seat 0) acted once preflop.

    Returns (LBR bot, its memory, the game)."""
    bot = LBRBot(target_cls, particles=particles, **kw)
    memory = {}
    bot.new_half(AgentConfig(), memory)
    g = Game(make_deal(deal_seed, 0), button_seat=0, hand_index=0)
    tgt = target_cls()
    tmem = {}
    tgt.new_half(AgentConfig(), tmem)
    g.apply(tgt.choose_action(g.observation(0), tmem))
    return bot, memory, g


def _belief(memory):
    hand = memory["lbr"].hand
    alive = hand.w > 0
    return [hand.hole_tuples[i] for i in np.flatnonzero(alive)], hand.w[alive]


# --------------------------------------------------------------------------- pure parts

def test_win_prob_on_the_river_is_exact_against_weighted_particles():
    deal = make_deal("wp", 3)
    hole, board = deal.hole_cards[0], deal.board
    rng = np.random.default_rng(1)
    unseen = [c for c in range(36) if c not in hole and c not in board]
    holes = np.array([sorted(rng.choice(unseen, 5, replace=False)) for _ in range(40)])
    w = rng.random(40)
    w[:5] = 0.0
    mine = omaha_score(hole, board)
    s = np.array([1.0 if mine > omaha_score(h, board) else 0.5 if mine == omaha_score(h, board)
                  else 0.0 for h in holes])
    assert win_prob(hole, board, holes, w, 10, rng) == pytest.approx((w * s).sum() / w.sum())


def test_win_prob_on_the_turn_matches_exact_enumeration():
    deal = make_deal("wp", 4)
    hole, board = deal.hole_cards[0], deal.board[:4]
    opp = deal.hole_cards[1]
    holes = np.array([opp])
    rivers = [c for c in range(36) if c not in hole and c not in board and c not in opp]
    exact = np.mean([1.0 if omaha_score(hole, board + (r,)) > omaha_score(opp, board + (r,))
                     else 0.5 if omaha_score(hole, board + (r,)) == omaha_score(opp, board + (r,))
                     else 0.0 for r in rivers])
    est = win_prob(hole, board, holes, np.ones(1), 20000, np.random.default_rng(2))
    assert est == pytest.approx(exact, abs=0.015)


def test_particle_wins_share_samples_across_weightings():
    """One estimate per particle per decision: any reweighting of the belief (the non-folders
    of each raise) averages the same samples, so a raise nobody folds to gets exactly the
    call's win probability instead of an independently noisy one."""
    deal = make_deal("wp", 5)
    hole = deal.hole_cards[0]
    rng = np.random.default_rng(3)
    unseen = [c for c in range(36) if c not in hole]
    holes = np.array([sorted(rng.choice(unseen, 5, replace=False)) for _ in range(300)])
    w = np.ones(300)
    w[::7] = 0.0
    e = particle_wins(hole, (), holes, w, 500, rng)
    assert e.shape == (300,) and (e[w == 0] == 0).all()
    assert ((e >= 0) & (e <= 1)).all()
    wp = (w * e).sum() / w.sum()
    assert wp == pytest.approx(0.5, abs=0.06)          # preflop vs random hands: near 1/2
    # each of the 257 live particles got ceil(500 / 257) = 2 board completions
    assert set(np.unique(e[w > 0] * 4)) <= {0, 1, 2, 3, 4}


def test_action_values_follow_the_check_call_afterwards_rule():
    g = Game(make_deal("ev", 0), button_seat=0)
    g.apply(CALL)        # button limps: 2 - 2, the big blind (seat 1) to act
    me = 1
    vals = action_values(g, me, wp_now=0.6, raises={HALF: (0.25, 0.7), POT: (0.5, 0.8)})
    assert FOLD not in vals                              # nothing owed: no fold
    assert vals[CALL] == pytest.approx(0.6 * 4 - 2)      # check down a 4-chip pot
    g2 = g.clone()
    g2.apply(HALF)
    x = g2.contributed[me]                               # raise to 4: the target calls to 4
    assert x == 4
    assert vals[HALF] == pytest.approx(0.25 * 2 + 0.75 * (0.7 * 2 * x - x))
    g3 = g.clone()
    g3.apply(POT)
    y = g3.contributed[me]
    assert y == 6
    assert vals[POT] == pytest.approx(0.5 * 2 + 0.5 * (0.8 * 2 * y - y))


def test_action_values_facing_a_bet():
    g = Game(make_deal("ev", 1), button_seat=0)
    g.apply(POT)         # button raises to 6, the big blind (seat 1) has 2 in
    vals = action_values(g, 1, wp_now=0.3, raises={})
    assert vals[FOLD] == -2
    assert vals[CALL] == pytest.approx(0.3 * 12 - 6)


def test_target_result_is_the_mirror_of_lbr_result():
    r = HandResult(hand_index=7, my_net=12, pot=24, went_to_showdown=True,
                   board=(1, 2, 3, 4, 5), opp_hole=(6, 7, 8, 9, 10), winner="me")
    t = target_result(r, lbr_hole=(11, 12, 13, 14, 15))
    assert (t.my_net, t.winner, t.opp_hole, t.pot, t.board) == \
        (-12, "opp", (11, 12, 13, 14, 15), 24, (1, 2, 3, 4, 5))
    r2 = HandResult(7, -1, 3, False, (), None, "opp")
    assert target_result(r2, lbr_hole=None) == HandResult(7, 1, 3, False, (), None, "me")
    r3 = HandResult(7, 0, 8, True, (1, 2, 3, 4, 5), (6, 7, 8, 9, 10), "split")
    assert target_result(r3, (11, 12, 13, 14, 15)).winner == "split"


# --------------------------------------------------------------------------- beliefs

def test_belief_keeps_only_hands_consistent_with_a_deterministic_target():
    for seed in ("b1", "b2", "b3"):
        bot, memory, g = _lbr_facing(AceRaiser, deal_seed=seed)
        raised = g.history[-1].action == POT
        obs = g.observation(1)
        assert bot.choose_action(obs, memory) in obs.legal_actions
        holes, w = _belief(memory)
        assert holes and all(_has_ace(h) == raised for h in holes)
        assert all(not set(h) & set(obs.hole_cards) for h in holes)


def test_belief_uses_exact_probabilities_for_a_distribution_target():
    factory = functools.partial(DistributionTarget, ace_probs)
    bot, memory, g = _lbr_facing(factory, deal_seed="b1", particles=3000)
    assert g.history[-1].action in (POT, CALL)
    obs = g.observation(1)
    bot.choose_action(obs, memory)
    hand = memory["lbr"].hand
    aces = np.array([_has_ace(h) for h in hand.hole_tuples])
    expect = 0.9 if g.history[-1].action == POT else 0.1
    ratio = hand.w[aces].mean() / hand.w[~aces].mean()
    assert ratio == pytest.approx(expect / (1 - expect), rel=1e-9)


def _lbr_first(target_cls, deal_seed, hand_index=0, particles=300):
    """LBR on the button (seat 0), first to act preflop. Returns (bot, memory, game)."""
    bot = LBRBot(target_cls, particles=particles)
    memory = {}
    bot.new_half(AgentConfig(), memory)
    return bot, memory, Game(make_deal(deal_seed, hand_index), button_seat=0,
                             hand_index=hand_index)


def test_fold_probability_of_a_stochastic_target_is_averaged_over_particles():
    bot, memory, g = _lbr_first(CoinFolder, "fp", particles=1000)
    bot.choose_action(g.observation(0), memory)
    d = memory["lbr"].last_decision
    for a in (HALF, POT):
        assert d["fold_prob"][a] == pytest.approx(0.5, abs=0.06)


def test_a_raise_nobody_folds_to_is_valued_with_the_calls_win_probability():
    bot, memory, g = _lbr_first(ZOO["calling_station"], "cs", particles=300)
    bot.choose_action(g.observation(0), memory)
    d = memory["lbr"].last_decision
    assert d["fold_prob"] == {HALF: 0.0, POT: 0.0}
    assert d["wp_after"] == {HALF: d["wp"], POT: d["wp"]}


def test_branch_responses_are_reused_when_the_target_answers():
    """The target's actual answer to LBR's raise keeps exactly the particles that were
    predicted to give it (same observation, same per-particle memory and rng)."""
    for seed in range(20):
        bot, memory, g = _lbr_first(CoinFolder, "br", hand_index=seed)
        a = bot.choose_action(g.observation(0), memory)
        if a not in (HALF, POT):
            continue
        hand = memory["lbr"].hand
        callers = np.flatnonzero((hand.w > 0) & (hand.branches[a][0][:, FOLD] == 0))
        g.apply(a)
        g.apply(CALL)                       # the target calls the raise: flop, LBR acts last
        g.apply(CALL)                       # the target checks the flop
        obs = g.observation(0)
        flop = set(obs.board)
        expect = sum(1 for i in callers if not flop & set(hand.hole_tuples[i]))
        bot.choose_action(obs, memory)
        assert memory["lbr"].hand is hand
        assert int((hand.w > 0).sum()) == expect
        return
    pytest.fail("LBR never raised first in")


def test_belief_is_redrawn_and_replayed_when_the_board_kills_every_particle():
    bot, memory, g = _lbr_first(AceRaiser, "rb", particles=50)
    bot.choose_action(g.observation(0), memory)        # LBR's belief for this hand
    hand = memory["lbr"].hand
    flop = g.deal.board[:3]
    hand.holes[:, 0] = flop[0]                         # every particle now holds a flop card
    tgt = AceRaiser()
    g.apply(CALL)                                      # LBR limps (whatever it chose)
    target_raised = tgt.choose_action(g.observation(1), {}) == POT
    g.apply(POT if target_raised else CALL)
    if target_raised:
        g.apply(CALL)                                  # LBR calls: the flop
    g.apply(tgt.choose_action(g.observation(1), {}))   # the target checks the flop
    obs = g.observation(0)
    assert obs.board == flop
    bot.choose_action(obs, memory)
    h = memory["lbr"]
    assert h.stats["rebuilds"] == 1 and h.hand is hand and hand.draws == 2
    alive = np.flatnonzero(hand.w > 0)
    assert len(alive) > 0 and hand.done == len(obs.action_history)
    for i in alive:              # fresh holes avoid every known card and agree with the
        hole = hand.hole_tuples[i]                     # target's replayed preflop action
        assert not set(hole) & (set(flop) | set(obs.hole_cards))
        assert _has_ace(hole) == target_raised


class _Stub:
    """A target model whose answers the test sets directly."""

    def __init__(self, P):
        self.P, self.calls = P, 0

    def respond(self, obs, hand, idx, branch):
        return self.P, None


def test_an_action_no_particle_explains_keeps_the_belief():
    bot, memory, g = _lbr_first(ZOO["calling_station"], "tr", particles=40)
    obs = g.observation(0)
    bot.choose_action(obs, memory)
    h = memory["lbr"]
    hand, n = h.hand, len(h.hand.w)
    hand.branch_at = -1
    P = np.zeros((n, 4))
    P[:, FOLD] = 1.0                                   # every particle folds; the target called
    h.model = _Stub(P)
    bot._observe(h, obs, 1, CALL, obs)
    assert h.stats["inconsistent"] == 1 and (hand.w > 0).sum() == n   # floored, not emptied
    P2 = np.zeros((n, 4))
    P2[: n // 2, CALL] = 1.0                           # half explain the next action
    P2[n // 2:, FOLD] = 1.0
    h.model = _Stub(P2)
    bot._observe(h, obs, 2, CALL, obs)
    assert (hand.w > 0).sum() == n // 2 and (hand.w[: n // 2] > 0).all()


# --------------------------------------------------------------------------- play

def test_lbr_raises_every_hand_against_a_target_that_always_folds_to_bets():
    # LBR's button raise wins the 2-chip big blind; as big blind it wins the 1-chip small
    # blind when the target folds first: 3 chips per paired deal = 1.5 bb/deal exactly.
    r = lbr_eval(FoldToBets, seeds=["fb"], deals=20, particles=50)
    assert r.mean_bb == pytest.approx(1.5) and r.ci_low == pytest.approx(1.5)
    assert r.deals == 20


def test_lbr_plays_the_heuristic_without_errors():
    h = play_half(make_lbr(HeuristicBot, particles=40, rollouts=100), HeuristicBot,
                  "lbr-h", 1, deals=6, time_bank_s=1e6, action_timeout_s=1e6)
    assert h.errors_a == 0 and h.errors_b == 0 and not h.forfeit_a


def test_lbr_eval_raises_when_lbr_errors(monkeypatch):
    def boom(self, obs, memory):
        raise RuntimeError("lbr bug")

    monkeypatch.setattr(LBRBot, "choose_action", boom)
    with pytest.raises(RuntimeError, match="LBR"):
        lbr_eval(ZOO["calling_station"], seeds=["e"], deals=5, particles=10)


def test_factories_are_picklable():
    for fac in (make_lbr(ZOO["always_pot"]), make_lbr(HeuristicBot, rollouts=100),
                make_lbr(functools.partial(DistributionTarget, ace_probs)),
                functools.partial(DistributionTarget, ace_probs)):
        pickle.loads(pickle.dumps(fac))


def test_distribution_target_plays_legally_by_sampling():
    h = play_half(functools.partial(DistributionTarget, ace_probs), ZOO["random"], "dt", 1,
                  deals=50)
    assert h.errors_a == 0 and h.actions_a.get("RAISE_POT", 0) > 0


# --------------------------------------------------------------------------- fail closed

_FC_DEALS = 12
_DEALT = {h for seed in ("fc", "fc1") for d in range(_FC_DEALS)       # "fc1": the CLI's seed
          for h in make_deal(seed, d).hole_cards}


class CrashesOffDeal:
    """Checks/calls with the holes the deal gives it and raises for any other hole. Only LBR's
    re-simulation asks about other holes, so the real target never errs."""

    def new_half(self, config, memory):
        pass

    def choose_action(self, obs, memory):
        if tuple(obs.hole_cards) not in _DEALT:
            raise RuntimeError("not my cards")
        return CALL

    def on_hand_end(self, result, memory):
        pass


class RaisesOnDeal(CrashesOffDeal):
    """Pot-raises with the holes the deal gives it, checks/calls with any other: it never
    errs, but LBR's re-simulation never predicts the real target's raises."""

    def choose_action(self, obs, memory):
        if tuple(obs.hole_cards) in _DEALT and POT in obs.legal_actions:
            return POT
        return CALL


def test_lbr_eval_fails_closed_when_the_re_simulated_target_raises():
    with pytest.raises(ModelCheckError, match="re-simulated") as caught:
        lbr_eval(CrashesOffDeal, seeds=["fc"], deals=_FC_DEALS, particles=30)
    r = caught.value.result
    assert r.target_errors == 0 and r.stats["model_errors"] > 0
    again = pickle.loads(pickle.dumps(caught.value))       # survives a worker boundary
    assert again.problems == caught.value.problems and again.result.deals == r.deals
    r2 = lbr_eval(CrashesOffDeal, seeds=["fc"], deals=_FC_DEALS, particles=30,
                  check_model=False)                       # the explicit opt-out
    assert r2.model_problems and r2.stats["model_errors"] == r.stats["model_errors"]


def test_lbr_eval_fails_closed_when_re_simulation_disagrees_with_the_target():
    with pytest.raises(ModelCheckError, match="no particle") as caught:
        lbr_eval(RaisesOnDeal, seeds=["fc"], deals=_FC_DEALS, particles=30)
    st = caught.value.result.stats
    assert st["model_errors"] == 0 and st["inconsistent"] > 0.02 * st["target_actions"] + 2


def test_cli_exits_non_zero_when_the_model_check_fails(monkeypatch, capsys):
    monkeypatch.setattr(lbr, "resolve_target", lambda spec: CrashesOffDeal)
    args = ["--target", "x", "--seed-prefix", "fc", "--seeds", "1", "--deals", str(_FC_DEALS),
            "--particles", "30"]
    assert lbr.main(args) == 2
    assert "MODEL PROBLEM" in capsys.readouterr().out
    assert lbr.main(args + ["--no-model-check"]) == 0


def test_model_problems_rules():
    ok = {"model_errors": 0, "shadow_errors": 0, "inconsistent": 1, "target_actions": 1000}
    assert model_problems(ok, target_errors=0, max_inconsistent=0.02) == []
    assert len(model_problems(dict(ok, model_errors=1), 0, 0.02)) == 1
    assert model_problems(dict(ok, model_errors=1), 3, 0.02) == []      # the target errs too
    assert len(model_problems(dict(ok, shadow_errors=2), 0, 0.02)) == 1
    assert len(model_problems(dict(ok, inconsistent=40), 0, 0.02)) == 1  # 4% > 2%
    assert model_problems(dict(ok, inconsistent=40), 0, None) == []     # no rate limit
    assert model_problems(dict(ok, inconsistent=2, target_actions=10), 0, 0.02) == []  # noise


class _HandEndCrash(FoldToBets):
    def on_hand_end(self, result, memory):
        raise RuntimeError("boom")


def test_shadow_hand_end_errors_are_counted():
    bot, memory = LBRBot(_HandEndCrash, particles=10), {}
    bot.new_half(AgentConfig(), memory)
    bot.on_hand_end(HandResult(0, 1, 3, False, (), None, "me"), memory)
    assert memory["lbr"].model.hand_end_errors == 1


class _ClaimsProbs(FoldToBets):
    """Has an action_probs method but is not the adapter, so nothing guarantees that its
    choose_action samples from it: LBR must re-simulate it."""

    def action_probs(self, obs, memory):
        return [1.0 / len(obs.legal_actions)] * len(obs.legal_actions)


def test_only_the_adapter_is_weighted_by_its_probabilities():
    for factory, model in ((_ClaimsProbs, lbr._Resim),
                           (functools.partial(DistributionTarget, ace_probs), lbr._Dist)):
        bot, memory = LBRBot(factory, particles=10), {}
        bot.new_half(AgentConfig(), memory)
        assert type(memory["lbr"].model) is model


def test_with_hole_is_dataclasses_replace():
    obs = Game(make_deal("wh", 0), 0).observation(0)
    hole = (0, 1, 2, 3, 4)
    assert lbr._with_hole(obs, hole) == dataclasses.replace(obs, hole_cards=hole)
    assert obs.hole_cards != hole


@pytest.mark.slow
def test_lbr_beats_always_pot_and_calling_station():
    for name in ("always_pot", "calling_station"):
        r = lbr_eval(ZOO[name], seeds=["lbr1", "lbr2"], deals=200)
        assert r.ci_low > 0, (name, r)
