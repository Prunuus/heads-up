"""Local Best Response (LBR) probe: a lower bound on how exploitable a target bot is.

LBR [Lisý & Bowling 2017] plays a fixed target and knows the target's policy, but not its
cards. Its belief over the target's 5 hole cards is a set of weighted particles: uniform
samples of the holes that avoid every card LBR can see, reweighted after each target action by
the probability that the target's policy gives that action with that hole (floored at TREMBLE,
so an action no particle explains keeps the particles that explain the most; particles under
PRUNE of the heaviest are dropped). A board card a particle holds kills it; when none is left,
a fresh set is drawn and the hand's target actions are replayed on it. At each decision LBR
plays the action with the highest immediate EV, assuming the target answers a raise by folding
with its policy's fold probability and otherwise calls (a re-raise counts as a call), and that
both players then check/call to showdown. In chips (LBR's final net):
    fold        -mine
    check/call  wp * pot - mine                                (after the call, then check down)
    raise       fp * theirs + (1 - fp) * (wp' * pot - mine)    (pot and mine after their call)
wp is P(win) + P(tie)/2 against the belief and wp' against the particles that do not fold
(fp = the belief's fold probability). Both average the same per-particle estimates, made once
per decision (exact on the river, else at least `rollouts` random board completions in all),
so the actions are compared on common samples. The engine (Game.clone/apply) supplies every
amount.

LBR's average win in bb/deal (both seats of each deal, like the judge) is the probe's number:
higher means more exploitable. It is a lower bound: LBR best-responds only locally.

Offline only: an LBR match gives both bots LBR_BANK_S as time bank and per-request timeout.
LBR is never cut short, and a time-managed target never runs short of its budget (the
heuristic's TimeManager always grants 64/128/256 rollouts by pot size), so its re-simulation
by LBR matches the target that plays, and a run is deterministic for fixed seeds.

Target contract (make_lbr(target) takes the BotFactory of the bot under test)
-----------------------------------------------------------------------------
LBR builds a private shadow of the target per half: target(), new_half with LBR's config into
a shadow memory, and on_hand_end with the target's side of every result. It then queries the
shadow for hypothetical holes; there are two kinds of target.

- BotLike target (default): deterministic re-simulation. At the start of a hand every particle
  gets a deep copy of the shadow memory, and every random.Random, numpy Generator or RandomState
  stored directly in that memory is reseeded per particle. A stochastic policy is thus
  deterministic per particle, and its randomness is averaged over the particles. The target's
  probability of an action is 1 when choose_action(obs with the particle's hole, the particle's
  memory) plays it, else 0, with the judge's handling: an exception, a non-action or an illegal
  action plays default_action. Hypothetical branches (LBR's candidate raises) run on copies, and
  the branch that happens is reused, so the same particle answers the same way. After the hand,
  a surviving particle's memory becomes the shadow memory. Requirements: per-half state lives in
  memory (instance or module state must not change decisions), and randomness comes from an rng
  stored at the top level of memory (any other randomness is still sampled per particle, but
  not reproducibly).
- Distribution target: a bot with action_probs(obs, memory) -> probabilities aligned with
  obs.legal_actions, equal to the distribution that its choose_action samples from. LBR then
  weights the particles by those probabilities exactly. action_probs must not modify memory;
  LBR calls it with the shadow memory, which only on_hand_end advances.
  DistributionTarget(probs_fn) is the adapter: probs_fn(obs, hole) -> probabilities over
  obs.legal_actions (hole == obs.hole_cards). For the blueprint, that is probs(node_of(obs),
  bucket of hole and board), masked to the legal actions. The adapter plays by sampling
  probs_fn with a numpy Generator kept in memory.

CLI (repo root; heavy runs: taskset -c 8-15 nice -n 19, --workers <= 8):
  .venv/bin/python -m sim.opponents.lbr --target heuristic --seeds 16 --deals 250 --workers 8
  --target: heuristic (sim.heuristic_bot at v1 defaults) | agent | a ZOO name | a bot file
Against the heuristic that is 4,000 deals, ~9 min on 8 free workers (about 1 s of LBR time per
deal, 0.2 s per decision).
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import multiprocessing
import os
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from bot import fasteval
from game import BIG_BLIND, Action, Deal, Game, HandResult, Observation, default_action
from judge_engine import coerce_action
from sim.match import BotFactory, HalfStats, play_half
from sim.tournament import mean_ci

# Belief size per hand. The plan's 2,000 costs 2x and moved LBR's value vs the heuristic by
# only +0.04 bb/deal [-0.72, +0.80] over 1,000 common deals; 500 lost 0.52 (docs/results/lbr.md).
PARTICLES = 1000
ROLLOUTS = 500            # min board completions per decision's win-probability estimates
LBR_BANK_S = 1e6          # time bank and per-request timeout of both bots in an LBR match
TREMBLE = 1e-6            # floor on the target's probability of an action it was seen to take
PRUNE = 1e-4              # particles lighter than this share of the heaviest are dropped
SEED = 20261107

FOLD, CALL, HALF, POT = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
_RAISES = (HALF, POT)
_DECK = np.arange(36, dtype=np.intp)
_ME, _TARGET = 0, 1       # seats in LBR's mirror of the hand


# ------------------------------------------------------------------------------ pure parts

def particle_wins(hole, board, holes: np.ndarray, w: np.ndarray, rollouts: int,
                  rng: np.random.Generator) -> np.ndarray:
    """(N,) P(win) + P(tie)/2 of `hole` against each particle with w > 0 (0 for the others).

    River: exact. Earlier: the mean over ceil(rollouts / live particles) uniform completions
    of the board (from the cards that neither hand nor the board holds) per live particle, so
    at least `rollouts` completions in all. LBR estimates this once per decision, and every
    reweighting of the belief averages the same samples.
    """
    alive = np.flatnonzero(np.asarray(w) > 0)
    if len(alive) == 0:
        raise ValueError("particle_wins needs a particle with positive weight")
    out = np.zeros(len(holes))
    mine1 = np.asarray(hole, dtype=np.intp)[None, :]
    nb = len(board)
    if nb == 5:
        b = np.asarray(board, dtype=np.intp)[None, :]
        mine = fasteval.omaha_batch(mine1, b)[0]
        theirs = fasteval.omaha_batch(holes[alive], b)
        out[alive] = (mine > theirs) + 0.5 * (mine == theirs)
        return out
    k = -(-rollouts // len(alive))
    opp = holes[np.repeat(alive, k)]
    keys = rng.random((len(opp), 36))
    keys[:, list(hole) + list(board)] = 2.0           # never drawn: above every uniform key
    np.put_along_axis(keys, opp, 2.0, axis=1)
    need = 5 - nb
    boards = np.empty((len(opp), 5), dtype=np.intp)
    boards[:, :nb] = np.asarray(board, dtype=np.intp)
    boards[:, nb:] = np.argpartition(keys, need - 1, axis=1)[:, :need]
    mine = fasteval.omaha_batch(mine1, boards)
    theirs = fasteval.omaha_batch(opp, boards)
    out[alive] = ((mine > theirs) + 0.5 * (mine == theirs)).reshape(len(alive), k).mean(axis=1)
    return out


def win_prob(hole, board, holes: np.ndarray, w: np.ndarray, rollouts: int,
             rng: np.random.Generator) -> float:
    """P(win) + P(tie)/2 of `hole` against the particle range (holes (N, 5), weights w >= 0)."""
    w = np.asarray(w, dtype=float)
    return float((w * particle_wins(hole, board, holes, w, rollouts, rng)).sum() / w.sum())


def action_values(g: Game, me: int, wp_now: float, raises: dict) -> dict:
    """LBR's EV (its final net in chips) of FOLD and CHECK_CALL, and of each raise in
    `raises` = {raise: (target fold probability, wp against the target's non-folds)}.

    g is the hand at LBR's decision, with seat `me` to act. Check/call afterwards: no more
    chips go in after LBR's call, or after the target calls LBR's raise.
    """
    legal = g.legal_actions()
    vals = {}
    if FOLD in legal:
        vals[FOLD] = -float(g.contributed[me])
    c = g.clone()
    c.apply(CALL)
    vals[CALL] = wp_now * c.pot - c.contributed[me]
    for a, (fp, wp) in raises.items():
        r = g.clone()
        r.apply(a)
        won = r.pot - r.contributed[me]               # the target folds: LBR takes the pot
        r.apply(CALL)                                 # the target calls, then check down
        vals[a] = fp * won + (1.0 - fp) * (wp * r.pot - r.contributed[me])
    return vals


def target_result(result: HandResult, lbr_hole) -> HandResult:
    """The target's side of LBR's HandResult (lbr_hole is the target's opp_hole at showdown)."""
    winner = {"me": "opp", "opp": "me"}.get(result.winner, result.winner)
    return HandResult(hand_index=result.hand_index, my_net=-result.my_net, pot=result.pot,
                      went_to_showdown=result.went_to_showdown, board=result.board,
                      opp_hole=tuple(lbr_hole) if result.went_to_showdown and lbr_hole else None,
                      winner=winner)


def _normalized(p, obs: Observation) -> np.ndarray:
    """p as probabilities over obs.legal_actions; ValueError if it is not a distribution."""
    a = np.asarray(p, dtype=float)
    if a.shape != (len(obs.legal_actions),) or not np.isfinite(a).all() or (a < 0).any() \
            or a.sum() <= 0:
        raise ValueError(f"not a distribution over {obs.legal_actions}: {p!r}")
    return a / a.sum()


# ------------------------------------------------------------------------- target models

def _with_hole(obs: Observation, hole: tuple) -> Observation:
    """obs with another hole. Same as dataclasses.replace(obs, hole_cards=hole), 15x cheaper:
    it runs once per particle per target decision."""
    o = object.__new__(Observation)
    o.__dict__.update(obs.__dict__)
    o.__dict__["hole_cards"] = hole
    return o


def _judged(bot, obs: Observation, memory: dict) -> Action:
    """The action the judge plays for bot here: its answer, or default_action when it raises
    or answers anything that is not a legal action."""
    try:
        raw = bot.choose_action(obs, memory)
    except Exception:
        return default_action(obs.to_call)
    a = coerce_action(raw) if raw is None or isinstance(raw, (int, str, float)) else None
    return a if a is not None and a in obs.legal_actions else default_action(obs.to_call)


def _reseed(memory: dict, seed: int) -> dict:
    """Reseed every rng stored directly in memory (in place); returns memory."""
    for k, v in list(memory.items()):
        if isinstance(v, np.random.Generator):
            memory[k] = np.random.default_rng(seed)
        elif isinstance(v, random.Random):
            v.seed(seed)
        elif isinstance(v, np.random.RandomState):
            v.seed(seed & 0xFFFFFFFF)
    return memory


class _Resim:
    """BotLike target: deterministic re-simulation, one copy of its memory per particle."""

    def __init__(self, bot, shadow: dict) -> None:
        self.bot = bot
        self.shadow = shadow
        self.calls = 0

    def particle_memories(self, seeds: np.ndarray) -> list:
        return [_reseed(copy.deepcopy(self.shadow), int(s)) for s in seeds]

    def respond(self, obs: Observation, hand: "_Hand", idx: np.ndarray, branch: bool):
        """(P, memories): P[i] one-hot on particle i's action for i in idx (zero rows
        elsewhere). A branch runs on copies of the particle memories; otherwise the particle
        memories advance in place."""
        P = np.zeros((len(hand.w), 4))
        mems = list(hand.mems) if branch else hand.mems
        holes = hand.hole_tuples
        for i in idx:
            m = copy.deepcopy(mems[i]) if branch else mems[i]
            P[i, _judged(self.bot, _with_hole(obs, holes[i]), m)] = 1.0
            mems[i] = m
        self.calls += len(idx)
        return P, mems

    def end_hand(self, hand: "_Hand | None", result: HandResult) -> None:
        if hand is not None and hand.mems is not None:
            self.shadow = hand.mems[int(np.argmax(hand.w))]   # a surviving particle's memory
        try:
            self.bot.on_hand_end(result, self.shadow)
        except Exception:
            pass


class _Dist:
    """Distribution target: exact probabilities from action_probs(obs, shadow memory)."""

    def __init__(self, bot, shadow: dict) -> None:
        self.bot = bot
        self.shadow = shadow
        self.calls = 0

    def particle_memories(self, seeds: np.ndarray) -> None:
        return None

    def respond(self, obs: Observation, hand: "_Hand", idx: np.ndarray, branch: bool):
        P = np.zeros((len(hand.w), 4))
        cols = [int(a) for a in obs.legal_actions]
        holes = hand.hole_tuples
        for i in idx:
            o = _with_hole(obs, holes[i])
            P[i, cols] = _normalized(self.bot.action_probs(o, self.shadow), o)
        self.calls += len(idx)
        return P, None

    def end_hand(self, hand: "_Hand | None", result: HandResult) -> None:
        try:
            self.bot.on_hand_end(result, self.shadow)
        except Exception:
            pass


class DistributionTarget:
    """BotLike adapter for a distribution target: plays by sampling probs_fn(obs, hole), a
    distribution over obs.legal_actions, and exposes it to LBR as action_probs."""

    def __init__(self, probs_fn, seed: int = SEED) -> None:
        self.probs_fn = probs_fn
        self.seed = seed

    def new_half(self, config, memory: dict) -> None:
        memory["rng"] = np.random.default_rng(self.seed)

    def action_probs(self, obs: Observation, memory: dict) -> np.ndarray:
        return _normalized(self.probs_fn(obs, obs.hole_cards), obs)

    def choose_action(self, obs: Observation, memory: dict) -> Action:
        p = self.action_probs(obs, memory)
        return obs.legal_actions[int(memory["rng"].choice(len(p), p=p))]

    def on_hand_end(self, result, memory: dict) -> None:
        return None


# ------------------------------------------------------------------------------ the bot

def _sample_holes(hole, board, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 5) sorted holes, uniform over the holes that avoid hole and board."""
    mask = np.ones(36, dtype=bool)
    mask[list(hole)] = False
    mask[list(board)] = False
    unseen = _DECK[mask]
    return np.sort(unseen[rng.random((n, len(unseen))).argsort(axis=1)[:, :5]], axis=1)


class _Hand:
    """LBR's belief during one hand."""

    def __init__(self, index: int, hole: tuple, rng: np.random.Generator) -> None:
        self.index = index
        self.hole = hole
        self.rng = rng
        self.draws = 0            # particle sets drawn this hand (1 + rebuilds)
        self.holes = self.hole_tuples = self.w = self.mems = None
        self.done = 0             # action_history entries absorbed
        self.branch_at = -1       # history index where the target answers LBR's last raise
        self.branches: dict = {}  # that raise -> (P, memories) of the target's answer

    def draw(self, board, n: int, model) -> None:
        """A fresh particle set; the hand's history is then absorbed again."""
        self.holes = _sample_holes(self.hole, board, n, self.rng)
        self.hole_tuples = [tuple(map(int, r)) for r in self.holes]
        self.w = np.ones(n)
        self.mems = model.particle_memories(self.rng.integers(0, 2 ** 62, size=n))
        self.draws += 1
        self.done, self.branch_at, self.branches = 0, -1, {}


class _Half:
    """Per-half LBR state (kept in memory["lbr"])."""

    def __init__(self, model) -> None:
        self.model = model
        self.hand: _Hand | None = None
        self.last_decision: dict | None = None
        self.stats = {"decisions": 0, "alive_sum": 0, "target_actions": 0,
                      "inconsistent": 0, "rebuilds": 0}


def _mirror_deal(obs: Observation) -> Deal:
    """LBR's hole in seat _ME, dummy cards for the target, the visible board plus filler."""
    used = set(obs.hole_cards) | set(obs.board)
    spare = [c for c in range(36) if c not in used]
    return Deal(hole_cards=(tuple(obs.hole_cards), tuple(spare[:5])),
                board=tuple(obs.board) + tuple(spare[5:10 - len(obs.board)]))


class LBRBot:
    """The LBR player (BotLike). Use make_lbr(target) for a picklable factory."""

    def __init__(self, target: BotFactory, rollouts: int = ROLLOUTS,
                 particles: int = PARTICLES, seed: int = SEED) -> None:
        self.target = target
        self.rollouts = int(rollouts)
        self.particles = int(particles)
        self.seed = int(seed)
        self.half: _Half | None = None       # the same object as memory["lbr"]

    def new_half(self, config, memory: dict) -> None:
        bot = self.target()
        shadow: dict = {}
        bot.new_half(config, shadow)
        dist = callable(getattr(bot, "action_probs", None))
        self.half = memory["lbr"] = _Half((_Dist if dist else _Resim)(bot, shadow))

    def choose_action(self, obs: Observation, memory: dict) -> Action:
        h = memory["lbr"]
        if h.hand is None or h.hand.index != obs.hand_index:
            h.hand = _Hand(obs.hand_index, tuple(obs.hole_cards),
                           np.random.default_rng([self.seed, obs.hand_index]))
            h.hand.draw(obs.board, self.particles, h.model)
        g = self._absorb(h, obs)
        return self._decide(h, g, obs)

    def on_hand_end(self, result: HandResult, memory: dict) -> None:
        h = memory["lbr"]
        hand = h.hand if h.hand is not None and h.hand.index == result.hand_index else None
        h.model.end_hand(hand, target_result(result, hand.hole if hand is not None else None))
        h.hand = None

    # -- belief ------------------------------------------------------------------------

    def _absorb(self, h: _Half, obs: Observation) -> Game:
        """Bring the belief up to obs; return LBR's mirror of the hand at obs."""
        hand = h.hand
        if obs.board:
            hand.w[np.isin(hand.holes, obs.board).any(axis=1)] = 0.0
        if not (hand.w > 0).any():
            h.stats["rebuilds"] += 1
            hand.draw(obs.board, self.particles, h.model)
        g = Game(_mirror_deal(obs), button_seat=_ME if obs.is_button else _TARGET,
                 hand_index=obs.hand_index)
        for k, rec in enumerate(obs.action_history):
            if k >= hand.done and not rec.is_me:
                self._observe(h, g.observation(_TARGET), k, rec.action, obs)
            g.apply(rec.action)
        hand.done = len(obs.action_history)
        if g.observation(_ME) != obs:
            raise RuntimeError(f"LBR's mirror of hand {obs.hand_index} diverged from the game")
        return g

    def _observe(self, h: _Half, tobs: Observation, k: int, action: Action,
                 obs: Observation) -> None:
        """Reweight by the target's probability of `action` at its observation tobs."""
        hand = h.hand
        cached = None
        if k == hand.branch_at and obs.action_history[k - 1].is_me:
            cached = hand.branches.get(obs.action_history[k - 1].action)
        if cached is not None:
            P, mems = cached
        else:
            P, mems = h.model.respond(tobs, hand, np.flatnonzero(hand.w > 0), branch=False)
        hand.mems = mems
        p = P[:, int(action)]
        h.stats["target_actions"] += 1
        if not (p[hand.w > 0] > 0).any():
            h.stats["inconsistent"] += 1
        w = hand.w * np.maximum(p, TREMBLE)
        top = w.max()
        w[w < PRUNE * top] = 0.0
        hand.w = w / top

    # -- decision ----------------------------------------------------------------------

    def _decide(self, h: _Half, g: Game, obs: Observation) -> Action:
        hand = h.hand
        w = hand.w
        alive = np.flatnonzero(w > 0)
        e = particle_wins(obs.hole_cards, obs.board, hand.holes, w, self.rollouts, hand.rng)
        wp_now = float((w * e).sum() / w.sum())
        raises, branches = {}, {}
        for a in obs.legal_actions:
            if a not in _RAISES:
                continue
            r = g.clone()
            r.apply(a)
            P, mems = h.model.respond(r.observation(_TARGET), hand, alive, branch=True)
            fold = P[:, FOLD]
            fp = float((w * fold).sum() / w.sum())
            stay = w * (1.0 - fold)
            wp = float((stay * e).sum() / stay.sum()) if stay.sum() > 0 else 0.0
            raises[a] = (fp, wp)
            branches[a] = (P, mems)
        vals = action_values(g, _ME, wp_now, raises)
        best = max(obs.legal_actions, key=lambda a: vals[a])    # the first of equal bests
        hand.branches = branches
        hand.branch_at = len(obs.action_history) + 1
        h.stats["decisions"] += 1
        h.stats["alive_sum"] += len(alive)
        h.last_decision = {"values": vals, "wp": wp_now, "alive": len(alive),
                           "fold_prob": {a: fp for a, (fp, _) in raises.items()},
                           "wp_after": {a: wp for a, (_, wp) in raises.items()}}
        return best


def make_lbr(target: BotFactory, rollouts: int = ROLLOUTS, particles: int = PARTICLES,
             seed: int = SEED) -> BotFactory:
    """Picklable factory of LBR players against `target` (see the module docstring)."""
    return functools.partial(LBRBot, target, rollouts=rollouts, particles=particles, seed=seed)


# --------------------------------------------------------------------------- evaluation

@dataclass
class LBRResult:
    """LBR's win against a target in bb/deal (both seats of every deal), with a 95% CI."""

    mean_bb: float
    ci_low: float
    ci_high: float
    deals: int                 # paired deals: each is played once in each seat
    decisions: int             # LBR decisions
    lbr_seconds: float         # LBR's thinking time (belief updates included)
    wall_seconds: float
    target_errors: int         # the target's own errors (played as default_action)
    stats: dict = field(default_factory=dict)   # belief diagnostics, summed over halves
    # bb per deal, seeds in order: for paired comparisons of two targets on the same deals
    per_deal: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    @property
    def ms_per_decision(self) -> float:
        return 1e3 * self.lbr_seconds / max(self.decisions, 1)


def lbr_half(target: BotFactory, seed, half: int, deals: int, rollouts: int = ROLLOUTS,
             particles: int = PARTICLES, lbr_seed: int = SEED) -> tuple:
    """One half of LBR (agent1) vs target with no time limits: (HalfStats, belief stats).

    Raises RuntimeError if LBR erred even once: its value would then be meaningless."""
    made: list = []

    def factory():
        made.append(LBRBot(target, rollouts, particles, lbr_seed))
        return made[-1]

    hs: HalfStats = play_half(factory, target, seed, half, deals, grace_s=0.0,
                              time_bank_s=LBR_BANK_S, action_timeout_s=LBR_BANK_S)
    if hs.errors_a or hs.forfeit_a:
        raise RuntimeError(f"LBR erred {hs.errors_a} time(s) in seed {seed} half {half}")
    stats = {}
    if made and made[0].half is not None:
        stats = dict(made[0].half.stats, target_calls=made[0].half.model.calls)
    return hs, stats


def _job(job: tuple) -> tuple:
    target, seed, half, deals, rollouts, particles, lbr_seed = job
    return (seed, half) + lbr_half(target, seed, half, deals, rollouts, particles, lbr_seed)


def lbr_eval(target: BotFactory, seeds: list, deals: int, workers: int = 1,
             rollouts: int = ROLLOUTS, particles: int = PARTICLES,
             lbr_seed: int = SEED) -> LBRResult:
    """LBR vs target over every seed (both halves each), pooled per deal like paired_diff."""
    seeds = [str(s) for s in seeds]
    jobs = [(target, s, half, deals, rollouts, particles, lbr_seed)
            for s in seeds for half in (1, 2)]
    t0 = time.perf_counter()
    if workers <= 1 or len(jobs) <= 1:
        out = [_job(j) for j in jobs]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(min(workers, len(jobs))) as pool:
            out = list(pool.imap(_job, jobs, chunksize=1))
    wall = time.perf_counter() - t0
    halves = {(s, half): (hs, st) for s, half, hs, st in out}
    per_deal = np.concatenate([
        (np.asarray(halves[(s, 1)][0].nets_a, dtype=float)
         + np.asarray(halves[(s, 2)][0].nets_a, dtype=float)) / BIG_BLIND for s in seeds])
    lo, mean, hi = mean_ci(per_deal)
    stats: dict = {}
    for _hs, st in halves.values():
        for k, v in st.items():
            stats[k] = stats.get(k, 0) + v
    hss = [hs for hs, _ in halves.values()]
    return LBRResult(mean, lo, hi, len(per_deal), stats.get("decisions", 0),
                     sum(hs.bank_a for hs in hss), wall, sum(hs.errors_b for hs in hss), stats,
                     per_deal)


# ----------------------------------------------------------------------------------- CLI

def resolve_target(spec: str) -> BotFactory:
    """'heuristic' -> sim.heuristic_bot.HeuristicBot (v1 defaults); else sim.tournament's
    resolve_bot ('agent', a ZOO name or a bot file)."""
    if spec == "heuristic":
        from sim.heuristic_bot import HeuristicBot
        return HeuristicBot
    from sim.tournament import resolve_bot
    return resolve_bot(spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sim.opponents.lbr")
    ap.add_argument("--target", required=True)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seed-prefix", default="lbr")
    ap.add_argument("--deals", type=int, default=250)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--particles", type=int, default=PARTICLES)
    ap.add_argument("--rollouts", type=int, default=ROLLOUTS)
    ap.add_argument("--out", help="write the result as JSON")
    args = ap.parse_args(argv)
    seeds = [f"{args.seed_prefix}{i}" for i in range(1, args.seeds + 1)]
    r = lbr_eval(resolve_target(args.target), seeds, args.deals, args.workers,
                 args.rollouts, args.particles)
    st = r.stats
    print(f"LBR vs {args.target}: {r.mean_bb:+.3f} bb/deal  CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
          f"  deals {r.deals}  (particles {args.particles}, rollouts {args.rollouts})")
    print(f"decisions {r.decisions}  {r.ms_per_decision:.1f} ms/decision  LBR {r.lbr_seconds:.0f} s"
          f"  wall {r.wall_seconds:.0f} s  target errors {r.target_errors}  load "
          f"{os.getloadavg()[0]:.1f}")
    if st.get("decisions"):
        print(f"belief: mean alive {st['alive_sum'] / st['decisions']:.0f}  "
              f"inconsistent {st['inconsistent']}/{st['target_actions']}  rebuilds {st['rebuilds']}"
              f"  target re-simulations {st['target_calls']}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"target": args.target, "seeds": seeds, "deals_per_seed": args.deals,
                       "particles": args.particles, "rollouts": args.rollouts,
                       **{k: getattr(r, k) for k in ("mean_bb", "ci_low", "ci_high", "deals",
                                                     "decisions", "lbr_seconds", "wall_seconds",
                                                     "target_errors", "stats")},
                       "ms_per_decision": r.ms_per_decision}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
