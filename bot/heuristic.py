"""Parameterized heuristic policy: Features + HeuristicParams -> a legal Action.

Module state is read-only; randomness comes only from the caller's numpy Generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bot.features import Features
from game import Action, Observation, Street

FOLD, CALL, HALF, POT = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT


@dataclass(frozen=True)
class HeuristicParams:
    """Tunable thresholds (equities are vs a random hand, in [0, 1]).

    open_min        preflop: raise first in (button) at or above this strength
    threebet_min    preflop: re-raise a raise at or above this strength
    call_margin     call when effective equity > pot odds + call_margin
    range_discount  per opponent raise this street, subtracted from equity (flop, turn, river)
    value_raise     raise when effective equity exceeds this (flop, turn, river)
    pot_vs_half_cut RAISE_POT above this effective equity, else RAISE_HALF
    bluff_freq      bet this often with a hand too weak to value-bet, in position when checked to
    nonnut_penalty  subtracted when the board is paired or flush-possible and we lack the nuts
    """

    open_min: float = 0.44
    threebet_min: float = 0.53
    call_margin: float = 0.02
    range_discount: tuple = (0.10, 0.08, 0.06)
    value_raise: tuple = (0.70, 0.75, 0.80)
    pot_vs_half_cut: float = 0.85
    bluff_freq: float = 0.10
    nonnut_penalty: float = 0.08


def _legal(obs: Observation, want: Action) -> Action:
    """want if legal; a raise falls back to the other raise size, then to check/call.
    Never folds when nothing is owed."""
    legal = obs.legal_actions
    if want == FOLD and obs.to_call <= 0:
        want = CALL
    if want in legal:
        return want
    if want in (HALF, POT):
        other = POT if want == HALF else HALF
        if other in legal:
            return other
    if CALL in legal:
        return CALL
    for a in legal:
        if a != FOLD or obs.to_call > 0:
            return a
    return CALL


def _preflop(obs: Observation, f: Features, p: HeuristicParams, can_raise: bool) -> Action:
    e = f.eq
    r = f.opp_raised_this_street
    if obs.to_call <= 0:                       # big blind, the button limped
        if can_raise and e >= p.threebet_min:
            return HALF
        return CALL
    if r == 0:                                 # button, first in
        if can_raise and e >= p.open_min:
            return POT if e >= p.threebet_min else HALF
        return CALL if e > f.pot_odds + p.call_margin else FOLD
    if can_raise and e >= p.threebet_min + 0.04 * (r - 1):
        return POT
    eff = e - p.range_discount[0] * r
    return CALL if eff > f.pot_odds + p.call_margin else FOLD


def _postflop(obs: Observation, f: Features, p: HeuristicParams, rng: np.random.Generator,
              can_raise: bool) -> Action:
    i = int(f.street) - 1
    eff = f.eq - p.range_discount[i] * f.opp_raised_this_street
    if (f.nut & 0b110) and not (f.nut & 1):
        eff -= p.nonnut_penalty
    if f.street == Street.RIVER and f.nut & 1:
        eff = max(eff, 1.0)                    # the nuts on the river never loses
    if can_raise and eff > p.value_raise[i]:
        return POT if eff > p.pot_vs_half_cut else HALF
    if obs.to_call > 0:
        if f.street == Street.RIVER and f.nut & 1:
            return CALL                        # never fold a hand that cannot lose
        return CALL if eff > f.pot_odds + p.call_margin else FOLD
    if can_raise and f.in_position and rng.random() < p.bluff_freq:
        return HALF
    return CALL


def heuristic_action(obs: Observation, feats: Features, params: HeuristicParams,
                     rng: np.random.Generator, opp=None) -> Action:
    """A legal action for obs. `opp` (an opponent model, Task 13) is accepted and unused."""
    legal = obs.legal_actions
    can_raise = HALF in legal or POT in legal
    if feats.street == Street.PREFLOP:
        want = _preflop(obs, feats, params, can_raise)
    else:
        want = _postflop(obs, feats, params, rng, can_raise)
    return _legal(obs, want)
