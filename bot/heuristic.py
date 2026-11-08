"""Parameterized heuristic policy: Features + HeuristicParams -> a legal Action.

Preflop: raise first in at open_min (pot-sized at threebet_min), re-raise at threebet_min,
otherwise call when strength minus a per-raise range discount beats the pot odds.

Postflop, facing a bet: the opponent is modelled as betting with equity >= bet_range
(+ raise_step per further raise), and our effective equity mixes the range model
(bot.features.range_equity, weight range_weight) with the brief's linear form
eq - range_discount * opp_raises; nonnut_penalty applies on paired/flush boards without
the nuts. Raise above value_raise, call above pot odds + call_margin, else fold.
Unopened: bet pot/half above value_raise, half above thin_value (lower in position after a
check: such ranges are capped), and in position bluff with probability bluff_freq.
Never folds when free; never folds the nuts on the river (raises them when it can).

Module state is read-only; randomness comes only from the caller's numpy Generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Real

import numpy as np

from bot.features import Features, range_equity
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
    nonnut_penalty  subtracted when the board is paired or flush-possible and we lack the nuts.
                    Facing a bet on a flush-possible board it counts twice: the range
                    model's d term already marks such a hand down (range_equity), and the
                    penalty comes off the mixed equity on top (T10 review M6). Unopened it
                    is the only marker. It is a tuned parameter like the rest.

    Added:
    range_weight    facing a bet postflop, weight of the range model (equity vs an opponent
                    whose equity is >= bet_range); the rest keeps the linear discount. 0
                    gives the brief's formula.
    bet_range       assumed equity floor of the opponent's first bet/raise of a street
                    (flop, turn, river)
    raise_step      added to that floor per further raise this street
    thin_value      unopened: bet half pot above this effective equity (flop, turn, river)
    pre_discount    preflop: subtracted from our strength facing one raise; facing the
                    k-th raise of the street, pre_discount * (1 + 0.5 * (k - 1))
    ip_thin_shift   in position after the opponent checks: thin_value lowered by this

    Every value must be a finite real number, and each per-street field a sequence of
    exactly 3 (flop, turn, river); anything else raises ValueError naming the field, so a
    bad parameter set fails where it is built instead of inside the policy, where the
    layer would catch it and silently fall back. Numbers are stored as float and
    sequences as tuples of float, so instances stay hashable.
    """

    open_min: float = 0.44
    threebet_min: float = 0.53
    call_margin: float = 0.02
    range_discount: tuple = (0.10, 0.08, 0.06)
    value_raise: tuple = (0.70, 0.75, 0.80)
    pot_vs_half_cut: float = 0.85
    bluff_freq: float = 0.10
    nonnut_penalty: float = 0.08
    range_weight: float = 0.7
    bet_range: tuple = (0.50, 0.55, 0.55)
    raise_step: float = 0.10
    thin_value: tuple = (0.62, 0.64, 0.66)
    pre_discount: float = 0.07
    ip_thin_shift: float = 0.06

    def __post_init__(self) -> None:
        # Fast path: a finite float (x - x == 0 is False for inf and NaN) or a tuple of 3
        # of them is already normal, so dataclasses.replace stays cheap.
        for name in _SCALAR_FIELDS:
            v = getattr(self, name)
            if type(v) is not float or v - v != 0.0:
                object.__setattr__(self, name, _finite(name, v))
        for name in _STREET_FIELDS:
            v = getattr(self, name)
            if not (type(v) is tuple and len(v) == 3
                    and all(type(x) is float and x - x == 0.0 for x in v)):
                if isinstance(v, (str, bytes)) or not hasattr(v, "__len__") or len(v) != 3:
                    raise ValueError(f"{name} must be 3 numbers (flop, turn, river), got {v!r}")
                object.__setattr__(self, name, tuple(_finite(name, x) for x in v))


_STREET_FIELDS = ("range_discount", "value_raise", "bet_range", "thin_value")
_SCALAR_FIELDS = tuple(f.name for f in fields(HeuristicParams) if f.name not in _STREET_FIELDS)


def _finite(name: str, x) -> float:
    """x as a float; ValueError unless it is a finite real number (bool is not one)."""
    if isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x):
        raise ValueError(f"{name} must be finite real numbers, got {x!r}")
    return float(x)


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
    k = obs.raises_this_street                 # voluntary raises so far (blinds excluded)
    if obs.to_call <= 0:                       # big blind, the button limped
        if can_raise and e >= p.threebet_min:
            return HALF
        return CALL
    if k == 0:                                 # button, first in
        if can_raise and e >= p.open_min:
            return POT if e >= p.threebet_min else HALF
        return CALL if e > f.pot_odds + p.call_margin else FOLD
    if can_raise and e >= p.threebet_min:
        return POT
    # Each raise narrows the raiser's range, by less every time (measured: top 40% of
    # hands costs us ~0.065 equity, top 15% ~0.10, top 5% ~0.14).
    eff = e - p.pre_discount * (1.0 + 0.5 * (k - 1))
    return CALL if eff > f.pot_odds + p.call_margin else FOLD


def _postflop(obs: Observation, f: Features, p: HeuristicParams, rng: np.random.Generator,
              can_raise: bool) -> Action:
    i = int(f.street) - 1
    e = f.eq
    r = f.opp_raised_this_street
    nonnut = bool(f.nut & 0b110) and not (f.nut & 1)
    river_nuts = f.street == Street.RIVER and bool(f.nut & 1)
    if obs.to_call > 0:
        if river_nuts:
            return POT if can_raise else CALL  # never fold a hand that cannot lose
        t = p.bet_range[i] + p.raise_step * max(obs.raises_this_street - 1, 0)
        linear = e - p.range_discount[i] * r
        w = p.range_weight
        eff = w * range_equity(f.street, e, f.nut, t) + (1.0 - w) * linear
        if nonnut:
            eff -= p.nonnut_penalty
        if can_raise and eff > p.value_raise[i]:
            return POT if eff > p.pot_vs_half_cut else HALF
        return CALL if eff > f.pot_odds + p.call_margin else FOLD
    eff = 1.0 if river_nuts else e - (p.nonnut_penalty if nonnut else 0.0)
    if can_raise:
        if eff > p.value_raise[i]:
            return POT if eff > p.pot_vs_half_cut else HALF
        if eff > p.thin_value[i] - (p.ip_thin_shift if f.in_position else 0.0):
            return HALF
        if f.in_position and rng.random() < p.bluff_freq:
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
