"""Decision features for the heuristic policy (and the layers built on it later).

compute_features(obs, iters, rng) turns an Observation into a frozen Features record:
- preflop: eq = now = the preflop table value (equity vs a random hand, 3,000 rollouts
  per hand class offline), nut = 0;
- postflop: eq = equity_vs_random with `iters` rollouts, nut = nut_flags, and now =
  strength_now's deterministic estimate (no rollouts). No policy reads `now`; its
  rollouts were 20-30% of flop/turn compute (T10 review M7, ablated in Task 12), so every
  rollout goes to eq. On the river eq is already our made hand's strength: now = eq.
`iters` comes from the caller's time manager. iters <= 0 means no Monte Carlo: eq uses
bot.equity's cheap deterministic estimate too (then eq == now) and rng is left untouched.

Module state is read-only; everything per-half lives with the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bot import equity, preflop
from game import Action, Observation, Street

_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

# Range model: P(we win at showdown | our equity vs random `eq`, our nut bits, and an
# opponent whose own equity vs random is >= t). Logistic per street on a grid of t:
#   logit P = a + b * logit(eq) + c * [nuts now] + d * [flush possible, not the nuts]
# Fitted offline (python -m train.fit_range_model --spots 100000, seed 20261107) on
# 100,000 random spots per street, both equities from 128 rollouts.
# Rows are t = 0.0, 0.1, ..., 0.9; t is interpolated linearly.
_RANGE_COEF = (
    # FLOP
    ((-0.01, 0.90, 0.22, 0.05), (-0.02, 0.90, 0.21, 0.05), (-0.04, 0.91, 0.21, 0.03),
     (-0.13, 0.90, 0.19, -0.17), (-0.36, 0.89, 0.12, -0.84), (-0.73, 0.91, 0.20, -1.22),
     (-1.15, 0.94, 0.42, -1.16), (-1.59, 0.97, 0.61, -1.24), (-2.05, 0.94, 0.80, -1.12),
     (-2.87, 0.92, 0.61, -0.91)),
    # TURN
    ((-0.01, 0.94, 0.21, -0.03), (-0.02, 0.94, 0.21, -0.04), (-0.10, 0.95, 0.19, -0.19),
     (-0.30, 0.98, 0.19, -0.52), (-0.67, 1.03, 0.25, -0.83), (-1.12, 1.09, 0.41, -0.89),
     (-1.58, 1.12, 0.67, -0.96), (-1.97, 1.07, 1.03, -1.14), (-2.41, 0.95, 1.45, -1.11),
     (-3.04, 0.83, 1.66, -1.30)),
    # RIVER
    ((0.01, 0.98, 0.61, 0.04), (-0.39, 1.20, -0.13, 0.04), (-0.82, 1.38, -0.62, 0.03),
     (-1.32, 1.54, -0.93, 0.04), (-1.82, 1.66, -1.12, -0.05), (-2.58, 1.80, -1.25, -0.02),
     (-3.35, 1.93, -1.20, -0.07), (-4.68, 2.15, -1.21, -0.07), (-5.84, 2.27, -0.81, -0.21),
     (-7.07, 2.07, 0.45, -0.35)),
)


def range_equity(street: int, eq: float, nut: int, t: float) -> float:
    """Equity vs an opponent range {equity vs random >= t} (postflop streets 1..3).

    t is clipped to [0, 0.9]; eq to [0.005, 0.995] before the logit.
    """
    rows = _RANGE_COEF[int(street) - 1]
    k = min(max(t, 0.0), 0.9) * 10.0
    i = min(int(k), 8)
    w = k - i
    lo, hi = rows[i], rows[i + 1]
    a, b, c, d = (lo[j] + w * (hi[j] - lo[j]) for j in range(4))
    e = min(max(eq, 0.005), 0.995)
    x = a + b * math.log(e / (1.0 - e))
    if nut & 1:
        x += max(c, 0.0)      # never below a non-nut hand of the same eq (river fit < 0: ties)
    elif nut & 4:
        x += d
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class Features:
    """What the heuristic decides from.

    street                  the betting round
    pre                     preflop strength of our hole (any street); None if unavailable
    eq                      equity vs a random hand (preflop: the table value)
    now                     strength of our made hand vs random hands on this board
                            (deterministic estimate; on the river, eq)
    nut                     bot.equity.nut_flags bits (0 preflop): 1 nuts now, 2 paired
                            board, 4 flush possible, 8 ace-flush holding
    spr                     effective stack behind after matching the bet, over the pot
                            after matching it
    pot_odds                share of the final pot we put in by calling (0 when free)
    in_position             we are the button, so we act last after the flop
    opp_raised_this_street  opponent bets/raises on this street

    Added (all default to "unknown/none" so the brief's constructor still works):
    opp_raises_before       opponent bets/raises on earlier streets of this hand
    my_raised_this_street   our own bets/raises on this street
    facing_frac             to_call / pot before the opponent's last bet (0 when free)
    iters                   rollouts used for eq (0: deterministic estimate)
    """

    street: Street
    pre: float | None
    eq: float
    now: float
    nut: int
    spr: float
    pot_odds: float
    in_position: bool
    opp_raised_this_street: int
    opp_raises_before: int = 0
    my_raised_this_street: int = 0
    facing_frac: float = 0.0
    iters: int = 0


def _preflop_value(hole) -> float | None:
    try:
        return preflop.preflop_strength(hole)
    except Exception:          # missing/corrupt table or an impossible hole: no prior
        return None


def compute_features(obs: Observation, iters: int, rng: np.random.Generator) -> Features:
    """Features for obs. Raises ValueError only for a board that is not 0, 3, 4 or 5 cards."""
    hole, board = obs.hole_cards, obs.board
    pre = _preflop_value(hole)
    nb = len(board)
    if nb == 0:
        eq = now = pre if pre is not None else 0.5
        nut = 0
        iters = 0
    elif nb in (3, 4, 5):
        iters = max(int(iters), 0)
        eq = equity.equity_vs_random(hole, board, iters, rng)
        now = eq if (iters == 0 or nb == 5) else equity.strength_now(hole, board, 0, rng)
        nut = equity.nut_flags(hole, board)
    else:
        raise ValueError(f"board must have 0, 3, 4 or 5 cards, got {nb}")

    street = obs.street
    opp_now = opp_before = mine = 0
    for a in obs.action_history:
        if a.action in _RAISES:
            if a.is_me:
                mine += a.street == street
            elif a.street == street:
                opp_now += 1
            else:
                opp_before += 1

    pot = max(obs.pot, 1)
    call = min(max(obs.to_call, 0), max(obs.my_stack, 0))
    behind = min(obs.my_stack - call, obs.opp_stack)
    spr = max(behind, 0) / (pot + call)
    pot_odds = call / (pot + call) if call > 0 else 0.0
    before_bet = pot - obs.to_call
    facing = obs.to_call / before_bet if obs.to_call > 0 and before_bet > 0 else 0.0
    return Features(street=street, pre=pre, eq=float(eq), now=float(now), nut=int(nut),
                    spr=float(spr), pot_odds=float(pot_odds), in_position=bool(obs.is_button),
                    opp_raised_this_street=opp_now, opp_raises_before=opp_before,
                    my_raised_this_street=mine, facing_frac=float(facing), iters=iters)
