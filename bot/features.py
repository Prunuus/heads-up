"""Decision features for the heuristic policy (and the layers built on it later).

compute_features(obs, iters, rng) turns an Observation into a frozen Features record:
- preflop: eq = now = the preflop table value (equity vs a random hand, 3,000 rollouts
  per hand class offline), nut = 0;
- postflop: eq = equity_vs_random and now = strength_now with `iters` rollouts each (on
  the river they are the same quantity, so it is computed once), nut = nut_flags.
`iters` comes from the caller's time manager. iters <= 0 means no Monte Carlo: both use
bot.equity's cheap deterministic estimate (then eq == now) and rng is left untouched.

Module state is read-only; everything per-half lives with the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bot import equity, preflop
from game import Action, Observation, Street

_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


@dataclass(frozen=True)
class Features:
    """What the heuristic decides from.

    street                  the betting round
    pre                     preflop strength of our hole (any street); None if unavailable
    eq                      equity vs a random hand (preflop: the table value)
    now                     strength of our made hand vs random hands on this board
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
    iters                   rollouts used for eq and now (0: deterministic estimate)
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
        now = eq if (iters == 0 or nb == 5) else equity.strength_now(hole, board, iters, rng)
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
