"""Time manager: count charged thinking time and size Monte Carlo per decision.

The judge charges every HTTP request max(0, elapsed - 1 ms) against an 8 s bank per
1,000-hand half. An empty bank means check/fold for the rest of the half. A request
delivers any finished hands (on_hand_end, once per hand) and then asks for one action.
The bot cannot see request boundaries, so it errs high:

- begin()/end() wrap choose_action and charge max(0, dt - GRACE_S). GRACE_S leaves
  0.25 ms of the free 1 ms for server and HTTP overhead (measured: ~0.23 ms per request).
- add(dt) charges on_hand_end work in full, with no grace.

iters() returns a rollout count from LADDER:
- Base level: 32 if pot <= 8, 64 if pot <= 30, else 128.
- Pace: projected = used / (hand_index + 1) * hands. Above target_s, drop one level per
  started 20% of overshoot. Below 0.5 * target_s, raise one level.
- Hard rule: used > 6.0 s (with the default 8 s bank) returns 0.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from game import Street

LADDER = (0, 16, 32, 64, 128, 256)
GRACE_S = 0.00075         # in-process free time per choose_action call
HARD_STOP_S = 6.0         # no Monte Carlo past this much use of a DEFAULT_BANK_S bank
DEFAULT_BANK_S = 8.0      # a retuned bank scales the hard stop in proportion
SMALL_POT = 8             # base level 32 at or below this pot
MEDIUM_POT = 30           # 64 at or below this pot, 128 above it
OVERSHOOT_STEP = 0.2      # one level down per started 20% of projected overshoot
_EPS = 1e-9               # keeps exact 20% boundaries from rounding up a level


class TimeManager:
    """Per-half bank accounting and a pot-weighted rollout budget. Lives in `memory`."""

    __slots__ = ("bank_s", "hands", "target_s", "clock", "stop_s", "used", "_t0")

    def __init__(self, bank_s: float = 8.0, hands: int = 1000, target_s: float = 3.0,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        if not (bank_s > 0 and hands > 0 and target_s > 0):
            raise ValueError("bank_s, hands and target_s must be positive")
        self.bank_s = bank_s
        self.hands = hands
        self.target_s = target_s
        self.clock = clock
        self.stop_s = HARD_STOP_S * bank_s / DEFAULT_BANK_S
        self.used = 0.0                    # charged seconds so far (never below the judge's count)
        self._t0: float | None = None

    def begin(self) -> None:
        """Start timing one choose_action call."""
        self._t0 = self.clock()

    def end(self) -> None:
        """Charge the time since begin() beyond GRACE_S. Without a begin(), do nothing."""
        t0 = self._t0
        if t0 is None:
            return
        self._t0 = None
        dt = self.clock() - t0 - GRACE_S
        if dt > 0.0:
            self.used += dt

    def add(self, dt: float) -> None:
        """Charge dt seconds in full (on_hand_end work gets no grace)."""
        if dt > 0.0:
            self.used += dt

    def iters(self, pot: int, street: Street, hand_index: int) -> int:
        """Rollouts this decision may spend. `street` is reserved; the rule ignores it."""
        used = self.used
        if used > self.stop_s:
            return 0
        level = 2 if pot <= SMALL_POT else 3 if pot <= MEDIUM_POT else 4
        target = self.target_s
        projected = used * self.hands / (hand_index + 1 if hand_index > 0 else 1)
        if projected > target:
            drop = math.ceil((projected - target) / (OVERSHOOT_STEP * target) - _EPS)
            level -= drop if drop > 1 else 1
            if level <= 0:
                return 0
        elif projected < 0.5 * target:
            level += 1
        return LADDER[level]
