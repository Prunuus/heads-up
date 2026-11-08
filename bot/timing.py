"""Time manager: count charged thinking time and size Monte Carlo per decision.

The judge charges every HTTP request max(0, elapsed - 1 ms) against an 8 s bank per
1,000-hand half. An empty bank means check/fold for the rest of the half. A request
delivers any finished hands (on_hand_end, once per hand) and then asks for one action.
The bot cannot see request boundaries, so it counts this way:

- begin()/end() wrap choose_action and charge max(0, dt - grace_s). The default grace_s
  (0.75 ms) leaves 0.25 ms of the free 1 ms for server and HTTP plumbing, measured at
  ~0.23 ms per request on the dev laptop. Plumbing is CPU-bound too: on a machine f
  times slower, pass grace_s=grace_for_cpu_factor(f), or the count falls below the
  judge's once plumbing passes 0.25 ms per request.
- add(dt) charges on_hand_end work in full, with no grace.

iters() returns a rollout count from LADDER:
- Base level: 32 if pot <= 8, 64 if pot <= 30, else 128.
- Pace: projected = used / (hand_index + 1) * hands. Above target_s, drop one level per
  started 20% of overshoot. Below 0.5 * target_s, raise one level. target_s defaults to
  0.375 * bank_s (3.0 s of the default 8 s bank).
- Hard rule: used > 6.0 s (with the default 8 s bank) returns 0.

The constructor never raises, because a raise inside new_half forfeits the half. An
unusable bank means no Monte Carlo at all; other unusable arguments get their defaults.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from game import Street

LADDER = (0, 16, 32, 64, 128, 256)
GRACE_S = 0.00075         # default free time per choose_action call (dev-laptop plumbing)
HARD_STOP_S = 6.0         # no Monte Carlo past this much use of a DEFAULT_BANK_S bank
DEFAULT_BANK_S = 8.0      # a retuned bank scales the hard stop in proportion
TARGET_FRACTION = 0.375   # default target_s as a share of the bank: 3.0 s of 8 s
DEFAULT_HANDS = 1000
SMALL_POT = 8             # base level 32 at or below this pot
MEDIUM_POT = 30           # 64 at or below this pot, 128 above it
OVERSHOOT_STEP = 0.2      # one level down per started 20% of projected overshoot
_EPS = 1e-9               # keeps exact 20% boundaries from rounding up a level


def _real(x: object) -> float | None:
    """x as a float, or None when it is not a number or is NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if v == v else None


def grace_for_cpu_factor(f: float) -> float:
    """grace_s for a CPU f times slower than the dev laptop.

    The judge's free 1 ms minus plumbing of 0.25 ms * f. Below zero once plumbing alone
    passes 1 ms (f > 4); end() then charges that overrun too. A non-number or f <= 0
    means "not measured" and gives the default GRACE_S.
    """
    v = _real(f)
    if v is None or not 0 < v < math.inf:
        return GRACE_S
    return 0.001 - 0.00025 * v


class TimeManager:
    """Per-half bank accounting and a pot-weighted rollout budget. Lives in `memory`."""

    __slots__ = ("bank_s", "hands", "target_s", "clock", "grace_s", "stop_s", "used", "_t0")

    def __init__(self, bank_s: float = 8.0, hands: int = 1000, target_s: float | None = None,
                 clock: Callable[[], float] = time.perf_counter,
                 grace_s: float = GRACE_S) -> None:
        bank, n, target, grace = _real(bank_s), _real(hands), _real(target_s), _real(grace_s)
        if bank is None or not bank > 0:
            bank, self.stop_s = 0.0, -1.0          # unusable bank: never any Monte Carlo
        else:
            self.stop_s = HARD_STOP_S * bank / DEFAULT_BANK_S
        if target is None or not 0 < target < math.inf:
            target = TARGET_FRACTION * (bank or DEFAULT_BANK_S)
        self.bank_s = bank
        self.hands = n if n is not None and 0 < n < math.inf else DEFAULT_HANDS
        self.target_s = target
        self.grace_s = grace if grace is not None and math.isfinite(grace) else GRACE_S
        self.clock = clock
        self.used = 0.0     # charged s; >= the judge's count while plumbing <= 1 ms - grace_s
        self._t0: float | None = None

    def begin(self) -> None:
        """Start timing one choose_action call."""
        self._t0 = self.clock()

    def end(self) -> None:
        """Charge the time since begin() beyond grace_s. Without a begin(), do nothing."""
        t0 = self._t0
        if t0 is None:
            return
        self._t0 = None
        dt = self.clock() - t0 - self.grace_s
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
            # Started 20% steps of overshoot. Capped at 5, which reaches 0 from any base
            # level and keeps an infinite overshoot out of math.ceil.
            over = min((projected / target - 1.0) / OVERSHOOT_STEP, 5.0)
            drop = math.ceil(over - _EPS)
            level -= drop if drop > 1 else 1
            if level <= 0:
                return 0
        elif projected < 0.5 * target:
            level += 1
        return LADDER[level]
