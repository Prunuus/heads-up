"""Archetype opponent zoo: practice opponents imitating likely rival bots.

Every bot is a top-level class (zero-arg constructor) or a functools.partial of
one, so every ZOO value is picklable. Per-half state lives in `memory`.
"""

from __future__ import annotations

import functools
import random
from collections import Counter
from itertools import combinations

from evaluator import (FLUSH, STRAIGHT, TWO_PAIR, card_rank, card_suit,
                       category_of, omaha_score, score_five)
from game import Action, default_action
from sim.match import BotFactory, bot_file_factory

NUM_CARDS = 36
_RAISES = (Action.RAISE_POT, Action.RAISE_HALF)


def _can_raise(obs) -> bool:
    return Action.RAISE_HALF in obs.legal_actions or Action.RAISE_POT in obs.legal_actions


def _legal_or(obs, preferred: Action, fallback: Action) -> Action:
    return preferred if preferred in obs.legal_actions else fallback


def _raise_pref(obs, preferred: Action) -> Action:
    """The preferred raise if legal, else the other raise, else CHECK_CALL."""
    if preferred in obs.legal_actions:
        return preferred
    for a in _RAISES:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL


class _Base:
    def new_half(self, config, memory) -> None:
        memory["rng"] = random.Random(12345)

    def on_hand_end(self, result, memory) -> None:
        return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PAIRS = tuple(combinations(range(5), 2))


def made_hand_now(hole, board) -> int:
    """Best score_five over (2 of 5 hole) x (3 of the visible board cards).
    Needs at least 3 board cards."""
    best = 0
    for i, j in _PAIRS:
        for tri in combinations(board, 3):
            s = score_five((hole[i], hole[j], *tri))
            if s > best:
                best = s
    return best


def mc_equity(hole, board, rollouts: int, rng: random.Random) -> float:
    """Equity vs one uniformly random 5-card hand, pure Python, ties = 0.5."""
    used = set(hole) | set(board)
    deck = [c for c in range(NUM_CARDS) if c not in used]
    need_board = 5 - len(board)
    wins = 0.0
    for _ in range(rollouts):
        draw = rng.sample(deck, 5 + need_board)
        full = tuple(board) + tuple(draw[5:])
        mine = omaha_score(hole, full)
        theirs = omaha_score(draw[:5], full)
        if mine > theirs:
            wins += 1.0
        elif mine == theirs:
            wins += 0.5
    return wins / rollouts


# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------

class GuideExample(_Base):
    """Participant guide section 4.1 strategy."""

    def choose_action(self, obs, memory) -> Action:
        if len(obs.board) < 5:
            if obs.to_call <= obs.big_blind:
                return Action.CHECK_CALL
            return Action.FOLD if Action.FOLD in obs.legal_actions else Action.CHECK_CALL
        if (omaha_score(obs.hole_cards, obs.board) >= (FLUSH << 20)
                and Action.RAISE_POT in obs.legal_actions):
            return Action.RAISE_POT
        return Action.CHECK_CALL


def nit_preflop_score(hole) -> float:
    """Simple strength feature score for a 5-card preflop hand."""
    ranks = [card_rank(c) for c in hole]
    suits = [card_suit(c) for c in hole]
    rc = Counter(ranks)
    score = 0.0
    for r, n in rc.items():
        if n >= 2:
            score += 2.0 + 0.5 * r + (n - 2) * 3.0
    score += sum(max(0, r - 5) for r in ranks) * 0.6      # Q=1, K=2, A=3 (x0.6)
    for s, n in Counter(suits).items():
        if n >= 2:
            hi = max(card_rank(c) for c in hole if card_suit(c) == s)
            score += 0.5 + 0.1 * hi                           # suited, esp. high
    uniq = sorted(set(ranks))
    best = 0
    for lo in range(0, 5):
        best = max(best, sum(1 for r in uniq if lo <= r <= lo + 4))
    score += 0.6 * max(0, best - 2)                           # connectedness
    return score


NIT_THRESHOLD = 10.3     # calibrated: ~20% of random hands pass
NIT_RAISE_THRESHOLD = 13.8  # top ~5%


class Nit(_Base):
    def choose_action(self, obs, memory) -> Action:
        free = obs.to_call == 0
        if len(obs.board) == 0:
            sc = nit_preflop_score(obs.hole_cards)
            if sc >= NIT_RAISE_THRESHOLD and _can_raise(obs):
                return _raise_pref(obs, Action.RAISE_POT)
            if sc >= NIT_THRESHOLD or free:
                return Action.CHECK_CALL
            return Action.FOLD
        cat = category_of(made_hand_now(obs.hole_cards, obs.board))
        if cat >= FLUSH and _can_raise(obs):
            return _raise_pref(obs, Action.RAISE_POT)
        if cat >= TWO_PAIR or free:
            return Action.CHECK_CALL
        return Action.FOLD


class Maniac(_Base):
    def choose_action(self, obs, memory) -> Action:
        rng = memory["rng"]
        if _can_raise(obs) and rng.random() < 0.70:
            return _raise_pref(obs, Action.RAISE_POT if rng.random() < 0.5 else Action.RAISE_HALF)
        return Action.CHECK_CALL


class McEquity(_Base):
    def __init__(self, rollouts: int = 64) -> None:
        self.rollouts = rollouts

    def choose_action(self, obs, memory) -> Action:
        eq = mc_equity(obs.hole_cards, obs.board, self.rollouts, memory["rng"])
        if obs.to_call > 0 and eq <= obs.to_call / (obs.pot + obs.to_call):
            return Action.FOLD if Action.FOLD in obs.legal_actions else Action.CHECK_CALL
        if eq > 0.7 and _can_raise(obs):
            return _raise_pref(obs, Action.RAISE_POT if eq > 0.8 else Action.RAISE_HALF)
        return Action.CHECK_CALL


class ThresholdCat(_Base):
    def choose_action(self, obs, memory) -> Action:
        if len(obs.board) == 0:
            if obs.to_call <= 3 * obs.big_blind:
                return Action.CHECK_CALL
            return Action.FOLD if Action.FOLD in obs.legal_actions else Action.CHECK_CALL
        cat = category_of(made_hand_now(obs.hole_cards, obs.board))
        if cat >= FLUSH and _can_raise(obs):
            return _raise_pref(obs, Action.RAISE_POT)
        if cat >= STRAIGHT or obs.to_call == 0:
            return Action.CHECK_CALL
        return Action.FOLD


class CheckFoldAfter(_Base):
    """mc_equity_fast for hand_index < n, then default_action forever."""

    def __init__(self, n: int) -> None:
        self.n = n
        self._inner = McEquity(64)

    def choose_action(self, obs, memory) -> Action:
        if obs.hand_index < self.n:
            return self._inner.choose_action(obs, memory)
        return default_action(obs.to_call)


def checkfold_after(n: int) -> BotFactory:
    return functools.partial(CheckFoldAfter, n)


ZOO: dict[str, BotFactory] = {
    "random": bot_file_factory("baseline-bots/baseline_random.py"),
    "calling_station": bot_file_factory("baseline-bots/baseline_calling_station.py"),
    "always_half": bot_file_factory("baseline-bots/baseline_always_half.py"),
    "always_pot": bot_file_factory("baseline-bots/baseline_always_pot.py"),
    "guide_example": GuideExample,
    "nit": Nit,
    "maniac": Maniac,
    "mc_equity_fast": functools.partial(McEquity, 64),
    "mc_equity_slow": functools.partial(McEquity, 500),
    "threshold_cat": ThresholdCat,
    "checkfold_after_300": checkfold_after(300),
}
