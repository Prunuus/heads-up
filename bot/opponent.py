"""Opponent model: how the opponent plays this half, plus reversible exploit detectors.

One OpponentModel lives in the per-half memory: memory["opp"] = OpponentModel() in
new_half. Call observe(obs) first thing in choose_action, and finish(result) in
on_hand_end for every finished hand, in any order and in batches. Neither method ever
raises: an exception in on_hand_end is an error, and 5 errors forfeit the match. Any
failure is swallowed and counted in `errors`.

Decisions. Each opponent decision is recorded with its street and facing_bet (the
opponent owed chips when it acted), its action (fold, check/call, bet/raise), and
whether a raise was legal for it then.
- observe() replays obs.action_history from the first record it has not seen. Blinds are
  not in the history: preflop starts with our blind as our street total. The stacks before
  each record are rebuilt backwards from the stacks in obs, so "a raise was legal" (fewer
  than raise_cap raises on the street, the opponent can more than call, we are not all-in)
  needs no config.
- finish() adds what the history cannot show, since HandResult has no action history
  (F11). At most one opponent decision follows our last action in a hand:
  - no showdown, we won, and we acted: our last action was a bet or raise and the
    opponent folded to it;
  - no showdown, we won, we never acted, and the pot is under two big blinds: we were the
    big blind and the opponent folded its button preflop (the judge never asks us to act);
  - showdown, and more chips went in after our last observation than our call: we bet or
    raised and the opponent called it (maybe all-in for less);
  - showdown after we checked first on the river: the opponent checked behind.
  Results are keyed by hand_index: out-of-order results are fine, and a repeated
  hand_index is ignored.

Statistics (whole half): fold_to_bet(street) and raise_rate(street) as (rate, n), with
rate 0.0 when n == 0, and vpip as (rate, n) over hands where the opponent made a preflop
decision.

Action class: (street, aggressive) as (int, bool), where aggressive means the opponent bet
or raised at least once on that street of the hand. At each showdown, every street on which
the opponent made a decision adds (class, category of omaha_score(opp_hole, board)).
omaha_score's table must be built at startup (agent.py calls evaluator.warmup()).

Detectors are computed from the current evidence on every call, so each turns off as soon
as the evidence stops supporting it. Station and maniac read only the most recent
decisions (STATION_WINDOW, MANIAC_WINDOW), so late in a half they turn off as fast as early.
- is_check_folder(): the opponent's last CF_STREAK (12) bet-facing decisions were all
  folds, and it has not bet or raised since the first of them. Any call, bet or raise
  starts the run over. The brief's "0 raises in the last 40 decisions" is scoped to the
  run on purpose: a bot that empties its bank mid-half raised shortly before, and a fixed
  40-decision window delays detection against checkfold_after(50) to hands 60-77 (8 seeds;
  1.3 opponent decisions per hand), past the required hand 70. The run detects at 60-61.
- is_station(): postflop fold-to-bet < 0.05 over the last STATION_WINDOW bet-facing
  postflop decisions, with n >= 60. Postflop only: preflop equities are compressed (F6),
  so even folding bots almost never fold preflop.
- is_maniac(): bet/raise rate > 0.55 over the last MANIAC_WINDOW decisions where a raise
  was legal (all streets), with n >= 80.
"""

from __future__ import annotations

from collections import deque

from evaluator import omaha_score
from game import BIG_BLIND, RAISE_CAP_PER_STREET

_FOLD, _CALL = 0, 1             # Action.FOLD, Action.CHECK_CALL; both raises are >= 2
_RAISE = 2
_PREFLOP, _RIVER = 0, 3
_STREETS = 4
_CATEGORIES = 9                 # HIGH_CARD (0) .. STRAIGHT_FLUSH (8)
_CATEGORY_SHIFT = 20

CF_STREAK = 12                  # check-folder: bet-facing folds in a row, no bet/raise between
STATION_MAX_FOLD = 0.05
STATION_MIN_N = 60
STATION_WINDOW = 150            # latest bet-facing postflop decisions
MANIAC_MIN_RATE = 0.55
MANIAC_MIN_N = 80
MANIAC_WINDOW = 200             # latest decisions where a raise was legal
SHOWDOWN_MIN_N = 5              # showdown_min_category answers only with this many samples
MAX_OPEN_HANDS = 8              # unfinished hands kept (normally 1)


def _positive_int(v, default: int) -> int:
    return v if type(v) is int and v > 0 else default


class _Hand:
    """One hand we have observed and not finished yet."""

    __slots__ = ("seen", "street", "my_total", "raises", "last", "acted", "aggr", "vpip")

    def __init__(self, my_total: int = 0) -> None:
        self.seen = 0               # action_history records already replayed
        self.street = _PREFLOP      # street of the latest replayed record
        self.my_total = my_total    # our chips in on that street
        self.raises = 0             # bets and raises on that street
        self.last = None            # our latest observation of the hand
        self.acted = [False, False, False, False]   # the opponent decided on street s
        self.aggr = [False, False, False, False]    # ... and bet or raised there
        self.vpip = False           # the opponent put chips in voluntarily preflop


class OpponentModel:
    """Per-half opponent statistics and detectors. See the module docstring."""

    def __init__(self, raise_cap: int = RAISE_CAP_PER_STREET, big_blind: int = BIG_BLIND) -> None:
        # Bad arguments get the defaults: a raise in new_half forfeits the half.
        self.raise_cap = _positive_int(raise_cap, RAISE_CAP_PER_STREET)
        self._bb = _positive_int(big_blind, BIG_BLIND)
        self._errors = 0                    # exceptions swallowed by observe/finish
        self._n = 0                         # opponent decisions recorded
        self._face_n = [0] * _STREETS       # bet-facing decisions per street
        self._face_fold = [0] * _STREETS
        self._raise_n = [0] * _STREETS      # decisions where a raise was legal
        self._raise_k = [0] * _STREETS
        self._vpip_n = 0
        self._vpip_k = 0
        self._sd: dict = {}                 # (street, aggressive) -> counts per category
        self._hands: dict = {}              # hand_index -> _Hand, unfinished
        self._done: set = set()             # finished hand indexes
        self._streak = 0                    # bet-facing folds in a row, no bet/raise between
        self._st_win: deque = deque(maxlen=STATION_WINDOW)
        self._st_folds = 0
        self._mn_win: deque = deque(maxlen=MANIAC_WINDOW)
        self._mn_raises = 0

    # ----- feeding -----------------------------------------------------------

    def observe(self, obs) -> None:
        """Record the opponent decisions in obs.action_history not seen yet."""
        try:
            self._observe(obs)
        except Exception:
            self._errors += 1

    def finish(self, result) -> None:
        """Infer the opponent's decision after our last action, and record showdowns."""
        try:
            self._finish(result)
        except Exception:
            self._errors += 1

    def _observe(self, obs) -> None:
        idx = obs.hand_index
        if idx in self._done:
            return
        hands = self._hands
        h = hands.get(idx)
        hist = obs.action_history
        n = len(hist)
        if h is None:
            bb = obs.big_blind
            if type(bb) is int and bb > 0:
                self._bb = bb
            h = _Hand(self._bb // 2 if obs.is_button else self._bb)
            hands[idx] = h
            if len(hands) > MAX_OPEN_HANDS:
                del hands[next(iter(hands))]        # the oldest unfinished hand
        start = h.seen
        if n < start:
            return                                  # an older observation of this hand
        h.seen = n                                  # a bad record is never replayed twice
        h.last = obs
        if n > start:
            self._replay(h, hist[start:], obs.my_stack, obs.opp_stack)

    def _replay(self, h: _Hand, new, my_stack: int, opp_stack: int) -> None:
        m = len(new)
        # Stacks just before each new record, rebuilt backwards from the stacks now.
        stacks = [None] * m
        me, op = my_stack, opp_stack
        for k in range(m - 1, -1, -1):
            r = new[k]
            if r.is_me:
                me += r.amount
            else:
                op += r.amount
            stacks[k] = (me, op)
        cap = self.raise_cap
        street, my_total, raises = h.street, h.my_total, h.raises
        for k in range(m):
            r = new[k]
            s = int(r.street)
            a = int(r.action)
            if not (0 <= s < _STREETS and 0 <= a <= 3):
                continue
            if s != street:
                street, my_total, raises = s, 0, 0
            if r.is_me:
                my_total = r.street_total
            else:
                owed = my_total - (r.street_total - r.amount)
                me, op = stacks[k]
                self._record(h, s, owed > 0 or a == _FOLD, a, raises < cap and op > owed and me > 0)
            if a >= _RAISE:
                raises += 1
        h.street, h.my_total, h.raises = street, my_total, raises

    def _finish(self, res) -> None:
        idx = res.hand_index
        if idx in self._done:
            return
        self._done.add(idx)
        h = self._hands.pop(idx, None)
        won = res.winner == "me"
        showdown = res.went_to_showdown
        if h is None or h.last is None:
            # We never acted. Only the button folding its first preflop decision ends a
            # hand before the big blind acts; the pot is then the two blinds.
            if won and not showdown and res.pot < 2 * self._bb:
                self._record(_Hand(), _PREFLOP, True, _FOLD, True)
            return
        o = h.last
        s = int(o.street)
        if not 0 <= s < _STREETS:
            return
        added = res.pot - o.pot                     # chips in after our last observation
        cap = self.raise_cap
        if not showdown:
            if won:
                # Our last action was a bet or raise of `added`, and the opponent folded.
                owed = o.my_bet + added - o.opp_bet
                could = o.raises_this_street + 1 < cap and o.opp_stack > owed and o.my_stack > added
                self._record(h, s, True, _FOLD, could)
            return
        call = min(o.to_call, o.my_stack)
        if added == call:
            # We checked or called. Only our river check as first to act leaves the
            # opponent a decision: it checked behind.
            if o.to_call == 0 and s == _RIVER and not any(r.street == _RIVER for r in o.action_history):
                could = o.raises_this_street < cap and o.opp_stack > 0 and o.my_stack > 0
                self._record(h, s, False, _CALL, could)
        elif added > call:
            # We bet or raised and the opponent called: find which raise matches the pot.
            could = False
            for to in o.raise_to.values():
                mine, owed = to - o.my_bet, to - o.opp_bet
                if mine + min(owed, o.opp_stack) == added:
                    could = o.raises_this_street + 1 < cap and o.opp_stack > owed and o.my_stack > mine
                    break
            self._record(h, s, True, _CALL, could)
        self._showdown(h, res)

    def _showdown(self, h: _Hand, res) -> None:
        hole, board = res.opp_hole, res.board
        if hole is None or len(board) != 5:
            return
        cat = omaha_score(hole, board) >> _CATEGORY_SHIFT
        if not 0 <= cat < _CATEGORIES:
            return
        sd = self._sd
        for s in range(_STREETS):
            if h.acted[s]:
                key = (s, h.aggr[s])
                counts = sd.get(key)
                if counts is None:
                    counts = sd[key] = [0] * _CATEGORIES
                counts[cat] += 1

    def _record(self, h: _Hand, s: int, facing: bool, a: int, could_raise: bool) -> None:
        """Count one opponent decision on street s (action a as an int)."""
        raised = a >= _RAISE
        self._n += 1
        if facing:
            self._face_n[s] += 1
            folded = a == _FOLD
            if folded:
                self._face_fold[s] += 1
                self._streak += 1
            else:
                self._streak = 0
            if s:
                win = self._st_win
                if len(win) == STATION_WINDOW:
                    self._st_folds -= win[0]
                win.append(folded)
                self._st_folds += folded
        if raised:
            self._streak = 0
        if could_raise:
            self._raise_n[s] += 1
            self._raise_k[s] += raised
            win = self._mn_win
            if len(win) == MANIAC_WINDOW:
                self._mn_raises -= win[0]
            win.append(raised)
            self._mn_raises += raised
        if s == _PREFLOP:
            if not h.acted[0]:
                self._vpip_n += 1
            if not h.vpip and (raised or (facing and a == _CALL)):
                h.vpip = True
                self._vpip_k += 1
        h.acted[s] = True
        if raised:
            h.aggr[s] = True

    # ----- statistics --------------------------------------------------------

    def fold_to_bet(self, street) -> tuple[float, int]:
        """(fold rate, n) over the opponent's bet-facing decisions on `street` this half."""
        return self._rate(self._face_fold, self._face_n, street)

    def raise_rate(self, street) -> tuple[float, int]:
        """(bet/raise rate, n) over its decisions on `street` where a raise was legal."""
        return self._rate(self._raise_k, self._raise_n, street)

    @staticmethod
    def _rate(k: list, n: list, street) -> tuple[float, int]:
        try:
            s = int(street)
        except Exception:
            return (0.0, 0)
        if not 0 <= s < _STREETS:
            return (0.0, 0)
        return (k[s] / n[s] if n[s] else 0.0, n[s])

    @property
    def errors(self) -> int:
        """Exceptions observe() and finish() swallowed (bad input); 0 in normal play."""
        return self._errors

    @property
    def decisions(self) -> int:
        """Opponent decisions recorded this half, inferred ones included."""
        return self._n

    @property
    def fold_streak(self) -> int:
        """Bet-facing folds in a row, with no call, bet or raise since the first of them."""
        return self._streak

    @property
    def vpip(self) -> tuple[float, int]:
        """(share of hands it called or raised preflop, n hands with a preflop decision)."""
        n = self._vpip_n
        return (self._vpip_k / n if n else 0.0, n)

    @property
    def recent_postflop_fold_to_bet(self) -> tuple[float, int]:
        """(fold rate, n) over the last STATION_WINDOW bet-facing postflop decisions."""
        n = len(self._st_win)
        return (self._st_folds / n if n else 0.0, n)

    @property
    def recent_raise_rate(self) -> tuple[float, int]:
        """(bet/raise rate, n) over the last MANIAC_WINDOW decisions where a raise was legal."""
        n = len(self._mn_win)
        return (self._mn_raises / n if n else 0.0, n)

    def showdown_count(self, action_class) -> int:
        """Showdowns recorded for an action class (street, aggressive)."""
        try:
            counts = self._sd.get((int(action_class[0]), bool(action_class[1])))
        except Exception:
            return 0
        return sum(counts) if counts else 0

    def showdown_min_category(self, action_class, min_n: int = SHOWDOWN_MIN_N) -> int | None:
        """Weakest category (0..8) the opponent showed down after this action class, or None
        with fewer than min_n showdowns for it. action_class is (street, aggressive)."""
        try:
            counts = self._sd.get((int(action_class[0]), bool(action_class[1])))
            if not counts or sum(counts) < max(1, min_n):
                return None
        except Exception:
            return None
        for cat, c in enumerate(counts):
            if c:
                return cat
        return None

    @staticmethod
    def current_class(obs) -> tuple[int, bool] | None:
        """The opponent's action class on obs.street so far (None if obs is unusable)."""
        try:
            s = int(obs.street)
            return (s, any(not r.is_me and r.street == s and r.action >= _RAISE
                           for r in obs.action_history))
        except Exception:
            return None

    # ----- detectors ---------------------------------------------------------

    def is_check_folder(self) -> bool:
        """Folds to every bet and never bets: e.g. a bot that ran out of time bank."""
        return self._streak >= CF_STREAK

    def is_station(self) -> bool:
        """Almost never folds to a postflop bet."""
        n = len(self._st_win)
        return n >= STATION_MIN_N and self._st_folds < STATION_MAX_FOLD * n

    def is_maniac(self) -> bool:
        """Bets or raises most of the time it can."""
        n = len(self._mn_win)
        return n >= MANIAC_MIN_N and self._mn_raises > MANIAC_MIN_RATE * n
