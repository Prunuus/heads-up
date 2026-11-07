"""In-process duplicate match simulator mirroring judge_engine.py (no HTTP).

A bot is any object with new_half(config, memory), choose_action(obs, memory)
and on_hand_end(result, memory). Factories are zero-arg callables returning a
bot; use bot_file_factory(path) (picklable) for bot files.
"""

from __future__ import annotations

import functools
import importlib.util
import itertools
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from game import (ACTION_TIMEOUT_SECONDS, Action, Street, ERROR_LIMIT,  # noqa: E402
                  TIME_BANK_SECONDS, AgentConfig, Game, button_for_deal,
                  default_action, make_deal)
from judge_engine import (AGENT1_WIN, AGENT2_WIN, BOTH_FORFEIT, CHOP,  # noqa: E402
                          AgentStats, HalfResult, agent1_seat_for_half,
                          coerce_action, combine_halves)

BotFactory = Callable[[], Any]

_VERDICTS = {AGENT1_WIN: "A", AGENT2_WIN: "B", CHOP: "CHOP", BOTH_FORFEIT: "BOTH_FORFEIT"}
_counter = itertools.count()


def load_bot_file(path: str) -> ModuleType:
    """Import a bot file under a unique module name (fresh globals each call)."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    name = f"_simbot_{next(_counter)}_{p.stem}"
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def bot_file_factory(path: str) -> Callable[[], ModuleType]:
    """Picklable factory for a bot file."""
    return functools.partial(load_bot_file, path)


@dataclass
class HalfStats:
    nets_a: list
    errors_a: int
    errors_b: int
    forfeit_a: bool
    forfeit_b: bool
    bank_a: float          # seconds charged to the time bank
    bank_b: float
    calls_a: int
    calls_b: int
    max_call_a: float      # longest single request of A, seconds
    # Action actually played (incl. defaults on error/bank-out), keyed by Action name.
    actions_a: dict = field(default_factory=dict)
    actions_b: dict = field(default_factory=dict)
    # Decision context counters: faced (to_call>0), faced_fold, pre (preflop decisions), pre_fold.
    ctx_a: dict = field(default_factory=dict)
    ctx_b: dict = field(default_factory=dict)
    bank_exhausted_a: bool = False
    bank_exhausted_b: bool = False


@dataclass
class MatchStats:
    mean_bb: float
    sd_bb: float
    ci_low: float
    ci_high: float
    verdict: str
    points_a: float
    halves: tuple


class _Handle:
    """Mirror of judge_engine.HttpAgentHandle around an in-process bot."""

    def __init__(self, factory: BotFactory, config: AgentConfig, grace_s: float,
                 action_timeout_s: float = ACTION_TIMEOUT_SECONDS) -> None:
        self.action_timeout_s = action_timeout_s
        self.grace_s = grace_s
        self.factory = factory
        self.config = config
        self.stats = AgentStats()
        self.pending: list = []
        self.max_call = 0.0
        self.bot = None
        self.memory: dict = {}
        self.actions: dict = {}
        self.ctx: dict = {}

    def _record_error(self) -> None:
        self.stats.errors += 1
        if self.stats.errors >= ERROR_LIMIT:
            self.stats.forfeited = True

    def _remaining_bank(self) -> float:
        return self.config.time_bank_seconds - self.stats.time_seconds

    def _usable(self) -> bool:
        if self.stats.forfeited:
            return False
        if self._remaining_bank() <= 0:
            self.stats.timed_out = True
            self.stats.skipped_calls += 1
            return False
        return True

    def _request(self, fn: Callable[[list], Any]):
        """Run one timed request. fn(hand_end_errors) returns the payload.
        Returns (ok, payload, raised): ok False means the reply is discarded."""
        bank_left = self._remaining_bank() + self.grace_s
        limited_by_bank = bank_left < self.action_timeout_s
        timeout = max(0.001, min(self.action_timeout_s, bank_left))
        he_errors: list = []
        raised = False
        payload = None
        start = time.perf_counter()
        try:
            payload = fn(he_errors)
        except Exception:
            raised = True
        elapsed = time.perf_counter() - start
        self.stats.wall_seconds += elapsed
        self.stats.time_seconds += max(0.0, elapsed - self.grace_s)
        self.stats.calls += 1
        self.max_call = max(self.max_call, elapsed)

        if elapsed > timeout:
            if limited_by_bank:
                self.stats.timed_out = True
            else:
                self._record_error()
            return False, None
        for _ in he_errors:
            self._record_error()
        if raised:
            self._record_error()
            return False, None
        return True, payload

    def start_half(self) -> None:
        self.stats = AgentStats()
        self.pending = []
        self.memory = {}
        self.actions = {}
        self.ctx = {}
        try:
            self.bot = self.factory()
        except Exception:
            self.stats.errors += 1
            self.stats.forfeited = True
            return

        def call(_errs):
            self.bot.new_half(self.config, self.memory)
            return True

        ok, _ = self._request(call)
        if not ok:
            self.stats.forfeited = True

    def _deliver(self, results: list, errs: list) -> None:
        for r in results:
            try:
                self.bot.on_hand_end(r, self.memory)
            except Exception:
                errs.append(1)

    def act(self, obs) -> Any:
        action = self._act(obs)
        name = Action(action).name
        self.actions[name] = self.actions.get(name, 0) + 1
        fold = name == "FOLD"
        c = self.ctx
        if obs.to_call > 0:
            c["faced"] = c.get("faced", 0) + 1
            c["faced_fold"] = c.get("faced_fold", 0) + fold
        if obs.street == Street.PREFLOP:
            c["pre"] = c.get("pre", 0) + 1
            c["pre_fold"] = c.get("pre_fold", 0) + fold
        return action

    def _act(self, obs) -> Any:
        fallback = default_action(obs.to_call)
        if not self._usable():
            return fallback
        results, self.pending = self.pending, []

        def call(errs):
            self._deliver(results, errs)
            return self.bot.choose_action(obs, self.memory)

        ok, raw = self._request(call)
        if not ok:
            return fallback
        # The HTTP server passes str/int/float/None through; anything else becomes repr().
        action = coerce_action(raw) if raw is None or isinstance(raw, (int, str, float)) else None
        if action is None or action not in obs.legal_actions:
            self._record_error()
            return fallback
        return action

    def hand_end(self, result) -> None:
        if not self.stats.forfeited:
            self.pending.append(result)

    def flush(self) -> None:
        results, self.pending = self.pending, []
        if results and self._usable():
            self._request(lambda errs: self._deliver(results, errs))


def play_half(fa: BotFactory, fb: BotFactory, seed: Any, half: int,
              deals: int = 1000, grace_s: float = 0.00075,
              time_bank_s: float = TIME_BANK_SECONDS,
              action_timeout_s: float = ACTION_TIMEOUT_SECONDS) -> HalfStats:
    """Play one half; A is agent1 (seat 0 in half 1, seat 1 in half 2)."""
    seed = str(seed)
    a_seat = agent1_seat_for_half(half)
    config = AgentConfig(num_hands=deals, time_bank_seconds=time_bank_s)
    ha, hb = _Handle(fa, config, grace_s, action_timeout_s), _Handle(fb, config, grace_s, action_timeout_s)
    bots = (ha, hb) if a_seat == 0 else (hb, ha)
    for b in bots:
        b.start_half()
    nets_a: list = []
    for d in range(deals):
        game = Game(make_deal(seed, d), button_for_deal(d), hand_index=d)
        while not game.is_terminal:
            seat = game.to_act
            game.apply(bots[seat].act(game.observation(seat)))
        nets = game.result().nets
        nets_a.append(nets[a_seat])
        for seat in (0, 1):
            bots[seat].hand_end(game.hand_result(seat))
    for b in bots:
        b.flush()
    return HalfStats(nets_a, ha.stats.errors, hb.stats.errors,
                     ha.stats.forfeited, hb.stats.forfeited,
                     ha.stats.time_seconds, hb.stats.time_seconds,
                     ha.stats.calls, hb.stats.calls, ha.max_call,
                     dict(ha.actions), dict(hb.actions), dict(ha.ctx), dict(hb.ctx),
                     ha.stats.timed_out, hb.stats.timed_out)


def _as_half_result(h: HalfStats, half: int, seed: str) -> HalfResult:
    return HalfResult(
        half=half, match_seed=seed, num_deals=len(h.nets_a),
        agent1_seat=agent1_seat_for_half(half),
        nets_agent1=list(h.nets_a), nets_agent2=[-x for x in h.nets_a],
        stats_agent1=AgentStats(errors=h.errors_a, forfeited=h.forfeit_a),
        stats_agent2=AgentStats(errors=h.errors_b, forfeited=h.forfeit_b))


def play_match(fa: BotFactory, fb: BotFactory, seed: Any, deals: int = 1000,
               grace_s: float = 0.00075) -> MatchStats:
    """Play both halves and score with the judge's own combine_halves."""
    seed = str(seed)
    h1 = play_half(fa, fb, seed, 1, deals, grace_s)
    h2 = play_half(fa, fb, seed, 2, deals, grace_s)
    r = combine_halves(_as_half_result(h1, 1, seed), _as_half_result(h2, 2, seed))
    return MatchStats(r.mean_bb, r.std_bb, r.ci[0], r.ci[1], _VERDICTS[r.verdict],
                      r.points["agent1"], (h1, h2))
