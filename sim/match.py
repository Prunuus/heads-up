"""In-process duplicate match simulator mirroring judge_engine.py (no HTTP).

A bot is any object with new_half(config, memory), choose_action(obs, memory)
and on_hand_end(result, memory). Factories are zero-arg callables returning a
bot; use bot_file_factory(path) (picklable) for bot files.

time_scale (ruling R30b) emulates a CPU that is `time_scale` times slower than this one,
per bot: every request's measured time is multiplied by it before it is charged to the
bank and compared with the timeouts. The default grace follows it: the judge's free 1 ms
minus SIM_PLUMBING_S of HTTP plumbing per request (which the in-process sim does not
run), and on a slower CPU that plumbing is slower too: grace_for_time_scale(s). A bot
that times itself would see the slow CPU too, so the bot is told: it gets the scale as
SimAgentConfig.time_scale (our PolicyBot reads it and scales its own clock and CPU
factor, bot/policy.py), and in a bot file every module-level global that is the time
module (under any name) or one of its clock functions (CLOCK_NAMES, imported under any
name) is replaced by a ScaledTime (or its clock), whose intervals run s times long. A
clock kept anywhere else (inside a class, a closure or a helper module; the archetypes,
which do fixed work and keep no clock) is not reached: such a bot spends as on this
machine while it is charged as on the slow one.
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
                  NETWORK_GRACE_SECONDS, TIME_BANK_SECONDS, AgentConfig, Game,
                  button_for_deal, default_action, make_deal)
from judge_engine import (AGENT1_WIN, AGENT2_WIN, BOTH_FORFEIT, CHOP,  # noqa: E402
                          AgentStats, HalfResult, agent1_seat_for_half,
                          coerce_action, combine_halves)

BotFactory = Callable[[], Any]

# HTTP plumbing per request (judge -> uvicorn -> agent.py and back) that the in-process
# sim does not run: ~0.25 ms on the dev machines (0.23 ms Mac, 0.29 ms PC). The judge
# charges every request beyond NETWORK_GRACE_SECONDS, plumbing included.
SIM_PLUMBING_S = 0.00025


def grace_for_time_scale(scale: float) -> float:
    """Free time per request in the sim for a CPU `scale` times slower than this one:
    the judge's 1 ms minus the plumbing, which scales too. Negative past scale 4 (the
    plumbing alone overruns the free ms, and the judge charges the overrun)."""
    return NETWORK_GRACE_SECONDS - SIM_PLUMBING_S * scale


@dataclass(frozen=True)
class SimAgentConfig(AgentConfig):
    """AgentConfig plus the sim's emulated CPU slowdown for this bot (1.0: this machine).
    The real judge sends a plain AgentConfig, which has no time_scale."""

    time_scale: float = 1.0


class ScaledTime:
    """Stand-in for the `time` module inside one bot file: every interval its clocks
    (perf_counter, monotonic, time, process_time and their _ns forms) measure is `scale`
    times as long, as a bot on a CPU that much slower would measure its own work. Each
    clock reads its real value at creation, so absolute times stay plausible. Everything
    else, sleep included, is the real module's."""

    def __init__(self, scale: float) -> None:
        self._scale = scale
        self._t0 = {n: getattr(time, n)() for n in
                    ("perf_counter", "monotonic", "time", "process_time")}

    def _scaled(self, name: str) -> float:
        t0 = self._t0[name]
        return t0 + (getattr(time, name)() - t0) * self._scale

    def perf_counter(self) -> float:
        return self._scaled("perf_counter")

    def monotonic(self) -> float:
        return self._scaled("monotonic")

    def time(self) -> float:
        return self._scaled("time")

    def process_time(self) -> float:
        return self._scaled("process_time")

    def perf_counter_ns(self) -> int:
        return int(self._scaled("perf_counter") * 1e9)

    def monotonic_ns(self) -> int:
        return int(self._scaled("monotonic") * 1e9)

    def time_ns(self) -> int:
        return int(self._scaled("time") * 1e9)

    def process_time_ns(self) -> int:
        return int(self._scaled("process_time") * 1e9)

    def __getattr__(self, name: str):
        return getattr(time, name)


CLOCK_NAMES = ("perf_counter", "monotonic", "time", "process_time",
               "perf_counter_ns", "monotonic_ns", "time_ns", "process_time_ns")


def _scale_bot_clock(bot: Any, scale: float) -> None:
    """In a bot file (a module), replace every module-level global that is the time module
    (whatever its name: `import time as _time` too) with a ScaledTime, and every one that
    is a clock function of it (`from time import perf_counter`, under any name) with that
    ScaledTime's version. One ScaledTime per bot, so all its clocks share their origin."""
    if scale == 1.0 or not isinstance(bot, ModuleType):
        return
    clocks = [(getattr(time, n), n) for n in CLOCK_NAMES]
    scaled = None
    for name, value in list(vars(bot).items()):
        if value is time:
            replacement = scaled = scaled or ScaledTime(scale)
        else:
            clock = next((n for fn, n in clocks if value is fn), None)
            if clock is None:
                continue
            scaled = scaled or ScaledTime(scale)
            replacement = getattr(scaled, clock)
        setattr(bot, name, replacement)


def _scales(time_scale) -> tuple:
    """(scale A, scale B) from one number or a pair; each must be finite and > 0."""
    pair = tuple(time_scale) if isinstance(time_scale, (tuple, list)) else (time_scale,) * 2
    if len(pair) != 2:
        raise ValueError(f"time_scale must be a number or a pair, got {time_scale!r}")
    out = []
    for v in pair:
        v = float(v)
        if not 0.0 < v < float("inf"):
            raise ValueError(f"time_scale must be finite and > 0, got {time_scale!r}")
        out.append(v)
    return tuple(out)

_VERDICTS = {AGENT1_WIN: "A", AGENT2_WIN: "B", CHOP: "CHOP", BOTH_FORFEIT: "BOTH_FORFEIT"}
_counter = itertools.count()

# Packages a bot file gets a fresh copy of on every load: the tournament starts each
# half in a new process, so module-level state in our runtime package must not cross
# halves, matches or candidates in the simulator either (ruling R22).
FRESH_PACKAGES: tuple = ("bot",)


def _evict(packages: tuple) -> dict:
    """Remove each package and its submodules from sys.modules; return what was removed."""
    gone = {}
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in packages):
            gone[name] = sys.modules.pop(name)
    return gone


def load_bot_file(path: str, fresh_packages: tuple = FRESH_PACKAGES) -> ModuleType:
    """Import a bot file the way the tournament starts a half: as a fresh process would.

    - The file runs under a unique module name, so every call gets fresh globals.
    - Every package in fresh_packages (default: our `bot`) is evicted from sys.modules
      before the file runs, so its imports of them run fresh too. Two bots in one match
      each get their own copy.
    - Afterwards sys.modules is put back as it was. The caller keeps its own copy of those
      packages (pickling, monkeypatching and isinstance keep working), and the loaded
      module with its private copies lives only as long as the caller holds it, so a long
      serial run does not keep one copy of the package (and its tables) per half.

    Bot code must import at load time (module top level): an import that first runs
    during play resolves against the caller's sys.modules, not the bot's private copy.
    For the same reason a bot must not pickle objects of its private package copy
    during play: pickle looks classes up by name in the caller's sys.modules, finds the
    caller's copy (not the same class object), and fails. Factories are pickled before
    any load, so they are unaffected.
    """
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    name = f"_simbot_{next(_counter)}_{p.stem}"
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    saved = _evict(fresh_packages)
    sys.modules[name] = module            # registered while it runs (dataclasses need it)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
        _evict(fresh_packages)
        sys.modules.update(saved)
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
                 action_timeout_s: float = ACTION_TIMEOUT_SECONDS,
                 time_scale: float = 1.0) -> None:
        self.action_timeout_s = action_timeout_s
        self.grace_s = grace_s
        self.time_scale = time_scale        # measured time x this = emulated time
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
        elapsed = (time.perf_counter() - start) * self.time_scale
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
        _scale_bot_clock(self.bot, self.time_scale)

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
              deals: int = 1000, grace_s: float | None = None,
              time_bank_s: float = TIME_BANK_SECONDS,
              action_timeout_s: float = ACTION_TIMEOUT_SECONDS,
              time_scale: float | tuple = 1.0) -> HalfStats:
    """Play one half; A is agent1 (seat 0 in half 1, seat 1 in half 2).

    time_scale: one number for both bots or a pair (A, B); see the module docstring.
    grace_s: free time per request for both bots; None gives each bot
    grace_for_time_scale(its scale) (0.75 ms at 1.0).
    """
    seed = str(seed)
    a_seat = agent1_seat_for_half(half)
    handles = []
    for factory, scale in zip((fa, fb), _scales(time_scale)):
        config = SimAgentConfig(num_hands=deals, time_bank_seconds=time_bank_s,
                                time_scale=scale)
        grace = grace_for_time_scale(scale) if grace_s is None else grace_s
        handles.append(_Handle(factory, config, grace, action_timeout_s, scale))
    ha, hb = handles
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
               grace_s: float | None = None, time_scale: float | tuple = 1.0) -> MatchStats:
    """Play both halves and score with the judge's own combine_halves."""
    seed = str(seed)
    h1 = play_half(fa, fb, seed, 1, deals, grace_s, time_scale=time_scale)
    h2 = play_half(fa, fb, seed, 2, deals, grace_s, time_scale=time_scale)
    r = combine_halves(_as_half_result(h1, 1, seed), _as_half_result(h2, 2, seed))
    return MatchStats(r.mean_bb, r.std_bb, r.ci[0], r.ci[1], _VERDICTS[r.verdict],
                      r.points["agent1"], (h1, h2))
