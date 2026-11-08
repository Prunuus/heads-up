"""Layered policy: base = CFR blueprint else heuristic; exploits then modify the base.

PolicyBot.choose_action (the module-level decide) runs, per decision:
0. Opponent model: memory["opp"].observe(obs) first (bot/opponent.py, Task 13);
   on_hand_end feeds it every finished hand with finish(result).
1. Blueprint: only if a blueprint is loaded and memory["config_ok"], meaning this half's
   game (stack, blinds, raise cap) is the one it was trained for. The slot stays empty
   until Task 20.
2. Heuristic: compute_features + heuristic_action, whenever the blueprint gives no answer.
   It reads every size from obs, so it also plays a retuned game.
3. Exploits: bot.exploit.apply_exploits modifies the base action from step 1 or 2,
   from the opponent model's evidence (Task 14).
So base = blueprint else heuristic, and exploits then modify the base. Each layer has
its own try/except and catches any Exception (a corrupt observation, a
missing table, a bug). A layer must answer with a legal Action. Otherwise it counts a
fault in memory["faults"] and the decision moves on: a failed base layer leaves the next
base layer to answer, and a failed exploit keeps the base action. If no layer answers
legally, the result is fallback_action(obs): CHECK_CALL when nothing is owed, else FOLD,
filtered by legal_actions. Five errors forfeit the half, so nothing here raises.

Time (rulings R9, R30a). TimeManager.begin()/end() wrap every choose_action in
try/finally, so the per-request grace applies once per call. on_hand_end work is charged
in full with add(dt). The grace (grace_for_cpu_factor) is sized by a CPU factor, where
1.0 is the dev PC. warmup() times a fixed workload against its dev-PC time. During play,
each preflop heuristic decision is a fixed workload too, and the median of the last 64
is compared with its dev-PC median. The larger of the two factors sizes the grace.

Simulated slow CPU (ruling R30b). sim.match can emulate a CPU s times slower: it charges
s x the measured time minus a grace of 1 ms - 0.25 ms * s, and sends s as
config.time_scale (sim_time_scale; the judge's AgentConfig has none, so s = 1). The bot
then measures what it would on such a CPU: its clock runs s times fast (decisions,
on_hand_end and the online samples, which therefore read s times the factor), and the
warmup factor, measured at this machine's speed, is multiplied by s. So the grace is
1 ms - 0.25 ms * max(s * warmup factor, online factor). Both factors are at least s, so
the bot's grace never exceeds the sim's and its own count stays at or above the sim's
charge, as with the real judge. In-process sims read online factors of about 1-2.5
times s (see PREFLOP_REF_S), so the bot over-counts there: the safe direction.

State (R22). Module state is what warmup() loads: written once, read-only afterwards.
Per-half state lives in memory (tm, time_scale, rng, opp, config_ok, cpu, faults,
exploits: the exploit layer's bot.exploit.STAT_KEYS counters). A PolicyBot holds only
its settings. Never pickle bot.* objects during play: sim.match.load_bot_file runs each
half on a private copy of the bot package, and pickle looks classes up in the caller's
copy. For the same reason every import here sits at module top.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

import evaluator
from bot import equity, exploit, preflop
from bot.exploit import apply_exploits
from bot.features import compute_features
from bot.heuristic import HeuristicParams, heuristic_action
from bot.opponent import OpponentModel
from bot.timing import TimeManager, grace_for_cpu_factor
from game import (BIG_BLIND, NUM_DEALS, RAISE_CAP_PER_STREET, SMALL_BLIND, STARTING_STACK,
                  TIME_BANK_SECONDS, Action, Game, Observation, Street, make_deal)

RNG_SEED = 20261107

# The game a blueprint is trained for (the Task 15 tree uses the default constants).
# config_ok compares the half's config with a loaded blueprint's own meta, or with this
# when no blueprint is loaded.
DEFAULT_META = MappingProxyType({"stack": STARTING_STACK, "sb": SMALL_BLIND,
                                 "bb": BIG_BLIND, "cap": RAISE_CAP_PER_STREET})
_CONFIG_KEYS = (("stack", "starting_stack"), ("sb", "small_blind"), ("bb", "big_blind"),
                ("cap", "raise_cap"))
FAULT_KINDS = ("blueprint", "heuristic", "exploit", "hand_end")

# CPU factor. 1.0 is the dev PC (Ryzen 7 9700X, WSL2, Python 3.12.3), measured 2026-11-07
# on the quiet machine (load ~2), pinned with taskset -c 0-7. Re-measure both constants if
# their workloads change.
# - REFERENCE_WORKLOAD_S: best of REF_RUNS runs of reference_workload, as warmup times it;
#   the median over 12 fresh processes (range 0.331-0.341 ms). With another job on CPUs
#   8-15 (load 8.5) the same measurement read 0.413 ms: a busy PC is a slower CPU.
# - PREFLOP_REF_S: median preflop heuristic-layer time (_heuristic_action, the sample the
#   online estimate takes) in 3 real-judge halves of 1000 deals: judge_engine.py over HTTP
#   to uvicorn agent.py, vs baseline random / calling station / always pot (medians 13.52,
#   14.04, 13.72 us; 15.7-16.1 us at load 12-15). The sample depends on what ran just
#   before it, so it must come from the real server: in-process sims read 13-34 us
#   depending on the opponent's code, and the FastAPI TestClient 60 us. Sims therefore
#   read online factors of about 1-2.5, which only shrinks the grace there (the safe
#   direction).
# F_MAX bounds the damage of a broken measurement: grace_for_cpu_factor(8) = -1 ms per call.
F_MAX = 8.0
REF_REPS = 20             # reference_workload repetitions per timed run (~0.35 ms)
REF_RUNS = 60             # warmup keeps the best of this many runs (~25 ms in all)
REFERENCE_WORKLOAD_S = 0.336e-3
PREFLOP_REF_S = 13.7e-6
WINDOW = 64               # online estimate: median of the last WINDOW preflop samples
MIN_SAMPLES = 16          # ... once there are at least this many

_clock = time.perf_counter


class _ScaledClock:
    """_clock times `scale`: every interval reads `scale` times as long, as on a CPU that
    much slower. Looks _clock up per call."""

    __slots__ = ("scale",)

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def __call__(self) -> float:
        return _clock() * self.scale


def sim_time_scale(config) -> float:
    """The CPU slowdown a simulator emulates for this half (sim.match.SimAgentConfig);
    1.0 for the judge's AgentConfig and for anything not a finite number > 0. Never
    raises."""
    try:
        s = float(getattr(config, "time_scale", 1.0))
    except Exception:
        return 1.0
    return s if 0.0 < s < math.inf else 1.0


# ----------------------------------------------------------------------------- helpers

def fallback_action(obs) -> Action:
    """CHECK_CALL when nothing is owed, else FOLD. If that is not in obs.legal_actions:
    CHECK_CALL, then any legal Action. Never raises."""
    try:
        want = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
    except Exception:
        return Action.CHECK_CALL
    try:
        legal = obs.legal_actions
        if want in legal:
            return want
        if Action.CHECK_CALL in legal:
            return Action.CHECK_CALL
        for a in legal:
            if isinstance(a, Action):
                return a
    except Exception:
        pass
    return want


def config_matches(config, meta) -> bool:
    """True when config is the game meta describes: stack, blinds and raise cap.
    The hand count and the time bank do not change the game. Never raises."""
    try:
        return all(meta[k] == getattr(config, attr) for k, attr in _CONFIG_KEYS)
    except Exception:
        return False


def _legal(action, obs) -> bool:
    try:
        return isinstance(action, Action) and action in obs.legal_actions
    except Exception:
        return False


def _fault(memory, kind: str) -> None:
    try:
        memory["faults"][kind] += 1
    except Exception:
        pass


def _time_manager(memory) -> TimeManager | None:
    try:
        tm = memory["tm"]
    except Exception:
        return None
    return tm if isinstance(tm, TimeManager) else None


def _opponent(memory) -> OpponentModel | None:
    try:
        opp = memory["opp"]
    except Exception:
        return None
    return opp if isinstance(opp, OpponentModel) else None


def _clamp_factor(f: float) -> float:
    """f floored at 1.0 and clamped to F_MAX; 1.0 (no adjustment) when f is not a number."""
    if f != f:
        return 1.0
    return min(max(f, 1.0), F_MAX)


# ------------------------------------------------------------------------- CPU factor

def reference_workload(payload: str, reps: int = REF_REPS) -> None:
    """A fixed stand-in for the server plumbing the bot cannot time: JSON in,
    Observation.from_dict, to_dict, JSON out. It uses no data file."""
    for _ in range(reps):
        obs = Observation.from_dict(json.loads(payload)["observation"])
        json.dumps({"observation": obs.to_dict(), "hand_end_errors": []})


def measure_cpu_factor(reference_s: float | None = None, runs: int = REF_RUNS) -> float:
    """Best of `runs` timings of reference_workload over its dev-PC time
    (REFERENCE_WORKLOAD_S), floored at 1.0 and clamped to F_MAX. 1.0 if anything goes
    wrong. Never raises."""
    ref = REFERENCE_WORKLOAD_S if reference_s is None else reference_s
    try:
        payload = json.dumps({"observation": _warmup_spots()[-1].to_dict(), "hand_results": []})
        best = math.inf
        for _ in range(runs):
            t0 = _clock()
            reference_workload(payload)
            best = min(best, _clock() - t0)
        return _clamp_factor(best / ref) if best < math.inf else 1.0
    except Exception:
        return 1.0


class CpuFactorEstimator:
    """Online CPU factor: the median of the last WINDOW preflop heuristic-layer times over
    their dev-PC median ref_s (default PREFLOP_REF_S), floored at 1.0 and clamped to F_MAX.

    Preemption spikes cannot move a median. A sample that is not a finite positive number
    is dropped. factor() is None until there are MIN_SAMPLES samples, or if ref_s is
    unusable. Never raises.
    """

    __slots__ = ("ref_s", "_samples")

    def __init__(self, ref_s: float | None = None) -> None:
        self.ref_s = PREFLOP_REF_S if ref_s is None else ref_s
        self._samples: deque = deque(maxlen=WINDOW)

    @property
    def n(self) -> int:
        """Samples in the window."""
        return len(self._samples)

    def add(self, dt) -> None:
        try:
            v = float(dt)
        except (TypeError, ValueError, OverflowError):
            return
        if 0.0 < v < math.inf:
            self._samples.append(v)

    def factor(self) -> float | None:
        try:
            s = self._samples
            ref = float(self.ref_s)
            if len(s) < MIN_SAMPLES or not 0.0 < ref < math.inf:
                return None
            return _clamp_factor(sorted(s)[len(s) // 2] / ref)
        except Exception:
            return None


# ----------------------------------------------------------------------------- warmup

@dataclass(frozen=True)
class Warmup:
    """What warmup() did: its duration, the measured CPU factor, the blueprint it loaded
    (None for none), and the names of the steps that failed."""

    seconds: float
    cpu_factor: float
    blueprint: object
    failed: tuple


WARMUP: Warmup | None = None


def _warmup_spots() -> list:
    """Every decision of one fixed hand played down to showdown. The first actor bets half
    pot on each postflop street, so all four streets and facing a bet are covered."""
    g = Game(make_deal("policy-warmup", 0), 0)
    spots, seen = [], set()
    while not g.is_terminal:
        obs = g.observation(g.to_act)
        spots.append(obs)
        bet = (obs.street not in seen and obs.street != Street.PREFLOP
               and Action.RAISE_HALF in obs.legal_actions)
        seen.add(obs.street)
        g.apply(Action.RAISE_HALF if bet else Action.CHECK_CALL)
    return spots


def _touch_preflop() -> None:
    preflop.preflop_strength((0, 4, 8, 12, 16))   # loads the table on first use


def _touch_nut_flags() -> None:
    # The first call per board width builds index caches, and the first np.unique call
    # costs 3-4 ms on the dev PC.
    hole = (0, 4, 8, 12, 16)
    for w in (3, 4, 5):
        equity.nut_flags(hole, tuple(range(20, 20 + w)))


def _dummy_decisions() -> None:
    rng = np.random.default_rng(RNG_SEED)
    params = HeuristicParams()
    for obs in _warmup_spots():
        for iters in (0, 16):
            heuristic_action(obs, compute_features(obs, iters, rng), params, rng)


def _load_blueprint():
    """The blueprint for the blueprint layer: Task 20 loads data/blueprint.npz here
    (bot.strategy.Blueprint, with .meta), and gives None if it is missing or unreadable.
    No blueprint exists yet."""
    return None


def warmup() -> None:
    """Load every table and touch every lazy cache before the first request. The first
    call in a process does the work; later calls return at once. Never raises: a failed
    step is listed in WARMUP.failed, and the bot plays on without it (the heuristic has
    defaults for a missing table)."""
    global WARMUP
    if WARMUP is not None:
        return
    t0 = _clock()
    failed = []

    def step(name, fn):
        try:
            return fn()
        except Exception:
            failed.append(name)
            return None

    step("evaluator", evaluator.warmup)       # omaha_score's table (showdown scoring)
    step("equity", equity.warmup)             # fasteval's table + the zero-budget reference
    step("preflop", _touch_preflop)
    step("nut_flags", _touch_nut_flags)
    blueprint = step("blueprint", _load_blueprint)
    step("decisions", _dummy_decisions)       # one compute_features + heuristic per spot
    factor = measure_cpu_factor()             # last: everything above is warm by now
    WARMUP = Warmup(_clock() - t0, factor, blueprint, tuple(failed))


# -------------------------------------------------------------------------- PolicyBot

class PolicyBot:
    """The shipped bot. BotLike: new_half / choose_action / on_hand_end.

    params         heuristic thresholds (default HeuristicParams())
    use_blueprint  consult the blueprint when one is loaded and the game matches it
    use_exploits   run the exploit layer on the base action

    The class itself and functools.partial(PolicyBot, ...) are picklable factories. The
    instance keeps settings only; everything per half lives in memory.
    """

    def __init__(self, params: HeuristicParams | None = None, use_blueprint: bool = True,
                 use_exploits: bool = True) -> None:
        warmup()                  # once per process (a sim factory call is untimed)
        w = WARMUP
        self.params = params if params is not None else HeuristicParams()
        self.use_blueprint = bool(use_blueprint)
        self.use_exploits = bool(use_exploits)
        self.blueprint = w.blueprint if self.use_blueprint and w is not None else None
        self.cpu_factor = w.cpu_factor if w is not None else 1.0

    def new_half(self, config, memory: dict) -> None:
        """Fresh per-half state. The same keys every time. Never raises.
        config.time_scale (a simulator's emulated slow CPU) scales the bot's clock and
        warmup factor (module docstring)."""
        try:
            bank, hands = config.time_bank_seconds, config.num_hands
        except Exception:
            bank, hands = TIME_BANK_SECONDS, NUM_DEALS
        try:
            cap, bb = config.raise_cap, config.big_blind
        except Exception:
            cap, bb = RAISE_CAP_PER_STREET, BIG_BLIND
        try:
            meta = DEFAULT_META if self.blueprint is None else self.blueprint.meta
        except Exception:
            meta = None
        scale = sim_time_scale(config)
        try:
            memory["tm"] = TimeManager(bank_s=bank, hands=hands,
                                       clock=_clock if scale == 1.0 else _ScaledClock(scale),
                                       grace_s=grace_for_cpu_factor(self.cpu_factor * scale))
            memory["time_scale"] = scale
            memory["rng"] = np.random.default_rng(RNG_SEED)
            memory["opp"] = OpponentModel(raise_cap=cap, big_blind=bb)   # bad values: defaults
            memory["config_ok"] = config_matches(config, meta)
            memory["cpu"] = CpuFactorEstimator()
            memory["faults"] = dict.fromkeys(FAULT_KINDS, 0)
            memory["exploits"] = dict.fromkeys(exploit.STAT_KEYS, 0)
        except Exception:
            pass

    def choose_action(self, obs, memory: dict) -> Action:
        """The layered decision (module docstring). Always a legal Action."""
        tm = _time_manager(memory)
        if tm is not None:
            tm.begin()
        try:
            return self._decide(obs, memory, tm)
        finally:
            if tm is not None:
                tm.end()

    def on_hand_end(self, result, memory: dict) -> None:
        """Feed the result to the opponent model and charge the time in full with add(dt)
        (R9). Never raises: each raise would count as its own error."""
        try:
            tm = _time_manager(memory)
            clock = tm.clock if tm is not None else _clock
            t0 = clock()
            opp = _opponent(memory)
            if opp is not None:
                opp.finish(result)          # never raises
            if tm is not None:
                tm.add(clock() - t0)
        except Exception:
            _fault(memory, "hand_end")

    # ------------------------------------------------------------------- the layers

    def _decide(self, obs, memory, tm) -> Action:
        try:
            opp = _opponent(memory)
            if opp is not None:
                opp.observe(obs)            # first: the layers read the updated model
        except Exception:                   # observe() never raises; belt and braces
            pass
        base = feats = None
        try:
            config_ok = memory["config_ok"] is True
        except Exception:
            config_ok = False
        if self.blueprint is not None and config_ok:
            try:
                base = self._blueprint_action(obs, memory)
                if base is not None and not _legal(base, obs):
                    _fault(memory, "blueprint")
                    base = None
            except Exception:
                _fault(memory, "blueprint")
                base = None
        if base is None:
            try:
                feats, base = self._heuristic_action(obs, memory, tm)
                if not _legal(base, obs):
                    _fault(memory, "heuristic")
                    base = None
            except Exception:
                _fault(memory, "heuristic")
                base = None
        action = base
        if base is not None and self.use_exploits:
            try:
                a = self._exploit_action(base, obs, feats, memory)
                if _legal(a, obs):
                    action = a
                else:
                    _fault(memory, "exploit")
            except Exception:
                _fault(memory, "exploit")
        return action if _legal(action, obs) else fallback_action(obs)

    def _blueprint_action(self, obs, memory):
        """Blueprint layer slot. Task 20 fills it: node_of -> bucket -> Blueprint.act.
        Returns a legal Action, or None for no answer (the heuristic then plays)."""
        return None

    def _heuristic_action(self, obs, memory, tm):
        """(features, action). The time manager sizes the Monte Carlo. Preflop the work is
        fixed (no Monte Carlo), so its time is also the online CPU-factor sample."""
        clock = tm.clock if tm is not None else _clock
        t0 = clock()
        rng = memory["rng"]
        iters = tm.iters(obs.pot, obs.street, obs.hand_index) if tm is not None else 0
        feats = compute_features(obs, iters, rng)
        action = heuristic_action(obs, feats, self.params, rng, memory["opp"])
        if feats.street == Street.PREFLOP:
            self._observe_cpu(memory, tm, clock() - t0)
        return feats, action

    def _exploit_action(self, base, obs, feats, memory):
        """Exploit layer: bot.exploit.apply_exploits on the base, with this bot's heuristic
        params and the half's counters. feats is None when the base came from the
        blueprint."""
        return apply_exploits(base, obs, feats, _opponent(memory), memory["rng"], self.params,
                              memory.get("exploits"))

    def _observe_cpu(self, memory, tm, dt: float) -> None:
        """Add an online sample and resize the grace: the larger of the (time-scaled)
        warmup and online factors. Never raises, so it can never cost the heuristic its
        answer."""
        try:
            est = memory["cpu"]
            est.add(dt)
            f = est.factor()
            if f is not None and tm is not None:
                warm = self.cpu_factor * memory.get("time_scale", 1.0)
                tm.grace_s = grace_for_cpu_factor(max(warm, f))
        except Exception:
            pass


# ------------------------------------------------- the plan's module-level interface

_DEFAULT_BOT: PolicyBot | None = None


def _default_bot() -> PolicyBot:
    global _DEFAULT_BOT
    if _DEFAULT_BOT is None:
        _DEFAULT_BOT = PolicyBot()
    return _DEFAULT_BOT


def start_half(config, memory: dict) -> None:
    """PolicyBot().new_half: memory gets tm, rng, opp, config_ok (plus cpu and faults)."""
    _default_bot().new_half(config, memory)


def decide(obs, memory: dict) -> Action:
    """PolicyBot().choose_action. Always a legal Action."""
    try:
        return _default_bot().choose_action(obs, memory)
    except Exception:
        return fallback_action(obs)


def hand_finished(result, memory: dict) -> None:
    """PolicyBot().on_hand_end. Never raises."""
    try:
        _default_bot().on_hand_end(result, memory)
    except Exception:
        pass
