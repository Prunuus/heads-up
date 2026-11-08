"""The layered policy (bot/policy.py) and its wiring into agent.py."""

import dataclasses
import functools
import importlib.util
import json
import math
import pickle
import random
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

import game
from bot import policy
from bot.heuristic import HeuristicParams
from bot.opponent import OpponentModel
from bot.timing import TimeManager, grace_for_cpu_factor
from game import Action, AgentConfig, Game, Street, make_deal
from sim.match import REPO_ROOT, load_bot_file, play_half
from sim.opponents import ZOO
from sim.tournament import resolve_bot

FALLBACK_STREETS = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)

# Warm up first, as agent.py does at import: tests below monkeypatch the layers, and
# the one warmup a process runs must not see those patches.
policy.warmup()


def _observations(seed="pol", hands=40):
    """Observations from real hands (both seats) with a raise-happy random mix of actions,
    so every street, facing bets, the raise cap and all-ins all show up."""
    rng = random.Random(seed)
    out = []
    for h in range(hands):
        g = Game(make_deal(seed, h), h % 2, hand_index=h)
        while not g.is_terminal:
            obs = g.observation(g.to_act)
            out.append(obs)
            legal = obs.legal_actions
            u = rng.random()
            raises = [a for a in legal if a in (Action.RAISE_HALF, Action.RAISE_POT)]
            if raises and u < 0.35:
                g.apply(rng.choice(raises))
            elif Action.FOLD in legal and u > 0.93:
                g.apply(Action.FOLD)
            else:
                g.apply(Action.CHECK_CALL)
    return out


def _expected_fallback(obs):
    return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD


def _fresh_bot(config=None, **kw):
    bot, memory = policy.PolicyBot(**kw), {}
    bot.new_half(config or AgentConfig(), memory)
    return bot, memory


# ------------------------------------------------------------- the plan's four tests

def test_layer_exception_falls_back_legally(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("heuristic failure")

    monkeypatch.setattr(policy, "heuristic_action", boom)
    bot, memory = _fresh_bot()
    obs_all = _observations()
    assert {o.street for o in obs_all} == set(FALLBACK_STREETS)
    for obs in obs_all:
        a = bot.choose_action(obs, memory)
        assert a in obs.legal_actions and a == _expected_fallback(obs)
    assert memory["faults"]["heuristic"] == len(obs_all)
    assert memory["tm"]._t0 is None                 # end() ran although the layer raised
    for obs in obs_all[:20]:                        # the module-level wrapper, same path
        memory = {}
        policy.start_half(AgentConfig(), memory)
        assert policy.decide(obs, memory) == _expected_fallback(obs)


def test_config_mismatch_uses_heuristic(monkeypatch):
    # Review Focus 1: a retuned game (here a 100-chip stack) must never reach the
    # blueprint; the heuristic, which reads sizes only from obs, plays instead.
    class FakeBlueprint:
        meta = {"stack": 80, "sb": 1, "bb": 2, "cap": 4, "buckets": 200, "iterations": 1}

    bot = policy.PolicyBot()
    bot.blueprint = FakeBlueprint()
    consulted = []

    def spy(obs, memory):
        consulted.append(obs)
        return None                                 # "no answer": fall through

    monkeypatch.setattr(bot, "_blueprint_action", spy)

    memory = {}
    bot.new_half(AgentConfig(), memory)             # the blueprint's own game
    assert memory["config_ok"] is True
    obs = _observations("cm0", 1)[0]
    assert bot.choose_action(obs, memory) in obs.legal_actions
    assert len(consulted) == 1                      # the slot is live when config matches

    consulted.clear()
    monkeypatch.setattr(game, "STARTING_STACK", 100)
    memory = {}
    bot.new_half(AgentConfig(starting_stack=100), memory)
    assert memory["config_ok"] is False
    obs_all = _observations("cm1", 40)
    assert max(o.my_stack + o.my_bet for o in obs_all) > 80     # really a 100-chip game
    for obs in obs_all:
        assert bot.choose_action(obs, memory) in obs.legal_actions
    assert consulted == []
    assert sum(memory["faults"].values()) == 0      # the heuristic answered every time


def test_warmup_under_6s():
    # A cold process, like the tournament's: import agent.py (evaluator table, our tables,
    # dummy decisions and the CPU measurement all run at import).
    code = ("import time; t0 = time.perf_counter(); import agent; "
            "from bot import policy; "
            "print(time.perf_counter() - t0, policy.WARMUP.seconds, policy.WARMUP.cpu_factor)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    total, warm, factor = map(float, r.stdout.split()[-3:])
    assert total < 6.0 and warm < 6.0
    assert 1.0 <= factor <= policy.F_MAX


class _Recorder:
    """agent.py as the sim loads it (a fresh bot-package copy), keeping each half's memory."""

    def __init__(self, memories):
        self.mod = load_bot_file("agent.py")
        self.memories = memories

    def new_half(self, config, memory):
        self.mod.new_half(config, memory)
        self.memories.append(memory)

    def choose_action(self, obs, memory):
        return self.mod.choose_action(obs, memory)

    def on_hand_end(self, result, memory):
        return self.mod.on_hand_end(result, memory)


def _assert_clean_half(name, i, deals):
    memories = []
    h = play_half(lambda: _Recorder(memories), ZOO[name], f"zoo-{name}", 1 + i % 2, deals=deals)
    assert h.errors_a == 0 and not h.forfeit_a, (name, h.errors_a)
    assert h.bank_a < 3.0, (name, h.bank_a)
    assert h.calls_a > deals                        # it really played
    (memory,) = memories
    assert sum(memory["faults"].values()) == 0, (name, memory["faults"])


@pytest.mark.slow
@pytest.mark.parametrize("i,name", list(enumerate(sorted(ZOO))))
def test_zero_errors_vs_whole_zoo(i, name):
    _assert_clean_half(name, i, deals=200)


SMOKE = ("random", "always_pot", "maniac", "nit", "llm_simple_rules")


@pytest.mark.parametrize("i,name", list(enumerate(SMOKE)))
def test_zero_errors_smoke(i, name):
    _assert_clean_half(name, i, deals=25)


# ------------------------------------------------------------------- more layers

def test_illegal_or_odd_layer_results_are_replaced(monkeypatch):
    obs_all = _observations("odd", 15)
    monkeypatch.setattr(policy, "heuristic_action", lambda *a, **k: Action.RAISE_POT)
    bot, memory = _fresh_bot()
    for obs in obs_all:                             # an Action, legal only sometimes
        want = Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else _expected_fallback(obs)
        assert bot.choose_action(obs, memory) is want
    assert 0 < memory["faults"]["heuristic"] < len(obs_all)
    for odd in ("RAISE_POT", 3, None, 7.5):         # not an Action at all: always replaced
        monkeypatch.setattr(policy, "heuristic_action", lambda *a, odd=odd, **k: odd)
        bot, memory = _fresh_bot()
        for obs in obs_all:
            assert bot.choose_action(obs, memory) is _expected_fallback(obs)
        assert memory["faults"]["heuristic"] == len(obs_all)


def test_exploit_and_blueprint_faults_keep_the_heuristic(monkeypatch):
    class FakeBlueprint:
        meta = dict(policy.DEFAULT_META)

    def boom(*a, **k):
        raise ValueError("layer failure")

    bot = policy.PolicyBot()
    bot.blueprint = FakeBlueprint()
    monkeypatch.setattr(bot, "_blueprint_action", boom)
    monkeypatch.setattr(bot, "_exploit_action", boom)
    memory = {}
    bot.new_half(AgentConfig(), memory)
    obs_all = _observations("exp", 20)
    for obs in obs_all:
        assert bot.choose_action(obs, memory) in obs.legal_actions
    f = memory["faults"]
    assert f["blueprint"] == f["exploit"] == len(obs_all) and f["heuristic"] == 0


def test_a_legal_exploit_change_replaces_the_base(monkeypatch):
    # T11 M3, the positive path: a legal answer from the exploit layer is played. An
    # illegal one is a fault, and the base it was given stands.
    bot, memory = _fresh_bot()
    seen = []

    def change(base, obs, feats, memory):
        other = [a for a in obs.legal_actions if a != base]
        seen.append((base, other[-1] if other else base))
        return seen[-1][1]

    monkeypatch.setattr(bot, "_exploit_action", change)
    obs_all = _observations("pos", 20)
    for obs in obs_all:
        assert bot.choose_action(obs, memory) == seen[-1][1]
    assert sum(b != a for b, a in seen) > len(obs_all) // 2
    assert memory["faults"]["exploit"] == 0

    bases = []

    def illegal(base, obs, feats, memory):
        bases.append(base)
        return Action.FOLD if obs.to_call == 0 else 99     # FOLD when free; not an Action

    monkeypatch.setattr(bot, "_exploit_action", illegal)
    for obs in obs_all:
        assert bot.choose_action(obs, memory) == bases[-1]
    assert memory["faults"]["exploit"] == len(obs_all)


def test_corrupt_observation_falls_through():
    bot, memory = _fresh_bot()
    obs = next(o for o in _observations("bad", 10) if o.street == Street.FLOP)
    for bad in (dataclasses.replace(obs, board=obs.board[:2]),        # ValueError
                dataclasses.replace(obs, hole_cards=(0, 1, 2, 3, 99)),  # IndexError
                dataclasses.replace(obs, action_history=None)):         # TypeError
        assert bot.choose_action(bad, memory) == _expected_fallback(obs)
    assert memory["faults"]["heuristic"] == 3


def test_fallback_action_is_filtered_by_legal():
    obs = _observations("fb", 1)[0]
    assert policy.fallback_action(obs) == _expected_fallback(obs)
    free = dataclasses.replace(obs, to_call=0, legal_actions=(Action.CHECK_CALL,))
    assert policy.fallback_action(free) == Action.CHECK_CALL
    odd = dataclasses.replace(obs, to_call=3, legal_actions=(Action.CHECK_CALL,))
    assert policy.fallback_action(odd) == Action.CHECK_CALL     # FOLD is not legal there
    assert policy.fallback_action(None) == Action.CHECK_CALL     # never raises


def test_hand_finished_never_raises_and_charges_time():
    bot, memory = _fresh_bot()
    g = Game(make_deal("hf", 0), 0)
    while not g.is_terminal:
        g.apply(Action.CHECK_CALL)
    tm = memory["tm"]
    before = tm.used
    for r in (g.hand_result(0), g.hand_result(1), None, {"junk": 1}, 42):
        assert bot.on_hand_end(r, memory) is None
        assert policy.hand_finished(r, memory) is None
    assert tm.used > before                         # add(dt), no grace
    for broken in ({}, {"tm": "not a time manager"}, None):
        assert bot.on_hand_end(g.hand_result(0), broken) is None
        assert policy.hand_finished(g.hand_result(0), broken) is None


def test_decide_without_start_half_is_still_legal():
    bot = policy.PolicyBot()
    for obs in _observations("nostart", 5):
        for memory in ({}, {"tm": None, "rng": "x"}):
            assert bot.choose_action(obs, memory) in obs.legal_actions


# ------------------------------------------------------------- start_half / config

def test_start_half_sets_the_planned_memory():
    bot, memory = _fresh_bot(AgentConfig(num_hands=500, time_bank_seconds=4.0))
    assert {"tm", "rng", "opp", "config_ok"} <= memory.keys()
    tm = memory["tm"]
    assert isinstance(tm, TimeManager) and tm.bank_s == 4.0 and tm.hands == 500
    assert tm.grace_s == pytest.approx(grace_for_cpu_factor(bot.cpu_factor))
    assert isinstance(memory["opp"], OpponentModel)
    assert memory["config_ok"] is True              # num_hands / bank do not change the game
    a, b = {}, {}
    bot.new_half(AgentConfig(), a)
    bot.new_half(AgentConfig(), b)
    assert a["rng"].random() == b["rng"].random()   # fixed seed
    assert a.keys() == b.keys()


@pytest.mark.parametrize("change", [{"starting_stack": 100}, {"big_blind": 4},
                                    {"small_blind": 2}, {"raise_cap": 3}])
def test_config_ok_is_false_for_any_game_change(change):
    _, memory = _fresh_bot(AgentConfig(**change))   # no blueprint loaded: game defaults
    assert memory["config_ok"] is False


def test_config_matches_never_raises():
    meta = dict(policy.DEFAULT_META)
    assert policy.config_matches(AgentConfig(), meta) is True
    assert policy.config_matches(AgentConfig(), {}) is False
    assert policy.config_matches(None, meta) is False
    assert policy.config_matches(AgentConfig(), None) is False
    bot = policy.PolicyBot()
    memory = {}
    bot.new_half(None, memory)                      # garbage config: defaults, no raise
    assert memory["config_ok"] is False and isinstance(memory["tm"], TimeManager)


def test_policy_bot_factories_are_picklable():
    p = HeuristicParams(bluff_freq=0.0)
    fac = pickle.loads(pickle.dumps(functools.partial(policy.PolicyBot, p, use_exploits=False)))
    b = fac()
    assert b.params == p and b.use_exploits is False and b.blueprint is None
    assert pickle.loads(pickle.dumps(policy.PolicyBot)) is policy.PolicyBot
    assert policy.PolicyBot().params == HeuristicParams()
    assert policy.PolicyBot(use_blueprint=False).blueprint is None


# ---------------------------------------------------------------- CPU factor (R30a)

def test_cpu_factor_measurement_floors_and_clamps():
    assert policy.measure_cpu_factor(reference_s=1e9) == 1.0
    assert policy.measure_cpu_factor(reference_s=1e-12) == policy.F_MAX
    assert policy.measure_cpu_factor(reference_s=float("nan")) == 1.0
    assert policy.measure_cpu_factor(reference_s=0.0) == 1.0
    assert policy.measure_cpu_factor(runs=0) == 1.0           # nothing measured: no change
    assert 1.0 <= policy.WARMUP.cpu_factor <= policy.F_MAX


def test_online_estimator_is_robust():
    ref = 1e-5
    est = policy.CpuFactorEstimator(ref_s=ref)
    assert est.factor() is None
    for bad in (0.0, -0.0, -1.0, float("nan"), float("inf"), -float("inf"), None, "x", [1]):
        est.add(bad)                                # ignored, never raises
    assert est.n == 0 and est.factor() is None
    for _ in range(policy.MIN_SAMPLES - 1):
        est.add(3 * ref)
    assert est.factor() is None                     # not enough evidence yet
    est.add(3 * ref)
    assert est.factor() == pytest.approx(3.0)
    for i in range(60):                             # a third of the samples are spikes
        est.add(3 * ref)
        if i % 2:
            est.add(1000 * ref)
    assert est.factor() == pytest.approx(3.0)
    for _ in range(200):
        est.add(0.4 * ref)                          # faster than the dev PC: floored
    assert est.factor() == 1.0                      # and the old samples are forgotten
    for _ in range(200):
        est.add(1e6 * ref)
    assert est.factor() == policy.F_MAX             # clamped
    assert policy.CpuFactorEstimator(ref_s=float("nan")).factor() is None
    bad_ref = policy.CpuFactorEstimator(ref_s=0.0)
    for _ in range(50):
        bad_ref.add(1e-4)
    assert bad_ref.factor() is None                 # a broken reference never yields a factor


def test_online_estimate_drives_the_grace(monkeypatch):
    # Pretend the dev PC were 1000x faster: every preflop sample reads as a very slow CPU.
    monkeypatch.setattr(policy, "PREFLOP_REF_S", 1e-9)
    bot, memory = _fresh_bot()
    bot.cpu_factor = 1.0
    tm = memory["tm"]
    pre = [o for o in _observations("cpu", 60) if o.street == Street.PREFLOP]
    assert len(pre) > policy.MIN_SAMPLES
    for obs in pre:
        bot.choose_action(obs, memory)
    assert memory["cpu"].n == min(len(pre), policy.WINDOW)
    assert tm.grace_s == pytest.approx(grace_for_cpu_factor(policy.F_MAX))
    # The other way round: online reads faster than warmup, so warmup's factor holds.
    monkeypatch.setattr(policy, "PREFLOP_REF_S", 1e3)
    bot, memory = _fresh_bot()
    bot.cpu_factor = 2.5
    memory["tm"].grace_s = grace_for_cpu_factor(2.5)
    for obs in pre:
        bot.choose_action(obs, memory)
    assert memory["tm"].grace_s == pytest.approx(grace_for_cpu_factor(2.5))


# ------------------------------------------------- sim time_scale (ruling R30b)

class _StepClock:
    """A fake perf_counter that advances `step` seconds per call."""

    def __init__(self, step):
        self.t, self.step = 1000.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def test_sim_time_scale_is_read_from_the_config():
    from sim.match import SimAgentConfig
    assert policy.sim_time_scale(SimAgentConfig(time_scale=2.5)) == 2.5
    assert policy.sim_time_scale(AgentConfig()) == 1.0         # the real judge's config
    for bad in (0.0, -1.0, float("nan"), float("inf"), "x", None):
        assert policy.sim_time_scale(SimAgentConfig(time_scale=bad)) == 1.0
    assert policy.sim_time_scale(None) == 1.0


def test_sim_time_scale_scales_the_clock_and_the_warmup_factor(monkeypatch):
    from sim.match import SimAgentConfig
    monkeypatch.setattr(policy, "_clock", _StepClock(0.001))
    bot = policy.PolicyBot()
    bot.cpu_factor = 1.2
    memory = {}
    bot.new_half(SimAgentConfig(time_scale=2.5), memory)
    tm = memory["tm"]
    assert memory["time_scale"] == 2.5
    assert tm.grace_s == pytest.approx(grace_for_cpu_factor(1.2 * 2.5))
    assert tm.clock() - tm.clock() == pytest.approx(-0.0025)  # each 1 ms step reads 2.5 ms
    tm.used = 0.0
    bot.on_hand_end(None, memory)                     # charged in full: one scaled step
    assert tm.used == pytest.approx(0.0025)
    # The judge's plain AgentConfig: this machine's clock and the warmup factor as is.
    memory = {}
    bot.new_half(AgentConfig(), memory)
    assert memory["time_scale"] == 1.0
    assert memory["tm"].grace_s == pytest.approx(grace_for_cpu_factor(1.2))
    assert memory["tm"].clock is policy._clock


@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_sim_time_scale_reaches_the_online_estimate(monkeypatch, scale):
    from sim.match import SimAgentConfig
    step = 20e-6             # each timed preflop decision reads 20 us x scale
    monkeypatch.setattr(policy, "_clock", _StepClock(step))
    bot = policy.PolicyBot()
    bot.cpu_factor = 1.0
    memory = {}
    bot.new_half(SimAgentConfig(time_scale=scale), memory)
    pre = [o for o in _observations("cpu", 60) if o.street == Street.PREFLOP]
    for obs in pre:
        bot.choose_action(obs, memory)
    online = step * scale / policy.PREFLOP_REF_S
    assert memory["cpu"].factor() == pytest.approx(online)
    assert memory["tm"].grace_s == pytest.approx(grace_for_cpu_factor(max(scale, online)))


def test_in_a_scaled_sim_the_bots_count_covers_the_charge():
    # The T9 invariant under emulation: the bot's grace never exceeds the sim's, so its
    # own count stays at or above what the sim charges.
    from sim.match import play_half
    seen = []

    class Rec(policy.PolicyBot):
        def new_half(self, config, memory):
            super().new_half(config, memory)
            seen.append(memory)

    for scale in (1.0, 3.0):
        seen.clear()
        h = play_half(Rec, ZOO["calling_station"], "ts", 1, deals=40, time_scale=scale)
        assert h.errors_a == 0 and seen[0]["time_scale"] == scale
        assert h.bank_a > 0.0 or scale == 1.0           # 3x: grace 0.25 ms, so it charges
        assert seen[0]["tm"].used >= 0.95 * h.bank_a


# ------------------------------------------------------------------------ warmup

def test_warmup_is_idempotent_and_recorded():
    policy.warmup()
    w = policy.WARMUP
    policy.warmup()
    assert policy.WARMUP is w                       # the second call does nothing
    assert w.failed == () and w.seconds < 6.0 and w.blueprint is None


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_missing_or_corrupt_table_never_crashes(tmp_path, monkeypatch, damage):
    from bot import preflop
    path = tmp_path / "preflop.npz"
    if damage == "corrupt":
        path.write_bytes((REPO_ROOT / "data" / "preflop.npz").read_bytes()[:1000])
    monkeypatch.setattr(preflop, "_DEFAULT_PATH", path)
    monkeypatch.setattr(preflop, "_MASKS", None)
    monkeypatch.setattr(preflop, "_STRENGTH", None)
    monkeypatch.setattr(policy, "WARMUP", None)
    policy.warmup()                                 # never raises
    assert "preflop" in policy.WARMUP.failed
    bot, memory = _fresh_bot()
    for obs in _observations("dmg", 30):
        assert bot.choose_action(obs, memory) in obs.legal_actions
    assert sum(memory["faults"].values()) == 0      # the heuristic copes (strength 0.5)


def test_a_failing_warmup_step_does_not_stop_the_rest(monkeypatch):
    from bot import equity

    def boom():
        raise RuntimeError("table build failed")

    monkeypatch.setattr(equity, "warmup", boom)
    monkeypatch.setattr(policy, "WARMUP", None)
    policy.warmup()
    assert "equity" in policy.WARMUP.failed
    assert 1.0 <= policy.WARMUP.cpu_factor <= policy.F_MAX


_FIRST_DECISION_PROBE = r"""
import json, statistics, time
from bot import policy
from bot.timing import TimeManager
from game import Action, AgentConfig, Game, make_deal
policy.warmup()
bot = policy.PolicyBot()
g = Game(make_deal("first-decision", 3), 0)
spots = []
while not g.is_terminal and len(spots) < 4:
    obs = g.observation(g.to_act)
    if not spots or spots[-1].street != obs.street:
        spots.append(obs)
    g.apply(Action.CHECK_CALL)
out = {}
memory = {}
bot.new_half(AgentConfig(), memory)
for obs in spots:                       # streets in play order, like one real hand
    t = []
    for _ in range(25):
        memory["tm"] = TimeManager()    # same iters every call: the pace starts fresh
        t0 = time.perf_counter()
        bot.choose_action(obs, memory)
        t.append(time.perf_counter() - t0)
    out[obs.street.name] = [1e3 * t[0], 1e3 * statistics.median(t[1:])]
print(json.dumps(out))
"""


def test_first_decision_per_street_costs_the_steady_state():
    # Warmup must touch every lazy cache: in a cold process the first decision on each
    # street is as fast as the ones after it. A lazy table or cache costs 3-15 ms every
    # cold start; a preemption spike does not repeat, so three cold starts may be tried.
    runs = []
    for _ in range(3):
        r = subprocess.run([sys.executable, "-c", _FIRST_DECISION_PROBE], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        res = json.loads(r.stdout.splitlines()[-1])
        runs.append(res)
        assert set(res) == {s.name for s in FALLBACK_STREETS}
        if all(first <= 2.0 * steady + 1.0 for first, steady in res.values()):
            return
    pytest.fail(f"first decision per street much slower than steady state (ms): {runs}")


# -------------------------------------------------------------- agent.py end to end

def _load_agent():
    spec = importlib.util.spec_from_file_location("agent_policy_test", REPO_ROOT / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agent_end_to_end_through_the_routes():
    module = _load_agent()
    assert isinstance(module.BOT, policy.PolicyBot)
    client = TestClient(module.app)
    body = {"half": 1, "seat": 0, "config": AgentConfig().to_dict()}
    assert client.post("/new-half", json=body).status_code == 200
    memory = module.MEMORY
    assert memory["config_ok"] is True and isinstance(memory["tm"], TimeManager)
    rng = random.Random(7)
    pending = []
    for hand in range(40):
        g = Game(make_deal("e2e", hand), hand % 2, hand_index=hand)
        while not g.is_terminal:
            obs = g.observation(g.to_act)
            if g.to_act == 0:
                r = client.post("/act", json={"observation": obs.to_dict(),
                                              "hand_results": pending})
                pending = []
                assert r.status_code == 200, r.json()
                assert r.json()["hand_end_errors"] == []
                a = Action[r.json()["action"]]
                assert a in obs.legal_actions
            else:
                a = rng.choice(obs.legal_actions)
            g.apply(a)
        pending.append(g.hand_result(0).to_dict())
    # The end-of-half batch: everything still pending plus 20 hands at once, including
    # hands we never acted in (the button folds first), splits and showdowns.
    for hand in range(40, 60):
        g = Game(make_deal("e2e", hand), 1 if hand % 3 == 0 else hand % 2, hand_index=hand)
        while not g.is_terminal:
            g.apply(Action.FOLD if hand % 3 == 0 else Action.CHECK_CALL)
        pending.append(g.hand_result(0).to_dict())
    r = client.post("/hand-end", json={"results": pending})
    assert r.status_code == 200 and r.json()["hand_end_errors"] == []
    assert memory["tm"].used >= 0.0 and sum(memory["faults"].values()) == 0
    body = {"half": 2, "seat": 1, "config": AgentConfig(starting_stack=100).to_dict()}
    assert client.post("/new-half", json=body).status_code == 200
    assert module.MEMORY["config_ok"] is False


def test_candidate_agent_resolves_and_pickles():
    fac = pickle.loads(pickle.dumps(resolve_bot("agent")))
    mod = fac()
    assert isinstance(mod.BOT, mod.policy.PolicyBot)
    h = play_half(fac, ZOO["calling_station"], "cand", 2, deals=20)
    assert h.errors_a == 0 and not h.forfeit_a
