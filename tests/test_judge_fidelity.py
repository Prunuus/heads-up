"""sim.match against the organizer's judge_engine, on the same bots and deals (audit I4).

Default suite: judge_engine.play_half and combine_halves run unchanged. Only
HttpAgentHandle._request is replaced, by a call into agent.py's own FastAPI app in
this process (fastapi.testclient). So the judge's timing, error, forfeit, bank and
hand-result logic and agent.py's server plumbing both run for real, and every number
the simulator reports is compared with the judge's.

Slow (-m slow): real uvicorn servers on free ports, with judge_engine.py run as a
subprocess the way a tournament pod runs a half, then `combine`.
"""
import enum
import functools
import http.client
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import judge_engine as je
import sim.match as sm
from game import TIME_BANK_SECONDS, Action, AgentConfig
from sim.match import REPO_ROOT, bot_file_factory, load_bot_file
from sim.opponents import ZOO

STARTER = bot_file_factory("sim/opponents/starter.py")
ERRMIX = bot_file_factory("tests/fixtures/errmix_bot.py")
VERDICT = {je.AGENT1_WIN: "A", je.AGENT2_WIN: "B", je.CHOP: "CHOP", je.BOTH_FORFEIT: "BOTH_FORFEIT"}


# ------------------------------------------------------------ the in-process judge

@pytest.fixture(scope="module")
def servers():
    """Two copies of agent.py's server (one per seat), each with an in-process client."""
    mods = [load_bot_file("agent.py") for _ in range(2)]
    clients = [TestClient(m.app, raise_server_exceptions=False) for m in mods]
    for c in clients:
        c.__enter__()
    yield list(zip(mods, clients))
    for c in clients:
        c.__exit__(None, None, None)


class InProcessHandle(je.HttpAgentHandle):
    """The judge's own handle; only the HTTP exchange goes to agent.py's app in process."""

    def __init__(self, factory, config, server):
        super().__init__("http://in-process:1", config)
        self.factory = factory
        self.module, self.client = server
        self.up = False

    def start_half(self):
        m = self.module
        m.MEMORY.clear()
        m.STATUS["started"] = False                      # a fresh server process per half
        try:
            bot = self.factory()
            m.new_half, m.choose_action, m.on_hand_end = (
                bot.new_half, bot.choose_action, bot.on_hand_end)
            self.up = True
        except Exception:
            self.up = False                              # the server never came up
        super().start_half()

    def _request(self, method, path, body, timeout):
        # Same contract as HttpAgentHandle._request: (status, parsed JSON), TimeoutError
        # past the deadline, ValueError for a reply that is not JSON.
        if not self.up:
            raise ConnectionRefusedError("the bot server is not running")
        headers = {"Content-Type": "application/json"}
        if self.module.TOKEN:
            headers["X-Agent-Token"] = self.module.TOKEN
        start = time.monotonic()
        r = self.client.request(method, path, headers=headers,
                                content=None if body is None else json.dumps(body))
        if time.monotonic() - start > timeout:
            raise TimeoutError("the reply did not finish before the deadline")
        raw = r.content
        try:
            data = json.loads(raw) if raw else None
        except ValueError as exc:
            raise ValueError(f"response was not JSON: {raw[:200]!r}") from exc
        return r.status_code, data


def judge_half(servers, fa, fb, seed, half, deals, time_bank_s=TIME_BANK_SECONDS):
    cfg = AgentConfig(num_hands=deals, time_bank_seconds=time_bank_s)
    return je.play_half(InProcessHandle(fa, cfg, servers[0]), InProcessHandle(fb, cfg, servers[1]),
                        seed, half, deals)


def sim_half(fa, fb, seed, half, deals, **kw):
    return sm.play_half(fa, fb, seed, half, deals, grace_s=je.NETWORK_GRACE_SECONDS, **kw)


def judge_facts(r):
    a, b = r.stats_agent1, r.stats_agent2
    return {"nets": list(r.nets_agent1), "errors": (a.errors, b.errors),
            "forfeited": (a.forfeited, b.forfeited), "calls": (a.calls, b.calls),
            "bank_empty": (a.timed_out, b.timed_out)}


def sim_facts(h):
    return {"nets": list(h.nets_a), "errors": (h.errors_a, h.errors_b),
            "forfeited": (h.forfeit_a, h.forfeit_b), "calls": (h.calls_a, h.calls_b),
            "bank_empty": (h.bank_exhausted_a, h.bank_exhausted_b)}


# ------------------------------------------------------------------- test bots

class _Bot:
    def new_half(self, config, memory):
        memory["n"] = 0

    def choose_action(self, obs, memory):
        memory["n"] += 1
        return Action.CHECK_CALL

    def on_hand_end(self, result, memory):
        pass


class _SlowEvery(_Bot):
    """Every k-th decision takes `delay`, then asks for a raise (discarded if too late)."""

    def __init__(self, k, delay):
        self.k, self.delay = k, delay

    def choose_action(self, obs, memory):
        memory["n"] += 1
        if memory["n"] % self.k == 0:
            time.sleep(self.delay)
            return Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL
        return Action.CHECK_CALL


class _SlowNewHalf(_Bot):
    def new_half(self, config, memory):
        time.sleep(0.05)
        memory["n"] = 0


class _SlowHandEnd(_Bot):
    def on_hand_end(self, result, memory):
        time.sleep(0.004)


class _StrAction(str, enum.Enum):
    CHECK_CALL = "CHECK_CALL"


class _IntAction(enum.IntEnum):
    ONE = 1


class _OddReturns(_Bot):
    """Every third decision returns the next odd value, otherwise CHECK_CALL."""

    def __init__(self, values):
        self.values = values

    def choose_action(self, obs, memory):
        memory["n"] += 1
        k, r = divmod(memory["n"], 3)
        if r == 0 and k <= len(self.values):
            return self.values[k - 1]
        return Action.CHECK_CALL


# group -> (values, how many of them the judge rejects). Each group stays under
# ERROR_LIMIT, so every value is exercised.
ODD_RETURNS = {
    "numpy-and-floats": ([np.int64(1), 1.0, np.float64(1.0), float("nan")], 4),
    "strings-bool-none": (["1", "fold", True, None], 4),
    "accepted": (["CHECK_CALL", 1, _StrAction.CHECK_CALL, _IntAction.ONE], 0),
    "containers": ([(1,), [1], {"a": 1}], 3),
}

PAIRS = {
    "starter-v-calling_station": (STARTER, ZOO["calling_station"]),
    "nit-v-maniac": (ZOO["nit"], ZOO["maniac"]),
    "threshold_cat-v-always_half": (ZOO["threshold_cat"], ZOO["always_half"]),
    "random-v-always_pot": (ZOO["random"], ZOO["always_pot"]),
}


# --------------------------------------------------------- fast: in-process judge

@pytest.mark.parametrize("half", [1, 2])
@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_deterministic_bots_match_the_judge(servers, pair, half):
    fa, fb = PAIRS[pair]
    assert sim_facts(sim_half(fa, fb, "j1", half, 40)) == judge_facts(
        judge_half(servers, fa, fb, "j1", half, 40))


@pytest.mark.parametrize("half", [1, 2])
def test_counted_errors_and_forfeit_match_the_judge(servers, half):
    s = sim_half(ERRMIX, STARTER, "j2", half, 40)
    assert s.errors_a == 5 and s.forfeit_a and not s.forfeit_b
    assert sim_facts(s) == judge_facts(judge_half(servers, ERRMIX, STARTER, "j2", half, 40))


@pytest.mark.parametrize("group", sorted(ODD_RETURNS))
def test_odd_return_values_match_the_judge(servers, group):
    values, rejected = ODD_RETURNS[group]
    f = functools.partial(_OddReturns, values)
    cs = ZOO["calling_station"]
    j = judge_half(servers, f, cs, "j3", 1, 15)
    assert j.stats_agent1.errors == rejected
    assert sim_facts(sim_half(f, cs, "j3", 1, 15)) == judge_facts(j)


@pytest.mark.parametrize("bot", ["slow-every-9th", "slow-new-half"])
def test_timeouts_match_the_judge(servers, monkeypatch, bot):
    monkeypatch.setattr(je, "ACTION_TIMEOUT_SECONDS", 0.02)      # the judge reads it per request
    f = {"slow-every-9th": functools.partial(_SlowEvery, 9, 0.04),
         "slow-new-half": _SlowNewHalf}[bot]
    cs = ZOO["calling_station"]
    s = sim_half(f, cs, "j4", 1, 30, action_timeout_s=0.02)
    assert s.errors_a > 0
    assert sim_facts(s) == judge_facts(judge_half(servers, f, cs, "j4", 1, 30))


@pytest.mark.parametrize("bot", ["slow-every-2nd", "slow-hand-end"])
def test_bank_out_matches_the_judge(servers, bot):
    f = {"slow-every-2nd": functools.partial(_SlowEvery, 2, 0.015),
         "slow-hand-end": _SlowHandEnd}[bot]
    cs = ZOO["calling_station"]
    sf = sim_facts(sim_half(f, cs, "j5", 1, 60, time_bank_s=0.15))
    jf = judge_facts(judge_half(servers, f, cs, "j5", 1, 60, time_bank_s=0.15))
    for k in ("nets", "calls"):          # where exactly the bank runs dry is timing jitter
        del sf[k], jf[k]
    assert sf == jf and sf["bank_empty"] == (True, False) and sf["errors"] == (0, 0)


@pytest.mark.parametrize("pair", ["starter-v-calling_station", "nit-v-maniac"])
def test_match_verdict_matches_judge_combine(servers, pair):
    fa, fb = PAIRS[pair]
    r = je.combine_halves(judge_half(servers, fa, fb, "j6", 1, 40),
                          judge_half(servers, fa, fb, "j6", 2, 40))
    m = sm.play_match(fa, fb, "j6", 40, grace_s=je.NETWORK_GRACE_SECONDS)
    assert (m.mean_bb, m.sd_bb, m.ci_low, m.ci_high) == pytest.approx(
        (r.mean_bb, r.std_bb, r.ci[0], r.ci[1]), abs=1e-12)
    assert (m.verdict, m.points_a) == (VERDICT[r.verdict], r.points["agent1"])


# ------------------------------------------------- slow: real servers, real judge

HTTP_BOTS = {"starter": "sim/opponents/starter.py",
             "calling_station": "baseline-bots/baseline_calling_station.py",
             "errmix": "tests/fixtures/errmix_bot.py",
             "slow_once": "tests/fixtures/slow_once_bot.py"}
# (agent1, agent2, seed, errors the judge must count for agent1 in each half)
HTTP_MATCHES = [("starter", "calling_station", "h1", 0), ("starter", "calling_station", "h2", 0),
                ("errmix", "starter", "h1", 5), ("slow_once", "calling_station", "h1", 1)]
HTTP_DEALS = 60


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _answers(port):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
        c.request("GET", "/")
        ok = c.getresponse().status == 200
        c.close()
        return ok
    except OSError:
        return False


@pytest.fixture(scope="module")
def http_bots(tmp_path_factory):
    env = {k: v for k, v in os.environ.items() if k not in ("AGENT_TOKEN", "PORT")}
    logs = tmp_path_factory.mktemp("bot_logs")
    procs, ports = [], {}
    try:
        for name, path in HTTP_BOTS.items():
            ports[name] = _free_port()
            procs.append(subprocess.Popen(
                [sys.executable, "tests/fixtures/serve_bot.py", path, str(ports[name])],
                cwd=REPO_ROOT, env=env, stdout=open(logs / f"{name}.log", "w"),
                stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 30
        for name, port in ports.items():
            while not _answers(port):
                assert time.monotonic() < deadline, f"{name} never answered; logs in {logs}"
                time.sleep(0.1)
        yield ports
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


@pytest.mark.slow
@pytest.mark.parametrize("a,b,seed,errors", HTTP_MATCHES)
def test_sim_matches_the_real_judge_over_http(http_bots, tmp_path, a, b, seed, errors):
    env = {k: v for k, v in os.environ.items()
           if k not in ("AGENT1_TOKEN", "AGENT2_TOKEN", "AGENT1_URL", "AGENT2_URL", "HALF",
                        "MATCH_SEED")}

    def judge(*args):
        r = subprocess.run([sys.executable, "judge_engine.py", *args], cwd=REPO_ROOT, env=env,
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout + r.stderr

    for half in (1, 2):
        judge("--agent1", str(http_bots[a]), "--agent2", str(http_bots[b]), "--half", str(half),
              "--seed", seed, "--deals", str(HTTP_DEALS), "--json", str(tmp_path / f"h{half}.json"))
    judge("combine", str(tmp_path / "h1.json"), str(tmp_path / "h2.json"),
          "--json", str(tmp_path / "match.json"))
    halves = [json.loads((tmp_path / f"h{h}.json").read_text()) for h in (1, 2)]
    match = json.loads((tmp_path / "match.json").read_text())
    assert [h["agent_stats"]["agent1"]["errors"] for h in halves] == [errors, errors]

    m = sm.play_match(bot_file_factory(HTTP_BOTS[a]), bot_file_factory(HTTP_BOTS[b]), seed,
                      HTTP_DEALS, grace_s=je.NETWORK_GRACE_SECONDS)
    for s, j in zip(m.halves, halves):
        st1, st2 = j["agent_stats"]["agent1"], j["agent_stats"]["agent2"]
        assert sim_facts(s) == {
            "nets": j["nets_agent1"], "errors": (st1["errors"], st2["errors"]),
            "forfeited": (st1["forfeited"], st2["forfeited"]), "calls": (st1["calls"], st2["calls"]),
            "bank_empty": (st1["timed_out"], st2["timed_out"])}
    assert (m.mean_bb, m.sd_bb, m.ci_low, m.ci_high) == pytest.approx(
        (match["mean_bb_per_deal"], match["std_bb"], *match["ci"]), abs=1e-12)
    assert (m.verdict, m.points_a) == (VERDICT[match["verdict"]], match["points"]["agent1"])
