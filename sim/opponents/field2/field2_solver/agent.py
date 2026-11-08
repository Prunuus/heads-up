"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "Solverline"
AGENT_NAME = "SolverlineCFR"


# ===== YOUR CODE HERE =====

import json
import random
import time
from itertools import combinations, permutations

import numpy as np

from evaluator import _score_ranks

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- vectorized Omaha evaluator (rank-count index tables) ----------
_PW5 = (5 ** np.arange(9)).astype(np.int64)


def _build_rank_tables():
    nf = np.zeros(5 ** 9, dtype=np.int32)
    fl = np.zeros(5 ** 9, dtype=np.int32)
    for combo in combinations(range(13), 5):
        ranks = [c - i for i, c in enumerate(combo)]
        cnt = np.bincount(ranks, minlength=9)
        if cnt.max() > 4:
            continue
        idx = int(sum(_PW5[r] for r in ranks))
        rd = tuple(sorted(ranks, reverse=True))
        nf[idx] = _score_ranks(rd, False)
        if cnt.max() == 1:
            fl[idx] = _score_ranks(rd, True)
    return nf, fl


_NF, _FL = _build_rank_tables()
_PA = np.array([p[0] for p in combinations(range(5), 2)])
_PB = np.array([p[1] for p in combinations(range(5), 2)])
_TRI = {m: np.array(list(combinations(range(m), 3))) for m in (3, 4, 5)}
_RANKP = _PW5[np.arange(36) >> 2]
_SUIT = np.arange(36) & 3


def omaha_vec(hole, board):
    """hole (N,5), board (N,m) with m in 3..5 -> best 2+3 score per row."""
    hp = _RANKP[hole]; hs = _SUIT[hole]
    pv = hp[:, _PA] + hp[:, _PB]
    ps = np.where(hs[:, _PA] == hs[:, _PB], hs[:, _PA], -1)
    t = _TRI[board.shape[1]]
    bp = _RANKP[board]; bs = _SUIT[board]
    tv = bp[:, t[:, 0]] + bp[:, t[:, 1]] + bp[:, t[:, 2]]
    ts = np.where((bs[:, t[:, 0]] == bs[:, t[:, 1]]) & (bs[:, t[:, 1]] == bs[:, t[:, 2]]), bs[:, t[:, 0]], -2)
    idx = pv[:, :, None] + tv[:, None, :]
    f = ps[:, :, None] == ts[:, None, :]
    sc = np.where(f, _FL[idx], _NF[idx])
    return sc.reshape(sc.shape[0], -1).max(axis=1)


# ---------- offline tables ----------
_PERMS = list(permutations(range(4)))
try:
    _pf = np.load(os.path.join(HERE, "preflop.npz"))
    PF_PCT = dict(zip(_pf["codes"].tolist(), _pf["pct"].tolist()))
except Exception:
    PF_PCT = {}
RB = 20
try:
    _rv = np.load(os.path.join(HERE, "river.npz"))
    RIVER = {k: v for k, v in zip(_rv["keys"].tolist(), _rv["strat"])}
except Exception:
    RIVER = {}


def pf_pct(hole):
    best = None
    for p in _PERMS:
        m = sorted((c & ~3) | p[c & 3] for c in hole)
        code = (((m[0] * 36 + m[1]) * 36 + m[2]) * 36 + m[3]) * 36 + m[4]
        if best is None or code < best:
            best = code
    return PF_PCT.get(best, 0.5)


# ---------- Monte Carlo equity vs a weighted opponent range ----------
NSAMP = 224


def mc_cache(obs, memory):
    key = (obs.hand_index, int(obs.street))
    c = memory["mc"]
    if c.get("key") == key:
        return c
    hole = list(obs.hole_cards); board = list(obs.board)
    dead = set(hole) | set(board)
    rem = np.array([x for x in range(36) if x not in dead])
    need = 5 - len(board)
    rng = memory["nprng"]
    tu = memory.get("t_used", 0.0)
    n = NSAMP if tu < 3.5 else (96 if tu < 5.5 else 40)   # time-bank guard
    cards = rng.permuted(np.tile(rem, (n, 1)), axis=1)[:, :5 + 2 * need]
    opp = cards[:, :5]
    bd = np.broadcast_to(np.array(board), (n, len(board)))
    full = np.concatenate([bd, cards[:, 5:5 + need]], axis=1)
    if need == 0:
        me = omaha_vec(np.array([hole]), np.array([board]))[0]
    else:
        me = omaha_vec(np.broadcast_to(np.array(hole), (n, 5)), full)
    op = omaha_vec(opp, full)
    res = (me > op) + 0.5 * (me == op)
    if need == 0:
        q = op.astype(float)
    else:
        made = omaha_vec(opp, bd)
        full2 = np.concatenate([bd, cards[:, 5 + need:5 + 2 * need]], axis=1)
        pot2 = omaha_vec(opp, full2)
        q = 0.5 * (made.argsort().argsort()) + 0.5 * (pot2.argsort().argsort())
    qp = (q.argsort().argsort() + 0.5) / n
    c.clear()
    c.update(key=key, res=res, q=qp)
    return c


def eq_vs(c, k):
    if k <= 0.01:
        return float(c["res"].mean())
    w = c["q"] ** k
    return float((w * c["res"]).sum() / w.sum())


# ---------- opponent model ----------
SIZE_W = {2: 0.75, 3: 1.0}
STREET_W = {0: 0.25, 1: 0.8, 2: 1.0, 3: 1.2}


def opp_k(history, upto_street, memory):
    """Range-tightening exponent from opponent actions on streets < upto_street
    (or <= when upto_street is given as street+0.5)."""
    st = memory["stats"]
    agg_f = 1.0
    if st["pf_dec"] >= 30:
        r = (st["pf_agg"] + 1) / (st["pf_dec"] + 3)
        agg_f = min(1.3, max(0.1, (0.30 / max(r, 0.05)) ** 1.3))
    k = 0.0
    for a in history:
        if a.street >= upto_street:
            break
        if a.is_me:
            continue
        if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            k += SIZE_W[int(a.action)] * STREET_W[int(a.street)] * (agg_f if a.street > 0 else 1.0)
        elif a.action == Action.CHECK_CALL and a.street > 0:
            k += 0.3 if a.amount > 0 else -0.1
    return max(0.0, k)


def _rand(memory):
    return memory["rng"].random()


def pick_raise(obs, pref):
    la = obs.legal_actions
    if pref in la:
        return pref
    for a in (Action.RAISE_POT, Action.RAISE_HALF):
        if a in la:
            return a
    return Action.CHECK_CALL


# ---------- preflop ----------
try:
    PF_E = np.load(os.path.join(HERE, "pf_range.npy"))
except Exception:
    PF_E = None


def pf_eq(p, f):
    """All-in equity of a hand at preflop percentile p vs the top-f fraction of hands."""
    if PF_E is None:
        return 0.35 + 0.2 * p
    i = min(19, max(0, int(p * 20)))
    j = min(19, max(0, int(round(f * 20)) - 1))
    return float(PF_E[i, j])


def preflop_action(obs, memory):
    p0 = pf_pct(obs.hole_cards)
    p = min(0.999, max(0.0, p0 + (_rand(memory) - 0.5) * 0.06))
    st = memory["stats"]
    la = obs.legal_actions
    can_raise = Action.RAISE_POT in la
    C = obs.to_call
    pot = obs.pot
    req = C / (pot + C) if C > 0 else 0.0
    nr = obs.raises_this_street
    f_open = (st["open"] + 3.5) / (st["open_opp"] + 5)        # opp SB open-raise rate
    f_3b = (st["tb"] + 1) / (st["tb_opp"] + 6)                 # opp 3-bet rate vs our open
    f_iso = (st["iso"] + 1) / (st["iso_opp"] + 4)              # opp BB raise vs our limp
    f_4b = (st["fb4"] + 1.5) / (st["fb4_opp"] + 5)             # opp re-raise vs our 3-bet
    if nr == 0:
        if C > 0:  # SB first decision
            if p > 0.30 and can_raise:
                return Action.RAISE_POT
            if p > 0.08:
                return Action.CHECK_CALL
            return Action.FOLD
        # BB after a limp
        if can_raise and (p > 0.72 or (0.12 < p < 0.22 and _rand(memory) < 0.25)):
            return Action.RAISE_POT
        return Action.CHECK_CALL
    # facing a raise: estimate the opponent's range as a top fraction of hands
    me_raised = any(a.is_me and a.street == 0 and a.action in (Action.RAISE_HALF, Action.RAISE_POT)
                    for a in obs.action_history)
    if nr == 1:
        f = f_open if not obs.is_button else f_iso
    elif nr == 2:
        f = f_3b if obs.is_button else f_iso * 0.5
    else:
        f = (f_open if not obs.is_button else f_3b) * f_4b
        if st["opp_pf_shove"] > 0.25:
            f = max(f, 0.5)
    f = min(1.0, max(0.05, f))
    e = pf_eq(p, f)
    stack_after = obs.my_stack - C
    realize = 1.0 if stack_after <= 0 else (0.97 if obs.is_button else 0.90)
    if can_raise and p > 0.70 and e > 0.555 and (nr <= 2 or e > 0.58):
        return Action.RAISE_POT
    if e * realize > req + 0.01:
        return Action.CHECK_CALL
    return Action.FOLD


# ---------- river (CFR table) ----------
_ACODE = {Action.FOLD: 0, Action.CHECK_CALL: 1, Action.RAISE_HALF: 2, Action.RAISE_POT: 3}
_ACH = "fchp"


def river_action(obs, memory, c):
    k0 = opp_k(obs.action_history, 3, memory)
    r = eq_vs(c, k0)
    seq = "".join(_ACH[_ACODE[a.action]] for a in obs.action_history if a.street == Street.RIVER)
    pot0 = obs.pot - obs.my_bet - obs.opp_bet
    strat = RIVER.get(f"{pot0}|{seq}")
    st = memory["stats"]
    if strat is None:
        return fallback_post(obs, memory, c)
    b = min(RB - 1, int(r * RB))
    probs = np.array(strat[b], dtype=float)
    # exploitation overrides, evidence-gated
    station = st["fb_opp"] >= 25 and st["fb"] / st["fb_opp"] < 0.18
    folder = st["fb_opp"] >= 25 and st["fb"] / st["fb_opp"] > 0.6
    if obs.to_call == 0:
        if station and r < 0.5:
            probs[2] = probs[3] = 0.0
        if station and r > 0.62:
            probs[2] += 0.5
        if folder and r < 0.3:
            probs[3] += 0.2
    else:
        maniac = st["pf_dec"] >= 40 and (st["pf_agg"] / st["pf_dec"]) > 0.55
        nit = st["pf_dec"] >= 40 and (st["pf_agg"] / st["pf_dec"]) < 0.15
        if maniac and r > 0.3:
            probs[1] += probs[0]; probs[0] = 0.0
        if nit and r < 0.75:
            probs[0] += probs[1] * 0.5; probs[1] *= 0.5
    la = obs.legal_actions
    for a in Action:
        if a not in la:
            probs[_ACODE[a]] = 0.0
    s = probs.sum()
    if s <= 0:
        return Action.CHECK_CALL
    x = _rand(memory) * s
    for a in (Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT):
        x -= probs[_ACODE[a]]
        if x <= 0 and a in la:
            return a
    return Action.CHECK_CALL


# ---------- flop / turn rules ----------
def fallback_post(obs, memory, c):
    st = memory["stats"]
    la = obs.legal_actions
    k = opp_k(obs.action_history, obs.street + 1, memory)
    e = eq_vs(c, k)
    C = obs.to_call
    pot = obs.pot
    can_raise = Action.RAISE_POT in la
    eff = min(obs.my_stack, obs.opp_stack + obs.opp_bet - obs.my_bet) if C == 0 else obs.my_stack
    spr = eff / max(pot, 1)
    fold_rate = (st["fb"] + 3) / (st["fb_opp"] + 8)
    station = st["fb_opp"] >= 25 and fold_rate < 0.2
    ip = obs.is_button
    u = _rand(memory)
    if C == 0:
        if not can_raise:
            return Action.CHECK_CALL
        if e > 0.70:
            if u < 0.12 and ip is False and spr > 2:
                return Action.CHECK_CALL  # occasional trap to protect checks
            return Action.RAISE_POT if u < 0.75 else Action.RAISE_HALF
        if e > 0.57 or (station and e > 0.53):
            return Action.RAISE_HALF if u < (0.75 if ip else 0.55) else Action.CHECK_CALL
        bl = (0.30 if ip else 0.18) * min(1.6, fold_rate / 0.4)
        if station:
            bl = 0.0
        if 0.30 < e < 0.50 and u < bl:
            return Action.RAISE_POT if u < bl * 0.5 else Action.RAISE_HALF
        return Action.CHECK_CALL
    req = C / (pot + C)
    allin_call = C >= obs.my_stack
    margin = 0.0 if allin_call else (0.05 if obs.street == Street.FLOP else 0.035)
    if not ip and not allin_call:
        margin += 0.02
    if can_raise and e > (0.72 if spr > 1 else 0.6):
        return Action.RAISE_POT if u < 0.7 else Action.RAISE_HALF
    if can_raise and not station and 0.40 < e < 0.50 and u < 0.06 and spr > 1.5:
        return Action.RAISE_POT
    if e > req + margin:
        return Action.CHECK_CALL
    return Action.FOLD


# ---------- stats from completed hands ----------
def _process_hand(seq, memory):
    """seq: list of (street, is_me, action, amount). Updates opponent stats."""
    st = memory["stats"]
    if not seq:
        return
    last_raiser = None   # per street: who made the last raise
    cur_st = -1
    pf_raises = 0
    for i, (s, me, a, amt) in enumerate(seq):
        if s != cur_st:
            cur_st = s; last_raiser = None
            n_r = 0
        facing = last_raiser is not None and last_raiser != me
        is_r = a in (2, 3)
        if not me:
            if s == 0:
                if i == 0:   # opp is the SB, first action
                    st["open_opp"] += 1; st["open"] += is_r
                elif n_r == 1 and facing and seq[i - 1][1] and seq[i - 1][2] in (2, 3) and i == 1:
                    st["tb_opp"] += 1; st["tb"] += is_r
                elif i == 1 and seq[0][1] and seq[0][2] == 1:
                    st["iso_opp"] += 1; st["iso"] += is_r
                if n_r == 2 and facing:
                    st["fb4_opp"] += 1; st["fb4"] += is_r
                if is_r and n_r >= 2:
                    st["opp_pf_shove"] = 0.9 * st["opp_pf_shove"] + 0.1
            else:
                st["pf_dec"] += 1; st["pf_agg"] += is_r
                if facing and last_raiser:
                    st["fb_opp"] += 1; st["fb"] += (a == 0)
        if is_r:
            last_raiser = me; n_r += 1


def _note_decision(obs, action, memory):
    seq = [(int(a.street), a.is_me, int(a.action), a.amount) for a in obs.action_history]
    seq.append((int(obs.street), True, int(action), 0))
    memory["last"] = [obs.hand_index, seq]


def new_half(config: AgentConfig, memory: dict) -> None:
    seed = random.SystemRandom().randrange(1 << 30)
    memory["rng"] = random.Random(seed)
    memory["nprng"] = np.random.default_rng(seed)
    memory["mc"] = {}
    memory["t_used"] = 0.0
    memory["last"] = [-1, []]
    memory["stats"] = {"open": 0, "open_opp": 0, "tb": 0, "tb_opp": 0, "iso": 0, "iso_opp": 0,
                       "pf_dec": 0, "pf_agg": 0, "fb": 0, "fb_opp": 0, "opp_pf_shove": 0.0,
                       "fb4": 0, "fb4_opp": 0}


def _choose(obs, memory):
    if obs.street == Street.PREFLOP:
        return preflop_action(obs, memory)
    if memory.get("t_used", 0.0) > 6.6:   # near the 8 s bank: zero-compute emergency play
        if obs.to_call == 0:
            return Action.CHECK_CALL
        return Action.CHECK_CALL if obs.to_call * 4 <= obs.pot else Action.FOLD
    c = mc_cache(obs, memory)
    if obs.street == Street.RIVER:
        return river_action(obs, memory, c)
    return fallback_post(obs, memory, c)


def choose_action(obs: Observation, memory: dict) -> Action:
    t0 = time.perf_counter()
    la = obs.legal_actions
    try:
        if "rng" not in memory:
            new_half(AgentConfig(), memory)
        a = _choose(obs, memory)
        if a not in la:
            a = Action.CHECK_CALL if Action.CHECK_CALL in la else la[0]
    except Exception:
        a = Action.CHECK_CALL if Action.CHECK_CALL in la else la[0]
    try:
        _note_decision(obs, a, memory)
        # charge our compute plus ~1 ms of request overhead against a private budget
        memory["t_used"] = memory.get("t_used", 0.0) + (time.perf_counter() - t0) + 0.001
    except Exception:
        pass
    return a


def on_hand_end(result: HandResult, memory: dict) -> None:
    try:
        if "last" not in memory:
            return
        hi, seq = memory["last"]
        if hi != result.hand_index or not seq:
            return
        seq = list(seq)
        if not result.went_to_showdown and result.my_net > 0 and seq[-1][1]:
            seq.append((seq[-1][0], False, 0, 0))   # opponent folded to our last action
        _process_hand(seq, memory)
        memory["last"] = [-1, []]
    except Exception:
        pass

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
