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
PARTICIPANT = "KChen"
AGENT_NAME = "ShortDeckHybrid"


# ===== YOUR CODE HERE =====

import random
import time
from collections import Counter
from itertools import combinations

import numpy as np

from evaluator import get_table

# ---------------------------------------------------------------------------
# Startup "training" (all generated at import time, ~2-3 s):
#   1. Turn the evaluator's hash table into sorted numpy arrays so a whole batch
#      of 5-card masks can be scored with one searchsorted call.
#   2. Monte-Carlo simulate preflop deals and fit a ridge regression from cheap
#      hole-card features -> equity vs a random hand (eq1) and equity vs the
#      better of two random hands (eq2, a proxy for "vs a betting range").
#   3. At play time: preflop uses the regression, postflop uses a vectorised
#      Monte-Carlo with an adaptive sample count tied to the time bank, an
#      online opponent model (aggression, fold-to-raise), and pot-odds logic.
# ---------------------------------------------------------------------------

_T = get_table()
_KEYS = np.fromiter(_T.keys(), dtype=np.int64, count=len(_T))
_VALS = np.fromiter(_T.values(), dtype=np.int64, count=len(_T))
_ORD = np.argsort(_KEYS)
_KEYS = _KEYS[_ORD]
_VALS = _VALS[_ORD]
del _ORD

_PI = np.array(list(combinations(range(5), 2)), dtype=np.int64)   # (10, 2)
_TI = np.array(list(combinations(range(5), 3)), dtype=np.int64)   # (10, 3)
_ONE = np.int64(1)


def _omaha_batch(hole: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Vectorised omaha_score. hole, board: (N, 5) int arrays -> (N,) scores."""
    hb = np.left_shift(_ONE, hole)
    bb = np.left_shift(_ONE, board)
    hm = hb[:, _PI[:, 0]] | hb[:, _PI[:, 1]]                         # (N, 10)
    bm = bb[:, _TI[:, 0]] | bb[:, _TI[:, 1]] | bb[:, _TI[:, 2]]      # (N, 10)
    masks = (hm[:, :, None] | bm[:, None, :]).reshape(-1)
    idx = np.searchsorted(_KEYS, masks)
    return _VALS[idx].reshape(hole.shape[0], 100).max(axis=1)


def _features(hole) -> list:
    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]
    rc = Counter(ranks)
    sc = Counter(suits)
    pairs = sum(v * (v - 1) // 2 for v in rc.values())
    trips = sum(1 for v in rc.values() if v >= 3)
    pair_hi = sum(r / 8.0 for r, v in rc.items() if v >= 2)
    suited_pairs = sum(v * (v - 1) // 2 for v in sc.values())
    ds = sum(1 for v in sc.values() if v >= 2)
    suit_top = 0.0
    for s, v in sc.items():
        if v >= 2:
            suit_top += max(r for r, ss in zip(ranks, suits) if ss == s) / 8.0
    ace_suited = 0.0
    for r, s in zip(ranks, suits):
        if r == 8 and sc[s] >= 2:
            ace_suited = 1.0
    aces = rc.get(8, 0)
    kings = rc.get(7, 0)
    broad = sum(1 for r in ranks if r >= 4)
    mean_rank = sum(ranks) / 40.0
    distinct = len(rc)
    rs = set(ranks)
    conn = 0
    for lo in range(0, 5):
        w = {lo, lo + 1, lo + 2, lo + 3, lo + 4}
        conn = max(conn, len(rs & w))
    conn = max(conn, len(rs & {8, 0, 1, 2, 3}))
    four_conn = 1.0 if conn >= 4 else 0.0
    dangler = 1.0 if (pairs == 0 and conn <= 3) else 0.0
    return [1.0, pairs, trips, pair_hi, suited_pairs, ds, suit_top, ace_suited,
            aces, kings, broad, mean_rank, distinct, conn, four_conn, dangler,
            pairs * ds, aces * ds]


def _train_preflop(samples: int = 12000, seed: int = 20260):
    rng = np.random.default_rng(seed)
    X, y1, y2 = [], [], []
    chunk = 3000
    for _ in range(samples // chunk):
        noise = rng.random((chunk, 36))
        deck = np.argsort(noise, axis=1)[:, :20].astype(np.int64)
        me, o1, o2, bd = deck[:, 0:5], deck[:, 5:10], deck[:, 10:15], deck[:, 15:20]
        # board needs 5 cards: take 15..19
        sm = _omaha_batch(me, bd)
        s1 = _omaha_batch(o1, bd)
        s2 = _omaha_batch(o2, bd)
        w1 = (sm > s1) + 0.5 * (sm == s1)
        so = np.maximum(s1, s2)
        w2 = (sm > so) + 0.5 * (sm == so)
        for i in range(chunk):
            X.append(_features(sorted(me[i].tolist())))
        y1.append(w1)
        y2.append(w2)
    X = np.array(X, dtype=np.float64)
    y1 = np.concatenate(y1)
    y2 = np.concatenate(y2)
    lam = 3.0
    A = X.T @ X + lam * np.eye(X.shape[1])
    b1 = np.linalg.solve(A, X.T @ y1)
    b2 = np.linalg.solve(A, X.T @ y2)
    return b1, b2


try:
    _B1, _B2 = _train_preflop()
except Exception:  # extremely defensive: fall back to a flat prior
    _B1 = _B2 = None


def _pre_equity(hole):
    """(eq vs random hand, eq vs better-of-two random hands) for 5 hole cards."""
    if _B1 is None:
        return 0.5, 0.33
    f = np.array(_features(hole), dtype=np.float64)
    e1 = float(f @ _B1)
    e2 = float(f @ _B2)
    return min(max(e1, 0.05), 0.95), min(max(e2, 0.03), 0.9)


def _mc_equity(hole, board, n, rng):
    """Postflop MC. Returns (eq vs random hand, eq vs better of two random)."""
    known = list(hole) + list(board)
    need = 5 - len(board)
    k = 10 + need
    noise = rng.random((n, 36))
    noise[:, known] = 2.0
    deck = np.argsort(noise, axis=1)[:, :k].astype(np.int64)
    o1 = deck[:, 0:5]
    o2 = deck[:, 5:10]
    if need:
        bd = np.empty((n, 5), dtype=np.int64)
        bd[:, :len(board)] = np.array(board, dtype=np.int64)
        bd[:, len(board):] = deck[:, 10:10 + need]
    else:
        bd = np.tile(np.array(board, dtype=np.int64), (n, 1))
    me = np.tile(np.array(hole, dtype=np.int64), (n, 1))
    sm = _omaha_batch(me, bd)
    s1 = _omaha_batch(o1, bd)
    s2 = _omaha_batch(o2, bd)
    w1 = (sm > s1) + 0.5 * (sm == s1)
    so = np.maximum(s1, s2)
    w2 = (sm > so) + 0.5 * (sm == so)
    return float(w1.mean()), float(w2.mean())


def new_half(config: AgentConfig, memory: dict) -> None:
    """Fresh state for a half: RNGs, opponent-model counters, time tracking."""
    memory["hands_played"] = 0
    memory["np_rng"] = np.random.default_rng(987654)
    memory["rng"] = random.Random(13579)
    memory["num_hands"] = max(1, int(getattr(config, "num_hands", 1000)))
    memory["t_used"] = 0.0
    memory["cost"] = 4e-5          # EMA seconds per MC sample
    memory["cur_hand"] = -1
    memory["hist_n"] = 0
    memory["o_tot"] = 0            # opponent decisions seen
    memory["o_rais"] = 0           # ... of which raises
    memory["my_raised"] = False
    memory["r_hands"] = 0          # hands where I raised
    memory["r_folds"] = 0          # ... and the opponent folded without showdown


def _opp_stats(memory):
    agg = (memory["o_rais"] + 2.0) / (memory["o_tot"] + 6.0)       # prior ~0.33
    foldr = (memory["r_folds"] + 2.0) / (memory["r_hands"] + 5.0)  # prior ~0.4
    return agg, foldr


def _update_opp(obs, memory):
    if obs.hand_index != memory["cur_hand"]:
        memory["cur_hand"] = obs.hand_index
        memory["hist_n"] = 0
        memory["my_raised"] = False
    hist = obs.action_history
    for a in hist[memory["hist_n"]:]:
        if not a.is_me:
            memory["o_tot"] += 1
            if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
                memory["o_rais"] += 1
    memory["hist_n"] = len(hist)


def _decide(obs, memory, eq1, eq2):
    legal = obs.legal_actions
    can_raise = Action.RAISE_POT in legal and Action.RAISE_HALF in legal
    tc = obs.to_call
    pot = obs.pot
    rnd = memory["rng"].random()
    agg, foldr = _opp_stats(memory)
    bluffp = min(0.22, max(0.02, (foldr - 0.3) * 0.5))

    if tc > 0 and obs.raises_this_street > 0:
        ratio = min(1.0, tc / max(pot - tc, 1))
        tight = min(1.0, max(0.3, 1.0 - (agg - 0.3) * 1.5))
        w = ratio * 0.85 * tight
    else:
        w = 0.0
    eff = eq1 + (eq2 - eq1) * w

    if tc > 0:
        need = tc / float(pot + tc)
        opening = obs.street == Street.PREFLOP and obs.raises_this_street == 0
        if can_raise:
            if opening:
                if eff >= 0.60:
                    return Action.RAISE_POT
                if eff >= 0.53:
                    return Action.RAISE_HALF
            elif eff >= max(0.64, need + 0.22):
                return Action.RAISE_POT if (eff >= 0.74 or rnd < 0.6) else Action.RAISE_HALF
            elif (obs.street != Street.RIVER and obs.raises_this_street <= 1
                  and 0.40 <= eff < need + 0.03 and rnd < bluffp * 0.4):
                return Action.RAISE_HALF
        margin = 0.035 + (0.0 if obs.is_button else 0.02)
        if opening:
            margin = 0.0
        if eff >= need + margin or tc >= obs.my_stack and eff >= need:
            return Action.CHECK_CALL
        return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL

    # nothing to call
    if can_raise:
        if eff >= 0.66:
            return Action.RAISE_POT if rnd < 0.6 else Action.RAISE_HALF
        if eff >= 0.56:
            if rnd < 0.8:
                return Action.RAISE_HALF
        elif eff >= 0.50:
            if rnd < 0.3:
                return Action.RAISE_HALF
        elif eff < 0.42 and rnd < bluffp:
            return Action.RAISE_HALF
    return Action.CHECK_CALL


def _safe(obs):
    if Action.CHECK_CALL in obs.legal_actions and obs.to_call == 0:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL


def choose_action(obs: Observation, memory: dict) -> Action:
    """Preflop: regression equity. Postflop: adaptive vectorised Monte Carlo.
    Then pot-odds / value / bluff logic using the online opponent model."""
    t0 = time.perf_counter()
    try:
        _update_opp(obs, memory)
        hole = list(obs.hole_cards)
        board = list(obs.board)
        if obs.street == Street.PREFLOP or len(board) < 3:
            eq1, eq2 = _pre_equity(hole)
        else:
            left = memory["num_hands"] - obs.hand_index
            budget_total = 3.5 - memory["t_used"]
            if budget_total < 0.15:
                # time nearly gone: play cheaply and safely
                if obs.to_call == 0:
                    act = Action.CHECK_CALL
                elif obs.to_call <= 0.15 * obs.pot:
                    act = Action.CHECK_CALL
                else:
                    act = _safe(obs)
                if act not in obs.legal_actions:
                    act = _safe(obs)
                return act
            per_dec = budget_total / max(1.0, left * 2.5)
            per_dec = min(0.006, max(0.0004, per_dec))
            n = int(per_dec / memory["cost"])
            n = max(24, min(400, n))
            t1 = time.perf_counter()
            eq1, eq2 = _mc_equity(hole, board, n, memory["np_rng"])
            dt = time.perf_counter() - t1
            memory["cost"] = 0.7 * memory["cost"] + 0.3 * (dt / n)
        act = _decide(obs, memory, eq1, eq2)
        if act not in obs.legal_actions:
            act = Action.CHECK_CALL if Action.CHECK_CALL in obs.legal_actions else _safe(obs)
        if act in (Action.RAISE_HALF, Action.RAISE_POT):
            memory["my_raised"] = True
        return act
    except Exception:
        return _safe(obs)
    finally:
        memory["t_used"] = memory.get("t_used", 0.0) + (time.perf_counter() - t0)


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Track how often the opponent folds when I raise (feeds bluff frequency)."""
    memory["hands_played"] = memory.get("hands_played", 0) + 1
    try:
        if memory.get("my_raised"):
            memory["r_hands"] += 1
            if (not result.went_to_showdown) and result.winner == "me":
                memory["r_folds"] += 1
        memory["my_raised"] = False
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
