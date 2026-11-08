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

import bisect
import random
import time
from itertools import combinations

from evaluator import get_table

_PAIRS = tuple(combinations(range(5), 2))
_TRIPLES = tuple(combinations(range(5), 3))
_RNG = random.Random(12345)

# Time management (seconds). Bank is 8s per half; we plan to use far less
# of it ourselves because network overhead is billed too.
PLAN_BUDGET = 4.2      # target total MC time per half
SOFT_LIMIT = 5.4       # beyond this: no MC at all, heuristics only
HARD_LIMIT = 6.8       # beyond this: near-instant trivial policy
MAX_DECISION_S = 0.03  # never spend more than this on a single decision


def _pre_raw(hole):
    """Heuristic preflop strength of 5 hole cards (higher = better)."""
    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]
    vals = []
    for i, j in _PAIRS:
        r1, r2 = ranks[i], ranks[j]
        v = 0.0
        if r1 == r2:
            v += 1.0 + 0.12 * r1
        else:
            v += 0.06 * (r1 + r2)
            gap = abs(r1 - r2)
            if gap == 1:
                v += 0.25
            elif gap == 2:
                v += 0.12
            elif gap == 3:
                v += 0.05
        if suits[i] == suits[j]:
            v += 0.25 + (0.15 if max(r1, r2) == 8 else 0.0)
        vals.append(v)
    vals.sort(reverse=True)
    s = sum(vals[:5]) + 0.3 * sum(vals[5:8])
    # rank-count structure: trips / quads in hand are poor in PLO5
    cnt = {}
    for r in ranks:
        cnt[r] = cnt.get(r, 0) + 1
    for n in cnt.values():
        if n >= 3:
            s -= 0.8 * (n - 2)
    # ace with good support
    if 8 in cnt:
        s += 0.2
    # connectedness of whole hand
    ur = sorted(cnt)
    s += 0.15 * max(sum(1 for r in ur if lo <= r <= lo + 4) for lo in range(0, 5))
    return s


def _build_pre_dist():
    rng = random.Random(777)
    deck = list(range(36))
    out = []
    for _ in range(3000):
        out.append(_pre_raw(rng.sample(deck, 5)))
    out.sort()
    return out


_PRE_DIST = _build_pre_dist()


def _pre_q(hole):
    return bisect.bisect_left(_PRE_DIST, _pre_raw(hole)) / len(_PRE_DIST)


def _equity(hole, board, n_samples, deadline):
    """Monte Carlo equity vs a uniformly random opponent hand."""
    table = get_table()
    known = set(hole) | set(board)
    pool = [c for c in range(36) if c not in known]
    need_board = 5 - len(board)
    hb = [1 << c for c in hole]
    hmasks = [hb[i] | hb[j] for i, j in _PAIRS]
    board = list(board)
    wins = 0.0
    done = 0
    sample = _RNG.sample
    clock = time.perf_counter
    for k in range(n_samples):
        draw = sample(pool, need_board + 5)
        full = board + draw[:need_board]
        opp = draw[need_board:]
        bb = [1 << c for c in full]
        bmasks = [bb[i] | bb[j] | bb[l] for i, j, l in _TRIPLES]
        ob = [1 << c for c in opp]
        omasks = [ob[i] | ob[j] for i, j in _PAIRS]
        mine = max([table[h | b] for h in hmasks for b in bmasks])
        theirs = max([table[h | b] for h in omasks for b in bmasks])
        if mine > theirs:
            wins += 1.0
        elif mine == theirs:
            wins += 0.5
        done += 1
        if (k & 7) == 7 and clock() > deadline:
            break
    if done == 0:
        return None, 0
    return wins / done, done


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["hands_played"] = 0
    memory["used"] = 0.0          # seconds spent in choose_action this half
    memory["decisions"] = 0
    memory["per_sample"] = 8e-5   # running estimate of seconds / MC sample
    memory["opp_raise"] = 1.0     # smoothed opponent aggression counts
    memory["opp_act"] = 4.0


def _legal(obs, act):
    return act if act in obs.legal_actions else None


def _safe_default(obs):
    if Action.CHECK_CALL in obs.legal_actions and obs.to_call <= 0:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return obs.legal_actions[0]


def _pick_raise(obs, big):
    if big and Action.RAISE_POT in obs.legal_actions:
        return Action.RAISE_POT
    if Action.RAISE_HALF in obs.legal_actions:
        return Action.RAISE_HALF
    if Action.RAISE_POT in obs.legal_actions:
        return Action.RAISE_POT
    return None


def _preflop(obs, memory):
    q = _pre_q(obs.hole_cards)
    legal = obs.legal_actions
    rs = obs.raises_this_street
    can_raise = Action.RAISE_HALF in legal or Action.RAISE_POT in legal
    if obs.to_call <= 0:
        if can_raise and q > 0.62 and rs < 2:
            return _pick_raise(obs, q > 0.88)
        return Action.CHECK_CALL
    odds = obs.to_call / float(obs.pot + obs.to_call)
    if can_raise and rs < 3:
        if q > 0.97 or (rs == 0 and q > (0.55 if obs.is_button else 0.70)):
            return _pick_raise(obs, q > 0.9)
        if rs == 1 and q > 0.92:
            return _pick_raise(obs, q > 0.96)
    need = 0.18 + 1.0 * odds + 0.07 * rs
    # opponent that raises a lot preflop is wider: loosen slightly
    agg = memory["opp_raise"] / memory["opp_act"]
    if agg > 0.45:
        need -= 0.05
    if q > need:
        return Action.CHECK_CALL
    if Action.FOLD in legal:
        return Action.FOLD
    return Action.CHECK_CALL


def _postflop(obs, memory, eq):
    legal = obs.legal_actions
    rs = obs.raises_this_street
    can_raise = Action.RAISE_HALF in legal or Action.RAISE_POT in legal
    if obs.to_call <= 0:
        if can_raise:
            if eq > 0.80:
                return _pick_raise(obs, True)
            if eq > 0.62:
                return _pick_raise(obs, False)
        return Action.CHECK_CALL
    odds = obs.to_call / float(obs.pot + obs.to_call)
    # opponent's betting range is stronger than random
    agg = memory["opp_raise"] / memory["opp_act"]
    adj = eq - 0.30 * odds - 0.03 * rs
    if agg > 0.5:
        adj += 0.03
    if can_raise and rs < 3 and adj > 0.74:
        return _pick_raise(obs, adj > 0.86)
    if adj > odds + 0.05:
        return Action.CHECK_CALL
    if Action.FOLD in legal:
        return Action.FOLD
    return Action.CHECK_CALL


def _fast_postflop(obs):
    """No-simulation fallback when the time bank is getting low."""
    if obs.to_call <= 0:
        return Action.CHECK_CALL
    if len(obs.board) == 5:
        sc = omaha_score(obs.hole_cards, obs.board)
        odds = obs.to_call / float(obs.pot + obs.to_call)
        cat = sc >> 20
        if cat >= 4 or (cat >= 2 and odds < 0.3):
            return Action.CHECK_CALL
    elif obs.to_call <= obs.pot * 0.25:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL


def choose_action(obs: Observation, memory: dict) -> Action:
    t0 = time.perf_counter()
    try:
        memory["decisions"] = memory.get("decisions", 0) + 1
        used = memory.get("used", 0.0)
        action = None
        if obs.street == Street.PREFLOP:
            action = _preflop(obs, memory) if used < HARD_LIMIT else None
        elif used >= SOFT_LIMIT:
            action = _fast_postflop(obs)
        else:
            hands_left = max(1, 1000 - obs.hand_index)
            # about 1.6 postflop decisions per hand on average
            est_left = hands_left * 1.6 + 2
            budget = max(0.0, PLAN_BUDGET - used) / est_left
            budget = min(MAX_DECISION_S, max(budget, 0.0006) * (1.0 + 0.5 * (len(obs.board) == 3)))
            per = memory.get("per_sample", 8e-5)
            n = int(budget / per)
            n = max(8, min(n, 200))
            if obs.to_call <= 0 and obs.my_stack <= 0:
                n = 0
            if n > 0:
                ts = time.perf_counter()
                eq, done = _equity(obs.hole_cards, obs.board, n, ts + MAX_DECISION_S)
                el = time.perf_counter() - ts
                if done:
                    memory["per_sample"] = 0.8 * per + 0.2 * (el / done)
                if eq is not None:
                    action = _postflop(obs, memory, eq)
            if action is None:
                action = _fast_postflop(obs)
        if action is None:
            action = _safe_default(obs)
    except Exception:
        action = _safe_default(obs)
    if action not in obs.legal_actions:
        action = _safe_default(obs)
    memory["used"] = memory.get("used", 0.0) + (time.perf_counter() - t0)
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    memory["hands_played"] = memory.get("hands_played", 0) + 1


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
