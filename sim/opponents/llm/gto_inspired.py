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
import random
import time
import traceback
from bisect import bisect_left

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
AGENT_NAME = "BalancedMixer"


# ===== YOUR CODE HERE =====
#
# Strategy: "balanced mixer".
#   * Preflop: a hand-feature strength score turned into a percentile; open /
#     defend / re-raise by percentile bands, with randomised frequencies.
#   * Postflop: Monte Carlo equity vs a random opponent hand (sample count is
#     scaled to the remaining time bank). Value bets with the strong part of
#     the range, a bluff frequency sized to the bet, thin-value / protection
#     bets in the middle, calls priced by pot odds with a shrink for the
#     opponent's betting range, and randomised thresholds so no action is a
#     deterministic function of hand strength.
#   * Light opponent tracking (raise frequency) nudges calling thresholds.

_RNG = random.Random()
_TIME_LIMIT = 6.0            # seconds of our own 8 s bank we allow ourselves
_PRE_CDF: list = []          # sorted sample of preflop strengths


def _pre_raw(hole) -> float:
    """Heuristic preflop strength of five hole cards (higher = better)."""
    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]
    rc = [0] * 9
    for r in ranks:
        rc[r] += 1
    sc = [0] * 4
    for s in suits:
        sc[s] += 1
    score = 0.0
    for r in range(9):
        n = rc[r]
        if n == 2:
            score += 1.0 + 0.16 * r
        elif n == 3:
            score += 0.9 + 0.08 * r - 0.8
        elif n == 4:
            score += 1.5 - 1.8
    score += 0.45 * sum(r / 8.0 for r in ranks)
    score += 0.35 * min(2, rc[8])
    # suitedness (nut-suit bonus)
    suited_groups = 0
    for s in range(4):
        n = sc[s]
        if n >= 2:
            suited_groups += 1
            score += 0.30 * min(n - 1, 2) - (0.25 if n >= 4 else 0.0)
            ace_in = any((c >> 2) == 8 and (c & 3) == s for c in hole)
            score += 0.3 if ace_in else 0.0
    if suited_groups >= 2:
        score += 0.3
    # connectivity: most cards inside any five-rank window (ace plays low too)
    present = [1 if rc[r] else 0 for r in range(9)]
    best = 0
    for lo in range(-1, 5):
        cnt = 0
        for r in range(lo, lo + 5):
            rr = 8 if r == -1 else r
            if present[rr]:
                cnt += 1
        best = max(best, cnt)
    score += 0.38 * (best - 2)
    return score


def _init_pre_cdf() -> None:
    if _PRE_CDF:
        return
    rng = random.Random(12345)
    vals = []
    for _ in range(4000):
        vals.append(_pre_raw(rng.sample(range(36), 5)))
    vals.sort()
    _PRE_CDF.extend(vals)


def _pre_pct(hole) -> float:
    return bisect_left(_PRE_CDF, _pre_raw(hole)) / len(_PRE_CDF)


_init_pre_cdf()


def new_half(config: AgentConfig, memory: dict) -> None:
    """Reset per-half state."""
    memory["hands_played"] = 0
    memory["spent"] = 0.0            # seconds of our time bank used
    memory["cost"] = 3e-5            # estimated seconds per equity sample
    memory["rng"] = random.Random()
    memory["seen_hand"] = -1         # opponent-action tracking
    memory["seen_idx"] = 0
    memory["opp_acts"] = 0
    memory["opp_raises"] = 0


def _track_opponent(obs: Observation, memory: dict) -> None:
    if obs.hand_index != memory["seen_hand"]:
        memory["seen_hand"] = obs.hand_index
        memory["seen_idx"] = 0
    hist = obs.action_history
    for a in hist[memory["seen_idx"]:]:
        if not a.is_me:
            memory["opp_acts"] += 1
            if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
                memory["opp_raises"] += 1
    memory["seen_idx"] = len(hist)


def _opp_aggr(memory: dict) -> float:
    """Smoothed fraction of opponent actions that are raises (prior 0.30)."""
    return (memory["opp_raises"] + 6.0) / (memory["opp_acts"] + 20.0)


def _equity(obs: Observation, memory: dict, rng: random.Random) -> float:
    """Monte Carlo equity vs a uniformly random opponent hand."""
    hole = list(obs.hole_cards)
    board = list(obs.board)
    known = set(hole) | set(board)
    deck = [c for c in range(36) if c not in known]
    missing = 5 - len(board)
    base = {Street.FLOP: 28, Street.TURN: 36, Street.RIVER: 48}[obs.street]
    # scale sample count to remaining budget
    hands_left = max(1, 1000 - obs.hand_index)
    allowance = (_TIME_LIMIT - memory["spent"]) / (hands_left * 3.0)
    n = int(allowance / max(memory["cost"], 1e-6))
    n = max(8, min(base * 2, n))
    if missing == 0:
        mine = omaha_score(hole, board)
    wins = 0.0
    t0 = time.perf_counter()
    for _ in range(n):
        s = rng.sample(deck, 5 + missing)
        opp = s[:5]
        full = board + s[5:] if missing else board
        if missing:
            m = omaha_score(hole, full)
        else:
            m = mine
        o = omaha_score(opp, full)
        if m > o:
            wins += 1.0
        elif m == o:
            wins += 0.5
    dt = time.perf_counter() - t0
    c = dt / n
    memory["cost"] = 0.8 * memory["cost"] + 0.2 * c
    return wins / n


def _pick_raise(obs: Observation, rng: random.Random, big: float) -> Action:
    """Choose RAISE_POT with probability `big`, else RAISE_HALF (if legal)."""
    la = obs.legal_actions
    if Action.RAISE_POT in la and (Action.RAISE_HALF not in la or rng.random() < big):
        return Action.RAISE_POT
    if Action.RAISE_HALF in la:
        return Action.RAISE_HALF
    return Action.CHECK_CALL


def _preflop(obs: Observation, memory: dict, rng: random.Random) -> Action:
    la = obs.legal_actions
    q = _pre_pct(obs.hole_cards)
    can_raise = Action.RAISE_HALF in la or Action.RAISE_POT in la
    r = rng.random()
    aggr = _opp_aggr(memory)

    if obs.to_call <= 0:
        # Big blind with the option.
        if can_raise:
            if q >= 0.65 and r < 0.65:
                return _pick_raise(obs, rng, 0.6)
            if q < 0.25 and r < 0.10:
                return _pick_raise(obs, rng, 0.5)      # bluff
        return Action.CHECK_CALL

    if obs.raises_this_street == 0:
        # Small blind / button facing the blind: open, limp or fold.
        if can_raise:
            if q >= 0.55 and r < 0.92:
                return _pick_raise(obs, rng, 0.55)
            if q >= 0.35 and r < 0.70:
                return _pick_raise(obs, rng, 0.25)
            if q < 0.35 and q >= 0.15 and r < 0.20:
                return _pick_raise(obs, rng, 0.25)     # bluff-open
        if q >= 0.20 or obs.to_call <= obs.big_blind // 2:
            return Action.CHECK_CALL
        return Action.FOLD if Action.FOLD in la else Action.CHECK_CALL

    # Facing a raise.
    need = obs.to_call / float(obs.pot + obs.to_call)
    q_call = 0.17 + 1.15 * need + 0.05 * (obs.raises_this_street - 1)
    q_call -= 0.12 * (aggr - 0.30)                      # opponent raises a lot: defend wider
    q_call += rng.uniform(-0.04, 0.04)
    if can_raise:
        if q >= 0.90 and r < 0.80:
            return _pick_raise(obs, rng, 0.7)
        if 0.40 <= q < 0.55 and r < 0.07 and obs.raises_this_street <= 1:
            return _pick_raise(obs, rng, 0.7)           # light 3-bet bluff
    if q >= q_call:
        return Action.CHECK_CALL
    return Action.FOLD if Action.FOLD in la else Action.CHECK_CALL


def _postflop(obs: Observation, memory: dict, rng: random.Random) -> Action:
    la = obs.legal_actions
    can_raise = Action.RAISE_HALF in la or Action.RAISE_POT in la
    e = _equity(obs, memory, rng)
    r = rng.random()
    aggr = _opp_aggr(memory)
    street = obs.street

    if obs.to_call <= 0:
        # --- we may bet or check ---
        if not can_raise:
            return Action.CHECK_CALL
        pos = 1.0 if obs.is_button else 0.8
        jitter = rng.uniform(-0.03, 0.03)
        if e + jitter >= 0.82:
            # monster: mostly pot, slow-play a little
            return _pick_raise(obs, rng, 0.7) if r < 0.85 else Action.CHECK_CALL
        if e + jitter >= 0.66:
            return _pick_raise(obs, rng, 0.35) if r < 0.80 else Action.CHECK_CALL
        if e >= 0.50:
            # thin value / protection
            return Action.RAISE_HALF if (r < 0.22 and Action.RAISE_HALF in la) else Action.CHECK_CALL
        if e < 0.36:
            # bluffs: more on earlier streets (draws), fewer on river
            p = {Street.FLOP: 0.30, Street.TURN: 0.26, Street.RIVER: 0.20}[street] * pos
            if r < p:
                return _pick_raise(obs, rng, 0.35)      # mostly half-pot bluffs
        return Action.CHECK_CALL

    # --- facing a bet ---
    prev_bet = max(1, obs.pot - obs.to_call)
    b = obs.to_call / float(prev_bet)                   # bet size relative to old pot
    need = obs.to_call / float(obs.pot + obs.to_call)
    shrink = 0.11 * min(1.2, b)
    e_adj = e - shrink
    e_adj += 0.10 * (aggr - 0.30)                       # aggressive opponent bluffs more
    thr = need + 0.02 + rng.uniform(-0.03, 0.03)

    if can_raise:
        if e_adj >= 0.78 and r < 0.80:
            return _pick_raise(obs, rng, 0.65)
        if e_adj < 0.20 and r < 0.05 and street != Street.RIVER:
            return _pick_raise(obs, rng, 0.6)           # raise-bluff with air
        if e_adj < 0.12 and r < 0.04 and street == Street.RIVER:
            return _pick_raise(obs, rng, 0.6)
    if e_adj >= thr:
        return Action.CHECK_CALL
    return Action.FOLD if Action.FOLD in la else Action.CHECK_CALL


def choose_action(obs: Observation, memory: dict) -> Action:
    """Return a legal action; never raises."""
    la = obs.legal_actions
    try:
        if len(la) == 1:
            return la[0]
        t0 = time.perf_counter()
        rng = memory.get("rng") or _RNG
        _track_opponent(obs, memory)
        if memory["spent"] > _TIME_LIMIT + 1.0:
            # bank nearly gone: very cheap fallback
            if obs.to_call == 0:
                return Action.CHECK_CALL
            need = obs.to_call / float(obs.pot + obs.to_call)
            return Action.CHECK_CALL if need < 0.25 else (
                Action.FOLD if Action.FOLD in la else Action.CHECK_CALL)
        if obs.street == Street.PREFLOP:
            act = _preflop(obs, memory, rng)
        else:
            act = _postflop(obs, memory, rng)
        memory["spent"] += time.perf_counter() - t0
        if act in la:
            return act
    except Exception:
        pass
    if Action.CHECK_CALL in la:
        return Action.CHECK_CALL
    return la[0]


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Count hands (opponent tracking happens inside choose_action)."""
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
