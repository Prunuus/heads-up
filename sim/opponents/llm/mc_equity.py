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
from itertools import combinations

from evaluator import get_table

_PAIRS = tuple(combinations(range(5), 2))
_TRIPLES = tuple(combinations(range(5), 3))
_ALL = tuple(range(36))

SIM_TOTAL_BUDGET = 3.0     # seconds of our own sim time per half (bank is 8s)
MAX_PER_DECISION = 0.008   # never sim longer than this for one decision
MIN_SIMS = 6


def new_half(config: AgentConfig, memory: dict) -> None:
    """Fresh per-half state: time accounting and light opponent stats."""
    memory["hands_played"] = 0
    memory["decisions"] = 0
    memory["used"] = 0.0
    memory["num_hands"] = getattr(config, "num_hands", 1000) or 1000
    memory["opp_raise_seen"] = 0
    memory["opp_actions_seen"] = 0
    memory["last_hand_counted"] = -1
    memory["rng"] = random.Random(12345)


def _equity(hole, board, deadline, rng):
    """Monte Carlo equity vs a uniformly random opponent hand.
    Runs until `deadline` (perf_counter) or a sim cap; always >= MIN_SIMS."""
    table = get_table()
    known = set(hole) | set(board)
    deck = [c for c in _ALL if c not in known]
    need = 5 - len(board)
    hb = [1 << c for c in hole]
    my_masks = [hb[i] | hb[j] for i, j in _PAIRS]
    sample = rng.sample
    pc = time.perf_counter
    fixed_board = need == 0
    if fixed_board:
        bb = [1 << c for c in board]
        bms = [bb[i] | bb[j] | bb[k] for i, j, k in _TRIPLES]
        my_fixed = max([table[h | b] for h in my_masks for b in bms])
    score = 0.0
    n = 0
    cap = 400
    while n < cap:
        if n >= MIN_SIMS and pc() >= deadline:
            break
        s = sample(deck, need + 5)
        if not fixed_board:
            full = list(board) + s[:need]
            bb = [1 << c for c in full]
            bms = [bb[i] | bb[j] | bb[k] for i, j, k in _TRIPLES]
            my = max([table[h | b] for h in my_masks for b in bms])
        else:
            my = my_fixed
        ob = [1 << c for c in s[need:]]
        om = [ob[i] | ob[j] for i, j in _PAIRS]
        opp = max([table[h | b] for h in om for b in bms])
        if my > opp:
            score += 1.0
        elif my == opp:
            score += 0.5
        n += 1
    return score / n, n


def _decide(obs: Observation, memory: dict) -> Action:
    legal = obs.legal_actions
    rng = memory["rng"]
    t0 = time.perf_counter()

    # --- time budget for this decision ---
    hands_done = max(memory["hands_played"], 1)
    dec_per_hand = max(1.5, memory["decisions"] / hands_done) if memory["decisions"] > 20 else 2.5
    hands_left = max(memory["num_hands"] - obs.hand_index, 1)
    remaining = max(SIM_TOTAL_BUDGET - memory["used"], 0.0)
    budget = remaining / (hands_left * dec_per_hand)
    budget *= 0.7 if obs.street == Street.PREFLOP else 1.2
    budget = min(budget, MAX_PER_DECISION)
    eq, _n = _equity(obs.hole_cards, obs.board, t0 + budget, rng)

    to_call = obs.to_call
    pot = obs.pot
    can_raise = Action.RAISE_POT in legal or Action.RAISE_HALF in legal

    def raise_with(prefer_pot):
        if prefer_pot and Action.RAISE_POT in legal:
            return Action.RAISE_POT
        if Action.RAISE_HALF in legal:
            return Action.RAISE_HALF
        if Action.RAISE_POT in legal:
            return Action.RAISE_POT
        return Action.CHECK_CALL

    # opponent aggression estimate (prior-weighted)
    seen = memory["opp_actions_seen"]
    agg = (memory["opp_raise_seen"] + 2) / (seen + 6)  # prior ~0.33

    if to_call > 0:
        pot_before = max(pot - to_call, 1)
        bet_ratio = to_call / pot_before
        pot_odds = to_call / (pot + to_call)
        # opponent's betting range is stronger than random: shade equity
        shade = 0.12 * min(1.0, bet_ratio) + 0.03 * max(0, obs.raises_this_street - 1)
        # a passive opponent's raise means more; an aggressive one means less
        shade *= 1.3 - min(agg, 0.7)
        if obs.street == Street.PREFLOP:
            shade *= 0.6
        e = eq - shade
        if e < pot_odds + 0.01:
            # occasional bluff-raise when we hold some equity and are cheap
            if (can_raise and obs.street != Street.RIVER and eq > 0.38
                    and bet_ratio < 0.7 and rng.random() < 0.06):
                return raise_with(False)
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
        if can_raise:
            if e > 0.72:
                return raise_with(True)
            if e > 0.62 and rng.random() < 0.85:
                return raise_with(e > 0.68)
            if e > 0.55 and bet_ratio < 0.6 and rng.random() < 0.25:
                return raise_with(False)
        return Action.CHECK_CALL

    # --- nothing to call: bet or check ---
    if can_raise:
        if eq > 0.74:
            # mostly bet pot; occasionally slow-play
            if rng.random() < 0.9:
                return raise_with(True)
            return Action.CHECK_CALL
        if eq > 0.60:
            return raise_with(eq > 0.67)
        if eq > 0.52 and rng.random() < 0.4:
            return raise_with(False)
        # semi-bluff / steal with weak hands occasionally
        if obs.street != Street.RIVER and eq > 0.33 and rng.random() < 0.10:
            return raise_with(False)
        if obs.street == Street.RIVER and eq < 0.30 and rng.random() < 0.07:
            return raise_with(False)
    return Action.CHECK_CALL


def choose_action(obs: Observation, memory: dict) -> Action:
    """Return a legal action: Monte Carlo equity vs pot odds."""
    t0 = time.perf_counter()
    legal = obs.legal_actions
    try:
        # opponent aggression tracking: count opp actions seen this decision point
        hist = obs.action_history
        if hist:
            last = hist[-1]
            if not last.is_me:
                memory["opp_actions_seen"] += 1
                if last.action in (Action.RAISE_HALF, Action.RAISE_POT):
                    memory["opp_raise_seen"] += 1
        action = _decide(obs, memory)
        if action not in legal:
            action = Action.CHECK_CALL if Action.CHECK_CALL in legal else legal[0]
    except Exception:
        action = Action.CHECK_CALL if Action.CHECK_CALL in legal else legal[0]
    memory["decisions"] += 1
    memory["used"] += time.perf_counter() - t0
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Count finished hands (used for time budgeting)."""
    memory["hands_played"] += 1

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
