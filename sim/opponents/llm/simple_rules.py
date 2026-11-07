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

# Simple rules-based strategy: estimate hand equity by Monte Carlo (random
# opponent hand, random completion of the board), then compare it with
# fixed thresholds and the pot odds. Every decision falls back to a legal
# action if anything goes wrong.

import random

from evaluator import NUM_CARDS

EQUITY_SAMPLES = 50          # Monte Carlo samples per decision (about 1 ms in total)
VERY_STRONG_EQUITY = 0.78    # bet/raise the pot
STRONG_EQUITY = 0.65         # bet/raise half the pot
CALL_MARGIN = 0.04           # call only when equity beats the pot odds by this much
MAX_RAISES_TO_RAISE = 3      # with 3 or more raises already on the street, just call or check


def _equity(hole, board, samples: int, seed: str) -> float:
    """Share of the pot we expect to win against a random opponent hand.

    The seed is built from the spot itself, so the same spot always gives
    the same estimate (repeatable and easy to debug).
    """
    rng = random.Random(seed)
    known = set(hole) | set(board)
    deck = [c for c in range(NUM_CARDS) if c not in known]
    missing_board = 5 - len(board)
    score = 0.0
    for _ in range(samples):
        draw = rng.sample(deck, 5 + missing_board)
        opp_hole = draw[:5]
        full_board = tuple(board) + tuple(draw[5:])
        mine = omaha_score(hole, full_board)
        theirs = omaha_score(opp_hole, full_board)
        if mine > theirs:
            score += 1.0
        elif mine == theirs:
            score += 0.5
    return score / samples


def _decide(obs: Observation) -> Action:
    """The strategy itself. choose_action() wraps it with a safety net."""
    hole = tuple(obs.hole_cards)
    board = tuple(obs.board)
    seed = f"{obs.hand_index}|{int(obs.street)}|{hole}|{board}"
    eq = _equity(hole, board, EQUITY_SAMPLES, seed)

    raise_ok = (Action.RAISE_HALF in obs.legal_actions
                and obs.raises_this_street < MAX_RAISES_TO_RAISE)
    if eq >= VERY_STRONG_EQUITY and raise_ok:
        return Action.RAISE_POT
    if eq >= STRONG_EQUITY and raise_ok:
        return Action.RAISE_HALF

    if obs.to_call > 0:
        # Pot odds: calling is worth it when our equity beats the share of
        # the final pot that the call costs us (plus a small safety margin).
        needed = obs.to_call / (obs.pot + obs.to_call)
        if eq >= needed + CALL_MARGIN:
            return Action.CHECK_CALL
        return Action.FOLD
    return Action.CHECK_CALL


def new_half(config: AgentConfig, memory: dict) -> None:
    """Nothing to carry between hands except a counter."""
    memory["hands_played"] = 0


def choose_action(obs: Observation, memory: dict) -> Action:
    """Pick an action from the hand strength and the pot odds."""
    try:
        action = _decide(obs)
    except Exception:
        action = Action.CHECK_CALL
    # Never return anything outside obs.legal_actions: check when possible, else fold.
    if action in obs.legal_actions:
        return action
    return Action.CHECK_CALL if Action.CHECK_CALL in obs.legal_actions else Action.FOLD


def on_hand_end(result: HandResult, memory: dict) -> None:
    """No learning in this version."""
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
