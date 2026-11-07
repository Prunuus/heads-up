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

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import NUM_CARDS, card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
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

# Strategy: aggressive, equity-driven pressure.
#   * Equity: a small Monte Carlo estimate of our win chance against a random
#     opponent hand (random runout), scored with omaha_score. The number of
#     samples is paced so we stay far inside the time bank.
#   * Pressure: value-raise with good hands, and bluff-raise often with weak
#     ones. Bluff frequency rises when the opponent folds to our raises.
#   * Calls are loose: we call anything within a few points of the pot odds and
#     fold only when we are clearly behind.
#   * Any exception falls back to a legal CHECK_CALL or FOLD.

TIME_BUDGET_PER_HALF = 2.5    # seconds of our own compute per half (bank is 8 s; keep well under)
SECONDS_PER_SIM = 1e-4        # rough cost of one Monte Carlo sample, used for pacing
MIN_SIMS, MAX_SIMS = 6, 40    # samples per decision
DECISIONS_PER_HAND = 3        # rough average number of our decisions per hand, for pacing
CALL_SLACK = 0.05             # call when equity is within this of the pot odds
VALUE_RAISE_EQUITY = 0.65     # raise pot with at least this equity
RAISE_EQUITY = 0.5            # raise half-pot with at least this equity


def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands).

    `memory` is a fresh, empty dict every half. Store anything you want to
    remember between hands in it, e.g. memory["opp_raises"] = 0.
    `config` holds the match settings (starting_stack, big_blind, ...).

    In the tournament, each half runs in a brand-new copy of your server,
    so nothing you learn in one half carries over to the other half.
    """
    memory["hands_played"] = 0
    memory["num_hands"] = config.num_hands
    memory["spent"] = 0.0          # seconds of our own thinking this half
    memory["raised_hand"] = None   # hand_index of the last hand where we raised
    memory["raise_spots"] = 0      # hands in which we raised
    memory["folds_to_raise"] = 0   # of those, hands the opponent folded to us


def _sims_for_decision(memory: dict, hand_index: int) -> int:
    """Pace the Monte Carlo sample count against the remaining time budget."""
    hands_left = max(1, memory.get("num_hands", 1000) - hand_index)
    budget_left = TIME_BUDGET_PER_HALF - memory.get("spent", 0.0)
    per_decision = budget_left / (hands_left * DECISIONS_PER_HAND)
    sims = int(per_decision / SECONDS_PER_SIM)
    return max(MIN_SIMS, min(MAX_SIMS, sims))


def _equity(hole, board, sims: int) -> float:
    """Estimated win probability (ties count half) against a random opponent hand."""
    known = set(hole) | set(board)
    deck = [c for c in range(NUM_CARDS) if c not in known]
    need_board = 5 - len(board)
    wins = 0.0
    for _ in range(sims):
        drawn = random.sample(deck, 5 + need_board)
        opp = drawn[:5]
        full_board = tuple(board) + tuple(drawn[5:])
        mine = omaha_score(hole, full_board)
        theirs = omaha_score(opp, full_board)
        if mine > theirs:
            wins += 1.0
        elif mine == theirs:
            wins += 0.5
    return wins / sims


def _fold_rate(memory: dict) -> float:
    """Opponent's estimated fold rate to our raises (prior 0.5)."""
    folds = memory.get("folds_to_raise", 0)
    spots = memory.get("raise_spots", 0)
    return (folds + 2) / (spots + 4)


def _bluff_probability(memory: dict) -> float:
    """How often we bluff. More when the opponent folds a lot, less when they call."""
    return min(0.7, max(0.2, 0.2 + 0.6 * _fold_rate(memory)))


def _decide(obs: Observation, memory: dict) -> Action:
    """The real decision. choose_action() wraps this with a safety net."""
    legal = obs.legal_actions
    can_raise = Action.RAISE_HALF in legal or Action.RAISE_POT in legal
    to_call = obs.to_call
    # Chance we need to win to make a call profitable: call / (pot after the call).
    needed = to_call / (obs.pot + to_call) if to_call > 0 else 0.0

    equity = _equity(obs.hole_cards, obs.board, _sims_for_decision(memory, obs.hand_index))
    bluff = _bluff_probability(memory)
    roll = random.random()

    size = None  # "big" = pot-sized raise, "half" = half-pot raise
    if can_raise:
        if to_call == 0:
            if equity >= VALUE_RAISE_EQUITY:
                size = "big"
            elif equity >= RAISE_EQUITY or roll < bluff:
                size = "half"
        else:
            if equity >= VALUE_RAISE_EQUITY:
                size = "big"
            elif equity >= RAISE_EQUITY and roll < 0.7:
                size = "half"
            elif equity < needed and roll < bluff * 0.4:
                size = "half"  # bluff-raise a bet we are not getting the price to call

    if size is not None:
        memory["raised_hand"] = obs.hand_index
        if size == "big" and Action.RAISE_POT in legal:
            return Action.RAISE_POT
        if Action.RAISE_HALF in legal:
            return Action.RAISE_HALF
        return Action.RAISE_POT

    if to_call == 0:
        return Action.CHECK_CALL
    if equity + CALL_SLACK >= needed:
        return Action.CHECK_CALL
    return Action.FOLD


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.

    What you can see in obs:
        hand_index          which hand of the half this is (0, 1, 2, ...)
        street              Street.PREFLOP, Street.FLOP, Street.TURN or Street.RIVER
        hole_cards          your 5 cards as ints, sorted ascending (0-35)
        board               board (community) cards seen so far (0, 3, 4 or 5 ints)
        pot                 total chips both players put in this hand
        to_call             chips you must add to match the opponent
        my_bet, opp_bet     chips you,opponent put in on THIS street
        my_stack, opp_stack chips each player still has behind
        is_button           True if you are the button (small blind)
        legal_actions       the Actions you may return right now
        raise_to            {RAISE_HALF: total, RAISE_POT: total} for legal raises
        min_raise_to        smallest normal raise total (all-ins may be lower)
        raises_this_street  raises so far this street (max 4)
        action_history      this hand's actions so far; each has
                            street, is_me, action, amount, street_total
        big_blind           size of the big blind in chips

    Actions: Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
    Cards are ints: card_rank(c) is 0 (six) .. 8 (ace), card_suit(c) is 0..3,
    and cards_to_str(obs.hole_cards) prints them like "6h 7c 9s Kd Ah".
    Once all 5 board cards are out, omaha_score(obs.hole_cards, obs.board)
    scores your hand (higher is better).
    """
    started = time.perf_counter()
    try:
        action = _decide(obs, memory)
    except Exception:
        action = None  # any bug falls back to the safe action below
    memory["spent"] = memory.get("spent", 0.0) + (time.perf_counter() - started)

    if action not in obs.legal_actions:
        action = Action.CHECK_CALL if Action.CHECK_CALL in obs.legal_actions else Action.FOLD
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand. Optional: learn from what happened.

    To save time, the judge sends finished hands along with your next
    decision request, so this always runs BEFORE choose_action() sees the
    next hand. (The last hands of a half arrive at the end of the half.)

    result has: hand_index, my_net (chips won or lost), pot,
    went_to_showdown, board (cards revealed), opp_hole (opponent's cards,
    only after a showdown, else None), winner ('me', 'opp' or 'split').
    """
    memory["hands_played"] = memory.get("hands_played", 0) + 1
    if memory.get("raised_hand") == result.hand_index:
        # We raised in this hand. If it ended without a showdown and we won,
        # the opponent folded to our pressure.
        memory["raise_spots"] = memory.get("raise_spots", 0) + 1
        if not result.went_to_showdown and result.winner == "me":
            memory["folds_to_raise"] = memory.get("folds_to_raise", 0) + 1
        memory["raised_hand"] = None

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
