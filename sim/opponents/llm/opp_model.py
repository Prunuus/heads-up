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

from evaluator import TWO_PAIR, ONE_PAIR, category_of

TIME_TARGET = 5.0      # seconds of the 8 s bank we plan to use per half
TIME_HARD = 6.6        # past this, stop simulating entirely


def _pre_score(hole) -> float:
    """Cheap preflop hand-quality heuristic (pairs, high cards, suits, connectivity)."""
    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]
    s = 0.0
    cnt = {}
    for r in ranks:
        cnt[r] = cnt.get(r, 0) + 1
    for r, k in cnt.items():
        if k == 2:
            s += 1.2 + 0.08 * r
        elif k == 3:
            s += 0.4
        elif k >= 4:
            s -= 1.0
    s += sum(r / 8.0 for r in ranks) * 0.45
    by_suit = {}
    for c in hole:
        by_suit.setdefault(c & 3, []).append(c >> 2)
    groups = sorted((sorted(v, reverse=True) for v in by_suit.values() if len(v) >= 2),
                    key=lambda v: -v[0])
    for g in groups[:2]:
        s += 0.35 + 0.35 * (g[0] / 8.0) + (0.1 if len(g) == 2 else -0.1 * (len(g) - 2))
    best = 0
    uniq = set(ranks)
    for lo in range(-1, 5):          # lo=-1 : ace-low window
        win = set(range(lo, lo + 5)) if lo >= 0 else {8, 0, 1, 2, 3}
        best = max(best, len(uniq & win))
    s += 0.3 * max(0, best - 2)
    return s


_PRE_TABLE = []


def _build_pre_table():
    rng = random.Random(2024)
    _PRE_TABLE.extend(sorted(_pre_score(rng.sample(range(36), 5)) for _ in range(2500)))


_build_pre_table()


def _pre_equity(hole) -> float:
    pct = bisect.bisect_left(_PRE_TABLE, _pre_score(hole)) / len(_PRE_TABLE)
    return 0.36 + 0.28 * pct


def _mc_equity(hole, board, deadline, max_sims, rng) -> float:
    dead = set(hole) | set(board)
    deck = [c for c in range(36) if c not in dead]
    need = 5 - len(board)
    k = 5 + need
    board = list(board)
    my_fixed = omaha_score(hole, board) if need == 0 else None
    total = 0.0
    n = 0
    while n < max_sims:
        draw = rng.sample(deck, k)
        full = board + draw[5:] if need else board
        mine = my_fixed if need == 0 else omaha_score(hole, full)
        theirs = omaha_score(draw[:5], full)
        if mine > theirs:
            total += 1.0
        elif mine == theirs:
            total += 0.5
        n += 1
        if n >= 6 and (n & 3) == 0 and time.perf_counter() > deadline:
            break
    return total / n if n else 0.5


def new_half(config: AgentConfig, memory: dict) -> None:
    """Set up opponent-model counters (Beta-style priors) and bookkeeping."""
    memory["hands_played"] = 0
    memory["rng"] = random.Random(987654321)
    memory["time_used"] = 0.0
    memory["num_hands"] = getattr(config, "num_hands", 1000)
    # opponent model
    memory["opp_raises"] = 0          # opponent raise actions seen
    memory["opp_acts"] = 0            # opponent non-fold actions seen
    memory["opp_pre_raises"] = 0
    memory["opp_pre_acts"] = 0
    memory["my_bet_hands"] = 0        # hands in which I bet/raised
    memory["opp_folds_to_me"] = 0     # of those, hands won by opp folding
    memory["opp_sd_raise"] = 0        # showdowns where opp had raised
    memory["opp_sd_weak"] = 0         # ... and showed a weak hand (<= one pair)
    # per-hand scratch
    memory["cur_hand"] = -1
    memory["hist_ptr"] = 0
    memory["i_bet"] = False
    memory["opp_raised"] = False


def _opp_stats(memory: dict):
    raise_rate = (memory["opp_raises"] + 2.5) / (memory["opp_acts"] + 10.0)         # prior 0.25
    fold_rate = (memory["opp_folds_to_me"] + 3.0) / (memory["my_bet_hands"] + 8.0)  # prior 0.375
    bluff = (memory["opp_sd_weak"] + 1.0) / (memory["opp_sd_raise"] + 4.0)          # prior 0.25
    return raise_rate, fold_rate, bluff


def _update_from_history(obs: Observation, memory: dict) -> None:
    if obs.hand_index != memory["cur_hand"]:
        memory["cur_hand"] = obs.hand_index
        memory["hist_ptr"] = 0
        memory["i_bet"] = False
        memory["opp_raised"] = False
    hist = obs.action_history
    for a in hist[memory["hist_ptr"]:]:
        if not a.is_me:
            memory["opp_acts"] += 1
            pre = a.street == Street.PREFLOP
            if pre:
                memory["opp_pre_acts"] += 1
            if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
                memory["opp_raises"] += 1
                memory["opp_raised"] = True
                if pre:
                    memory["opp_pre_raises"] += 1
    memory["hist_ptr"] = len(hist)


def _decide(obs: Observation, memory: dict) -> Action:
    _update_from_history(obs, memory)
    legal = obs.legal_actions
    rng = memory["rng"]
    raise_rate, fold_rate, bluff = _opp_stats(memory)
    to_call = obs.to_call
    can_half = Action.RAISE_HALF in legal
    can_pot = Action.RAISE_POT in legal
    can_raise = can_half or can_pot

    # ---- equity (vs a random hand), time-budgeted ----
    used = memory["time_used"]
    if obs.street == Street.PREFLOP:
        eq = _pre_equity(obs.hole_cards)
    elif used > TIME_HARD:
        eq = 0.5  # out of budget: neutral, play cheap
    else:
        remaining_dec = max(1.0, (memory["num_hands"] - memory["hands_played"]) * 2.6)
        per = max(0.0004, min(0.006, (TIME_TARGET - used) / remaining_dec))
        deadline = time.perf_counter() + per
        eq = _mc_equity(obs.hole_cards, obs.board, deadline, 200, rng)

    # ---- adjust equity for the range a bettor has ----
    facing_bet = to_call > 0 and (obs.street != Street.PREFLOP or to_call > obs.big_blind
                                  or memory["opp_raised"])
    eq_adj = eq
    if facing_bet:
        frac = to_call / max(1.0, obs.pot - to_call)
        tight = min(1.4, max(0.35, 1.45 - 1.2 * raise_rate - 1.0 * bluff))
        eq_adj = eq - (0.04 + 0.10 * min(frac, 1.5)) * tight
    if not obs.is_button and obs.street != Street.PREFLOP:
        eq_adj -= 0.01

    # ---- strategy thresholds that adapt to the opponent ----
    value_th = 0.64 - 0.08 * (0.4 - fold_rate)
    value_th = min(0.68, max(0.58, value_th))
    if obs.street == Street.PREFLOP:
        value_th += 0.02
    bluff_p = min(0.45, max(0.0, (fold_rate - 0.35) * 1.0))
    if obs.street == Street.RIVER:
        bluff_p *= 0.8

    pot_odds = to_call / (obs.pot + to_call) if to_call > 0 else 0.0

    def raise_act(strong: bool) -> Action:
        if strong and can_pot:
            return Action.RAISE_POT
        if can_half:
            return Action.RAISE_HALF
        return Action.RAISE_POT

    if to_call == 0:
        if can_raise and eq_adj >= value_th:
            return raise_act(eq_adj >= value_th + 0.12)
        if can_raise and eq_adj < 0.5 and rng.random() < bluff_p:
            return Action.RAISE_HALF if can_half else Action.RAISE_POT
        return Action.CHECK_CALL

    # facing a bet / raise
    extra = 0.04 * obs.raises_this_street
    if can_raise and eq_adj >= value_th + 0.08 + extra:
        return raise_act(eq_adj >= 0.80)
    if (can_raise and obs.raises_this_street <= 1 and 0.30 <= eq <= 0.5
            and fold_rate > 0.55 and rng.random() < (fold_rate - 0.5) * 0.5):
        return Action.RAISE_HALF if can_half else Action.RAISE_POT
    margin = 0.03 + (0.02 if obs.street == Street.PREFLOP and not obs.is_button else 0.0)
    if eq_adj > pot_odds + margin:
        return Action.CHECK_CALL
    if Action.FOLD in legal:
        return Action.FOLD
    return Action.CHECK_CALL


def choose_action(obs: Observation, memory: dict) -> Action:
    """Equity-based play (preflop heuristic, postflop Monte Carlo) that adapts
    bluffing, value-betting and calling to the tracked opponent profile."""
    t0 = time.perf_counter()
    try:
        act = _decide(obs, memory)
        if act in (Action.RAISE_HALF, Action.RAISE_POT) and act in obs.legal_actions:
            memory["i_bet"] = True
    except Exception:
        act = Action.CHECK_CALL if obs.to_call <= max(obs.big_blind, obs.pot * 0.25) else Action.FOLD
    if act not in obs.legal_actions:
        act = Action.CHECK_CALL
    memory["time_used"] = memory.get("time_used", 0.0) + (time.perf_counter() - t0)
    return act


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Update the opponent profile: fold-to-aggression and bluff-at-showdown."""
    memory["hands_played"] += 1
    if memory.get("i_bet"):
        memory["my_bet_hands"] += 1
        if result.winner == "me" and not result.went_to_showdown:
            memory["opp_folds_to_me"] += 1
    if result.went_to_showdown and memory.get("opp_raised") and result.opp_hole and len(result.board) == 5:
        try:
            cat = category_of(omaha_score(result.opp_hole, result.board))
            memory["opp_sd_raise"] += 1
            if cat <= ONE_PAIR:
                memory["opp_sd_weak"] += 1
        except Exception:
            pass
    memory["i_bet"] = False
    memory["opp_raised"] = False
    memory["cur_hand"] = -1  # next observation starts a fresh hand scratch

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
