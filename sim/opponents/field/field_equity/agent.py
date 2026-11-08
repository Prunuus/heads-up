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
PARTICIPANT = "EquityEdge"
AGENT_NAME = "RangeEquity"


# ===== YOUR CODE HERE =====
#
# EquityEdge: a range-aware equity engine.
#   * Vectorized numpy Omaha evaluator (exact same scores as evaluator.py).
#   * Monte Carlo equity against the opponent's ESTIMATED range: sampled
#     opponent hands are importance-weighted by how likely they were to take
#     the actions they took this hand (preflop table percentile, postflop
#     made-hand/draw strength), with action frequencies learned this half.
#   * Decisions from equity vs pot odds, position, SPR; sizing by equity.

import time
from itertools import combinations

import numpy as np

from evaluator import score_five

_HERE = os.path.dirname(os.path.abspath(__file__))

# ----- 5-card score tables, keyed by sum(5**rank) (+ flush offset) ---------
_KEY_BASE = 5 ** 9
_RK5 = np.array([5 ** (c >> 2) for c in range(36)], dtype=np.int32)
_SUIT = np.array([c & 3 for c in range(36)], dtype=np.int32)


def _build_score_table() -> np.ndarray:
    table = np.zeros(2 * _KEY_BASE, dtype=np.int32)
    # every 5-card rank multiset (each rank at most 4 times)
    for combo in combinations(range(9 + 4), 5):          # stars and bars
        ranks = []
        r = 0
        prev = -1
        # decode multiset from combinations-with-repetition encoding
        for i, x in enumerate(combo):
            ranks.append(x - i)
        counts = {}
        for rr in ranks:
            counts[rr] = counts.get(rr, 0) + 1
        if max(counts.values()) > 4:
            continue
        cards = []
        used = {}
        for rr in ranks:
            s = used.get(rr, 0)
            used[rr] = s + 1
            cards.append(rr * 4 + s)
        key = sum(5 ** rr for rr in ranks)
        if len(counts) == 5:
            # non-flush version: make sure suits differ
            nf = [rr * 4 for rr in ranks]
            nf[0] += 1
            table[key] = score_five(nf)
            table[key + _KEY_BASE] = score_five([rr * 4 for rr in ranks])
        else:
            table[key] = score_five(cards)
    return table


_SCORE_T = _build_score_table()

_PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.int64)      # (10, 2)
_TRIPLES5 = np.array(list(combinations(range(5), 3)), dtype=np.int64)   # (10, 3)
_TRIPLES4 = np.array(list(combinations(range(4), 3)), dtype=np.int64)   # (4, 3)


def _pair_parts(holes: np.ndarray):
    """holes (..., 5) -> pair keys (..., 10) and pair suit codes (..., 10)."""
    a = holes[..., _PAIRS[:, 0]]
    b = holes[..., _PAIRS[:, 1]]
    key = _RK5[a] + _RK5[b]
    sa = _SUIT[a]
    code = np.where(sa == _SUIT[b], sa, np.int32(4))
    return key, code


def _triple_parts(boards: np.ndarray, triples: np.ndarray):
    """boards (..., k) -> triple keys (..., T) and triple suit codes (..., T)."""
    a = boards[..., triples[:, 0]]
    b = boards[..., triples[:, 1]]
    c = boards[..., triples[:, 2]]
    key = _RK5[a] + _RK5[b] + _RK5[c]
    sa = _SUIT[a]
    code = np.where((sa == _SUIT[b]) & (sa == _SUIT[c]), sa, np.int32(5))
    return key, code


def _best_scores(pk, pc, tk, tc) -> np.ndarray:
    """pk/pc (N or 1, 10), tk/tc (N or 1, T) -> best Omaha score per row."""
    idx = pk[:, :, None] + tk[:, None, :]
    np.add(idx, np.int32(_KEY_BASE), out=idx, where=(pc[:, :, None] == tc[:, None, :]))
    return _SCORE_T.take(idx.reshape(idx.shape[0], -1)).max(axis=1)


def omaha_scores(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Vectorized omaha_score: holes (N,5) or (5,), boards (N,5) or (5,)."""
    holes = np.atleast_2d(holes)
    boards = np.atleast_2d(boards)
    pk, pc = _pair_parts(holes)
    tk, tc = _triple_parts(boards, _TRIPLES5)
    return _best_scores(pk, pc, tk, tc)


def partial_scores(holes: np.ndarray, board: np.ndarray, parts=None) -> np.ndarray:
    """Best 2+3 made hand on a 3, 4 or 5 card board (current strength), holes (N,5)."""
    b = np.asarray(board, dtype=np.int64)[None, :]
    tri = _TRIPLES4 if b.shape[1] == 4 else (_TRIPLES5 if b.shape[1] == 5 else np.array([[0, 1, 2]]))
    pk, pc = parts if parts is not None else _pair_parts(np.atleast_2d(holes))
    tk, tc = _triple_parts(b, tri)
    return _best_scores(pk, pc, tk, tc)


# ----- combinatorial index of a 5-card hand (preflop table) ----------------
_BINOM = np.zeros((37, 6), dtype=np.int64)
for _n in range(37):
    for _k in range(6):
        _BINOM[_n, _k] = 0 if _k > _n else int(np.round(np.prod([(_n - i) / (i + 1) for i in range(_k)])))


def hand_index(holes: np.ndarray) -> np.ndarray:
    """holes (N,5) -> index 0..C(36,5)-1 (cards need not be sorted)."""
    h = np.sort(np.atleast_2d(holes), axis=1)
    return (_BINOM[h[:, 0], 1] + _BINOM[h[:, 1], 2] + _BINOM[h[:, 2], 3]
            + _BINOM[h[:, 3], 4] + _BINOM[h[:, 4], 5])


# ----- straight-draw table over (hole rank mask, board rank mask) ----------
def _build_draw_table() -> np.ndarray:
    straights = [0b100001111]  # wheel A-6-7-8-9
    straights += [0b11111 << lo for lo in range(5)]
    m = np.arange(512)
    pop = np.array([bin(x).count("1") for x in range(512)])
    H = m[:, None]
    B = m[None, :]

    def has_straight(Bm):
        ok = np.zeros((512, 512), bool)
        for s in straights:
            ok |= (((s & ~(H | Bm)) == 0) & (pop[s & ~Bm] <= 2) & (pop[s & ~H] <= 3))
        return ok

    base = has_straight(B)
    outs = np.zeros((512, 512), np.int8)
    for r in range(9):
        outs += (has_straight(B | (1 << r)) & ~base & ((B >> r) & 1 == 0)).astype(np.int8)
    return outs


_DRAW_T = _build_draw_table()
_RANKBIT = np.array([1 << (c >> 2) for c in range(36)], dtype=np.int64)

# ----- preflop table: equity of each 5-card hand vs a random hand ----------
try:
    _PRE_EQ = np.load(os.path.join(_HERE, "preflop_eq.npy")).astype(np.float32)
    assert _PRE_EQ.shape == (376992,)
except Exception:  # fallback: crude feature score, never crash
    _PRE_EQ = None


def _fallback_pre(holes: np.ndarray) -> np.ndarray:
    holes = np.atleast_2d(holes)
    r = holes >> 2
    s = holes & 3
    hi = r.sum(axis=1) / 40.0
    suited = np.zeros(len(holes))
    for k in range(4):
        cnt = (s == k).sum(axis=1)
        suited += (cnt >= 2) * 0.03
    pairs = np.zeros(len(holes))
    for k in range(9):
        cnt = (r == k).sum(axis=1)
        pairs += (cnt == 2) * (0.02 + 0.004 * k)
    return (0.35 + 0.15 * hi + suited + pairs).astype(np.float32)


def pre_strength(holes: np.ndarray) -> np.ndarray:
    if _PRE_EQ is None:
        return _fallback_pre(holes)
    return _PRE_EQ[hand_index(holes)]


if _PRE_EQ is not None:
    _PRE_SORTED = np.sort(_PRE_EQ)
else:
    _PRE_SORTED = None


def pre_percentile(eq: float) -> float:
    if _PRE_SORTED is None:
        return float(min(1.0, max(0.0, (eq - 0.35) / 0.3)))
    return float(np.searchsorted(_PRE_SORTED, eq) / len(_PRE_SORTED))


# ----- random permutation banks for sampling the unseen cards --------------
_BANK_ROWS = 8192
_rng0 = np.random.default_rng(20261107)
_PERM_BANK = {R: _rng0.random((_BANK_ROWS, R)).argsort(axis=1)[:, :10].astype(np.int64)
              for R in (31, 28, 27, 26)}


# ----- parameters (tuned with the local harness) ---------------------------
P = {
    # preflop first-in (percentile of preflop equity table)
    "sb_raise": 0.50, "sb_call": 0.12,
    "bb_iso": 0.60,
    # range model
    "temp": 0.08, "floor": 0.10, "chk_trap": 0.35,
    # equity realization factors (call/check EV) by street, OOP / IP
    "real_oop": (0.88, 0.90, 0.95, 1.0), "real_ip": (0.95, 0.97, 1.0, 1.0),
    # preflop call margin over pot odds (positive = tighter)
    "pre_margin": 0.02,
    # bet/raise EV haircut (fraction of pot) to cover re-raises / variance
    "bet_margin": 0.05,
    "pre_raise_eq": 0.56,
    "samples": 250, "samples_big": 600,
}

# opponent model priors: spot -> pseudo-counts [fold/check, call, raise]
_PRIORS = {
    "pre_open": (1.0, 3.0, 4.0),
    "pre_vs_limp": (0.0, 5.0, 3.0),
    "pre_vs_raise": (2.5, 4.0, 1.5),
    "post_open": (0.0, 5.0, 4.0),
    "post_vs_bet": (3.5, 3.5, 1.0),
}


def new_half(config: AgentConfig, memory: dict) -> None:
    """Fresh per-half state: opponent stats, time accounting, rng."""
    memory["hands_played"] = 0
    memory["rng"] = np.random.default_rng(int(time.time() * 1000) % (2 ** 32))
    memory["stats"] = {k: [0, 0, 0] for k in _PRIORS}
    memory["sd"] = {"aggr": 0, "bluff": 0}
    memory["t_used"] = 0.0
    memory["n_dec"] = 0
    memory["num_hands"] = int(getattr(config, "num_hands", 1000) or 1000)
    memory["bank"] = float(getattr(config, "time_bank_seconds", 8.0) or 8.0)
    memory["hand"] = None
    memory["cost"] = 1.2e-6          # seconds per MC sample (EMA, measured)
    memory["fix"] = 2.5e-4           # seconds of non-MC work per decision (EMA)
    memory["mc_t"] = 0.0
    memory["n_mc"] = 0
    memory["slow"] = _SLOW           # current CPU slowdown estimate (re-measured)
    memory["oh_spent"] = 0.0         # estimated server overhead charged so far
    memory["load"] = 1.0             # measured MC time / idle-start MC time (EMA)


def _freq(memory, spot):
    c = memory["stats"][spot]
    pr = _PRIORS[spot]
    tot = c[0] + c[1] + c[2] + pr[0] + pr[1] + pr[2]
    return ((c[0] + pr[0]) / tot, (c[1] + pr[1]) / tot, (c[2] + pr[2]) / tot)


def _new_hand_state(obs):
    return {"idx": obs.hand_index, "processed": 0, "events": [], "awaiting": None,
            "last_my": None, "button": obs.is_button, "opp_aggr_streets": set()}


def _parse_history(obs, memory):
    """Walk this hand's history; record new opponent events and update stats."""
    hs = memory["hand"]
    my_bet = 1 if obs.is_button else 2
    opp_bet = 2 if obs.is_button else 1
    street = 0
    hist = obs.action_history
    for i, a in enumerate(hist):
        st = int(a.street)
        if st != street:
            street = st
            my_bet = opp_bet = 0
        if a.is_me:
            if i >= hs["processed"]:
                hs["awaiting"] = None
            my_bet = a.street_total
            continue
        to_call = my_bet - opp_bet
        act = int(a.action)
        cls = 0 if act == 0 else (1 if act == 1 else 2)
        if st == 0:
            if i == 0 and not obs.is_button:
                spot = "pre_open"
            elif to_call <= 0:
                spot = "pre_vs_limp"
            else:
                spot = "pre_vs_raise"
        else:
            spot = "post_open" if to_call <= 0 else "post_vs_bet"
        if i >= hs["processed"]:
            memory["stats"][spot][cls] += 1
            hs["events"].append((st, spot, cls))
            hs["awaiting"] = None
            if cls == 2 and st > 0:
                hs["opp_aggr_streets"].add(st)
        opp_bet = a.street_total
    hs["processed"] = len(hist)


def _weighted_pct(raw, w):
    order = np.argsort(raw, kind="stable")
    ws = w[order]
    cw = np.cumsum(ws)
    tot = cw[-1] if cw[-1] > 0 else 1.0
    pct = np.empty_like(raw, dtype=np.float64)
    pct[order] = (cw - 0.5 * ws) / tot
    # equal raw values share the same (average) percentile
    return pct


def _street_strength(opp, board, street, parts=None):
    """Percentile-ready raw strength of sampled opponent hands on a street."""
    b = board[:{1: 3, 2: 4, 3: 5}[street]]
    made = partial_scores(opp, b, parts).astype(np.float64)
    if street == 3:
        return made, None
    # draws: flush draw (2+ suited hole cards matching 2 board cards) and straight outs
    bsuits = np.array([c & 3 for c in b])
    fd = np.zeros(len(opp), bool)
    for s in range(4):
        nb = int((bsuits == s).sum())
        if nb == 2:
            fd |= ((opp & 3) == s).sum(axis=1) >= 2
    bmask = 0
    for c in b:
        bmask |= 1 << (c >> 2)
    hmask = np.bitwise_or.reduce(_RANKBIT[opp], axis=1)
    outs = _DRAW_T[hmask, bmask]
    draw = np.where(fd & (outs >= 2), 0.86, np.where(fd, 0.72, np.where(outs >= 3, 0.74, np.where(outs >= 2, 0.64, 0.0))))
    return made, draw


def _range_weights(opp, board, events, memory, cache, parts=None):
    """Importance weights for sampled opponent hands given their actions."""
    n = len(opp)
    w = np.ones(n)
    T = P["temp"]
    # target share of bluffs inside an aggressive range (learned at showdowns)
    b = (memory["sd"]["bluff"] + P["floor"] * 8.0) / (memory["sd"]["aggr"] + 8.0)
    b = min(0.45, max(0.02, b))
    last_pct = None
    for (st, spot, cls) in events:
        if st not in cache:
            if st == 0:
                cache[st] = (pre_strength(opp).astype(np.float64), None)
            else:
                cache[st] = _street_strength(opp, board, st, parts)
        made, draw = cache[st]
        pct = _weighted_pct(made, w)
        if draw is not None:
            pct = np.maximum(pct, draw)
        f = _freq(memory, spot)
        if cls == 2:     # bet / raise: top f_raise of the range (+ bluffs)
            fr = min(0.97, max(0.02, f[2]))
            floor = min(1.0, b * fr / ((1 - b) * (1 - fr)))
            w = w * (floor + (1 - floor) / (1 + np.exp(-(pct - (1 - fr)) / T)))
        elif cls == 1:
            if spot in ("pre_vs_limp", "post_open"):   # check: strong hands less likely
                fr = f[2]
                w = w * (1 - (1 - P["chk_trap"]) / (1 + np.exp(-(pct - (1 - fr)) / T)))
            else:        # call: drop the bottom f_fold of the range
                ff = f[0]
                w = w * (0.04 + 0.96 / (1 + np.exp(-(pct - ff) / T)))
                if spot == "pre_open":   # a limp caps the range a little
                    w = w * (1 - 0.5 / (1 + np.exp(-(pct - (1 - f[2])) / T)))
        last_pct = pct
    return w


def _equity(obs, memory, n, want_continue=None):
    """Weighted MC equity vs the estimated range.
    Returns (E, Ec, ff): Ec = equity vs the part of the range that continues
    against a bet, ff = chance the current range folds to a bet, given that
    the opponent folds the weakest `want_continue` share of all hands."""
    hero = np.array(obs.hole_cards, dtype=np.int64)
    board = np.array(obs.board, dtype=np.int64)
    known = np.zeros(36, bool)
    known[hero] = True
    known[board] = True
    rem = np.nonzero(~known)[0]
    R = len(rem)
    bank = _PERM_BANK[R]
    rng = memory["rng"]
    rows = rng.integers(0, bank.shape[0], n)
    cards = rem[bank[rows]]
    opp = cards[:, :5]
    k = len(board)
    hp = _pair_parts(hero[None, :])
    op = _pair_parts(opp)
    if k < 5:
        boards = np.concatenate([np.broadcast_to(board, (n, k)), cards[:, 5:10 - k]], axis=1)
        tp = _triple_parts(boards, _TRIPLES5)
        hs = _best_scores(hp[0], hp[1], tp[0], tp[1])
    else:
        tp = _triple_parts(board[None, :], _TRIPLES5)
        hs = _best_scores(hp[0], hp[1], tp[0], tp[1])[0]
    os_ = _best_scores(op[0], op[1], tp[0], tp[1])
    res = (hs > os_) + 0.5 * (hs == os_)
    cache = {}
    events = memory["hand"]["events"]
    w = _range_weights(opp, board, events, memory, cache, op)
    sw = w.sum()
    E = float((w * res).sum() / sw) if sw > 0 else float(res.mean())
    Ec, ff_eff = E, 0.0
    if want_continue is not None:
        st = int(obs.street)
        if st == 0:
            raw = pre_strength(opp).astype(np.float64)
            abs_pct = np.searchsorted(_PRE_SORTED, raw) / len(_PRE_SORTED) if _PRE_SORTED is not None \
                else _weighted_pct(raw, np.ones(n))
        else:
            if st not in cache:
                cache[st] = _street_strength(opp, board, st, op)
            made, draw = cache[st]
            abs_pct = _weighted_pct(made, np.ones(n))
            if draw is not None:
                abs_pct = np.maximum(abs_pct, draw)
        # the opponent folds hands below an ABSOLUTE strength threshold, so a
        # range that is already strong folds much less than its average
        cont = 0.02 + 0.98 / (1 + np.exp(-(abs_pct - want_continue) / P["temp"]))
        wc = w * cont
        swc = wc.sum()
        ff_eff = float(1.0 - swc / sw) if sw > 0 else want_continue
        Ec = float((wc * res).sum() / swc) if swc > 0 else E
    return E, Ec, ff_eff


def _make_bench():
    try:
        import json
        from game import Game, make_deal
        g = Game(make_deal(1, 0), 0)
        g.apply(Action.RAISE_POT)
        g.apply(Action.RAISE_POT)
        g.apply(Action.CHECK_CALL)
        return json, json.dumps({"observation": g.observation(g.to_act).to_dict(), "hand_results": []})
    except Exception:
        return None, None


_BENCH = _make_bench()


def _measure_slow(reps: int = 5, use_min: bool = False) -> float:
    """Wall-clock time of the judge-side parsing work (json + Observation),
    relative to our dev machine (~0.05 ms). Wall clock on purpose: when the CPU
    is shared or throttled, the server plumbing slows down the same way, and the
    judge charges wall time."""
    js, body = _BENCH
    if js is None:
        return 2.0
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        Observation.from_dict(js.loads(body)["observation"])
        ts.append(time.perf_counter() - t)
    ts.sort()
    t = ts[0] if use_min else ts[len(ts) // 2]
    return min(20.0, max(1.0, t / 5.0e-5))


def _calibrate() -> float:
    try:
        _measure_slow(20)
        return max(_measure_slow(40, use_min=True), _measure_slow(40))
    except Exception:
        return 2.0


_SLOW = _calibrate()


def _oh(slow: float) -> float:
    """Charged server overhead per request (s): ~1.15 ms of plumbing on the dev
    machine, scaled by the slowdown, minus the free 1 ms."""
    return max(0.0003, 0.0012 * slow - 0.001)


def _measure_cost0():
    """Fit MC time = F0 + C0 * n for a real flop decision at startup."""
    try:
        from game import Game, make_deal
        g = Game(make_deal(2, 0), 0)
        g.apply(Action.RAISE_POT)
        g.apply(Action.CHECK_CALL)
        obs = g.observation(g.to_act)
        mem = {}
        new_half(AgentConfig(), mem)
        # warm every street's code path once (page in tables, numpy caches)
        g2 = Game(make_deal(3, 0), 0)
        for a in (Action.RAISE_POT, Action.CHECK_CALL, Action.RAISE_HALF, Action.CHECK_CALL,
                  Action.CHECK_CALL, Action.RAISE_HALF, Action.CHECK_CALL, Action.RAISE_HALF):
            o2 = g2.observation(g2.to_act)
            mem["hand"] = _new_hand_state(o2)
            _parse_history(o2, mem)
            _equity(o2, mem, 200, want_continue=0.4)
            g2.apply(a)
        mem["hand"] = _new_hand_state(obs)
        _parse_history(obs, mem)
        best = {}
        for n in (100, 400, 100, 400, 100, 400):
            t = time.perf_counter()
            _equity(obs, mem, n, want_continue=0.4)
            best[n] = min(best.get(n, 1.0), time.perf_counter() - t)
        c0 = max(2e-7, (best[400] - best[100]) / 300.0)
        f0 = max(2e-5, best[100] - 100 * c0)
        return f0, c0
    except Exception:
        return 2.0e-4, 1.2e-6


_F0, _C0 = _measure_cost0()


def _budget_samples(obs, memory, big):
    """Pick the MC sample count so the half stays well inside the time bank,
    counting our measured compute plus the estimated per-request overhead."""
    n = P["samples_big"] if big else P["samples"]
    bank = memory["bank"]
    hands_left = max(1, memory["num_hands"] - obs.hand_index)
    dph = (memory["n_dec"] + 5.0) / (obs.hand_index + 2.0)      # decisions per hand
    future = hands_left * dph
    mc_frac = (memory["n_mc"] + 2.0) / (memory["n_dec"] + 3.0)  # share of decisions that run MC
    # load = measured MC time / startup benchmark (hand complexity + CPU
    # contention): used to predict the next MC's cost. The server overhead is
    # estimated from the (re-measured) parse benchmark. Measured compute is
    # also hard-capped, which bounds our share of the bank under any contention.
    load = memory["load"]
    oh_now = _oh(memory["slow"])
    oh_proj = memory["oh_spent"] + future * oh_now              # server overhead, whole half
    base = memory["t_used"] + future * memory["fix"]           # compute that is not MC
    soft = min(0.25 * bank, 0.62 * bank - oh_proj) - base
    afford = int((soft / (future * mc_frac) / load - _F0) / _C0)
    if afford < n:
        n = afford
    if n < 60:
        # graceful tier: a small MC is still far better than the no-MC fallback
        hard = min(0.33 * bank, 0.78 * bank - oh_proj) - base
        n = min(120, int((hard / (future * mc_frac) / load - _F0) / _C0)) if hard > 0 else 0
    return n


def _pick(obs, *prefs):
    for a in prefs:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL


def _cheap_decision(obs, memory):
    """No-MC play for when the time bank is tight: made-hand category rules."""
    st = int(obs.street)
    pot, C = obs.pot, obs.to_call
    can_raise = Action.RAISE_POT in obs.legal_actions
    hero = np.array(obs.hole_cards, dtype=np.int64)
    if st == 0:
        pct = pre_percentile(float(pre_strength(hero)[0]))
        if pct >= 0.9 and can_raise:
            return Action.RAISE_POT
        if C == 0 or pct >= 0.55 or (C <= 0.35 * pot and pct >= 0.3):
            return Action.CHECK_CALL
        return Action.FOLD
    cat = int(partial_scores(hero, np.array(obs.board, dtype=np.int64))[0]) >> 20
    if cat >= 5:                                   # full house or better
        return Action.RAISE_POT if can_raise else Action.CHECK_CALL
    if cat == 4 or (cat == 3 and st < 3):          # trips / straight
        if C == 0:
            return Action.RAISE_HALF if can_raise else Action.CHECK_CALL
        return Action.CHECK_CALL if C <= 0.7 * pot else Action.FOLD
    if C == 0:
        return Action.CHECK_CALL
    return Action.CHECK_CALL if (C <= 0.25 * pot and cat >= 2) else Action.FOLD


def _decide(obs, memory):
    st = int(obs.street)
    pot, C = obs.pot, obs.to_call
    legal = obs.legal_actions
    can_raise = Action.RAISE_POT in legal
    ip = obs.is_button if st > 0 else (not obs.is_button)

    # ---- preflop first-in decisions: table only (cheap) --------------------
    if st == 0:
        pct = pre_percentile(float(pre_strength(np.array(obs.hole_cards, dtype=np.int64))[0]))
        nh = len(obs.action_history)
        if nh == 0 and obs.is_button:          # SB open
            f_bb = _freq(memory, "pre_vs_raise")
            raise_th = P["sb_raise"] + 0.25 * (f_bb[2] - 0.2) - 0.25 * (f_bb[0] - 0.3)
            if pct >= raise_th and can_raise:
                return Action.RAISE_POT
            if pct >= P["sb_call"]:
                return Action.CHECK_CALL
            return Action.FOLD
        if C == 0 and not obs.is_button and obs.raises_this_street == 0:   # BB vs limp
            f_l = _freq(memory, "pre_vs_raise")
            iso = P["bb_iso"] - 0.3 * (f_l[0] - 0.3)
            if pct >= iso and can_raise:
                return Action.RAISE_POT
            return Action.CHECK_CALL

    big = (pot + C) >= 40
    n = _budget_samples(obs, memory, big)
    if n < 60:
        return _cheap_decision(obs, memory)

    spot = "pre_vs_raise" if st == 0 else "post_vs_bet"
    fq = _freq(memory, spot)
    ff, fr = fq[0], fq[2]
    t0 = time.perf_counter()
    E, Ec, ff = _equity(obs, memory, n, want_continue=ff)
    dt = time.perf_counter() - t0
    memory["mc_t"] = dt
    memory["n_mc"] += 1
    memory["cost"] = 0.8 * memory["cost"] + 0.2 * (dt / n)
    ratio = min(3.0, max(0.7, dt / (_F0 + _C0 * n)))
    memory["load"] = min(3.0, max(1.0, 0.95 * memory["load"] + 0.05 * ratio))

    my_stack, opp_stack = obs.my_stack, obs.opp_stack
    real = (P["real_ip"] if ip else P["real_oop"])[st]
    # little stack behind after calling -> we realize our equity fully
    behind = min(my_stack - min(C, my_stack), opp_stack)
    if behind <= 0.35 * (pot + C):
        real = 1.0
    Er = min(1.0, E * real) if E < 0.5 else E

    ev = {}
    if C > 0:
        ev[Action.FOLD] = 0.0
        callamt = min(C, my_stack)
        ev[Action.CHECK_CALL] = Er * (pot + callamt) - callamt
        if st == 0:
            ev[Action.CHECK_CALL] -= P["pre_margin"] * (pot + callamt)
    else:
        ev[Action.CHECK_CALL] = Er * pot
    if can_raise:
        for a in (Action.RAISE_HALF, Action.RAISE_POT):
            A = obs.raise_to[a] - obs.my_bet           # chips we add
            opp_add = min(A - C, opp_stack)
            Ecr = Ec
            evr = ff * pot + (1 - ff) * (Ecr * (pot + A + opp_add) - A)
            evr -= P["bet_margin"] * (pot + A)
            if st == 0 and E < P["pre_raise_eq"]:
                evr -= 0.1 * (pot + A)
            ev[a] = evr
    best = max(ev, key=lambda a: ev[a])
    if best not in legal:
        best = Action.CHECK_CALL
    return best


def choose_action(obs: Observation, memory: dict) -> Action:
    """Range-aware equity bot. Always returns a legal action."""
    t0 = time.perf_counter()
    try:
        if "stats" not in memory:          # new_half not run (should not happen)
            new_half(AgentConfig(), memory)
        memory["mc_t"] = 0.0
        hs = memory.get("hand")
        if hs is None or hs["idx"] != obs.hand_index:
            memory["hand"] = _new_hand_state(obs)
        _parse_history(obs, memory)
        action = _decide(obs, memory)
        if action not in obs.legal_actions:
            action = Action.CHECK_CALL
        hs = memory["hand"]
        hs["last_my"] = (int(obs.street), int(action), obs.to_call)
        if action in (Action.RAISE_HALF, Action.RAISE_POT):
            hs["awaiting"] = "pre_vs_raise" if obs.street == Street.PREFLOP else "post_vs_bet"
        elif action == Action.CHECK_CALL and obs.to_call == 0 and obs.street == Street.RIVER and not obs.is_button:
            hs["awaiting"] = "river_check"
        else:
            hs["awaiting"] = None
    except Exception:
        action = Action.CHECK_CALL if (obs.to_call == 0 or Action.FOLD not in obs.legal_actions) else Action.FOLD
    try:
        n_dec = memory.get("n_dec", 0)
        if n_dec % 20 == 0:
            memory["slow"] = 0.6 * memory.get("slow", _SLOW) + 0.4 * _measure_slow(5)
        if memory.get("mc_t", 0.0) == 0.0:      # no MC this decision: let the load estimate relax
            memory["load"] = 1.0 + 0.995 * (memory.get("load", 1.0) - 1.0)
        memory["oh_spent"] = memory.get("oh_spent", 0.0) + _oh(memory.get("slow", _SLOW))
        dt = time.perf_counter() - t0
        memory["t_used"] = memory.get("t_used", 0.0) + dt
        memory["n_dec"] = n_dec + 1
        memory["fix"] = 0.9 * memory.get("fix", 2.5e-4) + 0.1 * max(0.0, dt - memory.get("mc_t", 0.0))
    except Exception:
        pass
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Infer the opponent's last (unseen) action and learn showdown info."""
    try:
        memory["hands_played"] = memory.get("hands_played", 0) + 1
        hs = memory.get("hand")
        if hs is None or hs["idx"] != result.hand_index:
            # we never acted: the opponent folded its small blind
            if result.my_net > 0 and not result.went_to_showdown and len(result.board) == 0:
                memory["stats"]["pre_open"][0] += 1
            return
        aw = hs.get("awaiting")
        if aw in ("pre_vs_raise", "post_vs_bet"):
            if not result.went_to_showdown and result.winner == "me":
                memory["stats"][aw][0] += 1
            elif result.went_to_showdown:
                memory["stats"][aw][1] += 1
        elif aw == "river_check" and result.went_to_showdown:
            memory["stats"]["post_open"][1] += 1
        if result.went_to_showdown and result.opp_hole is not None and len(result.board) == 5:
            if 3 in hs["opp_aggr_streets"]:
                memory["sd"]["aggr"] += 1
                sc = omaha_scores(np.array(result.opp_hole, dtype=np.int64),
                                  np.array(result.board, dtype=np.int64))[0]
                if (sc >> 20) <= 1:          # river aggression with one pair or worse
                    memory["sd"]["bluff"] += 1
        memory["hand"] = None
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
