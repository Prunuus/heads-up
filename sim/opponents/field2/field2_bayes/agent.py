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
PARTICIPANT = "Posterior"
AGENT_NAME = "BayesReader"


# ===== YOUR CODE HERE =====
#
# "Posterior": Bayesian hand reading for heads-up Short Deck PLO5.
#
# Every hand we keep a weighted particle belief over the opponent's 5 cards.
# Each opponent action reweights the particles by its likelihood under a
# per-opponent action model: "the opponent takes action a with the slice of
# its (our modelled) range whose strength quantile falls in the band given by
# its observed action frequencies in this context" (street x facing x size),
# mixed with a strength-independent component lambda that is learned from
# showdowns (lambda ~ 0 = honest bettor, lambda ~ 1 = bets regardless of hand).
# We then pick the action with the best expected value against the belief
# (fold model for our bets comes from the same opponent model), softened by a
# softmax so we are hard to read.

import sys
import time as _time
from itertools import combinations as _combinations
from math import comb as _comb

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEBUG = bool(os.environ.get("POSTERIOR_DEBUG"))
_BIN = np.array([[_comb(n, k) for k in range(6)] for n in range(37)], dtype=np.int64)
_PAIRS10 = np.array(list(_combinations(range(5), 2)), dtype=np.int64)
_TRIPS10 = np.array(list(_combinations(range(5), 3)), dtype=np.int64)
_PID = np.zeros((36, 36), dtype=np.int64)
for _a in range(36):
    for _b in range(_a + 1, 36):
        _PID[_a, _b] = _PID[_b, _a] = _a + _b * (_b - 1) // 2
_ONE = np.int64(1)


def _colex(a):
    idx = np.zeros(a.shape[:-1], dtype=np.int64)
    for j in range(a.shape[-1]):
        idx += _BIN[a[..., j], j + 1]
    return idx


def _build_tables():
    """Fallback (only if the shipped .npy files are missing)."""
    from evaluator import _build_table
    table = _build_table()
    scores = np.fromiter(table.values(), dtype=np.int64, count=len(table))
    uniq = np.unique(scores)
    rank5 = (np.searchsorted(uniq, scores) + 1).astype(np.uint16)
    combos5 = np.array(list(_combinations(range(36), 5)), dtype=np.int64)
    r5 = np.zeros(len(combos5), dtype=np.uint16)
    r5[_colex(combos5)] = rank5
    tri = np.array(list(_combinations(range(36), 3)), dtype=np.int64)
    tri = tri[np.argsort(_colex(tri))]
    par = np.array(list(_combinations(range(36), 2)), dtype=np.int64)
    par = par[np.argsort(_colex(par))]
    tp = np.zeros((len(tri), len(par)), dtype=np.uint16)
    for s in range(0, len(tri), 1000):
        tt = tri[s:s + 1000]
        five = np.concatenate([np.repeat(tt[:, None, :], len(par), 1),
                               np.repeat(par[None, :, :], len(tt), 0)], axis=2)
        five.sort(axis=2)
        dup = (np.diff(five, axis=2) == 0).any(axis=2)
        r = r5[np.minimum(_colex(five), len(r5) - 1)]
        r[dup] = 0
        tp[s:s + 1000] = r
    # crude preflop strength: high cards + pairs + suitedness
    rk = combos5 >> 2
    st = combos5 & 3
    hs = rk.sum(1) / 40.0
    hs += 0.08 * np.array([len(set(r)) < 5 for r in rk[:, :]], dtype=float)
    hs += 0.05 * np.array([len(set(s)) < 4 for s in st[:, :]], dtype=float)
    hs = (np.argsort(np.argsort(hs)) / len(hs)) * 0.4 + 0.3
    pf = np.zeros(len(combos5), dtype=np.uint16)
    pf[_colex(combos5)] = np.round(hs * 65535).astype(np.uint16)
    return tp, pf


try:
    _TP = np.load(os.path.join(_HERE, "tp_rank.npy"))
    _PF = np.load(os.path.join(_HERE, "pf_hs.npy")).astype(np.float64) / 65535.0
    assert _TP.shape == (7140, 630) and _PF.shape == (376992,)
except Exception:  # pragma: no cover - safety net
    _TP, _pf = _build_tables()
    _PF = _pf.astype(np.float64) / 65535.0


def _hand_pairs(h):
    return _PID[h[:, _PAIRS10[:, 0]], h[:, _PAIRS10[:, 1]]]


def _mask_of(cards):
    m = 0
    for c in cards:
        m |= 1 << int(c)
    return m


# ---------------------------------------------------------------- equity core

def street_eval(my, board, parts, pmask, K, rng):
    """My equity e_i vs every particle and each particle's strength s_i
    (equity vs a random hand, approximated by its percentile among the
    particles, which are a uniform sample) over K sampled runouts."""
    board = [int(c) for c in board]
    known = _mask_of(my) | _mask_of(board)
    deck = np.array([c for c in range(36) if not (known >> c) & 1], dtype=np.int64)
    need = 5 - len(board)
    if need == 0:
        run = np.zeros((1, 0), dtype=np.int64)
    elif need == 1:
        run = deck[:, None] if K >= len(deck) else rng.choice(deck, size=K, replace=False)[:, None]
    elif need == 2:
        # stratified: random perfect matchings, so every unseen card is a turn/river once per matching
        m = max(1, K // (len(deck) // 2))
        half = len(deck) // 2
        run = np.concatenate([rng.permutation(deck)[:2 * half].reshape(half, 2) for _ in range(m)], 0)
    else:
        run = deck[np.argsort(rng.random((K, len(deck))), axis=1)[:, :need]]
    K = run.shape[0]
    boards = np.sort(np.concatenate([np.broadcast_to(np.array(board, dtype=np.int64), (K, len(board))), run], 1), 1)
    rmask = (_ONE << run).sum(1) if need else np.zeros(1, dtype=np.int64)
    pt = _TP[_colex(boards[:, _TRIPS10])].max(1).astype(np.int32)  # K x 630
    my_hp = _hand_pairs(np.array([sorted(my)], dtype=np.int64))[0]
    my_sc = pt[:, my_hp].max(1)  # K
    hp = _hand_pairs(parts)  # N x 10
    sc = pt[:, hp].max(2)  # K x N
    N = sc.shape[1]
    valid = (pmask[None, :] & rmask[:, None]) == 0
    cnt = valid.sum(0)
    win = (my_sc[:, None] > sc) + 0.5 * (my_sc[:, None] == sc)
    e = np.where(cnt > 0, (win * valid).sum(0) / np.maximum(cnt, 1), 0.5)
    srt = np.sort(sc, 1)
    off = (np.arange(K, dtype=np.int32) * 4096)[:, None]
    flat = (srt + off).ravel()
    qq = (sc + off).ravel()
    lo = np.searchsorted(flat, qq, 'left')
    hi = np.searchsorted(flat, qq, 'right')
    base = np.repeat(np.arange(K) * N, N)
    pct = (((lo + hi) * 0.5 - base) / N).reshape(K, N)
    s = np.where(cnt > 0, (pct * valid).sum(0) / np.maximum(cnt, 1), 0.5)
    aux = (pt, flat, off, N, rmask)
    return e, s, aux


def strength_of(hole, aux):
    """Strength (vs random) of a known hand using a street's stored runouts."""
    pt, flat, off, N, rmask = aux
    h = np.array([sorted(hole)], dtype=np.int64)
    sc = pt[:, _hand_pairs(h)[0]].max(1)  # K
    hm = _mask_of(hole)
    ok = (rmask & hm) == 0
    if not ok.any():
        return 0.5
    q = sc + off[:, 0]
    K = len(sc)
    lo = np.searchsorted(flat, q, 'left')
    hi = np.searchsorted(flat, q, 'right')
    pct = ((lo + hi) * 0.5 - np.arange(K) * N) / N
    return float(pct[ok].mean())


def preflop_strength(h5):
    return _PF[_colex(np.sort(np.asarray(h5, dtype=np.int64), axis=-1))]


# ---------------------------------------------------------- opponent model

A_FOLD, A_CALL, A_HALF, A_POT = 0, 1, 2, 3
_NKIND = 6
# prior action frequencies per context kind (fold, call/check, half, pot)
_PRIOR = np.array([
    [0.00, 0.55, 0.25, 0.20],   # 0 not facing a bet
    [0.15, 0.45, 0.15, 0.25],   # 1 preflop SB facing the blind
    [0.30, 0.52, 0.09, 0.09],   # 2 facing first bet, small
    [0.40, 0.46, 0.07, 0.07],   # 3 facing first bet, big
    [0.35, 0.50, 0.07, 0.08],   # 4 facing a raise, small
    [0.42, 0.46, 0.06, 0.06],   # 5 facing a raise, big
])
_PC = 6.0
_SIG = 0.07
_GFAC = float('1.5')
_LAM_GRID = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9])
_LAM_PRIOR = np.log(_LAM_GRID) * 1.0 + np.log(1 - _LAM_GRID) * 3.0
# polarisation: share of an aggressive action's mass that comes from the
# bottom of the range (bluffs), as balanced bots do
_PI_GRID = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
_PI_PRIOR = np.log(_PI_GRID + 0.05) * 1.0 + np.log(1 - _PI_GRID) * 4.0


def _kind(street, to_call, raises_before, pot):
    if to_call <= 0:
        return 0
    if street == 0 and raises_before == 0:
        return 1
    ratio = to_call / max(1, pot - to_call)
    big = 1 if ratio >= 0.75 else 0
    return (2 if raises_before <= 1 else 4) + big


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _thr_prob(q, a, f, avail):
    """P(a | strength quantile q) under the ordered-threshold model."""
    order = [x for x in range(4) if avail[x]]
    lo = sum(f[x] for x in order if x < a)
    hi = lo + f[a]
    p_lo = 1.0 if a == order[0] else _sig((q - lo) / _SIG)
    p_hi = 0.0 if a == order[-1] else _sig((q - hi) / _SIG)
    return np.clip(p_lo - p_hi, 0.0, 1.0)


def _freqs(memory, street, kind, avail):
    d = memory["cnt"][street, kind]
    f = (_PRIOR[kind] * _PC + d) / (_PC + d.sum())
    f = f * np.array(avail, dtype=float)
    tot = f.sum()
    if tot <= 0:
        f = np.array(avail, dtype=float)
        tot = f.sum()
    return f / tot


def _pol_prob(q, a, f, avail, pi):
    """Threshold model with a polarised bluff band for aggressive actions."""
    pt = _thr_prob(q, a, f, avail)
    if a >= 2 and pi > 0:
        pt = (1 - pi) * pt + pi * _sig((f[a] - q) / _SIG)
    return pt


def _act_prob(q, a, f, avail, lam, pi):
    return (1 - lam) * _pol_prob(q, a, f, avail, pi) + lam * f[a]


def _pi(memory):
    ll = memory["pi_ll"] + _PI_PRIOR
    p = np.exp(ll - ll.max())
    return float((p * _PI_GRID).sum() / p.sum())


def _lam(memory):
    ll = memory["lam_ll"] + _LAM_PRIOR
    p = np.exp(ll - ll.max())
    return float((p * _LAM_GRID).sum() / p.sum())


def _wquant(s, w):
    order = np.argsort(s, kind="stable")
    cw = np.cumsum(w[order])
    q = np.empty_like(s)
    q[order] = cw - 0.5 * w[order]
    return q, s[order], cw - 0.5 * w[order]


def _replay(history, is_button):
    """Yield (index, entry, street, to_call, raises_before, pot_before, avail)
    for every action in this hand's history, from the actor's view."""
    bets = {True: 1 if is_button else 2, False: 2 if is_button else 1}
    contrib = dict(bets)
    street = 0
    raises = 0
    out = []
    for i, h in enumerate(history):
        st = int(h.street)
        if st != street:
            street = st
            bets = {True: 0, False: 0}
            raises = 0
        me = h.is_me
        to_call = max(0, bets[not me] - bets[me])
        pot = contrib[True] + contrib[False]
        stack_actor = 80 - contrib[me]
        stack_other = 80 - contrib[not me]
        can_raise = raises < 4 and stack_actor > to_call and stack_other > 0
        avail = (to_call > 0, True, can_raise, can_raise)
        out.append((i, h, st, to_call, raises, pot, avail))
        bets[me] = h.street_total
        contrib[me] += h.amount
        if int(h.action) >= 2:
            raises += 1
    return out


# ---------------------------------------------------------- per-hand belief

def _new_hand(obs, memory):
    rng = memory["rng"]
    n0 = memory["n0"]
    my = np.array(obs.hole_cards, dtype=np.int64)
    deck = np.array([c for c in range(36) if c not in obs.hole_cards], dtype=np.int64)
    parts = np.sort(deck[np.argsort(rng.random((n0, len(deck))), axis=1)[:, :5]], 1)
    hand = {
        "idx": obs.hand_index, "my": my, "parts": parts,
        "pmask": (_ONE << parts).sum(1), "logw": np.zeros(n0),
        "sd": {}, "done": 0, "recs": [], "last": None, "button": obs.is_button,
    }
    memory["hand"] = hand
    return hand


def _street_data(hand, street, board, memory):
    sd = hand["sd"].get(street)
    if sd is not None:
        return sd
    rng = memory["rng"]
    bmask = _mask_of(board[:[0, 3, 4, 5][street]])
    alive = np.nonzero((hand["pmask"] & bmask) == 0)[0]
    parts = hand["parts"][alive]
    kk = memory["K"][street]
    if street == 0:
        # preflop: noise-free pairwise model calibrated offline on the strength table
        s = preflop_strength(parts)
        e = np.clip(0.5 + 0.95 * (float(preflop_strength(hand["my"])) - s), 0.03, 0.97)
        aux = None
    else:
        e, s, aux = street_eval(hand["my"], board[:[0, 3, 4, 5][street]], parts, hand["pmask"][alive], kk, rng)
    sd = {"alive": alive, "e": e, "s": s, "aux": aux}
    hand["sd"][street] = sd
    return sd


def _weights(hand, sd):
    lw = hand["logw"][sd["alive"]]
    w = np.exp(lw - lw.max())
    return w / w.sum()


def _observe_opp(hand, memory, street, a, kind, avail, board, count=True):
    if count:
        memory["cnt"][street, kind, a] += 1.0
    sd = _street_data(hand, street, board, memory)
    w = _weights(hand, sd)
    q, s_sorted, cw = _wquant(sd["s"], w)
    f = _freqs(memory, street, kind, avail)
    lam = _lam(memory)
    pi = _pi(memory)
    hand["logw"][sd["alive"]] += np.log(_act_prob(q, a, f, avail, lam, pi) + 1e-9)
    hand["recs"].append((street, a, f, avail, s_sorted, cw))


def _process_history(obs, hand, memory):
    board = list(obs.board)
    for (i, h, st, to_call, raises, pot, avail) in _replay(obs.action_history, obs.is_button):
        if i < hand["done"]:
            continue
        if not h.is_me:
            a = int(h.action)
            _observe_opp(hand, memory, st, a, _kind(st, to_call, raises, pot), avail, board)
    hand["done"] = len(obs.action_history)


# ---------------------------------------------------------- decision

def _decide(obs, memory, hand):
    street = int(obs.street)
    sd = _street_data(hand, street, list(obs.board), memory)
    w = _weights(hand, sd)
    e = sd["e"]
    q, _, _ = _wquant(sd["s"], w)
    lam = _lam(memory)
    pi = _pi(memory)
    pot, to_call = obs.pot, obs.to_call
    eq = float((w * e).sum())
    legal = obs.legal_actions
    evs = {}
    if Action.FOLD in legal:
        evs[Action.FOLD] = 0.0
    # realisation: out of position / early streets realise a bit less
    real = 1.0 if street == 3 else (0.97 if obs.is_button else 0.93)
    # stack commitment: vs an aggressive opponent the remaining stacks tend to
    # go in later, which a one-street EV ignores
    g = 0.0
    if street < 3:
        c = memory["cnt"][1:, 0].sum(0)
        agg = (c[2] + c[3] + 4.0 * 0.45) / (c.sum() + 4.0)
        g = float(np.clip((agg - 0.35) / 0.55, 0.0, 1.0)) * _GFAC

    def fut(e_vec, s_rem):
        if g <= 0 or s_rem <= 0:
            return 0.0
        d = 2 * e_vec - 1
        return g * s_rem * np.where(d > 0, d, 0.4 * d)

    s_rem_call = min(obs.my_stack - to_call, obs.opp_stack)
    evs[Action.CHECK_CALL] = float((w * (real * e * (pot + to_call) + fut(e, s_rem_call))).sum()) - to_call
    for act in (Action.RAISE_HALF, Action.RAISE_POT):
        if act not in legal:
            continue
        R = obs.raise_to[act]
        b = R - obs.my_bet
        opp_call = min(R - obs.opp_bet, obs.opp_stack)
        pot_after = pot + b
        kind = _kind(street, R - obs.opp_bet, obs.raises_this_street + 1, pot_after)
        can_r = (obs.raises_this_street + 1) < 4 and obs.opp_stack > opp_call and obs.my_stack - b > 0
        avail = (True, True, can_r, can_r)
        f = _freqs(memory, street, kind, avail)
        pf = _act_prob(q, A_FOLD, f, avail, lam, pi)
        pr = 0.0
        if can_r:
            pr = _act_prob(q, A_HALF, f, avail, lam, pi) + _act_prob(q, A_POT, f, avail, lam, pi)
        pot_called = pot + b + opp_call
        # if they re-raise we assume we continue only with our good part:
        # model it as a call where our equity is shaded down a little
        e_c = e - 0.14 * pr * (1 - e)
        ev_cont = real * e_c * pot_called - b + fut(e_c, min(obs.my_stack - b, obs.opp_stack - opp_call))
        evs[act] = float((w * (pf * pot + (1 - pf) * ev_cont)).sum())
    # softmax mixing (temperature relative to pot) for unpredictability
    acts = list(evs.keys())
    v = np.array([evs[a] for a in acts])
    tau = 0.025 * pot + 0.25
    p = np.exp((v - v.max()) / tau)
    p /= p.sum()
    choice = acts[int(memory["rng"].choice(len(acts), p=p))]
    return choice, evs, eq


def _cheap_action(obs):
    legal = obs.legal_actions
    if obs.to_call == 0:
        return Action.CHECK_CALL
    if obs.to_call * 4 <= obs.pot and Action.CHECK_CALL in legal:
        return Action.CHECK_CALL
    return Action.FOLD


_BENCH_REF = 0.00035  # seconds per benchmark street_eval on the dev machine


def _bench():
    """Rough CPU speed relative to the dev machine (>1 = slower)."""
    try:
        rng = np.random.default_rng(0)
        my = np.array([0, 5, 10, 15, 20], dtype=np.int64)
        deck = np.array([c for c in range(36) if c not in (0, 5, 10, 15, 20, 1, 2, 3)], dtype=np.int64)
        parts = np.sort(deck[np.argsort(rng.random((140, len(deck))), 1)[:, :5]], 1)
        pm = (_ONE << parts).sum(1)
        best = 1e9
        for _ in range(4):
            t = _time.perf_counter()
            street_eval(my, [1, 2, 3], parts, pm, 14, rng)
            best = min(best, _time.perf_counter() - t)
        return float(np.clip(best / _BENCH_REF, 1.0, 3.0))
    except Exception:
        return 2.0


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["hands_played"] = 0
    memory["rng"] = np.random.default_rng(int.from_bytes(os.urandom(4), "little"))
    memory["cnt"] = np.zeros((4, _NKIND, 4))
    memory["lam_ll"] = np.zeros(len(_LAM_GRID))
    memory["pi_ll"] = np.zeros(len(_PI_GRID))
    memory["hand"] = None
    memory["tused"] = 0.0
    memory["n0"] = 240
    memory["K"] = [0, 14, 14, 1]
    memory["mode"] = 0
    memory["calls"] = 0
    memory["troute"] = 0.0
    memory["ovh"] = _OVH * _bench()


_OVH = 0.00045      # assumed HTTP-stack charge per request outside the timed route (beyond the free 1 ms)
_BUDGET = 7.0      # seconds of the 8 s bank we allow ourselves per half


def _set_mode(memory, hand_index):
    est = max(memory["tused"], memory["troute"]) + memory["ovh"] * memory["calls"]
    proj = est / (hand_index + 1) * 1000.0
    if est > _BUDGET - 0.6 or memory["mode"] == 2:
        memory["mode"] = 2
    elif proj > _BUDGET - 1.2:
        memory["mode"] = 1
        memory["n0"], memory["K"] = 140, [0, 14, 8, 1]
    else:
        memory["mode"] = 0
        memory["n0"], memory["K"] = 240, [0, 14, 14, 1]


def choose_action(obs: Observation, memory: dict) -> Action:
    t0 = _time.perf_counter()
    memory["calls"] = memory.get("calls", 0) + 1
    try:
        if memory.get("mode", 0) == 2:
            return _cheap_action(obs)
        hand = memory.get("hand")
        if hand is None or hand["idx"] != obs.hand_index:
            _set_mode(memory, obs.hand_index)
            if memory["mode"] == 2:
                return _cheap_action(obs)
            hand = _new_hand(obs, memory)
        _process_history(obs, hand, memory)
        act, evs, eq = _decide(obs, memory, hand)
        if act not in obs.legal_actions:
            act = _cheap_action(obs)
        # remember what the opponent will face, to infer the response at hand end
        if act in (Action.RAISE_HALF, Action.RAISE_POT):
            R = obs.raise_to[act]
            b = R - obs.my_bet
            opp_call = min(R - obs.opp_bet, obs.opp_stack)
            kind = _kind(int(obs.street), R - obs.opp_bet, obs.raises_this_street + 1, obs.pot + b)
            can_r = (obs.raises_this_street + 1) < 4 and obs.opp_stack > opp_call and obs.my_stack - b > 0
            hand["last"] = ("bet", int(obs.street), kind, (True, True, can_r, can_r), list(obs.board))
        elif act == Action.CHECK_CALL and obs.to_call == 0 and int(obs.street) == 3:
            first = not any(h.street == Street.RIVER for h in obs.action_history)
            hand["last"] = ("check", 3, 0, (False, True, True, True), list(obs.board)) if first else None
        else:
            hand["last"] = None
        hand["done"] = len(obs.action_history) + 1
        return act
    except Exception:
        if _DEBUG:
            raise
        return _cheap_action(obs)
    finally:
        memory["tused"] = memory.get("tused", 0.0) + (_time.perf_counter() - t0)


def on_hand_end(result: HandResult, memory: dict) -> None:
    t0 = _time.perf_counter()
    try:
        memory["hands_played"] = memory.get("hands_played", 0) + 1
        if _DEBUG and result.hand_index % 250 == 249:
            print(f"[posterior] hand {result.hand_index} tused={memory['tused']:.2f} troute={memory['troute']:.2f} "
                  f"calls={memory['calls']} est={max(memory['tused'], memory['troute']) + memory['ovh'] * memory['calls']:.2f} mode={memory['mode']}",
                  file=sys.stderr, flush=True)
        hand = memory.get("hand")
        if hand is None or hand["idx"] != result.hand_index:
            return
        last = hand.get("last")
        if last is not None:
            kind_l, st, kind, avail, board = last
            if kind_l == "bet":
                a = A_CALL if result.went_to_showdown else (A_FOLD if result.my_net > 0 else None)
            else:
                a = A_CALL if result.went_to_showdown else None
            if a is not None and st in hand["sd"]:
                _observe_opp(hand, memory, st, a, kind, avail, board)
        # learn honesty (lambda) from the shown hand
        if result.opp_hole is not None and hand["recs"]:
            opp = list(result.opp_hole)
            s_true = {}
            for street, sd in hand["sd"].items():
                if street == 0:
                    s_true[0] = float(preflop_strength(np.array(opp)))
                elif sd["aux"] is not None:
                    s_true[street] = strength_of(opp, sd["aux"])
            ll = memory["lam_ll"]
            pl = memory["pi_ll"]
            lam_c, pi_c = _lam(memory), _pi(memory)
            for (street, a, f, avail, s_sorted, cw) in hand["recs"]:
                if street not in s_true:
                    continue
                qt = np.array([float(np.interp(s_true[street], s_sorted, cw)) if len(s_sorted) else 0.5])
                pt = float(_pol_prob(qt, a, f, avail, pi_c)[0])
                ll += np.log((1 - _LAM_GRID) * pt + _LAM_GRID * f[a] + 1e-6)
                if a >= 2:
                    thr = float(_thr_prob(qt, a, f, avail)[0])
                    bot = float(_sig((f[a] - qt[0]) / _SIG))
                    pl += np.log((1 - lam_c) * ((1 - _PI_GRID) * thr + _PI_GRID * bot) + lam_c * f[a] + 1e-6)
        # mild forgetting so we can follow an opponent that changes gears
        memory["cnt"] *= 0.997
        memory["lam_ll"] *= 0.995
        memory["pi_ll"] *= 0.995
    except Exception:
        if _DEBUG:
            raise
    finally:
        memory["tused"] = memory.get("tused", 0.0) + (_time.perf_counter() - t0)

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
    _t_route = _time.perf_counter()
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    MEMORY["troute"] = MEMORY.get("troute", 0.0) + (_time.perf_counter() - _t_route)  # time governor
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
