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
PARTICIPANT = "Metagame"
AGENT_NAME = "MetagameEnsemble"


# ===== YOUR CODE HERE =====
#
# Metagame: an ensemble of four parametric styles on top of one shared,
# range-aware Monte Carlo equity engine.  A Thompson-sampling bandit picks
# the style per two-hand block from luck-adjusted chip results, and fast
# detectors (over-folder, calling station, maniac, out-of-bank check/folder)
# overlay exploits on whichever style is active.

import time
from itertools import combinations
from math import comb

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---- lookup tables ----------------------------------------------------------
_PIDX = np.zeros((36, 36), dtype=np.int64)
for _a in range(36):
    for _b in range(36):
        _lo, _hi = min(_a, _b), max(_a, _b)
        _PIDX[_a, _b] = comb(_hi, 2) + _lo
_TIDX = np.zeros((36, 36, 36), dtype=np.int64)
for _a in range(36):
    for _b in range(36):
        for _c in range(36):
            _s = sorted((_a, _b, _c))
            _TIDX[_a, _b, _c] = comb(_s[2], 3) + comb(_s[1], 2) + _s[0]
_P2 = np.array(list(combinations(range(5), 2)))
_P3 = np.array(list(combinations(range(5), 3)))
_PA, _PB = _P2[:, 0], _P2[:, 1]
_TA, _TB, _TC = _P3[:, 0], _P3[:, 1], _P3[:, 2]
_CB = np.array([[comb(c, i + 1) for c in range(36)] for i in range(5)], dtype=np.int64)


def _build_ptab():
    """pair x triple -> hand-rank table (fallback if ptab.npy is missing)."""
    from evaluator import get_table
    tab = get_table()
    combos = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    masks = (np.int64(1) << combos).sum(1)
    scores = np.fromiter((tab[int(m)] for m in masks), dtype=np.int64, count=len(masks))
    _, inv = np.unique(scores, return_inverse=True)
    ranks = (inv + 1).astype(np.uint16)
    T = np.zeros(630 * 7140, dtype=np.uint16)
    for i, j in combinations(range(5), 2):
        r = [k for k in range(5) if k not in (i, j)]
        T[_PIDX[combos[:, i], combos[:, j]] * 7140 + _TIDX[combos[:, r[0]], combos[:, r[1]], combos[:, r[2]]]] = ranks
    return T


try:
    _T = np.load(os.path.join(_HERE, "ptab.npy"))
    assert _T.shape == (630 * 7140,)
except Exception:
    _T = _build_ptab()
try:
    _PF = np.load(os.path.join(_HERE, "pfeq.npy")).astype(np.float32)
    assert _PF.shape == (376992,)
except Exception:  # crude fallback: sum of ranks
    _cmb = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    _PF = ((_cmb >> 2).sum(1) / 40.0).astype(np.float32)
_order = np.argsort(_PF, kind="stable")
_PFP = np.empty(len(_PF), dtype=np.float32)
_PFP[_order] = np.arange(len(_PF), dtype=np.float32) / (len(_PF) - 1)
del _order


def _colex(cards_sorted):
    """colex index of sorted 5-card rows (N x 5) or a single sorted 5-tuple."""
    c = np.asarray(cards_sorted, dtype=np.int64)
    if c.ndim == 1:
        return int(sum(_CB[i, c[i]] for i in range(5)))
    return _CB[0, c[:, 0]] + _CB[1, c[:, 1]] + _CB[2, c[:, 2]] + _CB[3, c[:, 3]] + _CB[4, c[:, 4]]


def _pct_rank(x):
    """Percentile rank (0..1) of each element among x."""
    n = len(x)
    r = np.empty(n, dtype=np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(n)
    return r / max(1, n - 1)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _mc(hole, board, N, rng, ops):
    """Weighted Monte Carlo equity of `hole` vs an opponent range built by
    narrowing a random hand with `ops` = [(street, kind, freq), ...]."""
    hole = np.asarray(hole, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    used = np.zeros(36, bool)
    used[hole] = True
    used[board] = True
    deck = np.flatnonzero(~used)
    k = len(board)
    need = 10 - k
    s = deck[rng.random((N, len(deck))).argsort(1)[:, :need]]
    if k:
        boards = np.concatenate([np.broadcast_to(board, (N, k)), s[:, :5 - k]], 1)
    else:
        boards = s[:, :5]
    opp = s[:, 5 - k:]
    bt = _TIDX[boards[:, _TA], boards[:, _TB], boards[:, _TC]]
    mh = _PIDX[hole[_PA], hole[_PB]] * 7140
    my = _T[mh[None, :, None] + bt[:, None, :]].max((1, 2))
    oh = _PIDX[opp[:, _PA], opp[:, _PB]] * 7140
    osc = _T[oh[:, :, None] + bt[:, None, :]].max((1, 2))
    win = (my > osc) + 0.5 * (my == osc)
    if not ops:
        e = float(win.mean())
        return e, e
    strength = {}

    def get_strength(st):
        if st in strength:
            return strength[st]
        if 0 not in strength:
            strength[0] = _PFP[_colex(np.sort(opp, 1))].astype(np.float64)
        if st == 0:
            return strength[0]
        if st == 1:
            m = _T[oh + _TIDX[board[0], board[1], board[2]]].max(1)
            v = 0.6 * _pct_rank(m) + 0.4 * strength[0]
        elif st == 2:
            b4 = board[:4]
            t4 = np.array([_TIDX[b4[a], b4[b], b4[c]] for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))])
            m = _T[oh[:, :, None] + t4[None, None, :]].max((1, 2))
            v = 0.82 * _pct_rank(m) + 0.18 * strength[0]
        else:
            v = _pct_rank(osc + 0.01 * strength[0])
        v = _pct_rank(v)
        strength[st] = v
        return v

    w = np.ones(N)
    for st, kind, f in ops:
        p = get_strength(st)
        if kind == "agg":
            w *= 0.04 + 0.96 * _sig((p - (1.0 - f)) / 0.08)
        elif kind == "cont":
            w *= 0.06 + 0.94 * _sig((p - (1.0 - f)) / 0.10)
        elif kind == "chk":
            w *= 1.0 - 0.7 * _sig((p - (1.0 - 0.6 * f)) / 0.07)
        elif kind == "limp":  # not a raise, not a fold
            w *= (1.0 - 0.75 * _sig((p - (1.0 - f)) / 0.07))
    sw = w.sum()
    e = float((w * win).sum() / sw) if sw > 1e-9 else float(win.mean())
    return e, float(win.mean())


def _exact_equity(hole, opp, board_known, full_board):
    """Equity of hole vs opp (both known) with board_known fixed: exact on
    the flop/turn, sampled preflop."""
    hole = np.asarray(hole, dtype=np.int64)
    opp = np.asarray(opp, dtype=np.int64)
    k = len(board_known)
    used = np.zeros(36, bool)
    used[hole] = True
    used[opp] = True
    used[list(board_known)] = True
    deck = np.flatnonzero(~used)
    if k == 4:
        rest = deck[:, None]
    elif k == 3:
        rest = np.array(list(combinations(deck, 2)), dtype=np.int64)
    else:
        rng = np.random.default_rng(len(deck) * 7 + int(hole.sum()))
        rest = deck[rng.random((400, len(deck))).argsort(1)[:, :5]]
    n = len(rest)
    if k:
        boards = np.concatenate([np.broadcast_to(np.asarray(board_known, dtype=np.int64), (n, k)), rest], 1)
    else:
        boards = rest
    bt = _TIDX[boards[:, _TA], boards[:, _TB], boards[:, _TC]]
    mh = _PIDX[hole[_PA], hole[_PB]] * 7140
    oh = _PIDX[opp[_PA], opp[_PB]] * 7140
    my = _T[mh[None, :, None] + bt[:, None, :]].max((1, 2))
    osc = _T[oh[None, :, None] + bt[:, None, :]].max((1, 2))
    return float(((my > osc) + 0.5 * (my == osc)).mean())


# ---- styles -----------------------------------------------------------------
# open: button raise fraction; limp: button limp fraction (above the folds);
# iso: BB raise-vs-limp fraction; tb: 3-bet fraction; cm: call margin;
# bet/vr: value bet / value raise equity per street (flop, turn, river);
# blf: bluff-bet prob for weak hands; semi: semi-bluff prob; slow: slowplay
# prob; trust: how much opponent aggression narrows its range.
STYLES = {
    "bal": dict(open=0.84, limp=0.09, iso=0.36, tb=0.16, fb=0.06, cm=-0.01,
                bet=(0.58, 0.60, 0.62), vr=(0.70, 0.73, 0.78), blf=(0.24, 0.19, 0.19),
                semi=0.35, slow=0.06, potth=0.08, trust=1.0),
    "aggr": dict(open=0.90, limp=0.06, iso=0.42, tb=0.20, fb=0.08, cm=-0.02,
                 bet=(0.56, 0.58, 0.61), vr=(0.68, 0.71, 0.76), blf=(0.32, 0.26, 0.24),
                 semi=0.45, slow=0.03, potth=0.06, trust=0.95),
    "hyper": dict(open=0.95, limp=0.04, iso=0.55, tb=0.28, fb=0.10, cm=-0.03,
                  bet=(0.54, 0.56, 0.60), vr=(0.66, 0.69, 0.74), blf=(0.45, 0.38, 0.32),
                  semi=0.60, slow=0.02, potth=0.05, trust=0.9),
    "trap": dict(open=0.75, limp=0.18, iso=0.20, tb=0.10, fb=0.05, cm=-0.04,
                 bet=(0.62, 0.63, 0.62), vr=(0.75, 0.78, 0.80), blf=(0.06, 0.06, 0.08),
                 semi=0.15, slow=0.35, potth=0.12, trust=0.75),
}
# Preflop is shared: hyper's preflop aggression won every cross-test, so the
# arms differ only postflop (trap keeps a slightly tighter preflop).
for _k in ("bal", "aggr"):
    STYLES[_k].update({x: STYLES["hyper"][x] for x in ("open", "limp", "iso", "tb", "fb")})
STYLES["trap"].update({x: STYLES["aggr"][x] for x in ("open", "limp", "iso", "tb", "fb")})
ARMS = ("bal", "aggr", "hyper", "trap")
ARM_PRIOR = {"bal": -0.3, "aggr": 0.6, "hyper": 0.8, "trap": -1.0}
N_SAMPLES = 110
FIXED_ARM = None  # set by the local harness to disable the bandit


def _cnt(memory, key, n):
    st = memory["st"]
    if key not in st:
        st[key] = [0.0] * n
    return st[key]


PRIORS = {
    "pf_open": (6.0, 2.5, 1.5),       # opp button first action: raise, limp, fold
    "pf_vs_limp": (3.0, 7.0),         # opp BB after our limp: raise, check
    "pf_vs_raise": (1.5, 5.0, 3.5),   # opp facing our preflop raise: raise, call, fold
    "post_open": (4.0, 6.0),          # opp postflop not facing a bet: bet, check
    "post_vs_bet": (1.2, 4.3, 4.5),   # opp facing our postflop bet: raise, call, fold
}


def _freq(memory, key):
    c = _cnt(memory, key, len(PRIORS[key]))
    pr = PRIORS[key]
    tot = sum(c) + sum(pr)
    return [(c[i] + pr[i]) / tot for i in range(len(pr))]


def _raw(memory, key):
    c = _cnt(memory, key, len(PRIORS[key]))
    return c, sum(c)


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["rng"] = np.random.default_rng(int.from_bytes(os.urandom(4), "little"))
    memory["st"] = {}
    memory["hand"] = -1
    memory["seen"] = 0
    memory["arm"] = None
    memory["arm_hand"] = {}
    memory["bandit"] = {a: [0.0, 0.0, 0.0] for a in ARMS}
    memory["last"] = None          # (hand, street, action, facing) of our last decision
    memory["streak"] = [0, 0]      # opp consecutive default actions, folds among them
    memory["t_used"] = 0.0
    memory["n_dec"] = 0
    memory["pending_ops"] = []
    memory["sigma2"] = [0.0, 0.0]  # sum w, sum w*x^2 for pooled variance
    memory["hole"] = None


def _record_opp(memory, street, action, facing, raises_before, opp_is_button):
    """Update opponent counters for one observed opponent action."""
    if street == Street.PREFLOP:
        if facing and raises_before == 0 and opp_is_button:
            c = _cnt(memory, "pf_open", 3)
            c[{Action.RAISE_HALF: 0, Action.RAISE_POT: 0, Action.CHECK_CALL: 1, Action.FOLD: 2}[action]] += 1
        elif not facing:
            c = _cnt(memory, "pf_vs_limp", 2)
            c[0 if action in (Action.RAISE_HALF, Action.RAISE_POT) else 1] += 1
        else:
            c = _cnt(memory, "pf_vs_raise", 3)
            c[{Action.RAISE_HALF: 0, Action.RAISE_POT: 0, Action.CHECK_CALL: 1, Action.FOLD: 2}[action]] += 1
    else:
        if facing:
            c = _cnt(memory, "post_vs_bet", 3)
            c[{Action.RAISE_HALF: 0, Action.RAISE_POT: 0, Action.CHECK_CALL: 1, Action.FOLD: 2}[action]] += 1
        else:
            c = _cnt(memory, "post_open", 2)
            c[0 if action in (Action.RAISE_HALF, Action.RAISE_POT) else 1] += 1
    sk = memory["streak"]
    default = (action == Action.FOLD) if facing else (action == Action.CHECK_CALL)
    if default:
        sk[0] += 1
        if facing:
            sk[1] += 1
    else:
        sk[0] = 0
        sk[1] = 0


def _walk(obs):
    """Replay the hand history: returns list of (idx, street, is_me, action,
    facing, raises_before) entries."""
    out = []
    me_btn = obs.is_button
    street = None
    bm = bo = 0
    raises = 0
    for i, a in enumerate(obs.action_history):
        if a.street != street:
            street = a.street
            raises = 0
            if street == Street.PREFLOP:
                bm, bo = (1, 2) if me_btn else (2, 1)
            else:
                bm = bo = 0
        if a.is_me:
            facing = bo > bm
            bm = a.street_total
        else:
            facing = bm > bo
            bo = a.street_total
        out.append((i, a.street, a.is_me, a.action, facing, raises))
        if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            raises += 1
    return out


def _start_hand(obs, memory):
    memory["hand"] = obs.hand_index
    memory["seen"] = 0
    if obs.hand_index % 2 == 0 or memory["arm"] is None or memory["arm"] not in STYLES:
        memory["arm"] = _pick_arm(memory)
    memory["arm_hand"][obs.hand_index] = memory["arm"]
    if len(memory["arm_hand"]) > 50:
        for h in sorted(memory["arm_hand"])[:-20]:
            del memory["arm_hand"][h]


def _pick_arm(memory):
    if FIXED_ARM is not None:
        return FIXED_ARM
    rng = memory["rng"]
    s2w, s2 = memory["sigma2"]
    var = (s2 + 30 * 100.0) / (s2w + 30)
    sigma = max(3.0, var ** 0.5)
    best, bv = "bal", -1e9
    n0 = 60.0
    for a in ARMS:
        w, wx, _ = memory["bandit"][a]
        n = w + n0
        mean = (wx + ARM_PRIOR[a] * n0) / n
        v = mean + rng.standard_normal() * sigma / n ** 0.5
        if v > bv:
            best, bv = a, v
    return best


def _detectors(memory):
    """Return exploit flags from raw counts."""
    flags = {}
    c, n = _raw(memory, "post_vs_bet")
    cp, npf = _raw(memory, "pf_vs_raise")
    tot = n + npf
    if tot >= 14:
        fold = (c[2] + cp[2]) / tot
        cont = 1 - fold
        flags["fold"] = fold
        if fold >= 0.58:
            flags["folder"] = True
        if fold <= 0.22:
            flags["station"] = True
        flags["raise_vs"] = (c[0] + cp[0]) / tot
    co, no = _raw(memory, "post_open")
    po, npo = _raw(memory, "pf_open")
    if no + npo >= 20:
        agg = (co[0] + po[0]) / (no + npo)
        if agg >= 0.7 and flags.get("raise_vs", 0) >= 0.3:
            flags["maniac"] = True
    sk = memory["streak"]
    if sk[0] >= 10 and sk[1] >= 4:
        flags["oob"] = True
    return flags


def _ops_for(obs, memory, walk, trust):
    """Range-narrowing ops from the opponent's actions this hand."""
    ops = []
    opp_btn = not obs.is_button
    for i, st, is_me, act, facing, rb in walk:
        if is_me:
            continue
        agg = act in (Action.RAISE_HALF, Action.RAISE_POT)
        if st == Street.PREFLOP:
            if facing and rb == 0 and opp_btn:
                f = _freq(memory, "pf_open")
                if agg:
                    ops.append((0, "agg", min(1.0, f[0] / trust)))
                else:
                    ops.append((0, "limp", f[0]))
            elif not facing:
                if agg:
                    ops.append((0, "agg", min(1.0, _freq(memory, "pf_vs_limp")[0] / trust)))
            else:
                f = _freq(memory, "pf_vs_raise")
                if agg:
                    ops.append((0, "agg", min(1.0, f[0] * (0.7 if rb >= 2 else 1.0) / trust)))
                else:
                    ops.append((0, "cont", min(1.0, (f[0] + f[1]) / trust)))
        else:
            if facing:
                f = _freq(memory, "post_vs_bet")
                if agg:
                    ops.append((int(st), "agg", min(1.0, f[0] * (0.7 if rb >= 2 else 1.0) / trust)))
                else:
                    ops.append((int(st), "cont", min(1.0, (f[0] + f[1]) / trust)))
            else:
                f = _freq(memory, "post_open")
                if agg:
                    ops.append((int(st), "agg", min(1.0, f[0] / trust)))
                else:
                    ops.append((int(st), "chk", f[0]))
    return ops


def _legal_pick(obs, want):
    la = obs.legal_actions
    if want in la:
        return want
    if want in (Action.RAISE_HALF, Action.RAISE_POT):
        return Action.CHECK_CALL
    if want == Action.FOLD and Action.FOLD not in la:
        return Action.CHECK_CALL
    return Action.CHECK_CALL if Action.CHECK_CALL in la else la[0]


def _decide(obs, memory):
    rng = memory["rng"]
    if obs.hand_index != memory["hand"]:
        _start_hand(obs, memory)
    walk = _walk(obs)
    # update opponent stats with new opponent actions
    for i, st, is_me, act, facing, rb in walk[memory["seen"]:]:
        if not is_me:
            _record_opp(memory, st, act, facing, rb, not obs.is_button)
    memory["seen"] = len(walk)

    P = _STYLE_CACHE.get(memory["arm"]) or STYLES["bal"]
    flags = _detectors(memory)
    la = obs.legal_actions
    can_raise = Action.RAISE_POT in la
    C = obs.to_call
    pot = obs.pot
    street = int(obs.street)
    u = rng.random()

    # out-of-bank / auto check-folder: bet everything small
    if flags.get("oob"):
        if can_raise:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    blf_mult = 1.0
    semi = P["semi"]
    cm = P["cm"]
    slow = P["slow"]
    bet_adj = 0.0
    if flags.get("folder"):
        blf_mult = 1.0 + 6.0 * (flags["fold"] - 0.5)
        semi = min(0.9, semi * 2.0)
    if flags.get("station"):
        blf_mult = 0.0
        semi = 0.0
        bet_adj = -0.04
    if flags.get("maniac"):
        blf_mult *= 0.3
        slow = max(slow, 0.3)
        cm -= 0.02

    tl = memory["t_used"]
    # Pace control on our own compute (HTTP overhead is not visible here, so
    # keep our share of the 8 s bank small and degrade gracefully).
    pace = 1.0 * (obs.hand_index + 1) / 1000.0 + 0.15
    if tl < pace:
        N = N_SAMPLES
    elif tl < 1.6:
        N = 60
    elif tl < 2.2:
        N = 30
    else:
        N = 0

    if street == 0:
        pct = float(_PFP[_colex(obs.hole_cards)])
        raises = obs.raises_this_street
        if C > 0 and raises == 0 and obs.is_button:
            # first action as button
            if pct >= 1.0 - P["open"] * (1.15 if flags.get("folder") else 1.0):
                return _legal_pick(obs, Action.RAISE_POT)
            if pct >= 1.0 - P["open"] - P["limp"]:
                return Action.CHECK_CALL
            return Action.FOLD
        if C == 0:
            # BB after a limp
            iso = P["iso"] * (1.4 if flags.get("folder") else 1.0)
            if pct >= 1.0 - iso and can_raise:
                return Action.RAISE_POT
            return Action.CHECK_CALL
        # facing a raise
        ops = _ops_for(obs, memory, walk, P["trust"])
        if N > 0:
            e, _ = _mc(obs.hole_cards, (), N, rng, ops)
        else:
            e = 0.35 + 0.3 * pct
        req = C / (pot + C)
        allin = C >= obs.my_stack
        real = 1.0 if allin else (0.9 if obs.is_button else 0.86)
        tb = P["tb"] * (0.5 if raises >= 2 else 1.0)
        if can_raise and (pct >= 1.0 - tb or (raises >= 2 and e >= 0.6)):
            return Action.RAISE_POT
        if can_raise and raises == 1 and flags.get("folder") and pct >= 0.55 and u < 0.35:
            return Action.RAISE_POT
        if e * real >= req + cm:
            return Action.CHECK_CALL
        return Action.FOLD

    # ---------------- postflop ----------------
    ops = _ops_for(obs, memory, walk, P["trust"])
    if N > 0:
        e, er = _mc(obs.hole_cards, obs.board, N, rng, ops)
    else:
        e = 0.5
    si = street - 1
    bet_th = P["bet"][si] + bet_adj
    vr = P["vr"][si] + bet_adj
    in_pos = obs.is_button
    if C > 0:
        req = C / (pot + C)
        allin = C >= obs.my_stack
        if allin or street == 3:
            real = 1.0
        else:
            real = (0.9 if in_pos else 0.85) if street == 1 else (0.96 if in_pos else 0.93)
        if can_raise and e >= vr:
            if u < slow * 0.5 and street < 3 and e < 0.9:
                return Action.CHECK_CALL
            return Action.RAISE_POT
        if can_raise and street < 3 and 0.42 <= e < vr and u < semi * 0.35 and obs.raises_this_street <= 1:
            return Action.RAISE_POT
        if e * real >= req + cm:
            return Action.CHECK_CALL
        # bluff-raise on the river vs. frequent bettors that fold to raises
        if can_raise and street == 3 and flags.get("folder") and u < 0.15 * blf_mult and obs.raises_this_street <= 1:
            return Action.RAISE_POT
        return Action.FOLD
    # not facing a bet
    if e >= bet_th:
        if u < slow and street < 3 and e < 0.92:
            return Action.CHECK_CALL
        if e >= bet_th + P["potth"] or flags.get("station"):
            return _legal_pick(obs, Action.RAISE_POT)
        return _legal_pick(obs, Action.RAISE_HALF)
    if street < 3 and e >= 0.40 and u < semi:
        return _legal_pick(obs, Action.RAISE_HALF if street == 1 else Action.RAISE_POT)
    if e < 0.40 and u < min(0.85, P["blf"][si] * blf_mult):
        return _legal_pick(obs, Action.RAISE_HALF if (street < 3 or flags.get("folder")) else Action.RAISE_POT)
    return Action.CHECK_CALL


_STYLE_CACHE = dict(STYLES)


def choose_action(obs: Observation, memory: dict) -> Action:
    t0 = time.perf_counter()
    try:
        a = _decide(obs, memory)
        if a not in obs.legal_actions:
            a = _legal_pick(obs, a)
    except Exception:
        a = Action.CHECK_CALL if Action.CHECK_CALL in obs.legal_actions else obs.legal_actions[0]
    try:
        memory["last"] = (obs.hand_index, int(obs.street), a, obs.to_call > 0)
        memory["hole"] = (obs.hand_index, tuple(obs.hole_cards))
        memory["t_used"] = memory.get("t_used", 0.0) + (time.perf_counter() - t0)
        memory["n_dec"] = memory.get("n_dec", 0) + 1
    except Exception:
        pass
    return a


def on_hand_end(result: HandResult, memory: dict) -> None:
    try:
        _hand_end(result, memory)
    except Exception:
        pass


def _hand_end(result, memory):
    t0 = time.perf_counter()
    last = memory.get("last")
    reward = result.my_net / 2.0
    if last is not None and last[0] == result.hand_index:
        _, lst, lact, lfacing = last
        # infer the opponent's response to our last action
        if lact in (Action.RAISE_HALF, Action.RAISE_POT):
            resp = Action.CHECK_CALL if result.went_to_showdown else (
                Action.FOLD if result.winner == "me" else None)
            if resp is not None:
                rb = 1  # facing our raise
                _record_opp(memory, Street(lst), resp, True, rb, None)
        elif lact == Action.CHECK_CALL and not lfacing and result.went_to_showdown and lst == 3:
            _record_opp(memory, Street(lst), Action.CHECK_CALL, False, 0, None)
        # luck adjustment for all-ins before the river
        if result.went_to_showdown and lst < 3 and result.opp_hole is not None and len(result.board) == 5:
            # my_hole is not in HandResult; we stored it in memory at decision time
            hole = memory.get("hole")
            if hole is not None and hole[0] == result.hand_index:
                k = {0: 0, 1: 3, 2: 4}[lst]
                e = _exact_equity(hole[1], result.opp_hole, tuple(result.board[:k]), result.board)
                reward = (e - 0.5) * result.pot / 2.0
    reward = max(-25.0, min(25.0, reward))
    arm = memory["arm_hand"].get(result.hand_index)
    if arm is not None and FIXED_ARM is None:
        for a in ARMS:
            b = memory["bandit"][a]
            b[0] *= 0.998
            b[1] *= 0.998
        b = memory["bandit"][arm]
        b[0] += 1.0
        b[1] += reward
        s2 = memory["sigma2"]
        s2[0] += 1.0
        s2[1] += reward * reward
    memory["t_used"] = memory.get("t_used", 0.0) + (time.perf_counter() - t0)

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
