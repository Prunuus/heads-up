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
PARTICIPANT = "Equilibrium"
AGENT_NAME = "EquilibriumGTO"


# ===== YOUR CODE HERE =====
#
# "Equilibrium": a game-theory-first bot.
#   * Hand strength = Monte Carlo equity from a vectorised numpy evaluator
#     (one sample set per street, cached, so a street costs ~0.3-0.8 ms).
#   * Opponent range model: every opponent bet/raise/call narrows his range to
#     the top fraction f of hands, where f is HIS observed frequency of that
#     action in that spot (Bayesian counts with priors).  Equity is measured
#     against that weighted range, so a maniac's bets get called wide and a
#     nit's bets get respected, with no hand-written per-style rules.
#   * GTO principles: pot-odds/MDF-style calling, polarised bets with
#     bluff:value ratios matched to the bet size, mixed (randomised) actions,
#     some trapping to protect the checking range, position-aware margins.
#   * Exploit only with evidence: bluff frequency scales with the opponent's
#     observed fold-to-bet rate (prior = balanced), value thresholds with his
#     calling rate.
#   * Time guard: measures its own think time and shrinks the sample count if
#     the half's budget is at risk.
import random as _random
import time as _time
from itertools import combinations as _comb, combinations_with_replacement as _cwr

import numpy as np

from evaluator import score_five as _score_five

_FOLD, _CALL, _RH, _RP = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

# ---------------------------------------------------------------------------
# Vectorised short-deck evaluator.  A 5-card hand's rank multiset maps to the
# key sum(5**rank) (counts <= 4, so it is unique); flushes live at +_OFF.
# Scores are computed with the kit's own score_five, so they match omaha_score.
# ---------------------------------------------------------------------------
_OFF = 5 ** 9


def _build_table():
    T = np.zeros(2 * _OFF, dtype=np.int32)
    for combo in _cwr(range(9), 5):
        seen = {}
        cards = []
        for r in combo:
            k = seen.get(r, 0)
            seen[r] = k + 1
            cards.append(r * 4 + k)
        if max(seen.values()) > 4:
            continue
        key = sum(5 ** r for r in combo)
        if len(seen) == 5:
            T[key] = _score_five([combo[0] * 4, combo[1] * 4, combo[2] * 4, combo[3] * 4, combo[4] * 4 + 1])
            T[_OFF + key] = _score_five([r * 4 for r in combo])
        else:
            T[key] = _score_five(cards)
    return T


_T = _build_table()
_CK = np.array([5 ** (c >> 2) for c in range(36)], dtype=np.int64)
_CS = np.array([c & 3 for c in range(36)], dtype=np.int64)
_PK = (_CK[:, None] + _CK[None, :]).ravel()
_PS = np.where(_CS[:, None] == _CS[None, :], _CS[:, None], 4).ravel()
_TK = (_CK[:, None, None] + _CK[None, :, None] + _CK[None, None, :]).ravel()
_SAME3 = (_CS[:, None, None] == _CS[None, :, None]) & (_CS[None, :, None] == _CS[None, None, :])
_TS = np.where(_SAME3, _CS[:, None, None], 5).ravel()
_PI = np.array(list(_comb(range(5), 2)), dtype=np.int64)
_PA, _PB = _PI[:, 0], _PI[:, 1]
_TI5 = np.array(list(_comb(range(5), 3)), dtype=np.int64)
_TI4 = np.array(list(_comb(range(4), 3)), dtype=np.int64)
_TI3 = np.array([[0, 1, 2]], dtype=np.int64)
_SCALE = 1.0 / (1 << 20)
_SUIT4 = np.arange(4, dtype=np.int64)[None, None, :]


def _pairs(H):
    f = H[:, _PA] * 36 + H[:, _PB]
    return _PK[f], _PS[f]


def _tris(B, TI=_TI5):
    f = B[:, TI[:, 0]] * 1296 + B[:, TI[:, 1]] * 36 + B[:, TI[:, 2]]
    return _TK[f], _TS[f]


def _best(pk, ps, tk, ts):
    idx = pk[:, :, None] + tk[:, None, :]
    idx += _OFF * (ps[:, :, None] == ts[:, None, :])
    return _T[idx].max(axis=(1, 2))


def _omaha_np(H, B):
    pk, ps = _pairs(H)
    tk, ts = _tris(B)
    return _best(pk, ps, tk, ts)


# Preflop strength proxy (vectorised), used to rank sampled opponent hands.
_PF_W = (0.3146, 0.0555, -0.176, 0.0338, -0.0305, -0.0106, 0.0015)   # fitted to MC equity, R^2 0.87


def _pf_proxy(H):
    r = H >> 2
    s = H & 3
    hi = r.sum(axis=1) / 40.0
    eqr = r[:, _PA] == r[:, _PB]
    npair = eqr.sum(axis=1)
    pair_ok = np.minimum(npair, 2)
    trips = (npair >= 3)
    sc = np.stack([(s == k).sum(axis=1) for k in range(4)], axis=1)
    nsuit = (sc >= 2).sum(axis=1)
    over = np.maximum(sc - 3, 0).sum(axis=1)
    ace = (r == 8)
    ace_suited = ((ace[:, :, None]) & (s[:, :, None] == s[:, None, :]) & (r[:, None, :] != 8)).any(axis=(1, 2))
    d = np.abs(r[:, _PA] - r[:, _PB])
    conn = ((d >= 1) & (d <= 2)).sum(axis=1)
    w = _PF_W
    return (w[0] * hi + w[1] * pair_ok + w[2] * trips + w[3] * nsuit + w[4] * over
            + w[5] * ace_suited + w[6] * conn)


_PF_SORTED = np.sort(_pf_proxy(np.sort(np.argsort(np.random.default_rng(7).random((20000, 36)), axis=1)[:, :5], axis=1)))


def _pf_strength(hole):
    """Percentile (0..1) of a preflop hand among random hands, by the fitted proxy."""
    v = float(_pf_proxy(np.array(hole, dtype=np.int64)[None, :])[0])
    return float(np.searchsorted(_PF_SORTED, v)) / len(_PF_SORTED)


# Random position pools for sampling unseen cards: row = (missing board cards,
# then 5 opponent cards), as indices into the sorted list of unseen cards.
_POOL_N = 6000
_NPR = np.random.default_rng(20261107)
_POOLS = {nb: np.argsort(_NPR.random((_POOL_N, 31 - nb)), axis=1)[:, :(10 - nb)].astype(np.int64)
          for nb in (0, 3, 4, 5)}
_JIT = _NPR.random(_POOL_N) * 1e-7


def _street_samples(hole, board, K, rng):
    """Sample K (runout, opponent hand) pairs.  Returns (result, pct) where
    result is 1/0.5/0 for my win/tie/loss and pct the percentile of the
    opponent's current-street strength proxy among the samples."""
    nb = len(board)
    known = set(hole)
    known.update(board)
    rem = np.array([c for c in range(36) if c not in known], dtype=np.int64)
    st = rng.randrange(0, _POOL_N - K + 1)
    cards = rem[_POOLS[nb][st:st + K]]
    need = 5 - nb
    H = np.array(hole, dtype=np.int64)[None, :]
    mpk, mps = _pairs(H)
    O = cards[:, need:need + 5]
    opk, ops = _pairs(O)
    if need:
        B = np.empty((K, 5), dtype=np.int64)
        if nb:
            B[:, :nb] = board
        B[:, nb:] = cards[:, :need]
        tk, ts = _tris(B)
        my = _best(np.broadcast_to(mpk, (K, 10)), np.broadcast_to(mps, (K, 10)), tk, ts)
    else:
        tk, ts = _tris(np.array(board, dtype=np.int64)[None, :])
        my = _best(mpk, mps, tk, ts)[0]
    opp = _best(opk, ops, tk, ts)
    res = (my > opp).astype(np.float64) + 0.5 * (my == opp)
    if nb == 0:
        proxy = _pf_proxy(O)
    elif nb == 5:
        proxy = opp * _SCALE
    else:
        bb = np.array(board, dtype=np.int64)[None, :]
        btk, bts = _tris(bb, _TI3 if nb == 3 else _TI4)
        made = _best(opk, ops, btk, bts) * _SCALE
        bsuit = np.bincount(np.array(board, dtype=np.int64) & 3, minlength=4)
        osuit = ((O[:, :, None] & 3) == _SUIT4).sum(axis=1)
        fd = ((bsuit[None, :] == 2) & (osuit >= 2)).any(axis=1) & (made < 6.0)
        proxy = made + 1.9 * fd
    proxy = proxy + _JIT[st:st + K]
    order = np.argsort(proxy)
    pct = np.empty(K)
    pct[order] = (np.arange(K) + 0.5) / K
    return res, pct


def _eq_range(res, pct, f):
    if f >= 0.97:
        return float(res.mean())
    cut = 1.0 - f
    w = np.where(pct >= cut, 1.0, np.exp((pct - cut) / (0.35 * f)))
    return float(np.dot(w, res) / w.sum())


# ---------------------------------------------------------------------------
# River equilibrium table (dev/river_cfr.py): CFR+ on the exact river betting
# tree, 16 strength buckets per player, several stack-to-pot ratios.
# ---------------------------------------------------------------------------
import json as _json
import math as _math
import os as _os


def _load_cfr():
    try:
        path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "river_cfr.json")
        with open(path, encoding="utf-8") as fh:
            d = _json.load(fh)
        tabs = [{h: (acts, np.array(st, dtype=np.float64)) for h, (acts, st) in t.items()}
                for t in d["tables"]]
        spr = np.array([float(x) for x in d["spr"]])
        return {"N": int(d["N"]), "logspr": np.log(spr), "tables": tabs}
    except Exception:
        return None


_CFR = _load_cfr()
_HCH = {Action.CHECK_CALL: "c", Action.RAISE_HALF: "h", Action.RAISE_POT: "p", Action.FOLD: "f"}
_ACH = {"f": Action.FOLD, "c": Action.CHECK_CALL, "h": Action.RAISE_HALF, "p": Action.RAISE_POT}


def _cfr_probs(obs, memory, eq_start):
    """Equilibrium action probabilities for this river node, or None."""
    C = _CFR
    if C is None:
        return None
    hist = "".join(_HCH[a.action] for a in obs.action_history if a.street == Street.RIVER)
    P = obs.pot - obs.my_bet - obs.opp_bet
    if P <= 0:
        return None
    S = max(0.01, (memory.get("stack0", 80) - P / 2.0) / P)
    tab = C["tables"][int(np.argmin(np.abs(C["logspr"] - _math.log(S))))]
    ent = tab.get(hist)
    if ent is None:
        ent = tab.get(hist.replace("h", "p"))
        if ent is None:
            return None
    acts, st = ent
    N = C["N"]
    x = eq_start * N - 0.5
    i0 = int(_math.floor(x))
    fr = x - i0
    i0c = min(max(i0, 0), N - 1)
    i1c = min(max(i0 + 1, 0), N - 1)
    row = (1.0 - fr) * st[i0c] + fr * st[i1c]
    pa = {}
    for ch, pr in zip(acts, row):
        a = _ACH[ch]
        pa[a] = pa.get(a, 0.0) + float(pr)
    legal = obs.legal_actions
    for a in (_RH, _RP):
        if a in pa and a not in legal:
            other = _RP if a == _RH else _RH
            tgt = other if other in legal else _CALL
            pa[tgt] = pa.get(tgt, 0.0) + pa.pop(a)
    if _FOLD in pa and _FOLD not in legal:
        pa[_CALL] = pa.get(_CALL, 0.0) + pa.pop(_FOLD)
    for a in legal:
        pa.setdefault(a, 0.0)
    return pa


# ---------------------------------------------------------------------------
# Opponent model: counts of (fold, call/check, raise) per context, with priors.
# ---------------------------------------------------------------------------
_PRIOR = {
    "pf_open": (1.0, 2.0, 3.0),    # opp is SB, first action
    "pf_limp": (0.0, 3.0, 2.0),    # opp is BB, we limped (check / raise)
    "pf_face": (2.0, 3.0, 1.0),    # opp faces a preflop raise
    "po_first": (0.0, 3.3, 2.7),   # postflop, no bet to face (check / bet)
    "po_face": (2.4, 2.4, 1.2),    # postflop, facing a bet
}
_GAMMA = 0.7


def _freq(stats, ctx):
    c = stats[ctx]
    p = _PRIOR[ctx]
    n = c[0] + c[1] + c[2] + p[0] + p[1] + p[2]
    return ((c[0] + p[0]) / n, (c[1] + p[1]) / n, (c[2] + p[2]) / n)


def _replay(obs, hs, stats, stack0=80):
    """Walk this hand's history; for opponent actions not yet seen, update
    the stats and narrow the opponent's range fraction hs['f']."""
    hist = obs.action_history
    n = len(hist)
    if n <= hs["nproc"]:
        return
    me_sb = obs.is_button
    bets = {True: 1 if me_sb else 2, False: 2 if me_sb else 1}
    total = dict(bets)
    street = 0
    raises = 0
    for i, a in enumerate(hist):
        if a.street != street:
            street = a.street
            bets = {True: 0, False: 0}
            raises = 0
        mine = a.is_me
        facing = bets[not mine] > bets[mine]
        if i >= hs["nproc"] and not mine:
            hs["pending"] = None
            stack = stack0 - total[mine]
            opp_stack_left = stack0 - total[not mine]
            to_call = bets[not mine] - bets[mine]
            could_raise = raises < 4 and stack > to_call and opp_stack_left > 0
            if street == 0:
                if not facing:
                    ctx = "pf_limp"
                elif i == 0:
                    ctx = "pf_open"
                else:
                    ctx = "pf_face"
            else:
                ctx = "po_face" if facing else "po_first"
            kind = 0 if a.action == _FOLD else (2 if a.action in _RAISES else 1)
            if street == 3 and "f3" not in hs:
                hs["f3"] = hs["f"]
            pf, pc, pr = _freq(stats, ctx)
            if kind == 2:
                hs["f"] *= max(pr, 0.03) ** _GAMMA
                if street == 3:
                    hs["opp_riv_agg"] = True
            elif kind == 1 and facing:
                hs["f"] *= max(1.0 - pf, 0.05) ** (0.6 * _GAMMA)
            hs["f"] = max(hs["f"], 0.02)
            if could_raise:
                stats[ctx][kind] += 1
        if a.action in _RAISES:
            raises += 1
        bets[mine] = a.street_total
        total[mine] += a.amount
    hs["nproc"] = n


def _ramp(x, lo, hi):
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


# Preflop equity-vs-random quantiles (5%,10%,...,95%), measured offline.
_PF_Q = (0.392, 0.415, 0.428, 0.441, 0.450, 0.460, 0.469, 0.478, 0.487, 0.497,
         0.506, 0.517, 0.527, 0.538, 0.548, 0.557, 0.567, 0.583, 0.599)


def _pf_pct(eq):
    q = _PF_Q
    if eq <= q[0]:
        return 0.05 * max(0.0, (eq - (q[0] - 0.06)) / 0.06)
    for i in range(1, len(q)):
        if eq <= q[i]:
            return 0.05 * i + 0.05 * (eq - q[i - 1]) / max(1e-9, q[i] - q[i - 1])
    return min(1.0, 0.95 + 0.05 * (eq - q[-1]) / 0.06)


def _policy(obs, memory, hs, eq, eq0):
    rng = memory["rng"]
    stats = memory["stats"]
    legal = obs.legal_actions
    can_raise = _RP in legal
    street = int(obs.street)
    to_call = obs.to_call
    pot = obs.pot
    u = rng.random()
    v = rng.random()
    allin_call = to_call >= obs.my_stack

    # ---------------- preflop ----------------
    if street == 0:
        p = hs["pfp"]
        # steal more only when the opponent has shown he over-folds to raises
        pff = _freq(stats, "pf_face")[0]
        steal = max(0.0, min(0.25, 1.2 * (pff - 0.40)))
        if to_call > 0 and len(obs.action_history) == 0:
            # SB (button) opening decision; in position for the whole hand.
            if p < 0.10 + 0.06 * u - steal * 0.4:
                return _FOLD
            if can_raise and u < _ramp(p, 0.28 - steal, 0.48 - steal):
                return _RP if v < 0.85 else _RH
            return _CALL
        if to_call == 0:
            # BB after a limp: raise strong hands, check the rest.
            if can_raise and u < _ramp(p, 0.62 - steal, 0.78 - steal):
                return _RP
            return _CALL
        need = to_call / (pot + to_call)
        margin = 0.0 if allin_call else (0.025 if not obs.is_button else 0.005)
        if can_raise and u < _ramp(eq, 0.545, 0.60):
            return _RP
        if eq >= need + margin:
            return _CALL
        return _FOLD

    # ---------------- postflop ----------------
    ip = obs.is_button
    river = street == 3
    pf, pc, pr = _freq(stats, "po_face")      # opponent's response to our bets
    if river and hs.get("eq_start") is not None:
        pa = _cfr_probs(obs, memory, hs["eq_start"])
        if pa is not None:
            return _river_mix(obs, memory, hs, pa, eq, eq0, pf, u)
    if to_call == 0:
        # value threshold rises with how often the opponent folds to a bet
        base = (0.0, 0.57, 0.585, 0.60)[street]
        t_val = base + 0.30 * (pf - 0.35)
        t_big = t_val + 0.12
        if eq >= t_val:
            if not ip and eq >= t_big and u < 0.15:
                return _CALL                       # trap: protect the checking range
            if can_raise:
                big = eq >= t_big or obs.my_stack <= pot or pf < 0.22   # stations pay the big size
                if v < 0.2:
                    big = not big                  # mix sizes so size does not leak strength
                return _RP if big else _RH
            return _CALL
        # bluffs / semi-bluffs, ratio matched to a pot-sized bet (1 bluff : 2 value)
        xf = min(1.3, pf / 0.40)
        if can_raise:
            if river:
                if eq < 0.30 and u < 0.33 * xf:
                    return _RP
            else:
                lo = 0.30 if street == 1 else 0.28
                if lo <= eq < t_val - 0.04 and u < 0.20 * xf:
                    return _RP if v < 0.7 else _RH
        return _CALL

    # Showdown evidence: if his river bets often turn out to be bluffs, his
    # betting range is wider than the frequency model thinks.
    blf = memory["sdb"]
    b = (blf[0] + 1.2) / (blf[1] + 8.0)          # prior ~0.15 bluff share
    w = max(0.0, min(0.8, (b - 0.18) / 0.32))
    if not river:
        w *= 0.6
    eq = (1.0 - w) * eq + w * max(eq, eq0)
    need = to_call / (pot + to_call)
    if allin_call:
        margin = 0.0
    elif river:
        margin = 0.01
    else:
        margin = 0.015 if ip else 0.04
    t_raise = (0.0, 0.64, 0.66, 0.70)[street]
    if can_raise and eq >= t_raise and u < _ramp(eq, t_raise, t_raise + 0.06):
        return _RP
    if can_raise and not river and need <= eq < need + 0.12:
        fr, _, _ = _freq(stats, "po_face")
        if u < 0.07 * min(1.3, fr / 0.40):
            return _RP                             # semi-bluff raise
    if eq >= need + margin:
        return _CALL
    if can_raise and river and eq < 0.12 and u < 0.03 * min(1.3, pf / 0.40):
        return _RP                                 # rare river bluff-raise
    return _FOLD


def _river_mix(obs, memory, hs, pa, eq, eq0, pf, u):
    """Equilibrium river strategy + evidence-gated exploitative overlays."""
    to_call = obs.to_call
    eqs = hs["eq_start"]
    raises = [a for a in (_RH, _RP) if a in obs.legal_actions]
    passive = _FOLD if to_call > 0 else _CALL
    # (a) bluffs only pay if he folds: scale weak-hand raises by his fold rate
    if eqs < 0.5 and raises:
        xf = min(1.3, pf / 0.40)
        for a in raises:
            new = min(1.0, pa[a] * xf)
            pa[passive] = max(0.0, pa[passive] + pa[a] - new)
            pa[a] = new
    # (b) thin value against players who do not fold
    if to_call == 0 and raises and pf < 0.25:
        pv = _ramp(eqs, 0.55, 0.65)
        cur = sum(pa[a] for a in raises)
        if pv > cur:
            add = pv - cur
            take = min(add, pa[_CALL])
            pa[_CALL] -= take
            pa[_RP if _RP in raises else raises[0]] += take
    # (c) call/fold: blend equilibrium with the frequency model as evidence grows
    if to_call > 0:
        st = memory["stats"]
        n_po = sum(st["po_first"]) + sum(st["po_face"])
        lam = 0.35 + 0.5 * min(1.0, n_po / 150.0)
        p_raise = sum(pa[a] for a in raises)
        cont_cfr = pa[_CALL] / max(1e-9, pa[_CALL] + pa.get(_FOLD, 0.0))
        need = to_call / (obs.pot + to_call)
        margin = 0.0 if to_call >= obs.my_stack else 0.01
        blf = memory["sdb"]
        b = (blf[0] + 1.2) / (blf[1] + 8.0)       # shown-bluff share of his river bets
        w = max(0.0, min(0.8, (b - 0.18) / 0.32))
        eq = (1.0 - w) * eq + w * max(eq, eq0)
        model = 1.0 if eq >= need + margin else 0.0
        cont = (1.0 - lam) * cont_cfr + lam * model
        rest = max(0.0, 1.0 - p_raise)
        pa[_CALL] = rest * cont
        if _FOLD in pa:
            pa[_FOLD] = rest * (1.0 - cont)
    tot = sum(pa.values())
    if tot <= 0:
        return _CALL
    x = u * tot
    for a in (_FOLD, _CALL, _RH, _RP):
        if a in pa:
            x -= pa[a]
            if x <= 0:
                return a
    return _CALL


def _budget_k(memory, street, hand_index):
    """Samples for this street.  Targets ~1.1 ms of own think time per hand;
    on a slow or overloaded CPU the measured rate shrinks K, and hard caps
    (1.3 s -> K/3, 1.8 s -> K=32, 2.5 s -> no sampling; doubled on an unstarved CPU) protect the bank, because
    the judge also charges HTTP/JSON overhead that this code cannot see."""
    base = (300, 280, 240, 300)[street]
    used = memory.get("t_used", 0.0)
    m = _cap_mult(memory, hand_index)
    if used > 1.8 * m:
        return 32
    if used > 1.3 * m:
        base //= 3
    if hand_index >= 10:
        rate = used / (hand_index + 1)
        if rate > 0.0011 * m:
            base = int(base * 0.0011 * m / rate)
        # starved host (wall time >> our CPU time): every request is slow,
        # including the server overhead we cannot measure, so think less
        ratio = _starve_ratio(memory)
        if ratio > 1.3 and _CAP_MULT_OVERRIDE is None:
            base = int(base * 1.3 / ratio)
    return max(32, min(base, 1000))


_CAP_MULT_OVERRIDE = None     # testing hook (harness), never set in play


def _cap_mult(memory, hand_index):
    """2x budget on an unstarved CPU (our wall time ~ our CPU time: the server
    overhead then mostly fits in the judge's free 1 ms per request); 1x when
    the host is starving us."""
    if _CAP_MULT_OVERRIDE is not None:
        return _CAP_MULT_OVERRIDE
    if hand_index < 10:
        return 1.0
    return 2.0 if _starve_ratio(memory) < 1.25 else 1.0


def _starve_ratio(memory):
    """Wall time / CPU time of our own code.  An implausibly small CPU reading
    means the CPU clock is unusable: answer 1.3 (conservative caps, normal K)."""
    used = memory.get("t_used", 0.0)
    cpu = memory.get("t_cpu", 0.0)
    if used <= 0.0:
        return 1.0
    if cpu < 0.05 * used:
        return 1.3
    return used / cpu


def _new_hand_state(idx):
    return {"idx": idx, "nproc": 0, "f": 1.0, "street": -1, "res": None, "pct": None,
            "pending": None, "eq0": 0.5, "pfp": 0.5, "opp_riv_agg": False, "eq_start": None}


def _choose(obs, memory):
    rng = memory["rng"]
    hs = memory.get("hand")
    if hs is None or hs["idx"] != obs.hand_index:
        hs = _new_hand_state(obs.hand_index)
        memory["hand"] = hs
    _replay(obs, hs, memory["stats"], memory.get("stack0", 80))
    street = int(obs.street)
    if street == 0 and hs.get("pfp_done") is None:
        hs["pfp"] = _pf_strength(obs.hole_cards)
        hs["pfp_done"] = True
    # Preflop Monte Carlo equity is better than the fitted proxy; skip it only
    # in unraised spots when the time bank is getting used up.
    need_samples = street > 0 or (obs.to_call > 0 and len(obs.action_history) > 0)
    if hs["street"] != street and need_samples:
        K = _budget_k(memory, street, obs.hand_index)
        hs["res"], hs["pct"] = _street_samples(tuple(obs.hole_cards), tuple(obs.board), K, rng)
        hs["street"] = street
        hs["eq0"] = float(hs["res"].mean())
        if street == 0:
            hs["pfp"] = _pf_pct(hs["eq0"])
    if hs["res"] is None or hs["street"] != street:
        # unraised preflop spot: strength percentile is all the policy uses
        act = _policy(obs, memory, hs, 0.5, 0.5)
    else:
        act = _policy(obs, memory, hs, _current_eq(hs, memory, street), hs["eq0"])
    if act in _RAISES:
        hs["pending"] = "pf_face" if street == 0 else "po_face"
    elif obs.to_call == 0 and street == 3 and not obs.is_button:
        hs["pending"] = "po_first"
    else:
        hs["pending"] = None
    return act


def _current_eq(hs, memory, street):
    f = hs["f"]
    if f < 0.97:
        # honest bettors (his aggressive showdowns are rarely weak): narrow harder
        blf = memory["sdb"]
        b = (blf[0] + 1.2) / (blf[1] + 8.0)
        if b < 0.15:
            f = f ** (1.0 + 0.5 * (0.15 - b) / 0.15)
    eq = _eq_range(hs["res"], hs["pct"], f)
    if street == 3:
        f3 = hs.get("f3", hs["f"])
        hs["f3"] = f3
        if hs.get("eq_start") is None:
            if f3 < 0.97:
                blf = memory["sdb"]
                b = (blf[0] + 1.2) / (blf[1] + 8.0)
                if b < 0.15:
                    f3 = f3 ** (1.0 + 0.5 * (0.15 - b) / 0.15)
            hs["eq_start"] = _eq_range(hs["res"], hs["pct"], f3)
    return eq


def new_half(config: AgentConfig, memory: dict) -> None:
    """Fresh opponent model and RNG for every half."""
    memory["rng"] = _random.Random()
    memory["stack0"] = int(getattr(config, "starting_stack", 80) or 80)
    memory["t_used"] = 0.0
    memory["t_cpu"] = 0.0
    memory["stats"] = {k: [0, 0, 0] for k in _PRIOR}
    memory["hand"] = None
    memory["sdb"] = [0, 0]     # [river-aggression showdowns with a below-random hand, all such showdowns]


def choose_action(obs: Observation, memory: dict) -> Action:
    """Never raises; always returns a legal action."""
    t0 = _time.perf_counter()
    c0 = _time.process_time()
    legal = obs.legal_actions
    try:
        if "rng" not in memory:
            new_half(AgentConfig(), memory)
        if memory.get("t_used", 0.0) > 2.5 * _cap_mult(memory, obs.hand_index):
            # emergency mode (time bank at risk on a very slow host): no sampling
            if obs.to_call == 0:
                act = _CALL
            elif obs.to_call * 4 <= obs.pot:
                act = _CALL
            else:
                act = _FOLD
        else:
            act = _choose(obs, memory)
        if act not in legal:
            if act in _RAISES or obs.to_call == 0 or _FOLD not in legal:
                act = _CALL            # a wanted raise that is not allowed becomes a call
            else:
                act = _FOLD
    except Exception:
        act = _CALL if (obs.to_call == 0 or _FOLD not in legal) else _FOLD
    memory["t_used"] = memory.get("t_used", 0.0) + (_time.perf_counter() - t0)
    memory["t_cpu"] = memory.get("t_cpu", 0.0) + (_time.process_time() - c0)
    return act


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Close the opponent-response stats for actions the history never showed."""
    t0 = _time.perf_counter()
    c0 = _time.process_time()
    try:
        hs = memory.get("hand")
        if hs is not None and hs["idx"] == result.hand_index and hs.get("pending"):
            ctx = hs["pending"]
            if result.went_to_showdown:
                memory["stats"][ctx][1] += 1
            elif result.winner == "me" and ctx != "po_first":
                memory["stats"][ctx][0] += 1
            hs["pending"] = None
        if (hs is not None and hs["idx"] == result.hand_index and hs.get("opp_riv_agg")
                and result.went_to_showdown and result.opp_hole and len(result.board) == 5):
            res, _ = _street_samples(tuple(result.opp_hole), tuple(result.board), 160, memory["rng"])
            memory["sdb"][1] += 1
            if float(res.mean()) < 0.5:
                memory["sdb"][0] += 1
    except Exception:
        pass
    memory["t_used"] = memory.get("t_used", 0.0) + (_time.perf_counter() - t0)
    memory["t_cpu"] = memory.get("t_cpu", 0.0) + (_time.process_time() - c0)


# Everything built at import (lookup tables, sample pools, the kit's own
# evaluator table) lives for the whole process: move it out of the garbage
# collector's sight so collections during play stay cheap.
import gc as _gc

_gc.collect()
_gc.freeze()

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
