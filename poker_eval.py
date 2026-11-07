"""Exact visible-board Omaha features; no simulation or opponent equity claims."""
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from math import comb, gcd

from evaluator import get_table

FULL = "FULL_PAIR_v04"
REDUCED = "REDUCED_64_v04"
MODES = (FULL, REDUCED)
BITS = tuple(1 << c for c in range(36))
SEQUENCES = tuple(frozenset(range(i, i + 5)) for i in range(5)) + (frozenset((8, 0, 1, 2, 3)),)
SUITS = {(2, 2, 1): 12, (2, 1, 1, 1): 4, (3, 2): 7,
         (3, 1, 1): 2, (4, 1): -4, (5,): -9}


def preflop_score(hole):
    ranks = [0] * 9
    suits = [0] * 4
    for c in hole:
        ranks[c >> 2] += 1
        suits[c & 3] += 1
    high = sum(n * v for n, v in zip(ranks, (0, 0, 0, 0, 1, 2, 3, 4, 6)))
    pairs = [r for r in range(9) if ranks[r] >= 2]
    pair = ((2, 3, 4, 5, 7, 9, 12, 15, 18)[pairs[-1]] + 6 * (len(pairs) > 1)) if pairs else 0
    suited = SUITS[tuple(sorted((s for s in suits if s), reverse=True))]
    suited += 4 * any(c >> 2 == 8 and suits[c & 3] >= 2 for c in hole)
    con = 0
    for a, b in combinations((r for r in range(9) if ranks[r]), 2):
        con += 3 if b - a == 1 else 1 if b - a == 2 or (a, b) == (0, 8) else 0
    dup = sum(7 if n == 3 else 14 if n == 4 else 0 for n in ranks)
    return max(0, min(100, 2 * high + pair + suited + min(15, con) - dup))


def sample_indices(hole, board, salt=0):
    n = comb(36 - len(hole) - len(board), 2)
    payload = (b"PB4/REDUCED64\0" + bytes(sorted(hole))
               + bytes((len(board) - 2, len(board))) + bytes(board) + salt.to_bytes(4, "big"))
    digest = sha256(payload).digest()
    offset = int.from_bytes(digest[:8], "big") % n
    step = 1 + int.from_bytes(digest[8:16], "big") % (n - 1)
    while gcd(step, n) != 1:
        step = step % (n - 1) + 1
    return tuple(sorted((offset + j * step) % n for j in range(min(64, n))))


def board_masks(board):
    return tuple(BITS[a] | BITS[b] | BITS[c] for a, b, c in combinations(board, 3))


def pair_score(a, b, masks, table):
    mask = BITS[a] | BITS[b]
    best = 0
    for bm in masks:
        value = table[mask | bm]
        if value > best:
            best = value
    return best


def visible_score(hole, board):
    masks = board_masks(board)
    table = get_table()
    return max(pair_score(a, b, masks, table) for a, b in combinations(hole, 2))


def draw_routes(hole, board, category):
    """Each tuple contains a target and its exact marginal runout probability."""
    k = 5 - len(board)
    if k not in (1, 2) or category >= 6:
        return ()
    u = 36 - 5 - len(board)
    denom = comb(u, k)
    routes = []
    for suit in range(4):
        hs = sum(c & 3 == suit for c in hole)
        bs = sum(c & 3 == suit for c in board)
        need = 3 - bs
        outs = 9 - hs - bs
        if hs >= 2 and 0 < need <= k and outs >= need:
            p = sum(comb(outs, j) * comb(u - outs, k - j)
                    for j in range(need, min(k, outs) + 1)) / denom
            routes.append(("flush", suit, p))
    if category < 3:
        hr = {c >> 2 for c in hole}
        br = {c >> 2 for c in board}
        for seq in SEQUENCES:
            for pair in combinations(sorted(hr & seq), 2):
                target = seq - set(pair)
                missing = target - br
                if not 0 < len(missing) <= k:
                    continue
                outs = [4 - sum(c >> 2 == r for c in hole + board) for r in sorted(missing)]
                p = (1 - comb(u - outs[0], k) / denom if len(outs) == 1
                     else 2 * outs[0] * outs[1] / (u * (u - 1)))
                routes.append(("straight", tuple(sorted(target)), p))
    return tuple(routes)


@dataclass(frozen=True)
class Assessment:
    score: int
    n: int
    beating: int
    ties: int
    risk: float
    tie: float
    tier: str
    current: bool
    locked: bool
    vulnerable: bool
    danger: bool
    draw: float
    mode: str

    @property
    def category(self):
        return self.score >> 20


def assess(hole, board, mode=FULL, salt=0):
    if mode not in MODES:
        raise ValueError("unknown evaluator mode")
    table = get_table()
    masks = board_masks(board)
    score = max(pair_score(a, b, masks, table) for a, b in combinations(hole, 2))
    known = set(hole + board)
    unseen = tuple(c for c in range(36) if c not in known)
    pairs = tuple(combinations(unseen, 2))
    if mode == REDUCED:
        pairs = tuple(pairs[i] for i in sample_indices(hole, board, salt))
    beating = ties = 0
    for a, b in pairs:
        value = pair_score(a, b, masks, table)
        beating += value > score
        ties += value == score
    n = len(pairs)
    risk = beating / n
    cat = score >> 20
    suitmax = max(sum(c & 3 == s for c in board) for s in range(4))
    rankmax = max(sum(c >> 2 == r for c in board) for r in range(9))
    danger = suitmax >= 3 or rankmax >= 3 or (rankmax >= 2 and cat < 5)
    maximum = cat == 8 and (score >> 16 & 15) == 8
    current = maximum or (mode == FULL and beating == 0)
    locked = maximum or (mode == FULL and len(board) == 5 and beating == 0)
    if locked:
        tier = "NUTS"
    elif mode == FULL and (beating == 0 or risk <= .03):
        tier = "STRONG"
    elif mode == REDUCED and risk <= .03 and (not danger or cat >= 5):
        tier = "STRONG"
    else:
        tier = "MEDIUM" if risk <= .12 else "WEAK"
    draw = max((r[2] for r in draw_routes(hole, board, cat)), default=0.)
    return Assessment(score, n, beating, ties, risk, ties / n, tier, current,
                      locked, current and not locked, danger, draw, mode)
