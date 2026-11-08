"""Opponent-conditioned decision trees and legal-action guards."""
from game import Action, Observation, Street
from dataclasses import dataclass
from poker_eval import FULL, REDUCED, assess, preflop_score
from poker_memory import diagnostic, ingest, profile, remember_raise

CC, FOLD, HALF, POT = Action.CHECK_CALL, Action.FOLD, Action.RAISE_HALF, Action.RAISE_POT


@dataclass(frozen=True)
class Strategy:
    strong_defense: bool = False
    medium_defense: bool = False
    preflop_floor: int = 101
    trap: bool = False
    limp_aggressive: bool = False
    aggression_first: bool = False


# One fixed configuration per half. v04 remains the reproducible control.
STRATEGIES = {
    'v04': Strategy(),
    'v05_price': Strategy(strong_defense=True),
    'v05_medium': Strategy(True, True),
    'v05_defend35': Strategy(True, True, 35),
    'v05_defend30': Strategy(True, True, 30),
    'v05_trap': Strategy(True, True, 30, True),
    'v05_limp': Strategy(True, True, 30, True, True),
    'v05_profile': Strategy(aggression_first=True),
    'v05_aggressive30': Strategy(True, True, 30, True, False, True),
    'v05_aggressive_limp': Strategy(True, True, 30, True, True, True),
    'v05_limp20': Strategy(True, True, 20, True, True, True),
    'v05_limp25': Strategy(True, True, 25, True, True, True),
    'v05_limp35': Strategy(True, True, 35, True, True, True),
    'v05_limp40': Strategy(True, True, 40, True, True, True),
    'v05_limp45': Strategy(True, True, 45, True, True, True),
}


def strategy(memory):
    return STRATEGIES[memory.get('strategy', 'v04')]


def fallback(obs):
    legal = getattr(obs, "legal_actions", ())
    if not legal or any(not isinstance(a, Action) for a in legal):
        raise ValueError("unrecoverable legal action set: protocol failure")
    wanted = CC if getattr(obs, "to_call", 1) == 0 else FOLD
    if wanted in legal:
        return wanted
    return CC if CC in legal else legal[0]


def validate(obs, memory):
    if not isinstance(obs, Observation) or not memory["config_valid"]:
        raise ValueError("unusable observation/config")
    if not isinstance(obs.street, Street) or len(obs.hole_cards) != 5 or len(obs.board) != (0, 3, 4, 5)[obs.street]:
        raise ValueError("street/card count mismatch")
    cards = obs.hole_cards + obs.board
    if len(set(cards)) != len(cards) or any(type(c) is not int or not 0 <= c < 36 for c in cards):
        raise ValueError("invalid physical cards")
    for field in ("pot", "to_call", "my_bet", "opp_bet", "my_stack", "opp_stack", "raises_this_street", "min_raise_to", "hand_index"):
        v = getattr(obs, field)
        if type(v) is not int or v < 0:
            raise ValueError("invalid chip/index quantity " + field)
    cfg = memory["config"]
    if (type(obs.is_button) is not bool or type(obs.big_blind) is not int
            or obs.big_blind != cfg.big_blind
            or obs.pot < obs.my_bet + obs.opp_bet
            or obs.pot + obs.my_stack + obs.opp_stack != 2 * cfg.starting_stack
            or obs.my_bet + obs.my_stack > cfg.starting_stack
            or obs.opp_bet + obs.opp_stack > cfg.starting_stack
            or obs.to_call != max(0, obs.opp_bet - obs.my_bet)
            or obs.min_raise_to <= obs.opp_bet
            or obs.hand_index >= cfg.num_hands
            or obs.raises_this_street > cfg.raise_cap
            or len(obs.action_history) > 4 * (cfg.raise_cap + 2)):
        raise ValueError("inconsistent observation")


def raise_valid(obs, action, cap=None):
    if action not in (HALF, POT) or action not in obs.legal_actions:
        return False
    total = obs.raise_to.get(action)
    if type(total) is not int:
        return False
    cost = total - obs.my_bet
    all_in_to = obs.my_bet + obs.my_stack
    pot_raise_to = obs.opp_bet + obs.pot + obs.to_call
    return (obs.opp_stack > 0 and obs.my_stack > obs.to_call
            and obs.opp_bet < total <= min(all_in_to, pot_raise_to)
            and (total >= obs.min_raise_to or total == all_in_to)
            and (cap is None or cost <= cap * obs.my_stack))


def canonical(obs, action):
    if action in (HALF, POT):
        if not raise_valid(obs, action):
            action = CC
        elif (HALF in obs.legal_actions and POT in obs.legal_actions
              and obs.raise_to.get(HALF) == obs.raise_to.get(POT)):
            action = HALF
    return action if action in obs.legal_actions else fallback(obs)


def price(obs):
    call = min(obs.to_call, obs.my_stack)
    return call / (obs.pot + call) if call else 0.


def preflop(obs, memory, label, fold_pre):
    s = preflop_score(obs.hole_cards)
    cfg = memory["config"]
    policy = strategy(memory)
    if (obs.is_button and not obs.action_history and obs.raises_this_street == 0
            and obs.my_bet == cfg.small_blind and obs.opp_bet == obs.big_blind
            and obs.to_call == obs.big_blind - cfg.small_blind):
        if s < 20:
            return FOLD
        if label == 'AGGRESSIVE' and policy.limp_aggressive:
            return CC
        half, pot = (45, 65) if label == "STATION" else ((35, 55) if memory["adaptation"] and memory["resolved_pre"] >= 12 and fold_pre >= .52 else (40, 60))
        intended = POT if s >= pot else HALF if s >= half else CC
        return intended if intended == CC or raise_valid(obs, intended) else CC
    h = obs.action_history
    if (not obs.is_button and obs.to_call == 0 and obs.my_bet == obs.opp_bet == obs.big_blind
            and obs.raises_this_street == 0 and len(h) == 1 and not h[0].is_me
            and h[0].street == Street.PREFLOP and h[0].action == CC
            and h[0].amount == obs.big_blind - cfg.small_blind):
        intended = POT if s >= 65 else HALF if s >= 45 else CC
        return intended if intended == CC or raise_valid(obs, intended) else CC
    if obs.to_call == 0:
        return CC
    raises = [a for a in h if a.street == Street.PREFLOP and a.action in (HALF, POT)]
    if not raises or raises[-1].is_me or len(raises) != obs.raises_this_street:
        return fallback(obs)
    ceiling = .12 if s < 20 else .21 if s < 35 else .27 if s < 50 else .31 if s < 65 else .34
    ceiling -= .05 * (obs.raises_this_street >= 2) + .04 * (label == "TIGHT")
    ceiling += .02 * (label == "AGGRESSIVE")
    ceiling = round(max(.05, min(.34, ceiling)), 12)
    if label == 'AGGRESSIVE' and s >= policy.preflop_floor:
        ceiling = .34
    p = price(obs)
    if p > ceiling:
        return FOLD
    if label == 'AGGRESSIVE' and policy.trap:
        return CC
    if obs.raises_this_street == 1 and p <= ceiling - .04:
        if s >= 80 and raise_valid(obs, POT, .65):
            return POT
        if s >= 70 and raise_valid(obs, HALF, .60):
            return HALF
    return CC


def call_ceiling(obs, a, label, configuration='v04'):
    policy = STRATEGIES[configuration]
    base = {"WEAK": (.12, .10, .08), "MEDIUM": (.27, .26, .26),
            "STRONG": (.34,) * 3, "NUTS": (.34,) * 3}[a.tier][obs.street - 1]
    if obs.street != Street.RIVER and a.draw >= .10:
        base = max(base, min(.34, .08 + .70 * a.draw))
    base += .02 * (label == "AGGRESSIVE")
    base -= .04 * (label == "TIGHT")
    last = next((x for x in reversed(obs.action_history) if not x.is_me and x.street == obs.street), None)
    base -= .04 * (last is not None and last.action == POT)
    base -= .04 * (obs.raises_this_street >= 2)
    base -= .03 * (a.tier in ("WEAK", "MEDIUM") and a.danger)
    if label == 'AGGRESSIVE' and ((policy.strong_defense and a.tier == 'STRONG')
                                  or (policy.medium_defense and a.tier == 'MEDIUM')):
        return .34
    return .34 if a.locked else round(max(.03, min(.34, base)), 12)


def postflop(obs, memory, a, label, fold_post):
    river = obs.street == Street.RIVER
    k = a.category
    spr = min(obs.my_stack, obs.opp_stack) / max(obs.pot, 1)
    if min(obs.to_call, obs.my_stack) > 0:
        ceiling = call_ceiling(obs, a, label, memory.get('strategy', 'v04'))
        if price(obs) > ceiling:
            return FOLD
        if label == 'AGGRESSIVE' and strategy(memory).trap:
            return CC
        if a.locked:
            if a.tie < .30 and label in ("STATION", "AGGRESSIVE") and raise_valid(obs, POT):
                return POT
            return HALF if raise_valid(obs, HALF) else CC
        if a.vulnerable:
            return CC
        if a.tier == "STRONG":
            if river:
                qualified = label == "STATION" and k >= 5 and a.risk <= .015 and a.tie < .20
                cap = .60
            else:
                qualified = (a.risk <= .02 and k >= 5 and label in ("STATION", "AGGRESSIVE")
                             and price(obs) <= ceiling - .05 and not a.danger)
                cap = .55
            if qualified and raise_valid(obs, HALF, cap):
                return HALF
        return CC
    if not river and obs.raises_this_street:
        return CC
    if a.locked:
        pot_profile = label == "STATION" if river else label in ("STATION", "AGGRESSIVE")
        if pot_profile and a.tie < (.20 if river else .30) and raise_valid(obs, POT):
            return POT
        return HALF if raise_valid(obs, HALF) else CC
    if a.vulnerable:
        return HALF if not a.danger and spr >= .35 and raise_valid(obs, HALF, .50) else CC
    if a.tier == "STRONG":
        low_risk = (a.mode == FULL and a.risk <= .015) or (a.mode == REDUCED and a.risk == 0)
        if (label == "STATION" and k >= 5 and spr <= 2 and not a.danger and low_risk
                and (not river or a.tie < .20) and raise_valid(obs, POT, .65)):
            return POT
        half_ok = not a.danger if river else spr >= .35 and not (a.danger and k < 5)
        return HALF if half_ok and raise_valid(obs, HALF, .65) else CC
    if a.tier == "MEDIUM":
        station_value = label == "STATION" and k >= 1 and a.risk <= .10 and not a.danger
        continuation = (obs.street == Street.FLOP and memory["aggressor"] and k >= 1
                        and a.risk <= .08 and not a.danger and spr >= .75)
        if (station_value or continuation) and raise_valid(obs, HALF, .40):
            return HALF
    if not river and a.tier in ("WEAK", "MEDIUM") and a.draw >= .20 and label != "STATION" and spr >= .75 and not a.danger and raise_valid(obs, HALF, .40):
        return HALF
    if memory["pure_bluffs"] and label == "OVERFOLDER" and spr >= .75:
        if not river:
            positional = memory["aggressor"] if obs.street == Street.FLOP else obs.is_button
            bluff = fold_post >= .50 and k <= 2 and a.draw < .20 and not a.danger and positional
        else:
            checked = any(not x.is_me and x.street == Street.RIVER and x.action == CC and x.amount == 0 for x in obs.action_history)
            blocker = any(sum(c & 3 == s for c in obs.board) >= 3 and (32 + s) in obs.hole_cards
                          and sum(c & 3 == s for c in obs.hole_cards) < 2 for s in range(4)) and k != 6
            bluff = (memory["resolved_post"] >= 12 and fold_post >= .52 and obs.is_button
                     and k <= 1 and checked and blocker)
        if bluff and raise_valid(obs, HALF, .30):
            return HALF
    return CC


def choose(obs, memory):
    safe = fallback(obs)  # unrecoverable legal set is deliberately a protocol error
    try:
        validate(obs, memory)
        ingest(obs, memory)
        label, fp, fq = profile(memory, aggression_first=strategy(memory).aggression_first)
        if obs.street == Street.PREFLOP:
            action = preflop(obs, memory, label, fp)
        else:
            key = (obs.hand_index, obs.street, obs.board)
            if key not in memory["cache"]:
                memory["cache"][key] = assess(obs.hole_cards, obs.board, memory["mode"])
                memory["cache_misses"] += 1
            else:
                memory["cache_hits"] += 1
            action = postflop(obs, memory, memory["cache"][key], label, fq)
        action = canonical(obs, action)
        remember_raise(obs, action, memory)
        return action
    except Exception as exc:
        diagnostic(memory, f"choose_action: {type(exc).__name__}: {exc}")
        return safe
