"""Bounded, per-half, observed-only opponent statistics and terminal transactions."""
from hashlib import sha256
import math

from game import Action, AgentConfig, Street
from poker_eval import FULL, MODES

COUNTERS = ("completed", "first_observed", "vpip", "pre_raise_hands", "post_raises",
            "post_nonfold", "resolved_pre", "pre_folds", "resolved_post", "post_folds")


def diagnostic(memory, message):
    memory["internal_errors"] = memory.get("internal_errors", 0) + 1
    if memory.get("diagnostics_enabled") and len(memory.get("diagnostics", ())) < 32:
        memory["diagnostics"].append(str(message)[:160])


def initialize(config, memory, mode=FULL, adaptation=True, pure_bluffs=True, diagnostics=False):
    memory.clear()
    # Establish safe, complete defaults before inspecting supplied optional values.
    memory.update(config=AgentConfig(), mode=FULL, adaptation=adaptation,
                  pure_bluffs=pure_bluffs, hand=-1, cursor=0, prefix=(),
                  first_seen=False, pre_raise_seen=False, aggressor=False, hole=None,
                  pending=None, cache={}, cache_hits=0, cache_misses=0,
                  diagnostics_enabled=diagnostics, diagnostics=[], internal_errors=0,
                  processed=bytearray(1000), fingerprints=[None] * 1000,
                  conflicts=bytearray(1000), **dict.fromkeys(COUNTERS, 0))
    try:
        if mode not in MODES:
            raise ValueError("unrecognized fixed evaluator mode")
        if not isinstance(config, AgentConfig):
            raise ValueError("required AgentConfig missing")
        for field in ("starting_stack", "big_blind", "small_blind", "num_hands", "raise_cap"):
            v = getattr(config, field)
            if type(v) is not int or v <= 0:
                raise ValueError("invalid config " + field)
        if config.small_blind > config.big_blind or config.big_blind >= config.starting_stack:
            raise ValueError("inconsistent blinds/stack")
        if config.num_hands > 1000000 or not math.isfinite(config.time_bank_seconds) or config.time_bank_seconds <= 0:
            raise ValueError("invalid required config")
        memory.update(config=config, mode=mode, processed=bytearray(config.num_hands),
                      fingerprints=[None] * config.num_hands, conflicts=bytearray(config.num_hands))
    except Exception as exc:
        diagnostic(memory, f"new_half: {exc}")
        memory["config_valid"] = False
        return
    memory["config_valid"] = True


def _response_delta(street, fold):
    return {"resolved_pre" if street == Street.PREFLOP else "resolved_post": 1,
            "pre_folds" if street == Street.PREFLOP else "post_folds": int(fold)}


def ingest(obs, memory):
    """Validate a complete suffix before applying any of its observations."""
    h = obs.action_history
    new_hand = obs.hand_index != memory["hand"]
    if new_hand:
        if obs.hand_index < memory["hand"]:
            raise ValueError("backwards hand index")
        memory.update(hand=obs.hand_index, cursor=0, prefix=(), first_seen=False,
                      pre_raise_seen=False, aggressor=False, pending=None, cache={}, hole=tuple(obs.hole_cards))
    elif tuple(obs.hole_cards) != memory["hole"]:
        raise ValueError("hole cards changed within hand")
    old = memory["prefix"]
    if len(h) < len(old) or tuple(h[:len(old)]) != old:
        raise ValueError("shrinking or conflicting history")
    suffix = h[memory["cursor"]:]
    for a in suffix:
        if (not isinstance(a.action, Action) or not isinstance(a.street, Street)
                or type(a.is_me) is not bool or type(a.amount) is not int or a.amount < 0
                or type(a.street_total) is not int or a.street_total < 0):
            raise ValueError("malformed history")
    for index, a in enumerate(suffix, memory["cursor"]):
        if a.street == Street.PREFLOP and a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            memory["aggressor"] = a.is_me
        if a.is_me:
            continue
        pending = memory["pending"]
        if pending and index == pending["index"] + 1 and a.street == pending["street"]:
            for key, delta in _response_delta(a.street, a.action == Action.FOLD).items():
                memory[key] += delta
            memory["pending"] = None
        if a.street == Street.PREFLOP:
            if not memory["first_seen"]:
                memory["first_seen"] = True
                memory["first_observed"] += 1
                memory["vpip"] += a.action in (Action.RAISE_HALF, Action.RAISE_POT) or (a.action == Action.CHECK_CALL and a.amount > 0)
            if a.action in (Action.RAISE_HALF, Action.RAISE_POT) and not memory["pre_raise_seen"]:
                memory["pre_raise_seen"] = True
                memory["pre_raise_hands"] += 1
        elif a.action != Action.FOLD:
            memory["post_nonfold"] += 1
            memory["post_raises"] += a.action in (Action.RAISE_HALF, Action.RAISE_POT)
    memory.update(cursor=len(h), prefix=tuple(h))


def remember_raise(obs, action, memory):
    if action not in (Action.RAISE_HALF, Action.RAISE_POT):
        return
    pending = memory["pending"]
    candidate = dict(hand=obs.hand_index, street=obs.street, index=len(obs.action_history),
                     total=obs.raise_to[action], pot=obs.pot + obs.raise_to[action] - obs.my_bet,
                     board=tuple(obs.board),
                     ends_betting=(obs.raise_to[action] - obs.my_bet == obs.my_stack
                                   or obs.raise_to[action] - obs.opp_bet >= obs.opp_stack
                                   or obs.street == Street.RIVER))
    if pending == candidate:  # retried identical decision
        return
    if pending:
        diagnostic(memory, "discarded unresolved response before new raise")
    memory["pending"] = candidate


def finish(result, memory):
    try:
        i = result.hand_index
        if type(i) is not int or not 0 <= i < len(memory["processed"]):
            raise ValueError("invalid terminal index")
        if (type(result.my_net) is not int or type(result.pot) is not int or result.pot < 0
                or abs(result.my_net) > result.pot or type(result.went_to_showdown) is not bool
                or result.winner not in ("me", "opp", "split")):
            raise ValueError("invalid terminal primitives")
        b = tuple(result.board)
        opp = tuple(result.opp_hole) if result.opp_hole is not None else ()
        if len(b) not in (0, 3, 4, 5) or any(type(c) is not int or not 0 <= c < 36 for c in b + opp) or len(set(b + opp)) != len(b + opp):
            raise ValueError("invalid terminal cards")
        if result.went_to_showdown:
            if len(b) != 5 or len(opp) != 5:
                raise ValueError("incomplete showdown")
        elif opp or result.winner == "split":
            raise ValueError("inconsistent fold result")
        if i == memory['hand'] and memory['hole'] is not None:
            if set(memory['hole']) & set(b + opp):
                raise ValueError("terminal cards overlap hero")
            for key in memory['cache']:
                visible = key[2]
                if b[:len(visible)] != visible:
                    raise ValueError("terminal board changed observed cards")
        if ((result.winner == "me" and result.my_net < 0)
                or (result.winner == "opp" and result.my_net > 0)
                or (result.winner == "split" and result.my_net != 0)):
            raise ValueError("inconsistent terminal winner")
        fp = sha256(repr((i, result.my_net, result.pot, result.went_to_showdown,
                          b, opp, result.winner)).encode("ascii")).digest()[:16]
        if memory["processed"][i]:
            if memory["fingerprints"][i] != fp and not memory["conflicts"][i]:
                memory["conflicts"][i] = 1
                diagnostic(memory, "conflicting duplicate result " + str(i))
            return
        delta = {"completed": 1}
        pending = memory["pending"]
        clear = pending is not None and pending["hand"] == i
        if clear:
            if (not result.went_to_showdown and result.winner == "me"
                    and b == pending["board"] and result.pot == pending["pot"]):
                delta.update(_response_delta(pending["street"], True))
            elif result.went_to_showdown and pending["ends_betting"]:
                delta.update(_response_delta(pending["street"], False))
        # All scratch validation/calculations complete. No user code in commit phase.
        updates = {key: memory[key] + n for key, n in delta.items()}
        if clear:
            updates["pending"] = None
        memory.update(updates)
        memory["fingerprints"][i] = fp
        memory["processed"][i] = 1
    except Exception as exc:
        diagnostic(memory, f"on_hand_end: {exc}")


def profile(memory, aggression_first=False):
    n = memory["first_observed"]
    vpip = (memory["vpip"] + 6.6) / (n + 12)
    pr = (memory["pre_raise_hands"] + 3.6) / (n + 12)
    pa = (memory["post_raises"] + 3) / (memory["post_nonfold"] + 12)
    fp = (memory["pre_folds"] + 3.84) / (memory["resolved_pre"] + 12)
    fq = (memory["post_folds"] + 3.84) / (memory["resolved_post"] + 12)
    label = "UNKNOWN"
    aggressive = ((memory["post_nonfold"] >= 16 and pa >= .42) or (n >= 20 and pr >= .50))
    if memory["adaptation"]:
        if memory["resolved_post"] >= 12 and fq >= .50:
            label = "OVERFOLDER"
        elif aggression_first and aggressive:
            label = "AGGRESSIVE"
        elif n >= 20 and vpip >= .55 and memory["resolved_post"] >= 10 and fq <= .25:
            label = "STATION"
        elif aggressive:
            label = "AGGRESSIVE"
        elif n >= 30 and vpip <= .27 and pr <= .30:
            label = "TIGHT"
    return label, fp, fq
