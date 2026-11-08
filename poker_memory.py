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
                  street=None, board=(), is_button=None, contributions=None,
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
        if (config.num_hands > 1000000
                or type(config.time_bank_seconds) not in (int, float)
                or not math.isfinite(config.time_bank_seconds) or config.time_bank_seconds <= 0):
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
    """Commit observed facts only after validating the entire new context.

    The scratch copy shares read-only containers; this function only replaces
    fields and adds integer counters, so rejection leaves memory untouched.
    """
    if (type(obs.hand_index) is not int or not 0 <= obs.hand_index < len(memory["processed"])
            or memory["processed"][obs.hand_index]):
        raise ValueError("invalid or completed observed hand")
    state = memory.copy()
    h = obs.action_history
    new_hand = obs.hand_index != state["hand"]
    if new_hand:
        if obs.hand_index < state["hand"]:
            raise ValueError("backwards hand index")
        state.update(hand=obs.hand_index, cursor=0, prefix=(), first_seen=False,
                     pre_raise_seen=False, aggressor=False, pending=None, cache={},
                     hole=tuple(obs.hole_cards), street=obs.street,
                     board=tuple(obs.board), is_button=obs.is_button)
    else:
        if tuple(obs.hole_cards) != state["hole"] or obs.is_button != state["is_button"]:
            raise ValueError("hole cards or position changed within hand")
        if (obs.street < state["street"]
                or tuple(obs.board[:len(state["board"])]) != state["board"]
                or (obs.street == state["street"] and tuple(obs.board) != state["board"])):
            raise ValueError("backwards street or changed visible board")
    old = state["prefix"]
    if len(h) < len(old) or tuple(h[:len(old)]) != old:
        raise ValueError("shrinking or conflicting history")
    suffix = h[state["cursor"]:]
    previous = old[-1] if old else None
    for a in suffix:
        if (not isinstance(a.action, Action) or not isinstance(a.street, Street)
                or type(a.is_me) is not bool or type(a.amount) is not int or a.amount < 0
                or type(a.street_total) is not int or a.street_total < a.amount
                or (a.action in (Action.RAISE_HALF, Action.RAISE_POT) and a.amount == 0)):
            raise ValueError("malformed history")
        if (a.street > obs.street or (previous is not None
                and (a.street < previous.street
                     or (a.street == previous.street and a.is_me == previous.is_me)))):
            raise ValueError("inconsistent history order")
        previous = a
    pending = state["pending"]
    if pending and len(h) > pending["index"]:
        actual = h[pending["index"]]
        if not (actual.is_me and actual.street == pending["street"]
                and actual.action in (Action.RAISE_HALF, Action.RAISE_POT)
                and actual.street_total == pending["total"]
                and actual.amount == pending["cost"]):
            # The referee may substitute its default after a request fails.
            # A proposal is not evidence that a raise actually happened.
            state["pending"] = None
    for index, a in enumerate(suffix, state["cursor"]):
        if a.street == Street.PREFLOP and a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            state["aggressor"] = a.is_me
        if a.is_me:
            continue
        pending = state["pending"]
        if pending and index == pending["index"] + 1 and a.street == pending["street"]:
            for key, delta in _response_delta(a.street, a.action == Action.FOLD).items():
                state[key] += delta
            state["pending"] = None
        if a.street == Street.PREFLOP:
            if not state["first_seen"]:
                state["first_seen"] = True
                state["first_observed"] += 1
                state["vpip"] += a.action in (Action.RAISE_HALF, Action.RAISE_POT) or (a.action == Action.CHECK_CALL and a.amount > 0)
            if a.action in (Action.RAISE_HALF, Action.RAISE_POT) and not state["pre_raise_seen"]:
                state["pre_raise_seen"] = True
                state["pre_raise_hands"] += 1
        elif a.action != Action.FOLD:
            state["post_nonfold"] += 1
            state["post_raises"] += a.action in (Action.RAISE_HALF, Action.RAISE_POT)
    stack = state["config"].starting_stack
    state.update(cursor=len(h), prefix=tuple(h), street=obs.street, board=tuple(obs.board),
                 contributions=(stack - obs.my_stack, stack - obs.opp_stack))
    memory.update(state)


def remember_raise(obs, action, memory):
    pending = memory["pending"]
    if action not in (Action.RAISE_HALF, Action.RAISE_POT):
        if pending and pending["hand"] == obs.hand_index and pending["index"] == len(obs.action_history):
            memory["pending"] = None
        return
    candidate = dict(hand=obs.hand_index, street=obs.street, index=len(obs.action_history),
                     cost=obs.raise_to[action] - obs.my_bet,
                     total=obs.raise_to[action], pot=obs.pot + obs.raise_to[action] - obs.my_bet,
                     called_pot=obs.pot + obs.raise_to[action] - obs.my_bet
                                + min(obs.raise_to[action] - obs.opp_bet, obs.opp_stack),
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
        if not memory["config_valid"]:
            raise ValueError("unusable terminal config")
        i = result.hand_index
        if type(i) is not int or not 0 <= i < len(memory["processed"]):
            raise ValueError("invalid terminal index")
        if (type(result.my_net) is not int or type(result.pot) is not int or result.pot < 0
                or abs(result.my_net) > result.pot or type(result.went_to_showdown) is not bool
                or result.winner not in ("me", "opp", "split")):
            raise ValueError("invalid terminal primitives")
        stack = memory["config"].starting_stack
        if (result.pot > 2 * stack or abs(result.my_net) > stack
                or result.pot < memory["config"].small_blind + memory["config"].big_blind):
            raise ValueError("terminal chips exceed configured stacks/blinds")
        if result.went_to_showdown:
            expected = {"me": result.pot // 2, "opp": -result.pot // 2, "split": 0}
            if result.pot % 2 or result.my_net != expected[result.winner]:
                raise ValueError("inconsistent showdown payout")
        elif not (0 < abs(result.my_net) < result.pot / 2
                  and result.pot - abs(result.my_net) <= stack):
            raise ValueError("inconsistent fold payout")
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
            if result.went_to_showdown:
                final_contributions = (result.pot // 2,) * 2
            elif result.winner == "me":
                final_contributions = (result.pot - result.my_net, result.my_net)
            else:
                final_contributions = (-result.my_net, result.pot + result.my_net)
            if any(final < observed for final, observed
                   in zip(final_contributions, memory["contributions"])):
                raise ValueError("terminal payout contradicts observed contributions")
            if set(memory['hole']) & set(b + opp):
                raise ValueError("terminal cards overlap hero")
            for key in memory['cache']:
                visible = key[2]
                if b[:len(visible)] != visible:
                    raise ValueError("terminal board changed observed cards")
            if b[:len(memory['board'])] != memory['board']:
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
            elif (result.went_to_showdown and pending["ends_betting"]
                  and result.pot == pending["called_pot"]):
                # A proposed raise may have been replaced by the referee's
                # default. Only a matching final pot supports an immediate call.
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
