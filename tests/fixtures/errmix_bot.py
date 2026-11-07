"""Judge-fidelity fixture: a bot that makes each kind of counted error, then forfeits.

Per half: on_hand_end raises for hands 4, 5 and 6 (against a bot that folds as
button, two of those results arrive in one request, in either seat); choose_action
answers "nonsense" on its 4th decision, FOLD on its first free decision after the
8th (illegal: nothing to call) and raises on its 14th. The fifth of those errors
(ERROR_LIMIT) forfeits the half, and the rest is played with default actions.
"""
from game import Action


def new_half(config, memory):
    memory["n"] = 0
    memory["folded_free"] = False


def choose_action(obs, memory):
    memory["n"] += 1
    n = memory["n"]
    if n == 4:
        return "nonsense"
    if n > 8 and obs.to_call == 0 and not memory["folded_free"]:
        memory["folded_free"] = True
        return Action.FOLD
    if n == 14:
        raise RuntimeError("boom")
    if Action.RAISE_HALF in obs.legal_actions and n % 5 == 0:
        return Action.RAISE_HALF
    return Action.CHECK_CALL


def on_hand_end(result, memory):
    if result.hand_index in (4, 5, 6):
        raise RuntimeError("hand-end boom")
