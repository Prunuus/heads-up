"""Judge-fidelity fixture: answers its 5th decision of each half too late.

That decision takes SLEEP_S, past the judge's 1 s ACTION_TIMEOUT_SECONDS: one error,
and the late RAISE_POT is discarded in favour of the default action.
"""
import time

from game import Action

SLEEP_S = 1.05


def new_half(config, memory):
    memory["n"] = 0


def choose_action(obs, memory):
    memory["n"] += 1
    if memory["n"] == 5:
        time.sleep(SLEEP_S)
        return Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL
    return Action.CHECK_CALL


def on_hand_end(result, memory):
    pass
