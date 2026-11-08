"""Frozen copy of the starter strategy: check when free, otherwise fold.

Kept as a fixed reference opponent (agent.py gets rewired to the real bot).
"""

from game import Action, AgentConfig, HandResult, Observation


def new_half(config: AgentConfig, memory: dict) -> None:
    """Nothing to set up."""


def choose_action(obs: Observation, memory: dict) -> Action:
    """Check if it is free, else fold."""
    if obs.to_call == 0:
        return Action.CHECK_CALL
    return Action.FOLD


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Nothing to learn."""
