"""Simulator-isolation fixture: a bot that keeps state in an imported bot.* module.

The tournament starts every half in a fresh process, so this bot always sees
halves_started() == 1. sim.match must give the same result (tests/test_sim_match.py).
"""
import bot.timing as shared            # any module of our runtime package
from game import Action


def halves_started() -> int:
    return shared.__dict__.get("_fixture_halves_started", 0)


def new_half(config, memory):
    shared.__dict__["_fixture_halves_started"] = halves_started() + 1
    memory["halves_started"] = halves_started()


def choose_action(obs, memory):
    return Action.CHECK_CALL


def on_hand_end(result, memory):
    pass
