"""Opponent zoo: hand-written archetypes, LLM-generated bots and the agentic FIELD bots.

HOLDOUT and FIELD bots must never be tuned against; use TRAIN for development.
FIELD (sim/opponents/field/README.md) is our proxy for the real tournament field, for ship
decisions only.
"""
from sim.match import bot_file_factory
from sim.opponents.archetypes import ZOO as _ARCHETYPES

_LLM_NAMES = ["simple_rules", "beginner_2h", "aggressive", "mc_equity", "opp_model",
              "gto_inspired", "ml_minded", "time_careful"]
_FIELD_NAMES = ["field_equity", "field_gto", "field_exploit"]

ZOO = dict(_ARCHETYPES)
for _n in _LLM_NAMES:
    ZOO[f"llm_{_n}"] = bot_file_factory(f"sim/opponents/llm/{_n}.py")
for _n in _FIELD_NAMES:
    ZOO[_n] = bot_file_factory(f"sim/opponents/field/{_n}/agent.py")

# Never tune against these.
HOLDOUT: set[str] = {"llm_time_careful", "llm_aggressive", "llm_opp_model", "threshold_cat"}
FIELD: set[str] = set(_FIELD_NAMES)
TRAIN: set[str] = set(ZOO) - HOLDOUT - FIELD

# Generated bots that error/forfeit (left unfixed on purpose; realism).
FORFEIT_PRONE: set[str] = set()
