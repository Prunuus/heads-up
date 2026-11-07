"""Opponent zoo: hand-written archetypes plus LLM-generated bots.

HOLDOUT bots must never be tuned against; use TRAIN for development.
"""
from sim.match import bot_file_factory
from sim.opponents.archetypes import ZOO as _ARCHETYPES

_LLM_NAMES = ["simple_rules", "beginner_2h", "aggressive", "mc_equity", "opp_model",
              "gto_inspired", "ml_minded", "time_careful"]

ZOO = dict(_ARCHETYPES)
for _n in _LLM_NAMES:
    ZOO[f"llm_{_n}"] = bot_file_factory(f"sim/opponents/llm/{_n}.py")

# Never tune against these.
HOLDOUT: set[str] = {"llm_time_careful", "llm_aggressive", "llm_opp_model", "threshold_cat"}
TRAIN: set[str] = set(ZOO) - HOLDOUT

# Generated bots that error/forfeit (left unfixed on purpose; realism).
FORFEIT_PRONE: set[str] = set()
