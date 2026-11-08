"""Dev-only evaluation bot for the Task 10 heuristic (not the shipped PolicyBot).

HeuristicBot(params) is BotLike and picklable (functools.partial or the class itself is a
factory). Per half it keeps a TimeManager and a numpy Generator in `memory`; each decision
asks the manager for iters, then runs compute_features + heuristic_action.

The module also works as a bot file (`--candidate sim/heuristic_bot.py`, default params).

Evaluation CLI (repo root):
  .venv/bin/python -m sim.heuristic_bot --opponents train --seeds 3 --deals 1000 --workers 4
  [--params '{"bluff_freq": 0.05}'] [--out results.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import sys
import time

import numpy as np

from bot import equity, preflop
from bot.features import compute_features
from bot.heuristic import HeuristicParams, heuristic_action
from bot.timing import TimeManager
from game import Action, default_action

RNG_SEED = 20261107


class HeuristicBot:
    """compute_features + heuristic_action, with Monte Carlo sized by a TimeManager."""

    def __init__(self, params: HeuristicParams | None = None) -> None:
        self.params = params if params is not None else HeuristicParams()
        equity.warmup()            # the factory call is untimed, like a fresh process's startup
        preflop.load_preflop()

    def new_half(self, config, memory: dict) -> None:
        memory["tm"] = TimeManager(bank_s=config.time_bank_seconds, hands=config.num_hands)
        memory["rng"] = np.random.default_rng(RNG_SEED)

    def choose_action(self, obs, memory: dict) -> Action:
        tm = memory["tm"]
        tm.begin()
        try:
            rng = memory["rng"]
            feats = compute_features(obs, tm.iters(obs.pot, obs.street, obs.hand_index), rng)
            a = heuristic_action(obs, feats, self.params, rng)
            return a if a in obs.legal_actions else default_action(obs.to_call)
        finally:
            tm.end()

    def on_hand_end(self, result, memory: dict) -> None:
        return None


# --- bot-file interface (default params), state in memory only ---------------------------

def new_half(config, memory: dict) -> None:
    memory["bot"] = HeuristicBot()
    memory["bot"].new_half(config, memory)


def choose_action(obs, memory: dict) -> Action:
    return memory["bot"].choose_action(obs, memory)


def on_hand_end(result, memory: dict) -> None:
    return None


# --- evaluation CLI ---------------------------------------------------------------------

def make_params(overrides: dict | None) -> HeuristicParams:
    """HeuristicParams with JSON overrides (lists become tuples)."""
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in (overrides or {}).items()}
    return dataclasses.replace(HeuristicParams(), **kw)


def summarize(rep) -> list:
    """One row per opponent from a FocusReport."""
    rows = []
    for o, by_seed in rep.matches.items():
        ms = [by_seed[s] for s in sorted(by_seed)]
        halves = [h for m in ms for h in m.halves]
        rows.append({
            "opponent": o,
            "mean_bb": float(np.mean([m.mean_bb for m in ms])),
            "seeds": [{"seed": s, "mean_bb": m.mean_bb, "ci": [m.ci_low, m.ci_high],
                       "verdict": m.verdict} for s, m in zip(sorted(by_seed), ms)],
            "W": sum(m.verdict == "A" for m in ms), "L": sum(m.verdict == "B" for m in ms),
            "C": sum(m.verdict == "CHOP" for m in ms),
            "max_bank": max(h.bank_a for h in halves),
            "errors": sum(h.errors_a for h in halves),
            "max_call_ms": 1e3 * max(h.max_call_a for h in halves),
        })
    return rows


def print_rows(rows: list) -> None:
    print(f"{'opponent':22s} {'bb/deal':>8s}  {'W/L/C':>5s}  {'bank':>5s} {'err':>3s}  per-seed [ci]")
    for r in rows:
        seeds = "  ".join(f"{s['mean_bb']:+.2f}[{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}]{s['verdict'][0]}"
                          for s in r["seeds"])
        print(f"{r['opponent']:22s} {r['mean_bb']:+8.3f}  {r['W']}/{r['L']}/{r['C']}  "
              f"{r['max_bank']:5.2f} {r['errors']:3d}  {seeds}")
    print(f"mean bb/deal {np.mean([r['mean_bb'] for r in rows]):+.3f}  "
          f"W/L/C {sum(r['W'] for r in rows)}/{sum(r['L'] for r in rows)}/{sum(r['C'] for r in rows)}  "
          f"max bank {max(r['max_bank'] for r in rows):.2f} s  errors {sum(r['errors'] for r in rows)}")


def main(argv=None) -> int:
    from sim.tournament import focus_eval, select_opponents
    from sim.opponents import ZOO
    ap = argparse.ArgumentParser(prog="sim.heuristic_bot")
    ap.add_argument("--opponents", default="train",
                    help="train | holdout | all | comma-separated zoo names")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-prefix", default="s")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--params", default="{}", help="JSON overrides of HeuristicParams")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.opponents in ("train", "holdout", "all"):
        opps = select_opponents(args.opponents)
    else:
        opps = {n: ZOO[n] for n in args.opponents.split(",")}
    params = make_params(json.loads(args.params))
    seeds = [f"{args.seed_prefix}{i}" for i in range(1, args.seeds + 1)]
    t0 = time.time()
    rep = focus_eval(functools.partial(HeuristicBot, params), opps, seeds, args.deals,
                     args.workers)
    rows = summarize(rep)
    print_rows(rows)
    print(f"({time.time() - t0:.0f} s, load {os.getloadavg()[0]:.1f})")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"params": dataclasses.asdict(params), "seeds": seeds,
                       "deals": args.deals, "rows": rows}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
