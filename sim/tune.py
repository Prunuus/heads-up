"""Task 12 tuner: random search, then coordinate descent, over HeuristicParams.

Objective (D1): expected points + 0.01 * mean bb/deal of PolicyBot(params) against the
opponents. For each opponent, the per-deal results of every seed in the batch are pooled
into a mean m and a standard deviation sd, and the expected points are those of a
tournament match (MATCH_DEALS paired deals, the judge's 95% CI verdict) for a bot with
that edge: P(win) + 0.5 P(chop), with P(win) = Phi(m / se - Z), P(loss) = Phi(-m / se - Z),
se = sd / sqrt(MATCH_DEALS). That is the quantity the tournament pays, and it is smooth in
m, whereas the 0 / 0.5 / 1 verdicts of short tuning matches score a shorter match and
move in steps. The raw verdict points are logged next to it.

Search (search() is the pure part, tune() supplies the evaluator):
1. Random search: one batch of the base parameters plus random neighbours of them (each
   parameter moves to another value of its list with probability 3 / len(space)).
2. Coordinate descent from the best of that batch: for each parameter in turn, a batch
   of the incumbent and its neighbouring values (one step either way in the list). A
   challenger replaces the incumbent only if it beats it by more than min_gain in the
   same batch. Passes repeat until one moves nothing or the budget runs out.
Every candidate in a batch plays the same seeds (paired comparisons). With batch_seeds
set, consecutive batches take consecutive groups of that many seeds from the pool, so
the search sees a broad seed pool and the incumbent is re-measured on fresh seeds in
every batch, which damps the winner's curse (T10 review M8). The budget counts candidate
evaluations, incumbents included. Every parameter set is a HeuristicParams, whose
constructor rejects non-finite values and short tuples (M3).

Opponents: TRAIN only, by construction (train_opponents). The plan forbids tuning against
HOLDOUT, and every other split (FIELD, "all") is held out too, so the tuner takes the
TRAIN split or names of TRAIN members and looks their factories up in the ZOO itself;
anything else raises before a match is played. Held-out evaluation (the M1 gate) belongs
to sim.tournament / sim.heuristic_bot.

CLI (repo root; Track A pins to CPUs 0-7, ruling R32):
  taskset -c 0-7 .venv/bin/python -m sim.tune --budget 80 --seeds 8 --batch-seeds 2 \\
      --deals 500 --workers 6 --time-scale 2.5 --out tune.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from bot.heuristic import HeuristicParams
from bot.policy import PolicyBot
from sim.tournament import Z, _per_deal_bb, _run_job

MATCH_DEALS = 1000          # paired deals in a tournament match (both halves)
BB_WEIGHT = 0.01            # D1: expected points + 0.01 * mean bb/deal

# The search space: for each parameter, its base value and one step either side (per-street
# fields move all three streets together). default_space() centres it on the current
# defaults; Task 12's run used it centred on the T10 v1 values.
STEPS: dict = {
    "open_min": 0.04, "threebet_min": 0.03, "call_margin": 0.02, "range_discount": 0.04,
    "value_raise": 0.04, "pot_vs_half_cut": 0.05, "bluff_freq": 0.05, "nonnut_penalty": 0.04,
    "range_weight": 0.1, "bet_range": 0.04, "raise_step": 0.05, "thin_value": 0.03,
    "pre_discount": 0.03, "ip_thin_shift": 0.03,
}


def default_space(base: HeuristicParams | None = None) -> dict:
    """{field: [base - step, base, base + step]} for every field in STEPS (values rounded
    to 4 places; a per-street tuple shifts as a whole)."""
    base = base if base is not None else HeuristicParams()
    space = {}
    for name, step in STEPS.items():
        v = getattr(base, name)
        if isinstance(v, tuple):
            space[name] = [tuple(round(x + k * step, 4) for x in v) for k in (-1, 0, 1)]
        else:
            space[name] = [round(v + k * step, 4) for k in (-1, 0, 1)]
    return space


@dataclass(frozen=True)
class Score:
    """One candidate's result in one batch."""

    objective: float
    points: float              # mean expected points of a tournament match (module doc)
    match_points: float        # mean verdict points of the tuning matches themselves
    mean_bb: float             # mean bb/deal over opponents
    per_opponent: dict         # name -> (mean bb/deal, sd, expected points)
    errors: int
    forfeits: int
    max_bank: float


@dataclass(frozen=True)
class Step:
    """One evaluated candidate: batch number, phase, seeds, parameters, score."""

    batch: int
    phase: str
    seeds: tuple
    params: HeuristicParams
    score: Score


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def expected_points(mean: float, sd: float, deals: int = MATCH_DEALS) -> float:
    """Expected points of a `deals`-deal match for a true edge `mean` bb/deal with per-deal
    standard deviation `sd`, under the judge's verdict (95% CI on the mean)."""
    if not sd > 0.0:
        return 1.0 if mean > 0 else 0.0 if mean < 0 else 0.5
    se = sd / math.sqrt(deals)
    win, loss = _phi(mean / se - Z), _phi(-mean / se - Z)
    return win + 0.5 * (1.0 - win - loss)


def objective(points: float, mean_bb: float) -> float:
    """D1: expected points + 0.01 * mean bb/deal."""
    return points + BB_WEIGHT * mean_bb


def score_matches(matches: dict) -> Score:
    """Score one candidate from {opponent: [MatchStats, ...]} (the batch's seeds).
    A forfeited half scores -1: such a candidate can never win the search."""
    per, pts, raw, bbs = {}, [], [], []
    errors = forfeits = 0
    bank = 0.0
    for o, ms in matches.items():
        d = np.concatenate([_per_deal_bb(m) for m in ms])
        mu = float(d.mean())
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        e = expected_points(mu, sd)
        per[o] = (mu, sd, e)
        pts.append(e)
        bbs.append(mu)
        raw.extend(m.points_a for m in ms)
        for m in ms:
            for h in m.halves:
                errors += h.errors_a
                forfeits += h.forfeit_a
                bank = max(bank, h.bank_a)
    points, mean_bb = float(np.mean(pts)), float(np.mean(bbs))
    obj = -1.0 if forfeits else objective(points, mean_bb)
    return Score(obj, points, float(np.mean(raw)), mean_bb, per, errors, forfeits, bank)


# ------------------------------------------------------------------------- search

def _validated_space(space: dict) -> dict:
    """{name: [HeuristicParams-valid values]}; ValueError for an unknown name, an empty
    list or a value the HeuristicParams constructor rejects."""
    names = {f.name for f in dataclasses.fields(HeuristicParams)}
    out = {}
    for k, vals in space.items():
        if k not in names:
            raise ValueError(f"unknown HeuristicParams field {k!r}")
        vals = [tuple(v) if isinstance(v, list) else v for v in vals]
        if not vals:
            raise ValueError(f"no values for {k!r}")
        for v in vals:
            HeuristicParams(**{k: v})
        out[k] = vals
    return out


def _with(p: HeuristicParams, **kw) -> HeuristicParams:
    return dataclasses.replace(p, **kw)


def _random_neighbour(base: HeuristicParams, space: dict, rng: np.random.Generator):
    names = list(space)
    p_move = min(1.0, 3.0 / len(names))
    moved = [k for k in names if rng.random() < p_move] or [names[rng.integers(len(names))]]
    kw = {}
    for k in moved:
        others = [v for v in space[k] if v != getattr(base, k)] or space[k]
        kw[k] = others[rng.integers(len(others))]
    return _with(base, **kw)


def _neighbours(p: HeuristicParams, name: str, values: list) -> list:
    """The values one step either side of p's value in the list (all values if p's value
    is not in it), as parameter sets."""
    v = getattr(p, name)
    if v in values:
        i = values.index(v)
        picks = [values[j] for j in (i - 1, i + 1) if 0 <= j < len(values)]
    else:
        picks = list(values)
    return [_with(p, **{name: x}) for x in picks]


Evaluator = Callable[[list, int], list]     # (candidates, batch index) -> [Score]


def search(space: dict, base: HeuristicParams, budget: int, evaluate: Evaluator,
           rng: np.random.Generator, random_frac: float = 0.5, min_gain: float = 0.0,
           on_step: Callable[[Step], None] | None = None) -> tuple:
    """(best HeuristicParams, [Step, ...]). evaluate(candidates, batch) scores a batch on
    common seeds and is called at most `budget` candidates in total."""
    space = _validated_space(space)
    steps: list = []
    used = 0
    batch_no = 0

    def run(cands: list, phase: str) -> list:
        nonlocal used, batch_no
        scores = evaluate(cands, batch_no)
        for p, s in zip(cands, scores):
            st = Step(batch_no, phase, (), p, s)
            steps.append(st)
            if on_step is not None:
                on_step(st)
        used += len(cands)
        batch_no += 1
        return scores

    if budget < 1:
        return base, steps
    n_random = max(0, min(budget - 1, int(round(budget * random_frac))))
    cands = [base]
    tries = 0
    while len(cands) < 1 + n_random and tries < 50 * (n_random + 1):
        tries += 1
        c = _random_neighbour(base, space, rng)
        if c not in cands:
            cands.append(c)
    scores = run(cands, "random")
    best = max(range(len(cands)), key=lambda i: scores[i].objective)
    incumbent = cands[best]

    order = list(space)
    while budget - used >= 2:
        moved = False
        for name in order:
            room = budget - used - 1
            if room < 1:
                break
            challengers = _neighbours(incumbent, name, space[name])[:room]
            if not challengers:
                continue
            batch = [incumbent] + challengers
            scores = run(batch, f"coord:{name}")
            j = max(range(1, len(batch)), key=lambda i: scores[i].objective)
            if scores[j].objective > scores[0].objective + min_gain:
                incumbent, moved = batch[j], True
        if not moved:
            break
    return incumbent, steps


# --------------------------------------------------------------------------- tune

def train_opponents(which="train") -> dict:
    """{name: ZOO factory} for the TRAIN split ("train") or for a list or tuple of TRAIN
    member names. Anything else raises ValueError: "holdout", "all", a FIELD or unknown
    name, a bare name, and a dict of factories (no held-out bot can come in under a TRAIN
    name)."""
    from sim.opponents import TRAIN, ZOO
    if isinstance(which, str):
        if which != "train":
            raise ValueError(f"the tuner plays TRAIN opponents only, not {which!r}")
        return {k: v for k, v in ZOO.items() if k in TRAIN}
    if not isinstance(which, (list, tuple)) or not which \
            or not all(isinstance(n, str) for n in which):
        raise ValueError(f"opponents must be 'train' or a list of TRAIN names, got {which!r}")
    bad = [n for n in which if n not in TRAIN]
    if bad:
        raise ValueError(f"the tuner plays TRAIN opponents only; not in TRAIN: {bad}")
    return {n: ZOO[n] for n in which}


def _seed_groups(seeds: list, batch_seeds: int | None):
    """Batch k's seeds: all of them, or the k-th group of batch_seeds (cycling)."""
    seeds = [str(s) for s in seeds]
    if not batch_seeds or batch_seeds >= len(seeds):
        return lambda k: tuple(seeds)
    n = len(seeds)
    return lambda k: tuple(seeds[(k * batch_seeds + i) % n] for i in range(batch_seeds))


def tune(param_space: dict, opponents="train", seeds: list | None = None, deals: int = 500,
         budget: int = 80, workers: int = 6, time_scale: float = 2.5,
         base: HeuristicParams | None = None, batch_seeds: int | None = None,
         random_frac: float = 0.5, min_gain: float = 0.0, rng_seed: int = 0,
         on_step: Callable[[Step], None] | None = None) -> HeuristicParams:
    """The best HeuristicParams found for PolicyBot(params) (module docstring).

    opponents: "train" or a list of TRAIN names (train_opponents; nothing else is
    accepted). seeds: the seed pool (default t1-t4). time_scale: the sim's emulated CPU
    slowdown (ruling R30b).
    """
    opps = train_opponents(opponents)
    group = _seed_groups(seeds or [f"t{i}" for i in range(1, 5)], batch_seeds)
    base = base if base is not None else HeuristicParams()
    pool = None
    if workers > 1:
        pool = multiprocessing.get_context("spawn").Pool(workers)

    def evaluate(cands: list, batch: int) -> list:
        ss = group(batch)
        jobs = [(i, functools.partial(PolicyBot, p), o, f, s, deals, time_scale)
                for s in ss for o, f in opps.items() for i, p in enumerate(cands)]
        results = (pool.imap_unordered(_run_job, jobs, chunksize=1) if pool is not None
                   else map(_run_job, jobs))
        by = [{o: [] for o in opps} for _ in cands]
        for i, o, _s, m in results:
            by[i][o].append(m)
        return [score_matches(b) for b in by]

    def note(st: Step) -> None:
        if on_step is not None:
            on_step(dataclasses.replace(st, seeds=group(st.batch)))

    try:
        best, _ = search(param_space, base, budget, evaluate, np.random.default_rng(rng_seed),
                         random_frac, min_gain, note)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return best


# ---------------------------------------------------------------------------- CLI

def params_diff(p: HeuristicParams, ref: HeuristicParams | None = None) -> dict:
    """The fields of p that differ from ref (default HeuristicParams())."""
    ref = ref if ref is not None else HeuristicParams()
    return {f.name: getattr(p, f.name) for f in dataclasses.fields(p)
            if getattr(p, f.name) != getattr(ref, f.name)}


def _step_record(st: Step) -> dict:
    s = st.score
    return {"batch": st.batch, "phase": st.phase, "seeds": list(st.seeds),
            "params": params_diff(st.params), "objective": s.objective, "points": s.points,
            "match_points": s.match_points, "mean_bb": s.mean_bb,
            "per_opponent": {o: list(v) for o, v in s.per_opponent.items()},
            "errors": s.errors, "forfeits": s.forfeits, "max_bank": s.max_bank,
            "t": time.time()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sim.tune")
    ap.add_argument("--budget", type=int, default=80, help="candidate evaluations in all")
    ap.add_argument("--seeds", type=int, default=4, help="size of the seed pool")
    ap.add_argument("--seed-prefix", default="t")
    ap.add_argument("--batch-seeds", type=int, default=0,
                    help="seeds per batch, rotating through the pool (0: all of them)")
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--opponents", default="train",
                    help="train, or comma-separated TRAIN names (nothing held out)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--time-scale", type=float, default=2.5)
    ap.add_argument("--random-frac", type=float, default=0.5)
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", help="append one JSON line per evaluation here")
    args = ap.parse_args(argv)
    opponents = "train" if args.opponents == "train" else args.opponents.split(",")
    try:
        train_opponents(opponents)
    except ValueError as e:
        ap.error(str(e))
    seeds = [f"{args.seed_prefix}{i}" for i in range(1, args.seeds + 1)]
    t0 = time.time()
    out = open(args.out, "a") if args.out else None

    def on_step(st: Step) -> None:
        s = st.score
        print(f"[{time.time() - t0:6.0f}s b{st.batch:02d} {st.phase:22s}] obj {s.objective:.4f} "
              f"pts {s.points:.3f} raw {s.match_points:.3f} bb {s.mean_bb:+.3f} "
              f"err {s.errors} bank {s.max_bank:.2f}  {json.dumps(params_diff(st.params))}",
              flush=True)
        if out is not None:
            out.write(json.dumps(_step_record(st)) + "\n")
            out.flush()

    try:
        best = tune(default_space(), opponents, seeds, args.deals, args.budget,
                    args.workers, args.time_scale, batch_seeds=args.batch_seeds or None,
                    random_frac=args.random_frac, min_gain=args.min_gain,
                    rng_seed=args.rng_seed, on_step=on_step)
    finally:
        if out is not None:
            out.close()
    print(f"best (vs defaults): {json.dumps(params_diff(best))}  "
          f"({time.time() - t0:.0f} s, load {os.getloadavg()[0]:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
