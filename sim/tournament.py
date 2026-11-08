"""Round robin, focus evaluation and paired comparison on top of sim.match.

CLI (run from the repo root with .venv/bin/python):
  -m sim.tournament --candidate agent --opponents train --seeds 3 --deals 300
  -m sim.tournament --entrants zoo --seeds 1 --deals 300 --report docs/zoo_report.md
  --time-scale 2.5 emulates a CPU 2.5x slower than this one for every bot (sim.match).
"""

from __future__ import annotations

import argparse
import itertools
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from sim.match import (REPO_ROOT, BotFactory, MatchStats,  # noqa: E402
                       bot_file_factory, play_match)

BIG_BLIND = 2
Z = 1.96


class Row(NamedTuple):
    name: str
    points: float
    mean_bb: float
    losses: int
    forfeits: int
    max_bank: float


@dataclass
class Standings:
    rows: list
    # Per-entrant aggregates: actions, faced, faced_fold, pre, pre_fold,
    # errors, bank_exhausted_halves, halves.
    detail: dict = field(default_factory=dict)


@dataclass
class FocusReport:
    matches: dict            # opponent -> {seed: MatchStats}
    expected_points: float
    mean_bb: float


# ---------------------------------------------------------------- execution

def _run_job(job):
    name_a, fa, name_b, fb, seed, deals, time_scale = job
    return name_a, name_b, seed, play_match(fa, fb, seed, deals, time_scale=time_scale)


def _run_jobs(jobs: list, workers: int) -> list:
    if workers <= 1 or len(jobs) <= 1:
        return [_run_job(j) for j in jobs]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(min(workers, len(jobs))) as pool:
        # chunksize 1: match durations vary hugely (slow bots).
        return list(pool.imap(_run_job, jobs, chunksize=1))


# --------------------------------------------------------------- round robin

def run_round_robin(entrants: dict, seeds: list, deals: int,
                    workers: int = os.cpu_count() or 1, time_scale: float = 1.0) -> Standings:
    names = list(entrants)
    jobs = [(a, entrants[a], b, entrants[b], str(s), deals, time_scale)
            for a, b in itertools.combinations(names, 2) for s in seeds]
    results = _run_jobs(jobs, workers)

    pts = {n: 0.0 for n in names}
    bbs = {n: [] for n in names}
    losses = {n: 0 for n in names}
    forf = {n: 0 for n in names}
    bank = {n: 0.0 for n in names}
    detail = {n: {"actions": {}, "faced": 0, "faced_fold": 0, "pre": 0, "pre_fold": 0,
                  "errors": 0, "bank_exhausted_halves": 0, "halves": 0} for n in names}

    def absorb(n, h, side):
        d = detail[n]
        acts, ctx = (h.actions_a, h.ctx_a) if side == "a" else (h.actions_b, h.ctx_b)
        for k, v in acts.items():
            d["actions"][k] = d["actions"].get(k, 0) + v
        for k in ("faced", "faced_fold", "pre", "pre_fold"):
            d[k] += ctx.get(k, 0)
        if side == "a":
            d["errors"] += h.errors_a
            d["bank_exhausted_halves"] += h.bank_exhausted_a
            forf[n] += h.forfeit_a
            bank[n] = max(bank[n], h.bank_a)
        else:
            d["errors"] += h.errors_b
            d["bank_exhausted_halves"] += h.bank_exhausted_b
            forf[n] += h.forfeit_b
            bank[n] = max(bank[n], h.bank_b)
        d["halves"] += 1

    for a, b, _seed, m in results:
        both = m.verdict == "BOTH_FORFEIT"
        pts[a] += m.points_a
        pts[b] += 0.0 if both else 1.0 - m.points_a
        bbs[a].append(m.mean_bb)
        bbs[b].append(-m.mean_bb)
        losses[a] += m.verdict == "B"
        losses[b] += m.verdict == "A"
        for h in m.halves:
            absorb(a, h, "a")
            absorb(b, h, "b")

    rows = [Row(n, pts[n], float(np.mean(bbs[n])) if bbs[n] else 0.0,
                losses[n], forf[n], bank[n]) for n in names]
    rows.sort(key=lambda r: (-r.points, -r.mean_bb, r.name))
    return Standings(rows, detail)


# --------------------------------------------------------------- focus eval

def focus_eval(candidate: BotFactory, opponents: dict, seeds: list, deals: int,
               workers: int = os.cpu_count() or 1, time_scale: float = 1.0) -> FocusReport:
    jobs = [("cand", candidate, o, f, str(s), deals, time_scale)
            for o, f in opponents.items() for s in seeds]
    matches: dict = {o: {} for o in opponents}
    for _a, o, s, m in _run_jobs(jobs, workers):
        matches[o][s] = m
    allm = [m for d in matches.values() for m in d.values()]
    return FocusReport(matches,
                       float(np.mean([m.points_a for m in allm])) if allm else 0.0,
                       float(np.mean([m.mean_bb for m in allm])) if allm else 0.0)


# ------------------------------------------------------------ paired compare

def _per_deal_bb(m: MatchStats) -> np.ndarray:
    h1, h2 = m.halves
    return (np.asarray(h1.nets_a, dtype=float) + np.asarray(h2.nets_a, dtype=float)) / BIG_BLIND


def mean_ci(d) -> tuple:
    """(ci_low, mean, ci_high) for the mean of d: mean +- Z * sd / sqrt(N).

    sd is the sample standard deviation (ddof=1), 0 for a single value; N = len(d).
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = Z * sd / math.sqrt(n)
    return mean - half, mean, mean + half


def paired_diff(cand_a: BotFactory, cand_b: BotFactory, opponents: dict, seeds: list,
                deals: int, workers: int = 1, time_scale: float = 1.0) -> tuple:
    """Return (ci_low, mean, ci_high) of the per-deal bb difference A - B.

    Same opponents and seeds for both; all deals are pooled and the CI is
    mean_ci of the pooled per-deal differences, so N is the total pooled deals.
    """
    jobs = []
    for o, f in opponents.items():
        for s in seeds:
            jobs.append(("A", cand_a, o, f, str(s), deals, time_scale))
            jobs.append(("B", cand_b, o, f, str(s), deals, time_scale))
    res = {}
    for who, o, s, m in _run_jobs(jobs, workers):
        res[(who, o, str(s))] = _per_deal_bb(m)
    diffs = [res[("A", o, str(s))] - res[("B", o, str(s))] for o in opponents for s in seeds]
    return mean_ci(np.concatenate(diffs))


# ----------------------------------------------------------------------- CLI

def resolve_bot(spec: str) -> BotFactory:
    """'agent' -> agent.py, a ZOO name, or a path to a bot file."""
    from sim.opponents import ZOO
    if spec == "agent":
        return bot_file_factory("agent.py")
    if spec in ZOO:
        return ZOO[spec]
    p = Path(spec)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if p.is_file():
        return bot_file_factory(str(p))
    raise ValueError(f"cannot resolve bot {spec!r}: not 'agent', a ZOO name or a bot file")


OPPONENT_SPLITS = ("train", "holdout", "field", "all")     # "all" includes FIELD


def select_opponents(which: str) -> dict:
    from sim.opponents import FIELD, HOLDOUT, TRAIN, ZOO
    keep = {"train": TRAIN, "holdout": HOLDOUT, "field": FIELD, "all": set(ZOO)}[which]
    return {k: v for k, v in ZOO.items() if k in keep}


def _rate(num, den):
    return num / den if den else float("nan")


H_FACED_FOLD = 0.10    # H2 threshold: folds < 10% of decisions facing a bet
H_PRE_FOLD = 0.40      # H4 threshold: folds > 40% of preflop decisions


def hypotheses(st: Standings) -> list:
    """(label, description, stat, names) for each stat-derived hypothesis."""
    D, rows = st.detail, st.rows
    out = {
        "H1": [r.name for r in rows if D[r.name]["bank_exhausted_halves"] > 0],
        "H2": [r.name for r in rows if D[r.name]["faced"] > 0
               and _rate(D[r.name]["faced_fold"], D[r.name]["faced"]) < H_FACED_FOLD],
        "H4": [r.name for r in rows if D[r.name]["pre"] > 0
               and _rate(D[r.name]["pre_fold"], D[r.name]["pre"]) > H_PRE_FOLD],
        "H5": [r.name for r in rows
               if D[r.name]["actions"].get("RAISE_HALF", 0) == 0
               or D[r.name]["actions"].get("RAISE_POT", 0) == 0],
        "H6": [r.name for r in rows if D[r.name]["errors"] > 0 or r.forfeits > 0],
    }
    return out


def write_report(st: Standings, path: str, seeds: list, deals: int, wall_s: float) -> None:
    D = st.detail
    L = [f"# Zoo diversity report", "",
         f"Round robin of {len(st.rows)} zoo bots, seeds {seeds}, {deals} deals per half "
         f"(both halves per match), wall time {wall_s:.0f} s.", "",
         "## Standings", "",
         "| Bot | Points | mean bb/deal | Losses | Forfeit halves | Max bank (s) |",
         "|---|---:|---:|---:|---:|---:|"]
    for r in st.rows:
        L.append(f"| {r.name} | {r.points:.1f} | {r.mean_bb:+.3f} | {r.losses} | {r.forfeits} | {r.max_bank:.3f} |")
    L += ["", "## Behaviour", "",
          "Rates pool every decision the bot played across all its halves. "
          "Fold-facing = folds / decisions with to_call > 0; Pre fold = folds / preflop decisions.", "",
          "| Bot | Fold % | Call % | Half % | Pot % | Fold-facing % | Pre fold % | Errors | Bank-exhausted halves |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in st.rows:
        d = D[r.name]
        a = d["actions"]
        tot = sum(a.values()) or 1
        L.append("| {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {} | {}/{} |".format(
            r.name, 100 * a.get("FOLD", 0) / tot, 100 * a.get("CHECK_CALL", 0) / tot,
            100 * a.get("RAISE_HALF", 0) / tot, 100 * a.get("RAISE_POT", 0) / tot,
            100 * _rate(d["faced_fold"], d["faced"]), 100 * _rate(d["pre_fold"], d["pre"]),
            d["errors"], d["bank_exhausted_halves"], d["halves"]))
    h = hypotheses(st)
    n = len(st.rows)

    def fmt(k):
        return f"{len(h[k])}/{n}: " + (", ".join(h[k]) if h[k] else "none")

    L += ["", "## Hypotheses H1-H6", "",
          "Counts of zoo bots (out of %d) showing each behaviour, with the stat used." % n, "",
          f"- **H1, exhausts its time bank**: {fmt('H1')}. Stat: a half where the bank hit zero "
          "(`bank_exhausted` / timed_out flag). Max bank use per bot is in Standings.",
          f"- **H2, rarely folds facing bets**: {fmt('H2')}. Stat: fold-facing rate < {H_FACED_FOLD:.0%}.",
          "- **H3, category thresholds**: n/a from stats. From archetype code: `threshold_cat` "
          "(hand-category thresholds); the llm_* bots were not audited for this.",
          f"- **H4, folds a lot preflop**: {fmt('H4')}. Stat: preflop fold rate > {H_PRE_FOLD:.0%}.",
          f"- **H5, never uses RAISE_HALF or never uses RAISE_POT**: {fmt('H5')}. "
          "Stat: zero count of either action over all its halves.",
          f"- **H6, errors or forfeits**: {fmt('H6')}. Stat: error count > 0 or forfeited halves > 0.", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L))


def _seed_list(args) -> list:
    return [f"{args.seed_prefix}{i}" for i in range(1, args.seeds + 1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sim.tournament")
    ap.add_argument("--candidate")
    ap.add_argument("--opponents", choices=OPPONENT_SPLITS, default="train")
    ap.add_argument("--entrants", choices=["zoo"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-prefix", default="s")
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="emulate a CPU this many times slower than this one (both bots)")
    ap.add_argument("--report")
    args = ap.parse_args(argv)
    seeds = _seed_list(args)
    t0 = time.time()
    if args.entrants == "zoo":
        from sim.opponents import ZOO
        st = run_round_robin(dict(ZOO), seeds, args.deals, args.workers, args.time_scale)
        for r in st.rows:
            print(f"{r.name:22s} pts={r.points:6.1f} bb={r.mean_bb:+.3f} L={r.losses} F={r.forfeits} bank={r.max_bank:.3f}")
        if args.report:
            write_report(st, args.report, seeds, args.deals, time.time() - t0)
            print("wrote", args.report)
        return 0
    if not args.candidate:
        ap.error("need --candidate or --entrants zoo")
    opps = select_opponents(args.opponents)
    rep = focus_eval(resolve_bot(args.candidate), opps, seeds, args.deals, args.workers,
                     args.time_scale)
    print(f"{'opponent':22s} {'points':>7s} {'mean_bb':>9s}")
    for o, d in rep.matches.items():
        ms = list(d.values())
        print(f"{o:22s} {np.mean([m.points_a for m in ms]):7.2f} {np.mean([m.mean_bb for m in ms]):+9.3f}")
    print(f"expected_points={rep.expected_points:.3f} mean_bb={rep.mean_bb:+.3f} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
