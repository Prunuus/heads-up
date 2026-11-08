"""The v3 gate (Task 20, rulings R39, R42, R44): v3 (agent.py with the CFR blueprint) vs v2.

v3 is agent.py itself, loaded the way the tournament starts a half (it plays the blueprint
in data/blueprint.npz when that file loads, else the heuristic). v2 is
PolicyBot(use_blueprint=False): v2's heuristic base and exploit layer, decision for
decision. The purify variants are PolicyBot(purify=p).

Run from the repo root on free CPUs, after the final export:
    .venv/bin/python -m sim.gate_v3 --out train_data/gate_v3 --cpus 0-7
Sections (--sections, default abcde; c needs no runs of its own):
  a  head-to-head v3 vs v2: --h2h-seeds (5) x --deals (1000) per half, both halves of each
     duplicate match; the CI is over the pooled per-deal bb of v3.
  b  expected points vs HOLDOUT and vs FIELD for v3 and v2 at time scales --scales
     (1.0, 2.5): --focus-seeds (3) x --deals. Quiet protocol R39: --workers <= 2 pinned to
     --cpus while nothing else runs there; the opponents' max bank per half is reported.
     Also the paired v3 - v2 difference on the same matches.
  c  errors, forfeits and our max bank per half over every match of a and b.
  d  LBR(v3) vs LBR(v2): --lbr-particles (2000, T21 M5), --lbr-seeds (16) x --lbr-deals
     (250) per half, --lbr-workers (LBR has no clock: use the free cores). A ModelCheckError
     marks the value unreliable (reported, never used silently). LBR values are lower bounds.
  e  purify sweep {0, 0.05, 0.1, 0.2} (--purify): paired v3(purify) - v2 on TRAIN,
     --sweep-seeds (2) x --sweep-deals (1000). v2's matches are played once and shared.
     --extra v3_post v3_nofeat adds the tuning variants (blueprint postflop only; no
     maniac/station re-runs on a blueprint base) to the same table.
Every job's result is a JSON file under --out, written as soon as it finishes: kill the run
any time and rerun the same command to resume (finished jobs are skipped; a change of deals
or seeds makes new jobs). --summary-only prints the table from what is cached. One --out
holds one blueprint's results: a run with another data/blueprint.npz is refused (exit 3).
A provisional sanity run: --sections a --h2h-seeds 2 --deals 500 --workers 2.
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

from sim.match import REPO_ROOT, bot_file_factory, play_match
from sim.tournament import mean_ci

BIG_BLIND = 2
SECTIONS = "abcde"
PURIFY_SWEEP = (0.0, 0.05, 0.1, 0.2)


# ------------------------------------------------------------------------------- bots

def _policy_bot(**kw):
    """A PolicyBot from the caller's bot package (picklable through functools.partial)."""
    from bot.policy import PolicyBot
    return PolicyBot(**kw)


def bot(label: str):
    """Picklable factory for a gate label: v3, v2, v3_p<purify>, or a tuning variant:
    v3_post (the blueprint postflop only, the heuristic preflop) or v3_nofeat (no
    maniac/station re-runs on a blueprint base)."""
    if label == "v3":
        return bot_file_factory("agent.py")
    if label == "v2":
        return functools.partial(_policy_bot, use_blueprint=False)
    if label == "v3_post":
        from game import Street
        return functools.partial(_policy_bot,
                                 blueprint_streets=(Street.FLOP, Street.TURN, Street.RIVER))
    if label == "v3_nofeat":
        return functools.partial(_policy_bot, exploit_features=False)
    if label.startswith("v3_p"):
        return functools.partial(_policy_bot, purify=float(label[4:]))
    raise ValueError(f"unknown gate bot {label!r}")


def opponent(name: str):
    from sim.opponents import ZOO
    return ZOO[name]


def splits() -> dict:
    from sim.opponents import FIELD, HOLDOUT, TRAIN
    return {"holdout": sorted(HOLDOUT), "field": sorted(FIELD), "train": sorted(TRAIN)}


# ------------------------------------------------------------------------------- jobs

def _key(section, a, b, seed, deals, scale) -> str:
    return f"{section}__{a}__{b}__{seed}__{deals}__ts{scale:g}"


def _play(job: tuple) -> tuple:
    """One duplicate match; a JSON-able record of everything the summary needs."""
    key, a, b, b_is_bot, seed, deals, scale = job
    fb = bot(b) if b_is_bot else opponent(b)
    t0 = time.time()
    m = play_match(bot(a), fb, seed, deals, time_scale=scale)
    h1, h2 = m.halves
    rec = {
        "a": a, "b": b, "seed": seed, "deals": deals, "time_scale": scale,
        "points_a": m.points_a, "mean_bb": m.mean_bb, "verdict": m.verdict,
        "per_deal_bb": ((np.asarray(h1.nets_a, float) + np.asarray(h2.nets_a, float))
                        / BIG_BLIND).tolist(),
        "errors_a": [h.errors_a for h in m.halves], "errors_b": [h.errors_b for h in m.halves],
        "forfeit_a": [h.forfeit_a for h in m.halves], "forfeit_b": [h.forfeit_b for h in m.halves],
        "bank_a": [h.bank_a for h in m.halves], "bank_b": [h.bank_b for h in m.halves],
        "max_call_a": [h.max_call_a for h in m.halves],
        "wall_s": time.time() - t0, "load": os.getloadavg()[0],
    }
    return key, rec


def _run(jobs: list, out: Path, workers: int) -> dict:
    """Run the jobs not cached under out; return every job's record (cached or new)."""
    recs, todo = {}, []
    for j in jobs:
        f = out / f"{j[0]}.json"
        if f.is_file():
            recs[j[0]] = json.loads(f.read_text())
        else:
            todo.append(j)
    if todo:
        print(f"  {len(todo)} job(s) to run, {len(jobs) - len(todo)} cached, "
              f"{workers} worker(s), load {os.getloadavg()[0]:.1f}", flush=True)

    def save(key, rec):
        _write(out / f"{key}.json", rec)
        recs[key] = rec
        print(f"  done {key}: {rec['mean_bb']:+.3f} bb/deal, {rec['wall_s']:.0f} s, "
              f"load {rec['load']:.1f}", flush=True)

    if workers <= 1 or len(todo) <= 1:
        for j in todo:
            save(*_play(j))
    elif todo:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(min(workers, len(todo))) as pool:
            for key, rec in pool.imap_unordered(_play, todo, chunksize=1):
                save(key, rec)
    return recs


def jobs_a(args) -> list:
    return [(_key("a", "v3", "v2", s, args.deals, 1.0), "v3", "v2", True, s, args.deals, 1.0)
            for s in [f"g3h{i}" for i in range(1, args.h2h_seeds + 1)]]


def jobs_b(args) -> list:
    opps = splits()["holdout"] + splits()["field"]
    seeds = [f"g3f{i}" for i in range(1, args.focus_seeds + 1)]
    return [(_key("b", who, o, s, args.deals, ts), who, o, False, s, args.deals, ts)
            for ts in args.scales for who in ("v3", "v2") for o in opps for s in seeds]


def jobs_e(args) -> list:
    seeds = [f"g3p{i}" for i in range(1, args.sweep_seeds + 1)]
    labels = ["v2"] + [f"v3_p{p:g}" for p in args.purify] + list(args.extra)
    return [(_key("e", who, o, s, args.sweep_deals, 1.0), who, o, False, s, args.sweep_deals, 1.0)
            for who in labels for o in splits()["train"] for s in seeds]


def _jsonable(x):
    """json.dumps default: numpy scalars and arrays as plain numbers and lists."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON-able: {type(x).__name__}")


def _write(path: Path, rec: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(rec, default=_jsonable))
    os.replace(tmp, path)


def _lbr_chunks(args) -> list:
    """Seed groups of LBR jobs: each group fills --lbr-workers (two halves per seed) and is
    cached on its own, so an interrupted LBR run resumes at the next group."""
    seeds = [f"g3l{i}" for i in range(1, args.lbr_seeds + 1)]
    k = max(1, args.lbr_workers // 2)
    return [seeds[i:i + k] for i in range(0, len(seeds), k)]


def _lbr_file(out: Path, who: str, args, chunk: list) -> Path:
    return out / f"d__{who}__{args.lbr_particles}p__{args.lbr_deals}__{chunk[0]}-{chunk[-1]}.json"


def _lbr_combine(who: str, parts: list) -> dict:
    """One target's LBR value from its seed groups (the per-deal values pooled as lbr_eval
    pools them, the belief stats summed, model_problems re-checked on the whole run)."""
    from sim.opponents.lbr import MAX_INCONSISTENT, model_problems
    per_deal = np.concatenate([np.asarray(p["per_deal_bb"], float) for p in parts])
    stats: dict = {}
    for p in parts:
        for k, v in p["stats"].items():
            stats[k] = stats.get(k, 0) + v
    errors = sum(p["target_errors"] for p in parts)
    problems = model_problems(stats, errors, MAX_INCONSISTENT)
    lo, mean, hi = mean_ci(per_deal)
    decisions = sum(p["decisions"] for p in parts)
    return {"target": who, "mean_bb": mean, "ci_low": lo, "ci_high": hi, "deals": len(per_deal),
            "decisions": decisions, "wall_s": sum(p["wall_s"] for p in parts),
            "ms_per_decision": 1e3 * sum(p["lbr_seconds"] for p in parts) / max(decisions, 1),
            "target_errors": errors, "model_problems": problems,
            "model_check_failed": bool(problems), "stats": stats,
            "per_deal_bb": per_deal.tolist()}


def run_lbr(args, out: Path, run: bool = True) -> dict:
    """LBR vs v3 and v2 on common seeds, cached per seed group. ModelCheckError is applied
    to each target's whole run (lbr_eval's check, on the pooled stats)."""
    from sim.opponents.lbr import lbr_eval
    res = {}
    for who in ("v3", "v2"):
        parts = []
        for chunk in _lbr_chunks(args):
            f = _lbr_file(out, who, args, chunk)
            if f.is_file():
                parts.append(json.loads(f.read_text()))
                continue
            if not run:
                continue
            print(f"  LBR vs {who}: seeds {chunk[0]}..{chunk[-1]} x {args.lbr_deals} deals, "
                  f"{args.lbr_particles} particles, {args.lbr_workers} worker(s), "
                  f"load {os.getloadavg()[0]:.1f}", flush=True)
            r = lbr_eval(bot(who), chunk, args.lbr_deals, args.lbr_workers,
                         particles=args.lbr_particles, check_model=False)
            rec = {"seeds": chunk, "per_deal_bb": np.asarray(r.per_deal, float).tolist(),
                   "stats": r.stats, "target_errors": r.target_errors,
                   "decisions": r.decisions, "lbr_seconds": r.lbr_seconds,
                   "wall_s": r.wall_seconds}
            _write(f, rec)
            parts.append(json.loads(f.read_text()))
            print(f"  LBR vs {who} {chunk[0]}..{chunk[-1]}: {r.mean_bb:+.3f} bb/deal, "
                  f"{r.wall_seconds:.0f} s", flush=True)
        if parts and len(parts) == len(_lbr_chunks(args)):
            res[who] = _lbr_combine(who, parts)
    return res


# ---------------------------------------------------------------------------- summary

def _ci(d) -> str:
    lo, mean, hi = mean_ci(d)
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _pooled(recs) -> np.ndarray:
    return np.concatenate([np.asarray(r["per_deal_bb"]) for r in recs]) if recs else np.zeros(0)


def summarize(args, ra: dict, rb: dict, rd: dict, re_: dict) -> list:
    L = []
    if ra:
        rs = sorted(ra.values(), key=lambda r: r["seed"])
        L += ["(a) v3 vs v2 head-to-head, v3's bb/deal (paired duplicate halves)",
              f"    pooled {len(rs)} seeds x {rs[0]['deals']} deals: {_ci(_pooled(rs))}   "
              f"verdicts {' '.join(r['verdict'] for r in rs)}",
              "    per seed: " + "  ".join(f"{r['seed']} {r['mean_bb']:+.2f}" for r in rs), ""]
    if rb:
        sp = splits()
        L += ["(b) expected points vs HOLDOUT / FIELD (opp bank = opponent's max bank per half, s)",
              f"    {'scale':>5} {'split':8} {'v3 pts':>7} {'v2 pts':>7} {'v3 bb':>8} {'v2 bb':>8}"
              f"   v3 - v2 paired bb/deal [95% CI]"]
        for ts in args.scales:
            for split in ("holdout", "field"):
                pick = {who: sorted((r for r in rb.values() if r["a"] == who and r["b"] in sp[split]
                                     and r["time_scale"] == ts), key=lambda r: (r["b"], r["seed"]))
                        for who in ("v3", "v2")}
                if not pick["v3"] or not pick["v2"]:
                    continue
                pts = {w: np.mean([r["points_a"] for r in v]) for w, v in pick.items()}
                bb = {w: np.mean([r["mean_bb"] for r in v]) for w, v in pick.items()}
                pair = {(r["b"], r["seed"]): r for r in pick["v2"]}
                diffs = [np.asarray(r["per_deal_bb"]) - np.asarray(pair[(r["b"], r["seed"])]["per_deal_bb"])
                         for r in pick["v3"] if (r["b"], r["seed"]) in pair]
                L.append(f"    {ts:5g} {split:8} {pts['v3']:7.3f} {pts['v2']:7.3f} {bb['v3']:+8.3f} "
                         f"{bb['v2']:+8.3f}   {_ci(np.concatenate(diffs)) if diffs else 'n/a'}")
            for o in sp["holdout"] + sp["field"]:
                row = []
                for who in ("v3", "v2"):
                    rs = [r for r in rb.values() if r["a"] == who and r["b"] == o and r["time_scale"] == ts]
                    if rs:
                        row.append(f"{who} {np.mean([r['points_a'] for r in rs]):.2f} pts "
                                   f"{np.mean([r['mean_bb'] for r in rs]):+6.2f} bb "
                                   f"(opp bank {max(max(r['bank_b']) for r in rs):.2f})")
                if row:
                    L.append(f"      ts {ts:g} {o:18s} " + " | ".join(row))
        L.append("")
    allm = list(ra.values()) + list(rb.values())
    if allm:
        mine = [r for r in allm if r["a"] == "v3"]
        v3_err = sum(sum(r["errors_a"]) for r in allm if r["a"] == "v3")
        v3_ff = sum(sum(r["forfeit_a"]) for r in allm if r["a"] == "v3")
        v2_err = sum(sum(r["errors_b"]) for r in ra.values()) + sum(
            sum(r["errors_a"]) for r in rb.values() if r["a"] == "v2")
        v3_bank = max(max(r["bank_a"]) for r in mine)
        v3_call = max(max(r["max_call_a"]) for r in mine)
        L += ["(c) errors and bank (every match of a and b)",
              f"    v3: {v3_err} errors, {v3_ff} forfeited halves, max bank per half {v3_bank:.3f} s, "
              f"longest request {1e3 * v3_call:.1f} ms   |   v2: {v2_err} errors",
              f"    v3 bank per half by scale: " + "  ".join(
                  f"ts {ts:g}: {max([max(r['bank_a']) for r in mine if r['time_scale'] == ts] or [0]):.3f} s"
                  for ts in sorted({r['time_scale'] for r in mine})), ""]
    if rd:
        L.append("(d) LBR exploitability, LBR's bb/deal (lower = less exploitable; lower bounds)")
        for who in ("v3", "v2"):
            r = rd.get(who)
            if r:
                flag = "  MODEL CHECK FAILED: " + "; ".join(r["model_problems"]) if r["model_check_failed"] else ""
                L.append(f"    LBR({who}) = {r['mean_bb']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                         f"  deals {r['deals']}  {r['ms_per_decision']:.0f} ms/decision{flag}")
        if "v3" in rd and "v2" in rd and len(rd["v3"]["per_deal_bb"]) == len(rd["v2"]["per_deal_bb"]):
            d = np.asarray(rd["v3"]["per_deal_bb"]) - np.asarray(rd["v2"]["per_deal_bb"])
            L.append(f"    LBR(v3) - LBR(v2), paired on common deals: {_ci(d)}")
        L.append("")
    if re_:
        L.append("(e) purify sweep on TRAIN: v3(purify) - v2, paired bb/deal [95% CI]")
        v2 = {(r["b"], r["seed"]): r for r in re_.values() if r["a"] == "v2"}
        for lab in [f"v3_p{p:g}" for p in args.purify] + list(args.extra):
            rs = [r for r in re_.values() if r["a"] == lab and (r["b"], r["seed"]) in v2]
            if rs:
                d = np.concatenate([np.asarray(r["per_deal_bb"]) - np.asarray(v2[(r["b"], r["seed"])]["per_deal_bb"])
                                    for r in rs])
                pts = np.mean([r["points_a"] for r in rs])
                errs = sum(sum(r["errors_a"]) for r in rs)
                L.append(f"    {lab:9s} {_ci(d)}   points {pts:.3f} (v2 "
                         f"{np.mean([v['points_a'] for v in v2.values()]):.3f})   errors {errs}"
                         f"   ({len(rs)} matches)")
        L.append("")
    return L


# -------------------------------------------------------------------------------- CLI

def _same_blueprint(out: Path, bp) -> bool:
    """The cache under out holds results of one blueprint only (job keys do not name it):
    record its identity on the first run, refuse a run with another one."""
    ident = {"iterations": None if bp is None else int(bp.meta["iterations"]),
             "seed": None if bp is None else int(bp.meta["seed"]),
             "buckets_sha256": None if bp is None else str(bp.meta["buckets_sha256"])}
    f = out / "blueprint.json"
    if f.is_file():
        have = json.loads(f.read_text())
        if have != ident:
            print(f"refusing: {out} caches results of blueprint {have}, data/blueprint.npz is "
                  f"{ident}; pass a new --out")
            return False
    else:
        f.write_text(json.dumps(ident))
    return True


def _cpus(spec: str) -> set:
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sim.gate_v3", description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="train_data/gate_v3", help="cache directory (resumable)")
    ap.add_argument("--sections", default=SECTIONS)
    ap.add_argument("--cpus", help="pin this process and its workers, e.g. 0-7 (R39)")
    ap.add_argument("--workers", type=int, default=2, help="match workers (R39: <= 2)")
    ap.add_argument("--deals", type=int, default=1000, help="deals per half, sections a and b")
    ap.add_argument("--h2h-seeds", type=int, default=5)
    ap.add_argument("--focus-seeds", type=int, default=3)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.5])
    ap.add_argument("--lbr-seeds", type=int, default=16)
    ap.add_argument("--lbr-deals", type=int, default=250)
    ap.add_argument("--lbr-particles", type=int, default=2000)
    ap.add_argument("--lbr-workers", type=int, default=8)
    ap.add_argument("--sweep-seeds", type=int, default=2)
    ap.add_argument("--sweep-deals", type=int, default=1000)
    ap.add_argument("--purify", type=float, nargs="+", default=list(PURIFY_SWEEP))
    ap.add_argument("--extra", nargs="*", default=[],
                    help="more section-e variants vs v2 on TRAIN: v3_post, v3_nofeat")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)
    if args.cpus:
        os.sched_setaffinity(0, _cpus(args.cpus))
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    from bot import policy
    policy.warmup()
    w = policy.WARMUP
    bp = w.blueprint
    if not _same_blueprint(out, bp):
        return 3
    print(f"gate v3: blueprint {'iterations %s' % bp.meta['iterations'] if bp is not None else 'NONE (v3 = v2!)'}"
          f", warmup failed {w.failed}, cpus {sorted(os.sched_getaffinity(0))}, "
          f"load {os.getloadavg()[0]:.1f}, out {out}", flush=True)
    secs = set(args.sections)
    run = not args.summary_only
    t0 = time.time()
    ra = rb = rd = re_ = {}
    if "a" in secs or "c" in secs:
        print("(a) head-to-head", flush=True)
        ra = _run(jobs_a(args), out, args.workers) if run else _cached(jobs_a(args), out)
    if "b" in secs or "c" in secs:
        print("(b) HOLDOUT + FIELD", flush=True)
        rb = _run(jobs_b(args), out, args.workers) if run else _cached(jobs_b(args), out)
    if "e" in secs:
        print("(e) purify sweep", flush=True)
        re_ = _run(jobs_e(args), out, args.workers) if run else _cached(jobs_e(args), out)
    if "d" in secs:
        print("(d) LBR", flush=True)
        rd = run_lbr(args, out, run)
    lines = summarize(args, ra, rb, rd, re_)
    text = "\n".join([f"=== v3 gate summary ({time.time() - t0:.0f} s, load {os.getloadavg()[0]:.1f}) ===",
                      *lines])
    print(text)
    (out / "summary.txt").write_text(text + "\n")
    if any(r["model_check_failed"] for r in rd.values()):
        print("LBR's model of a target is unreliable (ModelCheckError): do not use that value")
        return 2
    return 0


def _cached(jobs: list, out: Path) -> dict:
    recs = {}
    for j in jobs:
        f = out / f"{j[0]}.json"
        if f.is_file():
            recs[j[0]] = json.loads(f.read_text())
    return recs


if __name__ == "__main__":
    sys.exit(main())
