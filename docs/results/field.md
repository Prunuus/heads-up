# FIELD: v1-untuned vs the agentic competitor bots (information run)

This is an information run for Task F, not a gate.
- **v1-untuned** is `bot.policy.PolicyBot()` at ff537d2, this branch's base.
- **FIELD** is the three agentic competitor bots in `sim/opponents/field/` (see the README
  there). They are held out: we test against them and never tune on them.
- **Matches.** Duplicate matches in the in-process sim (`sim.match.play_match`): seeds s1-s3,
  1,000 deals per half, both halves.
- **Paired bb/deal** is the judge's: (net in half 1 + net in half 2) / BB, per deal. Per seed,
  the CI and verdict are the judge's own (`combine_halves`). Pooled, the CI is `mean_ci` of the
  3,000 per-deal values.
- **Run.** 2026-11-07 20:01, 4 workers on CPUs 8-15 at nice 19, 57 s wall. The machine was
  heavily shared: load average 18-20 on 16 threads.

## Correction: the run below was distorted by load (read this first)

The 4-worker run below ran under heavy CPU contention: load average 18-20, plus its own 4 workers
on CPUs 8-15. That contention made all three FIELD bots' time guards degrade their own play. Its
numbers overstate v1-untuned against all three bots.

A serial rerun of the same 9 matches ran at 20:34-20:35, in one process, at load average 4-6.
Every FIELD bot was charged at most 0.13 s per half, so it played at full strength
(`serial_rerun.py` below):

| FIELD bot | v1-untuned bb/deal | 95% CI, pooled paired (3,000 deals) | s1 | s2 | s3 |
|---|---:|---|---|---|---|
| `field_equity` | **-1.677** | [-2.085, -1.270] | -2.15 [-2.89, -1.42] B | -1.83 [-2.52, -1.14] B | -1.04 [-1.73, -0.36] B |
| `field_gto` | **-1.161** | [-1.565, -0.758] | -1.29 [-2.01, -0.56] B | -1.24 [-1.90, -0.58] B | -0.96 [-1.67, -0.25] B |
| `field_exploit` | **+0.900** | [+0.269, +1.530] | +1.02 [-0.09, +2.13] CHOP | +1.98 [+0.88, +3.08] A | -0.30 [-1.37, +0.76] CHOP |

- **v1-untuned scored 2.0 points of 9:** 1 win, 2 chops and 6 losses (B is a FIELD win). The
  run below gave 6.5.
- **All 18 halves were clean.**
  - FIELD bank was 0.00-0.13 s per half, with 0 errors and 0 bank-outs.
  - Each FIELD bot raised 979-1,538 times per half.
  - The FIELD bots' own code swallowed 0 exceptions, per the `tests/test_field.py` hook. The run
    below had no such hook.
- **The hook does not change results.** On s1, serial, with and without it: v1 vs field_exploit
  scored +0.33 / -0.44, and vs field_equity -2.02 / -1.98. Both runs were at load average about
  10-19, and both were far from the run below.
- **Use these numbers, not the ones below,** until FIELD is rerun under controlled load. A
  dedicated 1-CPU container is closer to this quiet run than to the loaded one.
- **Two claims below are withdrawn:**
  - that v1's +3.9 vs field_exploit "is not an artifact of its throttle". The tier probe counted
    only its rules tier, not its load-driven 60-sample and lite tiers;
  - that "no half was played on check/fold autopilot". Error counts cannot see the bots' own
    catch-alls.

## Results (loaded run, superseded by the correction above)

| FIELD bot | v1-untuned bb/deal | 95% CI, pooled paired (3,000 deals) | s1 | s2 | s3 |
|---|---:|---|---|---|---|
| `field_equity` | **+0.219** | [-0.172, +0.610] | +0.16 [-0.54, +0.87] CHOP | +0.15 [-0.48, +0.78] CHOP | +0.34 [-0.35, +1.04] CHOP |
| `field_gto` | **-0.011** | [-0.393, +0.370] | -0.62 [-1.32, +0.09] CHOP | +0.83 [+0.38, +1.29] A | -0.25 [-1.03, +0.52] CHOP |
| `field_exploit` | **+3.927** | [+3.356, +4.499] | +3.94 [+2.93, +4.95] A | +3.51 [+2.49, +4.53] A | +4.33 [+3.39, +5.27] A |

A is a v1-untuned win. Over the 9 matches that is 4 wins, 5 chops and no losses, 6.5 points,
at this load. At low load it is 2.0 (correction above).
- **vs field_exploit:** v1-untuned won every seed, by +3.9 bb/deal. At low load it was +0.90.
- **vs field_equity:** a chop on every seed, +0.22 bb/deal. At low load it was three losses,
  -1.68.
- **vs field_gto:** a chop overall. Its one v1 win (s2) is the seed where field_gto spent the
  most bank. At low load it was three losses, -1.16.

## Bank use and errors per half

Bank is the time charged to the 8 s bank in the sim, with 0.75 ms free per request.

| FIELD bot | seed | half | FIELD bank (s) | FIELD errors | FIELD bank-out | FIELD forfeit | FIELD requests | v1 bank (s) | v1 errors | v1 bank-out | v1 max request (ms) |
|---|---|---:|---:|---:|---|---|---:|---:|---:|---|---:|
| `field_equity` | s1 | 1 | 1.95 | 0 | no | no | 2680 | 3.03 | 0 | no | 41.6 |
| `field_equity` | s1 | 2 | 1.96 | 0 | no | no | 2671 | 2.64 | 0 | no | 40.8 |
| `field_equity` | s2 | 1 | 1.80 | 0 | no | no | 2687 | 2.50 | 0 | no | 40.6 |
| `field_equity` | s2 | 2 | 2.08 | 0 | no | no | 2647 | 2.83 | 0 | no | 75.2 |
| `field_equity` | s3 | 1 | 2.18 | 0 | no | no | 2727 | 3.10 | 0 | no | 43.1 |
| `field_equity` | s3 | 2 | 2.27 | 0 | no | no | 2653 | 3.22 | 0 | no | 66.7 |
| `field_gto` | s1 | 1 | 1.99 | 0 | no | no | 2799 | 0.92 | 0 | no | 30.6 |
| `field_gto` | s1 | 2 | 3.50 | 0 | no | no | 2014 | 0.70 | 0 | no | 37.0 |
| `field_gto` | s2 | 1 | 3.97 | 0 | no | no | 1728 | 0.54 | 0 | no | 32.0 |
| `field_gto` | s2 | 2 | 4.24 | 0 | no | no | 1689 | 0.93 | 0 | no | 53.8 |
| `field_gto` | s3 | 1 | 2.15 | 0 | no | no | 2781 | 1.08 | 0 | no | 51.5 |
| `field_gto` | s3 | 2 | 2.05 | 0 | no | no | 2707 | 1.22 | 0 | no | 66.2 |
| `field_exploit` | s1 | 1 | 0.51 | 0 | no | no | 3080 | 1.13 | 0 | no | 19.4 |
| `field_exploit` | s1 | 2 | 0.88 | 0 | no | no | 3611 | 1.33 | 0 | no | 35.2 |
| `field_exploit` | s2 | 1 | 2.88 | 0 | no | no | 3413 | 1.94 | 0 | no | 35.6 |
| `field_exploit` | s2 | 2 | 0.91 | 0 | no | no | 3218 | 1.35 | 0 | no | 42.3 |
| `field_exploit` | s3 | 1 | 6.21 | 0 | no | no | 3565 | 2.45 | 0 | no | 101.7 |
| `field_exploit` | s3 | 2 | 0.44 | 0 | no | no | 3657 | 0.36 | 0 | no | 19.0 |

- **No FIELD bot ran out of bank in any half**, and none made an error or forfeited. So no result
  here is inflated by a bank-out.
  - Error counts cannot see the bots' own catch-alls, which check/call or fold on any exception.
    That was checked only in the serial rerun above, which swallowed 0 exceptions.
  - Their load-driven weaker tiers did inflate these results (correction above).
- v1-untuned made no errors and never banked out either. Its bank use peaked at 3.22 s, and its
  longest request was 101.7 ms (s3, half 1 vs field_exploit, under load).
- **FIELD bank use per half:**
  - field_equity: 1.80-2.27 s, mean 2.04 s.
  - field_gto: 1.99-4.24 s, mean 2.98 s.
  - field_exploit: 0.44-6.21 s, mean 1.97 s.

## The FIELD bots' time guards under tonight's load

No FIELD bot banked out, but they degrade their own play well before that. The cause is
machine load, not their compute. field_exploit's diagnostics (its stderr) measured its CPU at
1.8-2.8x its dev speed, and wall/CPU contention up to its 4.0 cap.

A read-only diagnostic rerun counted the decisions each bot made in its weakest tier, by the
bot's own thresholds. It used the same seeds and both halves, at 20:04-20:06 under the same load.
A wrapper read the bot's memory before each decision and never wrote to it (`probe_tiers.py`
below).

| FIELD bot, weakest tier | s1 h1 | s1 h2 | s2 h1 | s2 h2 | s3 h1 | s3 h2 | rerun, v1 per seed |
|---|---:|---:|---:|---:|---:|---:|---|
| `field_gto`, emergency mode: no sampling once its own time > 2.5 s (5 s on an unstarved CPU) | 15% | 22% | 37% | 4% | 0% | 6% | +0.47 CHOP, +0.52 CHOP, -0.83 B |
| `field_exploit`, rules only: once its own time > 38% of the bank | 0% | 0% | 0% | 0% | 0% | 0% | +3.81 A, +3.64 A, +4.29 A |

- **field_gto spent up to 37% of a half in its emergency mode.** That mode checks, calls only
  when to_call is at most a quarter of the pot, folds otherwise, and never raises. It started
  between hands 405 and 916 of a half.
  - In the main run, its three costliest halves (3.50-4.24 s) also had the fewest requests
    (1,689-2,014 vs about 2,750), which is what that fold-heavy mode looks like. s2, the seed
    where both halves look like that, is the only seed v1 won.
  - In the rerun, the seed where field_gto stayed at full strength longest (s3: 0% and 6%) was
    a v1 loss, -0.83 [-1.64, -0.01].
  - **So the ≈0 vs field_gto overstates v1.** The quiet serial rerun confirmed it: v1 lost all
    three seeds, -1.16 pooled (correction above).
- **field_exploit's loss to v1 was an artifact of its throttle.** The earlier claim that it was
  not is withdrawn.
  - The tier rerun found it never in its rules tier. But the probe did not count its 60-sample
    and lite tiers, which its load-driven projections choose.
  - At low load, v1's +3.9 became +0.90.
  - In the main run, each of its stderr lines carries its decision count. That maps each line
    to one half exactly (requests = decisions + 2). Its own time at the end of each half was s1
    1.21 / 1.33 s, s2 3.29 / 1.53 s and s3 6.39 / 0.81 s.
  - Two halves passed its 3.04 s rules cap: s3 half 1 by hand 500, and s2 half 1 after hand 750.
- **field_equity's tiers were not instrumented, but its throttle was active.** Here it was
  charged 1.80-2.27 s per half. At low load it was charged at most 0.13 s, and beat v1 on every
  seed (-1.68 pooled).

## Reproduce

Run from the repo root, with the two scripts below saved there. The deals are fixed by the seeds.
The numbers are not: `field_equity` seeds its RNG from the clock and `field_gto` from OS
entropy, and every bot's time guard reacts to machine load.

```
PYTHONPATH=. taskset -c 8-15 nice -n 19 .venv/bin/python field_info_run.py field_info.json
PYTHONPATH=. taskset -c 8-15 nice -n 19 .venv/bin/python probe_tiers.py field_gto s1 s2 s3
PYTHONPATH=. taskset -c 8-15 nice -n 19 .venv/bin/python probe_tiers.py field_exploit s1 s2 s3
PYTHONPATH=. taskset -c 8-15 nice -n 19 .venv/bin/python serial_rerun.py serial_rerun.json
PYTHONPATH=. taskset -c 8-15 nice -n 19 .venv/bin/python swallow_vs_v1.py
PYTHONPATH=. taskset -c 8-15 nice -n 19 .venv/bin/python control_no_hook.py field_exploit field_equity
```

The last three scripts belong to the correction. They import `tests/test_field.py` for its hook (`control_no_hook.py` does not), so run them from the repo root.

`--opponents field` in `sim.tournament` and `sim.heuristic_bot` selects the same three bots,
for the usual per-opponent points and mean bb/deal.

`field_info_run.py`, verbatim. It wrote the JSON that the loaded-run tables (Results, and Bank use and errors per half) were formatted from:

```python
"""Information run (Task F): v1-untuned = bot.policy.PolicyBot() vs each FIELD bot.

3 seeds x 1000 deals, duplicate (both halves). Per opponent: per-seed judge verdict and CI,
pooled paired CI (mean_ci of the 3000 per-deal paired bb), and per half each side's bank
use, errors, bank-out and forfeit flags. Run from the repo root.
"""
import json
import os
import sys
import time

import numpy as np

from bot.policy import PolicyBot
from sim.tournament import _per_deal_bb, focus_eval, mean_ci, select_opponents

SEEDS = ["s1", "s2", "s3"]
DEALS = 1000


def half_row(i, h):
    return {"half": i, "field_bank_s": h.bank_b, "field_errors": h.errors_b,
            "field_bank_out": h.bank_exhausted_b, "field_forfeit": h.forfeit_b,
            "field_calls": h.calls_b, "v1_bank_s": h.bank_a, "v1_errors": h.errors_a,
            "v1_bank_out": h.bank_exhausted_a, "v1_forfeit": h.forfeit_a,
            "v1_max_call_ms": 1e3 * h.max_call_a}


def main(out_path):
    t0, load0 = time.time(), os.getloadavg()
    rep = focus_eval(PolicyBot, select_opponents("field"), SEEDS, DEALS, workers=4)
    res = {"seeds": SEEDS, "deals": DEALS, "load_before": load0, "rows": {}}
    for opp, by_seed in rep.matches.items():
        ms = [by_seed[s] for s in SEEDS]
        lo, mid, hi = mean_ci(np.concatenate([_per_deal_bb(m) for m in ms]))
        res["rows"][opp] = {
            "pooled": {"mean_bb": mid, "ci": [lo, hi], "deals": DEALS * len(SEEDS)},
            "seeds": [{"seed": s, "mean_bb": m.mean_bb, "ci": [m.ci_low, m.ci_high],
                       "verdict": m.verdict, "halves": [half_row(i + 1, h) for i, h in enumerate(m.halves)]}
                      for s, m in zip(SEEDS, ms)]}
    res["wall_s"], res["load_after"] = time.time() - t0, os.getloadavg()
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    for opp, r in res["rows"].items():
        p = r["pooled"]
        print(f"{opp:14s} pooled {p['mean_bb']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]")
        for s in r["seeds"]:
            lo, hi = s["ci"]
            print(f"  {s['seed']}: {s['mean_bb']:+.3f} [{lo:+.3f}, {hi:+.3f}] {s['verdict']}")
            for h in s["halves"]:
                print("    half {half}: field bank {field_bank_s:.2f} s err {field_errors} "
                      "bank_out {field_bank_out} forfeit {field_forfeit} | v1 bank {v1_bank_s:.2f} s "
                      "err {v1_errors} bank_out {v1_bank_out}".format(**h))
    print(f"wall {res['wall_s']:.0f} s, load {load0[0]:.1f} -> {res['load_after'][0]:.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
```

`probe_tiers.py`, verbatim:

```python
"""Diagnostic (read-only): share of a FIELD bot's decisions made in its weakest time tier.

field_gto: its emergency mode (no sampling) once its own t_used > 2.5 * _cap_mult(...).
field_exploit: its rules-only tier once its own t_used > 0.38 * bank (hard cap in _samples).
The wrapper only reads the bot's memory before delegating; it never writes to it.
Usage (repo root): probe_tiers.py <field_gto|field_exploit> <seed> [<seed> ...]
"""
import functools
import sys

from bot.policy import PolicyBot
from sim.match import play_match
from sim.opponents import ZOO

PROBES = []


class Probe:
    def __init__(self, name):
        self.name, self.bot, self.n, self.weak, self.first_weak = name, ZOO[name](), 0, 0, None
        PROBES.append(self)

    def new_half(self, config, memory):
        self.bot.new_half(config, memory)

    def choose_action(self, obs, memory):
        used = memory.get("t_used", 0.0)
        if self.name == "field_gto":
            weak = used > 2.5 * self.bot._cap_mult(memory, obs.hand_index)
        else:
            weak = used > 0.38 * memory.get("bank", 8.0)
        self.n += 1
        self.weak += weak
        if weak and self.first_weak is None:
            self.first_weak = obs.hand_index
        return self.bot.choose_action(obs, memory)

    def on_hand_end(self, result, memory):
        self.bot.on_hand_end(result, memory)


name = sys.argv[1]
for seed in sys.argv[2:]:
    PROBES.clear()
    m = play_match(PolicyBot, functools.partial(Probe, name), seed, 1000)
    for half, (h, p) in enumerate(zip(m.halves, PROBES), 1):
        print(f"{name} {seed} half {half}: v1 {m.mean_bb:+.3f} [{m.ci_low:+.3f}, {m.ci_high:+.3f}] {m.verdict}; "
              f"bank {h.bank_b:.2f} s, requests {h.calls_b}; weakest tier {p.weak}/{p.n} decisions "
              f"({100 * p.weak / max(1, p.n):.0f}%), from hand {p.first_weak}", flush=True)
```

`serial_rerun.py`, verbatim (the correction table):

```python
"""Serial rerun of the information run (v1-untuned = PolicyBot() vs each FIELD bot, seeds s1-s3,
1000 deals per half, both halves) in ONE process, with tests/test_field.py's hook counting
exceptions swallowed in FIELD code. Writes JSON; prints one line per match."""
import importlib.util, json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
from bot.policy import PolicyBot
from sim.match import play_match
from sim.opponents import ZOO
from sim.tournament import _per_deal_bb, mean_ci
spec = importlib.util.spec_from_file_location("tf", "tests/test_field.py")
tf = importlib.util.module_from_spec(spec); spec.loader.exec_module(tf)
SEEDS = ["s1", "s2", "s3"]
res = {"load_before": os.getloadavg(), "rows": {}}
t0 = time.time()
for name in ("field_equity", "field_gto", "field_exploit"):
    ms, seeds = [], []
    for s in SEEDS:
        with tf.swallowed_exceptions() as sw:
            m = play_match(PolicyBot, ZOO[name], s, 1000)
        ms.append(m)
        halves = [{"field_bank_s": h.bank_b, "field_bank_out": h.bank_exhausted_b, "field_errors": h.errors_b,
                   "field_raises": h.actions_b.get("RAISE_HALF", 0) + h.actions_b.get("RAISE_POT", 0),
                   "field_actions": sum(h.actions_b.values()), "v1_bank_s": h.bank_a,
                   "v1_errors": h.errors_a, "v1_bank_out": h.bank_exhausted_a} for h in m.halves]
        seeds.append({"seed": s, "mean_bb": m.mean_bb, "ci": [m.ci_low, m.ci_high], "verdict": m.verdict,
                      "swallowed": len(sw), "halves": halves, "load": os.getloadavg()[0]})
        print(f"{name} {s}: v1 {m.mean_bb:+.3f} [{m.ci_low:+.3f}, {m.ci_high:+.3f}] {m.verdict}; "
              f"FIELD bank {' / '.join(f'{h['field_bank_s']:.2f}' for h in halves)} s, raises "
              f"{' / '.join(str(h['field_raises']) for h in halves)}, errors {sum(h['field_errors'] for h in halves)}, "
              f"bank-out {any(h['field_bank_out'] for h in halves)}, swallowed {len(sw)}; "
              f"load {os.getloadavg()[0]:.1f}", flush=True)
    lo, mid, hi = mean_ci(np.concatenate([_per_deal_bb(m) for m in ms]))
    res["rows"][name] = {"pooled": {"mean_bb": mid, "ci": [lo, hi]}, "seeds": seeds}
    print(f"{name} pooled: {mid:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
res["wall_s"], res["load_after"] = time.time() - t0, os.getloadavg()
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(f"wall {res['wall_s']:.0f} s, load {res['load_before'][0]:.1f} -> {res['load_after'][0]:.1f}")
```

`swallow_vs_v1.py`, verbatim (the s1 run with the hook):

```python
"""Info-run blind-spot check: each FIELD bot vs v1-untuned (PolicyBot()), seed s1, 1000 deals per
half, both halves, with tests/test_field.py's hook counting exceptions swallowed in FIELD code."""
import importlib.util, sys, time
sys.path.insert(0, ".")
from bot.policy import PolicyBot
from sim.match import play_match
from sim.opponents import FIELD, ZOO
spec = importlib.util.spec_from_file_location("tf", "tests/test_field.py")
tf = importlib.util.module_from_spec(spec); spec.loader.exec_module(tf)
for name in ("field_equity", "field_gto", "field_exploit"):
    t0 = time.time()
    with tf.swallowed_exceptions() as sw:
        m = play_match(PolicyBot, ZOO[name], "s1", 1000)
    for i, h in enumerate(m.halves, 1):
        raises = h.actions_b.get("RAISE_HALF", 0) + h.actions_b.get("RAISE_POT", 0)
        print(f"{name} s1 half {i}: FIELD raises {raises} of {sum(h.actions_b.values())} actions, "
              f"errors {h.errors_b}, bank {h.bank_b:.2f} s, bank-out {h.bank_exhausted_b}", flush=True)
    print(f"{name} s1: swallowed in FIELD code {len(sw)} {sw[:3]}; v1 {m.mean_bb:+.3f} "
          f"[{m.ci_low:+.3f}, {m.ci_high:+.3f}] {m.verdict} ({time.time() - t0:.0f} s)", flush=True)
```

`control_no_hook.py`, verbatim (the s1 run without it):

```python
"""Control for swallow_vs_v1.py: the same matches (seed s1, 1000 deals/half, serial, one process)
WITHOUT the sys.monitoring hook."""
import sys, time
sys.path.insert(0, ".")
from bot.policy import PolicyBot
from sim.match import play_match
from sim.opponents import ZOO
for name in sys.argv[1:]:
    t0 = time.time()
    m = play_match(PolicyBot, ZOO[name], "s1", 1000)
    banks = " / ".join(f"{h.bank_b:.2f}" for h in m.halves)
    raises = " / ".join(str(h.actions_b.get("RAISE_HALF", 0) + h.actions_b.get("RAISE_POT", 0)) for h in m.halves)
    print(f"{name} s1 no hook: v1 {m.mean_bb:+.3f} [{m.ci_low:+.3f}, {m.ci_high:+.3f}] {m.verdict}; "
          f"FIELD bank {banks} s; FIELD raises {raises} ({time.time() - t0:.0f} s)", flush=True)
```
