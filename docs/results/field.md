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

## Results

| FIELD bot | v1-untuned bb/deal | 95% CI, pooled paired (3,000 deals) | s1 | s2 | s3 |
|---|---:|---|---|---|---|
| `field_equity` | **+0.219** | [-0.172, +0.610] | +0.16 [-0.54, +0.87] CHOP | +0.15 [-0.48, +0.78] CHOP | +0.34 [-0.35, +1.04] CHOP |
| `field_gto` | **-0.011** | [-0.393, +0.370] | -0.62 [-1.32, +0.09] CHOP | +0.83 [+0.38, +1.29] A | -0.25 [-1.03, +0.52] CHOP |
| `field_exploit` | **+3.927** | [+3.356, +4.499] | +3.94 [+2.93, +4.95] A | +3.51 [+2.49, +4.53] A | +4.33 [+3.39, +5.27] A |

A is a v1-untuned win. Over the 9 matches that is 4 wins, 5 chops and no losses, 6.5 points.
- **vs field_exploit:** v1-untuned wins every seed, by +3.9 bb/deal. The win holds even in s1,
  where both of the bot's halves stayed under its full-Monte-Carlo cap (time tiers below).
- **vs field_equity:** a chop on every seed, +0.22 bb/deal.
- **vs field_gto:** a chop overall. Its one v1 win (s2) is the seed where field_gto spent the
  most bank. See the time tiers below before reading it as an edge.

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

- **No FIELD bot ran out of bank in any half**, and none made an error or forfeited. So no half
  was played on check/fold autopilot, and no result here is inflated by a bank-out.
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
  - **So the ≈0 vs field_gto likely overstates v1.** On a quiet machine field_gto plays its
    sampled strategy all half. Rerun this matchup on a quiet machine before reading it as a
    chop.
- **field_exploit's loss to v1 is not an artifact of its throttle.** In the rerun it never
  reached its rules tier, and v1 won all three seeds again by about the same margins.
  - In the main run, each of its stderr lines carries its decision count. That maps each line to
    one half exactly (requests = decisions + 2). Its own time at the end of each half was s1
    1.21 / 1.33 s, s2 3.29 / 1.53 s and s3 6.39 / 0.81 s.
  - Four halves stayed under its 1.76 s full-Monte-Carlo cap (22% of the bank). Two passed its
    3.04 s rules cap: s3 half 1 by hand 500 (3.30 s there), and s2 half 1 after hand 750 (2.50 s
    there).
  - s1, with both halves under the full-Monte-Carlo cap, scored +3.94. s3 scored +4.33 here and
    +4.29 in the rerun, where no half reached the rules tier.
  - Only the rules tier was counted. Its 60-sample and lite tiers depend on its projections,
    which the wrapper did not recompute.
- **field_equity's tiers were not instrumented.** Its bank use (1.80-2.27 s) and request counts
  were steady across halves.

## Reproduce

Run from the repo root, with the two scripts below saved there. The deals are fixed by the seeds.
The numbers are not: `field_equity` seeds its RNG from the clock and `field_gto` from OS
entropy, and every bot's time guard reacts to machine load.

```
PYTHONPATH=. taskset -c 8-15 nice -n 19 .venv/bin/python field_info_run.py field_info.json
PYTHONPATH=. taskset -c 8-15 nice -n 19 .venv/bin/python probe_tiers.py field_gto s1 s2 s3
PYTHONPATH=. taskset -c 8-15 nice -n 19 .venv/bin/python probe_tiers.py field_exploit s1 s2 s3
```

`--opponents field` in `sim.tournament` and `sim.heuristic_bot` selects the same three bots,
for the usual per-opponent points and mean bb/deal.

`field_info_run.py`, verbatim. It wrote the JSON the first two tables were formatted from:

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
