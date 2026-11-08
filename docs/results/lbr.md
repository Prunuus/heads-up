# Local Best Response (LBR) probe — results

`sim/opponents/lbr.py` (Task 21). LBR [Lisý & Bowling 2017] plays a target bot, knowing its
policy but not its cards, and best-responds one decision at a time. It assumes the target folds
to a raise with its policy's fold probability (otherwise calls), and that both players
check/call to showdown afterwards. Its average win in bb/deal (both seats of every deal,
like the judge) is a **lower bound on the target's exploitability**. Bigger means more
exploitable. The module docstring has the full rule and the target contract (BotLike
re-simulation, or `DistributionTarget` for the blueprint's `probs`).

Run (repo root; heavy runs pinned and niced, R32):

```
taskset -c 8-15 nice -n 19 .venv/bin/python -m sim.opponents.lbr --target heuristic \
    --seeds 16 --deals 250 --workers 8 [--particles N] [--rollouts R] [--out x.json]
```

`--target` is `heuristic` (`sim/heuristic_bot.py` at v1 defaults, "v1-untuned"), `agent`,
a ZOO name or a bot file. An LBR match gives both bots an effectively unlimited time bank
(`LBR_BANK_S`), so a time-managed target always plays its full-budget strategy, and LBR's
re-simulation of it matches the target that plays. Seeds are `lbr1..lbrN` (`--seed-prefix`).
A run is deterministic for fixed seeds. Checked: the per-deal values are identical with 1
and 2 workers.

## Results (defaults: 1,000 particles, 500 rollouts)

| Target | LBR bb/deal | 95% CI | Deals | Seeds × deals |
|---|---:|---|---:|---|
| heuristic v1-untuned | **+3.891** | [+3.445, +4.338] | 4,000 | lbr1–16 × 250 |
| always_pot | +6.085 | [+5.181, +6.989] | 2,000 | lbr1–20 × 100 |
| calling_station | +13.140 | [+12.387, +13.893] | 2,000 | lbr1–20 × 100 |

Belief health against the heuristic: 385 live particles per decision on average; 35 of 17,242 target actions
inconsistent (0.2%); 1 rebuild in 8,000 hands; 21.5M target re-simulations. "Inconsistent" counts target actions that no
live particle predicted. A low count means the re-simulation reproduces the target that plays.
The target made no errors.

v1, v2 and v3 are measured by the controller at T20 with the same command (`--target agent` or a
bot file) and seeds. `LBRResult.per_deal` allows a per-deal comparison of two targets on common
seeds: `mean_ci(r3.per_deal - r2.per_deal)`.

## Particles: accuracy vs cost

The plan says 2,000 particles × 500 rollouts. The same 1,000 deals (seeds `to1..to8` × 125)
were played against heuristic v1-untuned at three particle counts. Costs are from the nearly
idle machine (8 workers on CPUs 8–15):

| Particles | LBR bb/deal [95% CI] | vs 2,000, paired [95% CI] | ms/decision | LBR s/deal | Wall, 1,000 deals | Inconsistent |
|---:|---|---|---:|---:|---:|---:|
| 2,000 (plan) | +3.486 [+2.586, +4.387] | n/a | 412 | 2.04 | 265 s | 3/4,245 |
| **1,000** | +3.448 [+2.519, +4.377] | −0.038 [−0.800, +0.723] | 198 | 0.98 | 131 s | 14/4,274 |
| 500 | +2.962 [+2.029, +3.896] | −0.524 [−1.429, +0.381] | 91 | 0.46 | 62 s | 32/4,313 |

- The cost is linear in particles: each LBR decision re-simulates the target once per live
  particle for each candidate raise. Each target action costs one re-simulation per live
  particle, unless it answers LBR's raise, whose branch is reused.
- Rollouts are cheap. Win probabilities take a few ms per decision, against 100–400 ms for
  the re-simulation, so the plan's 500 stays.
- **Choice: 1,000 particles.** That halves the cost, and LBR's value moves by −0.04 bb/deal
  against the plan's setting, far inside the noise. 500 particles shows a hint of weakening
  (−0.52, not significant).
- At 1,000 particles, one standard evaluation (4,000 deals) takes ~9 min on 8 free workers.
  On CPUs shared with Track B's dataset job it takes 20 min (1,211 s, with ~43% of a core per worker) (the headline run above).
  Either way it is within the ~30 min budget. The per-deal SD is ~14.5 bb, so 4,000 deals
  give a CI of about ±0.45 bb/deal.
- Common random numbers mattered more than the particle count. The first version estimated
  each action's win probability from its own independent 500 completions. A raise that nobody
  folds to then got a `wp` ±0.03–0.04 off the call's, enough to flip close call/raise
  choices. Against always_pot, LBR measured +1.7 bb/deal [−1.0, +4.4] on 200 deals
  (`lbr1–2` × 100, 2,000 particles). With one per-particle estimate shared by all actions,
  it measured +5.04 [+3.75, +6.33] on 1,000 deals (`lbr1–10` × 100, a superset of those
  200). This was not a controlled pair, but the size is consistent with the decision trace.
  Every other number here uses the shared estimate.

## Caveats

- LBR is myopic (check/call afterwards), so its value is a lower bound and weaker against
  aggression. Against always_pot it calls pot bets with 1/3 < wp < 1/2, expecting no more
  betting, and then faces more bets.
- The target is measured with an unlimited time bank. That is its full-budget strategy, the
  same for every target.
- For BotLike targets, per-half state must live in `memory`, and randomness must come from an
  rng stored at the top level of `memory` (module docstring). A target that breaks this is
  still sampled, but LBR's model of it is then approximate.
- `lbr_eval` fails closed: it raises `ModelCheckError`, and the CLI exits with code 2. That
  happens when re-simulated decisions or the shadow's `on_hand_end` err while the real target
  never does, or when more than 2% of target actions (+2) are predicted by no particle.
  `--max-inconsistent` raises the limit; `--no-model-check` opts out.
  - The heuristic passes: a 50-deal run had 0 re-simulation errors and 3/230 inconsistent
    actions, and the headline run had 35/17,242 (0.2%).
  - Only the `DistributionTarget` adapter is weighted by its probabilities. Its
    `choose_action` samples them by construction; any other bot is re-simulated.
