# FIELD: the agentic competitor bots

Three bots, each built by a separate agent acting as an independent rival team. Each agent had
only the organizer's starter kit and the participant guide, the way other teams build their bots
with AI coding tools. FIELD is our closest proxy for the real tournament field.

**Held out, for ship decisions only: test against FIELD, never tune on it.**
- `sim.opponents.FIELD` holds the three ZOO keys below. TRAIN excludes them explicitly:
  `TRAIN = set(ZOO) - HOLDOUT - FIELD`.
- `--opponents field` selects them, in `sim.tournament` and in `sim.heuristic_bot`.
- `--opponents all` includes FIELD. So does the `--entrants zoo` round robin, which plays every
  ZOO bot.

## Files

The runtime files were copied unmodified from `/home/kchen/code/competitor-bots/` on 2026-11-07.
The `dev/` folders, reports and kit files were not copied. The sha256 below is that of each
source file, and `tests/test_field.py` checks every copy against it.

| file | sha256 | source |
|---|---|---|
| `field_equity/agent.py` | `df384314c2c0668df86e3c5602c0b9a70d28369eb577fa709aa414477c3384d7` | `agentic_equity/agent.py` |
| `field_equity/preflop_eq.npy` | `f86c5c013e1a1d287fc2a82205c43912e697945cbfbcb8360251ebdacb47da5f` | `agentic_equity/preflop_eq.npy` |
| `field_gto/agent.py` | `92b00a151c1e1cbd626cec8b2cab4016941747a3d4ba2b0efbf8d219a6a6d676` | `agentic_gto/agent.py` |
| `field_gto/river_cfr.json` | `b04f1535e5c320d5338c463c43670a3328c4f603f2e56d72113e1fc2ea067e75` | `agentic_gto/river_cfr.json` |
| `field_exploit/agent.py` | `40acd7ffbc994de48993c85cf6cce36ac08b082aa1086898858eac0109c76993` | `agentic_exploit/agent.py` |

The bots import the organizer's `game.py` and `evaluator.py`. Those resolve to this repo's root
copies, which are byte-identical to each team folder's:
- `game.py`: `bc00e3c2d1bf64cbd36bf5553dae218767f11b20f388e5c417d989919877f2db`
- `evaluator.py`: `e2a151a4c41828a035030f490c91b81015d04bc209abfb860ca3d0dce649e0b4`

## Loading

The bots are registered like the `llm_*` bots:
`ZOO[name] = sim.match.bot_file_factory("sim/opponents/field/<name>/agent.py")`, which is
picklable.

No adapter or edit was needed. Each bot finds its data file next to its `agent.py` through
`__file__`, and `load_bot_file` sets `__file__` to the copy. Both data files are optional to
their bots: a missing file silently falls back to weaker play. `tests/test_field.py` therefore
checks that `_PRE_EQ` and `_CFR` loaded.

Startup runs at import, inside the untimed factory call, like the tournament's startup. Measured
here under load (load average about 19 on 16 threads), it took 2.7-5.1 s for field_equity,
0.1-0.4 s for field_gto and 2.3-10.4 s for field_exploit.

## Differences in our in-process sim

- **field_gto runs `gc.collect(); gc.freeze()` at import.** In the tournament that touches only
  its own process. In the sim it freezes every object alive in the worker at that moment, ours
  included, so later collections in that worker skip them. That only makes collections cheaper.
  It is left as is, because the freeze is not what keeps memory (next point).
- **Memory.** FastAPI's endpoint caches (`lru_cache`, 4096 entries) hold every loaded
  `agent.py`'s route functions. So a dropped bot module keeps its globals and tables alive. That
  is true of the `llm_*` bots too, but the FIELD tables are larger. Each FIELD half leaves 12-17
  MB in a long-lived worker process (measured per load: 15 MB field_equity, 12 MB field_gto,
  17 MB field_exploit).
- **Logging.** field_exploit prints one diagnostic line to stderr every 250 hands.
- **Randomness.** Per half, field_equity seeds its RNG from the clock and field_gto from OS
  entropy. Their results vary from run to run, even on fixed seeds.
- **Time.** All three measure their own time and degrade under load (fewer Monte Carlo samples,
  then rules). Their bank use and their strength in the sim depend on machine load, like ours.
  At load average about 20, field_gto played up to 37% of a half in its no-sampling emergency
  mode. That contention made our first information run overstate v1-untuned by 1.2-3.0
  bb/deal against these bots, compared with a quiet serial rerun (`docs/results/field.md`).
  **Evaluate against FIELD on a quiet machine, and check the FIELD bots' bank use:** at full
  strength each was charged at most 0.13 s per half.

## The bots

### `field_equity`: EquityEdge ("RangeEquity")

From its REPORT.md. A range-aware equity engine:
- **Evaluation.** A vectorized exact evaluator (about 0.4 ms per 1,000 samples), plus a preflop
  table (`preflop_eq.npy`: the equity vs a random hand of all 376,992 five-card hands).
- **Range model.** Monte Carlo equity against the opponent's *estimated* range. Each sampled
  opponent hand is weighted by how likely it was to take this hand's actions. The likelihoods
  use per-half action frequencies on priors. `on_hand_end` infers the actions it never saw.
- **Decisions.** One-street EV comparisons of fold, call, check, half-pot and pot, with a
  range-conditional fold probability. Preflop first-in play uses table percentiles.
- **Time.** A throttle calibrated at startup cuts the samples (250/600, then 60-120, then rules
  with no Monte Carlo) to stay under 62% of the bank.
- **Reported results.** +5.6 to +10.4 bb/deal vs the four baselines, +1.2 vs a tight folder and
  +2.8 vs an equity-vs-random bot.
- **Known weaknesses.** The range model is heuristic, the EV looks one street ahead, the preflop
  thresholds are fixed, and play gets noisier when it throttles under CPU pressure.

### `field_gto`: Equilibrium ("EquilibriumGTO")

Its source folder had no REPORT.md when it was copied. Its `agent.py` is byte-identical to its
own `dev/versions/final.py`. This description is taken from the strategy comments in its YOUR
CODE section. A game-theory-first bot:
- **Evaluation.** Hand strength is Monte Carlo equity from a vectorised numpy evaluator, with
  one cached sample set per street (about 0.3-0.8 ms).
- **Range model.** Each opponent bet, raise or call narrows his range to the top fraction f of
  hands, where f is his observed frequency of that action in that spot (Bayesian counts with
  priors). Equity is measured against that weighted range.
- **GTO principles.** Pot-odds/MDF-style calling. Polarised bets, with bluff:value ratios
  matched to the bet size. Mixed (randomised) actions, and some trapping.
- **River.** Its river play mixes from a CFR+ equilibrium table (`river_cfr.json`). The table
  covers the exact river betting tree, with 16 strength buckets per player and several
  stack-to-pot ratios.
- **Exploits, only with evidence.** Its bluff frequency scales with the opponent's observed
  fold-to-bet rate, and its value thresholds with his calling rate.
- **Time.** A time guard shrinks the sample count when the half's budget is at risk.

### `field_exploit`: SharkTank ("SharkTankExploiter")

From its REPORT.md. A maximum-exploitation bot:
- **Evaluation.** At startup it builds numpy tables: a dense rank for every 2-hole-card plus
  3-board-card combination, and a fitted preflop-strength table. Each decision samples 200 (240
  preflop) opponent-hand and runout pairs.
- **Range model.** It re-weights the sampled hands by every action the opponent took this hand.
- **Decisions.** Every legal action is scored by a myopic chip EV against that range. The EV
  uses the opponent's learned fold rate per street and bet size.
- **Learning.** Decaying per-half counts (about a 50-observation window) on balanced priors,
  plus bluff rates read from showdowns, per bet size.
- **Counter-strategies.** The same formula value-bets vs stations, steals vs folders, calls down
  vs maniacs and folds vs nits. A detector raises every hand once the opponent looks timed out
  (check/fold autopilot).
- **Time.** A throttle steps down from full Monte Carlo to 60 samples, then a lite mode, then
  rules.
- **Reported results.** +5.0 to +12.2 bb/deal vs the baselines over HTTP, +4.7 vs an equity bot,
  and +0.7 vs a nit (-1.1 when throttled).
- **Known weaknesses.**
  - Its bank under heavy CPU oversubscription: one half banked out in its own judge runs.
  - Weaker degraded modes.
  - A myopic EV.
  - Slow learning: its first ~100 hands are close to its TAG prior.
