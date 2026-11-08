# FIELD2: the second-wave competitor bots

Three more bots built by separate agents acting as independent rival teams, from the organizer's
starter kit and participant guide (`/home/kchen/code/competitor-bots/hard_*`). They are a second
held-out split, used only for ship decisions.

**Held out: test against FIELD2, never tune on it, never change a bot's strategy.**
- `sim.opponents.FIELD2` holds the three ZOO keys below. TRAIN excludes them explicitly:
  `TRAIN = set(ZOO) - HOLDOUT - FIELD - FIELD2`. The tuner refuses FIELD2 (it accepts TRAIN only).
- `--opponents field2` selects them, in `sim.tournament` and in `sim.heuristic_bot`.
- `--opponents all` includes FIELD2. So does the `--entrants zoo` round robin.

## Files

Runtime files copied unmodified on 2026-11-08. The `dev/` folders, reports, harness scripts and
kit files (including `sdcore.py`) were not copied. The sha256 is that of each source file;
`tests/test_field2.py` checks every copy against it.

| file | sha256 | source |
|---|---|---|
| `field2_solver/agent.py` | `b106228f0f395b6fbced5cedd30e2a6c700c0cf17afe179e561e4d27590185f0` | `hard_solver/agent.py` |
| `field2_solver/preflop.npz` | `cae918c089b0bdbf8bc145e9475895df1dc163262f71e394cd457480c5b7bdf4` | `hard_solver/preflop.npz` |
| `field2_solver/river.npz` | `eaf8fc83ced66ae3224e0c939af708d07c030fe3406116991da2a0805c7af177` | `hard_solver/river.npz` |
| `field2_solver/pf_range.npy` | `0851eea61c6da6d79c7358029c809d93ce007c859798a45f6639e15a443d4002` | `hard_solver/pf_range.npy` |
| `field2_bayes/agent.py` | `9847e88874e18c55769b81e281789934b13940a8d326126e1ecbac988f5387a6` | `hard_bayes/agent.py` |
| `field2_bayes/tp_rank.npy` | `d5e3d01b4a34821070d85fe094a313727cb8b19c14fc473bebd959868eb7e26a` | `hard_bayes/tp_rank.npy` |
| `field2_bayes/pf_hs.npy` | `6ab8ef067af2cedb08b7fa18bf671e8eb66b3f5a0dacb301c0b9745133197c8c` | `hard_bayes/pf_hs.npy` |
| `field2_adaptive/agent.py` | `76e2ddeb65b2c2c2fd58fbb5305c286a786bf565b0df21b7d99a4dad0b81440e` | `hard_adaptive/agent.py` |
| `field2_adaptive/ptab.npy` | `37edbb5a5b2aaad08d17dd0be4d3572b441dff0a81c1041802ff62d7c2c5b966` | `hard_adaptive/ptab.npy` |
| `field2_adaptive/pfeq.npy` | `73102322789096d6bb95bee2e1eb40fd1affc7d1df98aee05c44e4e59a113281` | `hard_adaptive/pfeq.npy` |

The bots import the organizer's `game.py` and `evaluator.py`; the repo's root copies are
byte-identical to each team folder's (checked with `cmp`).

## Loading

`ZOO[name] = sim.match.bot_file_factory("sim/opponents/field2/<name>/agent.py")` (picklable). No
adapter or edit was needed. Each bot finds its data files next to `agent.py` through `__file__`.
A missing data file silently falls back (empty tables, or rebuilt tables at startup), so
`tests/test_field2.py` checks that the tables really loaded. Evaluate on a quiet machine: the bots
throttle their sampling under CPU load, as the FIELD bots do (see `sim/opponents/field/README.md`).

## The bots

### `field2_solver`: Solverline ("SolverlineCFR")

From its REPORT.md. Offline tables plus a solved river:
- **Preflop.** `preflop.npz` is equity vs random for all suit-canonical hands, as a percentile;
  `pf_range.npy` is the all-in equity of a hand at percentile p vs the top-f fraction of hands.
  It opens the top 70% with a pot raise and estimates the opponent's range from observed rates.
- **River.** `river.npz` is a CFR+ solution of the exact river tree (20 equity buckets), sampled
  from the mixed strategy.
- **Flop and turn.** Rules over Monte Carlo equity vs a weighted opponent range, with polarized
  value bets and semi-bluffs scaled by the opponent's fold-to-bet rate.
- **Safety.** A private time budget drops samples, then plays with zero compute.
- **Known weaknesses.** Too tight against high-frequency aggression, flop and turn are not
  solved, and its time use is high on loaded hosts.

### `field2_bayes`: Posterior

Its folder had no report; this is taken from the strategy comments in its `agent.py`. Bayesian
hand reading: a weighted particle belief over the opponent's five cards, reweighted after each
opponent action by its likelihood under a per-opponent action model (observed action frequencies
by street, facing and size, mixed with a strength-independent component learned from
showdowns). It picks the action with the best expected value against the belief, softened by a
softmax. Data: `tp_rank.npy` (rank table) and `pf_hs.npy` (preflop hand strength).

### `field2_adaptive`: Metagame

Its folder had no report; this is taken from the strategy comments in its `agent.py`. An ensemble
of four parametric styles on one shared range-aware Monte Carlo equity engine. A
Thompson-sampling bandit picks the style per two-hand block from luck-adjusted results, and fast
detectors (over-folder, calling station, maniac, out-of-bank check/folder) overlay exploits.
Data: `ptab.npy` (pair-rank table) and `pfeq.npy` (preflop equity).
