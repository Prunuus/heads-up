# Task F report: the agentic competitor bots as the FIELD split

Branch `kevin-field`, worktree `/home/kchen/code/heads-up-bundle-field`, base ff537d2.

## What I implemented

1. **Copies.** I copied the runtime files unmodified into `sim/opponents/field/<name>/`:
   - `field_equity`: `agent.py` and `preflop_eq.npy`
   - `field_gto`: `agent.py` and `river_cfr.json`
   - `field_exploit`: `agent.py`

   `sim/opponents/field/README.md` holds:
   - the sha256 table;
   - one paragraph per bot;
   - the split rules ("all" and the `--entrants zoo` round robin include FIELD);
   - the differences between our in-process sim and the tournament.
2. **Registration** in `sim/opponents/__init__.py`:
   - `ZOO[name] = bot_file_factory("sim/opponents/field/<name>/agent.py")`, picklable, as for
     the llm bots.
   - `FIELD = {"field_equity", "field_gto", "field_exploit"}`.
   - `TRAIN = set(ZOO) - HOLDOUT - FIELD`.
3. **CLIs.** `sim/tournament.py` gets `OPPONENT_SPLITS = ("train", "holdout", "field", "all")`.
   - `select_opponents` maps "field" to FIELD.
   - `sim.tournament --opponents` uses those choices.
   - `sim.heuristic_bot --opponents` uses the same tuple, for its check and its help text, so the
     two CLIs cannot drift apart again.
4. **Tests** in `tests/test_field.py` (details below). Two existing tests changed:
   - `tests/test_tournament.py::test_resolve_bot`: its partition invariant is now
     `HOLDOUT | TRAIN | FIELD == set(ZOO)`. The old invariant fails by design once TRAIN
     excludes FIELD.
   - `tests/test_zoo.py`: its per-bot 100-deal test skips FIELD, with a comment. test_field
     covers those bots, and 100-deal FIELD halves in the fast suite would defeat "a few seconds
     total".
5. **Information run** in `docs/results/field.md`: results, bank, errors, bank-outs, a time-tier
   diagnostic, and the scripts verbatim.

## Adapters or edits to a copied bot

None. Every file is byte-identical to its source, and `tests/test_field.py` enforces that against
the README table.
- **Imports.** The bots import `game` and `evaluator`, which resolve to the repo root. I
  re-checked that those are byte-identical to all three team folders' copies.
- **Data files.** Both are found through `__file__`, which `load_bot_file` sets to the copy.
- **Silent fallback.** Both data-file bots fall back silently to weaker play when the file is
  missing. So a test asserts `_PRE_EQ` and `_CFR` are loaded.

I considered one adapter and decided against it. `field_gto` runs `gc.collect(); gc.freeze()` at
import, which in the sim is process-wide: about 56k objects frozen per worker.
- I measured it with weakrefs. The freeze is not what keeps memory, and `gc.unfreeze()` before or
  after the load frees nothing more.
- The freeze only makes later collections cheaper. So I left the bot untouched and documented it
  in the README.

## sha256 (source file = copy)

| copy in `sim/opponents/field/` | sha256 | source in `/home/kchen/code/competitor-bots/` |
|---|---|---|
| `field_equity/agent.py` | `df384314c2c0668df86e3c5602c0b9a70d28369eb577fa709aa414477c3384d7` | `agentic_equity/agent.py` |
| `field_equity/preflop_eq.npy` | `f86c5c013e1a1d287fc2a82205c43912e697945cbfbcb8360251ebdacb47da5f` | `agentic_equity/preflop_eq.npy` |
| `field_gto/agent.py` | `92b00a151c1e1cbd626cec8b2cab4016941747a3d4ba2b0efbf8d219a6a6d676` | `agentic_gto/agent.py` |
| `field_gto/river_cfr.json` | `b04f1535e5c320d5338c463c43670a3328c4f603f2e56d72113e1fc2ea067e75` | `agentic_gto/river_cfr.json` |
| `field_exploit/agent.py` | `40acd7ffbc994de48993c85cf6cce36ac08b082aa1086898858eac0109c76993` | `agentic_exploit/agent.py` |

`game.py` (`bc00e3c2…77f2db`) and `evaluator.py` (`e2a151a4…e0b4`) are identical in the repo and
in all three team folders. I re-verified the sources against the copies at 19:58, after the
copy, with `sha256sum -c`.

## Information run: v1-untuned (`bot.policy.PolicyBot()` at ff537d2) vs FIELD

The run used seeds s1-s3, 1,000 deals per half, duplicate (both halves) and 4 workers on CPUs
8-15 at nice 19. It took 57 s at 20:01, at load average 18-20 on 16 threads. Paired bb/deal is
the judge's definition. Pooled is `mean_ci` over the 3,000 per-deal values.

| FIELD bot | v1 bb/deal | pooled 95% CI | s1 | s2 | s3 |
|---|---:|---|---|---|---|
| field_equity | +0.219 | [-0.172, +0.610] | +0.16 [-0.54, +0.87] CHOP | +0.15 [-0.48, +0.78] CHOP | +0.34 [-0.35, +1.04] CHOP |
| field_gto | -0.011 | [-0.393, +0.370] | -0.62 [-1.32, +0.09] CHOP | +0.83 [+0.38, +1.29] A | -0.25 [-1.03, +0.52] CHOP |
| field_exploit | +3.927 | [+3.356, +4.499] | +3.94 [+2.93, +4.95] A | +3.51 [+2.49, +4.53] A | +4.33 [+3.39, +5.27] A |

**Bank use per half** (seconds charged of 8 s; h1 / h2 per seed):

| FIELD bot | s1 | s2 | s3 |
|---|---|---|---|
| field_equity | 1.95 / 1.96 | 1.80 / 2.08 | 2.18 / 2.27 |
| field_gto | 1.99 / 3.50 | 3.97 / 4.24 | 2.15 / 2.05 |
| field_exploit | 0.51 / 0.88 | 2.88 / 0.91 | 6.21 / 0.44 |

- **Errors, forfeits, bank-outs:** 0 on both sides in all 18 halves. No FIELD half played
  check/fold for lack of bank.
- v1-untuned's bank peaked at 3.22 s, and its longest request was 101.7 ms.

**Time-tier diagnostic.** No FIELD bot banked out, but they degrade before that. A read-only
rerun counted the decisions made in each bot's weakest tier, by its own thresholds. It used the
same seeds, at 20:04-20:06.
- **field_gto** played 15 / 22 / 37 / 4 / 0 / 6% of its decisions per half in its no-sampling
  emergency mode.
  - The rerun's v1 results were +0.47, +0.52 and -0.83 (a loss on s3, the seed with the least
    emergency play).
  - In the main run, its costliest halves also had the fewest requests. That is the signature of
    the fold-heavy emergency mode, and s2 is v1's only win.
  - **So the ≈0 vs field_gto likely overstates v1.** It needs a quiet-machine rerun before
    anyone reads it as a chop.
- **field_exploit** was never in its rules tier in the rerun, and v1 won all three seeds again
  (+3.81, +3.64, +4.29).
  - In the main run, its stderr maps exactly to halves. Two halves passed its rules cap: s3
    half 1 by hand 500, and s2 half 1 after hand 750.
  - Four halves stayed under its full-Monte-Carlo cap. The +3.9 edge is robust to that.

Everything is in `docs/results/field.md`.

## Tests

`tests/test_field.py`:
- **Fast suite, 11 tests, 4.2 s:**
  - the names;
  - FIELD is disjoint from TRAIN and from HOLDOUT;
  - `select_opponents("field" / "train" / "all")`;
  - both CLIs accept `--opponents field`. A stand-in `focus_eval` captures the selected names,
    so no matches are played;
  - every copy matches its README sha256, and every file on disk is listed;
  - `_PRE_EQ` and `_CFR` are loaded;
  - each FIELD bot plays a 10-deal half vs calling_station with 0 errors and no forfeit. Each
    bot is loaded once, through its ZOO factory.
- **`@pytest.mark.slow`, 3 tests, 12.3 s:** a 200-deal half per bot, through `ZOO[name]`
  directly, so each half gets a fresh load.

**Full fast suite**, all runs `taskset -c 8-15 nice -n 19 .venv/bin/python -m pytest -q -p no:cacheprovider`:

| run | passed | failed | deselected | time |
|---|---:|---:|---:|---:|
| baseline at ff537d2, before any change | 230 | 8 | 27 | 455 s |
| after the change | 247 | 2 | 33 | 149 s |

- **Baseline failures.** All 8 are timing tests failing under load (average about 19):
  - `test_equity::test_timing_p99…`;
  - 5 `test_judge_fidelity` timeout and bank-out cases;
  - `test_policy::test_warmup_under_6s` (13.9 s);
  - `test_zoo[mc_equity_slow]`.
- **After the change**, the 2 failures (`test_equity::test_timing_p99_under_1_5ms_at_128_iters_flop`
  and `test_judge_fidelity::test_bank_out_matches_the_judge[slow-hand-end]`) are both in the
  baseline set. No new failure.
- **The counts reconcile.**
  - passed + failed grew by 11, which is my 11 new fast tests;
  - deselected grew by 6: my 3 slow tests, plus 3 new `test_policy::test_zero_errors_vs_whole_zoo[field_*]` cases (it parametrizes over all of ZOO).
- I ran those 3 new slow PolicyBot cases with `-m slow -k "test_zero_errors_vs_whole_zoo and field"`: 3 passed, 10.8 s.
- After my last README edits I reran `tests/test_field.py tests/test_zoo.py tests/test_tournament.py`: 45 passed, 3 deselected.

## TDD evidence

**RED.** I wrote `tests/test_field.py` and the updated `test_resolve_bot` before any
registration code:
```
$ taskset -c 8-15 nice -n 19 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_field.py
tests/test_field.py:10: in <module>
    from sim.opponents import FIELD, HOLDOUT, TRAIN, ZOO
E   ImportError: cannot import name 'FIELD' from 'sim.opponents' (.../sim/opponents/__init__.py)
ERROR tests/test_field.py
1 error in 0.30s
```
`tests/test_tournament.py` failed the same way at its import line. This was expected, because
FIELD and the ZOO keys did not exist yet.

**GREEN.** After the registration, the CLI changes and the README:
```
$ taskset -c 8-15 nice -n 19 .venv/bin/python -m pytest -p no:cacheprovider tests/test_field.py "tests/test_tournament.py::test_resolve_bot" -v
... 12 passed, 3 deselected in 4.21s
$ taskset -c 8-15 nice -n 19 .venv/bin/python -m pytest -p no:cacheprovider -m slow tests/test_field.py -v
... 3 passed, 11 deselected in 12.32s
```

## Commits (on `kevin-field`, not pushed)

- `6e2b7f2` Register the three agentic competitor bots as the held-out FIELD opponent split.
- `8d8ba51` Add the v1-untuned vs FIELD information run results.

The staged blobs of the five copied files hash to the source sha256s. At 20:11 I re-checked: the
sources still match the committed copies, and `agentic_gto/` still has no REPORT.md.

## Files changed

- new: `sim/opponents/field/README.md`
- new: `sim/opponents/field/field_equity/agent.py` and `preflop_eq.npy`
- new: `sim/opponents/field/field_gto/agent.py` and `river_cfr.json`
- new: `sim/opponents/field/field_exploit/agent.py`
- new: `tests/test_field.py`
- new: `docs/results/field.md`
- modified: `sim/opponents/__init__.py`, `sim/tournament.py`, `sim/heuristic_bot.py`,
  `tests/test_tournament.py`, `tests/test_zoo.py`

## Self-review findings

- I fixed three overstatements in my own docs before committing:
  - "the agent stopped without writing a report" was an inference; the README now says only
    that the folder had no REPORT.md when copied;
  - both startup timings were taken under load;
  - the field_exploit tiers had been inferred from charged bank, which ignores its 0.75 ms free
    per request. They are now taken from its own stderr, mapped exactly to halves by decision
    count.
- The Reproduce section embeds the exact scripts that produced the numbers, not a condensed
  rewrite.
- The diff touches no organizer files, no `bot/` and no `agent.py`. No copied bot file changed.

## Issues and concerns

1. **No REPORT.md for field_gto.** `agentic_gto/` had none at 19:44 or at 19:58. Its last write
   was 19:34, no process was running in the folder, and its `agent.py` equals its own
   `dev/versions/final.py`. The README paragraph for it comes from the strategy comments in its
   `agent.py`, and is labeled that way.
2. **The field_gto matchup is load-sensitive.** Its emergency mode covered up to 37% of a half
   at tonight's load. Treat v1 ≈ 0 vs field_gto as optimistic for us until it is rerun on a quiet
   machine. All FIELD numbers depend on load, ours too.
3. **Memory retention (pre-existing, harness-wide).** FastAPI's endpoint `lru_cache` (4096
   entries) pins every loaded `agent.py`'s globals, llm bots included. FIELD halves leave 12-17
   MB each in a long-lived worker.
   - That is fine for focus runs.
   - For an `--entrants zoo` round robin, which now includes FIELD, it can add up to GBs across
     workers with several seeds.
   - A fix belongs in `sim.match.load_bot_file`, outside this task.
4. **Merge with Track A's uncommitted work** in the main worktree:
   - `tests/test_tournament.py:138` (my edit) is adjacent to Track A's appended test, so expect a
     trivial conflict.
   - Track A's `time_scale` swaps a bot file's module-level `time`. That reaches field_equity and
     field_exploit, but not field_gto, which imports `time as _time`.
5. Pre-existing timing failures under load are unrelated to this task (see the suite comparison
   above).

## Late addendum

**The probe had already finished.** The background `probe_tiers.py` run (task `b7qumx0xe`)
finished its matches at 20:05:42, before I wrote this report. Its numbers are already in the
"Time-tier diagnostic" section above, and in `docs/results/field.md` (committed in `8d8ba51`).
So this addendum changes nothing in the commits. It restates field_gto's emergency-mode share
per seed, with the counts.

**Why the task looked unfinished.** Its wrapper shell never exited. Its wait loop, `while pgrep -f
"probe_tiers.py field_gto"`, matched the wrapper's own command line. At 20:13 I stopped that
background task: an idle `sleep 2` loop, PID 135902, now gone. No file and no commit was
touched. `kevin-field` is still clean at `8d8ba51`.

**field_gto in its emergency mode** (no sampling, once its own time > 2.5 s, or 5 s on an
unstarved CPU). It is a read-only rerun of v1-untuned (`PolicyBot()`) vs field_gto, seeds s1-s3
× 1,000 deals, both halves, at 20:04-20:06, load average about 19-20:

| seed | emergency decisions | share | per half | rerun v1 bb/deal [95% CI] |
|---|---:|---:|---|---|
| s1 | 914/4855 | 18.8% | h1 378/2466 (15.3%, from hand 683, bank 2.63 s); h2 536/2389 (22.4%, from hand 590, bank 2.92 s) | +0.465 [-0.266, +1.195] CHOP |
| s2 | 844/4893 | 17.2% | h1 727/1963 (37.0%, from hand 405, bank 3.05 s); h2 117/2930 (4.0%, from hand 916, bank 2.06 s) | +0.521 [-0.289, +1.331] CHOP |
| s3 | 165/6062 | 2.7% | h1 0/3132 (0.0%, never, bank 1.27 s); h2 165/2930 (5.6%, from hand 863, bank 1.54 s) | -0.829 [-1.644, -0.014] B |
| all | 1923/15810 | 12.2% | | |

**Reading.** The seed where field_gto played at full strength longest (s3, 2.7% emergency) is the
one v1-untuned lost. The two seeds with about 17-19% emergency play were chops leaning v1.
That is consistent with the main run's conclusion: v1 ≈ 0 vs field_gto is optimistic for us
until it is rerun on a quiet machine. For comparison, field_exploit was in its rules tier for
0% of its decisions in all six rerun halves.

## Fix report, round 1

The finding (Important): the half tests could not detect a FIELD bot silently falling back to
check/fold. All three bots catch every exception in `choose_action` and `on_hand_end` and play
check/call-or-fold instead, so `errors_a == 0 and not forfeit_a` stays green. The info run's
"0 errors" had the same blind spot.

### What changed

**`3ba2594`: Make the FIELD half tests fail when a bot silently falls back to check or fold.**
In `tests/test_field.py`:
- `swallowed_exceptions()` is a context manager. It hooks `sys.monitoring` `EXCEPTION_HANDLED`
  and collects every exception caught by an except block in a FIELD bot's own code, which is
  any code object whose file is under `sim/opponents/field/`.
  - It takes a free tool id and always frees it, setting events to 0 and unregistering the
    callback.
  - During play, the only handlers in the three bot files are in `choose_action` and
    `on_hand_end`. The others run at import (where they only fire if, say, a data file is
    missing) or in the server plumbing, which the sim never calls.
  - So the file-wide filter is the reviewer's scope today, plus any internal fallback a future
    copy adds.
- `_assert_played_its_strategy(h, swallowed)`: 0 errors, no forfeit, `swallowed == []`, and
  `RAISE_HALF + RAISE_POT > 0`. Every fallback only checks, calls or folds: the bots' three
  catch-alls, field_gto's emergency mode, and our harness's error and bank-out default.
- Both half tests now use the hook and the helper:
  - `test_field_bot_plays_a_short_half_errorless`, 10 deals, preloaded bot;
  - `test_field_bot_plays_200_deals_errorless`, slow; it loads through `ZOO[name]` inside the
    hook, so import-time swallows are caught too.

**`d78414a`: Correct the FIELD results with a serial rerun at low machine load.**
The new hook plus a serial rerun showed that the committed `docs/results/field.md` overstated
v1-untuned badly. Details are below.
- `docs/results/field.md` gets a "Correction (read this first)" section, with the serial
  rerun table.
- The loaded-run sections are relabeled as superseded, and the claims the correction refutes
  are fixed in place. Both are withdrawn: field_exploit's "+3.9 is not an artifact of its
  throttle", and "no half was played on check/fold autopilot".
- The three new scripts are added verbatim.
- The Time bullet in `sim/opponents/field/README.md` now says to evaluate FIELD on a quiet
  machine and to check FIELD bank use.

### Evidence that the new guard catches what the old assertion missed

The bots are clean today, so no classic RED is possible. Instead,
`/tmp/claude-1000/-home-kchen-code-heads-up-bundle/c0d55f53-d291-4faf-88d2-40523d153176/scratchpad/red_guard.py` sabotages a fresh load and applies the old assertion and the new helper:
- field_gto's `_choose` raises, so every decision falls into its catch-all;
- field_exploit's `_classify`, which only `on_hand_end` calls, raises. That is a learning-only
  failure.

```
$ cd /home/kchen/code/heads-up-bundle-field && taskset -c 8-15 nice -n 19 /home/kchen/code/heads-up-bundle/.venv/bin/python /tmp/claude-1000/-home-kchen-code-heads-up-bundle/c0d55f53-d291-4faf-88d2-40523d153176/scratchpad/red_guard.py 2>&1 | grep -v "^\[shark\]"
field_gto with _choose raising: OLD assertion passes=True (errors 0, forfeit False); raises 0; swallowed 25, first: ["choose_action: AttributeError('simulated harness incompatibility')"]
  NEW helper: FAILED as expected at: assert swallowed == []
field_exploit with _classify raising: OLD assertion passes=True (errors 0, forfeit False); raises 23; swallowed 10, first: ["on_hand_end: AttributeError('simulated harness incompatibility')"]
  NEW helper: FAILED as expected at: assert swallowed == []
```

- **field_gto case:** the old assertion is green while the bot makes 0 raises. The raise check
  alone would also catch it.
- **field_exploit case:** the old assertion is green, and so would a raise count alone be (23
  raises). The hook catches it.

### Covering tests (GREEN): the final committed file, full output

```
$ cd /home/kchen/code/heads-up-bundle-field && taskset -c 8-15 nice -n 19 /home/kchen/code/heads-up-bundle/.venv/bin/python -m pytest -v -p no:cacheprovider tests/test_field.py
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0 -- /home/kchen/code/heads-up-bundle/.venv/bin/python
rootdir: /home/kchen/code/heads-up-bundle-field
configfile: pytest.ini
plugins: anyio-4.16.0
collecting ... collected 14 items / 3 deselected / 11 selected

tests/test_field.py::test_field_names PASSED                             [  9%]
tests/test_field.py::test_field_is_disjoint_from_train_and_holdout PASSED [ 18%]
tests/test_field.py::test_select_opponents_field_and_all PASSED          [ 27%]
tests/test_field.py::test_clis_accept_opponents_field[tournament] PASSED [ 36%]
tests/test_field.py::test_clis_accept_opponents_field[heuristic_bot] PASSED [ 45%]
tests/test_field.py::test_copies_are_unmodified_and_listed PASSED        [ 54%]
tests/test_field.py::test_field_bot_loads_its_data_file[field_equity-_PRE_EQ] PASSED [ 63%]
tests/test_field.py::test_field_bot_loads_its_data_file[field_gto-_CFR] PASSED [ 72%]
tests/test_field.py::test_field_bot_plays_a_short_half_errorless[field_equity] PASSED [ 81%]
tests/test_field.py::test_field_bot_plays_a_short_half_errorless[field_exploit] PASSED [ 90%]
tests/test_field.py::test_field_bot_plays_a_short_half_errorless[field_gto] PASSED [100%]

======================= 11 passed, 3 deselected in 1.50s =======================
```

```
$ cd /home/kchen/code/heads-up-bundle-field && taskset -c 8-15 nice -n 19 /home/kchen/code/heads-up-bundle/.venv/bin/python -m pytest -v -p no:cacheprovider -m slow tests/test_field.py
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0 -- /home/kchen/code/heads-up-bundle/.venv/bin/python
rootdir: /home/kchen/code/heads-up-bundle-field
configfile: pytest.ini
plugins: anyio-4.16.0
collecting ... collected 14 items / 11 deselected / 3 selected

tests/test_field.py::test_field_bot_plays_200_deals_errorless[field_equity] PASSED [ 33%]
tests/test_field.py::test_field_bot_plays_200_deals_errorless[field_exploit] PASSED [ 66%]
tests/test_field.py::test_field_bot_plays_200_deals_errorless[field_gto] PASSED [100%]

======================= 3 passed, 11 deselected in 2.89s =======================
```

- Neither run prints a warnings summary: both are warning-free.
- After the README edit in `d78414a`, I reran the fast file: `11 passed, 3 deselected in 2.13s`,
  with no warnings.

**Raise margin.** The bot RNGs are not fixed (field_equity seeds from the clock, field_gto from
entropy), so I measured the margin. Five fresh loads each, 10-deal smoke half, as (raises,
swallowed, errors):

```
field_equity   (raises, swallowed, errors) x5: [(11, 0, 0), (13, 0, 0), (14, 0, 0), (12, 0, 0), (11, 0, 0)]
field_exploit  (raises, swallowed, errors) x5: [(23, 0, 0), (23, 0, 0), (23, 0, 0), (23, 0, 0), (23, 0, 0)]
field_gto      (raises, swallowed, errors) x5: [(10, 0, 0), (10, 0, 0), (11, 0, 0), (10, 0, 0), (10, 0, 0)]
```

The minimum is 10 raises against a threshold of > 0.

### The info run's blind spot, and what the check found

To close the blind spot for the matchups we care about, I put the same hook on the real
matchup, each FIELD bot vs `PolicyBot()`, run serially in one process.
- **s1 with the hook** (`swallow_vs_v1.py`, 20:31-20:32, load average about 19 and falling):

```
field_equity s1 half 1: FIELD raises 1202 of 2818 actions, errors 0, bank 0.48 s, bank-out False
field_equity s1 half 2: FIELD raises 1253 of 2816 actions, errors 0, bank 0.43 s, bank-out False
field_equity s1: swallowed in FIELD code 0 []; v1 -2.019 [-2.743, -1.295] B (12 s)
field_gto s1 half 1: FIELD raises 992 of 3152 actions, errors 0, bank 1.23 s, bank-out False
field_gto s1 half 2: FIELD raises 999 of 3127 actions, errors 0, bank 0.22 s, bank-out False
field_gto s1: swallowed in FIELD code 0 []; v1 -0.654 [-1.354, +0.046] CHOP (9 s)
field_exploit s1 half 1: FIELD raises 1379 of 2887 actions, errors 0, bank 0.09 s, bank-out False
field_exploit s1 half 2: FIELD raises 1545 of 2761 actions, errors 0, bank 0.07 s, bank-out False
field_exploit s1: swallowed in FIELD code 0 []; v1 +0.328 [-0.716, +1.371] CHOP (7 s)
```

- **s1 without the hook**, as a control (`control_no_hook.py`, 20:33, load average 11):

```
field_exploit s1 no hook: v1 -0.441 [-1.479, +0.597] CHOP; FIELD bank 0.00 / 0.00 s; FIELD raises 1483 / 1580 (7 s)
field_equity s1 no hook: v1 -1.979 [-2.659, -1.299] B; FIELD bank 0.05 / 0.02 s; FIELD raises 1212 / 1250 (4 s)
```

- **All 9 matches with the hook** (`serial_rerun.py`, 20:34-20:35, load average 4.0-5.8):

```
field_equity s1: v1 -2.155 [-2.892, -1.418] B; FIELD bank 0.10 / 0.08 s, raises 1225 / 1241, errors 0, bank-out False, swallowed 0; load 4.3
field_equity s2: v1 -1.832 [-2.524, -1.140] B; FIELD bank 0.08 / 0.09 s, raises 1217 / 1215, errors 0, bank-out False, swallowed 0; load 4.5
field_equity s3: v1 -1.045 [-1.730, -0.360] B; FIELD bank 0.13 / 0.09 s, raises 1226 / 1194, errors 0, bank-out False, swallowed 0; load 4.7
field_equity pooled: -1.677 [-2.085, -1.270]
field_gto s1: v1 -1.287 [-2.011, -0.563] B; FIELD bank 0.01 / 0.01 s, raises 991 / 1023, errors 0, bank-out False, swallowed 0; load 4.9
field_gto s2: v1 -1.236 [-1.895, -0.577] B; FIELD bank 0.04 / 0.05 s, raises 1018 / 1001, errors 0, bank-out False, swallowed 0; load 5.0
field_gto s3: v1 -0.961 [-1.672, -0.250] B; FIELD bank 0.01 / 0.00 s, raises 1010 / 979, errors 0, bank-out False, swallowed 0; load 5.4
field_gto pooled: -1.161 [-1.565, -0.758]
field_exploit s1: v1 +1.022 [-0.085, +2.129] CHOP; FIELD bank 0.03 / 0.04 s, raises 1249 / 1538, errors 0, bank-out False, swallowed 0; load 5.5
field_exploit s2: v1 +1.978 [+0.877, +3.078] A; FIELD bank 0.02 / 0.03 s, raises 1146 / 1380, errors 0, bank-out False, swallowed 0; load 5.6
field_exploit s3: v1 -0.300 [-1.365, +0.764] CHOP; FIELD bank 0.02 / 0.02 s, raises 1527 / 1520, errors 0, bank-out False, swallowed 0; load 5.8
field_exploit pooled: +0.900 [+0.269, +1.530]
wall 52 s, load 4.0 -> 5.8
```

What this shows:
- **No bot falls back silently** against v1 either: 0 swallowed exceptions in all 9 serial
  matches, and 979-1,538 FIELD raises per half.
- **The hook does not change results:** s1 with / without it gave +0.33 / -0.44 vs field_exploit
  and -2.02 / -1.98 vs field_equity.
- **The headline results were wrong.** At low contention, every FIELD bot was charged at most
  0.13 s per half and played at full strength. v1-untuned then scored:
  - field_equity -1.677 [-2.085, -1.270] (3 losses);
  - field_gto -1.161 [-1.565, -0.758] (3 losses);
  - field_exploit +0.900 [+0.269, +1.530] (1 win, 2 chops).

  That is 2.0 points of 9. The 4-worker run at load average 18-20 gave 6.5. That run's
  contention degraded all three bots, and it overstated v1 by 1.2-3.0 bb/deal.

**Retractions from my first report.** Two claims in the time-tier diagnostic were wrong:
- "the +3.9 edge is robust to that" (field_exploit). My tier probe counted only its rules tier,
  not its load-driven 60-sample and lite tiers.
- "v1 ≈ 0 vs field_gto". The quiet rerun confirms it overstated v1.

field_equity was never instrumented, and it is the bot most degraded by the contention. All of
this is corrected in `docs/results/field.md`.

### Concerns

1. **For ship decisions, treat the corrected numbers as the FIELD baseline.** v1-untuned loses
   to field_equity and field_gto at full strength, and edges field_exploit.
2. **FIELD evaluations must control contention.** Run serially, or with ≤ 2 workers, on a quiet
   machine, and check the FIELD bots' bank use: at full strength each is charged at most about
   0.13 s per half. A future FIELD gate could assert that as a sanity check, but that is not
   done here.
3. **The serial rerun is a single run at one moment's load.** The bots' RNGs and time guards are
   not deterministic. The controller may want its own rerun under the T20 protocol.
