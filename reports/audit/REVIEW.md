# Contest-guide code audit

Reviewed against the supplied **Heads Up Participant Guide.pdf**, particularly
pages 3–4 (participant-owned code), 7–9 (protocol and visible information),
10 (timeouts and fallback actions), and 11–12 (execution limits and testing).

The changes retain the registered identity, HTTP protocol, strategy parameters,
runtime dependencies, and short-deck Omaha scoring rules. Organizer-owned files
and the original supplied tests are unchanged. Historical design and qualification
artifacts remain intact; their frozen-source claims apply to their original code.

## Corrected defects

| Area | Defect | Correction |
|---|---|---|
| Observation transactions | A malformed first observation reset the previous hand before validation. A changed board could update opponent counters before being rejected. | Validate history, cards already observed, position, and street progression in scratch state, then commit together. |
| Opponent responses | The bot assumed its proposed raise was executed. After a referee timeout/default substitution it could count an opponent bet as a response to a nonexistent raise. | Verify the executed action's owner, street, amount, and total before recording a response. |
| Retried decisions | Replacing a proposed raise with a check/call left the old pending raise in memory. | Clear a superseded proposal for the same decision. |
| Raise guards | A purported raise could merely match the opponent's bet, fall below the minimum, or exceed the pot limit. | Enforce a positive raise increment, available stacks, pot limit, and minimum, with the legal short-all-in exception. |
| Chip/context validation | Impossible chip totals, mismatched blinds, invalid position values, and completed-hand observations were accepted. | Check configured stacks, conservation of chips, physical hand context, and completion status. |
| Terminal transactions | Impossible pot sizes and payouts could consume a hand's deduplication slot, preventing a subsequent correct result from being recorded. | Validate stack bounds and equal-stack showdown/fold payouts before committing or fingerprinting a result. |
| Evaluation helpers | Duplicate, overlapping, negative, or boolean cards could yield incorrect features or incidental lookup errors. Mixed list/tuple inputs failed during concatenation. Invalid sampler salts were accepted or raised incidental errors. | Normalize card sequences and validate physical cards, board sizes, hand categories, and the unsigned 32-bit salt domain. |
| HTTP research runner | Inherited `AGENT_TOKEN` was not sent by referee handles; empty latency lists crashed; a slow child could prevent cleanup. | Forward the token, define empty metrics, and kill children that exceed the termination grace period. |
| Test artifacts | Reachability tests overwrote archived qualification reports and depended on writable report directories. | Write coverage diagnostics to pytest temporary directories. |
| Current configuration validation | Existing experiment scripts defaulted to historical configurations and required historical freeze files. | Add `tools/validate_bot.py`, which reads the deployed settings from `agent.py`, starts fresh servers each half, and reserves a new report directory. Historical experiment workflows remain available. |

The initial 337-test suite passed before editing. Of the first 40 new regression
cases, 37 failed on the original code. The completed revision adds 47 regression
cases. These include physically valid Game-generated betting states and explicit
invalid-input cases; they do not weaken the supplied starter tests.

## Verification

- **384 tests pass on Python 3.13**: [JUnit](tests_python313.xml).
- **384 tests pass on Python 3.12** in an offline, read-only, one-CPU container:
  [JUnit](tests_python312.xml).
- The unchanged official build and restricted startup/action smoke test pass:
  [build log](build.log).
- Official rules, evaluator, judge, server plumbing, dependencies, Dockerfiles,
  build tools, original tests, and supplied design manifest match the pinned
  upstream source: [source verification](source_verification.json).
- The submission image adds **54,180 bytes** above the local official base,
  below the guide's 3 GiB allowance: [image sizes](images.json).

The test suite also retains the exhaustive preflop distribution check, independent
Omaha scoring and fragment references, runout checks, and more than 40,000
Game-driven decision checks across the evaluator/strategy configurations.

The first HTTP batch is retained under [matches/](matches/registry.json). It passed
all four baseline matches, but used a writable output mount to collect reports.
The final exported batch passes **four full duplicate matches / 8,000 hands**
against the four supplied baselines, with fresh processes each half and token
authentication enabled. There are **zero referee errors, internal errors,
forfeits, skipped decisions, or time-bank exhaustions**.

| Opponent | Verdict | Mean BB/deal | 95% paired interval |
|---|---|---:|---|
| baseline_random | Win | 2.8305 | [1.7451, 3.9159] |
| baseline_calling_station | Win | 3.2295 | [2.8171, 3.6419] |
| baseline_always_half | Win | 3.3985 | [2.2034, 4.5936] |
| baseline_always_pot | Win | 2.8425 | [1.2219, 4.4631] |

Maximum charged time is **0.893 seconds/half** (guide: 8 seconds); maximum
complete POST is **33.77 ms** (guide: 1,000 ms); maximum startup is
**1.52 seconds** (guide: 15 seconds).

The container enforces one CPU, 4 GiB RAM, UID 10001, no network, a read-only
root and repository, and `/tmp` as its only writable location (512 MiB).
Reports were exported before stopping the container. One earlier strict run
was repeated because Docker cleared its tmpfs reports at shutdown; repeated
runs use the same source and seed and are not independent strategy evidence.

Evidence: [qualified registry](qualified_matches/registry.json),
[container restrictions](qualified_container.json), [summary](summary.json).

## Reproducing the review

```powershell
.venv/Scripts/python -m pytest -q --basetemp=.work/pytest
.venv/Scripts/python research/check_sources.py --output reports/practice-source-verification.json
.venv/Scripts/python tools/validate_bot.py --output reports/practice-matches
```

Use a new match output directory. Run `tools/check_build.sh` in Git Bash or Linux
for the official build/smoke checks. For resource-qualified full matches, mount the
repository read-only in the built image, enforce the same runtime restrictions as
the build script, run `/repo/tools/validate_bot.py --output /tmp/matches` through
`docker exec` in a running container, and export the output through `docker exec`
before stopping it. Docker's archive API may not see tmpfs contents, and Docker
clears tmpfs at shutdown. The validation tool needs the repository's
research helpers; those helpers are excluded from the submission image.

## Limits

The strategy still uses visible-board fragment risk and heuristic betting
thresholds. Fragment risk is not five-card opponent equity or future-runout equity.
This review fixes identified engineering defects; it does not establish optimal
poker strategy or guarantee future tournament wins. The new match sample covers
the four supplied baselines on one duplicate-match seed. The actual tournament
field and organizer-hosted build were not tested here.
