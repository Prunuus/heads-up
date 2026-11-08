# Remaining-defect review

Reviewed the supplied **Heads Up Participant Guide.pdf** against checkout
`a480ca5`, which already contained the [previous audit](../audit/REVIEW.md).
The initial suite passed all 384 tests. This review found four additional defects.

Changes follow the participant-owned code boundary in pages 3-4 and 7-9 of
the guide. The organizer's game, evaluator, judge, original tests, Dockerfiles,
and server plumbing remain unchanged. Strategy thresholds, registered identity,
runtime requirements, and preserved design artifacts also remain unchanged.

## Corrections

| Defect | Correction |
|---|---|
| A pending proposed river/all-in raise was treated as confirmed whenever the hand reached showdown. A substituted check, call, or smaller wager could create a fictitious opponent response. | Store the pot expected after an immediate call. Infer a terminal nonfold response only when that pot matches and the response would end betting. Directly observed action-history confirmation remains available. |
| A terminal result could imply that a player had invested fewer chips than already observed. Such a result could consume the hand's deduplication slot and block the later correct result. | Preserve the latest observed contributions for the current hand. Reconstruct terminal contributions from the payout and reject decreasing investments before committing any counters or deduplication state. |
| The HTTP runner used fixed ports and trusted a healthy endpoint even if its newly launched child had exited. An existing server could be mistaken for the intended bot. | Select two distinct available local ports for each half and verify both child processes are alive before starting the referee. |
| Validation could hash the mounted checkout while subprocesses executed the container's older `/app` files. Results could be attributed to code that was never executed. | Compare checkout hashes against every executable source in the existing qualification manifest before reserving an output directory or starting matches. Reject mismatches with a rebuild instruction. |

Twelve regression cases were added. Ten failed before their corresponding fixes;
two positive river-call cases preserve valid terminal response attribution.
Poker transaction fixtures use the unchanged official `Game` state machine.
The runner tests simulate failed child launches and mismatched container sources.

## Verification

- **396 tests pass on Python 3.13:** [JUnit](tests_python313.xml).
- **396 tests pass on Python 3.12:** [JUnit](tests_python312.xml), executed offline
  with a read-only root and repository, one CPU, 4 GiB RAM, and UID 10001.
- The unchanged official build and restricted startup/action smoke checks pass:
  [build log](build.log).
- Protected official sources, server plumbing, and the preserved design manifest
  match the pinned upstream checkout: [source verification](source_verification.json).
- The submission image adds **54,796 bytes** above the local official base,
  below the guide's 3 GiB allowance: [images](images.json).

All **12 full duplicate matches / 24,000 hands** passed, with wins against every
baseline on all three seeds. There were **zero referee errors, internal errors,
forfeits, skipped decisions, or time-bank exhaustions**. Maximum charged thinking
time was **0.502 seconds per half** (guide: 8 seconds); maximum complete POST was
**33.26 ms** (guide: 1,000 ms); maximum startup was **1.53 seconds** (guide: 15 seconds).
Executable source hashes match the current checkout.
Full-match results and timing measurements are recorded in [summary.json](summary.json).
The three registered seeds are `corrective-a-20261107`,
`corrective-b-20261107`, and `corrective-c-20261107`, with all four supplied
opponents for each seed. Each half starts fresh processes. Token authentication
is enabled. Raw registries, half results, and bounded diagnostics are preserved
under `matches-a/`, `matches-b/`, and `matches-c/`.

The runtime container has no external network, a read-only root and repository,
one CPU, 4 GiB RAM, UID 10001, dropped capabilities, and no privilege escalation.
Only `/tmp` is writable, with a 512 MiB limit. Reports were exported before
removing the container. The separate test image adds test dependencies above the
verified submission image; they are absent from the submission image.
See [container configuration](container.json) and [execution log](container-run.log).

## Reproduction and limits

```powershell
.venv/Scripts/python -m pytest -q --basetemp=.work/pytest
.venv/Scripts/python research/check_sources.py --output reports/practice-sources.json
.venv/Scripts/python tools/validate_bot.py --output reports/practice-new --seed practice-new
```

Run `tools/check_build.sh` in Git Bash or Linux for the official build check.
For resource-qualified matches, run validation inside the built container with
the restrictions above, a read-only repository mount, and output under `/tmp`.
The image's executable files must match the checkout being qualified.
Use a new output directory per seed.

The fixes address demonstrated engineering defects. The strategy remains
heuristic; fragment threat is not five-card opponent equity or final showdown
equity. These baseline matches do not establish optimal strategy or performance
against the tournament field. Organizer-hosted execution and submission were
outside this review. Earlier frozen reports apply to their original sources.
