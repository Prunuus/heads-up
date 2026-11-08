# History validation and qualification audit

Reviewed checkout `a92922a` against the supplied **Heads Up Participant Guide.pdf**.
The initial 396-test suite passed. Changes stay within participant-owned modules
and development tooling, as required by pages 3-4 and 7-9. The organizer's rules,
evaluator, judge, baseline bots, original tests, Dockerfiles, and HTTP server
plumbing are unchanged. Registered identity, strategy thresholds, dependencies,
and the preserved design artifacts are also unchanged.

## Corrections

| Defect | Correction |
|---|---|
| Observation validation conserved the combined pot and stacks but accepted action amounts, street totals, raise counts, and individual investments that contradicted the visible history. These could change the opponent model or erase previously observed contributions. | Reconcile implicit blinds and every visible contribution with each player's street bet, remaining stack, pot, and raise count before committing any memory. Reject calls with the wrong amount, terminal folds in active histories, and overspent stacks. Invalid observations retain the existing legal fallback. |
| The qualification gate inspected only the candidate's referee statistics. It could pass a match against an opponent that errored, forfeited, or exhausted its time bank. It also did not verify terminal-result delivery. | Require clean statistics and zero skipped decisions for both bots, the requested hand count, and one processed terminal result per hand. Preserve the existing candidate latency and time-bank margins. |
| Executable source checks omitted the baseline opponents, and validation harness revisions were absent from registries. | Include all four baseline sources in image/checkout comparisons and source hashes. Record the validation CLI, match runner, and candidate launcher hashes separately. |
| Historical HTTP runners created their registry only after the first complete match. An early startup failure left the run unreserved. | Write the registry before launching any bot, so reruns cannot silently overwrite a failed attempt. |
| Historical HTTP runner commands exited successfully when runtime qualification failed. | Return a failing exit status for failed or empty qualifications, allowing callers to stop subsequent stages. |

Twenty regression cases were added, including real `Game` observations with
deliberately inconsistent histories and simulated failed qualifications. Fourteen
failure cases were explicitly reproduced before their fixes. Existing reachable
state tests continue to exercise tens of thousands of decisions.

## Verification

- **416 tests pass on Python 3.13:** [JUnit](tests_python313.xml).
- **416 tests pass on Python 3.12:** [JUnit](tests_python312.xml), offline with
  one CPU, 4 GiB RAM, non-root UID 10001, and a read-only repository/root.
- The unchanged official Docker build and restricted smoke test pass:
  [build log](build.log).
- Protected official sources, server plumbing, and the design manifest match
  the pinned upstream checkout: [source verification](source_verification.json).

All **12 complete duplicate matches / 24,000 hands** passed, with wins against
all four baselines on all three seeds. Both bots had **zero referee errors,
forfeits, skipped decisions, or time-bank exhaustions**. The candidate had zero
internal errors and processed all 24,000 terminal results. Maximum charged time
was **0.449 seconds per half** (guide: 8 seconds); maximum complete POST was
**26.07 ms** (guide: 1,000 ms); maximum startup was **1.52 seconds** (guide:
15 seconds). Executable and harness hashes match the final checkout.

Full-match evidence and resource measurements are recorded in
[summary.json](summary.json). Image sizes are recorded in [images.json](images.json),
and runtime restrictions in [container.json](container.json). The submission image
increment is below the guide's 3 GiB allowance. Each half starts fresh processes, with token
authentication enabled. The independent seeds are
`history-integrity-a-20261107`, `history-integrity-b-20261107`, and
`history-integrity-c-20261107`; historical final seeds were not reused.

## Reproduction and limits

```powershell
.venv/Scripts/python -m pytest -q --basetemp=.work/pytest
.venv/Scripts/python research/check_sources.py --output reports/practice-sources.json
.venv/Scripts/python tools/validate_bot.py --output reports/practice-new --seed practice-new
```

Run `tools/check_build.sh` in Git Bash or Linux for the official build check.
For resource-qualified matches, run validation inside the current image with
one CPU, 4 GiB RAM, no external network, read-only root and repository, UID
10001, and writable `/tmp` limited to 512 MiB. The executable image files must
match the checkout being qualified. Use a new output directory for each run.
The separate test image adds only development test dependencies above the
verified submission image; they are absent from the submission image.

These checks address demonstrated correctness and qualification defects.
The strategy remains heuristic, and baseline wins do not establish optimal
play or tournament-field performance. Organizer-hosted execution and contest
form submission are outside this audit. Earlier frozen reports apply to their
original source revisions.
