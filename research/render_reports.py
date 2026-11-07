"""Render release evidence from immutable run registries, without tuning."""
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'
def read(path):
    return json.loads((OUT/path).read_text(encoding='utf-8-sig'))

registries={name:read(name+'/registry.json') for name in ('E0','dev_A','dev_B','dev_C','final')}
assert len(registries['E0']['matches'])==24
assert all(len(registries[k]['matches'])==8 for k in ('dev_A','dev_B','dev_C','final'))
all_matches=[m for r in registries.values() for m in r['matches']]
assert all(m['num_deals']==1000 and m['runtime_pass'] for m in all_matches)
assert all(s['errors']==0 and not s['forfeited'] and not s['timed_out'] for m in all_matches for s in m['agent_stats']['agent1'])
selection=read('policy_selection.json')
freeze=read('final_freeze.json')
accuracy=read('accuracy.json')
lines=[]
rows=[]
for stage,registry in registries.items():
    for m in registry['matches']:
        row={'stage':stage,'mode':m['mode'],'opponent':m['opponent'],'seed':m['match_seed'],
             'deals':m['num_deals'],'mean_paired_bb_per_deal':m['mean_bb_per_deal'],
             'ci_low':m['ci'][0],'ci_high':m['ci'][1],'verdict':m['verdict'],
             'points':m['points']['agent1'],'configuration_hash':m['configuration_hash'],
             'early_100_paired_mean':statistics.mean(m['paired_bb'][:100]),
             'last_500_paired_mean':statistics.mean(m['paired_bb'][-500:])}
        for i,(s,x) in enumerate(zip(m['agent_stats']['agent1'],m['measurements']),1):
            for k in ('errors','forfeited','timed_out','calls','skipped_calls','time_seconds','wall_seconds'):
                row[f'half{i}_{k}']=s[k]
            row[f'half{i}_startup_seconds']=x['startup_seconds']
            for k in ('p50_ms','p95_ms','p99_ms','max_ms'):
                row[f'half{i}_{k}']=x['candidate_latency'][k]
            row[f'half{i}_internal_errors']=x['memory']['internal_errors']
            row[f'half{i}_observed_first_actions']=x['memory']['first_observed']
            row[f'half{i}_completed_hands']=x['memory']['completed']
        rows.append(row)
with (OUT/'match_results.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

e0=registries['E0']['matches']
final=registries['final']['matches']
points=sum(m['points']['agent1'] for m in final)
verdicts=Counter(m['verdict'] for m in final)
overall_errors=sum(s['errors'] for m in all_matches for s in m['agent_stats']['agent1'])
max_charged=max(s['time_seconds'] for m in all_matches for s in m['agent_stats']['agent1'])
max_ms=max(x['candidate_latency']['max_ms'] for m in all_matches for x in m['measurements'])
lines += ['# Release evidence', '',
          '**Engineering qualification: PASS. Competitive performance: mixed; this is an experimental submission candidate, not a demonstrated strong tournament bot.**', '',
          f'Identity: `3XKO / 3XKO`. Branch: `licheng`. Mode: `{freeze["mode"]}`; salt 0; adaptation enabled; pure bluffs disabled (development policy B).', '',
          'The participant guide requires team registration and a public repository/full pushed commit SHA. The repository is public; registration was not verified and no competition form was submitted.', '',
          f'Completed **{len(all_matches)} full duplicate matches / {len(all_matches)*2000:,} hands** across E0, development, and the one-time final audit. Candidate errors: **{overall_errors}**; illegal-action reports, internal hook errors, forfeits, and time-bank exhaustion: **0**. No strategic parameter was changed after viewing final results.', '',
          '## Correctness and official build', '',
          '- 305 tests passed on Windows Python 3.13 and on Linux Python 3.12 in an offline, read-only, one-CPU, 4-GB test container. Final frozen flags also passed all 305 Windows tests.',
          '- Includes 10,000 distinct complete river comparisons, 1,000 independent fragment references, all 376,992 preflop holdings, 100 runout-enumerated draw states, 100 malformed observations, and deterministic action/memory matrices.',
          '- The official Game-driven generator checked 10,006 decisions: 4,266 preflop, 3,485 flop, 1,657 turn, 598 river; 437 at the raise cap. Both seats and all four actions were covered.',
          '- Unchanged official `tools/check_build.sh` passed for both the initial and final configuration, including the restricted smoke test. Final smoke startup was approximately 0.8 seconds.',
          '- Official game/evaluator/judge and server plumbing are unchanged; the supplied design manifest passes. Source commit: `c83ab6eeb7660a2fd1fae7df224315d41fe985a9`.', '',
          'Evidence: [final tests](final_correctness.xml), [Python 3.12 tests](correctness_python312.xml), [final build](final_build.log), [source verification](final_source_verification.json), [coverage](reachability.json).', '',
          '## Evaluator qualification', '',
          'E0 ran both modes independently: six opponents (four official baselines plus behaviorally identical call-heavy/max-raise aliases), two stress seeds, both 1,000-hand halves, fresh HTTP servers per half. Both passed before mode selection.', '',
          '| Mode | Matches / halves | Max charged s/half | Max request ms | Max startup s | E0 | E1 |',
          '|---|---:|---:|---:|---:|---|---|']
for mode in ('FULL_PAIR_v04','REDUCED_64_v04'):
    ms=[m for m in e0 if m['mode']==mode]
    charge=max(s['time_seconds'] for m in ms for s in m['agent_stats']['agent1'])
    req=max(x['candidate_latency']['max_ms'] for m in ms for x in m['measurements'])
    startup=max(x['startup_seconds'] for m in ms for x in m['measurements'])
    lines.append(f'| {mode} | {len(ms)} / {2*len(ms)} | {charge:.3f} | {req:.2f} | {startup:.2f} | PASS | PASS |')
lines += ['',
          'Resource controls: one CPU quota, 4 GiB RAM, no external network, non-root UID 10001, read-only root, writable `/tmp`, dropped capabilities, and PID limit 256. Captured cgroup peak during E0 was 167,211,008 bytes (about 159.5 MiB) for judge and both bots together. The final official size check passed; the initial bot image increment was only 50,937 bytes.', '',
          f'Across all E0/development/final runs, maximum charged time was {max_charged:.3f} s/half and maximum measured complete POST latency was {max_ms:.2f} ms, below the 5.5 s / 100 ms acceptance margins and 8 s / 1 s official limits. Per-half percentiles, startup, charged time and errors are in [match_results.csv](match_results.csv).', '',
          'E1 used 480 fresh states (160 per street), including 102 FULL STRONG, 143 MEDIUM, 219 WEAK, and 16 NUTS. Coverage included 273 paired boards, 120 trips boards, 140 suited boards, 60 states near .03 and 37 near .12.', '',
          '- FULL severe STRONG mislabels: 0/102; Wilson 95% upper 3.63%. Mean exact current-five-card stronger risk in STRONG was 9.81%, versus 42.79% MEDIUM and 76.64% WEAK.',
          '- REDUCED severe STRONG mislabels: 0/100; Wilson upper 3.70%. Upward tier disagreement: 24/480 = 5.00% (Wilson upper 7.33%); downward: 67/480 = 13.96%.',
          '- Action disagreement: 44/1,004 = 4.38% on reachable postflop contexts with fixed nonmode inputs. Salts 1/2/3 disagreed in 46/50/41 contexts respectively. Salts were diagnostic only.',
          '- All six sampler fixtures pass on Python 3.12 and 3.13. Python 3.10 was NOT RUN because it was not installed; the acceptance requirement permits available runtimes.',
          '- All 240 reconstructed historical states pass official packed-score, pair-count, five-card-beat and tie-only checks. Independent slow direct enumeration agrees on one state per street for both historical and fresh sets.',
          '- Kings-full regression: exactly 13/325 beating fragments and 17,156/65,780 beating complete holdings. Salt 0 is MEDIUM; diagnostic salt 1 labels it STRONG, illustrating sampling risk.', '',
          'Fragment frequency, uniform current-five-card threat and final showdown equity are distinct. Calibration does not establish betting EV against real ranges. See [accuracy](accuracy.json), [action comparisons](action_comparison.csv), [salt diagnostics](salt_diagnostics.csv), [historical verification](pilot_official.csv), [fresh verification](fresh_official.csv), and [ordered freeze](freeze.json).', '',
          '## Development selection', '',
          '| Variant | Adaptation | Pure bluffs | Points / 8 | Mean paired BB/deal across match means |',
          '|---|---|---|---:|---:|']
for key,adapt,bluff in (('dev_A','off','off'),('dev_B','on','off'),('dev_C','on','on')):
    ms=registries[key]['matches']
    lines.append(f'| {key[-1]} | {adapt} | {bluff} | {sum(m["points"]["agent1"] for m in ms):.1f} | {statistics.mean(m["mean_bb_per_deal"] for m in ms):.4f} |')
lines += ['',
          'B met the prespecified promotion screen: +0.5 match points, +0.3494375 mean paired BB/deal versus A, and no opponent point regression. C was identical to B and was not promoted. All numeric preflop, tier, call and sizing thresholds remain v0.4 values. No E2/E3 threshold alternatives were proposed or run.', '',
          'Per-match candidate-minus-baseline means, variances and confidence intervals are in [policy_selection.json](policy_selection.json). The mean improvement averages these four selected opponents; it is not evidence about the unknown tournament field. Early 100 and final 500 paired-outcome diagnostics are included in the CSV; small-window results are not match verdicts.', '',
          '## One-time final audit', '',
          f'Frozen before access to final seeds: [final_freeze.json](final_freeze.json). Final score: **{points:.1f}/8 points**, {verdicts["AGENT1_WIN"]} wins, {verdicts["CHOP"]} chops, {verdicts["AGENT2_WIN"]} losses.', '',
          '| Opponent | Seed | Paired BB/deal | 95% CI | Verdict | Charged s, halves 1 / 2 | Max POST ms, halves 1 / 2 |',
          '|---|---|---:|---|---|---|---|']
for m in final:
    charges=' / '.join(f'{s["time_seconds"]:.3f}' for s in m['agent_stats']['agent1'])
    requests=' / '.join(f'{x["candidate_latency"]["max_ms"]:.2f}' for x in m['measurements'])
    verdict={'AGENT1_WIN':'WIN','AGENT2_WIN':'LOSS','CHOP':'CHOP'}[m['verdict']]
    lines.append(f'| {m["opponent"]} | {m["match_seed"]} | {m["mean_bb_per_deal"]:.4f} | [{m["ci"][0]:.4f}, {m["ci"][1]:.4f}] | {verdict} | {charges} | {requests} |')
lines += ['',
          '**The bot loses to RandomBot and AlwaysPot on the sealed audit.** Its conservative call ceilings can fold strong current hands to pot-sized aggression. This is a measured strategic limitation, not an engineering failure. The audit was not used to change the selected policy or thresholds. No claim of overall superiority, GTO play or tournament-winning strength is supported.', '',
          '## Not run and remaining submission actions', '',
          '- Optional alternative opening/call/sizing threshold experiments (E2/E3): NOT RUN; coefficients were kept frozen and the controlled E4 variants supplied the selection evidence.',
          '- Optional tight-value and overfolder archetypes: NOT RUN; the required official suite and timing stress opponents were completed. Pure-bluff exploitation of a deliberate overfolder therefore remains unvalidated and is disabled.',
          '- Python 3.10 fixture reproduction: NOT RUN; runtime not installed. Python 3.12 and 3.13 passed.',
          '- Remote organizer build and actual tournament: NOT RUN; require organizer submission. Registration status was not checked. No submission form was sent.',
          '- Use the full pushed commit SHA from the final response, verify the registered team name `3XKO`, and submit the public repository URL plus SHA if you choose to enter.', '',
          'The complete implementation, reproducible evidence and requirement mapping are committed on `licheng`. See [REQUIREMENTS.md](REQUIREMENTS.md) and [IMPLEMENTATION.md](IMPLEMENTATION.md).']
(OUT/'RELEASE.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

summary={'full_duplicate_matches':len(all_matches),'hands':len(all_matches)*2000,
         'candidate_errors':overall_errors,'max_charged_seconds':max_charged,'max_post_ms':max_ms,
         'final_points':points,'final_verdicts':dict(verdicts),'mode':freeze['mode'],
         'adaptation':freeze['adaptation'],'pure_bluffs':freeze['pure_bluffs'],
         'engineering_qualification':'PASS','competitive_strength':'MIXED_UNDERPERFORMS_RANDOM_AND_POT'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))
