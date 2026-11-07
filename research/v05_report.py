"""Publish all revision evidence, including the initial non-perfect audit."""
from collections import Counter,defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'reports/v05';B=ROOT/'reports/v05b'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
freeze=read(B/'final_freeze.json')
final_reg=read(B/'final/registry.json')
stress_reg=read(B/'stress/registry.json')
assert len(final_reg['matches'])==20 and len(stress_reg['matches'])==12
for name,h in freeze['hashes'].items():
    assert sha256((ROOT/name).read_bytes()).hexdigest()==h,name
for p in (B/'final_tests_python312.xml',B/'final_tests_python313.xml'):
    suites=ET.parse(p).getroot()
    assert sum(int(x.get('failures','0'))+int(x.get('errors','0')) for x in suites.iter('testsuite'))==0
    assert sum(int(x.get('tests','0')) for x in suites.iter('testsuite'))==337

registers=[]
for parent in (A,B):
    for p in parent.glob('*/registry.json'):
        r=read(p)
        if r.get('method','').startswith('unchanged official judge'):
            registers.append((p,r))
rows=[]
for p,r in registers:
    for m in r['matches']:
        row={'phase':p.parent.parent.name,'run':p.parent.name,'stage':r['stage'],
             'strategy':r['strategy'],'opponent':m['opponent'],'seed':m['match_seed'],
             'deals':m['num_deals'],'bb_per_deal':m['mean_bb_per_deal'],
             'ci_low':m['ci'][0],'ci_high':m['ci'][1],'verdict':m['verdict'],
             'runtime_pass':m['runtime_pass'],'configuration_hash':m['configuration_hash']}
        for n,(s,x) in enumerate(zip(m['agent_stats']['agent1'],m['measurements']),1):
            for k in ('errors','forfeited','timed_out','time_seconds','wall_seconds','calls','skipped_calls'):
                row[f'half{n}_{k}']=s[k]
            row[f'half{n}_startup_seconds']=x['startup_seconds']
            row[f'half{n}_internal_errors']=x['memory']['internal_errors']
            for k in ('max_ms','p50_ms','p95_ms','p99_ms'):
                row[f'half{n}_{k}']=x['candidate_latency'][k]
        rows.append(row)
with (A/'all_http_results.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

selected=[]
for p,r in registers:
    if r['strategy']==freeze['strategy']:
        selected+=r['matches']
assert len(selected)==48
engineering=all(m['runtime_pass'] for m in selected)
errors=sum(s['errors'] for m in selected for s in m['agent_stats']['agent1'])
internal=sum(x['memory']['internal_errors'] for m in selected for x in m['measurements'])
maxcharge=max(s['time_seconds'] for m in selected for s in m['agent_stats']['agent1'])
maxpost=max(x['candidate_latency']['max_ms'] for m in selected for x in m['measurements'])
maxstart=max(x['startup_seconds'] for m in selected for x in m['measurements'])
outcomes=Counter(m['verdict'] for m in final_reg['matches'])
perfect=outcomes['AGENT1_WIN']==20
summary={'strategy':freeze['strategy'],'parameters':freeze['parameters'],'mode':'FULL_PAIR_v04',
         'final_matches':20,'final_wins':outcomes['AGENT1_WIN'],'final_chops':outcomes['CHOP'],
         'final_losses':outcomes['AGENT2_WIN'],'observed_final_match_win_percent':outcomes['AGENT1_WIN']/20*100,
         'engineering_pass':engineering,'all_wins_audit_target_met':perfect,
         'selected_full_http_matches':len(selected),'selected_hands':len(selected)*2000,
         'selected_errors':errors,'selected_internal_errors':internal,
         'selected_max_charged_seconds':maxcharge,'selected_max_post_ms':maxpost,
         'selected_max_startup_seconds':maxstart,'all_attempted_http_matches':len(rows),
         'prior_phase_a_audit':{'wins':19,'chops':1,'losses':0},
         'future_win_guarantee':False}
(A/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')

selectionA=read(A/'selection.json');selectionB=read(B/'selection.json')
lines=['# Current v0.5 release evidence','',
       f'**Latest frozen audit: {outcomes["AGENT1_WIN"]} wins, {outcomes["CHOP"]} chops, {outcomes["AGENT2_WIN"]} losses in 20 full duplicate matches. Observed match win rate: {outcomes["AGENT1_WIN"]/20:.0%}.**','',
       f'Deployment: `{freeze["strategy"]}`, FULL_PAIR_v04, adaptation on, pure bluffs off, salt 0. Branch `licheng`; participant/bot `3XKO`. Engineering gates: {"PASS" if engineering else "FAIL"}.','',
       'This is an observed audit result, not a guarantee that every future match or every individual hand will be won. All matches use the official 1,000-deal duplicate format (2,000 hands), including its confidence-interval verdict. The tournament form was not submitted.','',
       '## What changed','',
       '- Prioritize demonstrated aggression over the calling-station label, using the original evidence thresholds. AlwaysPot was previously liable to become STATION merely because it never folds.',
       '- Against confirmed aggressive opponents, let MEDIUM/STRONG hands call ordinary pot prices without the contradictory pot/repeated-raise deductions.',
       '- Defend preflop scores >=20 against confirmed aggression; use calls and button limps to avoid unnecessary recursive reraising. Lower scores and nonaggressive price policies retain their original guards.',
       '- Keep the exact evaluator, feature score, sampler, draws, nuts proofs, legal guards, protocol and per-half memory contracts unchanged. No opponent identity, hidden cards, match seed or baseline RNG state is used by tournament logic.','',
       'See [STRATEGY.md](STRATEGY.md) and [REQUIREMENTS.md](REQUIREMENTS.md).','',
       '## Honest experiment history','',
       'The original v0.4 bot lost to RandomBot and AlwaysPot. Phase A isolated price, preflop-defense, trapping and profile-priority changes on new development seeds. Its selected score-30 policy won all 16 development/confirmation HTTP matches, but its sealed audit was **19 wins, 1 chop, 0 losses**. The non-perfect audit is retained in [the phase-A registry](final/registry.json); it was not overwritten or rerun.',
       f'Phase A improved the matched original-control development score by +4 match points and +{selectionA["candidates"]["v05_aggressive_limp"]["mean_delta_bb"]:.4f} mean paired BB/deal, with no opponent point regression.',
       '','Phase B was preregistered after that audit. It tested only the conditional preflop defense score, holding all other changes fixed, on new development/confirmation seeds. Because both policies were already winning all development matches, its promotion criterion required a BB improvement and no regression instead of an impossible increase beyond the match-point ceiling.',
       f'The selected score-20 revision won all 16 phase-B development/confirmation matches and added +{selectionB["mean_delta_bb"]:.4f} mean paired BB/deal versus score 30. Per-opponent changes were '+', '.join(f'{k}: {v:+.4f}' for k,v in selectionB['per_opponent_delta_bb'].items())+' BB/deal. These are development screening results, with per-match delta confidence intervals/variance disclosed in the selection files, not proof of population-wide superiority.',
       '','Only after promotion and a new source/configuration freeze did the final runner open five new audit seeds. No phase-A final seeds were reused, and no parameter was changed after reading the new final outcomes.',
       '','Evidence: [phase-A plan](experiment_plan.json), [phase-A selection](selection.json), [phase-B plan](../v05b/experiment_plan.json), [phase-B selection](../v05b/selection.json), [new final freeze](../v05b/final_freeze.json).','',
       'One rejected finalist (`v05_aggressive30`) had a 102.53 ms request during parallel development confirmation, exceeding the 100 ms acceptance margin despite zero referee errors and all match wins. That failed guard remains in its raw registry and was not hidden. The deployed policy passed its own complete checks.','',
       '## Latest untouched final audit','',
       '| Opponent | Wins / matches | Mean paired BB/deal across five matches | Lowest per-match CI bound |',
       '|---|---:|---:|---:|']
groups=defaultdict(list)
for m in final_reg['matches']:groups[m['opponent']].append(m)
for opponent,ms in groups.items():
    lines.append(f'| {opponent} | {sum(m["verdict"]=="AGENT1_WIN" for m in ms)} / {len(ms)} | {statistics.mean(m["mean_bb_per_deal"] for m in ms):.4f} | {min(m["ci"][0] for m in ms):.4f} |')
lines+=['','Every row below is one independent 1,000-deal duplicate match.','',
        '| Opponent | Seed | BB/deal | 95% paired CI | Verdict | Charged s, halves 1 / 2 | Max POST ms, halves 1 / 2 |',
        '|---|---|---:|---|---|---|---|']
for m in final_reg['matches']:
    charge=' / '.join(f'{s["time_seconds"]:.3f}' for s in m['agent_stats']['agent1'])
    req=' / '.join(f'{x["candidate_latency"]["max_ms"]:.2f}' for x in m['measurements'])
    lines.append(f'| {m["opponent"]} | {m["match_seed"]} | {m["mean_bb_per_deal"]:.4f} | [{m["ci"][0]:.4f}, {m["ci"][1]:.4f}] | {m["verdict"]} | {charge} | {req} |')
lines+=['','Raw final results: [registry](../v05b/final/registry.json). All development, stress and both audits are mapped to [all_http_results.csv](all_http_results.csv), including per-half errors, forfeits, time-bank status, startup, POST percentiles and source/configuration hashes.','',
        '## Correctness and runtime','',
        '- **337 tests pass on Python 3.12 and 3.13**, including the original 305 tests and new aggression, price-boundary, conditional trapping/limping and defense-floor cases.',
        '- Three revised configurations each undergo >=10,000 Game-driven decision checks, exercising every street, aggressive profiles and raise caps. Legacy scoring/fragment/draw/memory tests remain intact; the old policy is an explicit reproducible control.',
        '- Independent direct-hook screens and official HTTP results match all 24 compared training matches / 24,000 paired outcomes exactly. Direct screens are clearly labelled as screening, not runtime qualification.',
        '- The unchanged official Docker build and restricted smoke test pass. No runtime dependencies or referee-controlled files were added/modified.',
        f'- The deployed strategy completed {len(selected)} full HTTP matches / {len(selected)*2000:,} hands across development, confirmation, serialized stress and final audit: {errors} referee errors, {internal} internal errors, no illegal-action reports, no forfeits and no time-bank exhaustion.',
        f'- Across those runs: maximum charged time {maxcharge:.3f} s/half (acceptance <=5.5; official bank 8), maximum complete POST {maxpost:.2f} ms (acceptance <=100; official timeout 1,000), maximum startup {maxstart:.2f} s (acceptance <=12; official limit 15).',
        '- Qualification containers enforce one CPU, 4 GiB RAM, non-root UID, read-only root, writable `/tmp`, no external network, dropped capabilities and PID limit 256. Resource/image snapshots are retained beside the phase-B report.',
        '','Evidence: [Python 3.12 tests](../v05b/final_tests_python312.xml), [Python 3.13 tests](../v05b/final_tests_python313.xml), [official build](../v05b/official_build.log), [source verification](../v05b/source_verification.json), [runtime stress](../v05b/stress/registry.json), [screen/HTTP cross-check](screen_validation.json).','',
        '## Limits and not-run items','',
        '- A finite winning audit cannot establish a 100% future win probability. Different opponents, action-conditioned ranges and card variance can produce losses or chops.',
        '- The price ceilings remain heuristic risk policies. Fragment risk is not five-card equity, and current strength is not guaranteed final showdown strength.',
        '- Python 3.10: NOT RUN because it is not installed; available Python 3.12/3.13 were tested.',
        '- Custom tight-value/overfolder opponents and real tournament field: NOT RUN. Pure bluffs remain disabled. The measured claim covers the four official baselines.',
        '- Organizer-hosted build/tournament/form: NOT RUN; requires registration and submission. No form was submitted.',
        '- This revision was made and published on `licheng`, preserving the registered identity and public-repository workflow. Use the final response\'s full pushed commit SHA when submitting.']
(A/'RELEASE.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(B/'RELEASE.md').write_text('# Final refinement evidence\n\nSee the [combined current report](../v05/RELEASE.md). This directory retains the separately preregistered phase-B plan, selection, freeze, raw qualification, tests, and final audit.\n',encoding='utf8')
print(json.dumps(summary,indent=2))
