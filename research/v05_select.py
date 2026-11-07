"""Select solely from registered development/confirmation HTTP evidence."""
from collections import defaultdict
from datetime import datetime,timezone
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/v05'
plan=json.loads((OUT/'experiment_plan.json').read_text())
target=OUT/'final_freeze.json'
if target.exists():
    raise RuntimeError('Refusing to replace a final freeze')


def load(path):
    r=json.loads(path.read_text())
    assert len(r['matches'])==8 and r['deals']==1000
    assert r['stage']!='final'
    assert all(s['errors']==0 and not s['forfeited'] and not s['timed_out']
               for m in r['matches'] for s in m['agent_stats']['agent1'])
    return r


control=load(OUT/'http_dev_v04/registry.json')
baseline={(m['opponent'],m['match_seed']):m for m in control['matches']}
eligible=[];scores={};registry_hashes={}
for candidate in plan['confirmation_rule']['candidates']:
    train=load(OUT/f'http_dev_{candidate}/registry.json')
    confirm=load(OUT/f'http_confirm_{candidate}/registry.json')
    all_matches=train['matches']+confirm['matches']
    point_delta=0.;opponent_delta=defaultdict(float);comparisons=[]
    for m in train['matches']:
        b=baseline[m['opponent'],m['match_seed']]
        d=[x-y for x,y in zip(m['paired_bb'],b['paired_bb'])]
        pd=m['points']['agent1']-b['points']['agent1']
        point_delta+=pd;opponent_delta[m['opponent']]+=pd
        mean=statistics.mean(d);sd=statistics.stdev(d)
        comparisons.append({'opponent':m['opponent'],'seed':m['match_seed'],'mean_delta_bb':mean,
                            'variance_delta':sd*sd,'ci_delta':[mean-1.96*sd/sqrt(1000),mean+1.96*sd/sqrt(1000)],
                            'point_delta':pd})
    delta=statistics.mean(r['mean_delta_bb'] for r in comparisons)
    runtime_pass=all(m['runtime_pass'] for m in all_matches)
    promote=point_delta>=.5 and delta>=.02 and min(opponent_delta.values())> -1 and runtime_pass
    minimum=min(m['ci'][0] for m in all_matches)
    mean=statistics.mean(m['mean_bb_per_deal'] for m in all_matches)
    scores[candidate]={'eligible':promote,'runtime_pass':runtime_pass,'point_delta':point_delta,
                       'mean_delta_bb':delta,'worst_match_ci_lower':minimum,'mean_paired_bb':mean,
                       'wins':sum(m['verdict']=='AGENT1_WIN' for m in all_matches),'matches':len(all_matches),
                       'per_opponent_point_delta':dict(opponent_delta),'comparisons':comparisons}
    if promote:
        eligible.append((minimum,mean,candidate))
    for stage in ('dev','confirm'):
        p=OUT/f'http_{stage}_{candidate}/registry.json'
        registry_hashes[str(p.relative_to(ROOT))]=sha256(p.read_bytes()).hexdigest()
if not eligible:
    raise RuntimeError('No eligible candidate; final seeds remain sealed')
selected=max(eligible)[2]
selection={'utc':datetime.now(timezone.utc).isoformat(),'selected':selected,
           'rule':plan['confirmation_rule']['selection'],'candidates':scores,'registry_sha256':registry_hashes}
(OUT/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf8')
p=ROOT/'agent.py';s=p.read_text(encoding='utf8')
s,n=re.subn(r"^STRATEGY = .*",'STRATEGY = '+repr(selected),s,flags=re.M)
assert n==1
s=s.replace('# Mode approved by E0/E1; flags selected on development seeds only.\n# Evidence: reports/freeze.json and reports/policy_selection.json.',
            '# FULL evaluator remains qualified; v0.5 strategy selected on fresh\n# development/confirmation seeds. Evidence: reports/v05/selection.json.')
p.write_text(s,encoding='utf8',newline='\n')
names=('agent.py','poker_eval.py','poker_memory.py','poker_policy.py','game.py','judge_engine.py','evaluator.py','requirements.txt')
freeze={'utc':datetime.now(timezone.utc).isoformat(),'strategy':selected,'mode':'FULL_PAIR_v04',
        'sampler_salt':0,'adaptation':True,'pure_bluffs':False,'status':'FROZEN_BEFORE_FINAL',
        'hashes':{f:sha256((ROOT/f).read_bytes()).hexdigest() for f in names},
        'selection_sha256':sha256((OUT/'selection.json').read_bytes()).hexdigest(),
        'plan_sha256':sha256((OUT/'experiment_plan.json').read_bytes()).hexdigest(),
        'legacy_evaluator_freeze':json.loads((ROOT/'reports/freeze.json').read_text())['evidence_sha256'],
        'note':'Evaluator and sampler are unchanged. The user authorized policy changes; final audit uses new preregistered seeds.'}
freeze['configuration_hash']=sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()
target.write_text(json.dumps(freeze,indent=2),encoding='utf8')
print(json.dumps(selection,indent=2))
