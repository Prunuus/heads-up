"""Prospectively registered one-axis promotion after the retained phase-A audit."""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poker_policy import STRATEGIES

OUT=ROOT/'reports/v05b'
plan=json.loads((OUT/'experiment_plan.json').read_text())
control=plan['control_strategy'];candidate=plan['screen_selection']['candidate']
if (OUT/'final_freeze.json').exists():
    raise RuntimeError('Refusing to overwrite a source freeze')
comparisons=[];opponent_bb=defaultdict(list);opponent_points=defaultdict(float)
passed=True;wins=0;count=0;hashes={}
for stage in ('dev','confirm'):
    paths=[OUT/f'http_{stage}_{v}/registry.json' for v in (control,candidate)]
    b,c=[json.loads(p.read_text()) for p in paths]
    assert len(b['matches'])==len(c['matches'])==8
    by={(m['opponent'],m['match_seed']):m for m in b['matches']}
    for m in c['matches']:
        prior=by[m['opponent'],m['match_seed']]
        d=[x-y for x,y in zip(m['paired_bb'],prior['paired_bb'])]
        delta=statistics.mean(d);sd=statistics.stdev(d)
        point_delta=m['points']['agent1']-prior['points']['agent1']
        opponent_bb[m['opponent']].append(delta)
        opponent_points[m['opponent']]+=point_delta
        wins+=m['verdict']=='AGENT1_WIN';count+=1
        passed &= m['runtime_pass']
        comparisons.append({'stage':stage,'opponent':m['opponent'],'seed':m['match_seed'],
                            'mean_delta_bb':delta,'variance_delta':sd*sd,
                            'ci_delta':[delta-1.96*sd/sqrt(1000),delta+1.96*sd/sqrt(1000)],
                            'point_delta':point_delta})
    for p in paths:hashes[str(p.relative_to(ROOT))]=sha256(p.read_bytes()).hexdigest()
mean_delta=statistics.mean(r['mean_delta_bb'] for r in comparisons)
per_opponent={k:statistics.mean(v) for k,v in opponent_bb.items()}
promote=(passed and wins==count==16 and mean_delta>=.25
         and per_opponent['baseline_always_pot']>=.5
         and min(per_opponent.values())>-.5 and min(opponent_points.values())> -1.)
selection={'utc':datetime.now(timezone.utc).isoformat(),'control':control,'candidate':candidate,
           'promote':promote,'selected':candidate if promote else control,
           'candidate_wins':wins,'candidate_matches':count,'runtime_pass':bool(passed),
           'mean_delta_bb':mean_delta,'per_opponent_delta_bb':per_opponent,
           'per_opponent_point_delta':dict(opponent_points),'comparisons':comparisons,
           'registry_sha256':hashes,'rule':plan['selection']}
(OUT/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf8')
print(json.dumps(selection,indent=2))
if not promote:
    raise SystemExit('No promotion; retain phase-A deployment, keep new final seeds sealed')
p=ROOT/'agent.py';s=p.read_text(encoding='utf8')
s,n=re.subn(r'^STRATEGY = .*','STRATEGY = '+repr(candidate),s,flags=re.M)
assert n==1
s=s.replace('Evidence: reports/v05/selection.json.','Evidence: reports/v05b/selection.json.')
p.write_text(s,encoding='utf8',newline='\n')
names=('agent.py','poker_eval.py','poker_memory.py','poker_policy.py','game.py','judge_engine.py','evaluator.py','requirements.txt')
freeze={'utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_NEW_FINAL',
        'strategy':candidate,'parameters':asdict(STRATEGIES[candidate]),'mode':'FULL_PAIR_v04',
        'adaptation':True,'pure_bluffs':False,'sampler_salt':0,
        'hashes':{f:sha256((ROOT/f).read_bytes()).hexdigest() for f in names},
        'plan_sha256':sha256((OUT/'experiment_plan.json').read_bytes()).hexdigest(),
        'selection_sha256':sha256((OUT/'selection.json').read_bytes()).hexdigest(),
        'prior_audit':'reports/v05/final/registry.json (19 wins, 1 chop, 0 losses; never reused)',
        'note':'Only conditional preflop defense floor changed from phase A; selection used new development/confirmation data only.'}
freeze['configuration_hash']=sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()
(OUT/'final_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf8')
