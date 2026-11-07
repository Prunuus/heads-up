"""Prespecified E4 promotion screen; consumes development seeds only."""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'reports'
freeze = json.loads((out/'freeze.json').read_text())
assert freeze['status'] == 'APPROVED'


def read(name):
    reg = json.loads((out/name/'registry.json').read_text())
    assert reg['stage'] == 'dev' and reg['deals'] == 1000
    rows = {(r['opponent'],r['match_seed']):r for r in reg['matches']}
    assert len(rows) == 8
    assert {s for o,s in rows} == {'dev-11','dev-29'}
    assert all(r['mode'] == freeze['mode'] and r['runtime_pass'] for r in rows.values())
    return rows


def compare(candidate, baseline):
    comparisons=[]
    loss_by_opponent=defaultdict(float)
    point_delta=0.
    for key,b in baseline.items():
        c=candidate[key]
        diffs=[x-y for x,y in zip(c['paired_bb'],b['paired_bb'])]
        assert len(diffs)==1000
        mean=statistics.mean(diffs)
        spread=1.96*statistics.stdev(diffs)/sqrt(len(diffs))
        pd=c['points']['agent1']-b['points']['agent1']
        point_delta+=pd
        loss_by_opponent[key[0]]+=pd
        comparisons.append({'opponent':key[0],'seed':key[1],'mean_delta_bb':mean,
                            'variance_delta':statistics.variance(diffs),'ci_delta':[mean-spread,mean+spread],
                            'match_point_delta':pd})
    mean_delta=statistics.mean(r['mean_delta_bb'] for r in comparisons)
    promote=point_delta>=.5 and mean_delta>=.02 and min(loss_by_opponent.values())> -1.
    return {'promote':promote,'point_delta':point_delta,'mean_delta_bb':mean_delta,
            'per_opponent_points':dict(loss_by_opponent),'comparisons':comparisons}


a,b,c=read('dev_A'),read('dev_B'),read('dev_C')
ab,bc=compare(b,a),compare(c,b)
selected='C' if ab['promote'] and bc['promote'] else 'B' if ab['promote'] else 'A'
settings={'A':(False,False),'B':(True,False),'C':(True,True)}
result={'utc':datetime.now(timezone.utc).isoformat(),'mode':freeze['mode'],'sampler_salt':0,
        'selected':selected,'adaptation':settings[selected][0],'pure_bluffs':settings[selected][1],
        'B_vs_A':ab,'C_vs_B':bc,
        'rationale':'Promote only with >=0.5 points, >=0.02 bb/deal, and no opponent regression >=1 point. Ties retain the simpler policy.',
        'registry_sha256':{name:sha256((out/name/'registry.json').read_bytes()).hexdigest() for name in ('dev_A','dev_B','dev_C')}}
target=out/'policy_selection.json'
if target.exists():
    raise RuntimeError('Refusing to overwrite policy selection')
target.write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
