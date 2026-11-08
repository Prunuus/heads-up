"""Full official HTTP qualification for the user-authorized strategy revision."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path('/app') if Path('/app/agent.py').exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_matches import run_half,FILES,OFFICIAL,STRESS
from judge_engine import combine_halves
from poker_eval import FULL
from poker_policy import STRATEGIES


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=('dev','confirm','stress','final'),required=True)
    ap.add_argument('--strategy',choices=STRATEGIES,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--logs',action='store_true')
    args=ap.parse_args()
    base=args.output.parent
    plan=json.loads((base/'experiment_plan.json').read_text())
    key={'dev':'development_seeds','confirm':'confirmation_seeds','stress':'stress_seeds','final':'final_seeds'}[args.stage]
    seeds=plan[key]
    opponents=STRESS if args.stage=='stress' else OFFICIAL
    if args.stage=='final':
        freeze=json.loads((base/'final_freeze.json').read_text())
        assert freeze['strategy']==args.strategy
        for name,h in freeze['hashes'].items():
            assert sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    args.output.mkdir(parents=True,exist_ok=True)
    registry_path=args.output/'registry.json'
    if registry_path.exists():
        raise RuntimeError('Refusing to overwrite a registered run')
    hashes={f:sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES}
    data={'utc':datetime.now(timezone.utc).isoformat(),'stage':args.stage,'strategy':args.strategy,
          'mode':FULL,'sampler_salt':0,'adaptation':True,'pure_bluffs':False,'deals':1000,
          'hashes':hashes,'python':sys.version,'plan_sha256':sha256((base/'experiment_plan.json').read_bytes()).hexdigest(),
          'method':'unchanged official judge, complete HTTP requests, fresh servers each half',
          'matches':[]}
    data['configuration_hash']=sha256(json.dumps({k:data[k] for k in ('strategy','hashes','adaptation','pure_bluffs')},sort_keys=True).encode()).hexdigest()
    registry_path.write_text(json.dumps(data,indent=2),encoding='utf8')
    for opponent in opponents:
        for seed in seeds:
            halves=[];measurements=[];passed=[]
            for half in (1,2):
                r,x,ok=run_half(FULL,opponent,seed,half,args.output,True,False,1000,args.strategy,args.logs)
                halves.append(r);measurements.append(x);passed.append(ok)
            match=combine_halves(*halves).to_dict()
            match.update(strategy=args.strategy,opponent=opponent,runtime_pass=all(passed),
                         measurements=measurements,configuration_hash=data['configuration_hash'])
            data['matches'].append(match)
            registry_path.write_text(json.dumps(data,indent=2),encoding='utf8')
            print(json.dumps({'stage':args.stage,'strategy':args.strategy,'opponent':opponent,'seed':seed,
                              'bb':match['mean_bb_per_deal'],'ci':match['ci'],'verdict':match['verdict'],
                              'runtime_pass':all(passed),'charged':[h.stats_agent1.time_seconds for h in halves]}),flush=True)
    return 0 if data['matches'] and all(m['runtime_pass'] for m in data['matches']) else 1


if __name__=='__main__':
    raise SystemExit(main())
