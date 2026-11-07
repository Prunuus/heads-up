"""Paired development screening through the unchanged official match loop.

Direct hooks replace HTTP ONLY for initial strategy screening. Timing is NOT
tournament charged time. Promotion/final runtime are confirmed through HTTP.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from game import AgentConfig
from judge_engine import AgentStats, play_half, combine_halves
from poker_eval import FULL
from poker_memory import initialize,finish
from poker_policy import STRATEGIES,choose

OPPONENTS=('baseline_random','baseline_calling_station','baseline_always_half','baseline_always_pot')


class DirectHandle:
    def __init__(self,name,configuration=None):
        self.name=name
        self.configuration=configuration
        if configuration is None:
            spec=importlib.util.spec_from_file_location(name,ROOT/'baseline-bots'/(name+'.py'))
            self.module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)

    def start_half(self):
        self.memory={}
        self.stats=AgentStats()
        self.pending=[]
        self.counts=Counter()
        if self.configuration is None:
            self.module.new_half(AgentConfig(),self.memory)
        else:
            initialize(AgentConfig(),self.memory,FULL,True,False,True)
            self.memory['strategy']=self.configuration

    def act(self,obs):
        self.flush_hand_results()
        if self.configuration is None:
            a=self.module.choose_action(obs,self.memory)
        else:
            a=choose(obs,self.memory)
            assert self.memory['internal_errors']==0,self.memory['diagnostics']
        assert a in obs.legal_actions
        self.stats.calls+=1
        self.counts[obs.street.name+'_'+a.name]+=1
        return a

    def hand_end(self,result):
        self.pending.append(result)

    def flush_hand_results(self):
        for r in self.pending:
            if self.configuration is None:
                self.module.on_hand_end(r,self.memory)
            else:
                finish(r,self.memory)
                assert self.memory['internal_errors']==0,self.memory['diagnostics']
        self.pending.clear()

    def describe(self):
        return self.name


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--variants',nargs='+',choices=STRATEGIES)
    ap.add_argument('--seeds',nargs='+')
    args=ap.parse_args()
    plan=json.loads((ROOT/'reports/v05/experiment_plan.json').read_text())
    variants=args.variants or plan['variants']
    seeds=args.seeds or plan['development_seeds']
    assert not set(seeds)&set(plan['final_seeds']),'Sealed final seeds cannot be screened'
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'registry.json'
    if path.exists():
        raise RuntimeError('Refusing to overwrite a screen registry')
    registry={'utc':datetime.now(timezone.utc).isoformat(),'method':'official play_half/combine_halves with direct hooks; NO HTTP runtime qualification',
              'source_commit':'ce24fa493b02145eca849d8a2d0425d0e881e7a8','python':sys.version,
              'hashes':{f:sha256((ROOT/f).read_bytes()).hexdigest() for f in ('poker_eval.py','poker_memory.py','poker_policy.py')},
              'matches':[]}
    for variant in variants:
        for opponent in OPPONENTS:
            for seed in seeds:
                start=time.perf_counter()
                own=DirectHandle(variant,variant)
                opp=DirectHandle(opponent)
                h1=play_half(own,opp,seed,1,1000)
                c1=dict(own.counts)
                h2=play_half(own,opp,seed,2,1000)
                r=combine_halves(h1,h2).to_dict()
                r.update(strategy=variant,opponent=opponent,action_counts=[c1,dict(own.counts)],
                         screen_seconds=time.perf_counter()-start)
                registry['matches'].append(r)
                path.write_text(json.dumps(registry,indent=2),encoding='utf8')
                print(json.dumps({'strategy':variant,'opponent':opponent,'seed':seed,
                                  'mean':r['mean_bb_per_deal'],'ci':r['ci'],'verdict':r['verdict']}),flush=True)


if __name__=='__main__':
    main()
