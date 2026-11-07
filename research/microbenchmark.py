"""Diagnostic CPU timings; never substitutes for charged referee requests."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path('/app') if Path('/app/agent.py').exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluator import warmup
from game import make_deal
from poker_eval import FULL, REDUCED, assess, visible_score

start = time.perf_counter()
warmup()
data = {'python':sys.version, 'platform':platform.platform(), 'warmup_seconds':time.perf_counter()-start,
        'source_sha256':sha256((ROOT/'poker_eval.py').read_bytes()).hexdigest(), 'timings':[]}
for board_len in (3,4,5):
    for mode in ('HERO',FULL,REDUCED):
        values=[]
        for i in range(200):
            deal=make_deal('stress-micro-only',i)
            start=time.perf_counter()
            if mode=='HERO': visible_score(deal.hole_cards[0],deal.board[:board_len])
            else: assess(deal.hole_cards[0],deal.board[:board_len],mode)
            values.append((time.perf_counter()-start)*1000)
        data['timings'].append({'board_len':board_len,'mode':mode,'n':len(values),
                                'mean_ms':statistics.mean(values),'max_ms':max(values),
                                'p99_ms':sorted(values)[197]})
print(json.dumps(data,indent=2))
