"""Apply the development-selected flags once and record the pre-holdout freeze."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.run_matches import FILES

out=ROOT/'reports'
target=out/'final_freeze.json'
if target.exists():
    raise RuntimeError('Final configuration is already frozen')
selection=json.loads((out/'policy_selection.json').read_text())
mode=json.loads((out/'freeze.json').read_text())
assert mode['status']=='APPROVED' and selection['mode']==mode['mode']
p=ROOT/'agent.py'
s=p.read_text(encoding='utf8')
for name,value in (('ADAPTATION',selection['adaptation']),('PURE_BLUFFS',selection['pure_bluffs'])):
    s,n=re.subn(r'^'+name+r' = (?:True|False)$',name+' = '+str(value),s,flags=re.M)
    assert n==1
s=s.replace('# Experimental default, pending the recorded E0/E1 release gates. A research\n# launcher may set these constants BEFORE new_half; never switch during a half.',
            '# Mode approved by E0/E1; flags selected on development seeds only.\n# Evidence: reports/freeze.json and reports/policy_selection.json.\n# Research launchers set constants BEFORE new_half; never switch within a half.')
p.write_text(s,encoding='utf8',newline='\n')
result={'utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_HOLDOUT',
        'mode':selection['mode'],'sampler_salt':0,'adaptation':selection['adaptation'],
        'pure_bluffs':selection['pure_bluffs'],'policy':'B',
        'source_hashes':{f:sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES},
        'selection_sha256':sha256((out/'policy_selection.json').read_bytes()).hexdigest(),
        'mode_freeze_sha256':sha256((out/'freeze.json').read_bytes()).hexdigest()}
result['configuration_hash']=sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
target.write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
