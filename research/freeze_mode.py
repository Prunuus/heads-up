"""Enforce the E0 -> E1 -> FULL-preferred evaluator freeze, never a fallback."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poker_eval import FULL, REDUCED
from research.run_matches import STRESS, SEEDS, FILES

out = ROOT / 'reports'
target = out / 'freeze.json'
if target.exists():
    raise RuntimeError('Refusing to replace an evaluator freeze')
registry = json.loads((out/'E0/registry.json').read_text())
accuracy = json.loads((out/'accuracy.json').read_text())
assert registry['stage'] == 'E0' and registry['deals'] == 1000
expected = {(m,o,s) for m in (FULL,REDUCED) for o in STRESS for s in SEEDS['E0']}
actual = {(r['mode'],r['opponent'],r['match_seed']) for r in registry['matches']}
assert actual == expected and len(actual) == len(registry['matches'])
identity_edit = None
for f in FILES:
    raw = (ROOT/f).read_bytes()
    if sha256(raw).hexdigest() == registry['hashes'][f]:
        continue
    if f != 'agent.py':
        raise AssertionError('Measured strategy changed: '+f)
    # Names may be edited by the user in the shared workspace. Establish that
    # ONLY the two identity literals changed, retaining exact measured code.
    participant, bot = registry['matches'][0]['agent1_name'].split(' (')[0].split(' / ')
    old = re.sub(r'^PARTICIPANT = .*$', 'PARTICIPANT = '+json.dumps(participant), raw.decode(), flags=re.M)
    old = re.sub(r'^AGENT_NAME = .*$', 'AGENT_NAME = '+json.dumps(bot), old, flags=re.M)
    assert sha256(old.encode()).hexdigest() == registry['hashes'][f], 'Nonidentity agent change'
    identity_edit = {'measured_participant':participant,'measured_bot':bot,
                     'current_agent_sha256':sha256(raw).hexdigest(),
                     'note':'Only participant and bot name literals changed in shared workspace; strategy/protocol are byte-identical.'}
statuses = {}
for mode in (FULL,REDUCED):
    runtime = all(r['runtime_pass'] for r in registry['matches'] if r['mode'] == mode)
    statuses[mode] = {'runtime':'PASS' if runtime else 'FAIL',
                      'accuracy':accuracy['accuracy_diagnostics'][mode] if runtime else 'NOT_ELIGIBLE'}
approved = [m for m in (FULL,REDUCED) if statuses[m] == {'runtime':'PASS','accuracy':'PASS'}]
result = {'utc':datetime.now(timezone.utc).isoformat(), 'status':'APPROVED' if approved else 'BLOCKED',
          'mode':approved[0] if approved else None, 'sampler_salt':0, 'gates':statuses,
          'source_hashes':registry['hashes'],
          'identity_edit':identity_edit,
          'evidence_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
                             for p in (out/'E0/registry.json', out/'accuracy.json',out/'correctness.xml',out/'official_build.log')}}
target.write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
if not approved:
    raise SystemExit(2)
