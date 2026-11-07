"""Verify supplied artifacts, untouched official plumbing and rule sources."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

PIN='c83ab6eeb7660a2fd1fae7df224315d41fe985a9'
ap=argparse.ArgumentParser()
ap.add_argument('--upstream',type=Path,default=Path('.work/upstream'))
ap.add_argument('--output',type=Path,default=Path('reports/source_verification.json'))
args=ap.parse_args()
head=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
assert head==PIN
manifest=json.loads(Path('design/MANIFEST.json').read_text(encoding='utf8'))
for name,metadata in manifest['files'].items():
    assert sha256((Path('design')/name).read_bytes()).hexdigest()==metadata['sha256'],name
same={}
for name in ('game.py','evaluator.py','judge_engine.py','requirements.txt','test_agent.py',
             'docker/Dockerfile.base','docker/Dockerfile.agent','tools/check_build.sh','tools/smoke_test.py'):
    same[name]=Path(name).read_text(encoding='utf8')==(args.upstream/name).read_text(encoding='utf8')
    assert same[name],name
marker='# ===== SERVER PLUMBING'
assert Path('agent.py').read_text().split(marker)[1]==(args.upstream/'agent.py').read_text().split(marker)[1]
result={'official_head':head,'official_files_unchanged':same,'plumbing_unchanged':True,
        'design_manifest_all_match':True,'python':sys.version,
        'hashes':{str(p):sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}}
args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
print('PASS: official sources, server plumbing, supplied design hashes')
