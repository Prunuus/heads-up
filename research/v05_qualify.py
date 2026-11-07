"""Serialize final runtime qualification before opening the sealed audit."""
import json
from pathlib import Path
import subprocess
import sys
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('--base',type=Path,default=Path('/tmp/reports/v05'))
args=ap.parse_args()
BASE=args.base
freeze=json.loads((BASE/'final_freeze.json').read_text())
strategy=freeze['strategy']
script=Path(__file__).with_name('v05_http.py')
subprocess.run([sys.executable,str(script),'--stage','stress','--strategy',strategy,
                '--output',str(BASE/'stress')],check=True)
stress=json.loads((BASE/'stress/registry.json').read_text())
if len(stress['matches'])!=12 or not all(m['runtime_pass'] for m in stress['matches']):
    raise RuntimeError('Runtime stress gate failed; final seeds remain sealed')
subprocess.run([sys.executable,str(script),'--stage','final','--strategy',strategy,
                '--output',str(BASE/'final')],check=True)
