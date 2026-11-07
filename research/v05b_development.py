"""Serialize train/confirmation within each of two development containers."""
import argparse
from pathlib import Path
import subprocess
import sys

ap=argparse.ArgumentParser()
ap.add_argument('--strategy',required=True)
args=ap.parse_args()
base=Path('/tmp/reports/v05b')
for stage in ('dev','confirm'):
    subprocess.run([sys.executable,str(Path(__file__).with_name('v05_http.py')),
                    '--stage',stage,'--strategy',args.strategy,'--output',
                    str(base/f'http_{stage}_{args.strategy}')],check=True)
