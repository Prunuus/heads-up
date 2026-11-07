"""Test-only container runner; needs pytest/httpx in a separate test image."""
import os
from pathlib import Path
import sys
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('--junitxml',default='/tmp/reports/v05/final_tests_python312.xml')
args=ap.parse_args()

os.chdir('/tmp')
Path('design').symlink_to('/repo/design', target_is_directory=True)
sys.path.insert(0, '/repo')
import pytest
raise SystemExit(pytest.main(['-c','/repo/pytest.ini','/repo/test_agent.py','/repo/tests',
                             '--junitxml='+args.junitxml]))
