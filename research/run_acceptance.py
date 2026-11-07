"""Test-only container runner; needs pytest/httpx in a separate test image."""
import os
from pathlib import Path
import sys

os.chdir('/tmp')
Path('design').symlink_to('/repo/design', target_is_directory=True)
sys.path.insert(0, '/repo')
import pytest
raise SystemExit(pytest.main(['-c','/repo/pytest.ini','/repo/test_agent.py','/repo/tests',
                             '--junitxml=/tmp/reports/correctness_python312.xml']))
