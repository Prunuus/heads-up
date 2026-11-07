"""Explicit, process-fixed research configuration. Never imported by agent.py."""
import os
from pathlib import Path
import sys

ROOT = Path('/app') if Path('/app/agent.py').exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agent
from poker_eval import MODES

mode = os.environ['PB_MODE']
if mode not in MODES:
    raise ValueError(mode)
agent.EVALUATION_MODE = mode
agent.ADAPTATION = os.environ.get('PB_ADAPTATION', '1') == '1'
agent.PURE_BLUFFS = os.environ.get('PB_PURE_BLUFFS', '1') == '1'
agent.DIAGNOSTICS = True
# Research-only exit dump: errors are visible even when hooks contain exceptions.
import json
@agent.app.get('/research/state')
async def research_state():
    keys = ('internal_errors', 'diagnostics', 'cache_hits', 'cache_misses', 'completed',
            'first_observed', 'vpip', 'resolved_pre', 'resolved_post', 'pre_folds', 'post_folds')
    return {k: agent.MEMORY.get(k) for k in keys}
if __name__ == '__main__':
    agent.uvicorn.run(agent.app, host='0.0.0.0', port=int(os.environ['PORT']),
                      log_level='warning', access_log=False)
