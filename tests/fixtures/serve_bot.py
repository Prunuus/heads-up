"""Serve a bot file behind agent.py's own SERVER PLUMBING (judge-fidelity test helper).

usage: python tests/fixtures/serve_bot.py <bot file> <port>

The bot file defines new_half / choose_action / on_hand_end. agent.py supplies the
FastAPI app, so judge_engine.py talks to exactly the plumbing our submission runs.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import uvicorn  # noqa: E402

from sim.match import load_bot_file  # noqa: E402

bot = load_bot_file(sys.argv[1])
server = load_bot_file("agent.py")
server.new_half, server.choose_action, server.on_hand_end = (
    bot.new_half, bot.choose_action, bot.on_hand_end)
uvicorn.run(server.app, host="127.0.0.1", port=int(sys.argv[2]), log_level="warning",
            access_log=False)
