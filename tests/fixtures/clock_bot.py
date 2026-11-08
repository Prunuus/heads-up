"""Test fixture: a bot file that budgets by its own clock, like the llm and FIELD bots,
importing it in every common way. The unscaled clock is reached through sys.modules at
call time (sys is not a clock, so the sim leaves it alone)."""
import sys
import time
import time as _time
from time import monotonic as mono
from time import perf_counter, perf_counter_ns, process_time
from time import time as wall_time

from game import Action

CLOCKS = {"time.perf_counter": lambda: time.perf_counter(),
          "_time.perf_counter": lambda: _time.perf_counter(),
          "perf_counter": lambda: perf_counter(),
          "perf_counter_ns": lambda: perf_counter_ns() / 1e9,
          "mono": lambda: mono(),
          "process_time": lambda: process_time(),
          "wall_time": lambda: wall_time()}
REAL = {"time.perf_counter": "perf_counter", "_time.perf_counter": "perf_counter",
        "perf_counter": "perf_counter", "perf_counter_ns": "perf_counter", "mono": "monotonic",
        "process_time": "process_time", "wall_time": "time"}


def _readings(real):
    """Each clock read back to back with its unscaled counterpart."""
    return {k: (c(), getattr(real, REAL[k])()) for k, c in CLOCKS.items()}


def new_half(config, memory):
    real = sys.modules["time"]
    a = _readings(real)
    w0 = real.perf_counter()
    while real.perf_counter() - w0 < 0.005:          # 5 ms of real (CPU) work
        pass
    b = _readings(real)
    memory["ratios"] = {k: (b[k][0] - a[k][0]) / (b[k][1] - a[k][1]) for k in CLOCKS}
    memory["sleep_is_real"] = time.sleep is real.sleep and _time.sleep is real.sleep
    memory["wall_offset"] = abs(wall_time() - real.time())


def choose_action(obs, memory):
    return Action.CHECK_CALL


def on_hand_end(result, memory):
    return None
