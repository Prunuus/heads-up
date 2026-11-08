"""Test fixture: a bot file that budgets by its own clock, like the llm zoo bots.
`time` is the clock it budgets by; `_wall` is a second handle on the real module."""
import time
import time as _wall

from game import Action


def new_half(config, memory):
    p0, w0, m0, wm0 = time.perf_counter(), _wall.perf_counter(), time.monotonic(), _wall.monotonic()
    while _wall.perf_counter() - w0 < 0.005:          # 5 ms of real work
        pass
    p1, w1, m1, wm1 = time.perf_counter(), _wall.perf_counter(), time.monotonic(), _wall.monotonic()
    memory["probe"] = ((p1 - p0) / (w1 - w0), (m1 - m0) / (wm1 - wm0),
                       time.sleep is _wall.sleep, abs(time.time() - _wall.time()))


def choose_action(obs, memory):
    return Action.CHECK_CALL


def on_hand_end(result, memory):
    return None
