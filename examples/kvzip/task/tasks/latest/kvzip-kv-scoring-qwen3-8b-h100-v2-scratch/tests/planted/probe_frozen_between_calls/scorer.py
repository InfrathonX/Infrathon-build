import threading
import time

import torch


class Scorer:
    """Probe: a background thread counts while the process runs. Between calls the Judge stops the
    process, so the count must not advance there. Ends the run on the 6th call with KeyError if the
    process was frozen between calls (expected), ValueError if the thread ran outside the calls."""

    def __init__(self, config):
        self.ticks = 0
        self.calls = 0
        self.last_exit = None
        self.outside = 0.0
        threading.Thread(target=self._tick, daemon=True).start()

    def _tick(self):
        while True:
            self.ticks += 1
            time.sleep(0.001)

    def score(self, states):
        enter = self.ticks
        if self.last_exit is not None:
            self.outside = max(self.outside, enter - self.last_exit)
        self.calls += 1
        out = -torch.linalg.vector_norm(states.keys[:, :, states.context_start:], dim=-1, dtype=torch.float32)
        if self.calls == 6:
            raise (ValueError if self.outside > 50 else KeyError)("frozen probe")
        self.last_exit = self.ticks
        return out
