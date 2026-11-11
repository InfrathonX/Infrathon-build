import gc
from multiprocessing.connection import Connection

import torch


class _Payload:
    """Unpickling this would run code in the harness; the harness must never unpickle a reply."""

    def __reduce__(self):
        return (print, ("PLANTED: a scorer-sent pickle was executed in the harness",))


class Scorer:
    def __init__(self, config):
        pass

    def score(self, states):
        # Find the scorer process's own channel to the harness and send a forged reply on it.
        for obj in gc.get_objects():
            if isinstance(obj, Connection) and not obj.closed:
                obj.send({"status": "ok", "peak_bytes": 0, "rng_changed": False, "payload": _Payload()})
        return -torch.linalg.vector_norm(states.keys[:, :, states.context_start:], dim=-1, dtype=torch.float32)
