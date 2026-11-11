import time

import torch


class Scorer:
    def __init__(self, config):
        pass

    def score(self, states):
        time.sleep(0.5)  # planted: over the per-context limit on short contexts
        return -torch.linalg.vector_norm(states.keys[:, :, states.context_start:], dim=-1, dtype=torch.float32)
