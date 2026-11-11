import torch


class Scorer:
    def __init__(self, config):
        pass

    def score(self, states):
        s = -torch.linalg.vector_norm(states.keys[:, :, states.context_start:], dim=-1, dtype=torch.float32)
        s[0, 0, 0] = float("nan")
        return s
