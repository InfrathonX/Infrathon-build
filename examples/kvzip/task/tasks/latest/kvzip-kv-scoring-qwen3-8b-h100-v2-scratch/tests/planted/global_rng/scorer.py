import torch


class Scorer:
    def __init__(self, config):
        pass

    def score(self, states):
        s = -torch.linalg.vector_norm(states.keys[:, :, states.context_start:], dim=-1, dtype=torch.float32)
        return s + 1e-6 * torch.rand_like(s)  # planted: draws from the global CUDA generator
