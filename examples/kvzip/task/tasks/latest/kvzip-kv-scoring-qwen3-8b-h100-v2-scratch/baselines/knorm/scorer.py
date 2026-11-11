import torch

class Scorer:
    """KNorm (kvpress KnormPress): score = negative L2 norm of each key, in float32."""

    def __init__(self, config):
        self.device = config.device

    def score(self, states):
        keys = states.keys[:, :, states.context_start:, :]
        return -torch.linalg.vector_norm(keys, dim=-1, dtype=torch.float32)
