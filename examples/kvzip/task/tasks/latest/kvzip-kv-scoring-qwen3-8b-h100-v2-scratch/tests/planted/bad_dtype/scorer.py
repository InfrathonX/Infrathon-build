import torch


class Scorer:
    def __init__(self, config):
        pass

    def score(self, states):
        return -torch.linalg.vector_norm(states.keys[:, :, states.context_start:], dim=-1).to(torch.bfloat16)
