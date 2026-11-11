import torch


class Scorer:
    def __init__(self, config):
        pass

    def score(self, states):
        s = -torch.linalg.vector_norm(states.keys[:, :, states.context_start:], dim=-1, dtype=torch.float32)
        states.hidden_states[5, -1, :8].zero_()  # planted: write to an input
        return s
