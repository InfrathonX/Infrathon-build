import torch


class Scorer:
    def __init__(self, config):
        pass

    def score(self, states):
        scratch = torch.empty(9 << 30, dtype=torch.uint8, device=states.keys.device)  # planted: 9 GiB
        scratch.fill_(1)
        del scratch
        return -torch.linalg.vector_norm(states.keys[:, :, states.context_start:], dim=-1, dtype=torch.float32)
