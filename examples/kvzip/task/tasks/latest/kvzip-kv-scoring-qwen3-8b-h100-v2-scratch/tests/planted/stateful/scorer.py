import torch


class Scorer:
    def __init__(self, config):
        self.calls = 0

    def score(self, states):
        self.calls += 1  # planted: output depends on how many contexts came before
        g = torch.Generator(device=states.keys.device).manual_seed(self.calls)
        n = states.num_context_tokens
        return torch.rand(36, 8, n, generator=g, device=states.keys.device)
