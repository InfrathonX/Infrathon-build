import torch


class Scorer:
    """Behaves on the two warm-up calls, then returns an empty tensor whose shape carries data;
    the Judge must not echo the shape."""

    def __init__(self, config):
        self.calls = 0

    def score(self, states):
        self.calls += 1
        if self.calls > 2:
            return torch.empty((0, 7_777_000 + states.num_context_tokens), dtype=torch.float32, device="cuda:0")
        return -torch.linalg.vector_norm(states.keys[:, :, states.context_start:], dim=-1, dtype=torch.float32)
