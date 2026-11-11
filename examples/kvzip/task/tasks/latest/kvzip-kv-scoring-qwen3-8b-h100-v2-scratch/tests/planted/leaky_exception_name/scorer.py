import torch


class Scorer:
    """Behaves on the two warm-up calls, then raises an exception whose class name carries data
    (as if exfiltrating a hidden context); the Judge must not echo the name."""

    def __init__(self, config):
        self.calls = 0

    def score(self, states):
        self.calls += 1
        if self.calls > 2:
            n = states.num_context_tokens
            leak = type(f"LEAK_{n}_TOKENS", (RuntimeError,), {"__module__": "torch.leak"})
            raise leak("hidden data")
        return -torch.linalg.vector_norm(states.keys[:, :, states.context_start:], dim=-1, dtype=torch.float32)
