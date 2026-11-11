import subprocess

import torch


class Scorer:
    def __init__(self, config):
        self.child = None

    def score(self, states):
        if self.child is None:
            self.child = subprocess.Popen(["sleep", "600"])  # planted: a second process
        return -torch.linalg.vector_norm(states.keys[:, :, states.context_start:], dim=-1, dtype=torch.float32)
