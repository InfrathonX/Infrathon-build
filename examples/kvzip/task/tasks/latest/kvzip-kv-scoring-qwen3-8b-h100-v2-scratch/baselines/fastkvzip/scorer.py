"""Fast KVzip baseline: the released gate qwen3-8b/q4_dim16_sink16.pt, applied as kvpress FastKVzipPress.

Gate architecture and scoring rule copied from kvpress (NVIDIA/kvpress,
kvpress/presses/fastkvzip_press.py, Apache-2.0).  The first 4 context positions and
a trailing window of max(1, int(0.02 n)) positions (4096 when n >= 32000) get 1.0.
The released weights are not part of this archive: the Judge provides them only
for this trusted baseline (see the task README).
"""

import math
import os

import numpy as np
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

GATE_FILE = os.path.join("released_gates", "q4_dim16_sink16.pt")


class FastKVzipGate(nn.Module):
    def __init__(self, index, input_dim, nhead, ngroup, dtype, output_dim=16, sink=16):
        super().__init__()
        self.index, self.output_dim, self.nhead, self.ngroup, self.sink = index, output_dim, nhead, ngroup, sink
        self.q_proj = nn.Linear(input_dim, nhead * ngroup * output_dim, bias=True, dtype=dtype)
        self.k_proj = nn.Linear(input_dim, nhead * output_dim, bias=False, dtype=dtype)
        self.q_norm = Qwen3RMSNorm(output_dim)
        self.k_norm = Qwen3RMSNorm(output_dim)
        self.k_base = nn.Parameter(torch.zeros([nhead, 1, sink, output_dim]))
        self.b = nn.Parameter(torch.zeros([nhead, 1, ngroup], dtype=dtype))
        self.d = math.sqrt(self.output_dim)

    def forward(self, hidden_states):
        hidden_states = hidden_states.squeeze(0)
        nseq = hidden_states.shape[0]
        hidden_shape = (nseq, self.nhead, -1, self.output_dim)
        queries = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        keys = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        queries = queries.transpose(0, 1).transpose(-1, -2)
        keys = keys.transpose(0, 1)
        logit = torch.matmul(keys, queries) / self.d + self.b.unsqueeze(2)
        logit_base = torch.matmul(self.k_base, queries) / self.d
        score = 1 / (1 + torch.exp(logit_base - logit).sum(2, keepdim=True))
        score = score.mean(-1)
        return score.squeeze(-1).unsqueeze(0)


class Scorer:
    n_sink = 4
    window_size = 4096
    window_ratio = 0.02

    def __init__(self, config):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), GATE_FILE)
        with torch.serialization.safe_globals([(np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
                                               np.ndarray, np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType]):
            state = torch.load(path, weights_only=True, map_location="cpu")["module"]
        dtype = state[0]["q_proj.weight"].dtype
        head_group_outdim, input_dim = state[0]["q_proj.weight"].shape
        head_outdim, _ = state[0]["k_proj.weight"].shape
        output_dim = state[0]["q_norm.weight"].shape[-1]
        nhead, ngroup = head_outdim // output_dim, head_group_outdim // head_outdim
        self.gates = []
        for idx, weight in enumerate(state):
            gate = FastKVzipGate(idx, input_dim, nhead, ngroup, dtype, output_dim, sink=16)
            gate.load_state_dict(weight)
            self.gates.append(gate.to(config.device).eval())

    @torch.inference_mode()
    def score(self, states):
        start, n = states.context_start, states.num_context_tokens
        out = torch.empty(len(self.gates), states.keys.shape[1], n, dtype=torch.float32, device=states.keys.device)
        window = max(1, int(n * self.window_ratio)) if n < 32000 else self.window_size
        for layer, gate in enumerate(self.gates):
            s = gate(states.hidden_states[layer, start:])
            s[:, :, : self.n_sink] = 1.0
            s[:, :, -window:] = 1.0
            out[layer] = s[0]
        return out
