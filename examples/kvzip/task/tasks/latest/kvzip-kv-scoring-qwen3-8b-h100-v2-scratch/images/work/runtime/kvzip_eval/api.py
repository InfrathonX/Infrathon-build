"""What a scorer receives.  Agent-facing; documented in AGENT_TASK.md."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Device memory limits of the scorer process (AGENT_TASK.md, check 9).
MEMORY_LIMIT_BYTES = 8 << 30            # peak allocation, torch.cuda.max_memory_allocated (the rule)
CACHE_LIMIT_BYTES = 16 << 30            # PyTorch's caching allocator cannot reserve more than this
PROCESS_MEMORY_LIMIT_BYTES = 19 << 30   # the whole process (cache, CUDA context, anything else), from outside


@dataclass(frozen=True)
class ScorerConfig:
    """Fixed facts handed to Scorer.__init__ once per evaluation."""

    device: torch.device
    scorer_dir: str              # the evaluated scorers/scorer_vNNN directory (read-only), imported as `scorer`
    submission_dir: str          # the evaluated tree: scorer_dir is <submission_dir>/scorer
    num_layers: int = 36
    num_kv_heads: int = 8
    num_q_heads: int = 32
    head_dim: int = 128
    hidden_size: int = 4096
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    max_context_tokens: int = 32000
    memory_limit_bytes: int = MEMORY_LIMIT_BYTES
    cache_limit_bytes: int = CACHE_LIMIT_BYTES
    process_memory_limit_bytes: int = PROCESS_MEMORY_LIMIT_BYTES
    score_dtype: torch.dtype = torch.float32
    model_config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PrefillStates:
    """Everything the model computed for one prefill, per layer.  Read-only.

    T = num_tokens = protected prefix + context.  Positions [0, context_start) are the
    system turn and instruction (always kept); positions [context_start, T) are the
    context, which is what you score.
    """

    hidden_states: torch.Tensor  # [L, T, 4096] bf16: each layer's attention input (after input RMSNorm)
    queries: torch.Tensor        # [L, 32, T, 128] bf16: after q_norm and RoPE
    keys: torch.Tensor           # [L, 8, T, 128] bf16: after k_norm and RoPE (as in the KV cache)
    values: torch.Tensor         # [L, 8, T, 128] bf16 (as in the KV cache)
    context_start: int

    @property
    def num_tokens(self) -> int:
        return int(self.keys.shape[2])

    @property
    def num_context_tokens(self) -> int:
        return self.num_tokens - self.context_start

    def output_shape(self) -> tuple[int, int, int]:
        return (int(self.keys.shape[0]), int(self.keys.shape[1]), self.num_context_tokens)
