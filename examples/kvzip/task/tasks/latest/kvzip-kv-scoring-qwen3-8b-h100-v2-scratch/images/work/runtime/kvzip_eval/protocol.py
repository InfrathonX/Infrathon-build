"""The evaluation protocol: prompt layout, retention rule, workloads and groups."""

from __future__ import annotations

import torch

NUM_LAYERS = 36
NUM_KV_HEADS = 8
NUM_Q_HEADS = 32
HEAD_DIM = 128
HIDDEN_SIZE = 4096
RATIOS = (0.1, 0.2, 0.3)
# Decoding settings, in batch order: full cache first, then the three ratios.
SETTINGS = ("full", "0.1", "0.2", "0.3")
RATIO_TENTHS = {"0.1": 1, "0.2": 2, "0.3": 3}
MAX_CONTEXT_TOKENS = 32000
MIN_CONTEXT_TOKENS = 64
EOS_TOKEN_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>

SYSTEM_PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
INSTRUCTION_QA = "Given the context, answer to the following question or request without explanation.\n\n"
INSTRUCTION_REASON = "Given the context, answer to the following reasoning question.\n\n"
ASSISTANT_POSTFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# workload -> (group, metric, max new tokens): the public workloads. The Judge registers the
# hidden suite's other workloads (and their metrics) itself; they are not part of this image.
WORKLOADS = {
    "needle": ("retrieval", "include_normalized", 48),
    "scbench_kv": ("retrieval", "include_normalized", 96),
    "scbench_prefix_suffix": ("retrieval", "include_normalized", 128),
    "ruler_niah": ("retrieval", "ruler_all", 64),
    "squad": ("contextual_qa", "include_normalized", 256),
    "gsm": ("contextual_qa", "numeric_inclusion", 512),
    "ruler_qa": ("contextual_qa", "ruler_any", 32),
    "scbench_mf": ("redundancy", "exact_match", 48),
    "ruler_vt": ("redundancy", "ruler_all", 48),
}
GROUPS = ("retrieval", "contextual_qa", "redundancy")


def register_workloads(extra: dict[str, tuple[str, str, int]]) -> None:
    """Add workloads (group, metric, max new tokens) beyond the public ones."""
    for name, spec in extra.items():
        if spec[0] not in GROUPS or WORKLOADS.get(name, spec) != spec:
            raise ValueError(f"conflicting workload {name}: {spec}")
        WORKLOADS[name] = spec


def instruction_for(workload: str) -> str:
    return INSTRUCTION_REASON if workload == "gsm" else INSTRUCTION_QA


def question_text(question: str) -> str:
    """The text appended after the context: closes the user turn's content."""
    return "\n\n" + question.strip()


def keep_count(ratio_key: str, n_entries: int) -> int:
    """round(r * N) for r in {0.1, 0.2, 0.3}; N = 36 * 8 * n_ctx never makes it a tie."""
    tenths = RATIO_TENTHS[ratio_key]
    return (tenths * n_entries + 5) // 10


def retention_masks(scores: torch.Tensor) -> dict[str, torch.Tensor]:
    """One global ranking of a [36, 8, n_ctx] score tensor.

    Higher scores are kept first; ties keep the lower flat index in
    (layer, KV head, position) order.  Returns bool masks [36, 8, n_ctx] for
    each ratio key.  The retained sets are nested.
    """
    if scores.dim() != 3:
        raise ValueError("scores must be [layers, kv_heads, context]")
    flat = scores.reshape(-1)
    n = flat.numel()
    order = torch.sort(flat, descending=True, stable=True).indices
    rank = torch.empty_like(order)
    rank[order] = torch.arange(n, device=flat.device)
    masks = {}
    for key in RATIO_TENTHS:
        k = keep_count(key, n)
        masks[key] = (rank < k).view_as(scores)
    return masks
