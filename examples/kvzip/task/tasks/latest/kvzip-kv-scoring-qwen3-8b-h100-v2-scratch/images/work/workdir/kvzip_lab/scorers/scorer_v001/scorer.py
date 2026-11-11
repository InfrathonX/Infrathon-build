"""Your query-agnostic KV importance scorer (see AGENT_TASK.md).

FINAL_VERSION selects the evaluated version; its directory is imported as the package
`scorer`, and this module as `scorer.scorer`, which must define the class `Scorer`.
"""

import torch


class Scorer:
    def __init__(self, config):
        # config: kvzip_eval.api.ScorerConfig (device, scorer_dir, model shape, limits).
        # Load weights (relative to this file) and compile kernels here; not timed.
        self.config = config

    def score(self, states) -> torch.Tensor:
        # states: kvzip_eval.api.PrefillStates for one prefilled context (read-only).
        # Return a float32 tensor [36, 8, states.num_context_tokens] on config.device;
        # higher means keep.
        raise NotImplementedError("write the scorer")
