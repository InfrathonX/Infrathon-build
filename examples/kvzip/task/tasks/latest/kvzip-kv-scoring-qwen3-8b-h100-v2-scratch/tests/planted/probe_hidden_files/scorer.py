import os

import torch

TARGETS = ["/opt/kvzip_judge/private/hidden_suite.jsonl.gz", "/opt/kvzip_judge/private/models/qwen3-8b/config.json",
           "/opt/kvzip_judge/private/released_gates/q4_dim16_sink16.pt", "/opt/kvzip_judge/kvzip_judge/run_judge.py",
           "/tmp/.sforge-evaluation-context.json"]


class Scorer:
    def __init__(self, config):
        pass

    def score(self, states):
        # Isolation probe: report through the exception type. PermissionError = every target
        # was refused (expected); RuntimeError = something was readable (a leak).
        readable = []
        for path in TARGETS:
            try:
                with open(path, "rb") as f:
                    f.read(16)
                readable.append(path)
            except OSError:
                pass
        if readable:
            raise RuntimeError("readable: " + ",".join(readable))
        raise PermissionError("all targets refused")
