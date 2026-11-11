#!/usr/bin/env python3
"""Judge entry point (task JSON eval_cmd).

Reads the host's root-only evaluation context and hands the trusted requests to the
evaluation (kvzip_judge/run_judge.py):

* candidate_version - per-version replay: evaluate scorers/scorer_vNNN for this NNN instead of
  the submission's own FINAL_VERSION (set only by the host, never by the submission);
* private KVZIP_REFERENCE_BASELINE=fastkvzip - the trusted Fast KVzip reference baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), "/opt/kvzip/runtime"]

from kvzip_judge.run_judge import main  # noqa: E402

CONTEXT = Path("/tmp/.sforge-evaluation-context.json")


def evaluation_context() -> dict:
    if not CONTEXT.is_file():
        return {}
    try:
        raw = json.loads(CONTEXT.read_text())
    except (OSError, ValueError):
        return {}
    context = {"private": dict(raw.get("private") or {})}
    candidate_version = raw.get("candidate_version")
    if isinstance(candidate_version, int) and not isinstance(candidate_version, bool) and 1 <= candidate_version <= 999:
        context["candidate_version"] = candidate_version
    return context


if __name__ == "__main__":
    sys.exit(main(context=evaluation_context()))
