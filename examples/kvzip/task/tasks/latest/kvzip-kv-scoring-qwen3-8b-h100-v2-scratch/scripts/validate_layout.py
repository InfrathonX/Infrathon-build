#!/usr/bin/env python3
"""Static layout gate: the Work build context holds no Judge material, and the task is complete.

Run by scripts/build_images.sh before any build and by tests/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = [
    "README.md", "tasks/BENCHMARK.yaml", "tasks/kvzip_kv_scoring_qwen3_8b_h100_scratch_v2.json",
    "images/work/Dockerfile", "images/work/AGENT_TASK.md", "images/work/runtime/kvzip_eval/harness.py",
    "images/work/public_tests/run_public_tests.sh", "images/work/workdir/kvzip_lab/scorers/scorer_v001/scorer.py",
    "images/work/workdir/kvzip_lab/FINAL_VERSION",
    "images/judge/Dockerfile", "images/judge/Dockerfile.private", "images/judge/kvzip_judge/run_judge.py",
    "images/judge/kvzip_judge/weights_check.py",
    "images/judge/verify_kvzip.py", "scripts/replay_candidates.sh", "judge/suite_builder/build_suites.py",
    "judge/private/needles.json", "judge/suite_manifest.json", "llm_judge/POLICY.md", "assets/manifest.json",
]
# Names that may only ever exist on the Judge side.
JUDGE_ONLY_NAMES = ["kvzip_judge", "hidden_suite", "needles.json", "q4_dim16_sink16", "suite_builder",
                    "suite_manifest", "released_gates"]
# The hidden suite's workloads that the public suite does not have, and their metrics: the Work
# image must not reveal which task types the hidden suite holds (kvzip_judge/hidden_workloads.py).
HIDDEN_ONLY_TERMS = ["repoqa", "many_shot", "manyshot", "scbench_summary", "rouge", "tree_sitter", "tree-sitter",
                     "option_letter"]


def problems(task_root: Path) -> list[str]:
    out = [f"missing {rel}" for rel in REQUIRED if not (task_root / rel).is_file()]
    work = task_root / "images" / "work"
    hidden_needles = []
    needles = task_root / "judge" / "private" / "needles.json"
    if needles.is_file():
        for item in json.loads(needles.read_text()):
            hidden_needles += [item["needle"].strip(), item["answer"].strip()]
    for path in work.rglob("*"):
        rel = path.relative_to(task_root).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":  # excluded by images/work/.dockerignore
            continue
        if any(name in path.name for name in JUDGE_ONLY_NAMES):
            out.append(f"Judge-only name in the Work build context: {rel}")
        if path.is_file() and path.stat().st_size < (4 << 20):
            text = path.read_bytes().decode("utf-8", "ignore")
            for secret in hidden_needles:
                if secret and secret in text:
                    out.append(f"hidden needle text in the Work build context: {rel}")
            lowered = text.lower()
            for term in HIDDEN_ONLY_TERMS:
                if term in lowered:
                    out.append(f"hidden-only workload term {term!r} in the Work build context: {rel}")
    return out


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    found = problems(root)
    for p in found:
        print("LAYOUT:", p, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
