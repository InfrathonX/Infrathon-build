#!/usr/bin/env python3
"""Public tests: the evaluation protocol on the public suite, on this container's GPU.

Same prefill, scorer process, retention rule, masked decoding, metrics and checks
as the Judge, on public data.  Diagnostics only: the public score does not predict
the hidden score, and nothing here is recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.environ.get("KVZIP_RUNTIME_DIR", "/opt/kvzip/runtime"))

import torch  # noqa: E402

from kvzip_eval import protocol as P  # noqa: E402
from kvzip_eval.aggregate import aggregate, cost_check  # noqa: E402
from kvzip_eval.harness import ShardRunner, SubmissionInvalid, WorkerSpec, retained_overlap  # noqa: E402
from kvzip_eval.layout import check_tree, prepare_tree, selected_version  # noqa: E402
from kvzip_eval.suite import load_suite  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "public_suite.jsonl.gz"
MEMORY_LIMIT = 8 << 30


def select(rows, tier):
    if tier == "one":
        return [r for r in rows if r.workload == "squad"][:1]
    if tier == "quick":
        seen = defaultdict(int)
        out = []
        for r in rows:
            if seen[r.workload] < 2:
                seen[r.workload] += 1
                out.append(r)
        return out
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=["one", "quick", "full"], default="quick")
    ap.add_argument("--submission", default=os.getcwd(),
                    help="tree holding scorers/ and FINAL_VERSION (default: cwd)")
    ap.add_argument("--version", type=int, default=None, help="evaluate scorers/scorer_vNNN instead of FINAL_VERSION")
    ap.add_argument("--scorer-dir", default=None, help="evaluate this directory (holding scorer.py) directly")
    ap.add_argument("--workload", action="append", default=[], help="only these workloads (repeatable)")
    ap.add_argument("--model-dir", default=os.environ.get("KVZIP_MODEL_DIR", "/opt/models/qwen3-8b"))
    ap.add_argument("--out", default=None, help="results JSON (default: results/ next to this script)")
    ap.add_argument("--show-answers", type=int, default=2, help="answers printed per row and setting")
    args = ap.parse_args()

    scratch = f"/tmp/kvzip-public-{os.getpid()}"
    if args.scorer_dir:
        tree = Path(scratch) / "tree"
        tree.mkdir(parents=True)
        os.symlink(os.path.abspath(args.scorer_dir), tree / "scorer")
        label = f"scorer dir {args.scorer_dir}"
    else:
        problems = check_tree(args.submission, args.version)
        if problems:
            print("ADMISSION FAILED (the Judge would reject this tree):")
            for p in problems:
                print("  -", p)
            return 1
        version, _ = selected_version(args.submission, args.version)
        tree = prepare_tree(args.submission, Path(scratch) / "tree", version, copy=False)
        label = f"{args.submission}, scorers/scorer_v{version:03d}"
    rows = select(load_suite(DATA), args.tier)
    if args.workload:
        rows = [r for r in rows if r.workload in args.workload]
    if not rows:
        print("no rows selected")
        return 1
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    longest = 0
    for r in rows:
        prefix = tok.encode(P.SYSTEM_PREFIX + P.instruction_for(r.workload), add_special_tokens=False)
        longest = max(longest, len(prefix) + len(tok.encode(r.context, add_special_tokens=False)))
    max_tokens = max(4200, (longest + 127) // 128 * 128)
    print(f"public tier '{args.tier}': {len(rows)} rows; {label}")
    t0 = time.time()
    try:
        runner = ShardRunner(args.model_dir, torch.device("cuda:0"), max_tokens,
                             WorkerSpec(submission_dir=str(tree), scratch_dir=f"{scratch}/scorer"),
                             visible_device=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    except SubmissionInvalid as exc:
        print(f"INVALID [{exc.code}]: {exc.message}")
        if exc.detail:
            print(exc.detail)
        return 1
    print(f"scorer ready: construction {runner.ready['init_seconds']:.1f} s, "
          f"allocation after construction {runner.ready['peak_bytes'] / 2**30:.3f} GiB")
    results, invalid = [], None
    first = None
    try:
        for r in rows:
            res = runner.run_row(r, keep_scores=first is None)
            if "scores" in res:
                first = (r, res.pop("scores"))
            results.append(res)
            acc = " ".join(f"{k}={v:.3f}" for k, v in res["acc"].items())
            print(f"{r.id:46s} ctx={res['n_context']:6d}  prefill {1e3 * res['plain_s']:8.1f} ms  "
                  f"score {1e3 * res['score_s']:7.1f} ms  {acc}")
            shown = defaultdict(int)
            for qi, key, text, val in res["answers"]:
                if shown[key] < args.show_answers:
                    shown[key] += 1
                    print(f"      [{key:>4}] {val:.2f} q{qi}: {text[:120]!r}")
        overlap = retained_overlap(first[1], runner.rescore(first[0])) if first else 1.0
    except SubmissionInvalid as exc:
        invalid = exc
    finally:
        runner.close()
    if invalid is not None:
        print(f"\nINVALID [{invalid.code}]: {invalid.message}")
        if invalid.detail:
            print(invalid.detail)
        return 1
    agg = aggregate(results, drop_zero_full=True)
    cost = cost_check([(r["id"], r["plain_s"], r["score_s"]) for r in results])
    print("\nper workload (accuracy: full / 0.1 / 0.2 / 0.3; relative to full):")
    for t, a in sorted(agg["task_accuracy"].items()):
        if t not in agg["task_relative"]:
            print(f"  {t:24s} {a['full']:.3f} / {a['0.1']:.3f} / {a['0.2']:.3f} / {a['0.3']:.3f}   "
                  "left out of the public score: full-cache accuracy 0 on these rows")
            continue
        rel = agg["task_relative"][t]
        print(f"  {t:24s} {a['full']:.3f} / {a['0.1']:.3f} / {a['0.2']:.3f} / {a['0.3']:.3f}   "
              f"rel {rel['0.1']:.3f} / {rel['0.2']:.3f} / {rel['0.3']:.3f}")
    print("groups:", {g: round(v, 2) for g, v in agg["group_scores"].items()})
    print("ratios:", {r: round(v, 2) for r, v in agg["ratio_scores"].items()})
    print(f"PUBLIC SCORE ({args.tier}, groups present: {sorted(agg['group_scores'])}): {agg['score']:.3f}")
    print(f"cost: total scoring / prefill = {cost['total_ratio']:.4f} (limit 0.25); worst context at "
          f"{cost['worst_context_ratio_to_limit']:.3f} of its limit ({cost['worst_context_id']}); "
          f"floor = 10 x shortest prefill here = {1e3 * cost['per_context_floor_s']:.1f} ms")
    print(f"memory: peak scorer allocation {runner.peak_bytes / 2**30:.3f} GiB (limit 8 GiB); "
          f"PyTorch reserved at most {runner.reserved_bytes / 2**30:.3f} GiB (capped at 16 GiB; the "
          "whole scorer process, CUDA context and anything allocated outside PyTorch included, may hold 19 GiB)")
    print(f"stateless re-score overlap: {overlap:.4f} (Judge requires >= 0.99)")
    warnings = []
    if not cost["total_ok"]:
        warnings.append("total scoring time exceeds 25% of prefill")
    if not cost["per_context_ok"]:
        warnings.append(f"{cost['contexts_over_limit']} context(s) over the per-context limit")
    if runner.peak_bytes > MEMORY_LIMIT:
        warnings.append("peak allocation over 8 GiB")
    if overlap < 0.99:
        warnings.append("re-scoring a context changed its retained entries")
    print("CHECKS:", "PASS" if not warnings else "FAIL: " + "; ".join(warnings))
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "results" / f"public-{args.tier}-{int(t0)}.json"
    try:
        out.write_text(json.dumps({"tier": args.tier, "aggregate": agg, "cost": cost, "rows": results,
                                   "peak_bytes": runner.peak_bytes, "stateless_overlap": overlap},
                                  indent=1, default=str))
        print(f"details: {out}")
    except OSError:
        pass
    print(f"wall time {time.time() - t0:.0f} s")
    return 0 if not warnings and math.isfinite(agg["score"]) else 1


if __name__ == "__main__":
    sys.exit(main())
