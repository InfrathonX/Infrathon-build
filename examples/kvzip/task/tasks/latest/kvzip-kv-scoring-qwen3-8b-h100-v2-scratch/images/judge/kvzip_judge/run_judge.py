#!/usr/bin/env python3
"""KVzip KV-scoring Judge: the hidden suite on every GPU of this container.

One harness process per GPU (root) prefills, measures, masks and decodes; each runs
the submitted Scorer in its own process under an unprivileged user that can read
neither the model, the hidden suite nor any Judge file.  The agent-visible result
carries only the score, the three ratio and group scores, the overhead ratios and
the state of every check; per-workload and per-row detail goes to the private
artifact only.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

RUNTIME_DIR = os.environ.get("KVZIP_RUNTIME_DIR", "/opt/kvzip/runtime")
sys.path.insert(0, RUNTIME_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvzip_eval import protocol as P  # noqa: E402
from kvzip_eval.api import MEMORY_LIMIT_BYTES, PROCESS_MEMORY_LIMIT_BYTES  # noqa: E402
from kvzip_eval.aggregate import aggregate, cost_check  # noqa: E402
from kvzip_eval.suite import load_suite, suite_sha256  # noqa: E402

from kvzip_eval.layout import check_tree, prepare_tree, selected_version  # noqa: E402
from kvzip_judge import hidden_workloads  # noqa: E402

# The hidden suite's other workloads and metrics are Judge-only; register them in this process
# and (because shard processes re-import this module) in every shard.
hidden_workloads.register()

START = ">>>>> Start Structured Result"
END = ">>>>> End Structured Result"
CHECKS = ["admission", "weights", "scorer_init", "output", "inputs_unchanged", "rng_unchanged", "scorer_runtime",
          "single_gpu", "cost_total", "cost_per_context", "memory", "stateless", "sanity"]
# rejection codes raised by the harness -> the agent-visible check they fail
CODE_TO_CHECK = {"scorer_exception": "scorer_runtime", "scorer_crash": "scorer_runtime",
                 "scorer_protocol": "scorer_runtime", "inputs": "inputs_unchanged", "rng": "rng_unchanged"}
MEMORY_LIMIT = MEMORY_LIMIT_BYTES                    # peak allocation (the rule)
PROCESS_MEMORY_LIMIT = PROCESS_MEMORY_LIMIT_BYTES    # whole scorer process, sampled from outside (NVML)
STATELESS_MIN_OVERLAP = 0.99
# Fixed texts: the peak is reported by the scorer process itself, so its value is never echoed.
MEMORY_MESSAGE = "the scorer's peak allocation (torch.cuda.max_memory_allocated) exceeds 8 GiB"
PROCESS_MEMORY_MESSAGE = ("the scorer process held more than 19 GiB of device memory in total (PyTorch's cache, "
                          "the CUDA context and anything allocated outside PyTorch)")
STATELESS_MESSAGE = ("re-scoring a context gave different retained entries (the scorer must not carry state "
                     "across contexts)")
SCORER_UID_BASE = 20000
DEADLINE_S = float(os.environ.get("KVZIP_JUDGE_DEADLINE_S", "3300"))


def log(msg: str) -> None:
    print(f"[judge {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------
def shard_main(index: int, gpu_uuid: str, cfg: dict, task_q, result_q) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    runner = None
    try:
        import torch

        from kvzip_eval.harness import ShardRunner, SubmissionInvalid, WorkerSpec, retained_overlap

        rows = load_suite(cfg["suite"])
        uid = cfg["uid_base"] + index if cfg["drop_privileges"] else None
        spec = WorkerSpec(submission_dir=cfg["submission"], scratch_dir=f"{cfg['scratch']}/scorer-{index}",
                          uid=uid, gid=uid, verbose_errors=cfg["verbose_errors"])
        try:
            runner = ShardRunner(cfg["model_dir"], torch.device("cuda:0"), cfg["max_tokens"], spec,
                                 visible_device=gpu_uuid, use_graphs=True)
        except SubmissionInvalid as exc:
            result_q.put(("invalid", index, {"code": exc.code, "message": exc.message, "detail": exc.detail}))
            return
        result_q.put(("ready", index, {"init_seconds": runner.ready.get("init_seconds")}))
        probes = []
        overlaps = []
        min_plain = math.inf
        done_rows = 0
        while True:
            item = task_q.get()
            if item is None:
                break
            row = rows[item]
            res = runner.run_row(row, keep_scores=len(probes) < cfg["probes"])
            if "scores" in res:
                probes.append((item, res.pop("scores")))
            done_rows += 1
            min_plain = min(min_plain, res["plain_s"])
            if runner.peak_bytes > cfg["memory_limit"]:
                raise SubmissionInvalid("memory", MEMORY_MESSAGE, f"peak_bytes={runner.peak_bytes}")
            if done_rows == cfg["early_probe_after"]:
                # Early statelessness probe: re-score this shard's first context after later ones.
                item0, first = probes[0]
                overlaps.append(retained_overlap(first, runner.rescore(rows[item0])))
                if overlaps[-1] < cfg["stateless_min_overlap"]:
                    raise SubmissionInvalid("stateless", STATELESS_MESSAGE)
            if uid is not None:
                extra = _foreign_processes(uid, runner.scorer.pid)
                if extra:
                    raise SubmissionInvalid("single_gpu", "the scorer started other processes")
            limit = max(0.5 * res["plain_s"], 10.0 * min_plain)
            res["answers"] = [(qi, key, text[:300], val) for qi, key, text, val in res["answers"]]
            result_q.put(("row", index, res))
            if res["score_s"] > limit:
                raise SubmissionInvalid("cost_per_context", "a scoring call exceeded its per-context time limit")
        for item, first in probes:
            overlaps.append(retained_overlap(first, runner.rescore(rows[item])))
        result_q.put(("done", index, {"overlaps": overlaps, "peak_bytes": runner.peak_bytes,
                                      "reserved_bytes": runner.reserved_bytes,
                                      "harness_max_reserved": torch.cuda.max_memory_reserved()}))
    except Exception as exc:  # noqa: BLE001
        from kvzip_eval.harness import SubmissionInvalid

        if isinstance(exc, SubmissionInvalid):
            result_q.put(("invalid", index, {"code": exc.code, "message": exc.message, "detail": exc.detail}))
        else:
            result_q.put(("error", index, {"detail": traceback.format_exc()[-6000:]}))
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception:  # noqa: BLE001
                pass


def _foreign_processes(uid: int, allowed_pid: int) -> list[int]:
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == allowed_pid:
            continue
        try:
            with open(f"/proc/{entry}/status") as f:
                for line in f:
                    if line.startswith("Uid:"):
                        if int(line.split()[1]) == uid:
                            found.append(int(entry))
                        break
        except OSError:
            continue
    return found


# --------------------------------------------------------------------------------------------
class GpuMonitor(threading.Thread):
    """Samples per-GPU compute processes (nvidia-smi) for the single-GPU and memory checks."""

    def __init__(self, uuids: list[str], interval: float = 2.0):
        super().__init__(daemon=True)
        self.uuids, self.interval = set(uuids), interval
        self.max_procs = {u: 0 for u in uuids}
        self.max_second_mib = {u: 0 for u in uuids}
        self.max_used_mib = {u: 0 for u in uuids}
        self.foreign_gpu_procs = 0
        self.samples = 0
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=20).stdout
            except Exception:  # noqa: BLE001
                continue
            per = {u: [] for u in self.uuids}
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 3:
                    continue
                if parts[0] in per:
                    try:
                        per[parts[0]].append(int(parts[2]))
                    except ValueError:
                        per[parts[0]].append(0)
            try:
                used = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=20).stdout
                for line in used.strip().splitlines():
                    uuid, mib = [p.strip() for p in line.split(",")]
                    if uuid in self.max_used_mib:
                        self.max_used_mib[uuid] = max(self.max_used_mib[uuid], int(mib))
            except Exception:  # noqa: BLE001
                pass
            self.samples += 1
            for u, mems in per.items():
                self.max_procs[u] = max(self.max_procs[u], len(mems))
                if len(mems) >= 2:
                    self.max_second_mib[u] = max(self.max_second_mib[u], sorted(mems)[-2])


def list_gpus() -> list[str]:
    if os.environ.get("KVZIP_JUDGE_GPU_UUIDS"):  # host tests: restrict to these GPUs
        return os.environ["KVZIP_JUDGE_GPU_UUIDS"].split(",")
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True,
                         text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def estimate_cost(row) -> float:
    n = len(row.context) / 3.5
    return n * (1 + n / 20000) + 60 * row.max_new_tokens * (1 + len(row.questions) / 16)


# --------------------------------------------------------------------------------------------
def _write_private(private: dict, private_path: Path) -> None:
    private_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = private_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(private, indent=1, sort_keys=True, default=str))
    os.chmod(tmp, 0o600)
    tmp.replace(private_path)


def emit(result: dict, private: dict, private_path: Path) -> None:
    _write_private(private, private_path)
    print(START)
    print(json.dumps(result, sort_keys=True))
    print(END, flush=True)


def build_result(valid: bool, reason: str | None, checks: dict, agg: dict | None, cost: dict | None,
                 extra: dict | None = None) -> dict:
    """The agent-visible result: exactly the definition's feedback (score, ratio and group scores,
    overhead ratios, check states, rejection reason). Everything else stays in the private artifact."""
    metrics = {"evaluation_status": "valid" if valid else "invalid", "rejection_reason": reason,
               "checks": {c: checks.get(c, "not_run") for c in CHECKS}}
    if agg is not None and valid:  # an invalid submission gets no quality numbers (fail closed)
        metrics["ratio_scores"] = {k: round(v, 3) for k, v in agg["ratio_scores"].items()}
        metrics["group_scores"] = {k: round(v, 3) for k, v in agg["group_scores"].items()}
    if cost is not None:
        metrics["overhead"] = {"total_scoring_over_prefill": round(cost["total_ratio"], 4),
                               "total_limit": 0.25,
                               "worst_context_over_its_limit": round(cost["worst_context_ratio_to_limit"], 4),
                               "contexts_over_limit": cost["contexts_over_limit"]}
    metrics.update(extra or {})
    score = round(agg["score"], 4) if (valid and agg is not None) else 0.0
    metrics["score"] = score
    metrics["primary_score_name"] = "score"
    metrics["primary_score_unit"] = "(100 = as good as the full cache)"
    metrics["primary_score_formula"] = "100 x mean over r in {0.1,0.2,0.3} of mean over 3 groups of acc(task,r)/acc(task,full)"
    details = [{"name": c, "status": "FAILED" if checks.get(c) == "fail" else "PASSED", "message": None}
               for c in CHECKS if checks.get(c) in ("pass", "fail", "fired_reference_baseline")]
    summary = (f"valid; score {score:.2f}" if valid else f"invalid: {reason}")
    return {"valid": valid, "score": score, "summary": summary, "metrics": metrics, "details": details}


def main(context: dict | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", default="/app")
    ap.add_argument("--log-dir", default="/logs/verifier")
    ap.add_argument("--submission", default=None, help="submitted tree (default: the Work cwd under --app-dir)")
    ap.add_argument("--suite", default="/opt/kvzip_judge/private/hidden_suite.jsonl.gz")
    ap.add_argument("--suite-sha256", default=os.environ.get("KVZIP_HIDDEN_SUITE_SHA256", ""))
    ap.add_argument("--model-dir", default="/opt/kvzip_judge/private/models/qwen3-8b")
    ap.add_argument("--released-gates", default="/opt/kvzip_judge/private/released_gates/q4_dim16_sink16.pt")
    ap.add_argument("--scratch", default="/tmp/kvzip-judge")
    ap.add_argument("--private-name", default="kvzip_judge_private.json")
    ap.add_argument("--no-drop-privileges", action="store_true", help="host tests only")
    ap.add_argument("--limit-rows", type=int, default=0, help="host tests only")
    ap.add_argument("--reference-baseline", default="", help="host tests only (the Judge reads the private context)")
    args = ap.parse_args()
    t_start = time.time()
    private_path = Path(args.log_dir) / args.private_name
    checks: dict[str, str] = {}
    private: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def reject(code: str, message: str, detail=None, agg=None, cost=None, peak=None) -> int:
        checks[CODE_TO_CHECK.get(code, code)] = "fail"
        private["rejection"] = {"code": code, "message": message, "detail": detail}
        private.update(scorer_peak_bytes=peak, judge_seconds=round(time.time() - t_start, 1))
        emit(build_result(False, message, checks, agg, cost), private, private_path)
        return 0

    context = context or {}  # from verify_kvzip.py (the host's evaluation context)
    reference = (context.get("private") or {}).get("KVZIP_REFERENCE_BASELINE", "") or args.reference_baseline
    private["reference_baseline"] = reference or None

    src = Path(args.submission or (Path(args.app_dir) / "workdir" / "kvzip_lab"))
    override = context.get("candidate_version")  # per-version replay (host-side, trusted)
    problems = check_tree(src, override)
    if problems:
        return reject("admission", "submission layout: " + "; ".join(problems))
    version, _ = selected_version(src, override)
    private["evaluated_version"] = version
    checks["admission"] = "pass"

    # read-only copy the scorer processes can read; the Judge never runs code from src itself
    scratch = Path(args.scratch)
    if scratch.exists():
        shutil.rmtree(scratch)
    sub = prepare_tree(src, scratch / "submission", version, copy=True)
    if reference == "fastkvzip":
        (sub / "scorer" / "released_gates").mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.released_gates, sub / "scorer" / "released_gates" / "q4_dim16_sink16.pt")

    gpus = list_gpus()
    if not gpus:
        return reject("scorer_runtime", "no GPU visible to the Judge")
    scan_env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus[0]}
    # every submitted version is scanned, and the evaluated copy (which holds an injected reference gate)
    scan_dir = scratch / "scan"
    scan_dir.mkdir()
    if (src / "scorers").is_dir():
        shutil.copytree(src / "scorers", scan_dir / "scorers", symlinks=True)
    shutil.copytree(sub / "scorer", scan_dir / "evaluated", symlinks=True)
    scan = subprocess.run([sys.executable, "-m", "kvzip_judge.weights_check", "--submission", str(scan_dir),
                           "--released", args.released_gates], capture_output=True, text=True, env=scan_env,
                          cwd=str(Path(__file__).resolve().parents[1]), timeout=1200)
    try:
        scan_result = json.loads(scan.stdout)
    except ValueError:
        scan_result = {"unreadable": f"weights scan failed: {scan.stderr[-2000:]}"}
    private["weights_scan"] = scan_result
    if "unreadable" in scan_result:
        return reject("weights", f"weight file not readable: {scan_result['unreadable']}")
    if scan_result["uses_released_gates"]:
        if reference != "fastkvzip":
            return reject("weights", "submitted weights match the released Fast KVzip gates")
        checks["weights"] = "fired_reference_baseline"
    else:
        checks["weights"] = "pass"

    shutil.rmtree(scan_dir)
    for path in [scratch, *sub.rglob("*")]:
        os.chmod(path, 0o755 if path.is_dir() else 0o644)

    expected = args.suite_sha256
    if not expected:
        manifest = Path(args.suite).parent / "suite_manifest.json"
        expected = json.loads(manifest.read_text())["hidden"]["sha256"] if manifest.is_file() else ""
    if not expected or suite_sha256(args.suite) != expected:
        print("INFRASTRUCTURE FAILURE: the hidden suite is missing or differs from its frozen identity",
              file=sys.stderr, flush=True)
        return 3
    rows = load_suite(args.suite)
    private["suite_sha256"] = suite_sha256(args.suite)
    order = sorted(range(len(rows)), key=lambda i: -estimate_cost(rows[i]))
    if args.limit_rows:
        order = order[: args.limit_rows]
    # One of the shortest contexts first on every GPU: the per-context cost limit uses the suite's
    # shortest prefill, and an early short prefill makes the fail-fast bound tight from the start.
    shortest = sorted(order, key=lambda i: len(rows[i].context))[: len(list_gpus())]
    order = shortest + [i for i in order if i not in set(shortest)]
    max_tokens = 32128

    ctx = mp.get_context("spawn")
    task_q, result_q = ctx.Queue(), ctx.Queue()
    for i in order:
        task_q.put(i)
    for _ in gpus:
        task_q.put(None)
    cfg = {"suite": args.suite, "submission": str(sub), "scratch": str(scratch), "model_dir": args.model_dir,
           "max_tokens": max_tokens, "drop_privileges": not args.no_drop_privileges, "uid_base": SCORER_UID_BASE,
           "verbose_errors": False, "probes": 2, "memory_limit": MEMORY_LIMIT, "early_probe_after": 8,
           "stateless_min_overlap": STATELESS_MIN_OVERLAP}
    monitor = GpuMonitor(gpus)
    monitor.start()
    procs = [ctx.Process(target=shard_main, args=(i, g, cfg, task_q, result_q), daemon=True)
             for i, g in enumerate(gpus)]
    for p in procs:
        p.start()
    log(f"{len(gpus)} GPUs, {len(order)} rows")
    results, done, failure = [], {}, None
    while len(done) < len(procs):
        if time.time() - t_start > DEADLINE_S:
            failure = ("scorer_runtime", "the evaluation did not finish within the Judge time limit", None)
            break
        if any(n > 2 for n in monitor.max_procs.values()):
            failure = ("single_gpu", "a GPU ran more than the Judge's own process and its scorer", None)
            break
        if any((mib << 20) > PROCESS_MEMORY_LIMIT for mib in monitor.max_second_mib.values()):
            failure = ("memory", PROCESS_MEMORY_MESSAGE, dict(monitor.max_second_mib))
            break
        try:
            kind, index, payload = result_q.get(timeout=10)
        except Exception:  # noqa: BLE001
            dead = [i for i, p in enumerate(procs) if not p.is_alive() and i not in done]
            if dead:
                failure = ("infrastructure", f"Judge shard {dead[0]} exited unexpectedly", None)
                break
            continue
        if kind == "row":
            results.append(payload)
            if len(results) % 100 == 0:
                log(f"{len(results)}/{len(order)} rows")
        elif kind == "ready":
            log(f"shard {index} ready (scorer init {payload.get('init_seconds', 0):.1f}s)")
        elif kind == "done":
            done[index] = payload
        elif kind == "invalid":
            failure = (payload["code"], payload["message"], payload.get("detail"))
            break
        elif kind == "error":
            failure = ("infrastructure", "Judge shard failed", payload.get("detail"))
            break
    monitor.stop_event.set()
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=60)
    subprocess.run(["pkill", "-9", "-f", "kvzip_eval.worker"], capture_output=True)
    private["rows"] = results
    private["gpus"] = gpus
    private["gpu_monitor"] = {"samples": monitor.samples, "max_processes": monitor.max_procs,
                              "max_second_process_mib": monitor.max_second_mib,
                              "max_device_used_mib": monitor.max_used_mib}
    if failure is not None:
        code, message, detail = failure
        if code == "infrastructure":
            private["infrastructure_failure"] = {"message": message, "detail": detail}
            print(f"INFRASTRUCTURE FAILURE: {message}\n{detail or ''}", file=sys.stderr, flush=True)
            _write_private(private, private_path)
            return 3
        return reject(code, message, detail)
    for c in ("scorer_init", "output", "inputs_unchanged", "rng_unchanged", "scorer_runtime"):
        checks[c] = "pass"

    peak = max([int(d["peak_bytes"]) for d in done.values()] + [0])
    agg = aggregate(results)
    cost = cost_check([(r["id"], r["plain_s"], r["score_s"]) for r in results])
    private["aggregate"] = agg
    private["cost"] = cost
    private["stateless_overlaps"] = {i: d["overlaps"] for i, d in done.items()}
    private["harness_max_reserved_bytes"] = {i: d.get("harness_max_reserved") for i, d in done.items()}
    if any(n > 2 for n in monitor.max_procs.values()):
        return reject("single_gpu", "a GPU ran more than the Judge's own process and its scorer", agg=agg,
                      cost=cost, peak=peak)
    checks["single_gpu"] = "pass"
    checks["cost_total"] = "pass" if cost["total_ok"] else "fail"
    checks["cost_per_context"] = "pass" if cost["per_context_ok"] else "fail"
    if not cost["total_ok"]:
        return reject("cost_total", f"total scoring time is {100 * cost['total_ratio']:.1f}% of total prefill "
                      "time (limit 25%)", agg=agg, cost=cost, peak=peak)
    if not cost["per_context_ok"]:
        return reject("cost_per_context", f"{cost['contexts_over_limit']} scoring call(s) exceeded the "
                      "per-context limit", agg=agg, cost=cost, peak=peak)
    nvml_worker = max(monitor.max_second_mib.values()) << 20
    private["scorer_reserved_bytes"] = max([int(d.get("reserved_bytes", 0)) for d in done.values()] + [0])
    private["scorer_process_bytes"] = nvml_worker
    if peak > MEMORY_LIMIT:
        return reject("memory", MEMORY_MESSAGE, f"peak_bytes={peak}", agg=agg, cost=cost, peak=peak)
    if nvml_worker > PROCESS_MEMORY_LIMIT:
        return reject("memory", PROCESS_MEMORY_MESSAGE, f"nvml_worker_bytes={nvml_worker}", agg=agg, cost=cost,
                      peak=peak)
    checks["memory"] = "pass"
    overlaps = [o for d in done.values() for o in d["overlaps"]]
    if overlaps and min(overlaps) < STATELESS_MIN_OVERLAP:
        return reject("stateless", STATELESS_MESSAGE, agg=agg, cost=cost, peak=peak)
    checks["stateless"] = "pass"
    if agg["zero_full_accuracy_tasks"] or not agg["finite"]:
        # The full-cache accuracies are the evaluator's own denominators: a failure here is an
        # evaluation fault, reported as an infrastructure error (no score), never as a valid result.
        private["infrastructure_failure"] = {"message": "sanity check failed", "aggregate": agg}
        print("INFRASTRUCTURE FAILURE: sanity check failed: zero full-cache accuracy in "
              f"{agg['zero_full_accuracy_tasks']} or non-finite metrics", file=sys.stderr, flush=True)
        _write_private(private, private_path)
        return 3
    checks["sanity"] = "pass"
    private.update(scorer_peak_bytes=peak, judge_seconds=round(time.time() - t_start, 1))
    emit(build_result(True, None, checks, agg, cost), private, private_path)
    log(f"score {agg['score']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
