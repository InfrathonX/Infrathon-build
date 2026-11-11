"""One GPU's evaluation loop: prefill, score (in the scorer process), mask, decode, measure.

Used by the public tests (one GPU, scorer process under the agent's own user)
and by the Judge (one harness per GPU, scorer process under an unprivileged user).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from pathlib import Path

import torch

from . import ipc
from . import protocol as P
from .metrics import score_answer
from .model import RowDecoder, load_model, prefill, timed_prefill
from .suite import Row, Tokenized
from .vmm import SharedBuffer

RUNTIME_DIR = str(Path(__file__).resolve().parents[1])
INIT_TIMEOUT_S = 900.0
CALL_TIMEOUT_S = 300.0
MAX_DECODE_BATCH = 128
WARMUP_TEXT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
WARMUP_CONTEXT_TOKENS = (200, 4000)


class SubmissionInvalid(Exception):
    """The submission broke a rule; `code` names the check, the message is agent-safe."""

    def __init__(self, code: str, message: str, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


@dataclass
class WorkerSpec:
    submission_dir: str
    scratch_dir: str
    uid: int | None = None          # drop to this uid/gid (Judge); None = same user (public)
    gid: int | None = None
    verbose_errors: bool = True     # public: tracebacks; Judge: exception type only
    extra_env: dict = field(default_factory=dict)


class ScorerProcess:
    def __init__(self, spec: WorkerSpec, buffers: dict[str, SharedBuffer], visible_device: str, model_config: dict):
        self.spec = spec
        scratch = Path(spec.scratch_dir)
        for sub in ("home", "tmp", "triton", "cache"):
            (scratch / sub).mkdir(parents=True, exist_ok=True)
        if spec.uid is not None:
            for path in [scratch, *scratch.iterdir()]:
                os.chown(path, spec.uid, spec.gid)
                os.chmod(path, 0o700)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "HOME": str(scratch / "home"), "USER": "scorer", "LOGNAME": "scorer",
            "TMPDIR": str(scratch / "tmp"), "TRITON_HOME": str(scratch / "triton"),
            "TRITON_CACHE_DIR": str(scratch / "triton" / "cache"),
            "TORCHINDUCTOR_CACHE_DIR": str(scratch / "cache" / "inductor"),
            "XDG_CACHE_HOME": str(scratch / "cache"),
            "CUDA_VISIBLE_DEVICES": visible_device, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            **spec.extra_env,
        }
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        code = f"import sys; sys.path.insert(0, {RUNTIME_DIR!r}); from kvzip_eval.worker import main; main()"
        cmd = [sys.executable, "-I", "-c", code, str(child.fileno())]
        if spec.uid is not None:
            cmd = ["setpriv", f"--reuid={spec.uid}", f"--regid={spec.gid}", "--clear-groups", "--no-new-privs", *cmd]
        self.log_path = scratch / "scorer.log"
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(cmd, pass_fds=[child.fileno()], env=env, cwd=str(scratch),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        log.close()
        child.close()
        names = list(buffers)
        socket.send_fds(parent, [b"fds"], [buffers[n].fd for n in names])
        self.conn = Connection(os.dup(parent.fileno()))
        parent.close()
        ipc.send(self.conn, {"buffers": [[n, buffers[n].size] for n in names],
                             "submission_dir": spec.submission_dir, "model_config": model_config,
                             "verbose_errors": spec.verbose_errors})
        self.stopped = False

    @property
    def pid(self) -> int:
        return self.proc.pid

    def _recv(self, timeout: float) -> dict:
        """The next reply, validated (the scorer process is untrusted: JSON only, known fields)."""
        if not self.conn.poll(timeout):
            raise TimeoutError
        try:
            return ipc.checked_reply(ipc.recv(self.conn))
        except ipc.ProtocolError as exc:
            self.kill()
            raise SubmissionInvalid("scorer_protocol", "the scorer process sent an unexpected reply",
                                    str(exc)) from None

    def wait_ready(self) -> dict:
        try:
            msg = self._recv(INIT_TIMEOUT_S)
        except (TimeoutError, EOFError, OSError):
            self.kill()
            raise SubmissionInvalid("scorer_init", "the scorer process did not become ready "
                                    f"(crashed, or construction took longer than {INIT_TIMEOUT_S:.0f} s)",
                                    self.tail_log())
        if msg.get("status") != "ready":
            self.kill()
            raise SubmissionInvalid("scorer_init", "Scorer construction failed: "
                                    f"{ipc.exception_name(msg.get('error'))}", msg.get("detail"))
        self.require(msg, "init_seconds", "peak_bytes")
        self.pause()
        return msg

    def require(self, msg: dict, *keys: str) -> None:
        if any(k not in msg for k in keys):
            self.kill()
            raise SubmissionInvalid("scorer_protocol", "the scorer process sent an unexpected reply",
                                    f"missing {[k for k in keys if k not in msg]}")

    def pause(self):
        os.kill(self.proc.pid, signal.SIGSTOP)
        self.stopped = True

    def resume(self):
        os.kill(self.proc.pid, signal.SIGCONT)
        self.stopped = False

    def call(self, num_tokens: int, context_start: int) -> tuple[dict, float]:
        """One timed score() call.  The process runs only inside this window."""
        self.resume()
        t0 = time.perf_counter()
        ipc.send(self.conn, {"op": "score", "num_tokens": num_tokens, "context_start": context_start})
        try:
            msg = self._recv(CALL_TIMEOUT_S)
        except (TimeoutError, EOFError, OSError):
            self.kill()
            raise SubmissionInvalid("scorer_crash", "the scorer process died or did not answer "
                                    f"within {CALL_TIMEOUT_S:.0f} s", self.tail_log())
        elapsed = time.perf_counter() - t0
        self.pause()
        return msg, elapsed

    def stop(self):
        if self.proc.poll() is None:
            try:
                self.resume()
                ipc.send(self.conn, {"op": "stop"})
                self.proc.wait(timeout=30)
            except Exception:
                self.kill()

    def kill(self):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()

    def tail_log(self, limit: int = 4000) -> str:
        try:  # the scorer owns its scratch directory: never follow a link it planted
            fd = os.open(self.log_path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as f:
                f.seek(max(0, os.fstat(f.fileno()).st_size - limit))
                return f.read(limit).decode("utf-8", "replace")
        except OSError:
            return ""


def _fingerprint(t: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """Secret random projection of a bf16 tensor's bit pattern (exact for unchanged inputs)."""
    x = t.contiguous().view(torch.int16).view(-1, proj.shape[0]).to(torch.float32)
    return (x @ proj).sum(0, dtype=torch.float64)


class ShardRunner:
    def __init__(self, model_dir: str, device: torch.device, max_tokens: int, worker: WorkerSpec,
                 visible_device: str, log=print, use_graphs: bool = True):
        from transformers import AutoTokenizer

        self.device = device
        self.log = log
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = load_model(model_dir, device)
        self.max_tokens = max_tokens
        self.use_graphs = use_graphs
        torch.backends.cuda.matmul.allow_tf32 = False
        L, HKV, HQ, D, HID = P.NUM_LAYERS, P.NUM_KV_HEADS, P.NUM_Q_HEADS, P.HEAD_DIM, P.HIDDEN_SIZE
        dev_index = device.index or 0
        self.buffers = {
            "hidden": SharedBuffer.create(L * max_tokens * HID * 2, dev_index),
            "queries": SharedBuffer.create(L * HQ * max_tokens * D * 2, dev_index),
            "keys": SharedBuffer.create(L * HKV * max_tokens * D * 2, dev_index),
            "values": SharedBuffer.create(L * HKV * max_tokens * D * 2, dev_index),
            "scores": SharedBuffer.create(L * HKV * max_tokens * 4, dev_index),
        }
        self.hidden = self.buffers["hidden"].tensor.view(torch.bfloat16)
        self.queries = self.buffers["queries"].tensor.view(torch.bfloat16)
        self.keys = self.buffers["keys"].tensor.view(torch.bfloat16)
        self.values = self.buffers["values"].tensor.view(torch.bfloat16)
        self.scores = self.buffers["scores"].tensor.view(torch.float32)
        gen = torch.Generator(device=device)
        gen.manual_seed(int.from_bytes(os.urandom(8), "little"))
        self._proj = torch.randn(128, 4, device=device, generator=gen, dtype=torch.float32)
        with open(Path(model_dir) / "config.json") as f:
            model_config = json.load(f)
        self.worker_spec = worker
        self.scorer = ScorerProcess(worker, self.buffers, visible_device, model_config)
        self.ready = self.scorer.wait_ready()
        self.peak_bytes = max(0, self.ready.get("peak_bytes", 0))
        self.reserved_bytes = max(0, self.ready.get("reserved_bytes", 0))
        # Warm the model path and the scorer process (CUDA handles, kernel loading) outside
        # any timing: two untimed score() calls on fixed public filler text; outputs discarded.
        filler = self.tok.encode(WARMUP_TEXT * 400, add_special_tokens=False)
        prefix = self.tok.encode(P.SYSTEM_PREFIX + P.INSTRUCTION_QA, add_special_tokens=False)
        for n_ctx in WARMUP_CONTEXT_TOKENS:
            ids = torch.tensor([prefix + filler[:n_ctx]], device=device)
            k_ctx, v_ctx, _ = timed_prefill(self.model, ids)
            self.score_context(ids, len(prefix), k_ctx, v_ctx, phase="warm-up")
            del k_ctx, v_ctx

    # -- one row ---------------------------------------------------------------------
    def tokenize(self, row: Row) -> Tokenized:
        return Tokenized(self.tok, row)

    def score_context(self, ids: torch.Tensor, context_start: int, k_ref, v_ref,
                      phase: str = "score") -> tuple[torch.Tensor, float]:
        """Capture prefill states, run one timed score() call, check side effects."""
        n = ids.shape[1]
        L, HKV, HQ, D, HID = P.NUM_LAYERS, P.NUM_KV_HEADS, P.NUM_Q_HEADS, P.HEAD_DIM, P.HIDDEN_SIZE
        hidden = self.hidden[: L * n * HID].view(L, n, HID)
        queries = self.queries[: L * HQ * n * D].view(L, HQ, n, D)
        keys = self.keys[: L * HKV * n * D].view(L, HKV, n, D)
        values = self.values[: L * HKV * n * D].view(L, HKV, n, D)
        fps = []

        def capture(layer, h, q, k, v):
            hidden[layer].copy_(h)
            queries[layer].copy_(q)
            keys[layer].copy_(k)
            values[layer].copy_(v)
            fps.append(torch.stack([_fingerprint(h, self._proj), _fingerprint(q, self._proj)]))

        prefill(self.model, ids, capture=capture, keep_cache=False)
        fp_before = torch.stack(fps)
        torch.cuda.synchronize(self.device)
        reply, elapsed = self.scorer.call(n, context_start)
        status = reply.get("status")
        if status == "error":
            where = " during the untimed warm-up call" if phase == "warm-up" else ""
            raise SubmissionInvalid("scorer_exception", f"Scorer.score() raised "
                                    f"{ipc.exception_name(reply.get('error'))}{where}", reply.get("detail"))
        if status == "bad_output":
            # The Judge's message comes from a fixed table; the public tests show the worker's details.
            message = ipc.OUTPUT_PROBLEMS.get(reply.get("kind", ""), "invalid score tensor")
            if self.worker_spec.verbose_errors and reply.get("error"):
                message = reply["error"]
            raise SubmissionInvalid("output", message)
        if status != "ok":
            raise SubmissionInvalid("scorer_protocol", "the scorer process sent an unexpected reply")
        self.scorer.require(reply, "peak_bytes", "rng_changed")
        self.peak_bytes = max(self.peak_bytes, reply["peak_bytes"])
        self.reserved_bytes = max(self.reserved_bytes, reply.get("reserved_bytes", 0))
        if reply["rng_changed"]:
            raise SubmissionInvalid("rng", "Scorer.score() changed the global random number generator state")
        nctx = n - context_start
        scores = self.scores[: L * HKV * nctx].view(L, HKV, nctx).clone()
        changed = any(not torch.equal(keys[l], k_ref[l]) or not torch.equal(values[l], v_ref[l]) for l in range(L))
        if not changed:
            fp_after = torch.stack([torch.stack([_fingerprint(hidden[l], self._proj),
                                                 _fingerprint(queries[l], self._proj)]) for l in range(L)])
            changed = not torch.equal(fp_before, fp_after)
        if changed:
            raise SubmissionInvalid("inputs", "Scorer.score() modified its input tensors")
        if not bool(torch.isfinite(scores).all()):
            raise SubmissionInvalid("output", "score tensor has non-finite values")
        return scores, elapsed

    def run_row(self, row: Row, keep_scores: bool = False) -> dict:
        tk = self.tokenize(row)
        ids = torch.tensor([tk.prefix_ids + tk.context_ids], device=self.device)
        n, start = ids.shape[1], len(tk.prefix_ids)
        if n > self.max_tokens:
            raise RuntimeError(f"row {row.id}: {n} tokens exceed the harness buffers ({self.max_tokens})")
        k_ctx, v_ctx, plain_s = timed_prefill(self.model, ids)
        scores, score_s = self.score_context(ids, start, k_ctx, v_ctx)
        n_scores = scores.numel()
        masks = P.retention_masks(scores)
        keep = torch.ones(P.NUM_LAYERS, len(P.SETTINGS), P.NUM_KV_HEADS, n, dtype=torch.uint8, device=self.device)
        for s_idx, key in enumerate(P.SETTINGS[1:], start=1):
            keep[:, s_idx, :, start:] = masks[key].to(torch.uint8)
        retained = {key: int(masks[key].sum()) for key in masks}
        decoder = RowDecoder(self.model, k_ctx, v_ctx, keep, use_graphs=self.use_graphs)
        elements = [(qi, s) for qi in range(len(row.questions)) for s in range(len(P.SETTINGS))]
        outputs: dict[tuple[int, int], list[int]] = {}
        for c0 in range(0, len(elements), MAX_DECODE_BATCH):
            chunk = elements[c0:c0 + MAX_DECODE_BATCH]
            toks = decoder.run([tk.prompts[qi] for qi, _ in chunk], [s for _, s in chunk], row.max_new_tokens)
            outputs.update(zip(chunk, toks))
        del decoder, keep, k_ctx, v_ctx
        # The scorer process is stopped by now; its output must not have changed since score()
        # returned (work left running after the call would escape the timer).
        if not torch.equal(self.scores[:n_scores], scores.reshape(-1)):
            raise SubmissionInvalid("output", "the score tensor changed after score() returned "
                                    "(work must finish inside the call)")
        acc = {}
        answers = []
        for s_idx, key in enumerate(P.SETTINGS):
            vals = []
            for qi, q in enumerate(row.questions):
                text = self.tok.decode(outputs[(qi, s_idx)], skip_special_tokens=True)
                vals.append(score_answer(row.metric, text, q.refs, q.meta))
                answers.append((qi, key, text, vals[-1]))
            acc[key] = sum(vals) / len(vals)
        result = {"id": row.id, "workload": row.workload, "n_tokens": n, "n_context": n - start,
                  "plain_s": plain_s, "score_s": score_s, "acc": acc, "retained": retained,
                  "answers": answers}
        if keep_scores:
            result["scores"] = scores.cpu()
        return result

    def rescore(self, row: Row) -> torch.Tensor:
        tk = self.tokenize(row)
        ids = torch.tensor([tk.prefix_ids + tk.context_ids], device=self.device)
        k_ctx, v_ctx, _ = timed_prefill(self.model, ids)
        scores, _ = self.score_context(ids, len(tk.prefix_ids), k_ctx, v_ctx)
        return scores.cpu()

    def close(self):
        self.scorer.stop()


def retained_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Smallest fraction of retained entries shared by two scorings of one context."""
    ma, mb = P.retention_masks(a), P.retention_masks(b)
    worst = 1.0
    for key in ma:
        kept = int(ma[key].sum())
        if kept:
            worst = min(worst, int((ma[key] & mb[key]).sum()) / kept)
    return worst
