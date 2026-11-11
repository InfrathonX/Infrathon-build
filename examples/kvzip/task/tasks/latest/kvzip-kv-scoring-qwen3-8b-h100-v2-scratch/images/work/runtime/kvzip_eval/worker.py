"""The scorer process.  Runs the submitted Scorer and nothing else.

Started by the harness (in the Judge as an unprivileged user that can read
neither the model, the hidden suite, nor any Judge file).  It receives the
per-layer prefill states through shared device buffers and writes the score
tensor into a shared output buffer.  Every score() call is timed by the
harness, which stops this process between calls.
"""

from __future__ import annotations

import importlib
import os
import random
import socket
import sys
import time
import traceback
from multiprocessing.connection import Connection

import torch

from . import ipc
from .api import CACHE_LIMIT_BYTES, PrefillStates, ScorerConfig
from .vmm import SharedBuffer

L, HKV, HQ, D, HID = 36, 8, 32, 128, 4096


def _rng_state():
    state = [torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate()]
    np = sys.modules.get("numpy")
    if np is not None:
        state.append(np.random.get_state())
    return state


def _same_rng(a, b) -> bool:
    if len(a) != len(b):
        return len(a) == 3 and len(b) == 4  # numpy imported during the call: nothing to compare
    if not (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]):
        return False
    if len(a) == 4:
        np = sys.modules["numpy"]
        return (a[3][0] == b[3][0] and np.array_equal(a[3][1], b[3][1])
                and tuple(a[3][2:]) == tuple(b[3][2:]))
    return True


def _error_name(exc: BaseException) -> str:
    """The exception's class name; the harness shows it only if it is a builtin, PyTorch, NumPy or
    Triton exception name (anything else is reported as another type)."""
    return str(type(exc).__name__)[:200]


def main() -> None:
    sock = socket.socket(fileno=int(sys.argv[1]))
    _, fds, _, _ = socket.recv_fds(sock, 64, 8)
    conn = Connection(os.dup(sock.fileno()))
    init = ipc.recv(conn)
    verbose = bool(init["verbose_errors"])
    torch.cuda.init()
    device = torch.device("cuda:0")
    torch.zeros(1, device=device)
    # Bound PyTorch's cache (it frees cached blocks before it reaches the cap), so fragmentation
    # cannot grow the process past its device memory limit.
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, CACHE_LIMIT_BYTES / total), device)
    bufs = {}
    for (name, size), fd in zip(init["buffers"], fds):
        bufs[name] = SharedBuffer.open(fd, size).tensor
        os.close(fd)
    hidden = bufs["hidden"].view(torch.bfloat16)
    queries = bufs["queries"].view(torch.bfloat16)
    keys = bufs["keys"].view(torch.bfloat16)
    values = bufs["values"].view(torch.bfloat16)
    out = bufs["scores"].view(torch.float32)
    # Warm the kernels this runner itself uses so they are not charged to a call.
    probe = torch.ones(4, 4, device=device)
    torch.isfinite(probe).all().item()
    out[:16].copy_(probe.view(-1))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()

    submission_dir = init["submission_dir"]
    sys.path.insert(0, submission_dir)
    config = ScorerConfig(device=device, scorer_dir=os.path.join(submission_dir, "scorer"),
                          submission_dir=submission_dir, model_config=init["model_config"])
    t0 = time.perf_counter()
    try:
        module = importlib.import_module("scorer.scorer")
        scorer = module.Scorer(config)
        torch.cuda.synchronize()
    except BaseException as exc:  # noqa: BLE001 - every failure is reported, never raised
        ipc.send(conn, {"status": "error", "phase": "init", "error": _error_name(exc),
                        "detail": traceback.format_exc(limit=12)[-4000:]})
        return
    ipc.send(conn, {"status": "ready", "init_seconds": time.perf_counter() - t0,
                    "peak_bytes": torch.cuda.max_memory_allocated() - base_alloc,
                    "reserved_bytes": torch.cuda.max_memory_reserved()})

    while True:
        req = ipc.recv(conn)
        if req["op"] == "stop":
            ipc.send(conn, {"status": "stopped"})
            return
        n, start = int(req["num_tokens"]), int(req["context_start"])
        nctx = n - start
        states = PrefillStates(
            hidden_states=hidden[: L * n * HID].view(L, n, HID),
            queries=queries[: L * HQ * n * D].view(L, HQ, n, D),
            keys=keys[: L * HKV * n * D].view(L, HKV, n, D),
            values=values[: L * HKV * n * D].view(L, HKV, n, D),
            context_start=start,
        )
        before = _rng_state()
        reply = {"status": "ok"}
        try:
            result = scorer.score(states)
            if not isinstance(result, torch.Tensor):
                reply = {"status": "bad_output", "kind": "type",
                         "error": f"score() returned {type(result).__name__}, not a torch.Tensor"}
            elif result.dtype != torch.float32:
                reply = {"status": "bad_output", "kind": "dtype",
                         "error": f"score dtype {result.dtype}, expected torch.float32"}
            elif tuple(result.shape) != (L, HKV, nctx):
                reply = {"status": "bad_output", "kind": "shape",
                         "error": f"score shape {tuple(result.shape)}, expected {(L, HKV, nctx)}"}
            elif result.device != device:
                reply = {"status": "bad_output", "kind": "device",
                         "error": f"score device {result.device}, expected {device}"}
            elif not bool(torch.isfinite(result).all()):
                reply = {"status": "bad_output", "kind": "nonfinite", "error": "score tensor has non-finite values"}
            else:
                out[: L * HKV * nctx].view(L, HKV, nctx).copy_(result)
            del result
            torch.cuda.synchronize()
        except BaseException as exc:  # noqa: BLE001
            reply = {"status": "error", "phase": "score", "error": _error_name(exc)}
            if verbose:
                reply["detail"] = traceback.format_exc(limit=12)[-4000:]
        reply["rng_changed"] = not _same_rng(before, _rng_state())
        reply["peak_bytes"] = torch.cuda.max_memory_allocated() - base_alloc
        reply["reserved_bytes"] = torch.cuda.max_memory_reserved()
        ipc.send(conn, reply)


if __name__ == "__main__":
    main()
