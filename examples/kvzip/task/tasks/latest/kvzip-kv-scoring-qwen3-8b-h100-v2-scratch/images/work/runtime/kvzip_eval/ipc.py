"""Messages between the harness and the scorer process: length-bounded JSON objects.

The scorer process runs submitted code, so the harness never unpickles what it sends, accepts
only the reply fields and types below, and chooses every agent-visible word itself (exception
names from a fixed list, output problems from a fixed table).
"""

from __future__ import annotations

import builtins
import importlib
import json

MAX_MESSAGE_BYTES = 1 << 16
REPLY_STATUSES = frozenset({"ready", "error", "ok", "bad_output", "stopped"})
REPLY_FIELDS = {"status": str, "phase": str, "error": str, "kind": str, "detail": str,
                "init_seconds": float, "peak_bytes": int, "reserved_bytes": int, "rng_changed": bool}
OUTPUT_PROBLEMS = {
    "type": "score() did not return a torch.Tensor",
    "dtype": "score dtype is not torch.float32",
    "shape": "score shape is not [36, 8, context tokens]",
    "device": "score tensor is not on the scorer's device",
    "nonfinite": "score tensor has non-finite values",
}
OTHER_EXCEPTION = "an exception of a type outside Python, PyTorch, NumPy and Triton"
_EXCEPTION_MODULES = ("builtins", "torch", "torch.cuda", "torch._C", "torch.linalg", "numpy", "numpy.linalg",
                      "triton.compiler.errors", "triton.runtime.errors")
_safe_names: frozenset[str] | None = None


class ProtocolError(Exception):
    """The scorer process sent something that is not a well-formed reply."""


def send(conn, obj: dict) -> None:
    conn.send_bytes(json.dumps(obj).encode())


def recv(conn) -> dict:
    """One message; EOFError/OSError if the peer is gone or oversized, ProtocolError if malformed."""
    data = conn.recv_bytes(MAX_MESSAGE_BYTES)
    try:
        msg = json.loads(data)
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("not a JSON message") from exc
    if not isinstance(msg, dict):
        raise ProtocolError("not a JSON object")
    return msg


def checked_reply(msg: dict) -> dict:
    """A reply from the scorer process restricted to known fields of the right types."""
    out = {}
    for key, value in msg.items():
        kind = REPLY_FIELDS.get(key)
        if kind is None:
            raise ProtocolError("unknown reply field")
        if kind is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise ProtocolError(f"reply field {key} has the wrong type")
        out[key] = value[:4000] if isinstance(value, str) else value
    if out.get("status") not in REPLY_STATUSES:
        raise ProtocolError("unknown reply status")
    return out


def _exception_names() -> frozenset[str]:
    global _safe_names
    if _safe_names is None:
        names = set()
        for module_name in _EXCEPTION_MODULES:
            try:
                module = builtins if module_name == "builtins" else importlib.import_module(module_name)
            except ImportError:
                continue
            names.update(n for n, v in vars(module).items()
                         if isinstance(v, type) and issubclass(v, BaseException) and not n.startswith("_"))
        _safe_names = frozenset(names)
    return _safe_names


def exception_name(name) -> str:
    """The agent-visible name for an exception the scorer reported (a fixed list, never free text)."""
    return name if isinstance(name, str) and name in _exception_names() else OTHER_EXCEPTION
