"""The harness treats the scorer process as hostile: JSON replies only, known fields, fixed text."""

import pickle
import socket
from multiprocessing.connection import Connection

import pytest

from kvzip_eval import ipc


def _pair():
    a, b = socket.socketpair()
    return Connection(a.detach()), Connection(b.detach())


EXECUTED = []


def _mark(what):
    EXECUTED.append(what)


class _Payload:
    def __reduce__(self):
        return (_mark, ("ran",))


def test_a_pickled_reply_is_never_unpickled():
    harness, scorer = _pair()
    scorer.send({"status": "ok", "payload": _Payload()})  # what a hostile scorer can write
    with pytest.raises(ipc.ProtocolError):
        ipc.recv(harness)
    assert EXECUTED == []
    assert pickle.loads(pickle.dumps(_Payload())) is None and EXECUTED == ["ran"]  # the payload is live


def test_replies_are_restricted_to_known_fields_and_types():
    harness, scorer = _pair()
    ipc.send(scorer, {"status": "ok", "peak_bytes": 5, "rng_changed": False})
    assert ipc.checked_reply(ipc.recv(harness)) == {"status": "ok", "peak_bytes": 5, "rng_changed": False}
    for bad in ({"status": "ok", "extra": 1}, {"status": "ok", "peak_bytes": "5"}, {"status": "ok", "peak_bytes": True},
                {"status": "ok", "rng_changed": 0}, {"status": "done"}, {"peak_bytes": 1}):
        with pytest.raises(ipc.ProtocolError):
            ipc.checked_reply(bad)
    assert len(ipc.checked_reply({"status": "error", "detail": "x" * 10_000})["detail"]) == 4000


def test_oversized_messages_are_refused():
    harness, scorer = _pair()
    scorer.send_bytes(b"x" * (ipc.MAX_MESSAGE_BYTES + 1))
    with pytest.raises(OSError):
        ipc.recv(harness)


def test_agent_visible_text_comes_from_fixed_lists():
    assert ipc.exception_name("ValueError") == "ValueError"
    assert ipc.exception_name("OutOfMemoryError") == "OutOfMemoryError"
    for leak in ("LEAK_172_TOKENS", "ValueError: the hidden text", 7, None):
        assert ipc.exception_name(leak) == ipc.OTHER_EXCEPTION
    assert set(ipc.OUTPUT_PROBLEMS) == {"type", "dtype", "shape", "device", "nonfinite"}
    assert ipc.OUTPUT_PROBLEMS["shape"] == "score shape is not [36, 8, context tokens]"
