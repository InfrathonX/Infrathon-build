"""Hold the host's GPU locks for as long as a run is active.

Launchers inside a sandbox take locks that exist only inside it. These are the
host-level ones (the harness's own convention) that other runs on this machine check.
"""
from __future__ import annotations

import fcntl
import os
import subprocess


class GpuBusy(RuntimeError):
    pass


def gpu_table() -> dict[int, tuple[str, int]]:
    """index -> (uuid, MiB in use)"""
    done = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True)
    table = {}
    for line in done.stdout.splitlines():
        index, uuid, used = (part.strip() for part in line.split(","))
        table[int(index)] = (uuid, int(used))
    return table


def check_idle(indices: list[int], max_mib: int = 1024) -> None:
    table = gpu_table()
    busy = [f"GPU {i} ({table[i][1]} MiB in use)" for i in indices if table[i][1] > max_mib]
    if busy:
        raise GpuBusy("in use by something else: " + ", ".join(busy))


class HostGpuLocks:
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices
        self.handles: list[int] = []

    def acquire(self) -> None:
        table = gpu_table()
        for index in self.indices:
            uuid = table[index][0]
            for family in ("sforge", "mlsysforge"):
                path = f"/tmp/{family}-gpu-{uuid}.lock"
                fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o666)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    os.close(fd)
                    self.release()
                    raise GpuBusy(f"GPU {index}: {path} is held by another run") from exc
                self.handles.append(fd)

    def release(self) -> None:
        for fd in self.handles:
            os.close(fd)
        self.handles = []
