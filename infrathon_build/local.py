"""Running everything directly on this machine, without the Docker sandbox.

The builder is an ordinary local process working in the run's directory, with the host's Docker
daemon and GPUs. It has the same access to this machine as the user who started it, so use the
sandbox (`init --sandbox docker`) on a shared host or for anything you would not run yourself.

LocalHost offers the same interface as sandbox.Sandbox, so the pipeline does not care which one
it drives.
"""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .sandbox import RunResult


@dataclass
class LocalHost:
    run_id: str
    run_dir: Path
    input_dir: Path
    agent_runtimes: Path
    hooks_dir: Path
    guide_dir: Path
    gpus: list[int]
    codex_auth: Path | None = None
    uid: int = field(default_factory=os.getuid)
    gid: int = field(default_factory=os.getgid)

    builder = "local"
    runner = "local"

    def __post_init__(self) -> None:
        self.work = self.run_dir / "work"
        self.logs = self.run_dir / "logs"
        self.state_dir = self.run_dir / "agent-state"
        self.pid_file = self.run_dir / "agent.pgid"

    @property
    def paths(self) -> dict[str, str]:
        return {"work": str(self.work), "input": str(self.input_dir), "guide": str(self.guide_dir),
                "hooks": str(self.hooks_dir), "runtimes": str(self.agent_runtimes), "state": str(self.state_dir),
                "auth": str(self.codex_auth or "")}

    base_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")

    def gpu_index(self, index: int) -> int:
        """Local commands see every GPU of the host; the run's own are its configured indices."""
        return self.gpus[index]

    def environment(self) -> dict[str, str]:
        home = self.work / "home"
        return {
            # A home of its own keeps the builder's git, uv and tool settings out of the user's.
            "HOME": str(home),
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or str(Path.home() / ".cache/uv"),
            "HF_HOME": str(self.work / "hf"),
            "HF_HUB_CACHE": str(self.work / "hf/hub"),
            "INFRATHON_ASSETS_ROOT": str(self.work / "assets"),
            "SFORGE_VERSION_STORE_ROOT": str(self.work / "store"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_AUTOUPDATER": "1",
            "IBUILD_GPUS": ",".join(map(str, self.gpus)),
        }

    # ---- lifecycle: nothing to start or stop ---------------------------------------------

    def up(self) -> None:
        for sub in ("repo", "home", "build", "assets", "hf/hub", "store", "clean"):
            (self.work / sub).mkdir(parents=True, exist_ok=True)
        (self.state_dir / "claude").mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)

    def start_builder(self) -> None:
        self.up()

    def stop_builder(self) -> None:
        self.kill_agents()

    def start_runner(self, codex_auth: Path | None) -> None:
        self.codex_auth = codex_auth

    def stop_runner(self) -> None:
        pass

    def clear_nested_containers(self) -> None:
        """The host's Docker is shared: containers are the launchers' business, not ours to remove."""

    def install_codex_auth(self, source: Path) -> None:
        target = self.state_dir / "codex"
        target.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        (target / "auth.json").write_bytes(data)
        (target / "auth.json").chmod(0o600)

    def kill_agents(self) -> None:
        """Stop whatever the tool started that is still running (its whole process group)."""
        try:
            group = int(self.pid_file.read_text())
        except (OSError, ValueError):
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                break
            time.sleep(3)
        self.pid_file.unlink(missing_ok=True)

    def status(self) -> dict[str, bool]:
        return {}

    def down(self) -> None:
        self.kill_agents()

    def destroy(self) -> None:
        self.kill_agents()

    # ---- running things --------------------------------------------------------------------

    def exec_argv(self, container: str, command: str, *, user: str = "", workdir: str | None = None,
                  env: dict[str, str] | None = None, pass_env: list[str] | None = None, stdin: bool = False) -> list[str]:
        return ["bash", "-c", command]

    def run(self, container: str, command: str, *, log: Path | None = None, timeout: float | None = None,
            env: dict[str, str] | None = None, user: str = "", workdir: str | None = None,
            secret_env: dict[str, str] | None = None, watchdog=None) -> RunResult:
        """Run a shell command here, in its own process group, its output going only to a host file.
        Commands the sandbox runs as root run as the user; they are written to tolerate that."""
        full_env = {**os.environ, **self.environment(), **(env or {}), **(secret_env or {})}
        started = time.time()
        capture = log is None
        if capture:
            target = tempfile.TemporaryFile()
        else:
            log.parent.mkdir(parents=True, exist_ok=True)
            target = open(log, "ab")
        cwd = workdir or str(self.work)
        process = subprocess.Popen(["bash", "-c", command], stdout=target, stderr=subprocess.STDOUT, env=full_env,
                                   cwd=cwd, start_new_session=True)
        self.pid_file.write_text(str(process.pid))
        timed_out, aborted = False, None
        while True:
            try:
                process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                pass
            if timeout is not None and time.time() - started > timeout:
                timed_out = True
            elif watchdog is not None:
                aborted = watchdog()
            if timed_out or aborted:
                self.kill_agents()
                process.wait()
                break
        self.pid_file.unlink(missing_ok=True)
        output = ""
        if capture:
            target.seek(0)
            output = target.read().decode("utf-8", "replace")
        target.close()
        return RunResult(124 if timed_out else process.returncode, timed_out, time.time() - started, output, aborted)

    def write_file(self, relative: str, text: str) -> Path:
        path = self.work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def egress_rows(self) -> list[dict]:
        return []
