"""One run's sandbox: an egress gateway, a nested dockerd, and two agent containers.

    host
    ├─ this process: the only thing that can read the host's repositories and credentials
    └─ per-run bridge network
       ├─ proxy    gateway to the internet; policy and log (never visible inside)
       ├─ dind     nested dockerd: not privileged, sees no host image or disk
       ├─ builder  the builder agent, non-root, shares dind's network namespace
       └─ runner   the tool's checks and trials: same view, plus the trial's credentials

Builder and runner share dind's network namespace because the harness runs next to
the containers it starts: a Work container reaches the Judge server through its
docker bridge gateway, which only exists in that namespace.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import TOOL_ROOT

SANDBOX_IMAGE = "infrathon-build-sandbox:1.0"
PROXY_IMAGE = "infrathon-build-proxy:1.0"

# Names resolved to the gateway for dockerd and agent processes, so their paths can
# be filtered. Blob and object storage stays passthrough: it is content addressed,
# and intercepting tens of gigabytes would only add cost.
INTERCEPT_HOSTS = (
    "github.com", "api.github.com", "raw.githubusercontent.com", "codeload.github.com", "gist.github.com",
    "huggingface.co", "hub.docker.com", "registry-1.docker.io", "ghcr.io",
)

AGENT_ENV = {
    "HOME": "/work/home",
    "DOCKER_HOST": "unix:///ibt-sock/docker.sock",
    "NODE_EXTRA_CA_CERTS": "/ca/ibt-ca.crt",
    "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
    "REQUESTS_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
    "CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
    "UV_NATIVE_TLS": "1",
    "HF_HOME": "/work/hf",
    "HF_HUB_CACHE": "/work/hf/hub",
    "INFRATHON_ASSETS_ROOT": "/work/assets",
    "SFORGE_VERSION_STORE_ROOT": "/work/store",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_AUTOUPDATER": "1",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}


class SandboxError(RuntimeError):
    pass


def docker(*args: str, check: bool = True, capture: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(["docker", *args], capture_output=capture, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise SandboxError(f"docker {' '.join(args[:4])} failed: {(result.stderr or '')[-600:]}")
    return result


def subnet_for(run_id: str) -> tuple[str, str, str]:
    """A per-run /24 from a range the host's docker does not hand out by default."""
    octet = int(hashlib.sha256(run_id.encode()).hexdigest()[:4], 16) % 200 + 20
    return f"10.232.{octet}.0/24", f"10.232.{octet}.2", f"10.232.{octet}.3"


def ensure_images() -> None:
    for image, context in ((SANDBOX_IMAGE, TOOL_ROOT / "sandbox"), (PROXY_IMAGE, TOOL_ROOT / "sandbox/proxy")):
        if docker("image", "inspect", image, check=False).returncode != 0:
            docker("build", "-q", "-t", image, str(context), timeout=3600)


@dataclass
class Sandbox:
    run_id: str
    run_dir: Path
    input_dir: Path
    agent_runtimes: Path
    denylist: Path      # network policy: what the gateway refuses
    hooks_dir: Path     # agent settings and hook scripts, at /opt/ibt/hooks
    guide_dir: Path     # the builder's handbook for this run, at /guide
    gpus: list[int] | None = None  # host indices; None means every GPU
    uid: int = field(default_factory=os.getuid)
    gid: int = field(default_factory=os.getgid)

    # Where things are, as the agents and the tool's commands see them inside.
    paths = {"work": "/work", "input": "/input", "guide": "/guide", "hooks": "/opt/ibt/hooks",
             "runtimes": "/opt/agent-runtimes", "state": "/builder-state", "auth": "/run/ibt-secrets/codex-auth.json"}
    base_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    @staticmethod
    def gpu_index(index: int) -> int:
        """Inside, the run's GPUs are numbered from 0."""
        return index

    def __post_init__(self) -> None:
        safe = "".join(c if c.isalnum() else "-" for c in self.run_id).strip("-").lower()
        self.prefix = f"ibuild-{safe}"
        self.net = f"{self.prefix}-net"
        self.proxy = f"{self.prefix}-proxy"
        self.dind = f"{self.prefix}-dind"
        self.builder = f"{self.prefix}-builder"
        self.runner = f"{self.prefix}-runner"
        self.subnet, self.proxy_ip, self.dind_ip = subnet_for(self.run_id)
        self.work = self.run_dir / "work"
        self.logs = self.run_dir / "logs"
        self.ca_dir = self.run_dir / "ca"

    # ---- lifecycle ------------------------------------------------------------------

    def _exists(self, name: str) -> bool:
        return docker("inspect", "--format", "{{.Id}}", name, check=False).returncode == 0

    def _running(self, name: str) -> bool:
        result = docker("inspect", "--format", "{{.State.Running}}", name, check=False)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def up(self) -> None:
        ensure_images()
        for path in (self.work, self.logs / "proxy", self.ca_dir):
            path.mkdir(parents=True, exist_ok=True)
        for sub in ("repo", "home", "build", "assets", "hf/hub", "store", "clean"):
            (self.work / sub).mkdir(parents=True, exist_ok=True)
        self._start_proxy()
        self._start_dind()

    def _start_proxy(self) -> None:
        if self._running(self.proxy):
            return
        if not self._exists(self.proxy):
            if docker("network", "inspect", self.net, check=False).returncode != 0:
                docker("network", "create", "--subnet", self.subnet, self.net)
            docker("create", "--name", self.proxy, "--restart", "unless-stopped", "--network", "bridge",
                   "--cap-add", "NET_ADMIN", "--sysctl", "net.ipv4.ip_forward=1",
                   "-e", f"IBT_PROXY_IP={self.proxy_ip}",
                   "-v", f"{self.ca_dir}:/ca-out",
                   "-v", f"{self.logs / 'proxy'}:/logs",
                   "-v", f"{self.denylist}:/opt/ibt/denylist.json:ro",
                   PROXY_IMAGE)
            docker("network", "connect", "--ip", self.proxy_ip, self.net, self.proxy)
        (self.ca_dir / "ibt-ca.crt").unlink(missing_ok=True)
        docker("start", self.proxy)
        self._wait(lambda: (self.ca_dir / "ibt-ca.crt").exists() and self._port_open(self.proxy, 8081),
                   60, "egress gateway")

    def _port_open(self, container: str, port: int) -> bool:
        probe = f"import socket;s=socket.socket();s.settimeout(1);s.connect(('127.0.0.1',{port}))"
        return docker("exec", container, "python3", "-c", probe, check=False).returncode == 0

    def _start_dind(self) -> None:
        if self._running(self.dind):
            return
        if not self._exists(self.dind):
            for vol in ("data", "sock", "tmp"):
                docker("volume", "create", f"{self.prefix}-{vol}")
            docker("run", "-d", "--name", self.dind, "--restart", "unless-stopped",
                   "--cap-add=ALL", "--security-opt", "seccomp=unconfined",
                   "--security-opt", "apparmor=unconfined", "--security-opt", "systempaths=unconfined",
                   "--cgroupns=private", *self._gpu_args(),
                   "--network", self.net, "--ip", self.dind_ip,
                   "-e", f"IBT_PROXY_IP={self.proxy_ip}", "-e", f"IBT_SUBNET={self.subnet}",
                   "-e", "IBT_INTERCEPT_HOSTS=" + " ".join(INTERCEPT_HOSTS),
                   "-e", "DOCKER_HOST=unix:///ibt-sock/docker.sock",
                   "-v", f"{self.prefix}-data:/var/lib/docker",
                   "-v", f"{self.prefix}-sock:/ibt-sock",
                   "-v", f"{self.prefix}-tmp:/tmp",
                   "-v", f"{self.work}:/work",
                   "-v", f"{self.ca_dir}:/ca:ro",
                   "-v", f"{self.agent_runtimes}:/opt/agent-runtimes:ro",
                   "--tmpfs", "/run:rw",
                   "--entrypoint", "/usr/local/bin/dind-entry.sh", SANDBOX_IMAGE)
        else:
            docker("start", self.dind)
        self._wait(lambda: docker("exec", self.dind, "docker", "info", check=False).returncode == 0,
                   90, "nested dockerd")

    def _gpu_args(self) -> list[str]:
        """The sandbox sees only its own GPUs, so nothing in it can disturb another run's card."""
        # A comma inside this value has to be CSV-quoted, or docker reads several fields.
        devices = "all" if self.gpus is None else '"device=' + ",".join(map(str, self.gpus)) + '"'
        return ["--gpus", devices, "-e", "NVIDIA_DRIVER_CAPABILITIES=all"]

    def _agent_container_args(self) -> list[str]:
        return [
            "--network", f"container:{self.dind}",
            *self._gpu_args(),
            "--init", "--entrypoint", "/bin/sh",
            "-v", f"{self.prefix}-sock:/ibt-sock",
            "-v", f"{self.prefix}-tmp:/tmp",
            "-v", f"{self.work}:/work",
            "-v", f"{self.input_dir}:/input:ro",
            "-v", f"{self.ca_dir}:/ca:ro",
            "-v", f"{self.agent_runtimes}:/opt/agent-runtimes:ro",
            "-v", f"{self.guide_dir}:/guide:ro",
            "-v", f"{self.hooks_dir}:/opt/ibt/hooks:ro",
        ]

    def _ensure_identity(self, container: str) -> None:
        """Create the unprivileged account and trust the gateway's CA."""
        script = (
            f"getent group builder >/dev/null || groupadd -g {self.gid} builder; "
            f"id builder >/dev/null 2>&1 || useradd -u {self.uid} -g {self.gid} -G docker -d /work/home -M -s /bin/bash builder; "
            "cp /ca/ibt-ca.crt /usr/local/share/ca-certificates/ibt-ca.crt && update-ca-certificates >/dev/null; "
            "chmod 1777 /tmp"
        )
        docker("exec", "-u", "0", container, "sh", "-c", script)

    def start_builder(self) -> None:
        if self._running(self.builder):
            return
        if self._exists(self.builder):
            docker("start", self.builder)
        else:
            # The builder's own session history lives here, outside /work.
            docker("run", "-d", "--name", self.builder, "-v", f"{self.prefix}-state:/builder-state",
                   *self._agent_container_args(), SANDBOX_IMAGE, "-c", "sleep infinity")
        self._ensure_identity(self.builder)
        docker("exec", "-u", "0", self.builder, "chown", f"{self.uid}:{self.gid}", "/builder-state")

    def stop_builder(self) -> None:
        """Nothing the builder started survives into a check or a trial."""
        if self._running(self.builder):
            docker("stop", "-t", "5", self.builder, check=False)
        self.clear_nested_containers()

    def clear_nested_containers(self) -> None:
        if self._running(self.dind):
            docker("exec", self.dind, "sh", "-c",
                   'ids="$(docker ps -aq)"; [ -z "$ids" ] || docker rm -f $ids >/dev/null', check=False, timeout=300)

    def start_runner(self, codex_auth: Path | None) -> None:
        """Checks and trials. Only this container, and only when needed, sees the trial's credentials."""
        if self._exists(self.runner):
            docker("rm", "-f", self.runner, check=False)
        extra: list[str] = []
        if codex_auth is not None:
            extra += ["-v", f"{codex_auth}:/run/ibt-secrets/codex-auth.json:ro"]
        docker("run", "-d", "--name", self.runner, *self._agent_container_args(), *extra,
               SANDBOX_IMAGE, "-c", "sleep infinity")
        self._ensure_identity(self.runner)

    def stop_runner(self) -> None:
        docker("rm", "-f", self.runner, check=False)
        self.clear_nested_containers()

    def install_codex_auth(self, source: Path) -> None:
        """A Codex builder keeps its login in its own state volume, outside /work."""
        docker("exec", "-u", "builder", self.builder, "mkdir", "-p", "/builder-state/codex")
        docker("cp", str(source), f"{self.builder}:/builder-state/codex/auth.json")

    def kill_agents(self) -> None:
        for container in (self.builder, self.runner):
            docker("exec", "-u", "0", container, "sh", "-c", "pkill -TERM -u builder; sleep 3; pkill -KILL -u builder",
                   check=False)

    def status(self) -> dict[str, bool]:
        return {name: self._running(name) for name in (self.proxy, self.dind, self.builder, self.runner)}

    def down(self) -> None:
        for name in (self.runner, self.builder, self.dind, self.proxy):
            docker("rm", "-f", name, check=False)

    def destroy(self) -> None:
        self.down()
        for vol in ("data", "sock", "tmp", "state"):
            docker("volume", "rm", "-f", f"{self.prefix}-{vol}", check=False)
        docker("network", "rm", self.net, check=False)

    # ---- running things --------------------------------------------------------------

    def exec_argv(self, container: str, command: str, *, user: str = "builder", workdir: str | None = "/work",
                  env: dict[str, str] | None = None, pass_env: list[str] | None = None,
                  stdin: bool = False) -> list[str]:
        argv = ["docker", "exec", *(["-i"] if stdin else []), "-u", user]
        if workdir:
            argv += ["-w", workdir]
        for key, value in {**AGENT_ENV, **(env or {})}.items():
            argv += ["-e", f"{key}={value}"]
        # By name only, so a secret's value never appears on a host command line.
        for key in pass_env or []:
            argv += ["-e", key]
        return argv + [container, "bash", "-c", command]

    def run(self, container: str, command: str, *, log: Path | None = None, timeout: float | None = None,
            env: dict[str, str] | None = None, user: str = "builder", workdir: str | None = "/work",
            secret_env: dict[str, str] | None = None, watchdog=None) -> "RunResult":
        """Run a shell command in a container, its output going only to a host file.

        watchdog is called every few seconds; returning a string aborts the command
        with that reason (the host running out of disk, for example).
        """
        argv = self.exec_argv(container, command, user=user, workdir=workdir, env=env,
                              pass_env=list(secret_env or {}))
        started = time.time()
        capture = log is None
        if capture:
            target = tempfile.TemporaryFile()
        else:
            log.parent.mkdir(parents=True, exist_ok=True)
            target = open(log, "ab")
        process = subprocess.Popen(argv, stdout=target, stderr=subprocess.STDOUT,
                                   env={**os.environ, **(secret_env or {})})
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
                process.kill()
                process.wait()
                # The exec'd process outlives the docker CLI: stop it inside as well.
                docker("exec", "-u", "0", container, "sh", "-c",
                       "pkill -TERM -u builder; sleep 3; pkill -KILL -u builder", check=False)
                break
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

    @staticmethod
    def _wait(predicate, timeout: float, what: str) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return
            time.sleep(1)
        raise SandboxError(f"timed out waiting for {what}")

    def egress_rows(self) -> list[dict]:
        rows: list[dict] = []
        path = self.logs / "proxy" / "egress.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows


@dataclass
class RunResult:
    returncode: int
    timed_out: bool
    seconds: float
    output: str
    aborted: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.aborted
