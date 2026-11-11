#!/usr/bin/env python3
"""Task-local SForge entry: role-aware Docker containers for this task.

SForge's stock Docker adapter gives a Judge container its GPUs through
SFORGE_JUDGE_EXTRA_ENV, which is comma separated and so cannot carry a list of
devices.  This entry patches container creation (task-local, as the other
tasks' entries do) so that:

* Work gets the one GPU in KVZIP_WORK_GPU_UUID, the frozen model read-only at
  /opt/models/qwen3-8b, and only the launcher's internal network (reaching the
  outside through the restricted CONNECT sidecar);
* every Judge gets all GPUs in KVZIP_JUDGE_GPU_UUIDS, the model read-only below
  the root-only /opt/kvzip_judge/private, no network at all, and no new
  privileges.

KVZIP_MODEL_DIR is the host directory of Qwen/Qwen3-8B (the flat asset).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from docker.types import DeviceRequest
from sforge.harness import run_agent
from sforge.harness.agent.codex import CodexAgent
from sforge.harness.backend import AcceleratorAllocation, HANDOFF_BINDING_ANNOTATIONS
from sforge.harness.backend.docker_backend import (
    DockerBackend,
    DockerContainerHandle,
    _validated_docker_labels,
    _validated_handoff_binding_labels,
)

TASK_ID = "kvzip_kv_scoring_qwen3_8b_h100_scratch_v2"
UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
WORK_MODEL_MOUNT = "/opt/models/qwen3-8b"
JUDGE_MODEL_MOUNT = "/opt/kvzip_judge/private/models/qwen3-8b"
NETWORK_ENV = "SFORGE_ISOLATED_DOCKER_NETWORK"


def _role(image: str, name: str) -> str:
    image_work, image_judge = f".work.{TASK_ID}" in image, f".judge.{TASK_ID}" in image
    name_work = name.startswith(f"sforge.run.{TASK_ID}.")
    name_judge = f".judge.{TASK_ID}." in name
    if image_work and name_work and not (image_judge or name_judge):
        return "work"
    if image_judge and name_judge and not (image_work or name_work):
        return "judge"
    raise RuntimeError(f"cannot classify container image={image!r} name={name!r} for {TASK_ID}")


def _uuids(variable: str, single: bool) -> list[str]:
    raw = os.environ.get(variable, "").strip()
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values or any(UUID_RE.fullmatch(v) is None for v in values) or len(set(values)) != len(values):
        raise RuntimeError(f"set {variable} to {'one GPU UUID' if single else 'comma-separated GPU UUIDs'}")
    if single and len(values) != 1:
        raise RuntimeError(f"{variable} must name exactly one GPU")
    return values


def _model_dir() -> str:
    raw = os.environ.get("KVZIP_MODEL_DIR", "").strip()
    if not raw:
        raise RuntimeError("set KVZIP_MODEL_DIR to the Qwen3-8B directory (assets/fetch.sh lays it out)")
    path = Path(raw).resolve(strict=True)
    if not (path / "config.json").is_file() or not (path / "model.safetensors.index.json").is_file():
        raise RuntimeError(f"KVZIP_MODEL_DIR={path} is not a Qwen3-8B snapshot")
    return str(path)


def create_container(
    self: DockerBackend,
    image: str,
    name: str,
    *,
    command: str = "tail -f /dev/null",
    environment: dict[str, str] | None = None,
    extra_hosts: dict[str, str] | None = None,
    cap_drop: list[str] | None = None,
    cpu_limit: int | None = None,
    mem_limit: str | None = None,
    network_mode: str | None = None,
    user: str | None = None,
    annotations: dict[str, str] | None = None,
    mounts: list[tuple[str, str, str]] | None = None,
    accelerator_allocation: AcceleratorAllocation | None = None,
) -> DockerContainerHandle:
    identity = image
    if IMAGE_ID_RE.fullmatch(image):
        # Replay pins the task image by its ID: classify it by the task tags it still carries
        # (an untagged or foreign image stays unclassifiable).
        identity = " ".join(self.client.images.get(image).tags)
    role = _role(identity, name)
    if accelerator_allocation is not None:
        raise RuntimeError("this task runs Work and Judge on separate GPUs; no shared allocation")
    gpus = _uuids("KVZIP_WORK_GPU_UUID", True) if role == "work" else _uuids("KVZIP_JUDGE_GPU_UUIDS", False)
    env = dict(environment or {})
    budget = os.environ.get("MLSYSFORGE_AGENT_BUDGET_SECONDS", "").strip()
    if role == "work" and budget.isdigit():
        # The agent's own deadline (the run starts as this container is created).
        env["KVZIP_TIME_BUDGET_SECONDS"] = budget
        env["KVZIP_DEADLINE_UNIX"] = str(int(time.time()) + int(budget))
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["NVIDIA_VISIBLE_DEVICES"] = ",".join(gpus)
    env["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility"
    volumes = [f"{_model_dir()}:{WORK_MODEL_MOUNT if role == 'work' else JUDGE_MODEL_MOUNT}:ro"]
    if mounts:
        if role == "judge":
            raise RuntimeError("Judge containers accept no caller-supplied mounts")
        for host_path, container_path, mode in mounts:
            if mode != "ro":
                raise RuntimeError("Work containers accept only read-only bind mounts")
            volumes.append(f"{host_path}:{container_path}:{mode}")
    labels: dict[str, str] = {}
    if annotations:
        labels = _validated_docker_labels(annotations)
        if HANDOFF_BINDING_ANNOTATIONS.intersection(labels):
            labels = _validated_handoff_binding_labels(labels)
    labels.update({"infrathon.task_id": TASK_ID, "infrathon.role": role})
    kwargs: dict[str, Any] = {
        "volumes": volumes,
        "device_requests": [DeviceRequest(driver="nvidia", device_ids=gpus, capabilities=[["gpu"]])],
        "labels": labels,
        "shm_size": "8g",
    }
    if role == "judge":
        kwargs["network_mode"] = "none"
        kwargs["security_opt"] = ["no-new-privileges:true"]
    else:
        network = os.environ.get(NETWORK_ENV, "").strip()
        if not network:
            raise RuntimeError(f"{NETWORK_ENV} is required for Work")
        if self.client.networks.get(network).attrs.get("Internal") is not True:
            raise RuntimeError(f"refusing non-internal Work network {network}")
        kwargs["network"] = network
        if extra_hosts:
            kwargs["extra_hosts"] = extra_hosts
    if cpu_limit is not None:
        kwargs["nano_cpus"] = int(cpu_limit * 1e9)
    if mem_limit is not None:
        kwargs["mem_limit"] = mem_limit
    if cap_drop:
        kwargs["cap_drop"] = cap_drop
    if user:
        kwargs["user"] = user
    container = self.client.containers.create(image, name=name, detach=True, command=command,
                                              environment=env, **kwargs)
    return DockerContainerHandle(container)


DockerBackend.create_container = create_container

# internet follows the task JSON (exported by the launcher); the internal network and
# the CONNECT sidecar are the enforcement boundary, so SForge's own iptables mode is off.
WORK_INTERNET_ENV = "MLSYSFORGE_WORK_INTERNET"


def _task_internet(requested: Any) -> bool:
    value = os.environ.get(WORK_INTERNET_ENV, "").strip().lower()
    if value == "false":
        return False
    if value in {"", "true"}:
        return bool(requested)
    raise RuntimeError(f"{WORK_INTERNET_ENV} must be 'true' or 'false'")


_format_run_cmd = CodexAgent.format_run_cmd


def _format_with_internet(self: CodexAgent, *args: Any, **kwargs: Any) -> str:
    kwargs["internet"] = _task_internet(kwargs.get("internet", True))
    return _format_run_cmd(self, *args, **kwargs)


CodexAgent.format_run_cmd = _format_with_internet
_generate_prompt = run_agent.generate_evolve_prompt


def _prompt_with_internet(*args: Any, **kwargs: Any) -> str:
    kwargs["internet"] = _task_internet(kwargs.get("internet", True))
    return _generate_prompt(*args, **kwargs)


run_agent.generate_evolve_prompt = _prompt_with_internet

from sforge.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
