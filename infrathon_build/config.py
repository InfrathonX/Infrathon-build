"""Paths, credentials and the settings of one run.

Host settings come from the environment, or from `local.env` in the tool's home
(INFRATHON_BUILD_HOME, default: the repository root):

    IBUILD_CLAUDE_TOKEN_FILE   file holding a Claude Code OAuth token (or set CLAUDE_CODE_OAUTH_TOKEN)
    IBUILD_CODEX_AUTH_JSON     Codex auth.json, for the trial's agent and its integrity review
    IBUILD_AGENT_RUNTIMES      directory with claude-code-<version>/ and codex-<version>/ builds,
                               as Infrathon's launch/build-agent-runtime.sh makes them

Credential values are read only when an agent starts. They never go into a run
directory, a log, or a command line.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("INFRATHON_BUILD_HOME", TOOL_ROOT)).resolve()
RUNS_ROOT = HOME / "runs"


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def setting(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or read_env_file(HOME / "local.env").get(name) or default


def claude_token() -> str:
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if token:
        return token
    path = setting("IBUILD_CLAUDE_TOKEN_FILE")
    if not path:
        raise SystemExit("set CLAUDE_CODE_OAUTH_TOKEN, IBUILD_CLAUDE_TOKEN_FILE or ANTHROPIC_API_KEY")
    return Path(path).expanduser().read_text(encoding="utf-8").strip()


def claude_credentials() -> dict[str, str]:
    """What a Claude Code builder authenticates with: an OAuth token, or else an API key."""
    if not (os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or setting("IBUILD_CLAUDE_TOKEN_FILE")) \
            and setting("ANTHROPIC_API_KEY"):
        return {"ANTHROPIC_API_KEY": setting("ANTHROPIC_API_KEY")}
    return {"CLAUDE_CODE_OAUTH_TOKEN": claude_token()}


def codex_auth_path() -> Path:
    path = setting("IBUILD_CODEX_AUTH_JSON")
    if not path:
        raise SystemExit("set IBUILD_CODEX_AUTH_JSON to a Codex auth.json")
    return Path(path).expanduser()


def agent_runtimes() -> Path:
    path = setting("IBUILD_AGENT_RUNTIMES")
    if not path:
        raise SystemExit("set IBUILD_AGENT_RUNTIMES to the directory of agent runtime builds")
    return Path(path).expanduser().resolve()


def newest_runtime(prefix: str) -> str:
    """claude-code-2.1.287 or codex-0.159.3: the highest version present."""
    names = [p.name for p in agent_runtimes().glob(f"{prefix}-*") if p.is_dir()]
    if not names:
        raise SystemExit(f"no {prefix}-<version> under {agent_runtimes()}")
    return max(names, key=lambda n: [int(x) for x in re.findall(r"\d+", n)])


@dataclass
class RunConfig:
    run_id: str
    input_dir: str                    # the run's own copy of the expert's folder
    reference: str                    # Infrathon checkout the builder starts from
    sandbox: str = "none"             # none: run directly on this machine; docker: isolate in a sandbox
    network_policy: str = ""          # hosts and paths the sandbox gateway refuses (JSON); empty: none
    agent_settings: str = ""          # Claude Code settings for the builder; its directory is mounted beside it
    preflight: str = ""               # script run inside the sandbox before the builder starts; empty: none
    builder_agent: str = "claude-code"
    builder_model: str = "claude-opus-5-5"
    builder_effort: str = "max"
    solver_agent: str = "codex"       # the agent that attempts the task in a trial
    solver_model: str = "gpt-6.1-sol"
    solver_effort: str = "max"
    review_model: str = "gpt-6.1-sol"  # the Controller's integrity review during trials
    review_effort: str = "high"
    claude_runtime: str = ""
    codex_runtime: str = ""
    trial_hours: float = 6.0
    rounds: int = 2                   # trials at most; each is followed by a review and fixes
    gpus: list[int] = field(default_factory=lambda: list(range(8)))
    build_hours: float = 30.0
    fix_hours: float = 4.0
    review_hours: float = 8.0
    check_hours: float = 4.0
    check_tries: int = 3
    min_free_disk_gb: int = 40

    def to_json(self) -> dict:
        return asdict(self)

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def load(run_dir: Path) -> "RunConfig":
        return RunConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))


def review_env(config: RunConfig, auth_in_container: str, runtimes: str) -> dict[str, str]:
    """The Controller's integrity review, as real Infrathon runs configure it."""
    runtime = f"{runtimes}/{config.codex_runtime}"
    return {
        "CODEX_AUTH_JSON_PATH": auth_in_container,
        "SFORGE_AGENT_RUNTIME_DIR": runtime,
        "SFORGE_LLM_JUDGE_MODE": "enforce",
        "SFORGE_LLM_JUDGE_ENGINE": "agent",
        "SFORGE_LLM_JUDGE_AGENT": "codex",
        "SFORGE_LLM_JUDGE_AGENT_RUNTIME_DIR": runtime,
        "SFORGE_LLM_JUDGE_CODEX_AUTH_JSON": auth_in_container,
        "SFORGE_LLM_JUDGE_MODEL": config.review_model,
        "SFORGE_LLM_JUDGE_REASONING_EFFORT": config.review_effort,
    }


def free_disk_gb(path: Path) -> float:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 2**30
