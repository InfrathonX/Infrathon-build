"""A run directory of its own for every test, with fake agent runtimes and no credentials."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def git(repo: Path, *args: str) -> str:
    identity = ["-c", "user.name=test", "-c", "user.email=test@localhost", "-c", "commit.gpgsign=false"]
    return subprocess.run(["git", "-C", str(repo), *identity, *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def home(tmp_path, monkeypatch):
    runtimes = tmp_path / "agent_runtimes"
    (runtimes / "claude-code-2.1.0/bin").mkdir(parents=True)
    (runtimes / "codex-0.159.3/bin").mkdir(parents=True)
    monkeypatch.setenv("INFRATHON_BUILD_HOME", str(tmp_path))
    monkeypatch.setenv("IBUILD_AGENT_RUNTIMES", str(runtimes))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("IBUILD_CLAUDE_TOKEN_FILE", raising=False)
    monkeypatch.delenv("IBUILD_CODEX_AUTH_JSON", raising=False)
    # config.py reads the home at import; point it at this test's directory.
    import infrathon_build.config as config
    import infrathon_build.pipeline as pipeline
    monkeypatch.setattr(config, "HOME", tmp_path)
    monkeypatch.setattr(config, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(pipeline, "RUNS_ROOT", tmp_path / "runs")
    return tmp_path


@pytest.fixture
def reference(tmp_path):
    """The smallest tree that passes for an Infrathon checkout."""
    repo = tmp_path / "infrathon"
    (repo / "sforge").mkdir(parents=True)
    (repo / "sforge/__init__.py").write_text("")
    (repo / "tasks/latest/example-task").mkdir(parents=True)
    (repo / "tasks/latest/example-task/README.md").write_text("# Example\n")
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "Infrathon")
    return repo


@pytest.fixture
def expert_input(tmp_path):
    folder = tmp_path / "expert"
    (folder / "attachments").mkdir(parents=True)
    (folder / "TASK_DEFINITION.md").write_text("# A task\n")
    return folder


@pytest.fixture
def make_run(home, reference, expert_input):
    from infrathon_build.pipeline import Run, create_run

    def make(name: str = "r1", **overrides) -> Run:
        create_run(name, expert_input, reference, gpus=overrides.pop("gpus", [2, 5]), **overrides)
        return Run(name)
    return make


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
