"""Run a coding agent as one process, in the sandbox or on this machine, and read what it did.

Only this process keeps the transcript: it arrives on the agent's stdout and is written to
the run's logs.
"""
from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .sandbox import RunResult, Sandbox

LIMIT_TEXT = re.compile(r"usage limit|rate limit|rate_limit|limit reached|resets? (at|in)|quota|overloaded|429|too many requests",
                        re.IGNORECASE)


@dataclass
class AgentSpec:
    kind: str  # claude-code | codex
    model: str
    effort: str
    runtime: str  # directory name in the agent runtimes directory
    settings: str = ""  # Claude Code settings file, as the agent sees it


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        # Claude Code reports the session's running total, so a later value replaces an earlier one.
        self.cost_usd = max(self.cost_usd, other.cost_usd)


@dataclass
class AgentResult:
    returncode: int
    timed_out: bool
    seconds: float
    session_id: str | None
    error: str | None
    usage: Usage = field(default_factory=Usage)
    rate_limited: bool = False
    aborted: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def _read_jsonl(path: Path, start: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, "rb") as fh:
        fh.seek(start)
        for raw in fh:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


class Driver:
    def __init__(self, sandbox: Sandbox, spec: AgentSpec, container: str, secrets: dict[str, str]) -> None:
        self.sandbox = sandbox
        self.spec = spec
        self.container = container
        self.secrets = secrets

    def command(self, prompt_path: str, resume: str | None) -> tuple[str, dict[str, str]]:
        raise NotImplementedError

    def parse(self, rows: list[dict], result: RunResult) -> AgentResult:
        raise NotImplementedError

    def run(self, prompt_path: str, log: Path, timeout: float, resume: str | None = None, watchdog=None) -> AgentResult:
        command, env = self.command(prompt_path, resume)
        start = log.stat().st_size if log.exists() else 0
        result = self.sandbox.run(self.container, command, log=log, timeout=timeout, env=env,
                                  secret_env=self.secrets, watchdog=watchdog)
        rows = _read_jsonl(log, start) if log.exists() else []
        parsed = self.parse(rows, result)
        parsed.aborted = result.aborted
        return parsed


class ClaudeDriver(Driver):
    def command(self, prompt_path: str, resume: str | None) -> tuple[str, dict[str, str]]:
        paths = self.sandbox.paths
        args = [
            "claude", "-p", '"$(cat ' + shlex.quote(prompt_path) + ')"',
            "--model", shlex.quote(self.spec.model),
            "--output-format", "stream-json", "--verbose",
            "--dangerously-skip-permissions",
            "--settings", shlex.quote(self.spec.settings),
            "--add-dir", shlex.quote(paths["input"]), "--add-dir", shlex.quote(paths["guide"]),
        ]
        if self.spec.effort:
            args += ["--effort", shlex.quote(self.spec.effort)]
        if resume:
            args += ["--resume", shlex.quote(resume)]
        env = {
            "PATH": f"{paths['runtimes']}/{self.spec.runtime}/bin:{self.sandbox.base_path}",
            "CLAUDE_CONFIG_DIR": f"{paths['state']}/claude",
            "IS_SANDBOX": "1",
        }
        return " ".join(args), env

    def parse(self, rows: list[dict], result: RunResult) -> AgentResult:
        session_id, error, usage = None, None, Usage()
        for row in rows:
            if row.get("session_id"):
                session_id = row["session_id"]
            if row.get("type") == "result":
                if row.get("is_error"):
                    error = str(row.get("result") or row.get("subtype") or "agent error")[:800]
                info = row.get("usage") or {}
                usage.input_tokens += int(info.get("input_tokens") or 0)
                usage.output_tokens += int(info.get("output_tokens") or 0)
                usage.cached_tokens += int(info.get("cache_read_input_tokens") or 0)
                usage.cost_usd = max(usage.cost_usd, float(row.get("total_cost_usd") or 0))
        if result.returncode not in (0, 124) and not error:
            error = f"agent process exited with {result.returncode}"
        limited = bool(error and LIMIT_TEXT.search(error))
        return AgentResult(result.returncode, result.timed_out, result.seconds, session_id, error, usage, limited)


class CodexDriver(Driver):
    def command(self, prompt_path: str, resume: str | None) -> tuple[str, dict[str, str]]:
        paths = self.sandbox.paths
        args = [
            "codex", "exec",
            *(["resume", shlex.quote(resume)] if resume else []),
            "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--json",
            "-m", shlex.quote(self.spec.model),
            "-c", "model_reasoning_effort=" + shlex.quote(json.dumps(self.spec.effort)),
            '"$(cat ' + shlex.quote(prompt_path) + ')"',
        ]
        env = {
            "PATH": f"{paths['runtimes']}/{self.spec.runtime}/bin:{self.sandbox.base_path}",
            "CODEX_HOME": f"{paths['state']}/codex",
        }
        return " ".join(args), env

    def parse(self, rows: list[dict], result: RunResult) -> AgentResult:
        session_id, error, usage = None, None, Usage()
        for row in rows:
            kind = row.get("type")
            if kind == "thread.started":
                session_id = row.get("thread_id") or session_id
            elif kind == "turn.completed":
                info = row.get("usage") or {}
                usage.input_tokens += int(info.get("input_tokens") or 0)
                usage.output_tokens += int(info.get("output_tokens") or 0)
                usage.cached_tokens += int(info.get("cached_input_tokens") or 0)
            elif kind in ("turn.failed", "error"):
                error = json.dumps(row)[:800]
        if result.returncode not in (0, 124) and not error:
            error = f"agent process exited with {result.returncode}"
        limited = bool(error and LIMIT_TEXT.search(error))
        return AgentResult(result.returncode, result.timed_out, result.seconds, session_id, error, usage, limited)


def make_driver(sandbox: Sandbox, spec: AgentSpec, container: str, secrets: dict[str, str]) -> Driver:
    if spec.kind == "claude-code":
        return ClaudeDriver(sandbox, spec, container, secrets)
    if spec.kind == "codex":
        return CodexDriver(sandbox, spec, container, secrets)
    raise ValueError(f"unknown agent {spec.kind}")


def run_until_done(driver: Driver, prompt_path: str, log: Path, deadline: float, session: str | None = None,
                   watchdog=None, say=print, max_crashes: int = 3) -> AgentResult:
    """Keep one agent session going through crashes and rate limits, up to the deadline."""
    total = Usage()
    started = time.time()
    crashes = 0
    backoff = 600.0
    prompt = prompt_path
    while True:
        remaining = deadline - time.time()
        if remaining <= 30:
            return AgentResult(124, True, time.time() - started, session, "time budget used up", total)
        result = driver.run(prompt, log, remaining, resume=session, watchdog=watchdog)
        total.add(result.usage)
        session = result.session_id or session
        result.session_id = session
        result.usage = total
        result.seconds = time.time() - started
        if result.timed_out or result.aborted or not result.error:
            return result
        if result.rate_limited:
            wait = min(backoff, max(0.0, deadline - time.time() - 60))
            say(f"agent is rate limited; waiting {int(wait)} s: {result.error[:200]}")
            time.sleep(wait)
            backoff = min(backoff * 1.5, 3600.0)
        else:
            crashes += 1
            if crashes > max_crashes:
                return result
            say(f"agent crashed ({crashes}/{max_crashes}); resuming: {result.error[:200]}")
            time.sleep(30)
        if session:
            # A resumed session needs a prompt; the original is already in its history.
            driver.sandbox.write_file("build/.prompts/continue.md", "You were interrupted. Continue from where you stopped.\n")
            prompt = f"{driver.sandbox.paths['work']}/build/.prompts/continue.md"
