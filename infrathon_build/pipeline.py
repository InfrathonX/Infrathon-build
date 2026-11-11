"""The flow: build, check, trial, review and fix, deliver.

    setup    the reference checkout becomes the run's repo (in the sandbox, /work/repo)
    build    the builder agent writes the task, tests it, and hands over TASK.json
    check    the tool runs the task's check from a clean clone; a failure goes back to the builder
    trial    the tool runs the real launcher: another agent attempts the task for some hours
    review   the builder reads the trial, fixes what it shows about the task, and says whether
             another trial is needed; then check and trial again, up to the round limit
    deliver  patches and a report

One builder session runs through the whole flow, resumed at each step, so it remembers what it
built. Checks and trials are run by the tool from a clean clone, so they show what a fresh host
would see. Every finished step is recorded, so `run` picks up where it stopped.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from .agents import AgentSpec, make_driver, run_until_done
from .config import (
    RUNS_ROOT, TOOL_ROOT, RunConfig, claude_credentials, codex_auth_path, free_disk_gb, review_env,
)
from .hostlocks import GpuBusy, HostGpuLocks, check_idle
from .local import LocalHost
from .sandbox import Sandbox

PROMPTS = TOOL_ROOT / "prompts"
GUIDE = TOOL_ROOT / "guide"
# The framework's own environment, exactly as its lockfile pins it.
BOOTSTRAP = "uv sync -q --frozen --dev"
TASK_FIELDS = ("task", "task_id", "launch", "work_gpus", "judge_gpus", "check_cmd", "trial_cmd", "trial_output")


def log_line(run_dir: Path, text: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def fill(template: str, **values: object) -> str:
    """Replace only the named placeholders, leaving ${SHELL} and JSON braces alone."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def render_paths(text: str, paths: dict[str, str]) -> str:
    """Prompts and the guide name the sandbox's paths (/work, /input, /guide); map them to where they
    really are. Only paths that start there: images/work/ is a directory inside a task, not /work."""
    for name in ("work", "input", "guide"):
        text = re.sub(rf"(?<![\w./-])/{name}(?=[/`\s)]|$)", lambda m, p=paths[name]: p, text)
    return text


class Stop(Exception):
    """End the run here, with this status."""

    def __init__(self, status: str, why: str) -> None:
        super().__init__(why)
        self.status = status
        self.why = why


class Run:
    def __init__(self, run_id: str) -> None:
        self.dir = RUNS_ROOT / run_id
        self.config = RunConfig.load(self.dir)
        policy = Path(self.config.network_policy or TOOL_ROOT / "sandbox/proxy/default-policy.json").resolve()
        self.settings_file = Path(self.config.agent_settings or TOOL_ROOT / "sandbox/hooks/claude-settings.json").resolve()
        from .config import agent_runtimes
        common = dict(run_id=run_id, run_dir=self.dir, input_dir=Path(self.config.input_dir),
                      agent_runtimes=agent_runtimes(), hooks_dir=self.settings_file.parent, guide_dir=self.dir / "guide",
                      gpus=self.config.gpus)
        if self.config.sandbox == "docker":
            self.sandbox = Sandbox(denylist=policy, **common)
        elif self.config.sandbox == "none":
            self.sandbox = LocalHost(**common)
        else:
            raise SystemExit(f"unknown sandbox {self.config.sandbox!r}: use none or docker")
        self.W = self.sandbox.paths["work"]
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {
            "status": "created", "done": {}, "turns": [], "session": None, "base_commit": None, "costs": {},
        }
        self.locks = HostGpuLocks(self.config.gpus)

    # ---- helpers ------------------------------------------------------------------------

    def say(self, text: str) -> None:
        log_line(self.dir, text)

    def save(self) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)

    def watchdog(self) -> str | None:
        free = free_disk_gb(self.dir)
        if free < self.config.min_free_disk_gb:
            return f"host disk is down to {free:.0f} GB free (minimum {self.config.min_free_disk_gb})"
        return None

    def secret_values(self) -> list[str]:
        values: list[str] = []
        try:
            values += list(claude_credentials().values())
        except (OSError, SystemExit):
            pass
        try:
            data = json.loads(codex_auth_path().read_text(encoding="utf-8"))

            def walk(value, key=""):
                if isinstance(value, dict):
                    for k, v in value.items():
                        walk(v, str(k))
                elif isinstance(value, str) and len(value) >= 16 and any(m in key.lower() for m in ("token", "key", "secret")):
                    values.append(value)
            walk(data)
        except (OSError, ValueError, SystemExit):
            pass
        return values

    def copy_for_builder(self, source: Path, relative: str, limit: int = 3_000_000) -> str:
        """Logs live on the host, out of the builder's reach. It gets a redacted copy of the tail."""
        data = source.read_bytes()[-limit:].decode("utf-8", "replace") if source.exists() else ""
        for secret in self.secret_values():
            data = data.replace(secret, "<redacted>")
        self.sandbox.write_file(relative, data)
        return f"{self.W}/{relative}"

    def render(self, text: str) -> str:
        return render_paths(text, self.sandbox.paths)

    def prepare_guide(self) -> None:
        """The builder's handbook for this run: the guide, and the environment section for how it runs."""
        import shutil
        target = self.dir / "guide"
        # In place, so that a container's bind mount of this directory keeps seeing it.
        shutil.copytree(GUIDE / "reference_docs", target / "reference_docs", dirs_exist_ok=True)
        environment = "ENVIRONMENT-docker.md" if self.config.sandbox == "docker" else "ENVIRONMENT-local.md"
        gpus = ", ".join(str(self.sandbox.gpu_index(i)) for i in range(len(self.config.gpus)))
        for source, name in (("README.md", "README.md"), (environment, "ENVIRONMENT.md")):
            text = (GUIDE / source).read_text(encoding="utf-8").replace("{gpus}", gpus)
            (target / name).write_text(self.render(text), encoding="utf-8")

    def step(self, name: str, action):
        """Run a step once. A finished step's result is kept, so a restart skips it."""
        if name in self.state["done"]:
            return self.state["done"][name]
        self.state["current"] = name
        self.save()
        self.say(f"=== {name}")
        started = time.time()
        result = action()
        self.state["done"][name] = {"result": result, "seconds": round(time.time() - started), "ended": time.time()}
        self.state["current"] = None
        self.save()
        return self.state["done"][name]

    def head(self) -> str:
        done = subprocess.run(["git", "-C", str(self.sandbox.work / "repo"), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
        return done.stdout.strip()

    def task_json(self) -> tuple[dict | None, str | None]:
        path = self.sandbox.work / "build/TASK.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None, f"{self.W}/build/TASK.json does not exist"
        except ValueError as exc:
            return None, f"{self.W}/build/TASK.json is not valid JSON: {exc}"
        missing = [key for key in TASK_FIELDS if not data.get(key) and data.get(key) != 0]
        if missing:
            return None, f"{self.W}/build/TASK.json lacks {', '.join(missing)}"
        need = int(data["work_gpus"]) + int(data["judge_gpus"])
        if need > len(self.config.gpus):
            return None, f"TASK.json needs {need} GPUs; this run has {len(self.config.gpus)}"
        return data, None

    def gpu_values(self, task: dict) -> dict[str, str]:
        """Work GPUs first, then Judge GPUs, as nvidia-smi numbers them where the commands run."""
        index = self.sandbox.gpu_index
        work = [index(i) for i in range(int(task["work_gpus"]))]
        judge = [index(i) for i in range(len(work), len(work) + int(task["judge_gpus"]))]
        return {"work_gpu": str(work[0]) if work else "", "work_gpus": ",".join(map(str, work)),
                "judge_gpu": str(judge[0]) if judge else "", "judge_gpus": ",".join(map(str, judge))}

    # ---- the builder ----------------------------------------------------------------------

    def builder_spec(self) -> AgentSpec:
        kind = self.config.builder_agent
        runtime = self.config.claude_runtime if kind == "claude-code" else self.config.codex_runtime
        return AgentSpec(kind, self.config.builder_model, self.config.builder_effort, runtime,
                         f"{self.sandbox.paths['hooks']}/{self.settings_file.name}")

    def builder_turn(self, name: str, prompt: str, hours: float) -> dict:
        """One stretch of the builder's session. Its processes and containers do not outlive it."""
        spec = self.builder_spec()
        path = self.sandbox.write_file(f"build/.prompts/{name}.md", self.render(prompt))
        self.sandbox.start_builder()
        secrets: dict[str, str] = {}
        if spec.kind == "claude-code":
            secrets.update(claude_credentials())
        else:
            self.sandbox.install_codex_auth(codex_auth_path())
        driver = make_driver(self.sandbox, spec, self.sandbox.builder, secrets)
        log = self.dir / "logs/agent" / f"{name}.jsonl"
        session = self.state.get("session") or self.recover_session()
        prompt_path = f"{self.W}/build/.prompts/{path.name}"
        if session and log.exists() and log.stat().st_size:
            # The tool restarted in the middle of this turn: its instructions are already in the session.
            self.sandbox.write_file("build/.prompts/continue.md",
                                    "The tool was restarted while you were working. Every process you had running was "
                                    "stopped, including background jobs and containers. Check what finished, restart what "
                                    "did not, and continue from where you stopped.\n")
            prompt_path = f"{self.W}/build/.prompts/continue.md"
        self.say(f"builder {spec.kind} {spec.model} starts: {name}" + (f" (resuming {session})" if session else ""))
        outcome = run_until_done(driver, prompt_path, log, time.time() + hours * 3600,
                                 session=session, watchdog=self.watchdog, say=self.say)
        if outcome.session_id:
            # Keep this turn's own cost, not the session's running total.
            running = outcome.usage.cost_usd
            outcome.usage.cost_usd = max(0.0, running - self.state["costs"].get(outcome.session_id, 0.0))
            self.state["costs"][outcome.session_id] = running
            self.state["session"] = outcome.session_id
        row = {"name": name, **outcome.to_json()}
        self.state["turns"].append(row)
        self.save()
        self.sandbox.stop_builder()
        self.say(f"builder finished {name} in {int(outcome.seconds)} s, ${outcome.usage.cost_usd:.1f}; "
                 f"error: {outcome.error}")
        if outcome.aborted:
            raise Stop("paused", outcome.aborted)
        return row

    def recover_session(self) -> str | None:
        """If the tool stopped in the middle of a turn, the session is still in that turn's log."""
        for log in sorted((self.dir / "logs/agent").glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            with open(log, "rb") as fh:
                for raw in fh:
                    try:
                        row = json.loads(raw)
                    except ValueError:
                        continue
                    found = isinstance(row, dict) and (row.get("session_id") or row.get("thread_id"))
                    if found:
                        return str(found)
        return None

    # ---- steps ----------------------------------------------------------------------------

    def setup(self) -> dict:
        repo = self.sandbox.work / "repo"
        if not (repo / ".git").exists():
            if repo.exists():
                repo.rmdir()
            subprocess.run(["git", "clone", "-q", self.config.reference, str(repo)], check=True)
            # The builder needs the commit, not where on this host it came from.
            subprocess.run(["git", "-C", str(repo), "remote", "remove", "origin"], check=True)
            subprocess.run(["git", "-C", str(repo), "reflog", "expire", "--expire=now", "--all"], check=True)
        base = self.head()
        self.state["base_commit"] = base
        self.save()
        build = self.sandbox.work / "build"
        build.mkdir(parents=True, exist_ok=True)
        (build / "BASE_COMMIT").write_text(base + "\n", encoding="utf-8")
        tasks = sorted(p.name for p in (repo / "tasks/latest").iterdir() if p.is_dir())
        (build / "REFERENCE.md").write_text(
            f"# About {self.W}/repo\n\nAn Infrathon checkout with these example tasks:\n\n"
            + "".join(f"- `{name}`\n" for name in tasks)
            + "\nDocuments and tables may mention tasks that are not in this copy. A root `conftest.py`, if present, "
              "skips the few framework tests that need an absent task.\n", encoding="utf-8")
        store = self.sandbox.work / "store"
        store.mkdir(parents=True, exist_ok=True)
        (store / ".sforge-version-store.json").write_text(
            json.dumps({"schema_version": 1, "kind": "sforge-version-store", "store_id": self.config.run_id}) + "\n",
            encoding="utf-8")
        self.say(f"preparing the {'sandbox' if self.config.sandbox == 'docker' else 'run directory'}")
        self.sandbox.up()
        self.sandbox.start_builder()
        self.say("installing the framework environment and running its tests")
        # --global is the builder's own home (/work/home), never the user's.
        script = ("git config --global user.name builder && git config --global user.email builder@localhost && "
                  "git config --global init.defaultBranch main && git config --global --add safe.directory '*' && "
                  + BOOTSTRAP + " && .venv/bin/python -m pytest -q -p no:cacheprovider")
        log = self.dir / "evidence/setup/pytest.log"
        done = self.sandbox.run(self.sandbox.builder, script, log=log, timeout=3600, workdir=f"{self.W}/repo")
        if not done.ok:
            raise Stop("failed", f"the reference checkout's own tests fail; see {log}")
        result: dict = {"base_commit": base, "example_tasks": tasks}
        if self.config.preflight:
            rows = self.preflight()
            failed = [row.get("name") for row in rows if not row.get("pass")]
            if failed or not rows:
                raise Stop("failed", f"preflight failed ({len(rows)} checks): {failed}")
            result["preflight"] = f"{len(rows)} checks passed"
        self.sandbox.stop_builder()
        return result

    def preflight(self) -> list[dict]:
        """The operator's own checks, run where the builder runs. Each output line is one JSON check:
        {"name": ..., "pass": true|false, ...}. HOST_HOME names the host user's home."""
        self.sandbox.start_builder()
        extra = {"HOST_HOME": str(Path.home())}
        argv = self.sandbox.exec_argv(self.sandbox.builder, "bash -s", env=extra, stdin=True)
        with open(self.config.preflight, "rb") as script:
            done = subprocess.run(argv, stdin=script, capture_output=True, timeout=900, env={**os.environ, **extra},
                                  cwd=str(self.sandbox.work))
        rows = []
        for line in done.stdout.decode("utf-8", "replace").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        evidence = self.dir / "evidence/setup"
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "preflight.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        self.say(f"preflight: {len(rows)} checks, {sum(not r.get('pass') for r in rows)} failed")
        return rows

    def build(self) -> dict:
        return self.builder_turn("build", (PROMPTS / "build.md").read_text(encoding="utf-8"), self.config.build_hours)

    def fresh_clone(self, name: str, log: Path) -> str | None:
        """A clone of committed work only, in a new directory, with its own environment."""
        target = f"{self.W}/clean/{name}-{time.strftime('%m%d-%H%M%S')}"
        done = self.sandbox.run(self.sandbox.runner, f"git clone -q {self.W}/repo {target} && cd {target} && {BOOTSTRAP}",
                                log=log, timeout=1800)
        return target if done.ok else None

    @property
    def auth(self) -> str:
        """The trial agent's Codex credentials, where the check and trial commands see them."""
        return self.sandbox.paths["auth"] or str(codex_auth_path())

    def runner_env(self, task: dict | None) -> dict[str, str]:
        env = review_env(self.config, self.auth, self.sandbox.paths["runtimes"])
        if task:
            env.update({"IBUILD_WORK_GPUS": self.gpu_values(task)["work_gpus"],
                        "IBUILD_JUDGE_GPUS": self.gpu_values(task)["judge_gpus"]})
        return env

    def check(self, label: str) -> dict:
        """The task's own check, from a clean clone. Failures go back to the builder."""
        attempts = []
        for attempt in range(1, self.config.check_tries + 1):
            result = self.run_check(f"{label}-{attempt}")
            attempts.append(result)
            self.say(f"check {label}-{attempt}: {'pass' if result['passed'] else 'FAIL: ' + result['why']}")
            if result["passed"]:
                self.state["last_check"] = {"commit": result["commit"], "clone": result["clone"]}
                self.save()
                return {"passed": True, "attempts": attempts}
            if attempt == self.config.check_tries:
                break
            prompt = fill((PROMPTS / "check_failed.md").read_text(encoding="utf-8"),
                          commit=result.get("commit", "?"), why=result["why"], log=result.get("log", "(none)"),
                          tail=result.get("tail", ""))
            self.builder_turn(f"fix-check-{label}-{attempt}", prompt, self.config.fix_hours)
        raise Stop("blocked", f"check {label} still fails after {self.config.check_tries} tries: {attempts[-1]['why']}")

    def run_check(self, name: str) -> dict:
        task, problem = self.task_json()
        commit = self.head()
        if task is None:
            return {"passed": False, "why": problem, "commit": commit}
        evidence = self.dir / "evidence" / f"check-{name}"
        evidence.mkdir(parents=True, exist_ok=True)
        self.sandbox.stop_builder()
        self.sandbox.start_runner(codex_auth_path())
        try:
            clone = self.fresh_clone(f"check-{name}", evidence / "clone.log")
            if clone is None:
                log = self.copy_for_builder(evidence / "clone.log", f"build/feedback/check-{name}/clone.log")
                return {"passed": False, "why": "cloning and installing the framework failed", "commit": commit,
                        "log": log, "tail": self._tail(evidence / "clone.log")}
            self.register_task(task)
            command = fill(task["check_cmd"], **self.gpu_values(task), auth=self.auth, run_id=f"ibuild-check-{name}",
                           agent=self.config.solver_agent, model=self.config.solver_model,
                           effort=self.config.solver_effort, hours=1)
            hours = min(float(task.get("check_hours") or self.config.check_hours), 12.0)
            self.say(f"check {name}: {command}")
            log_path = evidence / "check.log"
            done = self.sandbox.run(self.sandbox.runner, command, log=log_path, timeout=hours * 3600, workdir=clone,
                                    env=self.runner_env(task), watchdog=self.watchdog)
        finally:
            self.sandbox.stop_runner()
        log = self.copy_for_builder(log_path, f"build/feedback/check-{name}/check.log")
        if done.aborted:
            raise Stop("paused", done.aborted)
        why = "" if done.ok else (f"it ran out of its {hours:g} h" if done.timed_out else f"it exited with {done.returncode}")
        return {"passed": done.ok, "why": why, "commit": commit, "clone": clone, "seconds": round(done.seconds),
                "log": log, "tail": self._tail(log_path)}

    @staticmethod
    def _tail(path: Path, lines: int = 60) -> str:
        if not path.exists():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    def register_task(self, task: dict) -> None:
        entry = self.sandbox.work / "store/tasks" / task["task"]
        if not (entry / "task.json").exists():
            entry.mkdir(parents=True, exist_ok=True)
            (entry / "task.json").write_text(json.dumps({"schema_version": 1, "task_directory": task["task"],
                                                         "task_id": task["task_id"]}) + "\n", encoding="utf-8")

    def checked_clone(self) -> str | None:
        """The clone where check_cmd last passed, if it holds the current HEAD: images and data are built there."""
        last = self.state.get("last_check") or {}
        if not last:
            passed = [a for step in self.state["done"].values() for a in (step.get("result") or {}).get("attempts", [])
                      if isinstance(a, dict) and a.get("passed") and a.get("clone")]
            last = passed[-1] if passed else {}
        return last.get("clone") if last.get("commit") == self.head() else None

    def trial(self, round_number: int) -> dict:
        """A trial that never got going (the launcher stopped at once, with nothing submitted) says
        nothing about the task's difficulty, so it goes back to the builder instead of using up a round."""
        for attempt in range(1, self.config.check_tries + 1):
            summary = self.run_trial(round_number, attempt)
            if summary["submissions"] or summary["wall_seconds"] >= 600:
                return summary
            why = f"the launcher exited with {summary['launcher_exit']} after {summary['wall_seconds']} s, with no submission"
            self.say(f"trial {round_number}: did not start: {why}")
            if attempt == self.config.check_tries:
                raise Stop("blocked", f"trial {round_number} never started after {attempt} tries: {why}")
            prompt = fill((PROMPTS / "trial_failed.md").read_text(encoding="utf-8"), round=round_number,
                          commit=summary["commit"][:12], why=why, log=summary["launcher_log"],
                          tail=self._tail(self.dir / "evidence" / f"trial-{round_number}-{attempt}" / "launcher.log"))
            self.builder_turn(f"fix-trial-{round_number}-{attempt}", prompt, self.config.fix_hours)
            self.check(f"{round_number}-t{attempt}")
        raise Stop("blocked", f"trial {round_number} never started")

    def run_trial(self, round_number: int, attempt: int) -> dict:
        task, problem = self.task_json()
        if task is None:
            raise Stop("blocked", problem)
        evidence = self.dir / "evidence" / f"trial-{round_number}-{attempt}"
        evidence.mkdir(parents=True, exist_ok=True)
        run_id = f"ibuild-trial-{round_number}-{time.strftime('%m%d%H%M')}"
        commit = self.head()
        self.sandbox.stop_builder()
        self.sandbox.start_runner(codex_auth_path())
        try:
            # Run where check_cmd just passed, as a maintainer would: build and check, then run.
            clone = self.checked_clone() or self.fresh_clone(f"trial-{round_number}", evidence / "clone.log")
            if clone is None:
                raise Stop("failed", f"could not make a clean clone for the trial; see {evidence / 'clone.log'}")
            # Containers in the check may have left root-owned outputs behind (on this machine, best effort).
            self.sandbox.run(self.sandbox.runner,
                             f"chown -R {self.sandbox.uid}:{self.sandbox.gid} {shlex.quote(clone)} 2>/dev/null; true",
                             user="0", timeout=600)
            self.register_task(task)
            hours = self.config.trial_hours
            command = fill(task["trial_cmd"], **self.gpu_values(task), auth=self.auth, run_id=run_id,
                           agent=self.config.solver_agent, model=self.config.solver_model,
                           effort=self.config.solver_effort,
                           hours=int(hours) if float(hours).is_integer() else hours)
            self.say(f"trial {round_number}: {self.config.solver_model} for {hours} h: {command}")
            started = time.time()
            done = self.sandbox.run(self.sandbox.runner, command, log=evidence / "launcher.log",
                                    timeout=hours * 3600 + 4 * 3600, workdir=clone, env=self.runner_env(task),
                                    watchdog=self.watchdog)
            output_dir = f"{clone}/{fill(task['trial_output'], run_id=run_id)}"
            # Nested containers write as root: let the builder read everything the run left.
            self.sandbox.run(self.sandbox.runner,
                             f"chmod -R a+rX {shlex.quote(clone)}/outputs {self.W}/store 2>/dev/null; true",
                             user="0", timeout=600)
            submissions = self.read_submissions(output_dir, task.get("score_key") or "score")
        finally:
            self.sandbox.stop_runner()
        valid = [s["score"] for s in submissions if s.get("valid") and isinstance(s.get("score"), (int, float))]
        summary = {
            "round": round_number, "commit": commit, "agent": self.config.solver_agent, "model": self.config.solver_model,
            "effort": self.config.solver_effort, "hours": hours, "wall_seconds": round(time.time() - started),
            "launcher_exit": done.returncode, "launcher_timed_out": done.timed_out, "run_id": run_id,
            "output_dir": output_dir, "submissions": submissions, "valid_scores": valid,
            "best": (max(valid) if valid else None),
        }
        (evidence / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        target = f"build/trials/round-{round_number}" if (submissions or summary["wall_seconds"] >= 600) \
            else f"build/feedback/trial-{round_number}-{attempt}"
        summary["launcher_log"] = self.copy_for_builder(evidence / "launcher.log", f"{target}/launcher.log")
        self.sandbox.write_file(f"{target}/SUMMARY.json", json.dumps(summary, indent=2) + "\n")
        self.say(f"trial {round_number}: {len(submissions)} submissions, {len(valid)} valid, best {summary['best']}; "
                 f"launcher exit {done.returncode}")
        if done.aborted:
            raise Stop("paused", done.aborted)
        return summary

    def read_submissions(self, output_dir: str, key: str) -> list[dict]:
        """What the Judge returned for each submission, read as root inside the sandbox."""
        script = r'''
import json, pathlib, sys
root, key = pathlib.Path(sys.argv[1]), sys.argv[2]
rows = []
for path in sorted(root.glob("**/report.json")) if root.exists() else []:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        rows.append({"report": str(path), "error": f"unreadable: {exc}"}); continue
    if not isinstance(data, dict):
        continue
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    rows.append({"report": str(path), "valid": bool(data.get("valid")), "score": data.get(key, metrics.get(key)),
                 "submitted_at": data.get("submitted_at"), "judge_seconds": data.get("runtime_seconds"),
                 "summary": str(data.get("summary") or data.get("reason") or "")[:600]})
rows.sort(key=lambda r: (r.get("submitted_at") or 0, r["report"]))
print(json.dumps(rows))
'''
        command = f"python3 -c {shlex.quote(script)} {shlex.quote(output_dir)} {shlex.quote(key)}"
        done = self.sandbox.run(self.sandbox.runner, command, user="0", timeout=600)
        try:
            return json.loads(done.output.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return [{"error": f"could not read submissions: {done.output[-400:]}"}]

    def review(self, round_number: int, summary: dict) -> dict:
        task_summary = json.dumps({k: v for k, v in summary.items() if k != "submissions"} |
                                  {"submissions": summary["submissions"][-40:]}, indent=2)
        prompt = fill((PROMPTS / "review.md").read_text(encoding="utf-8"), round=round_number,
                      solver=f"{self.config.solver_agent} ({self.config.solver_model}, effort {self.config.solver_effort})",
                      hours=summary["hours"], commit=summary["commit"][:12],
                      summary=f"{self.W}/build/trials/round-{round_number}/SUMMARY.json",
                      launcher_log=summary.get("launcher_log", ""), output_dir=summary["output_dir"],
                      summary_text=task_summary)
        before = self.head()
        self.builder_turn(f"review-{round_number}", prompt, self.config.review_hours)
        path = self.sandbox.work / f"build/reviews/round-{round_number}.json"
        try:
            review = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            review = {"decision": "missing", "why": f"no readable review file: {exc}"}
        review["changed_task"] = self.head() != before
        self.say(f"review {round_number}: {review.get('decision')}; task changed: {review['changed_task']}")
        return review

    # ---- main loop ------------------------------------------------------------------------

    def run(self) -> str:
        from .report import finalize
        try:
            if not self.sandbox.status().get(getattr(self.sandbox, "dind", ""), False):
                try:
                    check_idle(self.config.gpus)
                except GpuBusy as exc:
                    raise Stop("paused", f"GPUs are busy: {exc}") from exc
            try:
                self.locks.acquire()
            except GpuBusy as exc:
                raise Stop("paused", str(exc)) from exc
            problem = self.watchdog()
            if problem:
                raise Stop("paused", problem)
            self.state.update(status="running", pid=os.getpid(), why=None)
            self.state.setdefault("started", time.time())
            self.save()

            self.prepare_guide()
            if "setup" in self.state["done"]:
                self.sandbox.up()  # after a host restart, the gateway and nested dockerd come back first
            self.step("setup", self.setup)
            self.step("build", self.build)
            checked = None
            for round_number in range(1, self.config.rounds + 1):
                checked = self.step(f"check-{round_number}", lambda: self.check(str(round_number)))
                summary = self.step(f"trial-{round_number}", lambda: self.trial(round_number))["result"]
                review = self.step(f"review-{round_number}", lambda: self.review(round_number, summary))["result"]
                if review.get("decision") == "blocked":
                    raise Stop("blocked", f"the builder needs the expert: {review.get('why', '')[:400]}")
                if review.get("decision") != "retrial":
                    break
            last_checked = ((checked or {}).get("result") or {}).get("attempts", [{}])[-1].get("commit")
            if self.head() != last_checked:
                self.step("check-final", lambda: self.check("final"))
            self.state["status"] = "finished"
        except Stop as stop:
            self.state["status"] = stop.status
            self.state["why"] = stop.why
            self.say(f"stopped: {stop.status}: {stop.why}")
        except Exception as exc:  # the state must survive whatever happens
            self.state["status"] = "crashed"
            self.state["why"] = repr(exc)
            self.say(f"crashed: {exc!r}")
            self.save()
            raise
        finally:
            self.locks.release()
            self.save()
        if self.state["status"] in ("finished", "blocked"):
            self.sandbox.stop_builder()
            finalize(self)
        return self.state["status"]


def create_run(run_id: str, input_dir: Path, reference: Path, **overrides) -> Path:
    import shutil
    from .config import newest_runtime
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        raise SystemExit(f"run {run_id} already exists at {run_dir}")
    if not (input_dir / "TASK_DEFINITION.md").is_file():
        raise SystemExit(f"{input_dir} has no TASK_DEFINITION.md")
    if not (reference / ".git").exists() or not (reference / "sforge").is_dir():
        hint = (" (run `git submodule update --init` in the infrathon-build checkout, or pass --reference)"
                if reference == TOOL_ROOT / "infrathon" else "")
        raise SystemExit(f"{reference} is not an Infrathon git checkout{hint}")
    run_dir.mkdir(parents=True)
    shutil.copytree(input_dir, run_dir / "input")
    values = {k: v for k, v in overrides.items() if v is not None}
    values.setdefault("claude_runtime", newest_runtime("claude-code"))
    values.setdefault("codex_runtime", newest_runtime("codex"))
    config = RunConfig(run_id=run_id, input_dir=str(run_dir / "input"), reference=str(reference.resolve()), **values)
    config.save(run_dir)
    return run_dir


_ = re
