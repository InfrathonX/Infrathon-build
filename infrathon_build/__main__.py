"""infrathon-build: turn an expert's task definition into a runnable Infrathon task.

    python3 -m infrathon_build init RUN --input DIR --reference INFRATHON_CHECKOUT
    python3 -m infrathon_build run RUN --detach
    python3 -m infrathon_build status RUN
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .config import HOME, RUNS_ROOT, TOOL_ROOT, free_disk_gb
from .pipeline import Run, create_run, log_line


def parse_gpus(text: str | None) -> list[int] | None:
    """'0-7' or '0,1,4'."""
    if not text:
        return None
    gpus: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        gpus += list(range(int(first), int(last or first) + 1))
    return gpus


def cmd_init(args) -> int:
    run_dir = create_run(args.run, Path(args.input).resolve(), Path(args.reference).resolve(), sandbox=args.sandbox,
                         network_policy=str(Path(args.network_policy).resolve()) if args.network_policy else None,
                         agent_settings=str(Path(args.agent_settings).resolve()) if args.agent_settings else None,
                         preflight=str(Path(args.preflight).resolve()) if args.preflight else None,
                         gpus=parse_gpus(args.gpus), trial_hours=args.trial_hours, rounds=args.rounds,
                         builder_agent=args.builder, builder_model=args.builder_model,
                         solver_model=args.solver_model)
    print(f"created {run_dir}\nstart it with: python3 -m infrathon_build run {args.run} --detach")
    return 0


def session_name(run_id: str) -> str:
    return f"ibuild-{run_id}"


def cmd_run(args) -> int:
    run = Run(args.run)
    if args.detach:
        # A running tmux server does not pass this shell's environment on, so name the home explicitly.
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name(args.run), "-c", str(TOOL_ROOT),
                        f"INFRATHON_BUILD_HOME={shlex.quote(str(HOME))} {shlex.quote(sys.executable)} -m infrathon_build run {args.run} "
                        f">> {shlex.quote(str(run.dir))}/console.log 2>&1"], check=True)
        print(f"running in tmux session {session_name(args.run)}; follow it with: python3 -m infrathon_build status {args.run}")
        return 0
    status = run.run()
    print(status)
    return 0 if status == "finished" else 1


def cmd_status(args) -> int:
    run_dir = RUNS_ROOT / args.run
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8")) if (run_dir / "state.json").exists() else {}
    print(f"run {args.run}: {state.get('status', 'not started')}"
          + (f"  (now: {state['current']})" if state.get("current") else ""))
    if state.get("why"):
        print(f"  why: {state['why'][:600]}")
    for name, step in (state.get("done") or {}).items():
        result = step.get("result") or {}
        note = ""
        if name.startswith("trial-"):
            note = f"{len(result.get('submissions', []))} submissions, {len(result.get('valid_scores', []))} valid, best {result.get('best')}"
        elif name.startswith("review-"):
            note = f"decision {result.get('decision')}, task changed {result.get('changed_task')}"
        elif name.startswith("check-"):
            note = f"{len(result.get('attempts', []))} attempt(s)"
        print(f"  {name:12} {step.get('seconds', 0) / 3600:5.1f} h  {note}")
    cost = sum(t["usage"]["cost_usd"] for t in state.get("turns", []))
    print(f"  builder: {len(state.get('turns', []))} turn(s), ${cost:.1f} as its CLI reports it")
    print(f"  host disk free: {free_disk_gb(run_dir):.0f} GB")
    done = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True)
    busy = [line.replace(" ", "") for line in done.stdout.splitlines() if line.split(",")[1].strip() != "0"]
    print(f"  GPUs in use (index,MiB,util%): {busy or 'none'}")
    log = run_dir / "run.log"
    if log.exists():
        print("  recent:")
        for line in log.read_text(encoding="utf-8").splitlines()[-args.lines:]:
            print(f"    {line[:220]}")
    return 0


def _describe(item: dict) -> str | None:
    if item.get("type") == "text" and item.get("text", "").strip():
        return "says: " + " ".join(item["text"].split())[:300]
    if item.get("type") != "tool_use":
        return None
    name, given = item.get("name", "?"), item.get("input") or {}
    if name == "Bash":
        what = given.get("description") or (given.get("command") or "").strip().splitlines()[0]
    elif name in ("Read", "Write", "Edit", "NotebookEdit"):
        what = given.get("file_path", "")
    elif name in ("WebSearch", "WebFetch"):
        what = given.get("query") or given.get("url") or ""
    elif name in ("Grep", "Glob"):
        what = f"{given.get('pattern', '')} {given.get('path', '')}"
    elif name in ("Task", "Agent"):
        what = given.get("description", "")
    else:
        what = json.dumps(given)
    return f"{name}: {' '.join(str(what).split())[:200]}"


def cmd_log(args) -> int:
    """What the builder is doing, readable: its recent messages and tool calls."""
    logs = sorted((RUNS_ROOT / args.run / "logs/agent").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print("no builder transcript yet")
        return 0
    log = logs[-1]
    items: list[str] = []
    calls = 0
    with open(log, "rb") as fh:
        for raw in fh:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            content = (row.get("message") or {}).get("content") if isinstance(row, dict) else None
            if row.get("type") != "assistant" or not isinstance(content, list):
                continue
            for item in content:
                line = _describe(item)
                if line:
                    items.append(line)
                    calls += item.get("type") == "tool_use"
    age = (time.time() - log.stat().st_mtime) / 60
    print(f"turn {log.stem}: {calls} tool calls so far; last activity {age:.0f} min ago")
    for line in items[-args.lines:]:
        print(f"  {line}")
    return 0


def cmd_stop(args) -> int:
    """Stop the run now. The sandbox's data stays, and `run` resumes from the last finished step."""
    run = Run(args.run)
    subprocess.run(["tmux", "kill-session", "-t", session_name(args.run)], capture_output=True)
    run.sandbox.kill_agents()
    run.sandbox.stop_builder()
    run.sandbox.stop_runner()
    run.state["status"] = "stopped"
    run.state["why"] = args.why or "stopped by the operator"
    run.state["stopped_at"] = time.time()
    run.save()
    log_line(run.dir, f"stopped: {run.state['why']}")
    print("stopped; resume with: python3 -m infrathon_build run", args.run, "--detach")
    return 0


def cmd_report(args) -> int:
    from .report import finalize
    print(finalize(Run(args.run)))
    return 0


def cmd_preflight(args) -> int:
    run = Run(args.run)
    if not run.config.preflight:
        raise SystemExit("this run has no preflight script (init --preflight)")
    run.sandbox.up()
    rows = run.preflight()
    run.sandbox.stop_builder()
    for row in rows:
        print(("ok  " if row["pass"] else "FAIL"), row["name"], "|", row["observed"][:100].replace("\n", " "))
    return 0 if rows and all(row["pass"] for row in rows) else 1


def cmd_down(args) -> int:
    run = Run(args.run)
    (run.sandbox.destroy if args.destroy else run.sandbox.down)()
    print("containers and volumes removed" if args.destroy else "containers removed; data kept")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="infrathon_build", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a run from an expert's folder")
    init.add_argument("run")
    init.add_argument("--input", required=True, help="folder with TASK_DEFINITION.md and attachments/")
    init.add_argument("--reference", default=str(TOOL_ROOT / "infrathon"),
                      help="Infrathon git checkout the builder starts from (default: the infrathon/ submodule)")
    init.add_argument("--sandbox", choices=["none", "docker"], default="none",
                      help="none (default): run directly on this machine; docker: isolate the agents in a sandbox")
    init.add_argument("--network-policy", help="JSON of hosts and paths the sandbox gateway refuses (default: none)")
    init.add_argument("--agent-settings", help="Claude Code settings for the builder; hook scripts go beside it")
    init.add_argument("--preflight", help="script run inside the sandbox before the builder starts")
    init.add_argument("--gpus", help="host GPUs for this run, for example 0-7 (default: all eight)")
    init.add_argument("--trial-hours", type=float)
    init.add_argument("--rounds", type=int, help="trials at most (default 2)")
    init.add_argument("--builder", choices=["claude-code", "codex"])
    init.add_argument("--builder-model")
    init.add_argument("--solver-model")
    init.set_defaults(func=cmd_init)

    run = commands.add_parser("run", help="run, or resume after a stop")
    run.add_argument("run")
    run.add_argument("--detach", action="store_true", help="run in a tmux session")
    run.set_defaults(func=cmd_run)

    status = commands.add_parser("status")
    status.add_argument("run")
    status.add_argument("--lines", type=int, default=12)
    status.set_defaults(func=cmd_status)

    log = commands.add_parser("log", help="what the builder is doing: its recent messages and tool calls")
    log.add_argument("run")
    log.add_argument("--lines", type=int, default=25)
    log.set_defaults(func=cmd_log)

    stop = commands.add_parser("stop", help="stop now; data is kept")
    stop.add_argument("run")
    stop.add_argument("--why")
    stop.set_defaults(func=cmd_stop)

    report = commands.add_parser("report", help="write the report again")
    report.add_argument("run")
    report.set_defaults(func=cmd_report)

    preflight = commands.add_parser("preflight", help="run the preflight script again, where the builder runs")
    preflight.add_argument("run")
    preflight.set_defaults(func=cmd_preflight)

    down = commands.add_parser("down", help="remove the containers")
    down.add_argument("run")
    down.add_argument("--destroy", action="store_true", help="also remove the volumes")
    down.set_defaults(func=cmd_down)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
