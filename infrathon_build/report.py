"""The deliverable: the task as patches and a git bundle, and a report the builder cannot edit.

The report has two parts: what the tool observed (steps, checks, trials, cost, isolation),
and the builder's own report for the expert, appended as written.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(v).replace("\n", " ").replace("|", "\\|") for v in row) + " |" for row in rows]
    return "\n".join(lines)


def make_deliverable(run) -> dict:
    repo = run.sandbox.work / "repo"
    base = run.state.get("base_commit")
    out: dict = {}
    if not base or not (repo / ".git").exists():
        return out
    deliver = run.dir / "deliverable"
    deliver.mkdir(exist_ok=True)
    full = subprocess.run(["git", "-C", str(repo), "diff", "--binary", f"{base}..HEAD"], capture_output=True)
    (deliver / "task.patch").write_bytes(full.stdout)
    out["patch_bytes"] = len(full.stdout)
    out["secret_in_patch"] = any(secret.encode() in full.stdout for secret in run.secret_values())
    task = _load(run.sandbox.work / "build/TASK.json")
    if task.get("task") and task.get("launch"):
        only = subprocess.run(["git", "-C", str(repo), "diff", "--binary", f"{base}..HEAD", "--",
                               f"tasks/latest/{task['task']}", task["launch"]], capture_output=True)
        (deliver / "task-only.patch").write_bytes(only.stdout)
        out["task_only_patch_bytes"] = len(only.stdout)
    bundle = deliver / "task.bundle"
    done = subprocess.run(["git", "-C", str(repo), "bundle", "create", str(bundle), f"{base}..HEAD"],
                          capture_output=True, text=True)
    out["bundle"] = str(bundle) if done.returncode == 0 else f"failed: {done.stderr[-300:]}"
    log = subprocess.run(["git", "-C", str(repo), "log", "--oneline", f"{base}..HEAD"], capture_output=True, text=True)
    out["commits"] = log.stdout.strip().splitlines()
    return out


def finalize(run) -> Path:
    state = run.state
    config = run.config
    build = run.sandbox.work / "build"
    deliverable = make_deliverable(run)
    verdict = state["status"]
    if deliverable.get("secret_in_patch"):
        verdict = "blocked"
        state["why"] = "a credential is in the patch; it must not leave this machine"

    parts = [f"# Task build report: {config.run_id}", "",
             f"- **Status:** {verdict}" + (f": {state['why']}" if state.get("why") else ""),
             f"- **Input:** `{config.input_dir}`",
             f"- **Reference:** `{config.reference}` at `{(state.get('base_commit') or '')[:12]}`",
             f"- **Builder:** {config.builder_agent} `{config.builder_model}` (effort {config.builder_effort})",
             f"- **Trial agent:** {config.solver_agent} `{config.solver_model}` (effort {config.solver_effort}), "
             f"{config.trial_hours:g} h per trial, up to {config.rounds} trials",
             ""]

    rows = []
    started = state.get("started") or 0
    for name, step in state.get("done", {}).items():
        ended = step.get("ended") or 0
        rows.append([name, f"{step.get('seconds', 0) / 3600:.1f}",
                     f"{(ended - started) / 3600:.1f}" if started and ended >= started else ""])
    parts += ["## Steps", "", _table(["step", "hours", "done, hours after start"], rows), ""]

    turns = state.get("turns", [])
    if turns:
        rows = [[t["name"], f"{t['seconds'] / 3600:.1f}", t["usage"]["output_tokens"], f"{t['usage']['cost_usd']:.1f}",
                 (t.get("error") or "")[:80]] for t in turns]
        total = sum(t["usage"]["cost_usd"] for t in turns)
        parts += ["## Builder", "", _table(["turn", "hours", "tokens out", "cost $", "error"], rows), "",
                  f"Builder cost as its CLI reports it: ${total:.1f}. Trial agents and integrity reviews are not included.", ""]

    checks = [(name, step["result"]) for name, step in state.get("done", {}).items() if name.startswith("check-")]
    if checks:
        rows = [[name, a.get("commit", "")[:12], "pass" if a.get("passed") else "fail", f"{(a.get('seconds') or 0) / 60:.0f}",
                 a.get("why", "")] for name, result in checks for a in result.get("attempts", [])]
        parts += ["## Checks from a clean clone", "", _table(["step", "commit", "result", "minutes", "why"], rows), ""]

    for name, step in state.get("done", {}).items():
        if not name.startswith("trial-"):
            continue
        trial = step["result"]
        parts += [f"## Trial {trial['round']}", "",
                  f"{trial['model']} for {trial['hours']} h on commit `{trial['commit'][:12]}`: "
                  f"{len(trial['submissions'])} submissions, {len(trial['valid_scores'])} valid, best score "
                  f"**{trial['best']}**. Launcher exit {trial['launcher_exit']}"
                  + (" (timed out)" if trial.get("launcher_timed_out") else "") + ".", ""]
        rows = [[i + 1, s.get("valid"), s.get("score"),
                 round(s["judge_seconds"]) if isinstance(s.get("judge_seconds"), (int, float)) else "",
                 (s.get("summary") or s.get("error") or "")[:160]]
                for i, s in enumerate(trial["submissions"])]
        if rows:
            parts += [_table(["#", "valid", "score", "Judge s", "summary"], rows), ""]
        review = (state["done"].get(f"review-{trial['round']}") or {}).get("result") or {}
        if review:
            parts += [f"Builder's review: **{review.get('decision')}**. {review.get('why', '')}", ""]
            findings = review.get("findings") or []
            if findings:
                parts += [_table(["kind", "severity", "what", "fixed"],
                                 [[f.get("kind"), f.get("severity"), (f.get("what") or "")[:200], (f.get("fixed") or "")[:120]]
                                  for f in findings]), ""]

    if config.preflight or config.network_policy:
        setup = (state.get("done", {}).get("setup") or {}).get("result") or {}
        denied = [row for row in run.sandbox.egress_rows() if str(row.get("event", "")).startswith("deny")]
        parts += ["## Sandbox", "",
                  f"- preflight: {setup.get('preflight') or setup.get('probe') or 'not run'}",
                  f"- requests refused by the network policy: {len(denied)}", ""]

    parts += ["## Deliverable", "",
              f"- `deliverable/task.patch`: everything since the reference commit, {deliverable.get('patch_bytes', 0)} bytes",
              f"- `deliverable/task-only.patch`: the task and launcher directories only, "
              f"{deliverable.get('task_only_patch_bytes', 0)} bytes",
              f"- `deliverable/task.bundle`: the builder's commits, for `git fetch`",
              f"- commits: {len(deliverable.get('commits', []))}", ""]
    builder_report = build / "REPORT.md"
    parts += ["---", "", "# The builder's report", ""]
    parts += [builder_report.read_text(encoding="utf-8") if builder_report.exists() else "The builder wrote no report."]

    path = run.dir / "REPORT.md"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    state["verdict"] = verdict
    state["report"] = str(path)
    run.save()
    run.say(f"report: {path}")
    return path
