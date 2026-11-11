import json
import time

import pytest
from conftest import git, write_json

from infrathon_build.pipeline import Stop, fill, render_paths


def test_fill_replaces_only_named_placeholders():
    text = 'run.sh --gpu {work_gpu} --x ${HOME} --json \'{"a": 1}\' {unknown}'
    assert fill(text, work_gpu=3) == 'run.sh --gpu 3 --x ${HOME} --json \'{"a": 1}\' {unknown}'


def test_render_paths_maps_roots_but_not_lookalikes():
    paths = {"work": "/runs/r/work", "input": "/runs/r/input", "guide": "/runs/r/guide"}
    text = "`/work` and /work/repo; images/work/A.md; (/input/x) /guide/README.md; /workspace/x; /work"
    assert render_paths(text, paths) == (
        "`/runs/r/work` and /runs/r/work/repo; images/work/A.md; (/runs/r/input/x) "
        "/runs/r/guide/README.md; /workspace/x; /runs/r/work")


def test_create_run_refuses_a_missing_reference(home, expert_input, tmp_path):
    from infrathon_build.pipeline import create_run
    with pytest.raises(SystemExit, match="not an Infrathon git checkout"):
        create_run("bad", expert_input, tmp_path / "nowhere")
    assert not (home / "runs/bad").exists()


def test_local_run_defaults(make_run):
    run = make_run()
    assert run.config.sandbox == "none"
    assert run.config.claude_runtime == "claude-code-2.1.0"
    assert run.W == str(run.dir / "work")


def test_step_runs_once_and_survives_a_restart(make_run):
    from infrathon_build.pipeline import Run
    run = make_run()
    calls = []
    assert run.step("a", lambda: calls.append(1) or {"x": 1})["result"] == {"x": 1}
    assert run.step("a", lambda: calls.append(2))["result"] == {"x": 1}
    assert Run("r1").step("a", lambda: calls.append(3))["result"] == {"x": 1}
    assert calls == [1]


def test_task_json_is_validated(make_run):
    run = make_run()
    assert "does not exist" in run.task_json()[1]
    path = run.sandbox.work / "build/TASK.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{")
    assert "not valid JSON" in run.task_json()[1]
    write_json(path, {"task": "t"})
    assert "lacks task_id" in run.task_json()[1]
    full = {"task": "t", "task_id": "t1", "launch": "launch/t", "work_gpus": 1, "judge_gpus": 2,
            "check_cmd": "c", "trial_cmd": "r", "trial_output": "o"}
    write_json(path, full)
    assert "needs 3 GPUs" in run.task_json()[1]
    write_json(path, dict(full, judge_gpus=1))
    assert run.task_json() == (dict(full, judge_gpus=1), None)


def test_gpu_values_use_host_indices_on_this_machine(make_run):
    run = make_run(gpus=[2, 5, 6])
    values = run.gpu_values({"work_gpus": 1, "judge_gpus": 2})
    assert values == {"work_gpu": "2", "work_gpus": "2", "judge_gpu": "5", "judge_gpus": "5,6"}


def test_trial_reuses_the_clone_where_the_check_passed(make_run):
    run = make_run()
    run.sandbox.up()
    git(run.sandbox.work / "repo", "init", "-q", "-b", "main")
    git(run.sandbox.work / "repo", "commit", "-q", "--allow-empty", "-m", "base")
    head = run.head()
    run.state["last_check"] = {"commit": head, "clone": "/somewhere"}
    assert run.checked_clone() == "/somewhere"
    run.state["last_check"]["commit"] = "0" * 40
    assert run.checked_clone() is None


def test_guide_is_rendered_for_the_local_machine(make_run):
    run = make_run(gpus=[3])
    run.prepare_guide()
    environment = (run.dir / "guide/ENVIRONMENT.md").read_text()
    assert environment.startswith("# The machine: a shared host")
    assert "GPUs are **3**" in environment
    assert f"`{run.W}`" in environment
    readme = (run.dir / "guide/README.md").read_text()
    assert f"{run.W}/repo" in readme and "`/work/repo`" not in readme


def test_local_host_runs_captures_and_times_out(make_run, tmp_path):
    run = make_run()
    run.sandbox.up()
    done = run.sandbox.run("local", "echo $HOME; echo $IBUILD_GPUS", timeout=30)
    assert done.ok
    assert done.output.split() == [str(run.sandbox.work / "home"), "2,5"]
    started = time.time()
    slow = run.sandbox.run("local", "sleep 30 & sleep 30", timeout=1)
    assert slow.timed_out and not slow.ok and time.time() - started < 20


def test_stop_carries_status_and_reason():
    stop = Stop("blocked", "why")
    assert (stop.status, stop.why, str(stop)) == ("blocked", "why", "why")


def test_report_lists_steps_trials_and_the_builders_report(make_run):
    from infrathon_build.report import finalize
    run = make_run()
    repo = run.sandbox.work / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "commit", "-q", "--allow-empty", "-m", "base")
    run.state["base_commit"] = run.head()
    (repo / "tasks/latest/t").mkdir(parents=True)
    (repo / "tasks/latest/t/README.md").write_text("# T\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "Add t")
    write_json(run.sandbox.work / "build/TASK.json", {"task": "t", "launch": "launch/t"})
    (run.sandbox.work / "build/REPORT.md").write_text("# Builder says hello\n")
    run.state["done"] = {
        "trial-1": {"seconds": 3600, "ended": time.time(), "result": {
            "round": 1, "model": "m", "hours": 1, "commit": run.head(), "launcher_exit": 0,
            "submissions": [{"valid": True, "score": 5.0, "judge_seconds": 12.7, "summary": "ok"}],
            "valid_scores": [5.0], "best": 5.0}},
        "review-1": {"seconds": 60, "ended": time.time(), "result": {"decision": "ready", "why": "fine"}},
    }
    run.state["status"] = "finished"
    path = finalize(run)
    text = path.read_text()
    assert "**Status:** finished" in text
    assert "best score **5.0**" in text and "| 1 | True | 5.0 | 13 | ok |" in text
    assert "Builder's review: **ready**" in text
    assert text.rstrip().endswith("# Builder says hello")
    patch = (run.dir / "deliverable/task-only.patch").read_text()
    assert "tasks/latest/t/README.md" in patch
    assert json.loads((run.dir / "state.json").read_text())["verdict"] == "finished"
