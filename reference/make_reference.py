#!/usr/bin/env python3
"""Make the Infrathon checkout a builder starts from, with the example tasks it should read.

A builder learns the framework from real tasks, so it gets an Infrathon tree to read
and to add its task to. With every task (`--tasks all`) this is just an export. With
a subset, registries, task tables and the tests that count tasks are trimmed to match,
so the tree's own test suite still passes and every document describes what is there.

The result is one commit with no history, so the builder's work is a clean diff on top.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMON_PATHS = [
    ".gitignore", "LICENSE", "README.md", "docs", "pyproject.toml", "uv.lock", "sforge", "tests",
    "scripts/infrathon",
    "launch/common.sh", "launch/build-agent-runtime.sh", "launch/mlsysforge.env.example", "launch/README.md",
]

# Documents with a table row per task.
TASK_TABLE_DOCS = [
    "README.md", "launch/README.md", "scripts/infrathon/README.md",
    "docs/HOST_PREREQUISITES.md", "docs/NEW_HARDWARE.md", "docs/CPU_AFFINITY.md",
]

# Tests that exercise a specific task a subset may not contain.
SUBSET_IGNORES = {"tests/test_sd35m_metric_v3.py", "tests/test_wan22_v2.py"}
SUBSET_SKIPS = {
    "test_judge_server_checkpoint_api_with_real_orpo_spec",
    "test_per_version_enumeration_on_the_standalone_contract",
    "test_current_minference_private_files_load_without_rewriting_metadata",
    "test_excluded_files_do_not_change_the_source_digest",
}
SUBSET_CONFTEST = f'''"""This checkout holds a subset of the tasks: these tests need one it does not contain."""
import pytest

collect_ignore = {sorted(SUBSET_IGNORES)!r}
NEEDS_ABSENT_TASK = {sorted(SUBSET_SKIPS)!r}


def pytest_collection_modifyitems(config, items):
    marker = pytest.mark.skip(reason="needs a task that is not in this checkout")
    for item in items:
        if item.originalname in NEEDS_ABSENT_TASK:
            item.add_marker(marker)
'''


def git(repo: Path, *args: str, binary: bool = False):
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode()


def export(repo: Path, commit: str, paths: list[str], out: Path) -> None:
    data = git(repo, "archive", "--format=tar", commit, "--", *paths, binary=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(out, filter="tar")


def drop_absent_rows(text: str, names: set[str]) -> tuple[str, int]:
    """Tables list every task. A reader of the subset sees only the tasks it has."""
    if not names:
        return text, 0
    regex = re.compile("|".join(sorted((re.escape(n) for n in names), key=len, reverse=True)))
    kept, removed = [], 0
    for line in text.splitlines():
        if line.lstrip().startswith("|") and regex.search(line):
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept) + ("\n" if text.endswith("\n") else ""), removed


def trim_tables(out: Path, absent: set[str], report: dict) -> None:
    for rel in TASK_TABLE_DOCS:
        path = out / rel
        if path.exists():
            text, rows = drop_absent_rows(path.read_text(encoding="utf-8"), absent)
            path.write_text(text, encoding="utf-8")
            report["trimmed"][rel] = rows


def filter_registry(out: Path, keep: set[str], report: dict) -> None:
    path = out / "scripts/infrathon/tasks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    before = len(data["tasks"])
    data["tasks"] = {name: value for name, value in data["tasks"].items() if name in keep}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    report["trimmed"]["scripts/infrathon/tasks.json"] = before - len(data["tasks"])


def keep_subset_safe_asserts(text: str) -> tuple[str, int]:
    """In the task-list test, keep what a subset must satisfy (the task list matches the tree, the
    README table is current) and drop the rest of that test, which checks specific tasks' hardware."""
    lines = text.splitlines()
    kept: list[str] = []
    dropped = 0
    i = 0
    while i < len(lines):
        kept.append(lines[i])
        if lines[i].strip() == "assert tasks.table(document) in readme":
            i += 1
            while i < len(lines) and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
                dropped += bool(lines[i].strip())
                i += 1
            if i < len(lines):
                kept += ["", ""]
            continue
        i += 1
    return "\n".join(kept).rstrip("\n") + "\n", dropped


def _block_end(lines: list[str], start: int) -> int:
    i = start + 1
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
        i += 1
    return i


def patch_tests(out: Path, keep: set[str], report: dict) -> None:
    """Tests that count every task, or stand on one a subset lacks. Each change is recorded."""
    patched = report["patched_tests"]
    path = out / "tests/sforge/harness/test_llm_judge_task_policies.py"
    text = path.read_text(encoding="utf-8")
    if "assert len(TASK_DIRS) >= 20" in text:
        path.write_text(text.replace("assert len(TASK_DIRS) >= 20", "assert TASK_DIRS"), encoding="utf-8")
        patched["test_llm_judge_task_policies.py"] = "inventory size assertion relaxed"

    path = out / "tests/sforge/harness/test_latest_task_store_integration.py"
    text = path.read_text(encoding="utf-8")
    # Mapping entries are `"<task-dir>": (\n "<task_id>"\n),`
    entry = re.compile(r'^\s*"([a-z0-9.\-]+)":\s*\(\s*\n\s*"[^"]+"\s*\n\s*\),?\s*\n', re.MULTILINE)
    removed = [m.group(1) for m in entry.finditer(text) if m.group(1) not in keep]
    path.write_text(entry.sub(lambda m: m.group(0) if m.group(1) in keep else "", text), encoding="utf-8")
    patched["test_latest_task_store_integration.py"] = f"removed {len(removed)} absent tasks"

    path = out / "tests/sforge/harness/test_resource_profile.py"
    text = path.read_text(encoding="utf-8")
    # The refuse case stands on a task a subset may lack. Keep the test, drop that parameter.
    text = text.replace('@pytest.mark.parametrize("task_id, expected", [(PAGED, "allow"), (QVG, "refuse")])',
                        '@pytest.mark.parametrize("task_id, expected", [(PAGED, "allow")])')
    lines = text.splitlines()
    out_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r'^QVG\s*=', line):
            i += 1
            continue
        if line.startswith("def ") and any("QVG" in body for body in lines[i:_block_end(lines, i)]):
            end = _block_end(lines, i)
            patched.setdefault("test_resource_profile.py", []).append(line.split("(")[0][4:])
            i = end
            continue
        out_lines.append(line)
        i += 1
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    path = out / "tests/sforge/harness/test_pyromind_inventory.py"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        # The fixture only needs an opaque deployment name.
        path.write_text(text.replace("'qvg'", "'deployment-a'"), encoding="utf-8")
        patched["test_pyromind_inventory.py"] = "fixture deployment renamed"

    path = out / "tests/test_infrathon_assets.py"
    text, dropped = keep_subset_safe_asserts(path.read_text(encoding="utf-8"))
    path.write_text(text, encoding="utf-8")
    patched["test_infrathon_assets.py"] = f"dropped {dropped} lines checking specific tasks' hardware"

    path = out / "tests/test_infrathon_image_record.py"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("assert len(TASKS) == 6", "assert TASKS"), encoding="utf-8")
    patched["test_infrathon_image_record.py"] = "inventory size assertion relaxed"

    (out / "conftest.py").write_text(SUBSET_CONFTEST, encoding="utf-8")
    report["skipped_tests"] = sorted(SUBSET_SKIPS) + sorted(SUBSET_IGNORES)


def make(repo: Path, commit: str, tasks_file: Path | None, out: Path) -> dict:
    """Export the tree into out (not yet committed) and return what was kept and changed."""
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    commit = git(repo, "rev-parse", "--short=12", commit).strip()
    all_tasks = [t.rsplit("/", 1)[-1] for t in git(repo, "ls-tree", "--name-only", commit, "tasks/latest/").split()]
    if tasks_file is None:
        keep = set(all_tasks)
        paths = [".gitignore", "LICENSE", "README.md", "docs", "pyproject.toml", "uv.lock",
                 "sforge", "tests", "scripts", "launch", "tasks"]
    else:
        spec = json.loads(tasks_file.read_text(encoding="utf-8"))
        task_map: dict[str, str] = spec["tasks"]
        for name in task_map:
            if name not in all_tasks:
                raise SystemExit(f"{name} is not a task at {commit}")
        keep = set(task_map)
        paths = COMMON_PATHS + list(spec.get("extra_paths", [])) + list(task_map.values()) \
            + [f"tasks/latest/{name}" for name in task_map]
    out.mkdir(parents=True)
    export(repo, commit, paths, out)
    report: dict = {"source_commit": commit, "tasks": sorted(keep), "absent": sorted(set(all_tasks) - keep),
                    "trimmed": {}, "patched_tests": {}, "skipped_tests": []}
    if keep != set(all_tasks):
        trim_tables(out, set(all_tasks) - keep, report)
        filter_registry(out, keep, report)
        patch_tests(out, keep, report)
    return report


def commit_tree(out: Path, message: str) -> str:
    identity = ["-c", "user.name=reference", "-c", "user.email=reference@localhost", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", "-b", "main", str(out)], check=True)
    subprocess.run(["git", "-C", str(out), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(out), *identity, "commit", "-q", "-m", message], check=True)
    return git(out, "rev-parse", "HEAD").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="an Infrathon git checkout")
    parser.add_argument("--commit", default="HEAD", help="committed state to export (default HEAD)")
    parser.add_argument("--tasks", default=str(ROOT / "reference/tasks.json"),
                        help="JSON with the example tasks and their launch directories, or 'all'")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="where to record what was kept and changed")
    args = parser.parse_args()
    report = make(args.source, args.commit, None if args.tasks == "all" else Path(args.tasks), args.out)
    report["commit"] = commit_tree(args.out, f"Infrathon at {report['source_commit']}: {len(report['tasks'])} tasks")
    manifest = args.manifest or args.out.parent / f"{args.out.name}.manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("source_commit", "tasks", "commit")}, indent=2))
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
