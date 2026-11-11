"""The submitted tree and its admission rules (the Judge applies the same rules).

    scorers/scorer_vNNN/   one directory per scorer version: scorer.py (class Scorer), its own
                           modules, weights and the code that trained them
    FINAL_VERSION          the integer NNN of the version that is evaluated
    notes.md               notes (never affects the score)

The selected version is evaluated as the package ``scorer`` (a copy of
``scorers/scorer_vNNN``), so ``from scorer import x`` and relative imports work.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
from pathlib import Path

TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv", ".tsv",
                 ".log", ".sh",
                 ".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp"}
WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".npy", ".npz"}
MAX_TEXT_BYTES = 1 << 20
SUBMITTED_TOP = {"scorers", "FINAL_VERSION", "notes.md"}
VERSION_RE = re.compile(r"scorer_v([0-9]{3})")


def version_dir(root: str | Path, version: int) -> Path:
    return Path(root) / "scorers" / f"scorer_v{version:03d}"


def selected_version(root: str | Path, override: int | None = None) -> tuple[int | None, list[str]]:
    """The evaluated version (override, else FINAL_VERSION) and any problem with it."""
    root = Path(root)
    if override is not None:
        version = int(override)
    else:
        selector = root / "FINAL_VERSION"
        if not selector.is_file() or selector.is_symlink():
            return None, ["FINAL_VERSION is missing"]
        text = selector.read_text(errors="replace").strip()
        if not re.fullmatch(r"[0-9]{1,3}", text) or int(text) < 1:
            return None, [f"FINAL_VERSION must hold one integer 1-999, not {text[:20]!r}"]
        version = int(text)
    entry = version_dir(root, version) / "scorer.py"
    if not entry.is_file() or entry.is_symlink():
        return version, [f"scorers/scorer_v{version:03d}/scorer.py is missing"]
    return version, []


def check_tree(root: str | Path, override: int | None = None) -> list[str]:
    """Problems with the submitted tree; empty when admissible."""
    root = Path(root)
    _, problems = selected_version(root, override)
    for top in sorted(os.listdir(root)) if root.is_dir() else []:
        if top not in SUBMITTED_TOP:
            continue
        base = root / top
        if top == "scorers" and base.is_dir() and not base.is_symlink():
            for child in sorted(os.listdir(base)):
                if not VERSION_RE.fullmatch(child):
                    problems.append(f"scorers/{child}: only scorer_vNNN directories belong in scorers/")
        walk = os.walk(base, followlinks=False) if base.is_dir() and not base.is_symlink() else \
            [(str(root), [], [top])]
        for dirpath, dirnames, filenames in walk:
            for name in dirnames + filenames:
                path = Path(dirpath) / name
                rel = path.relative_to(root)
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    problems.append(f"{rel}: symbolic links are not accepted")
                elif stat.S_ISDIR(mode):
                    continue
                elif not stat.S_ISREG(mode):
                    problems.append(f"{rel}: not a regular file")
                elif "__pycache__" in rel.parts or path.suffix == ".pyc" or rel.as_posix() == "FINAL_VERSION":
                    continue
                else:
                    suffix = path.suffix.lower()
                    if suffix in TEXT_SUFFIXES:
                        if path.stat().st_size > MAX_TEXT_BYTES:
                            problems.append(f"{rel}: text and source files are limited to 1 MiB")
                    elif suffix not in WEIGHT_SUFFIXES:
                        problems.append(f"{rel}: file type {suffix or '(none)'} is not accepted")
    return problems[:20]


def prepare_tree(root: str | Path, dest: str | Path, version: int, copy: bool = True) -> Path:
    """Lay out the evaluated tree: <dest>/scorer = scorers/scorer_vNNN (copied or linked), notes.md."""
    root, dest = Path(root), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    src = version_dir(root, version)
    if copy:
        shutil.copytree(src, dest / "scorer", symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        os.symlink(src.resolve(), dest / "scorer")
    if (root / "notes.md").is_file():
        shutil.copy2(root / "notes.md", dest / "notes.md")
    return dest
