#!/usr/bin/env python3
"""Pack a submission archive, as sforge-submit does.

  make_archive.py TREE ARCHIVE         TREE holds scorers/, FINAL_VERSION, notes.md
  make_archive.py SCORER_DIR ARCHIVE   SCORER_DIR holds scorer.py: packed as scorers/scorer_v001/
                                       with FINAL_VERSION 1 (and its notes.md as notes.md)
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

SKIP_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints"}


def _files(base: Path):
    return [p for p in sorted(base.rglob("*")) if p.is_file() and not (SKIP_PARTS & set(p.parts)) and p.suffix != ".pyc"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("archive")
    args = ap.parse_args()
    root = Path(args.source)
    members: dict[str, bytes] = {}
    if (root / "FINAL_VERSION").is_file():
        for top in ("scorers", "FINAL_VERSION", "notes.md"):
            path = root / top
            for f in ([path] if path.is_file() else _files(path) if path.is_dir() else []):
                members[f.relative_to(root).as_posix()] = f.read_bytes()
    elif (root / "scorer.py").is_file():
        for f in _files(root):
            rel = f.relative_to(root).as_posix()
            if rel == "notes.md":
                members["notes.md"] = f.read_bytes()
            else:
                members[f"scorers/scorer_v001/{rel}"] = f.read_bytes()
        members["FINAL_VERSION"] = b"1\n"
        members.setdefault("notes.md", b"# Notes\n")
    else:
        raise SystemExit(f"{root} holds neither FINAL_VERSION nor scorer.py")
    with tarfile.open(args.archive, "w:gz") as tar:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    print(f"{args.archive}: {len(members)} files")


if __name__ == "__main__":
    main()
