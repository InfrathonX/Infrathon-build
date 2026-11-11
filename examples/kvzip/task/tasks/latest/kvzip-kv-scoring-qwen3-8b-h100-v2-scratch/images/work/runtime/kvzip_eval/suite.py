"""Suite files: one JSON object per row (gzip'd JSON lines).

{"id", "workload", "context", "questions": [{"question", "refs", "answer_prefix", "meta"}]}
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import protocol as P


@dataclass
class Question:
    question: str
    refs: list[str]
    answer_prefix: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Row:
    id: str
    workload: str
    context: str
    questions: list[Question]

    @property
    def group(self) -> str:
        return P.WORKLOADS[self.workload][0]

    @property
    def metric(self) -> str:
        return P.WORKLOADS[self.workload][1]

    @property
    def max_new_tokens(self) -> int:
        return P.WORKLOADS[self.workload][2]


def row_from_dict(d: dict) -> Row:
    if d["workload"] not in P.WORKLOADS:
        raise ValueError(f"unknown workload {d['workload']!r}")
    return Row(id=d["id"], workload=d["workload"], context=d["context"],
               questions=[Question(**q) for q in d["questions"]])


def row_to_dict(row: Row) -> dict:
    return {"id": row.id, "workload": row.workload, "context": row.context,
            "questions": [{"question": q.question, "refs": q.refs, "answer_prefix": q.answer_prefix,
                           "meta": q.meta} for q in row.questions]}


def load_suite(path: str | Path) -> list[Row]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [row_from_dict(json.loads(line)) for line in f if line.strip()]


def suite_bytes(rows: list[Row]) -> bytes:
    return "".join(json.dumps(row_to_dict(r), ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()


def write_suite(path: str | Path, rows: list[Row]) -> str:
    """gzip'd JSON lines; returns the sha256 of the uncompressed content (the suite identity)."""
    data = suite_bytes(rows)
    with open(Path(path), "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(data)
    return hashlib.sha256(data).hexdigest()


def suite_sha256(path: str | Path) -> str:
    """sha256 of a suite file's uncompressed content."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Tokenized:
    """Token ids of one row under the protocol's prompt layout."""

    def __init__(self, tokenizer, row: Row):
        enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
        self.prefix_ids = enc(P.SYSTEM_PREFIX + P.instruction_for(row.workload))
        self.context_ids = enc(row.context)
        postfix = enc(P.ASSISTANT_POSTFIX)
        self.prompts = []
        for q in row.questions:
            ids = enc(P.question_text(q.question)) + postfix
            if q.answer_prefix:
                ids = ids + enc(q.answer_prefix)
            self.prompts.append(ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prefix_ids) + len(self.context_ids)
