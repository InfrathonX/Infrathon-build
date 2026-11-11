"""The generated suites: frozen identities, the definition's sizes, and no public/hidden overlap.

Needs the suites built by scripts/build_suites.sh (KVZIP_SUITE_DIR or the asset root);
launch/kvzip-kv-scoring/check.sh runs these with them mounted.
"""

import hashlib
import json
from collections import Counter

import pytest

from conftest import TASK_ROOT, suite_dir
from kvzip_eval.suite import load_suite, suite_sha256

SUITES = suite_dir()
PUBLIC = SUITES / "public" / "public_suite.jsonl.gz" if SUITES else None
HIDDEN = SUITES / "judge" / "hidden_suite.jsonl.gz" if SUITES else None
pytestmark = pytest.mark.skipif(not (PUBLIC and PUBLIC.is_file() and HIDDEN.is_file()),
                                reason="suites not built (scripts/build_suites.sh)")
MANIFEST = json.loads((TASK_ROOT / "judge" / "suite_manifest.json").read_text())


@pytest.fixture(scope="module")
def public():
    return load_suite(PUBLIC)


@pytest.fixture(scope="module")
def hidden():
    return load_suite(HIDDEN)


def test_suites_match_their_frozen_identities():
    assert suite_sha256(PUBLIC) == MANIFEST["public"]["sha256"]
    assert suite_sha256(HIDDEN) == MANIFEST["hidden"]["sha256"]


def test_suite_sizes_follow_the_definition(public, hidden):
    assert Counter(r.workload for r in public) == {
        "squad": 10, "gsm": 10, "needle": 11, "scbench_kv": 8, "scbench_prefix_suffix": 8, "scbench_mf": 8,
        "ruler_niah": 4, "ruler_qa": 2, "ruler_vt": 2}
    assert sum(len(r.questions) for r in public) == 419
    assert Counter(r.workload for r in hidden) == {
        "needle": 96, "scbench_kv": 150, "scbench_prefix_suffix": 150, "scbench_repoqa": 22, "ruler_niah": 192,
        "squad": 250, "gsm": 200, "ruler_qa": 152, "scbench_mf": 250, "scbench_many_shot": 54,
        "scbench_summary": 62, "ruler_vt": 234}
    assert sum(len(r.questions) for r in hidden) == 6956


def _source_keys(row):
    keys = {("context", hashlib.sha256(row.context.encode()).hexdigest())}
    for q in row.questions:
        meta = q.meta or {}
        if row.workload.startswith("ruler"):
            keys.add(("ruler", meta["task"], meta["length"], meta["index"]))
        if row.workload == "squad":
            keys.add(("squad", meta["source_id"]))
        if row.workload == "gsm":
            keys.add(("gsm", row.id.split("/")[1], meta["source_index"]))
    return keys


def test_public_and_hidden_share_no_source_row_or_context(public, hidden):
    pub = set().union(*(_source_keys(r) for r in public))
    hid = set().union(*(_source_keys(r) for r in hidden))
    assert not pub & hid
    assert len({r.id for r in public} | {r.id for r in hidden}) == len(public) + len(hidden)


def test_hidden_needles_and_essays_stay_hidden(public, hidden):
    needles = json.loads((TASK_ROOT / "judge" / "private" / "needles.json").read_text())
    public_text = "\n".join(r.context + "".join(q.question for q in r.questions) for r in public)
    for n in needles:
        assert n["needle"].strip() not in public_text and n["question"] not in public_text
    hidden_needle_rows = [r for r in hidden if r.workload == "needle"]
    assert all(any(n["needle"].strip() in r.context for n in needles) for r in hidden_needle_rows)


def test_contexts_are_within_the_protocol_lengths(public, hidden):
    # character bounds that bracket 64..32,000 Qwen3 tokens (the builder checks tokens exactly)
    for r in public + hidden:
        assert 150 <= len(r.context) <= 32000 * 8, r.id
