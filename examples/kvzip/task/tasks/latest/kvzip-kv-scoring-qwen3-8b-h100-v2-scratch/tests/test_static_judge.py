"""Judge-side static logic: admission, released-gate detection, bounded feedback, layout."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import TASK_ROOT
from kvzip_eval.layout import check_tree, prepare_tree, selected_version
from kvzip_judge.run_judge import CHECKS, build_result
from kvzip_judge.weights_check import UnreadableWeights, load_released, scan


def test_layout_validator_passes_and_work_context_has_no_judge_material():
    out = subprocess.run([sys.executable, str(TASK_ROOT / "scripts" / "validate_layout.py"), str(TASK_ROOT)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_admission_accepts_the_starting_workspace_and_rejects_bad_trees(tmp_path):
    workspace = TASK_ROOT / "images" / "work" / "workdir" / "kvzip_lab"
    assert check_tree(workspace) == [] and selected_version(workspace) == (1, [])
    assert check_tree(tmp_path) == ["FINAL_VERSION is missing"]
    (tmp_path / "FINAL_VERSION").write_text("2\n")
    assert check_tree(tmp_path) == ["scorers/scorer_v002/scorer.py is missing"]
    v2 = tmp_path / "scorers" / "scorer_v002"
    v2.mkdir(parents=True)
    (v2 / "scorer.py").write_text("class Scorer: pass\n")
    assert check_tree(tmp_path) == [] and selected_version(tmp_path, 7)[1] == ["scorers/scorer_v007/scorer.py is missing"]
    (v2 / "blob.so").write_bytes(b"\x7fELF")
    (v2 / "big.py").write_text("x = 1\n" * 300000)
    os.symlink("/etc/passwd", v2 / "link.txt")
    (tmp_path / "scorers" / "stray").mkdir()
    problems = " ".join(check_tree(tmp_path))
    assert "blob.so" in problems and "big.py" in problems and "link.txt" in problems and "stray" in problems
    (tmp_path / "FINAL_VERSION").write_text("two")
    assert "FINAL_VERSION must hold one integer" in " ".join(check_tree(tmp_path))


def test_the_selected_version_is_evaluated_as_package_scorer(tmp_path):
    for v in (1, 3):
        d = tmp_path / "src" / "scorers" / f"scorer_v{v:03d}"
        d.mkdir(parents=True)
        (d / "scorer.py").write_text(f"VERSION = {v}\n")
    (tmp_path / "src" / "notes.md").write_text("n\n")
    tree = prepare_tree(tmp_path / "src", tmp_path / "tree", 3)
    assert (tree / "scorer" / "scorer.py").read_text() == "VERSION = 3\n" and (tree / "notes.md").is_file()


def _gate_file():
    root = os.environ.get("KVZIP_GATES_DIR", "/gates")
    path = Path(root) / "qwen3-8b" / "q4_dim16_sink16.pt"
    return path if path.is_file() else None


@pytest.mark.skipif(_gate_file() is None, reason="released gate not available (assets/fetch.sh)")
def test_weights_check_detects_disguised_released_gates_and_passes_independent_weights(tmp_path):
    from safetensors.torch import save_file

    released = load_released(_gate_file())
    q = released[:2048].clone()  # q_proj rows of layers 0-3 (bf16 values as float)

    def case(name, writer):
        d = tmp_path / name / "scorer"
        d.mkdir(parents=True)
        writer(d)
        return scan(tmp_path / name, released)

    s = q.abs().amax(1, keepdim=True) / 127
    groups = q.reshape(-1, 64)
    s4 = groups.abs().amax(1, keepdim=True) / 7
    nibbles = (torch.round(groups / s4).clamp(-8, 7).to(torch.int8).reshape(-1, 2) & 0xF).to(torch.uint8)
    packed = (nibbles[:, 0] | (nibbles[:, 1] << 4)).to(torch.uint8)          # int4, two's complement
    padded = torch.cat([q, 0.01 * torch.randn(q.shape[0], 2)], 1)           # 4098 inputs, as a gate with extras
    poisoned = q.clone()
    poisoned[:, 0] = float("nan")
    flagged = {
        "renamed_fp16": case("a", lambda d: save_file({"w": q.half()}, str(d / "w.safetensors"))),
        "transposed_fp32": case("b", lambda d: torch.save({"w": q.T.contiguous()}, d / "w.pt")),
        "reshaped": case("c", lambda d: np.save(d / "w.npy", q.reshape(32, 64, 4096).numpy())),
        "int8_rowscaled": case("d", lambda d: save_file({"q": torch.round(q / s).to(torch.int8), "s": s},
                                                         str(d / "w.safetensors"))),
        "noisy_finetune": case("e", lambda d: save_file({"w": (q + 0.3 * q.std() * torch.randn_like(q)).bfloat16()},
                                                         str(d / "w.safetensors"))),
        "int4_packed": case("h", lambda d: save_file({"p": packed, "s": s4}, str(d / "w.safetensors"))),
        "padded_4098": case("i", lambda d: torch.save({"fc1.weight": padded.bfloat16()}, d / "g.pt")),
        "nan_poisoned": case("j", lambda d: save_file({"w": poisoned}, str(d / "w.safetensors"))),
    }
    for name, result in flagged.items():
        assert result["uses_released_gates"], (name, result)
    clean = case("f", lambda d: save_file({"w": torch.randn(2048, 4096) * 0.01}, str(d / "w.safetensors")))
    assert not clean["uses_released_gates"] and clean["max_row_correlation"] < 0.3
    with pytest.raises(UnreadableWeights):
        case("g", lambda d: (d / "w.pt").write_bytes(b"not a checkpoint"))


@pytest.mark.skipif(_gate_file() is None, reason="released gate not available (assets/fetch.sh)")
def test_weights_check_ignores_vectors_that_only_share_massive_activation_coordinates(tmp_path):
    """Trial regression. Qwen3's mean attention input (layers 8, 17, 26, over this task's brief,
    tests/fixtures) puts most of its energy on a few massive-activation coordinates, and so do
    many released rows: Pearson correlation exceeds 0.9 on 84 released rows although nothing
    was copied, and the Pearson-based detector rejected an honest submission that stored such
    means. Rank correlation must not fire on them."""
    released = load_released(_gate_file())
    fixture = json.loads((TASK_ROOT / "tests" / "fixtures" / "qwen3_mean_attention_input.json").read_text())
    means = torch.tensor(fixture["values"], dtype=torch.float32)
    unit = lambda x: torch.nn.functional.normalize(x - x.mean(1, keepdim=True), dim=1)  # noqa: E731
    assert int(((unit(means) @ unit(released).T).abs() >= 0.8).any(0).sum()) >= 64  # the old rule fired
    d = tmp_path / "stats" / "scorer"
    d.mkdir(parents=True)
    torch.save({"feature_mean": means.flatten(), "base_mean": means.flatten().clone()}, d / "adaptive.pt")
    result = scan(tmp_path / "stats", released)
    assert not result["uses_released_gates"] and result["max_row_correlation"] < 0.5, result


def test_agent_visible_result_is_bounded():
    agg = {"score": 50.0, "ratio_scores": {"0.1": 40.0, "0.2": 50.0, "0.3": 60.0},
           "group_scores": {"retrieval": 1.0, "contextual_qa": 2.0, "redundancy": 3.0},
           "task_accuracy": {"squad": {"full": 1.0}}, "task_relative": {"squad": {}}, "groups": {}}
    cost = {"total_ratio": 0.1, "worst_context_ratio_to_limit": 0.5, "contexts_over_limit": 0,
            "worst_context_id": "squad/train/0001"}
    result = build_result(True, None, {c: "pass" for c in CHECKS}, agg, cost)
    text = json.dumps(result)
    assert "squad" not in text and "task_accuracy" not in text and "worst_context_id" not in text
    assert set(result["metrics"]) == {"evaluation_status", "rejection_reason", "checks",
                                      "ratio_scores", "group_scores", "overhead", "score",
                                      "primary_score_name", "primary_score_unit", "primary_score_formula"}
    assert all(d["status"] == "PASSED" for d in result["details"])
    rejected_late = build_result(False, "reason", {"memory": "fail"}, agg, cost)
    assert "ratio_scores" not in rejected_late["metrics"] and "group_scores" not in rejected_late["metrics"]
    assert rejected_late["score"] == 0.0 and "overhead" in rejected_late["metrics"]
    invalid = build_result(False, "reason", {"admission": "fail"}, None, None)
    assert invalid["score"] == 0.0 and invalid["valid"] is False
    assert invalid["details"] == [{"name": "admission", "status": "FAILED", "message": None}]


def test_memory_limits_match_the_brief():
    """Check 9: the allocation rule, PyTorch's cache cap and the whole-process limit are separate,
    and each rejection names its own limit (trial 2 rejected three honest scorers for their
    PyTorch cache under a message about the allocation peak)."""
    from kvzip_eval import api
    from kvzip_judge import run_judge as J

    assert (api.MEMORY_LIMIT_BYTES, api.CACHE_LIMIT_BYTES, api.PROCESS_MEMORY_LIMIT_BYTES) == (8 << 30, 16 << 30, 19 << 30)
    assert J.MEMORY_LIMIT == api.MEMORY_LIMIT_BYTES and J.PROCESS_MEMORY_LIMIT == api.PROCESS_MEMORY_LIMIT_BYTES
    assert "max_memory_allocated" in J.MEMORY_MESSAGE and "8 GiB" in J.MEMORY_MESSAGE
    assert "19 GiB" in J.PROCESS_MEMORY_MESSAGE
    brief = (TASK_ROOT / "images" / "work" / "AGENT_TASK.md").read_text()
    for text in ("at most 8 GiB", "16 GiB", "19 GiB"):
        assert text in brief
