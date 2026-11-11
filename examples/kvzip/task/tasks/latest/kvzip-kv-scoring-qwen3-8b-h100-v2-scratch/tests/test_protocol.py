import itertools

import pytest
import torch

from kvzip_eval import protocol as P


def test_keep_count_is_round_r_times_n_and_never_a_tie():
    for n_ctx in range(1, 2000):
        n = 36 * 8 * n_ctx
        for key, tenths in P.RATIO_TENTHS.items():
            exact = tenths * n / 10
            assert abs(exact - round(exact)) != 0.5
            assert P.keep_count(key, n) == round(exact)


def test_retention_is_one_global_ranking_with_lower_index_tie_break():
    scores = torch.zeros(36, 8, 10)
    masks = P.retention_masks(scores)
    flat = masks["0.1"].reshape(-1)
    k = P.keep_count("0.1", flat.numel())
    assert flat[:k].all() and not flat[k:].any()  # all ties: the lowest flat indices win


def test_retention_follows_scores_and_sets_are_nested():
    g = torch.Generator().manual_seed(0)
    scores = torch.randn(36, 8, 77, generator=g)
    scores[3, 2, 5] = 100.0
    masks = P.retention_masks(scores)
    for key in ("0.1", "0.2", "0.3"):
        assert int(masks[key].sum()) == P.keep_count(key, scores.numel())
        kept, dropped = scores[masks[key]], scores[~masks[key]]
        assert kept.min() >= dropped.max()
    assert masks["0.1"][3, 2, 5]
    assert (masks["0.1"] <= masks["0.2"]).all() and (masks["0.2"] <= masks["0.3"]).all()


def test_ties_across_layers_prefer_lower_layer_then_head_then_position():
    scores = torch.zeros(36, 8, 4)
    scores[10] = 1.0
    scores[20] = 1.0  # same value as layer 10: layer 10 must be kept first
    masks = P.retention_masks(scores)
    k = P.keep_count("0.1", scores.numel())  # 115 entries of 1152; layer 10 holds 32
    assert masks["0.1"][10].all()
    assert int(masks["0.1"][20].sum()) == 32
    assert int(masks["0.1"][0].sum()) == 32 and int(masks["0.1"][1].sum()) == k - 96


def test_prompt_layout_matches_the_definition():
    assert P.SYSTEM_PREFIX == "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    assert P.instruction_for("gsm").startswith("Given the context, answer to the following reasoning question.")
    assert P.instruction_for("squad").startswith(
        "Given the context, answer to the following question or request without explanation.")
    assert P.question_text("  What?  ") == "\n\nWhat?"
    assert P.ASSISTANT_POSTFIX == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    from conftest import PUBLIC_WORKLOADS
    assert sorted(PUBLIC_WORKLOADS) == ["gsm", "needle", "ruler_niah", "ruler_qa", "ruler_vt", "scbench_kv",
                                        "scbench_mf", "scbench_prefix_suffix", "squad"]  # the Work runtime
    assert {w: v[2] for w, v in P.WORKLOADS.items()} == {  # with the Judge's registration
        "needle": 48, "scbench_kv": 96, "scbench_prefix_suffix": 128, "scbench_repoqa": 512, "ruler_niah": 64,
        "squad": 256, "gsm": 512, "ruler_qa": 32, "scbench_mf": 48, "scbench_many_shot": 96,
        "scbench_summary": 256, "ruler_vt": 48}
    groups = {g: sorted(w for w, v in P.WORKLOADS.items() if v[0] == g) for g in P.GROUPS}
    assert groups["retrieval"] == ["needle", "ruler_niah", "scbench_kv", "scbench_prefix_suffix", "scbench_repoqa"]
    assert groups["contextual_qa"] == ["gsm", "ruler_qa", "squad"]
    assert groups["redundancy"] == ["ruler_vt", "scbench_many_shot", "scbench_mf", "scbench_summary"]
