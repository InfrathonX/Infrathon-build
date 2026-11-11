import math

from kvzip_eval.aggregate import aggregate, cost_check


def _row(w, full, r1, r2, r3):
    return {"workload": w, "acc": {"full": full, "0.1": r1, "0.2": r2, "0.3": r3}}


def test_score_is_mean_over_ratios_of_mean_over_groups_of_task_relative_accuracy():
    rows = [_row("squad", 1.0, 0.5, 0.5, 0.5), _row("squad", 0.5, 0.5, 0.5, 0.5),   # acc full .75, r .5
            _row("needle", 1.0, 0.0, 1.0, 1.0),
            _row("scbench_many_shot", 0.5, 1.0, 0.5, 0.5)]
    a = aggregate(rows)
    assert math.isclose(a["task_relative"]["squad"]["0.1"], 0.5 / 0.75)
    assert math.isclose(a["groups"]["redundancy"]["0.1"], 2.0)  # no cap
    r1 = (0.0 + 0.5 / 0.75 + 2.0) / 3
    r2 = (1.0 + 0.5 / 0.75 + 1.0) / 3
    assert math.isclose(a["ratio_scores"]["0.1"], 100 * r1)
    assert math.isclose(a["score"], 100 * (r1 + 2 * r2) / 3)


def test_zero_full_accuracy_is_flagged():
    a = aggregate([_row("squad", 0.0, 0.0, 0.0, 0.0), _row("needle", 1.0, 1, 1, 1)])
    assert a["zero_full_accuracy_tasks"] == ["squad"] and not a["finite"]
    b = aggregate([_row("squad", 0.0, 0.0, 0.0, 0.0), _row("needle", 1.0, 1, 1, 1)], drop_zero_full=True)
    assert b["finite"] and b["score"] == 100.0


def test_cost_rule():
    ok = cost_check([("a", 1.0, 0.15), ("b", 0.01, 0.09)])  # floor = 10 x 0.01 = 0.1 s
    assert ok["total_ok"] and ok["per_context_ok"]
    assert not cost_check([("a", 1.0, 0.6), ("b", 0.01, 0.0)])["per_context_ok"]   # over 50% of its prefill
    assert not cost_check([("a", 1.0, 0.0), ("b", 0.01, 0.11)])["per_context_ok"]  # over the floor
    assert not cost_check([("a", 1.0, 0.3), ("b", 1.0, 0.3)])["total_ok"]          # over 25% in total
