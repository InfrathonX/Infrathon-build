"""Score aggregation and the cost rule (definition section 4.2 / 4.3)."""

from __future__ import annotations

import math
from collections import defaultdict

from . import protocol as P

COST_TOTAL_FRACTION = 0.25
COST_CONTEXT_FRACTION = 0.5
COST_FLOOR_MULTIPLE = 10.0


def aggregate(rows: list[dict], drop_zero_full: bool = False) -> dict:
    """rows: [{"workload", "acc": {setting: mean over the row's questions}}].

    drop_zero_full (public diagnostics only): leave out tasks whose full-cache accuracy is 0
    instead of producing a non-finite score.  The Judge never drops a task.

    acc(task, s) = mean over rows; rel(task, r) = acc(task, r) / acc(task, full);
    group(g, r) = mean of rel over the tasks of g present; ratio(r) = mean over groups;
    score = 100 * mean over r.  No cap.
    """
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r.get("id", "")):  # canonical order: bit-identical sums
        by_task[r["workload"]].append(r["acc"])
    acc = {t: {s: sum(a[s] for a in v) / len(v) for s in P.SETTINGS} for t, v in by_task.items()}
    zero_full = sorted(t for t, a in acc.items() if a["full"] <= 0)
    rel = {t: {r: (a[r] / a["full"] if a["full"] > 0 else math.nan) for r in P.SETTINGS[1:]}
           for t, a in acc.items() if not (drop_zero_full and t in zero_full)}
    groups = {}
    for g in P.GROUPS:
        tasks = [t for t in rel if P.WORKLOADS[t][0] == g]
        if tasks:
            groups[g] = {r: sum(rel[t][r] for t in tasks) / len(tasks) for r in P.SETTINGS[1:]}
    ratio = {r: sum(groups[g][r] for g in groups) / len(groups) for r in P.SETTINGS[1:]} if groups else {}
    score = 100.0 * sum(ratio.values()) / len(ratio) if ratio else math.nan
    group_scores = {g: 100.0 * sum(v.values()) / len(v) for g, v in groups.items()}
    ratio_scores = {r: 100.0 * v for r, v in ratio.items()}
    finite = all(math.isfinite(x) for x in [score, *group_scores.values(), *ratio_scores.values()])
    return {"score": score, "ratio_scores": ratio_scores, "group_scores": group_scores,
            "groups": groups, "task_accuracy": acc, "task_relative": rel,
            "zero_full_accuracy_tasks": zero_full, "finite": finite}


def cost_check(timings: list[tuple[str, float, float]]) -> dict:
    """timings: (row id, plain prefill seconds, scoring seconds) for every scored context."""
    timings = sorted(timings)
    total_plain = sum(t[1] for t in timings)
    total_score = sum(t[2] for t in timings)
    shortest = min(t[1] for t in timings)
    floor = COST_FLOOR_MULTIPLE * shortest
    worst_margin, worst = -math.inf, None
    over = 0
    for rid, plain, sc in timings:
        limit = max(COST_CONTEXT_FRACTION * plain, floor)
        margin = sc / limit
        over += sc > limit
        if margin > worst_margin:
            worst_margin, worst = margin, rid
    total_ratio = total_score / total_plain
    return {"total_scoring_s": total_score, "total_prefill_s": total_plain,
            "total_ratio": total_ratio, "total_ok": total_ratio <= COST_TOTAL_FRACTION,
            "shortest_prefill_s": shortest, "per_context_floor_s": floor,
            "worst_context_ratio_to_limit": worst_margin, "contexts_over_limit": over,
            "per_context_ok": over == 0, "worst_context_id": worst}
