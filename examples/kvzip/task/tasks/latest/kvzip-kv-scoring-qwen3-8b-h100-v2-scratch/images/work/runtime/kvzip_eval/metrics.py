"""Per-question metrics of the public workloads.

Adapted from KVzip (snu-mllab/KVzip results/metric.py; MIT) and kvpress (NVIDIA/kvpress
RULER string match; Apache-2.0).  The Judge registers the metrics of the hidden suite's
other workloads in EXTRA_METRICS.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation

_WORD_TO_NUMBER = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                   "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_NUM_WORD_RE = re.compile(r"\b(" + "|".join(_WORD_TO_NUMBER) + r")\b")
_PUNCT = set(string.punctuation)
_CONTROL_RE = re.compile(r"[\x00-\x1f]")


def normalize_answer(s: str) -> str:
    """KVzip / SQuAD normalisation: lower, drop punctuation and articles, squash spaces, words->digits."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return _NUM_WORD_RE.sub(lambda m: _WORD_TO_NUMBER[m.group()], s)


def include_normalized(pred: str, refs: list[str]) -> float:
    p = normalize_answer(pred)
    return float(any(normalize_answer(r) in p for r in refs))


def exact_match(pred: str, refs: list[str]) -> float:
    return float(any(pred.strip() == r.strip() for r in refs))


def _ruler_clean(pred: str) -> str:
    return _CONTROL_RE.sub("", pred.strip()).strip()


def ruler_all(pred: str, refs: list[str]) -> float:
    p = _ruler_clean(pred).lower()
    return sum(1.0 for r in refs if r.lower() in p) / len(refs)


def ruler_any(pred: str, refs: list[str]) -> float:
    p = _ruler_clean(pred).lower()
    return max(1.0 if r.lower() in p else 0.0 for r in refs)


_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_decimal(text: str) -> Decimal | None:
    text = text.replace(",", "").strip()
    if text.endswith("."):
        text = text[:-1]
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def numeric_inclusion(pred: str, refs: list[str]) -> float:
    """GSM8K: the reference's final number (after '####') appears as a number in the answer."""
    targets = []
    for r in refs:
        value = _to_decimal(r.strip().split("####")[-1])
        if value is not None:
            targets.append(value)
    if not targets:
        return 0.0
    found = {_to_decimal(m.group()) for m in _NUMBER_RE.finditer(pred)}
    found.discard(None)
    return float(any(t in found for t in targets))


# name -> fn(pred, refs, meta): metrics registered by the Judge for its other workloads
EXTRA_METRICS: dict = {}


def score_answer(metric: str, pred: str, refs: list[str], meta: dict | None = None) -> float:
    if pred.endswith("</s>"):
        pred = pred[:-4]
    if not pred.strip():
        return 0.0
    if metric == "include_normalized":
        return include_normalized(pred, refs)
    if metric == "exact_match":
        return exact_match(pred, refs)
    if metric == "ruler_all":
        return ruler_all(pred, refs)
    if metric == "ruler_any":
        return ruler_any(pred, refs)
    if metric == "numeric_inclusion":
        return numeric_inclusion(pred, refs)
    if metric in EXTRA_METRICS:
        return EXTRA_METRICS[metric](pred, refs, meta or {})
    raise ValueError(f"unknown metric {metric}")
