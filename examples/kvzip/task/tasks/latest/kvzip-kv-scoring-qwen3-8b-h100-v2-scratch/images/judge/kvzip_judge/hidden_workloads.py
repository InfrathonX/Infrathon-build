"""The hidden suite's workloads beyond the public ones, and their metrics (Judge only).

The Work image's runtime knows only the public workloads; naming the others there would tell
the agent which task types the hidden suite holds. register() adds them to the runtime's
protocol and metrics; run_judge imports this module before it reads the hidden suite.

Metrics adapted from KVzip (snu-mllab/KVzip results/metric.py, results/repo_qa_utils.py, after
evalplus/repoqa; MIT).
"""

from __future__ import annotations

import re

from kvzip_eval import metrics as M
from kvzip_eval import protocol as P

# workload -> (group, metric, max new tokens)
WORKLOADS = {
    "scbench_repoqa": ("retrieval", "repoqa_bleu", 512),
    "scbench_many_shot": ("redundancy", "option_letter", 96),
    "scbench_summary": ("redundancy", "rouge_l", 256),
}


def option_letter(pred: str, refs: list[str]) -> float:
    """KVzip include_score_manyshot; refs[0] is the full option line, e.g. '(D) Named Entities'."""
    ref = refs[0]
    if "(" in pred and "(" in ref:
        p = pred.split("(")[1].split(")")[0]
        r = ref.split("(")[1].split(")")[0]
        return float(p == r)
    if ref[0] == "(":
        ref = ref.split(")")[1].strip()
    return float(M.normalize_answer(ref) in M.normalize_answer(pred))


def rouge_l(pred: str, refs: list[str]) -> float:
    from rouge import Rouge

    try:
        scores = Rouge().get_scores([pred], [refs[0]], avg=True)
    except Exception:  # upstream behaviour: unscorable text counts 0
        return 0.0
    return float(scores["rouge-l"]["f"])


_FUNCTION_QUERY = {
    "python": "(function_definition name: (_)) @fdef",
    "java": "(method_declaration name: (_)) @fdef",
    "typescript": "(function_declaration name: (_)) @fdef",
    "rust": "(function_item name: (_)) @fdef",
    "cpp": "(function_definition declarator: (function_declarator declarator: (identifier))) @fdef",
    "go": "(function_declaration name: (_)) @fdef",
}


def _sanitize_output(model_output: str, lang: str) -> str:
    model_output = model_output.strip()
    pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
    blocks = re.findall(pattern, model_output, re.DOTALL | re.MULTILINE)
    if not blocks:
        return model_output
    # tree-sitter is part of the protocol (the Judge image installs it); without it the metric
    # would silently change, so a missing parser is an evaluation error, not a fallback.
    from tree_sitter_languages import get_language, get_parser

    parser = get_parser(lang)
    query = get_language(lang).query(_FUNCTION_QUERY[lang])
    for block in blocks:
        try:
            data = bytes(block, "utf8")
            tree = parser.parse(data)
            for node, _ in query.captures(tree.root_node):
                return data[node.start_byte:node.end_byte].decode("utf8")
        except Exception:
            pass
    return blocks[0]


def function_similarity(candidate: str, reference: str) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    cand = re.split(r"\s+", candidate.strip())
    ref = re.split(r"\s+", reference.strip())
    return float(sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method4))


def repoqa_bleu(pred: str, refs: list[str], lang: str) -> float:
    """Pass if BLEU >= 0.8 against the reference function; an empty answer fails."""
    if not pred.strip():
        return 0.0
    return float(function_similarity(_sanitize_output(pred, lang), refs[0]) >= 0.8)


METRICS = {
    "option_letter": lambda pred, refs, meta: option_letter(pred, refs),
    "rouge_l": lambda pred, refs, meta: rouge_l(pred, refs),
    "repoqa_bleu": lambda pred, refs, meta: repoqa_bleu(pred, refs, meta["lang"]),
}


def register() -> None:
    P.register_workloads(WORKLOADS)
    M.EXTRA_METRICS.update(METRICS)
