#!/usr/bin/env python3
"""Build the public and hidden suites from pinned sources, deterministically.

Host-side, Judge-only tool (never copied into an image).  Run it through
scripts/build_suites.sh, which mounts the pinned inputs:

  --tokenizer   Qwen/Qwen3-8B snapshot (revision b968826d9c46dd6066d109eabc6255188de91218)
  --hf-cache    Hugging Face hub cache holding the dataset snapshots below
  --essays      KVzip data/needle/PaulGrahamEssays at commit 5d8472975d286e3da6b52178af39d18904330c0d
  --needles     judge/private/needles.json (the three hidden needles)

The rules are the task definition's (README "Public / Private Test"); seeds are
fixed per suite and differ between public and hidden, so a rebuild reproduces
the same bytes, which scripts/build_suites.sh checks against judge/suite_manifest.json.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
import uuid
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "images" / "work" / "runtime"))
from kvzip_eval.suite import Question, Row, write_suite  # noqa: E402

SQUAD = ("rajpurkar/squad", "dataset", "7b6d24c440a36b6815f21b70d25016731768db1f")
GSM8K = ("openai/gsm8k", "dataset", "740312add88f781978c0658806c59bc2815b9866")
SCBENCH = ("Jang-Hyun/SCBench-preprocessed", "dataset", "a079be919d3131822c202180ffd4dc322968de45")
RULER = ("simonjegou/ruler", "dataset", "24adceac8a0e6532936e8d721cd9e9084d2e4686")
MAX_CTX, MIN_CTX = 32000, 64

UPSTREAM_NEEDLE = {
    "needle": "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
    "question": "Based on the content of the book, the best thing to do in San Francisco is: ",
    "answer": "Eat a sandwich and sit in Dolores Park on a sunny day.",
}
PUBLIC_NEEDLE_GRID = [(500, 0), (500, 50), (500, 100), (2000, 10), (2000, 40), (2000, 70), (2000, 90),
                      (8000, 20), (8000, 50), (8000, 80), (8000, 100)]
NIAH_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
              "niah_multikey_3", "niah_multivalue", "niah_multiquery"]

KV_KEY_Q = '\nKey: "{}"\nThe value associated with the specified key is: '
KV_VAL_Q = '\nValue: "{}\nThe key associated with the specified value is: '
PS_INSTRUCTION = ("Given a list of words called 'dictionary', return the word in the dictionary that has both "
                  "the prefix 'pref' and the suffix 'suff'. If there is more than one valid answer, return the "
                  "latest of them. \n\nFor example:\ndictionary = ")
PS_Q = 'Given prefix = "{}"\nsuffix = "{}"\n\nWhat is the word in the dictionary that has both the prefix and the suffix?\n'
PS_ALPHABET = "+-ABCDEFGHIJKLMNOPQRSTUVWXYZ^_abcdefghijklmnopqrstuvwxyz"
MF_Q = {
    "max": "What is the largest number of the list?\n\nYou should answer with only one number, no other words. The largest number of the list is: ",
    "min": "What is the smallest number of the list?\n\nYou should answer with only one number, no other words. The smallest number of the list is: ",
    "max7": "What is the largest number with 7 in it?\n\nYou should answer with only one number, no other words. The largest number with 7 in it is: ",
    "max3": "What is the largest number with 3 at the end?\n\nYou should answer with only one number, no other words. The largest number with 3 at the end is: ",
    "min7": "What is the smallest number with 7 in it?\n\nYou should answer with only one number, no other words. The smallest number with 7 in it is: ",
    "min3": "What is the smallest number with 3 at the end?\n\nYou should answer with only one number, no other words. The smallest number with 3 at the end is: ",
}


class Builder:
    def __init__(self, args):
        self.tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        self.hf = Path(args.hf_cache)
        self.essays = Path(args.essays)
        self.needles = json.loads(Path(args.needles).read_text())

    def enc(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def ntok(self, s: str) -> int:
        return len(self.enc(s))

    def snapshot(self, spec) -> Path:
        repo, kind, rev = spec
        path = self.hf / f"{kind}s--{repo.replace('/', '--')}" / "snapshots" / rev
        if not path.is_dir():
            raise SystemExit(f"missing pinned snapshot {repo}@{rev} under {self.hf}")
        return path

    # -- natural-text workloads ---------------------------------------------------------------
    def squad(self, split: str, n_rows: int) -> list[Row]:
        df = pd.read_parquet(self.snapshot(SQUAD) / "plain_text" / f"{split}-00000-of-00001.parquet")
        order, qas = [], {}
        for r in df.itertuples():
            if r.context not in qas:
                qas[r.context] = []
                order.append(r.context)
            qas[r.context].append((r.id, r.question, list(dict.fromkeys(r.answers["text"]))))
        rows = []
        for ctx in order:
            if self.ntok(ctx) < MIN_CTX:
                continue
            qs = [Question(question=q, refs=refs, meta={"source_id": sid}) for sid, q, refs in qas[ctx]]
            rows.append(Row(id=f"squad/{split}/{len(rows):04d}", workload="squad", context=ctx, questions=qs))
            if len(rows) == n_rows:
                break
        return rows

    def gsm_eligible(self, split: str) -> list[tuple[int, str, str, str]]:
        df = pd.read_parquet(self.snapshot(GSM8K) / "main" / f"{split}-00000-of-00001.parquet")
        out = []
        for i, r in enumerate(df.itertuples()):
            st = r.question.split(". ")
            ctx = ". ".join(st[:-1]).strip() + "."
            if self.ntok(ctx) >= 72:
                out.append((i, ctx, st[-1].strip(), r.answer))
        return out

    def gsm_rows(self, items, split) -> list[Row]:
        return [Row(id=f"gsm/{split}/{i:05d}", workload="gsm", context=ctx,
                    questions=[Question(question=q, refs=[a], meta={"source_index": i})])
                for i, ctx, q, a in items]

    # -- needle in a haystack (KVzip data/needle generator, final_context_length_buffer=0) -------
    def haystack(self, names: list[str], length: int) -> str:
        context = ""
        while self.ntok(context) < length:
            for name in names:
                context += (self.essays / name).read_text(encoding="utf-8")
        return context

    def needle_context(self, haystack: str, needle: str, length: int, depth: int) -> tuple[str, int]:
        tokens = self.enc(haystack)
        if len(tokens) > length:
            haystack = self.tok.decode(tokens[:length])
        tokens_needle = self.enc(needle)
        tokens_context = self.enc(haystack)
        if len(tokens_context) + len(tokens_needle) > length:
            tokens_context = tokens_context[:length - len(tokens_needle)]
        if depth == 100:
            new, point = tokens_context + tokens_needle, len(tokens_context)
        else:
            point = int(len(tokens_context) * (depth / 100))
            new = tokens_context[:point]
            period_tokens = self.enc(".") + self.enc(".\n")
            while new and new[-1] not in period_tokens:
                point -= 1
                new = tokens_context[:point]
            new = new + tokens_needle + tokens_context[point:]
        return self.tok.decode(new).strip(), point

    def essay_split(self) -> tuple[list[str], list[str]]:
        names = sorted(p.name for p in self.essays.glob("*.txt"))
        if len(names) != 49:
            raise SystemExit(f"expected the 49 KVzip essays, found {len(names)}")
        even = [n for n in names if int(hashlib.sha256(n.encode()).hexdigest(), 16) % 2 == 0]
        odd = [n for n in names if n not in even]
        return even, odd

    def needle_rows(self, essays, needles, grid, tag) -> list[Row]:
        rows, seen = [], set()
        for ni, nd in enumerate(needles):
            for length in sorted({g[0] for g in grid}):
                hay = self.haystack(essays, length)
                for (gl, depth) in grid:
                    if gl != length:
                        continue
                    ctx, point = self.needle_context(hay, nd["needle"], length, depth)
                    key = (ni, length, point)
                    if key in seen:  # lands on the same sentence boundary as an earlier depth
                        continue
                    seen.add(key)
                    rows.append(Row(id=f"needle/{tag}/n{ni}-{length}-d{depth:03d}", workload="needle", context=ctx,
                                    questions=[Question(question=nd["question"], refs=[nd["answer"]],
                                                        meta={"length": length, "depth": depth})]))
        return rows

    # -- SCBench-format generators (follow the upstream *_short files) --------------------------
    @staticmethod
    def kv_row(rng: random.Random, rid: str, j: int, m: int) -> Row:
        n = rng.randint(361, 365)
        pairs = {}
        while len(pairs) < n:
            k = str(uuid.UUID(int=rng.getrandbits(128), version=4))
            pairs[k] = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        keys = list(pairs)
        start = min(n - 5, int((j + rng.random()) * (n - 4) / m))
        picked = list(range(start, start + 5))
        rng.shuffle(picked)
        qs = [Question(question=KV_KEY_Q.format(keys[i]), refs=[pairs[keys[i]]], meta={"position": i}) for i in picked]
        qs += [Question(question=KV_VAL_Q.format(pairs[keys[i]]), refs=[keys[i]], meta={"position": i}) for i in picked]
        return Row(id=rid, workload="scbench_kv", context="JSON data:\n" + json.dumps(pairs), questions=qs)

    @staticmethod
    def ps_row(rng: random.Random, rid: str) -> Row:
        while True:
            n = rng.randint(1004, 1010)
            words = ["".join(rng.choice(PS_ALPHABET) for _ in range(25)) for _ in range(n)]
            picks = rng.sample(range(n), 5)
            ok = True
            for i in picks:
                match = [k for k, w in enumerate(words) if w.startswith(words[i][:5]) and w.endswith(words[i][-5:])]
                ok &= match[-1] == i
            if ok:
                break
        qs = [Question(question=PS_Q.format(words[i][:5], words[i][-5:]), refs=[words[i]], meta={"position": i})
              for i in picks]
        return Row(id=rid, workload="scbench_prefix_suffix", context=PS_INSTRUCTION + repr(words) + "\n", questions=qs)

    @staticmethod
    def mf_row(rng: random.Random, rid: str) -> Row:
        n = rng.randint(6002, 6008)
        lo, hi = rng.randint(1, 99), rng.randint(100, 999)
        lst = [rng.randint(lo, hi) for _ in range(n)]
        with7 = [x for x in lst if "7" in str(x)]
        end3 = [x for x in lst if str(x).endswith("3")]
        answers = {"max": max(lst), "min": min(lst), "max7": max(with7), "max3": max(end3),
                   "min7": min(with7), "min3": min(end3)}
        kinds = list(MF_Q)
        rng.shuffle(kinds)
        qs = [Question(question=MF_Q[k], refs=[str(answers[k])], meta={"kind": k}) for k in kinds]
        return Row(id=rid, workload="scbench_mf", context=str(lst), questions=qs)

    def generated(self, workload: str, seed: str, count: int, tag: str) -> list[Row]:
        rng = random.Random(f"kvzip-scoring-v2/{seed}/{workload}")
        rows = []
        for j in range(count):
            rid = f"{workload}/{tag}/gen{j:03d}"
            if workload == "scbench_kv":
                rows.append(self.kv_row(rng, rid, j, count))
            elif workload == "scbench_prefix_suffix":
                rows.append(self.ps_row(rng, rid))
            else:
                rows.append(self.mf_row(rng, rid))
        return rows

    def scbench_upstream(self, name: str, workload: str) -> list[Row]:
        df = pd.read_parquet(self.snapshot(SCBENCH) / f"{name}.parquet")
        rows = []
        for i, r in enumerate(df.itertuples()):
            prompts = list(r.prompts)
            qs = []
            for qi, (q, gt) in enumerate(zip(prompts[1:], list(r.ground_truth))):
                gt = ", ".join(map(str, gt)) if isinstance(gt, (list, tuple)) else str(gt)
                meta = {}
                refs = [gt]
                if workload == "scbench_many_shot":
                    cand = [line.strip() for line in q.split("\n") if f"({gt})" in line]
                    if len(cand) != 1:
                        raise SystemExit(f"{name} row {i}: cannot parse option ({gt})")
                    refs = [cand[0]]
                    meta = {"letter": gt}
                if workload == "scbench_repoqa":
                    meta = {"lang": r.lang, "repo": r.repo, "func_name": str(list(r.func_name)[qi])}
                qs.append(Question(question=q, refs=refs, meta=meta))
            rows.append(Row(id=f"{workload}/upstream/{i:03d}", workload=workload, context=prompts[0], questions=qs))
        return rows

    # -- RULER -------------------------------------------------------------------------------------
    def ruler_frame(self, length: str) -> pd.DataFrame:
        return pd.read_parquet(self.snapshot(RULER) / length / "test-00000-of-00001.parquet")

    def ruler_rows(self, workload: str, picks: dict, tag: str) -> list[Row]:
        rows = []
        for (task, length), idxs in picks.items():
            df = self.ruler_frame(length)
            sub = df[df.task == task]
            for i in idxs:
                r = sub.iloc[i]
                rows.append(Row(id=f"{workload}/{tag}/{task}-{length}-{i:03d}", workload=workload, context=r.context,
                                questions=[Question(question=r.question, refs=[str(a) for a in r.answer],
                                                    answer_prefix=r.answer_prefix,
                                                    meta={"task": task, "length": int(length), "index": int(i)})]))
        return rows

    @staticmethod
    def ruler_draw(seed: str, cells: list[tuple[str, str]], k: int, exclude: dict) -> dict:
        out = {}
        for task, length in cells:
            rng = random.Random(f"kvzip-scoring-v2/{seed}/ruler/{task}/{length}")
            pool = [i for i in range(500) if i not in exclude.get((task, length), set())]
            out[(task, length)] = sorted(rng.sample(pool, k))
        return out

    # -- suites ------------------------------------------------------------------------------------
    def public(self) -> list[Row]:
        pub_essays, _ = self.essay_split()
        rows = self.squad("validation", 10)
        rows += self.gsm_rows(self.gsm_eligible("train")[:10], "train")
        rows += self.needle_rows(pub_essays, [UPSTREAM_NEEDLE], PUBLIC_NEEDLE_GRID, "public")
        for w in ("scbench_kv", "scbench_prefix_suffix", "scbench_mf"):
            rows += self.generated(w, "public", 8, "public")
        niah = self.ruler_draw("public", [("niah_single_1", "8192"), ("niah_multikey_2", "8192"),
                                          ("niah_multivalue", "8192"), ("niah_multiquery", "16384")], 1, {})
        qa = self.ruler_draw("public", [("qa_1", "8192"), ("qa_2", "16384")], 1, {})
        vt = self.ruler_draw("public", [("vt", "8192"), ("vt", "16384")], 1, {})
        rows += self.ruler_rows("ruler_niah", niah, "public")
        rows += self.ruler_rows("ruler_qa", qa, "public")
        rows += self.ruler_rows("ruler_vt", vt, "public")
        self._public_ruler = {**niah, **qa, **vt}
        return rows

    def hidden(self) -> list[Row]:
        _, hid_essays = self.essay_split()
        exclude = {k: set(v) for k, v in self._public_ruler.items()}
        rows = self.needle_rows(hid_essays, self.needles,
                                [(L, d) for L in (500, 2000, 8000) for d in range(0, 101, 10)], "hidden")
        rows += self.scbench_upstream("scbench_kv_short", "scbench_kv") + self.generated("scbench_kv", "hidden", 50, "hidden")
        rows += (self.scbench_upstream("scbench_prefix_suffix_short", "scbench_prefix_suffix")
                 + self.generated("scbench_prefix_suffix", "hidden", 50, "hidden"))
        rows += self.scbench_upstream("scbench_repoqa_short", "scbench_repoqa")
        cells = [(t, L) for t in NIAH_TASKS for L in ("8192", "16384")]
        rows += self.ruler_rows("ruler_niah", self.ruler_draw("hidden", cells, 12, exclude), "hidden")
        rows += self.squad("train", 250)
        train = self.gsm_eligible("train")
        rows += self.gsm_rows(self.gsm_eligible("test"), "test") + self.gsm_rows(train[10:62], "train")
        qa_cells = [(t, L) for t in ("qa_1", "qa_2") for L in ("8192", "16384")]
        rows += self.ruler_rows("ruler_qa", self.ruler_draw("hidden", qa_cells, 38, exclude), "hidden")
        rows += self.scbench_upstream("scbench_mf_short", "scbench_mf") + self.generated("scbench_mf", "hidden", 150, "hidden")
        rows += self.scbench_upstream("scbench_many_shot", "scbench_many_shot")
        rows += self.scbench_upstream("scbench_summary_short", "scbench_summary")
        vt_cells = [("vt", "8192"), ("vt", "16384")]
        rows += self.ruler_rows("ruler_vt", self.ruler_draw("hidden", vt_cells, 117, exclude), "hidden")
        return rows


def summarize(builder: Builder, rows: list[Row]) -> dict:
    stats: dict[str, dict] = {}
    for r in rows:
        n = builder.ntok(r.context)
        if not (MIN_CTX <= n <= MAX_CTX):
            raise SystemExit(f"{r.id}: context has {n} tokens, outside [{MIN_CTX}, {MAX_CTX}]")
        s = stats.setdefault(r.workload, {"rows": 0, "questions": 0, "min_tokens": n, "max_tokens": n})
        s["rows"] += 1
        s["questions"] += len(r.questions)
        s["min_tokens"] = min(s["min_tokens"], n)
        s["max_tokens"] = max(s["max_tokens"], n)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--hf-cache", required=True)
    ap.add_argument("--essays", required=True)
    ap.add_argument("--needles", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    b = Builder(args)
    out = Path(args.out)
    (out / "public").mkdir(parents=True, exist_ok=True)
    (out / "judge").mkdir(parents=True, exist_ok=True)
    pub, hid = b.public(), b.hidden()
    report = {"public": {"sha256": write_suite(out / "public" / "public_suite.jsonl.gz", pub),
                         "rows": len(pub), "questions": sum(len(r.questions) for r in pub),
                         "workloads": summarize(b, pub)},
              "hidden": {"sha256": write_suite(out / "judge" / "hidden_suite.jsonl.gz", hid),
                         "rows": len(hid), "questions": sum(len(r.questions) for r in hid),
                         "workloads": summarize(b, hid)}}
    (out / "judge" / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: {kk: v[kk] for kk in ("sha256", "rows", "questions")} for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
