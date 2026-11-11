# KVzip KV Scoring: query-agnostic KV importance for Qwen3-8B

Category: inference, module level. Start: scratch (empty scorer).

## 1. Task

Long-context inference is expensive partly because of the KV cache. KVzip showed that most of the
cache can be dropped without hurting answers, and that the choice can be made before any question is
seen. Its scorer reconstructs the context, which costs about twice a prefill. Fast KVzip replaced that
with small gates trained offline.

The agent writes a cheap, query-agnostic importance scorer for frozen Qwen3-8B, following the KVzip
evaluation protocol. After the context is prefilled, the scorer gives every
`(layer, KV head, context position)` entry one score. We keep the globally highest 10%, 20% and 30%,
then measure how much answer quality survives compared with the full cache.

What makes it hard:

- **Query-agnostic:** the scorer never sees the question.
- **One ranking:** one score tensor serves all three ratios.
- **Global:** scores compete across all 36 layers and 8 KV heads, so they must be calibrated.
- **Cheap:** the cost limit (§4.3) rules out KVzip's own reconstruction scorer.

The scorer only sees what the model computes over the context during the prefill: hidden states,
queries, keys and values for each layer. The agent starts from an empty scorer, with 24 hours and
access to public websites.

Rules:

- **No token identity:** the scorer may not work out which token or kind of token sits at a position.
  Statistics that merely correlate with identity, such as a key norm, are fine.
- **No weights someone else trained:** learned components must be trained by the agent in the run.
  The released Fast KVzip gates can be studied but not used.
- **No lookups of known answers:** no answer tables built from benchmark data.
- **Real work on every call:** no results cached across contexts, and no different behavior when not
  timed.

## 2. Evaluation protocol

Public and hidden tests share this protocol.

**Prompt.** The prefill is the Qwen3 chat system turn, then a user turn holding an instruction and the
context:

- QA rows: `Given the context, answer to the following question or request without explanation.`
- GSM8K rows: `Given the context, answer to the following reasoning question.`

The system turn and instruction form a protected prefix. Contexts are 64 to
32,000 Qwen3 tokens.

**Scoring and retention.** Score the context once. With `N = 36 × 8 × context tokens`, keep exactly
`round(r × N)` entries for r = 0.1, 0.2 and 0.3, by one global ranking. Ties keep the lower index in
`(layer, KV head, position)` order. The prefix and all question and answer tokens are always visible;
there is no other protected sink or window.

**Answering.** Each question is appended separately to the same compressed cache: `\n\n{question}`
closes the user turn, then an assistant turn opens with an empty `<think>\n\n</think>` block, so
thinking is off. RULER rows then add their answer prefix. Decoding is greedy up to the workload's
maximum new tokens. Full-cache answers are decoded in the same run through the same masked path.

## 3. Public tests (P)

For getting a scorer working and checking correctness. They don't predict the hidden score.

| workload | group | source | selection | rows | questions | context tokens |
|---|---|---|---|---:|---:|---|
| `squad` | contextual QA | SQuAD validation | first 10 distinct contexts of at least 64 tokens, all their questions | 10 | 222 | 73–324 |
| `gsm` | contextual QA | GSM8K train | first 10 rows whose context (problem minus last sentence) has at least 72 tokens | 10 | 10 | 75–101 |
| `needle` | retrieval | KVzip needle generator, upstream needle | public half of the essays; (length, depth %) = (500: 0, 50, 100), (2k: 10, 40, 70, 90), (8k: 20, 50, 80, 100) | 11 | 11 | 499–8,000 |
| `scbench_kv` | retrieval | generated, SCBench kv format | seed "public" | 8 | 80 | ~24.5k |
| `scbench_prefix_suffix` | retrieval | generated, SCBench prefix-suffix format | seed "public" | 8 | 40 | ~18.9k |
| `scbench_mf` | redundancy | generated, SCBench Math.Find format | seed "public" | 8 | 48 | 28.2k–29.6k |
| `ruler_niah` | retrieval | RULER test | one random row each: niah_single_1, niah_multikey_2, niah_multivalue at 8k; niah_multiquery at 16k | 4 | 4 | 7.5k–16.0k |
| `ruler_qa` | contextual QA | RULER test | one each: qa_1 at 8k, qa_2 at 16k | 2 | 2 | 6.1k–17.0k |
| `ruler_vt` | redundancy | RULER test | one each: vt at 8k and 16k | 2 | 2 | 7.9k–16.3k |
| **total** | | | 9 workloads | **63** | **419** | 73–29,580 |

**Essay split:** the 49 KVzip essay files are split by whether the SHA-256 of the file name is even
(public) or odd (hidden).

**Generated formats:** each follows the upstream `_short` file exactly.

- kv: a JSON object of 361–365 random UUID pairs; ask for 10 keys.
- prefix-suffix: 1,004–1,010 random 25-character strings; each question gives a prefix and a suffix.
- Math.Find: a Python list of 6,002–6,008 integers; ask 6 questions: the largest and smallest
  number, overall, containing a 7, and ending in 3.

Seeds are fixed and differ between public and hidden.

**Tiers:** `one` (the first SQuAD row), `quick` (the first two rows of each workload, 18 rows) and
`full` (all 63).

## 4. Hidden tests (E)

### 4.1 Cases

| workload | group | source | selection | rows | context tokens |
|---|---|---|---|---:|---|
| `needle` | retrieval | KVzip needle generator, `attachments/hidden/needles.json` | hidden half of the essays; 3 needles × lengths 500 / 2k / 8k × depths 0–100% in steps of 10; drop rows that land on the same sentence boundary | 96 | 499–8,000 |
| `scbench_kv` | retrieval | SCBench `scbench_kv_short` | all 100 upstream rows plus 50 generated | 150 | 24.3k–24.8k |
| `scbench_prefix_suffix` | retrieval | SCBench `scbench_prefix_suffix_short` | all 100 plus 50 generated | 150 | 18.7k–19.0k |
| `scbench_repoqa` | retrieval | SCBench `scbench_repoqa_short` | all rows | 22 | 9.5k–20.8k |
| `ruler_niah` | retrieval | RULER test, 8k and 16k | 8 niah tasks × 2 lengths × 12 random rows, excluding public rows | 192 | 7.5k–21.1k |
| `squad` | contextual QA | SQuAD train | first 250 distinct contexts of at least 64 tokens, all their questions | 250 | 70–502 |
| `gsm` | contextual QA | GSM8K test, then train | all 148 eligible test rows (at least 72 context tokens), then the next 52 eligible train rows after the public ones | 200 | 72–171 |
| `ruler_qa` | contextual QA | RULER test | qa_1 and qa_2 × 2 lengths × 38 | 152 | 5.8k–17.2k |
| `scbench_mf` | redundancy | SCBench `scbench_mf_short` | all 100 plus 150 generated | 250 | 24.2k–30.0k |
| `scbench_many_shot` | redundancy | SCBench `scbench_many_shot` | all rows | 54 | 18.6k–26.5k |
| `scbench_summary` | redundancy | SCBench `scbench_summary_short` | all rows | 62 | 9.4k–18.3k |
| `ruler_vt` | redundancy | RULER test | vt × 2 lengths × 117 | 234 | 7.9k–16.3k |
| **total** | | | 12 workloads | **1,812** | 70–29,991 |

The suite has 6,956 questions. It follows upstream KVzip (`eval.py --num 100`) scaled up: 99 rows gave
a standard error of about 8 points, and this scale brings it to about 1. SCBench's `_short` variants
are used because of the model's window, as upstream does for Gemma3-12B.

Hidden and public rows share no source row or context. RepoQA, many-shot and summary have too few rows
to split, so they are hidden only.

Don't redistribute the hidden text. Rebuild it from the sources and freeze it, so every rebuild is
identical.

### 4.2 Metric

| workload | per-question metric | max new tokens |
|---|---|---:|
| `needle` | normalized answer inclusion (KVzip) | 48 |
| `scbench_kv` | inclusion | 96 |
| `scbench_prefix_suffix` | inclusion | 128 |
| `scbench_repoqa` | pass if BLEU ≥ 0.8 against the reference function; an empty answer fails | 512 |
| `ruler_niah` | all reference values present (RULER) | 64 |
| `squad` | inclusion | 256 |
| `gsm` | numeric inclusion | 512 |
| `ruler_qa` | any reference value present (RULER) | 32 |
| `scbench_mf` | exact match | 48 |
| `scbench_many_shot` | option letter | 96 |
| `scbench_summary` | ROUGE-L | 256 |
| `ruler_vt` | all reference values present (RULER) | 48 |

```text
acc(row, r)   = mean over the row's questions
acc(task, r)  = mean over the task's rows
rel(task, r)  = acc(task, r) / acc(task, full cache)
group(g, r)   = mean of rel(task, r) over the tasks in g
ratio(r)      = mean of group(g, r) over retrieval, contextual QA and redundancy
score         = 100 × mean of ratio(r) over r = 0.1, 0.2, 0.3
```

Higher is better, and 100 means as good as the full cache.

- Average first, then divide, with no cap, as upstream does. Many-shot can exceed 1.
- Groups weigh the same regardless of row counts.
- An invalid submission scores 0.

### 4.3 Checks

A submission is valid only if all of these pass:

- **Output:** every score tensor has the exact dtype, shape and device, and is finite.
- **Side effects:** inputs and global RNG are unchanged, and no other GPU is used.
- **Cost:** total scoring time is at most 25% of total plain-prefill time for the same contexts on the
  same GPU. Each context is at most the larger of 50% of its own prefill and 10× the suite's
  shortest prefill.
- **Memory:** peak scorer allocation is at most 8 GiB.
- **Weights:** submitted weights are not the released Fast KVzip gates, even renamed, reshaped, cast or
  quantized.
- **Review:** an LLM reads the code and notes against the rules in §1.
- **Sanity:** all metrics are finite, and every workload's full-cache accuracy is positive.

### 4.4 Feedback

Each submission returns:

- the score
- the three ratio scores and the three group scores
- the overhead ratios and the state of every check
- for a rejection, the reason

There are no per-workload or per-row results. The run's result is its best valid score within the
time budget.

## 5. Resources (R)

| environment | allocation |
|---|---|
| Work | 1 × H100 80 GB, 8 CPUs, 32 GiB |
| Judge | 7 × H100 80 GB, 8 CPUs and 8 GiB per GPU, plus 32 GiB |

Software: PyTorch 2.8.0 with CUDA 12.9, Triton 3.4.0 and transformers 4.57.1, with SDPA attention.
KVzip, flash-attn, xformers and FlashInfer are not installed. A submission should take under an hour
on the Judge.

## 6. Time (T)

24 hours, including tool time and waiting for results. At most one submission every 5 minutes.

## 7. Baselines

| baseline | definition | expected |
|---|---|---|
| KNorm | score = negative L2 norm of each key | far below Fast KVzip, retrieval near zero |
| Fast KVzip | released gate `qwen3-8b/q4_dim16_sink16.pt`, as in kvpress `FastKVzipPress`: the first 4 positions and a trailing window of `max(1, int(0.02 n))` positions (4,096 when n ≥ 32,000) are set to 1.0 | high, but below 100 at r = 0.1; not a ceiling |

Don't use the Fast KVzip repo's own window rule, a fixed 4,096 positions from 16,000 tokens up. On long
contexts that window covers the whole budget at r = 0.1, which turns the gate into pure recency.

## 8. Attachments

| file | what it is |
|---|---|
| `attachments/public/kvzip/eval.py` | KVzip's evaluation loop |
| `attachments/public/kvzip/metric.py`, `repo_qa_utils.py` | per-workload metrics, including RepoQA |
| `attachments/public/kvzip/needle_data.py` | the needle generator |
| `attachments/public/kvzip/score.py` | KVzip's reconstruction scoring |
| `attachments/public/kvpress/knorm_press.py` | KNorm |
| `attachments/public/kvpress/fastkvzip_press.py` | Fast KVzip gate and window rule |
| `attachments/public/kvpress/ruler_metrics.py` | RULER string match |
| `attachments/hidden/needles.json` | 3 hidden needles (needle, question, answer). Judge only: never in Work |

## 9. Sources

| what | url |
|---|---|
| Qwen3-8B | https://huggingface.co/Qwen/Qwen3-8B |
| KVzip paper | https://arxiv.org/abs/2505.23416 |
| KVzip code, needle generator and essays | https://github.com/snu-mllab/KVzip |
| Fast KVzip paper | https://arxiv.org/abs/2601.17668 |
| Fast KVzip code | https://github.com/Janghyun1230/FastKVzip |
| Fast KVzip weights | https://huggingface.co/Jang-Hyun/Fast-KVzip |
| kvpress | https://github.com/NVIDIA/kvpress |
| SCBench (preprocessed) | https://huggingface.co/datasets/Jang-Hyun/SCBench-preprocessed |
| SQuAD | https://huggingface.co/datasets/rajpurkar/squad |
| GSM8K | https://huggingface.co/datasets/openai/gsm8k |
| RULER | https://huggingface.co/datasets/simonjegou/ruler |
