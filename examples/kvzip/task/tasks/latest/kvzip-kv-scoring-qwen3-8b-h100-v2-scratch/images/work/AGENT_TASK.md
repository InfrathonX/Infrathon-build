# Query-agnostic KV importance scoring for Qwen3-8B

## Goal

Long-context inference is expensive partly because of the KV cache. Most of the cache can
be dropped without hurting answers, and the choice can be made before any question is seen.
Your job is a **cheap, query-agnostic importance scorer** for the frozen `Qwen/Qwen3-8B`
(bf16): after a context is prefilled, your scorer gives every
`(layer, KV head, context position)` entry one score. The evaluator keeps the globally
highest 10%, 20% and 30% of the entries and measures how much answer quality survives
compared with the full cache. Higher is better; 100 means as good as the full cache.

What makes it hard:

- **Query-agnostic:** the scorer never sees the question.
- **One ranking:** one score tensor serves all three ratios.
- **Global:** scores compete across all 36 layers and 8 KV heads, so they must be calibrated
  against each other.
- **Cheap:** the scoring time limit (below) rules out reconstruction-style scorers that cost
  about another prefill.

You start from an empty scorer. Your budget is the run's wall time (24 hours in a benchmark
run), including the time you wait for the Judge. The environment variable `KVZIP_DEADLINE_UNIX`
holds the end of your run (`date -d @$KVZIP_DEADLINE_UNIX`); at that moment the submitted tree
is captured and judged once more.

## What you write

The submitted tree in `/app/workdir/kvzip_lab` (nothing else is sent to the Judge; keep all
three in place, the workspace is captured from them when the time budget ends):

```text
scorers/scorer_vNNN/   one directory per scorer version: scorer.py, its own modules, its
                       weights, and the code that trained them (e.g. scorers/scorer_vNNN/train/)
FINAL_VERSION          the integer NNN of the version the Judge evaluates (starts at 1)
notes.md               your notes; never affects the score
```

Only the selected version runs, but every directory in `scorers/` is submitted, its weights
are checked, and it counts toward the archive size limit (check 1): keep old versions with
large weights outside the submitted tree. The selected directory is imported as the package
`scorer` (a copy of it in the Judge), so `from scorer import my_module` and relative imports
work; it must not import other versions. `scorer.py` must define:

```python
class Scorer:
    def __init__(self, config): ...            # once per evaluation, not timed
    def score(self, states) -> torch.Tensor:   # once per context, timed
```

`config` (`kvzip_eval.api.ScorerConfig`, see `/opt/kvzip/runtime/kvzip_eval/api.py`):
`device` (always `cuda:0`), `scorer_dir` (the evaluated version's directory, read-only; load
your weights from there, or relative to `__file__`), the model's shape
(`num_layers=36`, `num_kv_heads=8`, `num_q_heads=32`, `head_dim=128`, `hidden_size=4096`,
`rope_theta`, `rms_norm_eps`), `max_context_tokens=32000`, `memory_limit_bytes` (8 GiB) and
`model_config` (Qwen3-8B's `config.json` as a dict).

`states` (`kvzip_eval.api.PrefillStates`) holds everything the model computed for one
prefill, per layer, as **read-only** bf16 CUDA tensors, with `T` = protected prefix + context
tokens:

| field | shape | what it is |
|---|---|---|
| `hidden_states` | `[36, T, 4096]` | each layer's attention input (after its input RMSNorm) |
| `queries` | `[36, 32, T, 128]` | after `q_norm` and RoPE |
| `keys` | `[36, 8, T, 128]` | after `k_norm` and RoPE, exactly as in the KV cache |
| `values` | `[36, 8, T, 128]` | as in the KV cache |
| `context_start` | int | positions `[0, context_start)` are the protected prefix, `[context_start, T)` the context |

Return a **float32** tensor of shape `[36, 8, states.num_context_tokens]` on `cuda:0`, all
values finite; a higher score means "keep". Your scorer runs in its own process with its own
CUDA context; it sees no token ids, no text, no model weights and nothing about the
questions.

## Evaluation protocol

The public tests and the Judge use the same protocol (`/opt/kvzip/runtime/kvzip_eval/`).

- **Prompt.** The prefill is the Qwen3 chat system turn, then a user turn holding an
  instruction and the context. QA rows use the instruction
  `Given the context, answer to the following question or request without explanation.`,
  GSM8K rows `Given the context, answer to the following reasoning question.`; the exact
  strings are in `protocol.py`. The system turn and instruction form the protected prefix.
  Contexts are 64 to 32,000 Qwen3 tokens.
- **Retention.** Your scores are computed once per context. With `N = 36 x 8 x context
  tokens`, the evaluator keeps exactly `round(r x N)` entries for r = 0.1, 0.2 and 0.3 from
  one global ranking; ties keep the lower index in `(layer, KV head, position)` order. The
  prefix and all question and answer tokens are always visible; there is no other protected
  sink or window (if you want one, score it high yourself).
- **Answering.** Each question is appended separately to the same compressed cache:
  `\n\n{question}` closes the user turn, then an assistant turn opens with an empty
  `<think>\n\n</think>` block (thinking off); RULER rows then add their answer prefix.
  Decoding is greedy up to the workload's maximum new tokens. Full-cache answers are decoded
  in the same evaluation through the same masked attention path.

## Metric

```text
acc(row, r)   = mean of the row's per-question metric
acc(task, r)  = mean over the task's rows
rel(task, r)  = acc(task, r) / acc(task, full cache)
group(g, r)   = mean of rel(task, r) over the tasks in group g
ratio(r)      = mean of group(g, r) over the groups retrieval, contextual QA, redundancy
score         = 100 x mean of ratio(r) over r = 0.1, 0.2, 0.3
```

Averages are taken first, then divided, with no cap (a task can exceed 1); groups weigh the
same whatever their row counts. An invalid submission scores 0.

The public suite (63 rows, 419 questions; in the Work image) has these workloads:

| workload | group | per-question metric | max new tokens |
|---|---|---|---:|
| `needle` (needle in a haystack of essays) | retrieval | normalized answer inclusion | 48 |
| `scbench_kv` (UUID key-value JSON) | retrieval | normalized inclusion | 96 |
| `scbench_prefix_suffix` (word dictionary) | retrieval | normalized inclusion | 128 |
| `ruler_niah` (RULER needle tasks) | retrieval | all reference values present | 64 |
| `squad` | contextual QA | normalized inclusion | 256 |
| `gsm` (GSM8K: problem minus last sentence as context) | contextual QA | the reference number appears as a number in the answer | 512 |
| `ruler_qa` | contextual QA | any reference value present | 32 |
| `scbench_mf` (find numbers in a long list) | redundancy | exact match | 48 |
| `ruler_vt` (variable tracking) | redundancy | all reference values present | 48 |

The hidden suite is much larger and broader than the public one: more rows, more and
different task types in the same three groups, and other sources. The public score does not
predict the hidden score; use the public tests to get a scorer working and correct, and use
your own data to make it good.

## Rules

- **Query-agnostic.** The scorer never sees the question (the process boundary enforces it).
- **No token identity.** The scorer may not work out which token or kind of token sits at a
  position: no tokenizers, no vocabulary or embedding tables, no inverting hidden states or
  keys back to tokens. Statistics that merely correlate with identity, such as a key norm, or
  a learned function of the hidden states, are fine.
- **No weights someone else trained.** Learned components must be trained by you in this run
  (in the Work container you may run the frozen model as much as you like to produce training
  data). The scorer sees the model only through `states`: do not ship or load copies of the
  model's own weights. The released Fast KVzip gates can be studied but not used, neither as
  weights nor as a teacher for your own training: the Judge detects them even renamed,
  reshaped, cast, quantized or fine-tuned. Put your training code in the version directory
  (e.g. `scorers/scorer_vNNN/train/`) and describe in `notes.md` how and on which data you
  trained.
- **No lookups of known answers.** No answer tables or position tables built from benchmark
  data.
- **No training on the benchmark's own data.** Do not train, tune or validate learned
  components on SQuAD, GSM8K, the RULER or SCBench data, or the Paul Graham essays (any split,
  including the public suite's rows): the suites draw their rows from these. Other text, and
  tasks you generate yourself, also of the same kinds, are fine. The public suite is there to
  check and select scorers.
- **Real work on every call.** No results cached across contexts, no state carried from one
  context to the next, and no different behavior when not timed. The scorer's output for a
  context may depend only on that context's prefill states and on parameters fixed before
  the evaluation starts. All work happens inside `score()`: no background threads, other
  processes or work left running after it returns, and no tampering with timers, memory
  statistics or the runtime (allocate device memory through PyTorch).
- **Stay inside the submission.** The scorer may read the submitted tree and the installed
  Python packages, nothing else; it has no network and may not start processes.

An integrity review (an LLM reading your submitted code and notes against these rules)
runs before the GPU Judge; a confident violation rejects the submission.

## Checks: every reason a submission is rejected

A submission is valid only if all of these pass; otherwise it scores 0 and you get the
reason.

1. **Admission.** `FINAL_VERSION` holds one integer N and `scorers/scorer_vNNN/scorer.py`
   exists; `scorers/` holds only `scorer_vNNN` directories. Submitted files are regular files
   (no symbolic links) of these types only: text and source (`.py .md .txt .json .jsonl .yaml .yml .toml
   .cfg .ini .csv .tsv .log .sh .cu .cuh .c .cc .cpp .h .hpp`, at most 1 MiB each) and weights
   (`.safetensors .pt .pth .bin .npy .npz`). The compressed submission archive is limited to
   128 MiB (512 MiB uncompressed).
2. **Weights.** Every weight file in every submitted version must load with `safetensors`,
   `torch.load(weights_only=True)` or `numpy.load(allow_pickle=False)`, and none may hold the
   released Fast KVzip gates in any form. Weights you trained yourself and statistics of the
   model's activations (means, scales, centroids) do not trigger this check.
3. **Construction.** `Scorer(config)` must succeed within 15 minutes (it runs with no network).
4. **Output.** Every score tensor has dtype float32, shape `[36, 8, context tokens]`, device
   `cuda:0`, and only finite values.
5. **Exceptions.** Neither `Scorer(config)` nor `score()` may raise. The Judge reports only the
   exception's type, and only for Python, PyTorch, NumPy and Triton exception types; it never
   reports messages, tracebacks or the values you returned (the public tests show all of them).
   The scorer process talks to the evaluator only through its return value: anything else it
   sends on the evaluator's channel rejects the submission.
6. **Side effects.** The input tensors must be unchanged after `score()`, and the global random
   number generators (torch CPU and CUDA, Python `random`, NumPy) must be in the same state
   (use your own `torch.Generator` if you need randomness).
7. **One GPU, one process.** The scorer process must not start other processes or use any GPU
   but its own.
8. **Cost.** Scoring is timed per call (the scorer process is stopped between calls). Total
   scoring time must be at most 25% of the total plain-prefill time of the same contexts on the
   same GPU, and each context at most the larger of 50% of its own prefill time and 10x the
   suite's shortest prefill time. Two untimed warm-up calls on filler text run first; compile
   your kernels in `__init__` too, because a call that compiles is charged.
9. **Memory.** Peak allocation of the scorer process (`torch.cuda.max_memory_allocated`, your
   weights included; the read-only inputs are not counted) at most 8 GiB. PyTorch's cache in
   the scorer process is capped at 16 GiB of reserved memory: the allocator releases cached
   blocks before it reaches the cap, and an allocation that still does not fit raises
   `OutOfMemoryError`. The whole process (that cache, the CUDA context and anything allocated
   outside PyTorch) may hold at most 19 GiB of device memory, measured from outside the
   process. The public tests apply the same cap and print your peak allocation and peak
   reserved memory.
10. **Stateless.** At the end the Judge re-scores some contexts; at least 99% of the retained
    entries must be the same as the first time.
11. **Integrity review** (above).

The Judge also fails an evaluation that it cannot complete itself (for example a full-cache
accuracy of zero); that is an infrastructure error, not a verdict on your submission.

## Public tests

```bash
/workspace/public_tests/run_public_tests.sh --tier one     # first SQuAD row: does it run?
/workspace/public_tests/run_public_tests.sh --tier quick    # 2 rows per workload (18)
/workspace/public_tests/run_public_tests.sh --tier full     # all 63 rows
# options: --submission DIR (default: cwd), --version N (instead of FINAL_VERSION),
#          --scorer-dir DIR (a directory holding scorer.py), --workload NAME (repeatable),
#          --show-answers N
```

They run the same harness, scorer process and checks as the Judge on this container's GPU
and print per-row accuracy at each ratio, timings against the cost limits, peak memory and
the statelessness probe (`full` takes about 4 minutes). The public score they print is a
diagnostic: it is not your run's score and does not predict the hidden one.

## Environment

- One H100 80 GB GPU, 8 CPUs, 32 GiB RAM. The model is at `/opt/models/qwen3-8b` (read-only);
  the evaluation runtime is in `/opt/kvzip/runtime` (read-only; read it, it is the protocol).
- PyTorch 2.8.0 (CUDA 12.9), Triton 3.4.0, transformers 4.57.1 with SDPA attention, pandas,
  pyarrow. KVzip, flash-attn, xformers and FlashInfer are not installed. You may install
  packages for your own experiments; the scorer runs in the Judge with only the packages of
  this image.

## Submitting

Run `sforge-submit` to send the submitted tree to the Judge; it waits for the result. At most
one submission every 5 minutes. After the integrity review, the Judge evaluates the hidden
suite in about 15 minutes on GPUs of its own (your GPU stays free; you may run `sforge-submit`
in the background and keep working). Only one submission can be in flight: while one waits
for or runs in the Judge, `sforge-submit` refuses another ("an agent submission is already
queued or running"), so wait for its result. Each result returns the score, the three ratio
scores, the three group scores, the scoring-overhead ratios and the state of every check, or
the reason for a rejection; there are no per-workload or per-row results (the public tests
show your peak memory and timings). Your run's result is its best valid score within the time
budget.

`sforge-save-version` keeps at most 1 GiB of saved versions per run in total, and every save
stores the whole submitted tree; once that space is used up, saving fails, while submissions
(saved automatically) still work. With large weights, keep the submitted tree lean.
