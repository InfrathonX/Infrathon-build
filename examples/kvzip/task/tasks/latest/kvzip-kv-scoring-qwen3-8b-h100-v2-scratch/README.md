# KVzip KV Scoring: query-agnostic KV importance for Qwen3-8B

The agent writes a cheap, query-agnostic importance scorer for the KV cache of the frozen
`Qwen/Qwen3-8B` (bf16), following the KVzip evaluation protocol. After a context is
prefilled, the scorer gives every `(layer, KV head, context position)` entry one score; the
evaluator keeps the globally highest 10%, 20% and 30% of the entries and measures how much
answer quality survives against the full cache. This is a scratch task (inference, module
level): the workspace starts with an empty scorer. KVzip showed that most of the cache can
be dropped before any question is seen, but its scorer reconstructs the context at about
twice the cost of a prefill; Fast KVzip replaced that with small gates trained offline. The
cost limit here rules out the reconstruction scorer, and the released Fast KVzip gates may
be studied but not used.

## The task

The agent works in `/app/workdir/kvzip_lab` inside the Work container (one GPU). The
submitted tree is:

```text
scorers/scorer_vNNN/   one directory per scorer version: scorer.py (class Scorer), its modules,
                       weights and the code that trained them
FINAL_VERSION          the integer NNN of the evaluated version
notes.md               how the scorer works and how its parameters were produced
```

The selected version is evaluated as the package `scorer` (a copy in the Judge). The task JSON
declares this as its version enumeration, so a run's retained versions can be re-scored one by
one (`launch/kvzip-kv-scoring/replay.sh --per-version`; the Judge honours the host's
`candidate_version`).

Interface (`/opt/kvzip/runtime/kvzip_eval/api.py`, read-only):

- `Scorer(config)` once per evaluation, not timed (at most 15 minutes). `config` carries the
  device, the submitted tree's path, the model's shape and the limits.
- `score(states) -> Tensor` once per context, timed. `states` holds, per layer, the
  attention input after the input RMSNorm `[36, T, 4096]`, the queries after `q_norm` and
  RoPE `[36, 32, T, 128]`, and the keys and values as in the KV cache `[36, 8, T, 128]`, for
  the protected prefix and the context (`context_start` separates them). It returns a
  float32 `[36, 8, context tokens]` tensor on `cuda:0`.

The scorer runs in its own process: in the Judge under an unprivileged user that can read
neither the model, the hidden suite nor any Judge file, with no network, receiving the
states through shared device buffers (CUDA VMM memory exported as file descriptors). It sees
no token ids, no text and nothing about the questions. The harness treats that process as
hostile: replies are length-bounded JSON with known fields only (never unpickled,
`kvzip_eval/ipc.py`), and every agent-visible word about it comes from fixed lists
(exception type names of Python, PyTorch, NumPy and Triton; fixed output-problem texts; no
values the scorer reported).

Rules (definition section 1): query-agnostic; no token identity; no weights someone else
trained (learned parameters are trained by the agent in the run; the released Fast KVzip
gates can be studied, not used); no lookups of known answers; real work on every call (no
caching or state across contexts, no different behaviour when not timed). The brief adds
what follows from them for the agent: the scorer reads only the submitted tree, starts no
processes, leaves no work running after `score()`, and does not ship copies of the model's
weights.

Public tests in the Work container (`/workspace/public_tests/run_public_tests.sh --tier
one|quick|full`) run the same harness, scorer process and checks on the public suite; they
are diagnostics: their public score is not the run's score.

## Evaluation

**Protocol** (`kvzip_eval/protocol.py`, `model.py`, `attention.py`). The prefill is the Qwen3
chat system turn (`You are a helpful assistant.`), then a user turn holding the KVzip
instruction and the context; the system turn and instruction are the protected prefix. The
score tensor is ranked once, globally: `round(r x N)` entries are kept for r = 0.1, 0.2,
0.3 with `N = 36 x 8 x context tokens`, ties keeping the lower `(layer, head, position)`
index. Each question is appended separately as `\n\n{question}`, closing the user turn,
followed by an assistant turn with an empty `<think>\n\n</think>` block and, for RULER, its
answer prefix; decoding is greedy to the workload's limit. Every element (question x
setting) of a row is decoded in one batch through the same masked attention (a Triton
kernel with a per-row retention mask over the shared prefilled cache, float32 softmax), the
full cache included, so the full-cache answers are decoded in the same evaluation through
the same path.

**Metric.**

```text
acc(task, r) = mean over rows of the mean over the row's questions
rel(task, r) = acc(task, r) / acc(task, full)          (no cap)
group(g, r)  = mean of rel over the tasks of g         (retrieval, contextual QA, redundancy)
score        = 100 x mean over r in {0.1,0.2,0.3} of the mean over the three groups
```

Per-question metrics: normalized answer inclusion (needle, SCBench kv and prefix-suffix,
SQuAD), RULER all/any reference present (niah, vt / qa), GSM8K the reference number among
the numbers of the answer, exact match (Math.Find), the option letter (many-shot, KVzip's
rule), ROUGE-L (summary), and RepoQA pass at BLEU >= 0.8 against the reference function
(empty answer fails).

**Checks** (any failure: invalid, score 0, with the reason): admission (layout, file types,
sizes, no links); weights (every weight file loadable without pickles; none holds the
released gates, renamed, reshaped, transposed, cast, int8/int4/fp8 quantized or fine-tuned);
construction; output (float32, shape, device, finite; unchanged after the call returns);
exceptions; inputs unchanged (exact for keys and values, secret random projection for hidden
states and queries); global RNGs unchanged; one GPU and one process (process table and NVML
per-GPU process count); cost (total scoring <= 25% of total plain prefill; each context <=
max(50% of its prefill, 10x the suite's shortest prefill); the scorer process is stopped
between calls); memory (peak `torch.cuda.max_memory_allocated` of the scorer process <= 8
GiB, NVML footprint as a cross-check); statelessness (contexts re-scored after others keep
>= 99% of their retained entries); the LLM integrity review (`llm_judge/POLICY.md`). A
full-cache accuracy of zero on any workload is an evaluator failure (infrastructure error,
no score).

**Feedback.** Exactly the definition's list: the score, the three ratio scores, the three group
scores, the overhead ratios and the state of every check, or the rejection reason; no
per-workload or per-row results (those, the scorer's peak allocation and the Judge's wall time
go only to the controller-only private artifact).

**Suites.** Public (in the Work image): 63 rows, 419 questions, 9 workloads (SQuAD validation,
GSM8K train, needles in the public half of the KVzip essays, SCBench-format kv /
prefix-suffix / Math.Find generated with the public seed, RULER 8k/16k rows). Hidden (Judge
only): 1,812 rows, 6,956 questions, 12 workloads, contexts of 70 to 30,026 tokens: needles
(three hidden needles in the other half of the essays), SCBench `_short` kv, prefix-suffix,
RepoQA, Math.Find, many-shot and summary (all upstream rows plus generated ones for kv,
prefix-suffix and Math.Find), RULER niah / qa / vt rows not in the public suite, SQuAD train,
GSM8K test and train. The selection rules are the definition's; `judge/suite_builder/`
regenerates both suites from the pinned sources and `judge/suite_manifest.json` freezes
their identities (sha256 of the content). Public and hidden share no source row or context
(`tests/test_suites.py`).

**Submissions.** Manual only: `sforge-submit` sends the tree through the Judge server (rate
limit: one per 300 s; integrity review) to a Judge container on the seven Judge GPUs, which
evaluates the whole hidden suite in about 13 minutes. Each accepted submission is kept as
`submission.tar.gz` with its `report.json`; at the deadline the workspace is captured from
the stopped container as `final_archive.tar.gz` and evaluated once more the same way. The
run's result is its best valid score.

## Hardware

Eight GPUs per run: Work on one (`--work-gpu`), the Judge on seven (`--judge-gpus`); the
Judge measures while Work keeps going. Each Judge GPU holds the model, the staging buffers of
the scorer's inputs for contexts up to 32k tokens and the scorer's process; the measured
peak is in the report and in the task JSON's capability envelope. Work: 8 CPUs, 32 GiB;
Judge: 56 CPUs, 88 GiB. Only the H100 80 GB (SXM) has been qualified.

| hardware type | status |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | untested |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

## Running

```bash
T=tasks/latest/kvzip-kv-scoring-qwen3-8b-h100-v2-scratch
$T/scripts/build_images.sh       # fetch + verify inputs, build suites (frozen identities), images
launch/kvzip-kv-scoring/check.sh --work-gpu G --judge-gpus G1,...,G7   # qualification
launch/kvzip-kv-scoring/run.sh --model MODEL [--agent codex|claude-code] --auth auth.json \
    --work-gpu G --judge-gpus G1,...,G7 [--hours 24] [--dry-run]
# one archive through the Judge (integrity review as configured):
$T/scripts/eval_archive.sh --judge-gpus G1,...,G7 ARCHIVE
# the planted submissions (tests/planted/README.md):
$T/scripts/run_planted.sh --judge-gpus G1,...,G7
# re-score a run's saved candidates, or each retained scorer_vNNN once:
launch/kvzip-kv-scoring/replay.sh --origin-run RUN_ID --measurement-set NAME --judge-gpus G1,...,G7 [--per-version]
```

`INFRATHON_ASSETS_ROOT` receives the inputs (`assets/manifest.json`): the model laid out as
plain files and mounted read-only (`KVZIP_MODEL_DIR` overrides), the released Fast KVzip
gate, the KVzip essays (git, pinned commit) and the datasets in the Hugging Face cache. The
generated suites go to `$INFRATHON_ASSETS_ROOT/<task>/generated/` (the hidden one mode 0600)
and enter only the host-built Judge holdout layer (`images/judge/Dockerfile.private`).
Outputs go to `outputs/kvzip-kv-scoring-qwen3-8b-h100-v2-scratch/` (`runs/<run id>/`).

The task-local entry `scripts/sforge_kvzip_entry.py` gives Work its GPU, the model mount and
the launcher's internal network (outside access only through the restricted CONNECT
sidecar, public hosts on 80/443), and gives every Judge container its seven GPUs, the model
below a root-only directory, no network and no new privileges.

## Baselines

Measured on the hidden suite on H100 80 GB SXM through the real Judge; re-running a submission gives
the same score bit for bit (`launch/kvzip-kv-scoring/check.sh` re-measures them and checks the
definition's expectations):

| baseline | definition | score | ratio 0.1 / 0.2 / 0.3 | retrieval / contextual QA / redundancy |
|---|---|---|---|---|
| starting stub | empty scorer | invalid (`NotImplementedError`) | | |
| KNorm (`baselines/knorm`) | negative L2 norm of each key | 24.29 | 23.33 / 23.37 / 26.16 | 0.31 / 20.08 / 52.47 |
| Fast KVzip (`baselines/fastkvzip`) | released gate as kvpress `FastKVzipPress` | 89.25 | 72.23 / 95.08 / 100.45 | 89.84 / 78.49 / 99.42 |

The Fast KVzip baseline is a trusted reference: its archive does not contain the gate (it
exceeds the 128 MiB archive limit and would be rejected); the host asks the Judge for it
through the private evaluation context (`eval_archive.sh --reference-baseline fastkvzip`),
the Judge copies the gate into the scorer's tree, and the weights check is reported as
fired instead of rejecting.

## Files

```text
images/work/        Work image: AGENT_TASK.md, workdir/kvzip_lab (starting stub in scorers/scorer_v001),
                    runtime/kvzip_eval (protocol, model, masked attention, metrics, harness,
                    scorer process and its JSON channel), public_tests/ (runner; the public
                    suite is added at build)
images/judge/       public Judge (verify_kvzip.py entry; kvzip_judge: run_judge.py,
                    weights_check.py, hidden_workloads.py: the hidden-only workloads and
                    their metrics, which the Work image does not name) and
                    Dockerfile.private (hidden suite, its identity, released gate)
judge/              suite_builder/build_suites.py, private/needles.json (hidden needles),
                    suite_manifest.json (frozen suite identities)
baselines/          KNorm and Fast KVzip scorers (scripts/make_archive.py packs them as
                    scorers/scorer_v001 with FINAL_VERSION 1)
scripts/            build, suites, entry, eval, run, replay, planted runner, layout validator,
                    proxy
tests/              CPU tests (protocol, metrics, aggregation, suites and their disjointness,
                    admission, released-gate detection, feedback bounds, the scorer channel);
                    planted/ cases (19, scripts/run_planted.sh)
llm_judge/          integrity-review policy
assets/             input manifest and fetch.sh
```

## Upstream and license

- KVzip (`snu-mllab/KVzip`, MIT) at `5d8472975d286e3da6b52178af39d18904330c0d`: prompt
  template, needle generator (`data/needle`), metrics (`results/metric.py`,
  `repo_qa_utils.py`), the 49 essays (fetched, not vendored). Ported, with the changes listed
  in the report: separate tokenization of prompt pieces kept; questions appended as
  `\n\n{question}` (no `Q:`), sorted essay order, GSM8K number-aware inclusion, RepoQA pass
  against the reference function with empty answers failing.
- kvpress (`NVIDIA/kvpress`, Apache-2.0): KNorm and Fast KVzip press (baselines), RULER string
  match.
- Fast KVzip gate `Jang-Hyun/Fast-KVzip@67ce1265f510a946bf22486a30269f99066b3cbb`
  (`qwen3-8b/q4_dim16_sink16.pt`): Judge only.
- `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` (Apache-2.0).
- Data: SQuAD (`rajpurkar/squad@7b6d24c4`, CC BY-SA 4.0), GSM8K (`openai/gsm8k@740312ad`, MIT),
  SCBench preprocessed by KVzip (`Jang-Hyun/SCBench-preprocessed@a079be91`), RULER
  (`simonjegou/ruler@24adceac`). Suites are rebuilt from these at build time, never
  redistributed.
- `scripts/restricted_proxy.py` is the restricted CONNECT sidecar of the other tasks.

The task's own code is under the root `LICENSE` of the repository that holds it.
