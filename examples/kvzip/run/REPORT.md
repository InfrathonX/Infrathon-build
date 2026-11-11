# Task build report: kvzip

- **Status:** finished
- **Input:** `examples/kvzip/input`
- **Reference:** an Infrathon checkout with five example tasks, at `2551a2388262`
- **Builder:** claude-code `claude-opus-5-5` (effort max)
- **Trial agent:** codex `gpt-6.1-sol` (effort max), 6 h per trial, up to 2 trials

## Steps

| step | hours | done, hours after start |
|---|---|---|
| setup | 0.1 | 0.1 |
| build | 3.8 | 4.7 |
| check-1 | 0.5 | 5.2 |
| trial-1 | 6.3 | 11.6 |
| review-1 | 1.2 | 12.8 |
| check-2 | 0.5 | 13.3 |
| trial-2 | 6.5 | 19.7 |
| review-2 | 0.9 | 20.6 |
| check-final | 0.5 | 21.1 |

## Builder

| turn | hours | tokens out | cost $ | error |
|---|---|---|---|---|
| build | 3.8 | 398885 | 64.5 |  |
| review-1 | 1.2 | 124813 | 23.0 |  |
| review-2 | 0.9 | 61263 | 15.3 |  |

Builder cost as its CLI reports it: $102.8. Trial agents and integrity reviews are not included.

## Checks from a clean clone

| step | commit | result | minutes | why |
|---|---|---|---|---|
| check-1 | 3fc64ae535ee | pass | 29 |  |
| check-2 | 4894250b50ff | pass | 30 |  |
| check-final | df5cadc039e6 | pass | 29 |  |

## Trial 1

gpt-6.1-sol for 6.0 h on commit `3fc64ae535ee`: 18 submissions, 16 valid, best score **98.3487**. Launcher exit 0.

| # | valid | score | Judge s | summary |
|---|---|---|---|---|
| 1 | True | 42.5937 | 800 | valid; score 42.59 |
| 2 | True | 84.1859 | 764 | valid; score 84.19 |
| 3 | True | 90.2851 | 762 | valid; score 90.29 |
| 4 | True | 92.6166 | 766 | valid; score 92.62 |
| 5 | False | 0.0 | 5 | invalid: submitted weights match the released Fast KVzip gates |
| 6 | False | 0.0 | 4 | invalid: submitted weights match the released Fast KVzip gates |
| 7 | True | 93.1542 | 769 | valid; score 93.15 |
| 8 | True | 93.3411 | 766 | valid; score 93.34 |
| 9 | True | 94.5029 | 782 | valid; score 94.50 |
| 10 | True | 94.5919 | 784 | valid; score 94.59 |
| 11 | True | 94.4152 | 785 | valid; score 94.42 |
| 12 | True | 94.3913 | 799 | valid; score 94.39 |
| 13 | True | 94.8752 | 787 | valid; score 94.88 |
| 14 | True | 97.3544 | 779 | valid; score 97.35 |
| 15 | True | 98.1246 | 779 | valid; score 98.12 |
| 16 | True | 98.3487 | 784 | valid; score 98.35 |
| 17 | True | 97.907 | 780 | valid; score 97.91 |
| 18 | True | 98.3487 | 784 | valid; score 98.35 |

Builder's review: **retrial**. The task works and the trial's best submission (98.35, scorer v020) is a legitimate, query-agnostic scorer of exactly the kind the expert wanted to measure: Fast KVzip-style gates the agent trained itself on its own reconstruction-attention targets, plus context-adaptive head budgets, probe features and learned generic-query templates. It used no benchmark data, no released gates and no token identity, and the integrity review passed it. But the trial exposed seven task problems, all now fixed in 3836737 and 4894250. (1) trial_cmd could not start from a fresh clone. (2) A Pearson-based weights check falsely rejected two honest submissions; one of them is valid and scores 92.99 under the fixed check. (3) The same check never compared tensors whose width is not a multiple of 4096. (4) The Work image's runtime named the three hidden-only workloads and their metrics, and the agent built training data to match them. (5-7) Three brief statements were wrong or missing: second submissions are refused, not queued; the budget was hard-coded at 24 h in a 6 h run; saved versions have a 1 GiB quota. Each fix is verified: fresh-clone trials ran end to end, the trial's archives were re-scanned and re-judged, the Work image was scanned, and check_cmd passed from a clean clone with identical baseline scores. What only another trial can show is how agents behave and score once the hidden workload names are gone and the deadline and submission rules are stated correctly. That retrial also tells the expert how much of this round's near-saturated 98.35 came from the leak. The expert should separately decide whether the uncapped metric leaves enough headroom above Fast KVzip; I recorded that question in the report and did not change the metric.

| kind | severity | what | fixed |
|---|---|---|---|
| infrastructure | blocking | trial_cmd from a fresh clone exited after 1 s. The trial runner required a per-checkout image build record (outputs/<task>/build/image-manifest.json), and only the checkout where check_cmd ran had one | 3836737: run_fixed_budget.sh builds the images (idempotent, about 10 s with a warm Docker cache) when the record is miss |
| task_problem | major | False positive in the released-gate weights check rejected two honest submissions. Their only 4096-aligned tensor was a 3x4096 mean of Qwen3 attention inputs (layers 8/17/26), used for feature centeri | 3836737: rank (Spearman) correlation instead of Pearson, plus at least 64 distinct matching candidate rows. Regression t |
| task_problem | major | Blind spots in the weights check, found while fixing the false positive. Tensors whose element count or width is not a multiple of 4096 were never compared. The agent's 4098-wide gate layers were not  | 3836737: flat rows aligned at both ends, first and last 4096 values of wider rows and columns, int4 in both encodings, r |
| task_problem | major | The Work image revealed the hidden suite's extra task types. The read-only runtime protocol.py listed scbench_repoqa, scbench_many_shot and scbench_summary with their metrics and token limits; metrics | 4894250: hidden-only workloads and metrics moved to images/judge/kvzip_judge/hidden_workloads.py, registered by the Judg |
| task_problem | minor | The brief said a submission made while another is being judged waits for it. The server actually refuses it with HTTP 409 'an agent submission is already queued or running'. | 4894250: the brief states that only one submission can be in flight and a second is refused. |
| task_problem | minor | The brief hard-coded a 24-hour budget. The run had 6 hours and the agent had no way to know its deadline; it was cut off while testing its last version (v022). | 4894250: the Work container gets KVZIP_DEADLINE_UNIX and KVZIP_TIME_BUDGET_SECONDS (from the launcher's budget); the bri |
| task_problem | minor | Two undocumented size limits. sforge-save-version hit the framework's 1 GiB per-run checkpoint quota after about 16 saves of a 60-113 MB tree. The brief's 'keep old versions if you like' inflated ever | 3836737 and 4894250: the brief says every scorers/ version is submitted, scanned and counted toward the archive limit, a |
| observation | major | The score nearly saturates. Fast KVzip's level (89.25) was passed at 33 minutes (agent-3, 90.29). The agent then climbed steadily to 98.35 at 5.2 h (ratios 92.8/100.8/101.5; redundancy group 101.9). O |  |
| observation | minor | The best submission is what the expert hoped to measure. Its parts: per-layer MLP gates on hidden states, trained by the agent on its own KVzip-style reconstruction attention (frozen model replaying W |  |
| observation | minor | The public suite is a weak diagnostic for strong scorers. Both public ruler_qa rows have full-cache accuracy 0 under RULER's string match, so they are excluded from the public score. Public Math.Find  |  |
| observation | minor | Judge time and determinism are as designed. Each submission took 762-800 s in the Judge plus 24-70 s of integrity review; 17 submissions fit in 6 h. The final evaluation of v020 reproduced 98.3487 exa |  |
| agent_limit | minor | The agent declared itself finished at about 1h17m and continued only because the stop hook sent it back. It did not track its budget, made two slips in its own training data (synthetic math under the  |  |

## Trial 2

gpt-6.1-sol for 6.0 h on commit `4894250b50ff`: 22 submissions, 19 valid, best score **94.8211**. Launcher exit 0.

| # | valid | score | Judge s | summary |
|---|---|---|---|---|
| 1 | True | 44.8974 | 818 | valid; score 44.90 |
| 2 | True | 78.1323 | 774 | valid; score 78.13 |
| 3 | True | 86.445 | 769 | valid; score 86.44 |
| 4 | True | 87.7562 | 775 | valid; score 87.76 |
| 5 | True | 89.8763 | 775 | valid; score 89.88 |
| 6 | True | 90.3896 | 772 | valid; score 90.39 |
| 7 | True | 92.3323 | 773 | valid; score 92.33 |
| 8 | True | 93.0993 | 764 | valid; score 93.10 |
| 9 | True | 93.5489 | 769 | valid; score 93.55 |
| 10 | True | 93.6396 | 766 | valid; score 93.64 |
| 11 | True | 94.1949 | 772 | valid; score 94.19 |
| 12 | False | 0.0 | 779 | invalid: the scorer's peak device memory allocation exceeds 8 GiB |
| 13 | False | 0.0 | 771 | invalid: the scorer's peak device memory allocation exceeds 8 GiB |
| 14 | False | 0.0 | 767 | invalid: the scorer's peak device memory allocation exceeds 8 GiB |
| 15 | True | 91.1721 | 769 | valid; score 91.17 |
| 16 | True | 94.8211 | 773 | valid; score 94.82 |
| 17 | True | 93.9989 | 782 | valid; score 94.00 |
| 18 | True | 93.1037 | 793 | valid; score 93.10 |
| 19 | True | 93.3935 | 784 | valid; score 93.39 |
| 20 | True | 94.769 | 775 | valid; score 94.77 |
| 21 | True | 94.7379 | 775 | valid; score 94.74 |
| 22 | True | 94.8211 | 772 | valid; score 94.82 |

Builder's review: **retrial**. The task ran as designed: 21 submissions, Judge runs of 13-14 minutes, deterministic results, and a best submission (94.82, scorer v018) that is a legitimate, query-agnostic scorer of the kind the expert wanted. It uses per-layer gates trained by the agent on its own KVzip-style reconstruction targets and full-cache answers, over Gutenberg, WikiText-2, Nemotron and synthetic data, plus received-attention features from a streaming Triton kernel. Without the hidden workload names that leaked in round 1, the run ended about 5.6 points above Fast KVzip and plateaued at 93-95, with headroom left at r = 0.1. The trial did expose one major task problem. An undocumented outside memory cross-check (process footprint at most 8 + 2 GiB) rejected three honest submissions under a message about the allocation peak, although their peak allocation was 2.5-3.5 GiB; the public tests could not reproduce it, and the agent lost three Judge slots chasing it. Re-judged with the fix, one of them is valid at 94.85, above the run's best. Commit df5cadc fixes it with three documented limits that the public tests share: the stated allocation rule, a 16 GiB cap on PyTorch's cache, and a 19 GiB whole-process limit. It also fixes three smaller problems: invalid results no longer return quality numbers; the 'public tests never produce a score' wording, which contradicted the runner and the framework's version-capture rule, is corrected; and the brief and integrity policy now forbid training on the benchmark's own source data. The memory fix is verified mechanically: the trial archive was re-judged and planted cases confirm both rejections. The new training-data rule and the reworded brief change what the agent is told and what the integrity review enforces. Only another trial can show that agents follow them and that the review applies the rule without rejecting honest training data, so I recommend one more round.

| kind | severity | what | fixed |
|---|---|---|---|
| task_problem | major | The memory check rejected three honest submissions. The brief's rule is peak allocation (torch.cuda.max_memory_allocated) at most 8 GiB, but the Judge also compared the scorer process's NVML footprint | df5cadc: the verdict follows the stated rule (allocation at most 8 GiB). PyTorch's cache in the scorer process is capped |
| task_problem | minor | Invalid submissions still returned ratio and group scores when they were rejected after the full suite ran (e.g. the memory rejections). That gives quality feedback for rule-breaking configurations, a | df5cadc: build_result emits ratio and group scores only for valid results; overhead ratios stay (they explain cost rejec |
| task_problem | minor | The brief said the public tests 'never produce a score', but the runner prints 'PUBLIC SCORE', and the framework's instructions ask the agent to save a version before any score-producing public test.  | df5cadc: the brief, agent_query and README call it a diagnostic public score that is not the run's score. |
| task_problem | minor | Loophole, not used. Nothing forbade training, tuning or validating on the datasets the hidden suite draws rows from (SQuAD train, GSM8K train/test, the RULER and SCBench releases, the Paul Graham essa | df5cadc: new brief rule 'No training on the benchmark's own data' (no training, tuning or validation on those sources in |
| observation | minor | The best submission (agent-16, v018, 94.82) is legitimate and measures what the expert intended. It uses per-layer MLP gates on attention inputs with learned context-mean, document, local-pool, positi |  |
| observation | minor | Score level and saturation after closing round 1's leak. The best was 94.82 (round 1: 98.35), and the run plateaued at 93.1-94.8 for its last 2.5 hours. Ratios 0.2 and 0.3 sit near 99; r = 0.1 (about  |  |
| observation | minor | The public suite is a weak diagnostic for strong scorers. Public ruler_qa full-cache accuracy is 0 (excluded), public Math.Find full-cache accuracy is low, and the quick tier reads 98-99 while the Jud |  |
| observation | minor | Judge time, determinism and the round-1 fixes held. Judge runs took 763-818 s; the final evaluation reproduced 94.8211; the deadline variable was used correctly; there were no 409/429 errors; archives |  |
| agent_limit | minor | The agent ran 32-45 GiB training jobs on the Work GPU next to its public tests, which ran out of memory twice (about 10 minutes). It never saved a version, misdiagnosed the memory rejection as a peak  |  |

## Sandbox

- preflight: 30 checks passed
- requests refused by the network policy: 10

## Deliverable

- `deliverable/task.patch`: everything since the reference commit, 482965 bytes
- `deliverable/task-only.patch`: the task and launcher directories only, 476892 bytes
- `deliverable/task.bundle`: the builder's commits, for `git fetch`
- commits: 9

---

# The builder's report

# KVzip KV Scoring for Qwen3-8B: build report

Task directory `tasks/latest/kvzip-kv-scoring-qwen3-8b-h100-v2-scratch`, task id
`kvzip_kv_scoring_qwen3_8b_h100_scratch_v2`, launcher `launch/kvzip-kv-scoring/` (`run.sh` runs
a benchmark run, `check.sh` qualifies the task, `replay.sh` re-judges saved versions).

This report is for the expert who wrote the definition:
- Part 1 says what was built.
- Part 2 lists every decision I made where the definition left room, with the reason and what
  changing it would cost.
- Part 3 gives the measurements.
- Part 4 lists what is not done, not enforced, or still open.

"Measured" means I ran it on this machine (8 × H100 80 GB SXM). Anything estimated or assumed
is marked as such.

## Results at a glance

| | definition expects | measured (real Judge, hidden suite, 7 × H100) |
|---|---|---|
| starting workspace (empty scorer) | (scratch start) | invalid, score 0: `Scorer.score() raised NotImplementedError during the untimed warm-up call`; Judge 15–20 s |
| KNorm | far below Fast KVzip, retrieval near zero | **24.29**: ratios 23.33 / 23.37 / 26.16; retrieval **0.31**, contextual QA 20.08, redundancy 52.47 |
| Fast KVzip | high, below 100 at r = 0.1, not a ceiling | **89.25**: ratios **72.23** / 95.08 / 100.45; retrieval 89.84, contextual QA 78.49, redundancy 99.42 |
| Judge time per submission | under an hour | 12.6–13.0 min inside the Judge for both baselines (758–782 s); 13.1–13.4 min wall time |
| scoring overhead | total ≤ 25 %; per context ≤ max(50 % own prefill, 10 × shortest prefill) | KNorm 0.54 % total; Fast KVzip 8.1 % total, worst context at 0.48–0.53 of its own limit |
| scorer peak allocation | ≤ 8 GiB | KNorm 0.09 GiB, Fast KVzip 0.65 GiB |
| planted bad submissions | each rejected | 19 of 19 rejected for the intended reason |
| trial round 1 (codex, 6 h) | (not specified) | best **98.35**, legitimate (Part 5). Seven task problems found and fixed: fresh-clone start, weights false positive and blind spots, hidden workload names in the Work image, three brief statements |
| trial round 2 (codex, 6 h) | (not specified) | best **94.82**, legitimate (Part 6). Three honest submissions falsely rejected by the memory cross-check; fixed, and one of them re-judged at 94.85. Also fixed: quality numbers for invalid results, public-score wording, benchmark-data training rule |
| `check_cmd` from a clean clone of HEAD | passes | **PASSED** at the final HEAD (`df5cadc`, after the trial-2 fixes) in 30 minutes; earlier at `4894250` and `3fc64ae` |

Both baselines land where the definition says. Re-running a submission gives the same score bit
for bit.

## 1. What was built

**The agent's side (Work image, one GPU).**
- The brief, `AGENT_TASK.md`.
- The starting workspace `/app/workdir/kvzip_lab`. Its three paths are what gets submitted:

  ```text
  scorers/scorer_v001/scorer.py   the empty scorer: class Scorer, score() raises NotImplementedError
  FINAL_VERSION                   "1", the version the Judge evaluates
  notes.md                        the agent's notes (training method and data go here)
  ```

- The read-only evaluation runtime `/opt/kvzip/runtime/kvzip_eval`. It *is* the protocol: prompt,
  ranking, masked decoding, metrics, scorer process and admission rules. It covers the public
  workloads only. The Judge registers the hidden suite's other workloads and their metrics
  from `kvzip_judge/hidden_workloads.py` (Part 5, fix 4).
- The public suite (63 rows, 419 questions) and its runner
  (`/workspace/public_tests/run_public_tests.sh --tier one|quick|full`).
- The model, mounted read-only at `/opt/models/qwen3-8b`.

The agent can keep many scorer versions side by side. `FINAL_VERSION` picks the one that is
judged, and that directory is imported as the package `scorer`. Training code lives in the
version directory (for example `scorers/scorer_v003/train/`), so each version carries the code
that produced its weights.

**The scorer interface.**
- `Scorer(config)` runs once. It is not timed and may take at most 15 minutes.
- `score(states) -> float32 [36, 8, n_ctx]` runs once per context and is timed.
- `states` covers the protected prefix and the context, for all 36 layers:
  - the attention input after the layer's input RMSNorm, `[36, T, 4096]`;
  - the queries after `q_norm` and RoPE, `[36, 32, T, 128]`;
  - the keys and values exactly as in the KV cache, `[36, 8, T, 128]`;
  - `context_start`.

**The Judge (seven GPUs, no network).** The entry point is `images/judge/verify_kvzip.py`, which
calls `kvzip_judge/run_judge.py`. The Judge first:
- checks the submitted tree (admission rules);
- scans every weight file of every version for the released gates;
- copies the selected version;
- starts one evaluation process per GPU.

Each process loads the model and takes rows from a shared queue: one of the shortest contexts
first, then the largest first. For each row it:
1. prefills prefix + context and times it ("plain prefill");
2. prefills again, copying every layer's states into shared device buffers;
3. runs one timed `score()` call in the scorer process;
4. checks the output, the inputs, the RNG, memory and processes;
5. ranks once, then decodes every question at full cache, 0.1, 0.2 and 0.3 in one batch through
   the same masked attention;
6. scores the answers.

Statelessness is probed after 8 rows and again at the end. The Judge then aggregates, applies the
cost rule and writes two things: the agent-visible result, and a private artifact only the
operator sees (per-workload and per-row results, answers, timings, GPU samples, peak memory).

**Isolation of the submission.**
- The scorer runs in its own process as an unprivileged user, one per GPU. That user cannot read
  the model (mounted under a root-only directory), the hidden suite, the released gate, the Judge
  code or the evaluation context.
- It has no network, because the Judge container has none.
- It sees no token ids, text or questions.
- It receives the states through CUDA virtual-memory buffers exported as file descriptors.
- The harness stops it with SIGSTOP between calls, times each call from outside, and compares its
  inputs afterwards.
- Its replies to the harness are length-bounded JSON, checked field by field. Nothing it says
  reaches the agent as free text (decision 39).
- Its stdout never reaches the Judge's output.

**Data.** `judge/suite_builder/build_suites.py` rebuilds both suites from pinned public sources:
- the Qwen3 tokenizer;
- SQuAD and GSM8K;
- KVzip's preprocessed SCBench;
- RULER;
- the KVzip essays at a pinned commit;
- the three hidden needles, committed under `judge/private/`.

Seeds are fixed and differ between the public and hidden suites. The content sha256 of both
suites is frozen in `judge/suite_manifest.json`, and every build checks it. The hidden suite is
never committed or published. It exists only in the Judge's private image layer, built on the
host. Sizes and sources follow your section 4.1 exactly (1,812 rows, 6,956 questions; Part 3
has the per-workload counts).

**Baselines.**
- `baselines/knorm`: kvpress `KnormPress`.
- `baselines/fastkvzip`: the kvpress `FastKVzipPress` gate with your window rule, using the
  released `qwen3-8b/q4_dim16_sink16.pt`.

**Framework integration.**
- Task JSON with version enumeration (`scorers/scorer_vNNN`, selected by `FINAL_VERSION`) and a
  capability envelope.
- Launcher: `run.sh`, `check.sh`, `replay.sh`.
- A task-local SForge entry wrapper, `scripts/sforge_kvzip_entry.py` (decision 36).
- The restricted CONNECT proxy sidecar for Work, shared with the other tasks.
- Per-version replay, `scripts/replay_candidates.sh`.
- Registry rows: `scripts/infrathon/tasks.json` and its README table, the top-level README,
  `launch/README.md`, `sforge/harness/resource_profiles.json`, and the framework test's list of
  expected registrations (`tests/sforge/harness/test_latest_task_store_integration.py`).
- Version-store registration.
- The integrity policy `llm_judge/POLICY.md`.
- 29 CPU task tests.
- 19 planted submissions (`tests/planted/`, run by `scripts/run_planted.sh`).

No file under `sforge/` was changed except the resource profile entry.

## 2. Decisions made for you

Each item: what I decided, why, and what changing it costs.

### Interface and protocol

1. **One `score()` call per context, with all layers at once**, rather than a per-layer callback
   during the prefill.
   - Why: one timed call per context keeps the timing simple, and cross-layer calibration is
     natural with all layers in hand.
   - Cost: up to 23.7 GB of staging memory per Judge GPU at 32k tokens (measured, Part 3). A
     per-layer API would save that memory and let scorers overlap with the prefill.
   - To change: a new runtime API and a new timing rule.
2. **"Hidden states" means each layer's attention input after its input RMSNorm.** That is what
   Fast KVzip's gates consume (kvpress' forward hook). The residual stream before the norm is not
   given, because it would add 9.4 GB of staging at 32k. Adding it is a one-field API change.
3. **The states include the protected prefix** (positions before `context_start`), because the
   scorer "sees what the model computes during the prefill". Scores cover only the context. The
   prefix length differs between the QA and GSM8K instructions, so a scorer can tell the two
   prompt types apart. I judged that harmless: it is not token identity at a position.
4. **Scores are float32** ("exact dtype"). kvpress computes KNorm in float32 for the same reason:
   in bf16 most scores tie. The shape is `[36, 8, n_ctx]` on `cuda:0`, all values finite.
5. **Questions are appended exactly as your text says.**
   - The sequence is `\n\n{question.strip()}`, then
     `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`, then the RULER answer prefix
     with no separator (as in kvpress).
   - KVzip's own code prefixes questions with `Q: `, and its GSM8K query adds a "The answer is"
     instruction. I followed your text, not the code.
   - The system turn is KVzip's (`You are a helpful assistant.`).
   - The prefix, context, question, postfix and answer prefix are tokenized separately and
     concatenated, as KVzip does.
   - Stop tokens are `<|im_end|>` and `<|endoftext|>`.
6. **Masked decoding uses my own Triton kernel.**
   - It attends over the shared prefilled cache with a per-row retention mask (the prefix is
     always visible), plus each element's own appended tokens. Scores and softmax are float32,
     operands bf16.
   - All four settings of all questions of a row are decoded in one batch (chunks of up to 128
     elements) with CUDA graphs. The full cache goes through the same kernel with an all-ones
     mask.
   - Why not SDPA: it cannot take per-row masks over a shared cache without materializing them.
     SDPA is used for the prefills.
   - The prefill layer loop is HF's Qwen3 layer written out. Its logits and KV cache are
     bit-identical to `transformers` (measured). Greedy full-cache outputs match
     `transformers.generate` on the examples I compared.
   - Cost of changing the decode path: it changes every number and needs a new qualification.
7. **Determinism.** Re-running a submission gives the same score bit for bit. Rows are
   aggregated in id order, and the full-cache denominators are identical across submissions
   (measured). Measured repeats:
   - KNorm 24.2869 in six Judge runs: two qualification runs, a manual evaluation, a
     `sforge-submit` from Work, a trial's final evaluation and a per-version replay;
   - Fast KVzip 89.2526 in both qualification runs.

### Metrics (definition 4.2, where it was not fully specified)

8. **needle, scbench_kv, scbench_prefix_suffix, squad use KVzip's normalized inclusion.** SQuAD
   validation (public) has several gold answers, and any one counts. SQuAD train (hidden) has one
   per question.
9. **gsm "numeric inclusion".** The reference number (after `####`, commas removed) must appear
   as a number among the numbers of the answer: `5` does not match `15`, and `1,250.00` matches
   `1250`. KVzip uses a plain substring after "the answer is", but our prompt does not ask for
   that phrase. Changing it is a one-function change in `metrics.py`.
10. **scbench_repoqa.**
    - KVzip's tree-sitter sanitizer extracts the function from the first code block. The answer
      passes if its BLEU (method 4) against the reference function is at least 0.8.
    - An empty answer fails (KVzip skipped empty answers).
    - KVzip also requires the reference to be the best match among the row's five functions. I
      followed your wording instead.
    - A missing tree-sitter is an error, not a silent fallback. Measured: without it, full-cache
      RepoQA accuracy drops from 0.78 to 0.46.
11. **The remaining metrics.**
    - scbench_mf: exact match after trimming whitespace.
    - many_shot: KVzip's rule (the letter in the first parenthesis, else option-text inclusion).
    - summary: ROUGE-L F from the `rouge` package, as in KVzip.
    - RULER: kvpress' string match (control characters stripped, case-insensitive); all
      references must be present for niah and vt, any one for qa.

### Data

12. **Essays are concatenated in file-name order.** KVzip concatenates them in `glob` order,
    which depends on the file system. With sorted order the hidden grid yields 99 candidate rows,
    and exactly 96 after dropping same-boundary duplicates, as your table says.
13. **Generated SCBench formats.** Each matches the upstream `_short` files byte for byte in
    template.
    - *kv*: 361 to 365 UUID4 pairs as `json.dumps`, five consecutive pairs per row at a
      stratified depth. They are asked as five key-to-value and then five value-to-key questions
      with the upstream templates, including the upstream's missing closing quote. Your text says
      "ask for 10 keys"; the upstream file asks 5 values and 5 keys, and you said to follow the
      upstream exactly.
    - *prefix-suffix*: 1,004 to 1,010 words of 25 characters from the upstream alphabet. Five
      questions; the answer is the latest match.
    - *Math.Find*: 6,002 to 6,008 integers uniform in `[lo, hi]`, with `lo` in 1..99 and `hi` in
      100..999 (the upstream ranges). The six questions are shuffled per row.
14. **RULER draws** use a seeded `random.Random` per (task, length), and the hidden draws exclude
    the public rows.
    - Consequence: both public `ruler_qa` rows have full-cache accuracy 0 under RULER's string
      match ("5 million" against "five million"; "Salford City" against "Salford City Football
      Club").
    - The hidden `ruler_qa` full-cache accuracy is 0.71.
    - The public runner leaves a zero-full workload out of its diagnostic score and says so.
    - I kept the draw rather than cherry-pick rows.
15. **Suite identity is the sha256 of the uncompressed JSON lines**, because gzip bytes can vary
    with the zlib version. A host build and a container build gave the same identity (measured).

### Checks (definition 4.3)

16. **Cost.**
    - Each call is timed by the harness, from sending the request to receiving the reply. The
      scorer process sends the reply after synchronizing its device. The time includes the IPC
      round trip (about 0.2 ms).
    - "Plain prefill" is one synchronized timing of the same layer loop without copies, per
      context, on the same GPU.
    - The shortest prefill is the minimum over the suite (measured 14.4 ms, so the floor is
      144 ms).
    - Two untimed warm-up calls on fixed filler text (200 and 4,000 tokens) come first. A fresh
      CUDA process pays about 200 ms on its first kernels (measured), which would break the
      floor for short contexts.
    - The per-context rule also fails fast during the run, but only where that is provably
      sound: each GPU processes one of the shortest contexts first, so the floor is known before
      any long context is judged.
17. **Inputs unchanged.**
    - Keys and values are compared exactly with the KV cache.
    - Hidden states and queries are compared through a secret random projection of their bit
      patterns, computed before and after the call (full copies would not fit in memory).
    - The global RNG states (torch CPU and CUDA, Python, NumPy) are compared inside the scorer
      process.
18. **No other GPU.** Two mechanisms:
    - no other process may run under the scorer's uid (process table, checked after every row);
    - nvidia-smi samples, taken every 2 s, must show at most 2 compute processes per GPU. This
      fails fast.
19. **Memory** (revised after trial round 2, Part 6). There are three limits, each with its own
    rejection message, and the public tests use the same ones:
    - **The rule:** the scorer process's peak allocation, `torch.cuda.max_memory_allocated`, is
      at most 8 GiB. Its weights count; the shared input buffers do not. This check fails fast
      after the offending call.
    - **PyTorch's cache in the scorer process is capped at 16 GiB of reserved memory**
      (`set_per_process_memory_fraction`, set before the submission is imported). The
      allocator releases cached blocks before it reaches the cap, so fragmentation cannot grow
      the process without bound. An allocation that still does not fit raises
      `OutOfMemoryError`, both in the Judge and in the public tests.
    - **The whole process may hold at most 19 GiB of device memory**, sampled from outside with
      nvidia-smi every 2 s. This covers the cache, the CUDA context and anything allocated
      outside PyTorch. An honest scorer cannot exceed it: the cache is capped at 16 GiB and the
      context takes about 0.8 GiB. So the limit catches memory hidden from PyTorch's
      statistics (planted `outside_pytorch_memory`, 20 GiB through the CUDA driver API). It
      also fails fast.
    - What was there before: the first version compared the process footprint with 8 + 2 GiB,
      a limit the brief did not mention, under a message about the allocation peak. It
      rejected three honest trial submissions, whose peak allocation was 2.5–3.5 GiB but
      whose PyTorch cache had grown to 12–17 GiB from fragmentation.
    - The 19 GiB keeps a Judge GPU safe: the harness peaks at about 50 GB, and 50 GB plus 20 GB
      is under 80 GB.
20. **"Real work on every call" and no caching.**
    - Every call is timed and every result is used, so there is no untimed call to behave
      differently in.
    - The scorer is frozen between calls. Its output buffer is re-checked after the row is
      decoded, to catch changes after return.
    - Two contexts per GPU are re-scored after later contexts, once after 8 rows and again at the
      end, and must keep at least 99 % of their retained entries.
    - A scorer that caches results would pass this probe. Caching gains nothing, though, because
      no hidden context repeats.
21. **Weights** (revised after trial round 1, Part 5).
    - Every weight file must load with safetensors, `torch.load(weights_only=True)` or numpy
      without pickles. Unreadable weights are rejected.
    - Every tensor yields candidate rows of 4,096 values, which are compared with the 23,040
      released q/k projection rows. Candidate rows come from:
      - flat order, aligned at the tensor's start and at its end (any order-preserving
        reshape, or a few elements of padding);
      - the first and last 4,096 values of each row, and of each column, when they are wider
        than 4,096 (a gate layer with two extra inputs is 4,098 wide);
      - for 8-bit tensors: signed and unsigned values, packed int4 (both nibble orders, offset
        and two's complement) and raw fp8 bytes;
      - for 16-bit integer tensors: their bits read as fp16 and bf16.
      Non-finite values count as 0, so NaN poisoning does not hide a row.
    - A row matches at **rank (Spearman) correlation** ≥ 0.8. The submission is rejected when
      at least 64 released rows are matched by at least 64 distinct candidate rows.
    - Why ranks: the first version used Pearson correlation and rejected two honest trial
      submissions. Qwen3's attention input concentrates most of its energy on three
      massive-activation coordinates, and so do many released rows. A plain mean of hidden
      states therefore reached Pearson 0.92 on 84–90 released rows. Its rank correlation is 0.22.
    - Measured, all caught (rank r 0.99–1.0): renamed, full gate, transposed, reshaped,
      flat-offset, int8 per-row, packed int4 (both encodings), raw fp8 bytes, the fp8 dtype,
      "fine-tuned" (+30 % noise), padded to 4,098 or 4,099 inputs, NaN-poisoned, and bf16
      bits stored as int16.
    - Measured, none caught:
      - every weight file of all 18 trial submissions (max 0.47: the agent's own
        reconstruction gates);
      - Qwen3-8B's own q, k, v, gate and up projections and 20,000 embedding rows (max 0.52);
      - random weights (0.09).
    - The scan covers every version in `scorers/`, not only the evaluated one, so a gate parked
      in an old version still rejects.
    - Not covered: a copy whose columns are permuted, or whose rows are mixed by a rotation.
      Those are caught only by the integrity review (Part 4).
22. **Admission.** Not in your list, but it follows from the weights rule and the archive limit.
    - `FINAL_VERSION` holds one integer from 1 to 999, and `scorers/scorer_vNNN/scorer.py`
      exists.
    - `scorers/` holds only `scorer_vNNN` directories.
    - Only regular files are accepted: text and source at most 1 MiB each, weights only in the
      formats above, and no other binaries (no prebuilt `.so`; kernels are built from submitted
      source).
    - Without this, the weights scan could be evaded with an unknown binary format.
23. **Sanity.** A zero full-cache accuracy or a non-finite metric makes the evaluation an
    infrastructure error: no score and nothing charged to the candidate, rather than a valid
    result or a verdict on the candidate. It never fired on the hidden suite; the lowest
    full-cache workload accuracy is Math.Find at 0.32.
24. **Feedback is exactly your 4.4 list.**
    - It contains: the score; the ratio scores (100 × ratio(r)); the group scores (100 × mean
      over r of group(g, r)); the overhead ratios (total scoring over prefill, the worst context
      against its own limit, and the count of contexts over their limit); the state of every
      check; and the reason for a rejection.
    - The scorer's peak allocation and the Judge's wall time go only to the private artifact.
      The agent gets both for the public suite from the public tests.
    - Every word of a rejection reason is chosen by the trusted harness, never copied from the
      scorer process (decision 39): exception type names from a fixed list, fixed texts for
      output problems, no reported values. Messages and tracebacks go to the private artifact.
      The public tests show all of them.
    - The rejection reason never names a context.
    - An invalid submission gets no ratio or group scores, even when it was rejected after the
      whole suite ran (trial round 2 showed agents using those numbers). Overhead ratios stay,
      because they explain cost rejections.

### Rules (definition section 1) and the brief

25. **"No weights someone else trained" also covers the frozen model's own weights** inside the
    scorer. The scorer cannot read them in the Judge anyway. The brief forbids shipping copies of
    them, and also forbids using the released gates as a training teacher. Your text says "the
    scorer only sees what the model computes"; I read copying the weights in as a way around
    that.
26. **What the brief adds for the agent**, each traceable to one of your rules:
    - no background threads or processes, and no work left running after return;
    - no tampering with timers or memory statistics;
    - read only the submitted tree.

    The integrity policy mirrors the brief line by line.
27. **The brief says the hidden suite is "much larger and broader … more and different task
    types in the same three groups, and other sources".** It names neither the hidden workloads
    nor their sizes. Since trial round 1 the Work image does not name them either: its runtime
    used to list them (Part 5, fix 4). The layout validator now enforces this.

### Fast KVzip baseline

28. **The Fast KVzip baseline is evaluated as a trusted reference.**
    - The problem: the released gate (189 MB) is larger than the framework's 128 MiB archive
      limit, and the weights check would reject it anyway. So the baseline archive holds only the
      scorer code.
    - How it runs: the host asks the Judge for this one trusted reference through the private
      evaluation context (`eval_archive.sh --reference-baseline fastkvzip`). The Judge copies the
      gate into the scorer's tree and reports the weights check as `fired_reference_baseline`
      instead of rejecting. Every other check runs as usual.
    - The integrity review is skipped for it, because the review would correctly reject it.
    - The window rule is yours: the first 4 context positions and the last `max(1, int(0.02 n))`
      positions (4,096 at n ≥ 32,000) are set to 1.

### Resources, timing, operations

29. **Resources.**
    - Judge: 7 GPUs, 56 CPUs and 88 GiB (your "8 CPUs and 8 GiB per GPU plus 32 GiB").
    - Judge timeout: 3,600 s, with an internal deadline of 3,300 s.
    - Work: 1 GPU, 8 CPUs, 32 GiB.
    - Submission cooldown 300 s; budget 24 h (the launcher default). The Work container gets
      the run's actual deadline (`KVZIP_DEADLINE_UNIX`), so a shorter run is not told 24 hours
      (Part 5, fix 6).
30. **One submission in flight.** `sforge-submit` waits for the result. While one submission
    is queued or running, the server refuses another with HTTP 409; the trial showed this, and
    the first brief wrongly said it would wait. The Judge lock also serializes evaluations
    across runs that share a Judge. The brief says all of this, and that a result takes about
    15 minutes after the review.
31. **At the deadline** the workspace is frozen as `final_archive.tar.gz` and judged once more.
    That result counts like a submission (framework convention: the last state is work done
    within the budget).
    - The snapshot needs every submitted path to exist. Measured: an earlier layout whose
      starting workspace lacked one path made the final snapshot fail.
    - The starting workspace now has all three paths, and the brief says to keep them in place.
32. **Work's network.** Work is on an internal Docker network. Its only way out is the restricted
    CONNECT proxy sidecar the other tasks use: public hosts on ports 80 and 443, internal
    addresses refused, and the Judge port relayed. The Judge has no network at all. The brief
    says nothing about internet access, as the guide requires.
33. **Naming.** The directory and task id use `v2`, matching the version store the tool provided.

### Submitted tree and framework

34. **A versioned submitted tree** (`scorers/scorer_vNNN/` + `FINAL_VERSION` + `notes.md`)
    rather than a single `scorer/` directory.
    - Why: the framework's test of the latest tasks requires this shape (a code root,
      `FINAL_VERSION`, `notes.md`). For tasks that enumerate versions, it also requires
      per-version replay wiring.
    - Benefits: you can re-judge every version an agent made, and training code stays next to the
      weights it produced.
    - Cost: the agent manages one more file. An evaluation copies only the selected version, but
      the weights scan covers all versions.
    - To change back: the framework test would need an exemption.
35. **Per-version replay.** `launch/kvzip-kv-scoring/replay.sh` (which runs
    `scripts/replay_candidates.sh`) re-judges a closed run's saved candidates. With
    `--per-version` it re-judges every distinct `scorer_vNNN` in them instead. For a replay, the
    Judge takes the version from the host's private evaluation context (`candidate_version`)
    instead of `FINAL_VERSION`. Measured on the smoke-3 trial: v001 (the stub) is invalid and
    v002 (KNorm) scores 24.2869, as in every other KNorm run (Part 3.9).
36. **Task-local entry wrapper** (`scripts/sforge_kvzip_entry.py`) around SForge's Docker
    backend.
    - Work gets exactly its one GPU, the model mount and the internal network.
    - The Judge gets its seven GPUs, the model under a root-only path, no network and
      no-new-privileges. The seven GPUs cannot go through the framework's own route, because its
      Judge environment option is comma-separated and so cannot carry a device list.
    - Images are recognized by their task tags. A replay pins the image by its digest, so a
      digest is classified by the tags it still carries. I found and fixed this during the
      replay test.
    - It changes no framework file, but patches one method (`DockerBackend.create_container`)
      that has to be kept in step with the framework.
37. **`check.sh` evaluates through `sforge eval`.** That command runs the Controller's integrity
    review (as configured by `SFORGE_LLM_JUDGE_*`) and then the same `judge_submission` path the
    Judge server uses for `sforge-submit`.
    - With the review on, the review may stop the starting stub before the Judge; `check.sh`
      accepts either outcome for the stub.
    - KNorm must pass the review.
    - Fast KVzip skips the review (decision 28).
    - The trial path itself (a Judge server, `sforge-submit` from Work, the final snapshot) was
      tested in three short trials (Part 3).
38. **Measurement profile.** The capability envelope asks for 7 GPUs of compute capability ≥ 8.0
    with ≥ 64 GiB each. Scores are compared only within one measurement profile. The registry
    marks H100 80 GB SXM (700 W) as qualified and H100 NVL (94 GB, 400 W) as untested. The cost
    rule is relative (scoring against prefill on the same GPU), so it should transfer to other
    GPUs; bf16 greedy decoding, and so the score, may differ slightly across GPU types.
39. **The scorer process is hostile, and so is everything it says.** The harness (root in the
    Judge) and the scorer process exchange length-bounded JSON objects (`kvzip_eval/ipc.py`).
    - Replies are never unpickled. An earlier version used Python's `multiprocessing`
      `Connection.recv()`, which unpickles. A scorer could have found its own channel and sent a
      pickle that runs code as root in the Judge. I found this in my own audit and replaced it.
      The planted `forged_reply` case sends such a pickle: it is refused as "the scorer process
      sent an unexpected reply", and nothing runs.
    - Replies may hold only known fields of the right types (status, exception type name, output
      problem kind, peak bytes, RNG flag, init seconds, private detail). Anything else rejects.
    - Agent-visible text is chosen by the harness: exception names only from the builtin,
      PyTorch, NumPy and Triton exception classes, otherwise "an exception of a type outside
      Python, PyTorch, NumPy and Triton". Output problems use a fixed table (no shapes, dtypes or
      class names). The memory rejection has no number, because the scorer reports its own peak.
    - Without this, a scorer could encode hidden text in an exception class name or a returned
      shape and read it back in the rejection reason. The planted `leaky_exception_name` and
      `leaky_shape` cases try exactly that and get only the fixed texts.
    - A hostile scorer can still choose *which* allowed reason it triggers: a few bits per
      submission, like the timing channel (Part 4).
    - The scorer's log is read with `O_NOFOLLOW` (the scorer owns its scratch directory), and only
      into the private artifact.

40. **No training on the benchmark's own data** (added after trial round 2; your definition
    only forbids answer tables).
    - The rule: learned components may not be trained, tuned or validated on SQuAD, GSM8K, the
      RULER or SCBench data, or the Paul Graham essays, in any split. Their own generated tasks,
      also of the same kinds, and any other text are fine. The public suite is for checking
      and selecting.
    - Why: the hidden suite draws its rows from those releases (SQuAD train, GSM8K test and
      train, RULER and SCBench rows, the essays), all downloadable from the Work container.
      Training on them is contamination that no mechanical check can see.
    - Enforcement: the brief and the integrity policy (training code and notes); no
      mechanical check.
    - Neither trial agent did it. Both used WikiText-2, Project Gutenberg, Nemotron web-text
      samples and their own synthetic tasks.
    - Cost: agents need other data, which both trials found easily. To drop the rule, delete
      it from the brief and the policy.

## 3. Measurements

### 3.1 Baselines on the hidden suite (real Judge, 7 × H100 80 GB SXM)

| submission | valid | score | ratio 0.1 / 0.2 / 0.3 | retrieval / contextual QA / redundancy | scoring / prefill | worst context / its limit | Judge time |
|---|---|---:|---|---|---:|---:|---:|
| starting stub | no (`NotImplementedError` in the warm-up call) | 0 | | | | | 15–20 s |
| KNorm | yes | 24.2869 | 23.33 / 23.37 / 26.16 | 0.31 / 20.08 / 52.47 | 0.54 % | 0.06–0.10 | 758–766 s |
| Fast KVzip | yes (weights check `fired_reference_baseline`) | 89.2526 | 72.23 / 95.08 / 100.45 | 89.84 / 78.49 / 99.42 | 8.1 % | 0.48–0.53 | 776–782 s |

Every other check passed on both baselines, and every statelessness probe kept 100 % of the
retained entries.

Development runs on the host, before tree-sitter was made mandatory (decision 10), gave 24.36
and 89.86. The difference came from RepoQA alone.

### 3.2 Per workload (from the Judge's private artifact)

Full-cache accuracy is identical for both baselines (same suite, same decoder). The other columns
are rel(task, r) = acc(r) / acc(full).

| workload | group | full-cache acc | KNorm 0.1 / 0.2 / 0.3 | Fast KVzip 0.1 / 0.2 / 0.3 |
|---|---|---:|---|---|
| needle | retrieval | 0.760 | 0.000 / 0.000 / 0.000 | 1.110 / 1.123 / 1.082 |
| ruler_niah | retrieval | 0.991 | 0.000 / 0.000 / 0.000 | 0.874 / 0.976 / 0.991 |
| scbench_kv | retrieval | 0.886 | 0.000 / 0.000 / 0.000 | 0.481 / 0.920 / 1.037 |
| scbench_prefix_suffix | retrieval | 0.597 | 0.000 / 0.000 / 0.000 | 0.357 / 0.761 / 1.020 |
| scbench_repoqa | retrieval | 0.782 | 0.012 / 0.012 / 0.023 | 0.837 / 0.930 / 0.977 |
| gsm | contextual QA | 0.885 | 0.017 / 0.017 / 0.079 | 0.085 / 0.808 / 0.972 |
| ruler_qa | contextual QA | 0.711 | 0.352 / 0.370 / 0.389 | 0.861 / 0.963 / 0.991 |
| squad | contextual QA | 0.894 | 0.153 / 0.178 / 0.251 | 0.429 / 0.970 / 0.986 |
| ruler_vt | redundancy | 1.000 | 0.000 / 0.000 / 0.000 | 1.000 / 1.000 / 1.000 |
| scbench_many_shot | redundancy | 0.441 | 1.336 / 1.076 / 1.059 | 1.008 / 0.992 / 0.975 |
| scbench_mf | redundancy | 0.317 | 0.074 / 0.238 / 0.341 | 1.044 / 1.034 / 1.065 |
| scbench_summary | redundancy | 0.427 | 0.684 / 0.728 / 0.762 | 0.855 / 0.960 / 0.997 |

What the table shows:
- KNorm loses every retrieval workload and variable tracking at every ratio. Its score comes
  from many-shot (above 1, uncapped as specified), summary and ruler_qa.
- Fast KVzip's weak spots are the short contexts at r = 0.1: GSM8K at 0.085 and SQuAD at 0.43.
  There the budget is a few dozen entries per head and the trailing window is 1 to 10 positions.
  Long-context KV and prefix-suffix retrieval are also weak at 0.1. This is the headroom the
  task leaves.
- Fast KVzip's 100.45 at r = 0.3 comes from rel > 1 on needle, kv, mf and prefix-suffix (no
  cap, as specified).

### 3.3 Hidden suite as built

This information is for you only; the brief does not give it.

| workload | rows | questions | context tokens min / median / max | plain prefill, s (sum) |
|---|---:|---:|---|---:|
| needle | 96 | 96 | 499 / 1,999 / 8,000 | 13 |
| scbench_kv | 150 | 1,500 | 24,320 / 24,567 / 24,757 | 190 |
| scbench_prefix_suffix | 150 | 750 | 18,746 / 18,883 / 19,038 | 132 |
| scbench_repoqa | 22 | 110 | 9,452 / 14,936 / 20,775 | 14 |
| ruler_niah | 192 | 192 | 7,497 / 15,647 / 21,043 | 104 |
| squad | 250 | 1,890 | 70 / 172 / 502 | 4 |
| gsm | 200 | 200 | 72 / 84 / 171 | 3 |
| ruler_qa | 152 | 152 | 5,780 / 12,239 / 17,215 | 70 |
| scbench_mf | 250 | 1,500 | 24,157 / 29,409 / 30,026 | 403 |
| scbench_many_shot | 54 | 270 | 18,614 / 25,808 / 26,474 | 66 |
| scbench_summary | 62 | 62 | 9,381 / 13,130 / 18,315 | 35 |
| ruler_vt | 234 | 234 | 7,872 / 16,273 / 16,285 | 120 |
| **total** | **1,812** | **6,956** | 70 / – / 30,026 | **1,155** |

Rows and questions match your table exactly. My generated Math.Find rows reach 30,026 tokens,
against the 29,991 your table gives; the generated rows come from my seeds, and the difference is
within the 32,000-token limit. The public suite matches your section 3: 63 rows, 419 questions.
The public and hidden suites share no context (task test).

### 3.4 Public suite (Work image, one H100)

- Fast KVzip, `full` tier: diagnostic public score 84.9 in 3.5 minutes. The brief says "about
  4 minutes".
- KNorm, `quick` tier: 13.06, all checks pass.
- Starting stub, `one` tier: rejected for `NotImplementedError`.

### 3.5 Timing

- **Prefill on one H100** (the Judge's layer loop, bf16):

  | context tokens | 128 | 1k | 8k | 16k | 24.5k | 32k |
  |---|---|---|---|---|---|---|
  | prefill time | 13.6 ms | 35 ms | 320 ms | 743 ms | 1.29 s | 1.89 s |

- **Decode** with CUDA graphs: 13–24 ms per step for a whole batch, from short contexts to 32k.
- **Judge wall time inside the container:**
  - stub: 15–20 s;
  - KNorm: 758–766 s over six runs;
  - Fast KVzip: 776–782 s over two runs.

  Container start and the weights scan add about 20 s. The brief says "about 15 minutes".
- **Worst case (estimated, not measured).** Suppose every compressed answer ran to its
  workload's token limit. Decoding would then take about 277,000 batch steps in total. At 13 ms
  (short contexts) to 20 ms (long) per step, that is about 4,400 GPU-seconds, or about
  10.5 minutes per GPU. With prefills and model loading it comes to about 18 minutes, inside
  the 55-minute internal deadline and the 60-minute Judge timeout. KNorm, whose compressed
  answers are mostly wrong, took no longer than Fast KVzip.
- **Cost rule inputs:**
  - total plain prefill of the hidden suite: 1,155 s (summed over GPUs);
  - shortest prefill: 14.4 ms, so the per-context floor is 144 ms;
  - total scoring allowance: about 289 s.
- **Scoring time used:**
  - Fast KVzip: 93.3–93.8 s in total (8.1 %). Its worst context reaches 0.48–0.53 of its limit
    (run-to-run timing noise). In both runs it is a short context where the 144 ms floor
    applies: a GSM8K context in one run, a 2,000-token needle context in the other.
  - KNorm: 6.3 s (0.54 %).

### 3.6 Memory

- **Judge, per GPU:**
  - harness, PyTorch reserved: 26.2–26.4 GiB;
  - VMM staging buffers: 23.7 GB at 32k;
  - device peak: 51.2 GB of 80 GB.
- **Scorer process, peak allocation:**
  - KNorm: 0.089 GiB;
  - Fast KVzip: 0.65 GiB;
  - KNorm's NVML footprint, CUDA context included: about 0.8 GB.
- **Work:** the public tests run the same harness on the one Work GPU, and the full tier fits.

### 3.7 Weights detector

See decision 21. Measured with the rank-correlation version:
- every disguised copy of the released gates is caught (rank r 0.99–1.0);
- the trial's honest weight files peak at 0.47 and are not caught;
- Qwen3-8B's own weights peak at 0.52;
- random weights reach 0.09.

Under the first, Pearson-based version, Qwen3-8B's own q/k projections reached 0.58, and plain
means of the model's hidden states reached 0.92. That is the false positive found in the trial.

### 3.8 Planted submissions (real Judge, hidden suite)

Each case was written to trigger one check (`tests/planted/README.md`, `scripts/run_planted.sh`).
- All 19 were rejected for the intended reason (`outside_pytorch_memory` was added after trial round 2).
- The 15 original cases ran first (first pass, before the scorer channel was hardened).
- After the hardening, 11 cases ran again: the three new ones, plus every case whose reason text
  changed or that touches the channel (second pass).
- The reasons below are what the agent sees now.

| planted | expected | agent-visible reason | time |
|---|---|---|---:|
| bad_admission (a `.so` file) | admission | `submission layout: ...fast_kernels.so: file type .so is not accepted` | 22 s |
| released_gates_fp8 | weights | `submitted weights match the released Fast KVzip gates` | 28 s |
| init_raises | construction | `Scorer construction failed: ValueError` | 41 s |
| bad_dtype | output | `score dtype is not torch.float32` | 41 s |
| nan_output | output | `score tensor has non-finite values` | 40 s |
| modifies_inputs | side effects | `Scorer.score() modified its input tensors` | 42 s |
| global_rng | side effects | `Scorer.score() changed the global random number generator state` | 41 s |
| spawns_process | one process | `the scorer started other processes` | 48 s |
| other_gpus | one GPU | `the scorer started other processes` | 51 s |
| slow | cost (per context) | `a scoring call exceeded its per-context time limit` | 48 s |
| memory_hog (9 GiB) | memory (allocation) | `the scorer's peak allocation (torch.cuda.max_memory_allocated) exceeds 8 GiB` | 48 s |
| outside_pytorch_memory (20 GiB through the CUDA driver API, round 2) | memory (whole process) | `the scorer process held more than 19 GiB of device memory in total (...)` | 41 s |
| stateful | stateless | `re-scoring a context gave different retained entries ...` | 81 s |
| probe_hidden_files | cannot read Judge files | `... raised PermissionError ...` (type only) | 37 s |
| probe_network | no network | `... raised OSError ...` (type only) | 41 s |
| probe_frozen_between_calls | frozen between calls | `Scorer.score() raised KeyError` (type only) | 59 s |
| forged_reply (pickle on the scorer's channel) | channel | `the scorer process sent an unexpected reply`; the pickle never ran | 38 s |
| leaky_exception_name (class name carries data, hidden contexts only) | no free text | `Scorer.score() raised an exception of a type outside Python, PyTorch, NumPy and Triton` | 42 s |
| leaky_shape (shape carries data, hidden contexts only) | no reported values | `score shape is not [36, 8, context tokens]` | 42 s |

Which pass each case was last run in:
- Second pass: forged_reply, leaky_exception_name, leaky_shape, bad_dtype, nan_output,
  init_raises, memory_hog, global_rng, modifies_inputs, probe_hidden_files, slow.
- First pass, then re-checked after the move to the versioned layout: bad_admission,
  released_gates_fp8.
- First pass only: spawns_process, other_gpus, stateful, probe_network,
  probe_frozen_between_calls. Those checks did not change.

Planted cases do not cover:
- the total-cost rule (25 %): only the per-context rule fired;
- the NVML memory cross-check;
- the output-changed-after-return check;
- the infrastructure-error path for a zero full-cache accuracy.

The total-cost rule shares its timings and code path (`cost_check`, unit-tested on CPU) with the
per-context rule.

### 3.9 Trial path (three short trials with a dummy agent credential)

`run.sh` ran three times with a 10-minute budget. Each run started the Judge server, the internal
network, the sidecar and the Work container. At the deadline it took the final snapshot, judged
it, and wrote `completion.json`.

The agent could not authenticate with the dummy credential, so I acted as the agent through
`docker exec`.

- **smoke-1.** KNorm submitted with `sforge-submit` from inside Work, through the Controller and
  the Judge: valid, 24.2869.
  - The feedback carried the score, ratio and group scores, overhead ratios and check states. At
    that time it also carried the peak allocation and the Judge's time; both have since moved
    to the private artifact (decision 24).
  - The current result fields are confirmed in the replay below.
  - The final snapshot then failed, because the starting workspace lacked one submitted path
    (decision 31). I fixed that.
- **smoke-2.** Final snapshot and final evaluation of the unchanged starting workspace: invalid,
  `NotImplementedError`, and `completion.json` written.
- **smoke-3** (versioned layout). I added `scorers/scorer_v002` (KNorm), set `FINAL_VERSION` to 2
  and saved the candidate with `sforge-save-version`. The final archive was judged: valid,
  24.2869, `completion.json` best valid 24.2869.
- **Per-version replay of smoke-3**
  (`replay.sh --origin-run smoke-3 --measurement-set test --per-version`):
  - v001: invalid, `NotImplementedError` (20 s);
  - v002: valid, 24.2869 (765 s). Its result holds only the definition's fields: score, ratio
    and group scores, overhead ratios, check states and reason, plus the score's name, unit and
    formula;
  - the results are written to `versions_curve.json`.

  The first attempt failed: the replay pins the Judge image by its digest, which the task entry
  could not classify (decision 36). I fixed that and re-ran it.

### 3.10 Qualification runs (`check_cmd` from a clean clone)

- `check-1` at an earlier HEAD (before the versioned layout): PASSED in about 30 minutes.
- `check-4` at `df5cadc`, after the trial-2 fixes: PASSED in 30 minutes, with the same scores (Part 6).
- `check-3` at `4894250`, after the trial-1 fixes: PASSED in 29 minutes, with the same scores (Part 5).
- `check-2` at `3fc64ae` (the commit trial round 1 ran): PASSED in 29 minutes. It ran from a fresh
  `git clone /work/repo`, after `uv sync --frozen --dev`, with `SFORGE_LLM_JUDGE_MODE=off`.
  - images current;
  - 111 framework tests passed, 27 task tests passed;
  - public tests: the stub was rejected; KNorm `quick` scored 13.06, all checks pass;
  - Judge: stub invalid in 42 s, KNorm 24.2869 in 786 s, Fast KVzip 89.2526 in 805 s. These
    wall times include the container start.

`check.sh` builds the images (idempotent), then runs:
- the task tests and the framework tests that cover the task's registry entries;
- the public tests: the stub must be rejected, and KNorm `quick` must pass;
- the stub, KNorm and Fast KVzip through the Judge, comparing them with the definition's
  expectations:
  - stub invalid with score 0;
  - both baselines valid;
  - Fast KVzip − KNorm ≥ 30;
  - KNorm retrieval ≤ 5;
  - Fast KVzip ≥ 75 and below 100 at r = 0.1;
  - the weights check fires on the gate.

## 4. Not done, not enforced, or open

**Rules enforced only by the integrity review (no mechanical check):**
1. **No token identity.** There is no mechanical check. The scorer gets no ids, text or
   vocabulary, but it could in principle learn an inverse of the hidden states. Review only.
2. **No weights someone else trained**, beyond the released Fast KVzip gates. The detector
   knows only the released gates. Weights downloaded from elsewhere, or trained with the released
   gates as a teacher, are caught only by the review and the agent's `notes.md`.
3. **No lookups of known answers, and no position tables built from benchmark data.** Review
   only. Hidden contexts never repeat, which limits the value of such tables.
4. **No different behavior when not timed, and no caching across contexts.** Every call is
   timed and used. The statelessness probe catches state that changes results, but not a cache
   of identical results (decision 20).

**Mechanical checks with known limits:**
5. **The RNG check and `torch.cuda.max_memory_allocated` run inside the scorer's own process.**
   Hostile code could patch them.
   - Mitigations: the whole-process limit measured from outside (19 GiB, decision 19), and the
     review rule against tampering with timers, memory statistics or the runtime.
   - Not airtight: a scorer that fakes its allocation statistics can use up to the 19 GiB
     process limit undetected, beyond the 8 GiB rule.
6. **GPU work after return.** The scorer process synchronizes its device before replying and is
   stopped right after the reply.
   - A scorer could enqueue work in the short window between the reply and the stop.
   - Changing its output after return is caught: the output is re-checked after decoding.
   - Extra work would only slow the Judge's decoding, and the review forbids it.
7. **Side channels through feedback.** A hostile scorer could encode a few bits of hidden
   information in its timing (and so in the overhead ratios the definition returns), or in
   which of the allowed rejection reasons it triggers. Free text and reported values are closed
   off (decision 39). The 5-minute cooldown and the review limit this. The score itself is the
   same kind of channel and cannot be closed.
8. **Reading files.** The scorer can read world-readable files: installed packages and system
   files. That is allowed by design. It cannot read the model, the hidden suite, the gate, the
   Judge code or the evaluation context (planted probe: `PermissionError`).
9. **The weights scan runs as root in the Judge**, on untrusted files, but only with safe
   loaders: safetensors, `torch.load(weights_only=True)` in PyTorch 2.8, and NumPy without
   pickles. A flaw in one of those loaders would compromise the Judge. Running the scan as an
   unprivileged user would need the released rows handed to it separately; I did not do that.

**Not exercised here:**
10. **The LLM integrity review was not run** (no review credentials in this environment). The
    policy passes the framework's policy tests. All runs here used `SFORGE_LLM_JUDGE_MODE=off`.
    With the review on, `check.sh` expects it to pass KNorm. A review that rejected KNorm would
    fail the check, which is what should happen. The review may stop the starting stub; the
    check accepts that.
11. **One real agent has run** (Part 5): codex, 6 hours. Before that I only drove the trial
    path with a dummy credential. `--agent claude-code` (the launcher's runpy shim) has not
    been run end to end. Like every launcher in this framework, `run.sh` needs an agent runtime: either
    `SFORGE_AGENT_RUNTIME_DIR`, or `<clone>/../agent_runtimes/codex-<pin>`, built with
    `launch/build-agent-runtime.sh`.
12. **H100 NVL (94 GB, 400 W)** is untested. It is registered as untested.

**Open questions for you:**
13. **"Ask for 10 keys" against the upstream's 5 + 5** (decision 13). Changing it is a generator
    change and a new hidden identity.
14. **The 144 ms per-context floor (10 × the shortest prefill) is tight** for scorers with large
    fixed Python overhead on the 70-token contexts. 36 layers × several small kernels fit easily
    (Fast KVzip uses up to 53 % of it). A scorer that loops in Python over layers and heads may not.
    This is the definition's rule; the brief states it.
15. **Public `ruler_qa` has a full-cache accuracy of 0** (decision 14). The public runner leaves
    it out of its diagnostic score.
16. **The prefix length differs between the QA and GSM8K instructions** (decision 3), so a
    scorer can tell the two prompt types apart.
17. **Judge memory:** 51 GB of 80 GB per GPU. GPUs with less than 64 GiB are excluded by the
    capability envelope.

## 5. Trial round 1 (codex, 6 hours) and what changed

The tool ran one trial from a clean clone of `3fc64ae`: codex (`gpt-6.1-sol`, effort max)
worked on the task for 6 hours through `trial_cmd`, with the integrity review on.

### What happened

- **The first attempt never started.** `trial_cmd` from a fresh clone exited after 1 second:
  `ERROR: no image build record; run scripts/build_images.sh`. The build record lived in the
  `outputs/` of the checkout where `check_cmd` had built the images, and the trial ran in a
  different checkout. The tool re-ran the trial inside the check's checkout.
- **The run itself.** 17 submissions; 15 valid. Every Judge run took 762–800 s, plus 24–70 s
  for the integrity review. The final snapshot reproduced the best score exactly (98.3487).

  | min after start | submission | score | ratio 0.1 / 0.2 / 0.3 |
  |---:|---|---:|---|
  | 3 | agent-1 | 42.59 | 36.0 / 43.2 / 48.6 |
  | 18 | agent-2 | 84.19 | 67.2 / 88.0 / 97.4 |
  | 33 | agent-3 | 90.29 | 75.2 / 95.3 / 100.3 |
  | 49 | agent-4 | 92.62 | 82.2 / 96.0 / 99.6 |
  | 63, 71 | agent-5, agent-6 | rejected: "submitted weights match the released Fast KVzip gates" | |
  | 106–231 | agent-7 to agent-13 | 93.15 to 94.88 | |
  | 250 | agent-14 | 97.35 | 90.6 / 100.5 / 101.0 |
  | 310 | agent-16 (best) | **98.35** | 92.8 / 100.8 / 101.5 |
  | 334 | agent-17 | 97.91 | |

- **The best submission is legitimate and measures what you intended** (scorer v020, read in
  full). It has four parts:
  - Per-layer MLP gates (4,096 hidden values + log length + relative position → 128 → 8
    heads). The agent trained them itself on its own reconstruction-attention targets: the
    frozen model replays WikiText-2 training prose and generated contexts. This is Fast
    KVzip's recipe redone from scratch.
  - A context-adaptive head budget: a ridge regression on mean hidden states, trained on
    answer-loss ablations of generated tasks.
  - Probe features from sampled context queries (also RoPE-shifted past the context end) and
    from value norms.
  - 128 learned "generic query" centroids per query head, rotated to the answer position and
    attended against the keys. They are fitted on query vectors of question/answer suffixes
    the model generated for synthetic tasks.

  Cost and memory: 15.6 % of prefill, worst context at 0.51 of its limit, 2.1 GiB peak. The
  integrity review passed every submission (confidence 0.97–0.99). The transcript shows:
  - no benchmark data (its only download was WikiText-2 train), no web search, and no
    released gates or KVzip code;
  - no hyperparameters chosen on the public suite; no token identity; no caching or timing
    tricks.
- **No loophole explains the jumps.**
  - 42.6 → 84.2 is the first trained gate.
  - 94.9 → 97.4 is the generic-query templates.
  - Each later step changes one learned component.

### Task problems found and fixed

1. **Fresh-clone trial start (infrastructure, blocking).** `run_fixed_budget.sh` now builds the
   images itself when the checkout has no build record, or a stale one. The build is
   idempotent and takes about 10 s when Docker already holds the layers. Verified with two
   2-minute trials from fresh clones (`smoke-r1`, `smoke-r1b`): each built, ran, took the final
   snapshot, judged it and wrote `completion.json`. Commit `3836737`.
2. **Weights check: false positive (major).** agent-5 and agent-6 were honest, and were
   rejected anyway.
   - Their only 4,096-aligned tensor was a 3 × 4,096 mean of attention inputs, used to center
     features.
   - Qwen3's massive-activation coordinates carry about 80 % of that vector's energy, and
     about 75 % of many released rows'. That gave Pearson 0.92 against 90 released rows.
   - The agent lost two submissions and a line of work, and never found the cause.
   - I re-judged the rejected agent-5 archive with the fix: it is **valid, 92.99**, better
     than the 92.62 the agent had at the time.
   - Fix: rank correlation plus a distinct-row count (decision 21). Tests include a fixture of
     real Qwen3 mean activations, on which the old rule fired for 84 released rows.
   - All 18 trial archives now pass (max 0.47), and all disguised copies are still caught.
     The planted fp8 copy still fires in the real Judge. Commit `3836737`.
3. **Weights check: blind spots (major).** While fixing it I found that tensors whose width is
   not a multiple of 4,096 were never compared at all. The agent's 4,098-wide gate layers
   were not scanned: the scan for agent-4 compared zero rows. So a released gate padded with
   two input columns would have passed. Raw fp8 bytes, two's-complement int4 and NaN-poisoned
   rows were also missed. All are now caught (decision 21). Commit `3836737`.
4. **The Work image named the hidden-only workloads (major).**
   - The read-only runtime's `protocol.py` listed `scbench_repoqa`, `scbench_many_shot` and
     `scbench_summary` with their metrics and token limits. `metrics.py` held their metrics,
     and their packages (rouge, tree-sitter) were installed.
   - The brief deliberately names no hidden workloads. The agent read these and built
     matching synthetic training families: summaries, few-shot labels and function quotation.
   - Fix: they now live in `images/judge/kvzip_judge/hidden_workloads.py`, which the Judge
     registers at start; their packages are in the Judge image only. The layout validator
     rejects these terms in the Work build context.
   - A scan of the built Work image finds none of them. Scores are unchanged (qualification
     below). Commit `4894250`.
5. **The brief said a second submission waits (minor).** In fact the server refuses it with
   HTTP 409 ("an agent submission is already queued or running"). The brief now says so.
   Commit `4894250`.
6. **The brief promised 24 hours to a 6-hour run (minor).** The agent never tracked the budget
   and was cut off while testing its last version. The Work container now carries
   `KVZIP_DEADLINE_UNIX` and `KVZIP_TIME_BUDGET_SECONDS`, and the brief points to them ("24
   hours in a benchmark run"). Verified inside a Work container. Commit `4894250`.
7. **Undocumented save quota and archive growth (minor).**
   - `sforge-save-version` failed with `checkpoint storage quota exceeded (HTTP 429)` after
     about 16 saves (about 1.05 GB). That is the framework's 1 GiB-per-run checkpoint quota;
     submissions are not affected.
   - The brief's "keep old versions if you like" also grew the archive, because every
     version is submitted and scanned. The trial's archives reached 113 MB of the 128 MiB
     limit.
   - The brief now states both. Commits `3836737` and `4894250`.

### Agent limits (not task problems)

- The agent declared itself finished after 1 h 17 min and kept working only because the stop
  hook sent it back.
- Two slips in its own training data: synthetic math under the QA instruction, and validation
  answers capped at 64 tokens.
- It never diagnosed the weights rejection.

### For your decision: the score nearly saturates

- The agent passed Fast KVzip's level (90.29 against 89.25) with its third submission, at
  33 minutes.
- It then climbed steadily for 4.5 hours to 98.35. The climb did not stall, and the metric
  separated its versions throughout.
- But ratio(0.2) and ratio(0.3) are already above 100, because relative accuracy is uncapped:
  the redundancy group reached 101.9. Only r = 0.1 has headroom left: 92.8, which is about 2.4
  score points.
- Late candidates differ by 0.2–0.5 points. A 24-hour run by a strong agent will likely end
  near 99–100.

This is your metric as specified, so I did not change it. If you want more room above Fast
KVzip, options are:
- add a harder ratio (0.05);
- weight r = 0.1 more;
- cap each task's relative accuracy at 1;
- make the hidden suite harder at the same ratios.

The next trial (with the hidden-workload names removed) will show how much of this round's
score came from tailoring to them.

The public suite is also a weak diagnostic for strong scorers:
- both public `ruler_qa` rows have full-cache accuracy 0 (decision 14);
- Math.Find's full-cache accuracy is low;
- the quick tier sits near 99–100.

It follows your section 3, so I left it as specified.

### Qualification after the fixes

- `check-3` at `4894250` passed from a fresh clone (`git clone /work/repo`, `uv sync
  --frozen --dev`, review off) in 29 minutes:
  - images current;
  - 111 framework tests and 28 task tests passed;
  - public tests: the stub was rejected; KNorm `quick` scored 13.06, all checks pass;
  - Judge: stub invalid (40 s); KNorm **24.2869** (785 s); Fast KVzip **89.2526** (805 s),
    with the weights check `fired_reference_baseline`.
- Both baseline scores are bit-identical to every earlier run. So moving the hidden-only
  workloads to the Judge, and the new weights check, changed no score.


## 6. Trial round 2 (codex, 6 hours) and what changed

The tool ran a second trial from a clean clone of `4894250` (the round-1 fixes), again codex
`gpt-6.1-sol` at effort max for 6 hours with the integrity review on. The run used the new
`KVZIP_DEADLINE_UNIX` correctly. It got no 409 or 429 errors and kept its archives under
97 MB.

### What happened

- **21 submissions; 18 valid.** Judge time was 763–818 s each, and the final evaluation
  reproduced the best score exactly.
  - Progress: 44.90 (agent-1), 78.13, 86.45, 87.76, 89.88 (agent-5, Fast KVzip's level, at
    about 1 h 15 min).
  - Then 90.39, 92.33, 93.10, 93.55, 93.64, 94.19 (agent-11, at 3 h).
  - Three memory rejections (agent-12, 13, 14).
  - Then 91.17 and **94.82** (agent-16, the best), followed by four more between 93.10 and
    94.77.
- **The best submission is legitimate** (scorer v018, read in full).
  - What it computes: per-layer MLP gates (4,096 → about 320 → 8) on each position's attention
    input, with learned terms for the context mean, a document summary, a 5-token local pool,
    four position features and the log value norm. It adds mean and max attention received
    from 16 evenly spaced context queries per head, computed by its own streaming Triton
    kernel.
  - Training: KVzip-style reconstruction attention first, then distillation through a hard
    global top-k mask against the model's full-cache answers.
  - Training data: Project Gutenberg books, WikiText-2, NVIDIA Nemotron web-text samples, the
    Python sources and its own synthetic tasks. No benchmark data, no released gates, no
    caching or timing tricks.
  - Cost and memory: 4.9 % of prefill and 0.33 GiB peak allocation. The integrity review
    passed all 22 verdicts.
- **Score level without the leaked names.** This round ended at 94.82, against 98.35 in round
  1, and it stayed between 93.1 and 94.8 for its last 2.5 hours. Ratios 0.2 and 0.3 sit near
  99; r = 0.1 (about 86) decides the score.

  That fits the round-1 leak being worth about 3.5 points. It is an inference from two single
  runs, not a measurement. The score is still near the top, but 6-hour runs now end about
  5–6 points above Fast KVzip rather than 9, and r = 0.1 has room left.

### Task problems found and fixed (commit `df5cadc`)

1. **The memory check rejected three honest submissions (major).**
   - agent-12, 13 and 14 were rejected with "the scorer's peak device memory allocation
     exceeds 8 GiB". Their peak allocation was 2.5–3.5 GiB.
   - The undocumented outside check fired: the process footprint was 12.0–17.3 GiB against
     8 + 2 GiB. The footprint came from PyTorch's caching allocator fragmenting across
     contexts of very different lengths.
   - The public tests reported 2.5 GiB and PASS, so the agent could not reproduce the failure.
     It spent three Judge slots (about an hour) bounding allocations that were never the
     problem.
   - Re-judged with the fix, the agent-13 archive is **valid at 94.85**, higher than the run's
     recorded best: peak allocation 2.5 GiB, PyTorch cache 15.3 GiB (capped), process
     16.0 GiB.
   - Fix (decision 19): the verdict follows the stated rule; PyTorch's cache is capped at
     16 GiB the same way in the public tests; a separate, documented 19 GiB whole-process limit
     replaces the old cross-check; each limit has its own message.
   - New planted case `outside_pytorch_memory` is rejected by the process limit, and
     `memory_hog` by the allocation rule.
2. **Invalid submissions still returned ratio and group scores (minor).** The agent used the
   quality numbers of its rejected runs. Invalid results now carry none (decision 24).
3. **"The public tests never produce a score" contradicted the runner (minor).** The runner
   prints `PUBLIC SCORE`, and the framework asks agents to save a version before any
   score-producing public test. The agent never saved a version. The brief, the agent query
   and the README now call it a diagnostic public score that is not the run's score.
4. **Benchmark data as training data (loophole, not used).** Nothing forbade training on the
   suites' source datasets, which the hidden suite draws from and the Work container can
   download. The brief and the integrity policy now forbid training, tuning or validating on
   them (decision 40).

### Agent limits (not task problems)

- It ran its own 32–45 GiB training jobs on the Work GPU next to the public tests, which ran
  out of memory twice (about 10 minutes).
- It never saved a version (partly the wording in fix 3).
- It misdiagnosed the memory rejection as a peak on some hidden context.
- Its last five candidates did not beat agent-16.

### For your decision (unchanged from round 1, less pressing)

- **Saturation.** Without the leak the score ends near 95 after 6 hours, with headroom only at
  r = 0.1. The options from Part 5 still apply if you want more room.
- **Layer-0 gates.** A learned function of layer-0 hidden states is a learned function of the
  token, because layer 0's input is the token embedding. Your rule and the Fast KVzip baseline
  allow it, and the brief says so. Whether that is the line you want is your call.

### Qualification after the fixes

- `check-4` at `df5cadc` passed from a fresh clone (review off) in 30 minutes:
  - images current;
  - 111 framework tests and 29 task tests passed;
  - public tests: the stub was rejected; KNorm `quick` scored 13.06, all checks pass;
  - Judge: stub invalid (41 s); KNorm **24.2869** (781 s); Fast KVzip
    **89.2526** (802 s), with the weights check `fired_reference_baseline`.
- Both baseline scores are again bit-identical to every earlier run. The memory cap, which is
  far above either baseline's use, did not change them.

