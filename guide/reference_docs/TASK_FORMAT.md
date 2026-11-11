> Written for an earlier version of the repository (20 tasks). Some files, tests and documents it names are not in the released Infrathon. Where this disagrees with /work/repo, the repository wins.

# Task format

What a task directory contains, and the contract every task in `tasks/latest/`
satisfies. Read this before adding a task or auditing an existing one.

## Directory anatomy

Taking `tasks/latest/standalone-dynamic-grouped-fp8-h100-v3-scratch/`
as the reference shape — every one of the 20 tasks has `tasks/`, `images/`,
`scripts/`, and `tests/`; 19 also have `judge/` and `docs/`:

```text
<task-directory>/
├── README.md                  # the task's own entry point (required, all 20)
├── experiment-<profile>.yaml  # one per launch profile (model, effort, budget)
│
├── tasks/
│   ├── <task_id>.json         # the SForge task spec — see below
│   └── BENCHMARK.yaml         # base image + apt packages for this family
│
├── images/
│   ├── work/                  # Dockerfile + the agent-visible workspace
│   │   ├── AGENT_TASK.md      #   the brief the agent reads first
│   │   ├── benchmark/         #   the public test runner
│   │   └── workdir/<lab>/     #   the submitted tree's starting state
│   └── judge/                 # Dockerfile + verifier + private manifests
│
├── judge/run_judge.py         # the scoring entry point (EAGLE-3 keeps it only
│                              #   under images/judge/)
├── runtime/                   # trusted runtime shared by Judge and oracle (15/20)
├── oracle/                    # host-only reference measurement (15/20)
├── host_reference/            # upstream parity implementations, host-only (5/20)
│
├── docs/
│   ├── DESIGN.md              # what the task asks and why
│   ├── SCORING_CONTRACT.md    # authoritative score formula and hard gates
│   ├── SOURCE_PROVENANCE.md   # upstream repo, commit, license, what was taken
│   ├── SECURITY.md            # the threat model and what blocks each escape
│   └── RUN_RESULTS.md         # measured qualification and model-run evidence
│
├── llm_judge/POLICY.md        # REQUIRED integrity-review policy: what the task
│                              #   requires, what counts as circumventing it, what is
│                              #   allowed (agent-safe; LLM_JUDGE_POLICY_AUTHORING.md)
├── scripts/                   # see the table below
└── tests/                     # task-local static and unit validation (no GPU)
```

Not every task carries every `docs/` file — see the coverage note at the end.
The task `README.md` is always present and is always the authoritative index for
that task.

## The task spec

`tasks/<task_id>.json` is what SForge loads. The fields that matter:

| Field | Meaning |
| --- | --- |
| `task_id` | Stable identifier; also the key the version store registers. |
| `base_image` | Key into `BENCHMARK.yaml`'s `base_images`. |
| `cwd` | The workspace path inside the Work container. |
| `submit_paths` | What the agent's submission archive contains. |
| `submit_exclude` | Patterns stripped from the archive (`.git/`, `auth.json`, `build/`, caches). |
| `candidate_identity_exclude` | Submitted files that never change validity or score — typically `notes.md`. Archived verbatim, but excluded from candidate identity so a notes-only edit does not create a new replay point. |
| `version_enumeration` | Declares that the tree retains one directory per implementation plus a selector file, e.g. `candidates/kernel_vNNN` + `FINAL_VERSION`. Enables per-version replay. |
| `work.agent_query` | The task brief handed to the agent. |
| `judge.eval_cmd` | What runs inside the Judge container. |
| `judge.private_artifact_path` | Where the structured result lands. |
| `judge.parser` | `structured_json` for all 20 tasks. |
| `judge.selection` | `valid_then_score` for all 20 tasks. |
| `judge.score_direction` | `maximize`, except Flash-KMeans v7 (`minimize`). |

Only ever list a path under `candidate_identity_exclude` if the Judge ignores it
beyond existence and size checks. Replay measures one representative archive per
candidate, so if an excluded file could flip the admission outcome, that
difference would be invisible on the curve.

## Standard scripts

`scripts/lib.sh` is sourced by every other script in the task and resolves the
repository root by walking up for a directory containing both `pyproject.toml`
and `sforge/`. It also sets `TASK_ID`, image tags, and the output paths.

| Script | Present in | What it does |
| --- | --- | --- |
| `lib.sh` | 20/20 | Shared path/ID resolution and preflight helpers. |
| `build_images.sh` | 20/20 | Builds the Work and Judge images. |
| `validate_layout.py` | 19/20 | Static gate: fails if Judge-only material leaks into the Work context, or the layout drifts. |
| `smoke_test.sh` | 19/20 | Fast end-to-end sanity check. |
| `eval_archive.sh` | 19/20 | Score one `submission.tar.gz` through the real Judge. |
| `replay_candidates.sh` | 19/20 | Batch re-score every saved candidate — see `sforge/harness/version_store.py`. |
| `status.sh` | 18/20 | Current image/lease/GPU state. |
| `run_fixed_budget.sh` | 15/20 | The formal run entry the `launch/` wrapper delegates to. |

One task deviates:

- **LLaDA H100 v2** has no `replay_candidates.sh`. Its Judge needs live host
  artifact fingerprints, so stored versions are scored one at a time through
  `eval_archive.sh`.

## The contract every task satisfies

These are enforced by `tests/test_repository_layout.py` and
`tests/test_launch_entrypoints.py`, which run without a GPU:

1. **One directory per task family** under `tasks/latest/`.
2. **Manual submission only.** Every `experiment*.yaml` sets
   `disable_auto_eval: true`; no runner passes `--max-submissions 0`; any script
   using `--eval-interval` also passes `--disable-auto-eval`.
3. **Centralised outputs.** Sourcing `scripts/lib.sh` yields the repository root,
   the task root, and `outputs/<task-directory>/` — nothing writes into the task
   source tree.
4. **Documented submission and replay policy.** Every task `README.md` has an
   `## Evaluation submission policy` section naming both `submission.tar.gz` and
   `final_archive.tar.gz`.
5. **No baked-in agent.** No Work Dockerfile installs a Codex CLI, Node from
   `nodejs.org/dist`, or a sudo install shim. The agent arrives as a
   host-mounted runtime.
6. **One Codex CLI pin.** Every task wrapper's `CODEX_CLI_VERSION` default equals
   the single pin in `launch/common.sh`.
7. **A launcher per task.** `launch/<name>/` covers every directory in
   `tasks/latest/`, and every `run.sh` calls the proxy guard.

## Upstream sources

**No upstream source tree is vendored into this repository.** Each task records
its upstream in `docs/SOURCE_PROVENANCE.md` (or the equivalent README section):
repository URL, commit SHA, license, and exactly what was taken.

Two host-only tools need an actual upstream checkout, which you supply:

| Tool | Variable | Upstream, pinned |
| --- | --- | --- |
| `flash-kmeans-…-v7-scratch/scripts/make_host_calibration_archives.sh` | `FLASH_KMEANS_UPSTREAM` | `svg-project/flash-kmeans` @ `12ed49b2eddc207dfc5ed27dba244af2365bd9c6` |
| `standalone-k-grouped-fp8-wgrad-…/oracle/run_oracle.sh` | `DEEPGEMM_SOURCE` | `deepseek-ai/DeepGEMM` @ `559d79fb6994a58b8a15b4b93bf13ccc16edf247` (submodules: cutlass `f3fde583…`, fmt `553ec11e…`) |

Both fail immediately with an actionable message when the variable is unset or
the path is not a real checkout. Neither is needed to build images, run an
agent, or score a submission — they exist to produce host-side reference
measurements.


## The task's specification document

Every task also has one document under `docs/tasks/`, and they all follow the
same structure so they can be read side by side. A new task's document must match
it:

```text
# <Name> — <hardware> v<N> (seed|scratch)

| Item | Value |        fixed 11 rows, in this order:
                     Task directory · `task_id` · Upstream project · Category ·
                     Starting point · Hardware · Base image · Score · Status ·
                     GPU envelope · Rebuild and run

## Task Definition          what the agent must do, and what makes it hard
## Metric                   score formula in a code block, then the hard gates
## Public / Private Test    one table: what each evaluation tier is, and how big
## Source                   one table: upstream URL, commit pin, license

---

# Detailed specification

## 1. The upstream               what the upstream project does; what was and was
                                 not derived from it; deliberate deviations
## 2. The task                   the frozen surface vs the editable surface, the
                                 ABI, and the mathematical contract
## 3. The agent's instructions   the brief the agent actually receives
## 4. The public test            the disclosed suite the agent drives itself
## 5. The Judge's hidden test    anchor and holdout, gates, measurement protocol
## 6. Runs and acceptance        qualification evidence and model runs, then:
### Changing GPUs and comparability   what changes with the card, what must be
                                      re-measured, when scores are comparable
### Known limitations            design boundaries and unresolved problems
### History and verification     dated entries, oldest first, including the
                                 clean-clone rebuild record P5 requires
### Maintenance and replay       rebuild / replay / re-measure, pointing at the
                                 task's REBUILD_PROMPT.md
```

`GPU envelope` carries the profile id, the declared floor and the measured Judge and
Work peaks, sourced from the task JSON and
`sforge/harness/resource_profiles.json`; `Rebuild and run`
points at the task's rebuild layer. `Hardware` describes the *current* deployment
form — a single whole-GPU lease or a dual NVLink pair — not a GPU model.

Deeper structure goes in `###` subsections; do not add a `## 7.`. The four
overview sections must be readable on one screen — details belong in the detailed specification.

Three rules the per-task documents are linted against: a date-looking token may
appear only from `## 6.` onwards; a file is cited by its bare name, never
`file.py:123`; and no host-private string (an absolute home path, a GPU index or
UUID as a default, a driver version, a proxy address, a branch name) appears at
all. Hidden suites are described by their *generation rule*, never by the drawn
instance. The template is
[TASK_DOC_TEMPLATE.md](TASK_DOC_TEMPLATE.md).

## Validating a task without a GPU

```bash
# Repository-wide contract (fast, no GPU, no Docker):
python3 -m pytest tests/ -q

# One task's own static and unit validation:
python3 tasks/latest/<task>/scripts/validate_layout.py tasks/latest/<task>
python3 -m pytest tasks/latest/<task>/tests -q
```

GPU tasks still need the hardware, the built images, and the private host-side
datasets named in that task's directory before anything can actually be scored.

## Documentation coverage

15 of 20 tasks carry the full `docs/` set. The gaps are historical, and in each
case the material exists in the task `README.md` or in
`docs/tasks/<task>.md` instead:

| Task | Missing from `docs/` |
| --- | --- |
| `flash-kmeans-…-v7-scratch` | `DESIGN.md`, `SCORING_CONTRACT.md`, `SOURCE_PROVENANCE.md` — has `design.md`, `PAPER_TO_TASK.md`, `PAPER_AND_REPO_NOTES.md` |
| `concurrent-agent-serving-…-v24` | `DESIGN.md`, `SOURCE_PROVENANCE.md` |
| `llada-h100-…-v2-vanilla-seed` | `DESIGN.md`, `SOURCE_PROVENANCE.md` |
| `qwen3-1p7b-orpo-…-v4`, `qwen3-4b-to-1p7b-gkd-…-v1` | `SOURCE_PROVENANCE.md` (upstream is `huggingface/trl`, pinned in the task `README.md`) |
| `qwen3-4b-to-1p7b-gkd-…-v1` | also `RUN_RESULTS.md` — no model run recorded yet |
