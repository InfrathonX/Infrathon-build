# Building an Infrathon task

An expert has defined a long-horizon AI infrastructure task. You turn the definition into a
runnable Infrathon task, so that agents can later be measured on it. The tool around you checks
your work from a clean clone and runs a trial, in which another agent attempts the task through the
real launcher. Your statement that something works counts for nothing; the check and the trial do.

## What you have

| path | what it is | you can write |
|---|---|---|
| `/input/TASK_DEFINITION.md`, `/input/attachments/` | the expert's definition and files | no |
| `/work/repo` | Infrathon, in git, with a few example tasks. **The deliverable is your commits here.** | yes |
| `/work/build` | your notes, `TASK.json`, `REPORT.md`; the tool also puts logs and trial results here | yes |
| `/work/assets` | `INFRATHON_ASSETS_ROOT`: data your task fetches or generates | yes |
| `/work/hf` | Hugging Face cache (`HF_HOME`) | yes |
| `/work/store` | version store for runs (`SFORGE_VERSION_STORE_ROOT`) | yes |
| `/work/clean` | where clean clones for checks and trials are made | yes |
| `/guide` | this guide, `ENVIRONMENT.md` and `reference_docs/` | no |

`ENVIRONMENT.md` describes the machine you work on: read it before anything else.

`reference_docs/` holds the framework's format and portability rules and a template for the task's
README. They predate the current repository layout: where one disagrees with `/work/repo`, the
repository wins. `/work/build/REFERENCE.md` lists the example tasks.

Nothing about this machine belongs in the task. A delivered task must work on a normal host with
direct internet, so never put this machine's paths, addresses or workarounds into a Dockerfile,
script or document.

## What an Infrathon task is

A task is τ = (W, P, E, R, T): a workspace the agent starts from, public tests it can run, private
tests it cannot see, resources and a time budget. Expert definitions follow the same outline.

- The agent works in a **Work** container: the brief (`AGENT_TASK.md`), the starting workspace and the
  public tests. It saves versions and submits.
- The **Controller** copies only the paths the brief allows, checks the submission (rate limits, the
  integrity review), and sends it on.
- The **Judge**, in a separate container the agent cannot see, runs the private tests and returns a
  result: valid or rejected with a reason, the score, and only the feedback the definition allows.

Learn the layout from the examples in `/work/repo/tasks/latest/`, and the harness's interface from
`sforge/harness/task_spec.py`, `docs/LLM_JUDGE.md` and `docs/LLM_JUDGE_POLICY_AUTHORING.md`. Pick the
example closest in shape to your task, and keep what you can justify: some examples carry hardening
from many past incidents, and their size is not the minimum to reproduce.

The deliverable, as commits on `/work/repo`:

- `tasks/latest/<task>/`: the whole task: images, brief, starting workspace, public tests, Judge,
  hidden-suite builder, integrity policy (`llm_judge/POLICY.md`), task tests, README
- `tasks/latest/<task>/portable/`: the rebuild layer, as the example tasks have it. Its
  `REBUILD_PROMPT.md` lets someone with nothing but the repository rebuild and run the task: host
  prerequisites, GPU memory envelope, external inputs with their pins, the frozen instances, the build
  command, post-build verification with the expected results, the formal run command, the run's
  artifacts and success criteria, and what to do when something fails. The manifests beside it
  record the suites, resources and protocols the same way the examples do
  (`reference_docs/PORTABILITY_CONTRACT.md` explains them).
- `tasks/latest/<task>/oracle/`: the reference solutions and baselines you measured, with notes on
  what each is and the score it got, as the examples keep theirs
- `launch/<dir>/run.sh`: the entry point a benchmark run uses
- `scripts/infrathon/tasks.json` and the README tables the framework's tests compare
- `sforge/harness/resource_profiles.json`, if your deployment needs an entry

Do not change the framework. `sforge/` stays as it is. If it truly cannot do what the task needs,
say so in the report rather than patching it.

## Rules

**The expert decides the task; you implement it.** Their test cases, metric, checks, feedback,
resources and time budget are the specification. Implement them as written. If something they asked
for is impossible or contradictory, implement the closest faithful version and record the gap in the
report, with the evidence. Never drop a requirement quietly, and never report it as done when it is not.

**Decide what the definition leaves open, and write it down.** Each decision goes in the report with
its reason and what it would cost to change, written for someone who knows the science but not this
framework.

**The checkout is the source of truth for the framework.** Where a document or a copy elsewhere
differs from `/work/repo`, the checkout wins. Search the web freely for the upstream projects, papers,
datasets and models the task is about.

**Say what you measured and what you assumed**, and keep the two apart.

### Work and Judge

- The Work image holds the brief, the starting workspace and the public tests. It never holds Judge
  code, hidden data, oracles or reference solutions.
- Public and hidden test data do not overlap. Prove it in a task test.
- Every evaluation input is pinned (fetched at a fixed revision), generated (from a shipped generator
  with a fixed seed, its result's hash recorded) or committed on a Judge-only path. Nothing may exist
  only on this machine. `hf download` always takes `--revision`.
- Hidden data the definition says not to redistribute is rebuilt from its sources and frozen by hash,
  never committed.

### The Judge

- **Treat every submission as hostile.** Assume it will read files it should not, detect that it is
  being measured, change its inputs, cache across calls, or use more than its share of resources.
  Decide which check stops each.
- **Every rule in the definition becomes a Judge check, an integrity-review rule, or a line in the
  report saying it is not enforced.** A check that has never fired on a submission written to trigger
  it is not known to work.
- **Fail closed.** Anything unexpected makes a submission invalid, with a reason. It never gets a score
  by default.
- **The score has to mean something.** A useless submission must score low, and the expert's baselines
  must land where the definition says they should.
- **Bound the feedback** to what the definition allows. No per-item hidden information.
- **Time the Judge.** A submission is synchronous, so a slow Judge eats the agent's budget. Measure how
  long one takes, keep it within the definition's limit, and tell the agent in the brief.
- **Measure, then set limits.** Memory floors and timeouts come from measurements on these GPUs, with
  margin. Thresholds compare within one run, never against an absolute number measured once.

### The brief

- It states the goal, the interface, the metric, the rules and how to submit, so that every reason the
  Judge can reject a submission is in it.
- It does not reveal hidden test contents or sizes, hidden reference scores, or anything the agent
  could fit to. It may say that the hidden test is larger and broader than the public one.
- It never names an agent product, and says nothing about internet access either way.

### The integrity policy

Every submission is read by an LLM reviewer before the Judge runs it, against the brief and the
task's `llm_judge/POLICY.md`. Write the policy in exactly the layout
`/work/repo/docs/LLM_JUDGE_POLICY_AUTHORING.md` prescribes, and read `docs/LLM_JUDGE.md` for how the
review uses it; the example tasks' policies follow it and are your models. Those documents are the
authority on the format, over anything here.

- The brief is the authority. The policy shows, in this task's terms, where the brief's lines run:
  what the task requires, and concrete examples on both sides of each line, including legitimate
  work that could look suspicious. It never adds a rule the brief does not state.
- It contains nothing the agent must not know: no hidden test contents or sizes, no reference
  scores, no Judge internals.
- Every rule of the definition that no Judge check enforces is covered here, and the report says
  which rules rely on the review alone.
- The framework's policy tests (`tests/sforge/harness/test_llm_judge_task_policies.py`) pass.

### Portability

No absolute home paths, GPU UUIDs or indices as defaults, internal addresses, driver versions or
branch names in anything you deliver. A default that is wrong everywhere but here is worse than none:
fail with the name of the variable to set.

## The handoff: `/work/build/TASK.json`

The tool drives your task from outside with this file. Its commands run from the root of a clean
clone of your HEAD, made with `git clone /work/repo` into a new directory under `/work/clean`, with
the framework installed into its own `.venv` (`uv sync --frozen --dev`). `/work/assets`, `/work/hf`
and `/work/store` are shared, so data you prepared once is still there.

```json
{
  "task": "<directory name under tasks/latest>",
  "task_id": "<the task_id the launcher and version store use>",
  "launch": "launch/<dir>",
  "work_gpus": 1,
  "judge_gpus": 1,
  "check_cmd": "launch/<dir>/check.sh --work-gpu {work_gpu} --judge-gpus {judge_gpus}",
  "check_hours": 3,
  "trial_cmd": "launch/<dir>/run.sh --model {model} --effort {effort} --hours {hours} --run-id {run_id} --work-gpu {work_gpu} --judge-gpus {judge_gpus}",
  "trial_output": "outputs/<task>/runs/{run_id}",
  "score_key": "score"
}
```

- `check_cmd` exits 0 only if everything passed. It builds the images and prepares the data (both
  idempotent, so a second run is quick), runs the public tests in the Work image, and puts the
  starting workspace and each of the expert's baselines through the real Controller and Judge, failing
  if a result is not what the definition expects. Print each result. It must finish within
  `check_hours`.
- `trial_cmd` runs one benchmark run, exactly as a real one would: the agent in the Work container
  for `{hours}` hours, submissions through the Controller to the Judge. It returns when the run ends.
  It runs in the same clone right after `check_cmd` passed there, so it can rely on what
  `check_cmd` built in that clone.
- `trial_output` is where that run's results land, relative to the clone. The tool reads each
  submission's `report.json` below it.
- The tool fills in `{model}`, `{effort}` and `{agent}` (the trial agent), `{hours}`, `{run_id}`,
  `{work_gpu}` and `{work_gpus}`, `{judge_gpu}` and `{judge_gpus}` (comma-separated indices as
  `nvidia-smi` numbers them), and `{auth}` (the trial agent's Codex `auth.json`, also in
  `CODEX_AUTH_JSON_PATH`). Work GPUs come first, then Judge GPUs.
- Register the task in the version store the launcher uses, as the examples do.
