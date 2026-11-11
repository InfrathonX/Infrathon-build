> Written for an earlier version of the repository (20 tasks). Some files, tests and documents it names are not in the released Infrathon. Where this disagrees with /work/repo, the repository wins.

# Architecture

How a run actually executes, and where the trust boundaries are.

## The short version

A run has three participants:

```text
┌─────────────────────────── trusted host ───────────────────────────┐
│                                                                     │
│  sforge run                    sforge serve                         │
│  ├─ starts the Work container  ├─ Judge HTTP server (FastAPI)       │
│  ├─ drives the agent           ├─ owns the private test data        │
│  ├─ extracts submissions       ├─ starts a Judge container per      │
│  └─ writes run logs            │  evaluation                        │
│                                └─ appends to the version store      │
│                                                                     │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ HTTP only, whitelisted by host iptables
┌──────────────────────▼──────────── untrusted ───────────────────────┐
│  Work container                                                      │
│  ├─ the coding agent (Codex / Claude Code)                           │
│  ├─ the task workspace (the submitted tree lives here)               │
│  ├─ public tests the agent may run freely                            │
│  └─ sforge-submit / sforge-save-version (curl wrappers)              │
└──────────────────────────────────────────────────────────────────────┘
```

The Work container is the only place untrusted code runs. It never sees the
private test data, the scoring internals, the version store, or the host
filesystem. Everything it can reach, it reaches over HTTP through a host-side
whitelist.

## The three containers

**Base image** — the shared runtime for a task family, declared in the task's
`tasks/BENCHMARK.yaml` (an official image such as
`pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`, plus a short apt package list).
Its content hash feeds the Work and Judge image tags, so a base change forces a
rebuild of both.

**Work image** — what the agent lives in: the task workspace, the public test
runner, the seeded or empty starting code, and `AGENT_TASK.md`. It contains no
private data, no reference solution, and — deliberately — no agent CLI
(see [Agent runtimes](#agent-runtimes)).

**Judge image** — the scoring side: the trusted runtime, the private case
manifests, the verifier, and `run_judge.py`. It is built from a separate
context that the Work build never touches. Several tasks add a
`validate_layout.py` that fails the build if a Judge-only path ever appears in
the Work context.

## Public and private evaluation planes

Two ways to be measured, and they are not the same thing.

The **public plane** is a disclosed case suite the agent runs itself inside the
Work container, as often as it likes. It guides the agent's search; nothing about
it is recorded as a score.

The **private plane** is the Judge: a held-out case set the agent never sees, run
in the Judge container by trusted code against a host-extracted submission
archive. Its output is the score. Many tasks add an *anchor* suite — a disclosed
case set scored by the Judge's own protocol — so a public reading can be related
to the Judge's yardstick without revealing the holdout.

The Judge server enforces the split at the API level: a container-side session
token carries capabilities, and in a fixed session the worker's token permits
checkpoint saves only — it cannot submit and cannot read evaluation history.

## Submission policy

Every task in this repository runs with **manual submission only**:
`disable_auto_eval: true` in the experiment YAML, `--disable-auto-eval` on
direct runners. Scheduled background evaluation is off. The agent decides when
to spend a submission by calling `sforge-submit` itself.

Each accepted submission is retained as an exact `submission.tar.gz`. When the
run ends, the workspace is captured once more as `final_archive.tar.gz` from
the **stopped** container, so the terminal state is immutable. An intermediate
working tree that was never submitted is not checkpointed by this mechanism —
that is what the version store is for.

## Integrity review before the Judge (mandatory)

By default (`SFORGE_LLM_JUDGE_MODE=enforce`), every agent submission is first
reviewed by a coding agent (Codex or Claude Code from a host agent runtime) in
a CPU-only container built from the task's Judge image. It reads the
submission, its Work-image baseline, the agent's briefs and the task's
`llm_judge/POLICY.md`, and decides only whether the agent stayed inside the
task's required scope -- never how good the code is. In `enforce` mode a
verified violation ends the submission before any GPU work (Work is never
paused, no Judge container starts) and the agent is told which rule it broke.
See [LLM_JUDGE.md](https://github.com/InfrathonX/Infrathon/blob/main/docs/LLM_JUDGE.md).

## One GPU per run: the Work/Judge lease

A single-GPU task does not reserve one card for the agent and a second for the
Judge. Each run is allocated **one** physical GPU, and Work and Judge take turns
on it under a lease that is the authority on who owns the device. (The four
tasks whose workload is itself two-rank keep their pair; see
`sforge/harness/resource_profiles.json`.)

Every `(run_id, task_id)` holds a never-reused `lease_id` and a monotonic
`epoch`. The interactive cycle is:

```text
UNBOUND -> WORK_ACTIVE -> PAUSE_REQUESTED -> WORK_PAUSING -> WORK_PAUSED
        -> WORK_GPU_IDLE_ATTESTED
        -> JUDGE_STARTING -> JUDGE_RUNNING -> JUDGE_CLEANING -> JUDGE_CLEANED
        -> JUDGE_GPU_IDLE_ATTESTED
        -> WORK_UNPAUSING -> WORK_ACTIVE
```

`WORK_PAUSED -> WORK_UNPAUSING -> WORK_ACTIVE` is also reachable directly, but
only as a busy-retry in which the Judge was never started and Work is confirmed
to still own the allocation; the agent gets a `RESOURCE_BUSY` that does not
consume a submission, and the retry runs under a new epoch.

The deadline path is irreversible and never returns to `WORK_ACTIVE`:

```text
WORK_ACTIVE | WORK_PAUSED | WORK_GPU_IDLE_ATTESTED | JUDGE_GPU_IDLE_ATTESTED
  -> FINALIZING -> FINAL_WORK_FROZEN -> FINAL_ARCHIVE_READY
  -> FINAL_WORK_DESTROYING -> FINAL_WORK_DESTROYED -> FINAL_GPU_IDLE_ATTESTED
  -> FINAL_JUDGE_STARTING -> FINAL_JUDGE_RUNNING -> FINAL_JUDGE_CLEANING
  -> FINAL_JUDGE_CLEANED -> FINAL_JUDGE_GPU_IDLE_ATTESTED -> CLOSED
```

A run that is not to be scored takes the explicit exit
`FINAL_GPU_IDLE_ATTESTED -> CHECKPOINT_ONLY_CLOSING -> CLOSED`, so it can
neither slip into the scoring path nor leave a resumable paused Work container
behind on the machine.

The rules the coordinator enforces:

* **No Judge is created** until Work is proven paused *and* the allocation is
  proven free of context. Pause first, attest second: checking and then pausing
  leaves a window in which the agent takes the device back.
* **Work is not unpaused** until the Judge container is proven destroyed and the
  allocation proven free again.
* An operation carrying a stale epoch is answered with a conflict.
* **Anything uncertain fails closed.** The built-in NVML observer does no
  cgroup/PID attribution, so it never guesses that a lingering context belongs
  to Work and resumes anyway. Persistent busy means `FENCED` — a state an
  operator reconciles, not one a run recovers from.
* **The lock key is the allocation, not the `task_id`**; otherwise two different
  tasks can still collide on one physical card. `allocation_key` is a content
  digest over the canonicalised physical device set, generated by the scheduler,
  so an index and a UUID cannot alias each other. Across processes the lock is a
  host flock or a database transaction, never an in-process `threading.Lock`.
* **Idle is not `memory.used == 0`.** Driver and display reservations can be
  non-zero. What is checked is the compute, graphics and MPS process lists, with
  an acceptable idle-memory envelope belonging to that host's qualification —
  and the allocation has to be held under a host lock throughout, or an ordinary
  process can open the device between two samples and no NVML snapshot proves
  anything.

The terminal state is `FINAL_WORK_FROZEN`, not `FINAL_STOPPED`, and the
difference is deliberate: the deadline can land while the Judge is running and
Work is already paused. Stopping a paused container would mean unpausing it
first, letting the agent write after the deadline. So the coordinator writes a
durable `never_resume` tombstone, captures the archive off the frozen
filesystem, and only then destroys Work and attests the GPU idle. On every other
path a resumable pause is refused as a terminal state.

The durable lease record binds at minimum: the lease ID; the allocation digest
and the exact device UUIDs; the full Work and Judge container IDs with their
reserved labels; the epoch and operation IDs; the submission and archive
digests; every idle attestation with the provenance of the observer that
produced it; and the recovery policy (`unpause_same_exec`,
`restart_agent_session` or `never_resume`). It is written with `fsync` + atomic
rename + directory fsync after each accepted transition. Every entry point that
could start or unpause a container passes through this fence: holding a Docker
handle is not authority, and the persisted lease and its tombstone outrank any
in-process object. The state machine is the coordinator's ordering constraint,
not the hardware proof — only an attestation a real observer produced and the
durable layer accepted advances an idle state.

That buys safety, not full crash-resume liveness. A coordinator that dies
between `FINAL_WORK_FROZEN` and `FINAL_WORK_DESTROYED` does not rebuild its
authority from the JSON on restart; an operator reconciles the exact container
IDs, the allocation and the NVML state. Completed final-Work and final-Judge
requests are idempotently retryable when identity and artifact digests match.

Baseline caches sit deliberately outside this machine: they have their own
durable state keyed by digest, and the Judge coordinator makes the cache ready
before candidate evaluation according to the metric's baseline policy
(see [Baseline is not one thing](#baseline-is-not-one-thing)).

## Network isolation

Every current task sets `internet: true`: the agent's Work environment has
internet access and live web search. The policy and how each runtime applies it
are in [INTERNET_AND_WEB_SEARCH.md](https://github.com/InfrathonX/Infrathon/blob/main/docs/INTERNET_AND_WEB_SEARCH.md). The mechanism
below applies only to a task that sets `internet: false`.

With `internet: false`, Work containers get no general internet. `network_isolation.py` installs
per-container iptables/ip6tables chains **on the host network namespace**,
whitelisting only the endpoints a run needs (Judge server, model API). The
container has no `NET_ADMIN`, so it cannot alter them. IPv6 is blocked outright;
whitelisted hostnames resolve to IPv4 and are injected via `extra_hosts`.

Several tasks tighten this further: the Work container sits on an internal Docker
network reachable only through a restricted CONNECT proxy sidecar, with
`SFORGE_HTTP_PROXY` pointed at it. `launch/common.sh` refuses a run whose proxy
is host loopback — that address resolves to the container itself and would fail
every request.

## Agent runtimes

No task image bakes in an agent CLI. The toolchain is a host directory built by
`launch/build-agent-runtime.sh` carrying a `.sforge-agent-runtime.json` marker;
when `SFORGE_AGENT_RUNTIME_DIR` names one, the Docker backend bind-mounts it
read-only at `/opt/agent-runtime`, `BASH_ENV` puts its `bin/` on `PATH`, and the
agent's network install step becomes an offline probe of the mounted toolchain.

So a CLI bump never rebuilds 20 task images: agent kind (`codex`, `claude-code`)
and version are chosen per launch and recorded in `agent_runtime.json` under the
run's log directory. `tests/test_launch_entrypoints.py` asserts no Work image
reintroduces a baked agent.

## Scoring

The Judge writes a structured JSON artifact; `parser: structured_json` reads it.
Grading is fail-closed: a malformed, missing, or unparseable result is a
failure, never a silently-zero score.

All 20 tasks use `selection: valid_then_score`: validity is a hard gate, and
among valid results the best score wins. (The harness also implements
`pass_rate_first`, where correctness pass rate dominates and score breaks ties;
no task here uses it.)

`score_direction` is `maximize` everywhere except Flash-KMeans v7, which
minimises a family-weighted geometric mean of warmed latencies in milliseconds. Most kernel tasks score raw effective TFLOPS
with no cap and no baseline ratio. The per-task `docs/SCORING_CONTRACT.md` is
authoritative for the exact formula and hard gates.

### Baseline is not one thing

Three different objects get called "the baseline", and conflating them is how a
correctness check gets skipped, or a candidate gets scored 0 for the evaluator's
failure:

* a **correctness oracle** proves the output is semantically right, and is run or
  verified for every case;
* a **qualification baseline** proves the task is solvable and has headroom on
  this card, and never enters a score;
* a **scoring baseline** is the numerator or denominator of the metric, and is
  the only one of the three that needs a caching protocol.

Every metric declares which it uses:

| `baseline_policy` | meaning | used by |
| --- | --- | --- |
| `none` | the score depends on no performance baseline | nearly every raw-score kernel and engine task here |
| `cached_once_per_campaign` | generated once per (suite, profile, evaluator, image, protocol) and then frozen | a separable, stable reference: StreamDiffusionV2's seed calibration, concurrent-serving's Direct evidence |
| `paired_per_candidate` | reference and candidate interleaved in the same round (RC/CR or AB/BA) | ORPO and GKD, where allocator, NCCL and clock state move the ratio |

`none` means *the official score is not divided by a timing baseline*; such a
task still has an oracle, and may have a quality reference. A cache key binds at
least the baseline implementation digest, the suite/manifest digest, the
task-side capability requirement, the attested measurement profile, the
evaluator and image digests, the seed/warm-up/repeat/order protocol and the
aggregation; the cached result also records the hardware and software inventory
and every raw per-case measurement. A change of profile or evaluator opens a new
epoch instead of silently reusing the old value, and a cache miss is claimed
single-flight so one key can never produce two disagreeing references.

Two rules follow, and both are about not charging the candidate for the
evaluator's problems:

* **A failed baseline is an evaluator/infrastructure error.** It does not consume
  a submission and it does not score the candidate 0. Only a candidate that
  violates a correctness gate receives the failing score its task contract
  defines.
* **Public tests never trigger the hidden baseline.**

## Where a run's output goes

```text
outputs/<task-directory>/
├── build/     # image build logs
├── launch/    # tmux launch metadata, when the launcher detaches
└── runs/      # per-run logs, judge history, agent_runtime.json, archives
```

`OUTPUT_ROOT` and `OUTPUT_SUBDIR` relocate this; `LOGS_DIR` sets an exact path
and takes precedence. Nothing is written inside the task source tree.

## Further reading

- [TASK_FORMAT.md](TASK_FORMAT.md) — what a task directory must contain.
- `sforge/harness/version_store.py` — capturing and re-scoring the agent's
  intermediate solutions.
- [LLM_JUDGE.md](https://github.com/InfrathonX/Infrathon/blob/main/docs/LLM_JUDGE.md) — the mandatory integrity review in front of the
  Judge.
- [sforge/NOTICE](https://github.com/InfrathonX/Infrathon/blob/main/sforge/NOTICE) — where the `sforge/` runtime comes from.
