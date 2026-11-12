<h1 align="center">Infrathon-build</h1>

<p align="center"><b>Turn an expert's task definition into a runnable Infrathon task, then find out whether an agent can actually do it</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.11995"><img src="https://img.shields.io/badge/arXiv-2611.11995-b31b1b?logo=arxiv&logoColor=white" alt="arXiv paper"></a>
  <a href="https://infrathonx.github.io"><img src="https://img.shields.io/badge/Homepage-0969da?logo=googlechrome&logoColor=white" alt="Homepage"></a>
  <a href="https://infrathonx.github.io/blog.html"><img src="https://img.shields.io/badge/Blog-8250df?logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2UtbGluZWNhcD0icm91bmQiPjxwYXRoIGQ9Ik01IDNoMTBsNCA0djE0SDV6IE0xNSAzdjVoNCBNOCAxMmg4IE04IDE2aDgiLz48L3N2Zz4=" alt="Blog"></a>
  <a href="https://github.com/InfrathonX/Infrathon"><img src="https://img.shields.io/badge/Benchmark-24292f?logo=github&logoColor=white" alt="Infrathon benchmark"></a>
  <a href="https://huggingface.co/datasets/Infrathon/Infrathon"><img src="https://img.shields.io/badge/Dataset-ffcc4d?logo=huggingface&logoColor=black" alt="Hugging Face dataset"></a>
  <a href="https://hub.docker.com/r/infrathon/infrathon"><img src="https://img.shields.io/badge/Docker%20Hub-2496ED?logo=docker&logoColor=white" alt="Docker Hub images"></a>
</p>

An expert writes the task: what the agent optimizes, the public and private test cases, the metric,
the checks, the resources and the time budget (τ = W, P, E, R, T). A builder agent implements it in
an Infrathon checkout, using the tasks already there as examples. The tool then checks the task
from a clean clone and runs a trial: a different agent attempts it through the real launcher, as a
benchmark run would. The builder reads what happened, fixes what the trial shows is wrong with the
task, and the tool runs another trial if needed. The result is a patch for the Infrathon repository
and a report for the expert.

<p align="center"><img src="docs/assets/flow.svg" width="100%" alt="The flow: an expert's definition and Infrathon's example tasks go to build, check, trial, review and deliver; inside a trial the agent submits through the Controller's integrity review to the Judge"></p>

| step | who | what |
|---|---|---|
| setup | tool | the Infrathon checkout becomes the run's repository; its own tests must pass |
| build | builder agent | reads the definition and the example tasks, writes the task, tests it on the GPUs, writes `TASK.json` and `REPORT.md` |
| check | tool | runs the task's `check_cmd` from a clean clone: images, public tests, the starting workspace and the expert's baselines through the real Judge. A failure goes back to the builder to fix; the check gets 3 tries |
| trial | tool | runs the task's `trial_cmd` in that clone: another agent works on the task for some hours, submitting to the Judge |
| review | builder agent | reads the trial (transcript, submissions, Judge reports), separates task problems from the agent's limits, fixes the task, and decides: `ready`, `retrial` or `blocked` |
| deliver | tool | `task.patch`, `task-only.patch`, `task.bundle` and `REPORT.md` |

One builder session runs through the whole flow, resumed at each step, so it remembers what it
built. The checks and trials run from a clean clone, so they show what a fresh host would see.

**Example:** [`examples/kvzip/`](examples/kvzip/) follows one run from an expert's definition of a
KV-cache scoring task to the delivered task: two six-hour trials, the eleven task problems they
exposed and the fixes, and the integrity review's calibration.

## Quick start

Needs a Linux host with Docker, the NVIDIA container toolkit, GPUs, `git`, `uv` and `tmux`, and the
agent runtimes Infrathon's launchers use, built with its `launch/build-agent-runtime.sh`.

```bash
git clone --recursive https://github.com/InfrathonX/Infrathon-build infrathon-build && cd infrathon-build
pip install -e .            # or run it in place: python3 -m infrathon_build ...

cat > local.env <<'EOF'
# the builder (or ANTHROPIC_API_KEY here, or CLAUDE_CODE_OAUTH_TOKEN exported in the shell)
IBUILD_CLAUDE_TOKEN_FILE=/path/to/claude-oauth-token
# the trial agent and the integrity review
IBUILD_CODEX_AUTH_JSON=/path/to/codex/auth.json
# holds claude-code-<version>/ and codex-<version>/
IBUILD_AGENT_RUNTIMES=/path/to/agent_runtimes
EOF

infrathon-build init my-task --input /path/to/expert-folder --gpus 0-7
infrathon-build run my-task --detach
infrathon-build status my-task      # where it is, trial scores, cost, disk and GPUs
infrathon-build log my-task         # what the builder is doing right now
```

The run stops on its own when it is done. `stop` halts it at any point, and `run` resumes from the
last finished step.

Options of `init`:

| option | default | |
|---|---|---|
| `--reference DIR` | `infrathon/` (the submodule) | the Infrathon checkout the builder starts from |
| `--sandbox none\|docker` | `none` | run on this machine, or isolate the agents (below) |
| `--gpus` | `0-7` | the GPUs this run may use, for example `0-7` or `0,2,4` |
| `--trial-hours` | `6` | the trial agent's budget per trial |
| `--rounds` | `2` | trials at most |
| `--builder claude-code\|codex`, `--builder-model` | Claude Code, `claude-opus-5-5` | who builds the task; with `--builder codex`, also pass `--builder-model` |
| `--solver-model` | `gpt-6.1-sol` (Codex) | who attempts it in trials |

## The expert's input

```text
expert-folder/
  TASK_DEFINITION.md     τ = (W, P, E, R, T): see templates/TASK_DEFINITION.md
  attachments/public/    files the agent may see
  attachments/hidden/    Judge-only files
```

The more exactly the definition states the test cases, metric, checks and baselines, the less the
builder decides on its own. Everything it does decide is listed in its report.

## What you get

Under `runs/<run>/`:

| path | what |
|---|---|
| `REPORT.md` | the report: steps and timing, builder cost, every check, every trial submission and score, each review's findings, then the builder's own report for the expert (what it built, decisions it made, measurements, what is not done or not enforced) |
| `deliverable/task-only.patch` | the task and its launcher, to apply to Infrathon |
| `deliverable/task.patch` | everything since the reference commit, registry and table entries included |
| `deliverable/task.bundle` | the builder's commits, for `git fetch` |
| `work/repo` | the repository with the builder's commits |
| `work/build/` | `TASK.json`, the reviews (`reviews/round-N.json`), trial summaries, the builder's notes |
| `work/clean/` | the clean clones of checks and trials, with each trial's agent transcript, submissions and Judge reports |
| `logs/agent/`, `evidence/` | the builder's transcripts, check and launcher logs |

## Time and cost

A module-level task on eight H100s took about 21 hours end to end: 4 hours to build, half an hour per
check, two trials of six hours, and about an hour per review. The builder (Claude Code, Opus, maximum
effort) cost about $100 as Claude Code reports it; the trial agent and the integrity reviews come on
top. Most of the build time goes into verification: the Judge on the full private suite, and one
deliberately rule-breaking submission per check.

## Running on this machine, or in a sandbox

With `--sandbox none` (the default) the builder is an ordinary process in the run's directory,
using the host's Docker and the run's GPUs. Its home is inside the run, so its git and tool settings
stay there, and its guide tells it to keep to its own directory, containers and GPUs. It has the same
access to the machine as you, though: use the sandbox on a shared host, or for anything you would not
run yourself.

With `--sandbox docker` each run gets its own Docker network with an egress gateway (mitmproxy:
hostname and path policy, a log of every connection), a nested Docker daemon that sees only the
run's GPUs and refuses privileged or host-mounting containers, and two agent containers that share
its network namespace: the builder, and a runner for the tool's checks and trials. Only the runner
sees the trial's credentials, and only while it runs. The builder works as a non-root user and
cannot reach the host's files, Docker daemon, credentials or private network.

More `init` options:

- `--network-policy FILE` (sandbox): hosts and paths the gateway refuses (see `sandbox/proxy/default-policy.json`).
- `--agent-settings FILE`: Claude Code settings for the builder, for example hooks; scripts they call
  sit in the same directory.
- `--preflight SCRIPT`: checks run where the builder runs, before it starts, one JSON line each
  (`{"name": ..., "pass": true, "observed": "..."}`); any failure stops the run.

## Example tasks

The builder learns the framework from the tasks in its Infrathon checkout, and adds its task next to
them. By default that is the whole of Infrathon, from the `infrathon/` submodule; update it with
`git submodule update --remote infrathon`. To give the builder only some tasks,
`python3 reference/make_reference.py --source <Infrathon checkout> --out <dir>` exports the tasks listed in
`reference/tasks.json`, trims registries and tables to match, and keeps the tree's own tests passing;
pass `--reference <dir>` to `init`.

## When something goes wrong

| status | meaning | what to do |
|---|---|---|
| `paused` | the run's GPUs are busy, or the disk is nearly full | free them, then `run` again |
| `failed` | setup failed: the Infrathon checkout's own tests, a preflight check, or a clean clone | read `runs/<run>/run.log`, fix the cause, then `run` again |
| `blocked` | the check still failed after 3 tries, a trial never started, or the builder needs the expert | read `REPORT.md` and `work/build/feedback/`; answer in the definition, then start a new run |
| `crashed` | the tool itself failed | read `runs/<run>/run.log` (and `console.log` with `--detach`); `run` resumes from the last finished step |
| `stopped` | you stopped it | `run` resumes |

## Development

```bash
pip install -e '.[test]' && pytest
```

| path | what |
|---|---|
| `infrathon_build/` | the tool: `pipeline.py` (the flow), `local.py` and `sandbox.py` (where it runs), `agents.py`, `report.py` |
| `prompts/` | what the builder is told at each step |
| `guide/` | the builder's handbook, with one environment section per way of running |
| `sandbox/` | images for `--sandbox docker`: nested Docker daemon and its authorization plugin, egress gateway, agent settings |
| `reference/` | the reference checkout generator and its default example tasks |
| `templates/` | the expert's definition template |
| `tests/` | unit tests |

## Citation

```bibtex
@article{hu2026infrathon,
  title   = {Infrathon: Benchmarking Language Agents for Real-World, Long-Horizon AI Infra Optimization},
  author  = {Hu, Zhengyu and Xiao, Zheyuan and Wang, Yike and Ivison, Hamish and Wang, Haoyu and
             Yuan, Yige and Shao, Rulin and Li, Yuetai and Lyu, Niqi and Ding, Yicheng and Yang, Shu and
             Zeng, Zhichen and Zhu, Banghua and Pei, Jiaxin and Ding, Kaize and Dasigi, Pradeep and
             Roth, Dan and Hajishirzi, Hannaneh and Smith, Noah A. and Xiao, Teng and Poovendran, Radha},
  journal = {arXiv preprint arXiv:2611.11995},
  year    = {2026}
}
```

## License

infrathon-build is released under the [PolyForm Noncommercial License 1.0.0](LICENSE): free for
research, teaching, personal use and use by noncommercial organizations. Any commercial use, such
as selling it, offering it as a paid service or using it in a commercial product, needs a separate
license from the authors.

Files under `examples/kvzip/input/attachments/` are third-party code (KVzip, MIT; kvpress,
Apache-2.0) and keep the licenses in their headers. The `infrathon/` submodule has its own license.
