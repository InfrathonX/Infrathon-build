> Written for an earlier version of the repository (20 tasks). Some files, tests and documents it names are not in the released Infrathon. Where this disagrees with /work/repo, the repository wins.

# The portability contract

**The delivery layer must never be what limits rebuilding or reproducing a task
on another machine.**

Everything in this repository was authored on one particular box, with one
particular set of GPUs, one driver, one home directory and one egress proxy.
None of that is part of any task. A task is a scientific object: a frozen set
of inputs, a build recipe, an evaluation procedure and a scoring contract. If
an agent with a clean clone of this repository, an admissible GPU host and no
other access cannot rebuild a task and reproduce its numbers, the task is not
finished — however well it runs here.

This document is the rule set. `tests/test_portability_contract.py` is its
mechanical half: it is parametrised over every `tasks/latest/<task>/` and fails
the build on the violations it can detect by reading. The test can only catch
what a scan can catch; the rest is on whoever writes the task.

Read this before adding a task, before changing a launcher, and before
recording any number as a property of a task.

---

## Two different promises

Most confusion about portability comes from conflating these. They are
different claims, they need different evidence, and mixing them up is how a
host-bound fact ends up frozen in a scientific contract.

**Reproduce the same identity** — another host runs *the same evaluation
instance* and gets a score comparable to the one in the record. This requires
the inputs to be bit-identical: the same model snapshot at the same revision,
the same frozen holdout draw, the same corpus shards. Scores are only
comparable across hosts within one identity. A redraw of the holdout is a
*new* identity, and must be recorded as such — its scores do not belong on the
same axis as the old ones.

**Re-instantiate on new hardware** — another host with a *different* GPU builds
the task and runs it successfully, producing its own valid numbers. Here the
inputs are reconstructed from a recipe rather than copied byte-for-byte,
because some of them cannot be identical: a GPU capture, a kernel timing, a
memory peak. What carries across is the *procedure* and the *sanity band*, not
the bytes.

The line between them is the line between a **hash-bound** input and a
**recipe-bound** input. A byte hash may be frozen in a task's scientific
contract **only** if the bytes are in-tree, or deterministically generated from
public-pinned inputs on any GPU. A GPU capture is not deterministic across GPU
models, so captures are recipe-bound, never hash-bound. Freezing the hash of a
capture silently converts "re-instantiate on new hardware" into "impossible on
any other card".

---

## Six layers, and only two of them are the task

Every hardware-shaped fact recorded about a task belongs to exactly one of six
layers. The first two are the task. The rest describe a machine, a moment or an
experiment, and none of them may become an admission gate or a scoring input.

| layer | part of the task? | examples |
| --- | :---: | --- |
| **TaskCore** — frozen and hashed | yes | the workload's mathematical semantics, the ABI (dtype / shape / layout), the public and hidden case lists and their weights, the oracle and its tolerances, the metric formula and direction, the hard gates, the agent prompt |
| **Capability envelope** | yes | `compute capability >= x.y`, the memory floor, device count and interconnect class, the software ABI (torch / triton / CUDA), dtype support (BF16 / FP8 / NVFP4) |
| **Qualification** | no | a GPU model name, a coefficient of variation measured on one fleet, a timeout chosen for one card's speed |
| **Deployment** | no | GPU index or UUID, container and session names, ports, `WORK_GPU` / `JUDGE_GPU` |
| **Measurement provenance** | no | driver, image ID, clocks, calibration epoch, "measured on <date>" |
| **An absolute threshold from one card, used as a gate** | never | `>= 40 TFLOPS to pass`, `must beat 34.87 ms`, a `>= 120 TFLOP/s` bucket used as a gate rather than as agent-facing diagnostics |

The last row is less a layer than the failure mode the other five exist to
prevent: a number measured once, on one card, promoted into the definition of
the task. There is none left in this repository, and none may be added.

Three questions map onto the first three layers, and each is answered by a
different kind of thing:

* **What is this task?** TaskCore answers. Its bytes are identical on every machine.
* **May this card enter?** The capability envelope answers, as a boolean
  predicate that code evaluates automatically.
* **Can this card's scorecard be believed?** Qualification answers — and it is
  not a predicate but an experiment. Does the image build here, does the seed
  pass every gate, is the memory floor really enough, is the Judge timeout
  enough, is the replay CV of one submission small enough, does a same-run
  relative gate still have sane headroom? Only a run on the target card can say.

In one line: **TaskCore decides what is compared, the capability envelope
decides who may enter, and qualification decides whether this card's scorecard
is trustworthy.** Moving to a new card changes neither of the first two — that is
what P1–P6 buy — and always requires redoing the third, which no design can
avoid.

Qualification's output is a **measurement profile**: GPU model, compute
capability, memory, driver, image IDs, software ABI and calibration protocol.
**A raw score — TFLOPS, tok/s, ms, GB/s — is comparable only within one
measurement profile.** Tasks state this mechanically as
`portable.metrics.*.comparability = same_measurement_profile`. Across profiles
only a relative quantity compares (a ratio against a reference re-measured on
the same card in the same run), and it has to be labelled as the normalised view
it is rather than presented as a raw measurement.

---

## P1. Every evaluation input is one of three classes, and the class is explicit

| class | rule | examples |
| --- | --- | --- |
| **public-pinned** | fetched by a documented command from a public source at a pinned revision / commit / digest-pinned image | model snapshots, datasets, upstream checkouts, base images |
| **generated** | derived deterministically from the task's own frozen rules plus public-pinned inputs; the generator and its seed ship in the task; the produced instance's SHA is recorded so a rebuild can prove identity | final holdout draws, corpus splits, per-deployment captures (recipe + sanity bands, bytes may differ per GPU) |
| **in-tree** | committed inside the task directory on a Judge-only path that `scripts/validate_layout.py` proves never enters the Work build context | needle pairs, shadow rows, hidden corpus shards, frozen holdout instances |


The inputs table in the task's rebuild layer is where the class is declared.
One row per input, naming the class and the exact command or path that
produces it.

## P2. Holdout draws are frozen, not per-build

A `scripts/generate_final_manifest.py` that draws from `secrets.SystemRandom()`
produces a different final suite on every build. Two hosts running "the same"
task then score different evaluations, and no comparison between them means
anything — including a comparison against the historical record.

So:

* The qualification draw — the instance every historical score was measured on
  — is committed in-tree on the Judge-only path `judge/private/<final file>`,
  and is the default build input.
* `portable/final-suite-manifest.json` records its SHA-256 under
  `frozen_instance`. A task without a portable layer records it wherever its
  final-suite record lives, and says so in its rebuild layer: flux-moe uses
  `judge/private/MANIFEST.json`, because its task JSON's `portable` block is
  parsed as an exact-keyed capability envelope that rejects extra fields. The
  requirement is the record, not the filename.
* The generator takes `--seed`, defaulting to a value derived deterministically
  from the suite id (e.g. the first 8 bytes of `sha256("<suite_id>|<task_id>")`),
  so `--redraw` produces a *reproducible* new instance.
* A redraw is a new suite identity and must be recorded as one.

The same rule applies to fixtures the task cannot commit. An
`external_fixtures` entry in `portable/*-suite-manifest.json` must carry
`recipe_sha256` — the hash of the recipe that regenerates it. A bare `sha256`
with no in-tree bytes is a host-bound input wearing a digest, and fails.

## P3. Hardware floors are measured, and recorded with the measurement


`portable.execution_profiles[*].min_memory_bytes` in the task JSON equals the
measured peak; the spec header carries both. Judge-side peaks above 80 GiB must
be justified in the spec §6 with the reducing change identified.

Recording the measurement means recording the *card and driver it came off*.
That is why `GPU_MEMORY_ENVELOPES.md`, `docs/RUN_RESULTS.md` and
`docs/SINGLE_GPU_MIGRATION.md` legitimately name a GPU UUID and a driver
version, and are exempted from the P4 scan for exactly those two rules. The
distinction is load-bearing: naming the card a number came off is evidence,
**defaulting** to that card is host-binding.

### Three rules the first non-Hopper rebuild produced

The 2026-09-12 A6000 pass is where the floor rule above comes from, and it
produced two more that generalise to any new card:

* **A hardware literal usually lives in the reference implementation, not in the
  Judge.** Upstream kernels carry per-GPU tuning tables, hard-coded
  shared-memory budgets and fixed tile shapes. None of that is in the contract,
  but all of it can make the qualification step "the reference implementation
  passes the Judge on this card" fail. The fix is to give the reference
  implementation a **new candidate identity** — a compatibility variant, scored
  as its own thing — never to edit the Judge, because editing the Judge edits
  the task.
* **A same-run probe is portable only if its physical model is portable.**
  LMCache's v1 bandwidth probe measured a `cudaMemcpy` on the copy engine. On a
  Gen5 host that reads far above any candidate, so the gate never fired; on a
  Gen3 host it reads *below* the real rate of a kernel reading pinned host
  memory directly, and it rejected correct candidates. Re-measuring the
  reference on the target card is necessary but not sufficient — the quantity
  measured has to be the quantity the gate is about. That is why the calibration
  protocol has a v2, and why a protocol change is a new measurement-profile
  identity rather than a fix to the old one.

## P4. No host-private strings in delivered files

No absolute `/home/<user>` path, no host GPU index or UUID as a default, no
internal proxy address, no driver version and no branch name in any script,
launcher, Dockerfile, JSON or doc inside `tasks/latest/<task>/`, `launch/` or
`docs/`.

Defaults use `${HOME}` / `Path.home()`, take an env var, derive the value
(`docker network inspect bridge` for the bridge gateway), or **die with the
variable name**:

```bash
GPU_UUIDS="${ORPO_GPU_UUIDS:?set ORPO_GPU_UUIDS to the two physical GPU UUIDs
of the NVLink pair this run owns, comma separated
(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)}"
```

A default that is wrong on every machine but one is worse than no default: it
runs, and produces a result against the wrong hardware. Dying with the name of
the variable to set is a complete instruction.

Two things are *not* host-private and the test knows it:

* `/home/agent` and `/home/candidate` are accounts the images create. Inside a
  `docker run` line they are container paths, and they move with the image.
* `GPU-00000000-0000-0000-0000-000000000000` is the documented all-zero
  placeholder, not a device.

Everything else needs an exemption entry in the test with a reason string.
Exemptions are deliberate and reviewable; there is no ambient allowance.

There is no open violation carried this way today. There was one until
2026-09-13: ORPO's `scripts/sforge_dual_h100_entry.py` and
`scripts/replay_candidates.sh` defaulted to this host's GPU 6/7 UUIDs, and the
task's own tests exercised them with `ORPO_GPU_UUIDS` unset and asserted
success, so the default could only go together with a change to those tests.
A cleanup pass made both call sites die with the variable name, changed the two tests to
name a placeholder pair, and deleted the entries. The same pass removed the
remaining index defaults — flux-moe's `FLUXMOE_GPU_INDICES:-6,7`,
StreamDiffusionV2's `SDV2_GPU_INDICES:-2,3` and ORPO's `--gpu-indices` default.

Carrying a violation as an exemption entry remains the right move while one is
genuinely blocked: an exemption that documents a violation is not the same as
one that accepts a practice, and the reason string has to make clear which it
is — and say what clears it.

## P5. The rebuild layer is executable by an agent with nothing else

`docs/HOST_PREREQUISITES.md` plus the task's rebuild layer plus the task
directory must be sufficient. No tribal knowledge, no "ask the author", no
path that only resolves here.

Each task's rebuild layer is one of:

| layer | tasks |
| --- | --- |
| `tasks/latest/<task>/portable/REBUILD_PROMPT.md` | the fifteen tasks with a full portable contract (`portable/*.json`) |
| `tasks/latest/<task>/docs/REBUILD.md` | the five without one: concurrent-serving, flux-moe, ORPO, GKD, StreamDiffusionV2 |

It must contain, in order:

1. Host prerequisites and the GPU envelope.
2. **The inputs table**, one row per input with its P1 class and how to obtain it.
3. The build command.
4. The formal-run command with the frozen budget contract.
5. Artifacts and success criteria.

A rebuild an agent cannot verify is not a rebuild. The proof is a clean-clone
rebuild recorded in the task's documentation.

## P6. Enforcement

`tests/test_portability_contract.py` runs in the repo suite
(`uv run --frozen python -m pytest -q`) and fails on:

* P4 strings in the delivery layer of any task, or of `docs/` and `launch/`;
* a `generate_final_manifest.py` that draws from a system random source with no
  `--seed` path — in **every** task that has one, portable layer or not (the
  check was scoped to `portable/` tasks until 2026-09-13, which is how flux-moe
  kept a per-build draw long enough for a clean-clone rebuild to score it on a
  suite identity nothing else had used);
* an `external_fixtures` entry with no `recipe_sha256`;
* an inputs-table row still marked owner-provided (agent credentials excepted):
  since 2026-09-18 every evaluation input of every task is public-pinned,
  generated by a shipped recipe, or committed in-tree, and the test keeps it so;
* a task with no row in `GPU_MEMORY_ENVELOPES.md`;
* a rebuild layer that is missing, has no inputs table, or names no
  verification command.

Work that is mid-flight is marked `xfail` with the package that will clear it,
never deleted from the test — the marker is the record that the gap is known
and owned. A strict `xfail` turns into a visible failure the moment the gap
closes, so the marker cannot outlive the work.

Anything exempt carries a reason string in the test. If you find yourself
wanting a new exemption, first check whether what you actually have is a
measurement record (legitimate, P3) or a host-bound default (not, P4).
