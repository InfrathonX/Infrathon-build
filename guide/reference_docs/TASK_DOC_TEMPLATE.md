> Written for an earlier version of the repository (20 tasks). Some files, tests and documents it names are not in the released Infrathon. Where this disagrees with /work/repo, the repository wins.

# <Task title, for example "SageAttention-derived Quantized Attention BF16 — H100 v1 (seed)">

<!-- The shape of docs/tasks/<task>.md: an `| Item | Value |` table at the top, then the overview
     sections (Task Definition / Metric / Public / Private Test / Source), then `## 1.`–`## 6.` under
     `# Detailed specification`, ending with `### Maintenance and replay`.

     The operational layer (host prerequisites, getting inputs, build, smoke, the formal run command,
     where outputs land) does not go here. It lives in the task's portable/REBUILD_PROMPT.md (a task
     without a portable layer: docs/REBUILD.md) and in docs/HOST_PREREQUISITES.md.

     Rules:
     1. Heading levels and order are fixed, as below; dated records appear only in "## 6." and after.
     2. No `file:line` references. To point at code, give the bare file name, at most one per paragraph.
     3. No host-private information (absolute /home paths, local GPU indices or UUIDs, driver versions,
        internal addresses, `.venv/bin`, branch names).
     4. For the hidden suite, write only the generation rule (grid, families, quotas, sources and pins,
        sampling strategy, seed strategy, what is redrawn at each build, which inputs each deployment
        captures itself), never the drawn instances (case ids, shapes, seeds, needle contents).
     5. Each number appears once, with its source file named; a memory floor gives both the declared
        value and the measured peak.
     6. Everything is written in English. -->

| Item | Value |
| --- | --- |
| Task directory | `tasks/latest/<task>/` |
| `task_id` | `<task_id>` |
| Upstream project | [<repo>](<url>) @ `<commit>` (<license>); a scratch task says "no upstream port (from scratch)" and names the paper or reference it follows |
| Category | GPU / Machine Learning Systems |
| Starting point | **seed** / **scratch**: one sentence on what `candidates/kernel_v001` (or the equivalent start) is |
| Hardware | one whole-H100 lease (Judge and Work take turns on the same card) / a dual NVLink pair: the **current** form |
| Base image | `<image key>` (`<docker tag>`) |
| Score | `<primary metric>` in short, its direction (`maximize`/`minimize`) + `valid_then_score` |
| Status | one sentence: whether scoring is qualified and enabled, and the card and result of the latest real-hardware check |
| GPU envelope | `execution_profiles` profile id, declared memory ≥ N GiB, measured Judge peak M GiB / Work peak K GiB (sources: the task JSON, `GPU_MEMORY_ENVELOPES.md`) |
| Rebuild and run | `tasks/latest/<task>/portable/REBUILD_PROMPT.md` (without a portable layer: `tasks/latest/<task>/docs/REBUILD.md`); host prerequisites in `HOST_PREREQUISITES.md` |

## Task Definition

The task in one sentence, then two or three paragraphs: what is optimized, why it is hard, and where
the frozen surface ends and the editable surface begins.

## Metric

```text
<formula, in a code block>
```

Direction, the validity rule, raw or relative quantity, and the number and nature of the hard gates
(one paragraph).

## Public / Private Test

A table of the grid or families (cell → geometry or regime → disclosed slot), and the three-tier
suite table:

| Tier | Contents |
| --- | --- |
| Public | case count, what is exposed, the `--quick` subset |
| Anchor (interactive) | suite id, its relation to the public suite, whether it counts in the final score |
| Private final | suite id, what is redrawn at each build; sampling happens only in the Judge |

## Source

| Component | Source | Pin |
| --- | --- | --- |
| reference implementation / paper / fixture / dataset | … | commit, revision, SHA-256 |

The upstream is a **source of evidence**: say whether it enters the Work or Judge image.

---

# Detailed specification

## 1. The upstream

The upstream project's or paper's technical claim, the headroom ladder, and lineage (which task the
harness inherits from). A scratch task says "there is no upstream implementation", and gives the
origin of the problem itself and its reference numbers instead.

## 2. The task

The frozen surface: the entry signature (code block), tensor shapes, dtypes and layouts, the
mathematical semantics, the allowed and forbidden implementation surface (import allowlist, the
restricted subset of host code, what is allowed inside a kernel), the workload grid, and the input
distribution with its calibration.

## 3. The agent's instructions

Sources: the task JSON's `agent_query` and `images/work/AGENT_TASK.md`. List what the agent is told,
item by item (ABI, precision contract, grid, allowed implementation surface, development commands,
the seed's reference numbers).

## 4. The public test

The public suite's entry command, coverage, the readings it exposes, and how it mirrors the Judge's
private report; the interactive Judge suite (`sforge-submit` / standalone anchor), its cooldown and
feedback boundary; known usability defects.

## 5. The Judge's hidden test

### 5.1 <calibration evidence for the precision or correctness contract>
### 5.2 <edge fixtures / hard gates as measured>
### 5.3 Score
### 5.4 Adversarial evidence (planted probes)

The frozen part of the final holdout and the part drawn at each build, the two-phase protocol, the
timeout ladder, the fail-closed cases, and the private artifacts.

## 6. Runs and acceptance

Qualification records, each model run (date, card, model, budget, score), incidents and fixes, oldest
first. **Dates appear only in this section and after it.**

### Changing GPUs and comparability

Which quantities change with the card (timeouts, tolerances, reference scores, profile keys,
per-deployment captured inputs, per-build holdout), what must be re-measured on the same model or a
different one, and when scores are comparable.

### Known limitations

Design boundaries and unresolved problems, one sentence each, with no dated narrative.

### History and verification

Dated entries, newest last; clean-clone rebuild records also go here.

### Maintenance and replay

```bash
TASK=tasks/latest/<task>
# command skeleton for restoring inputs / build / smoke / the formal run; details in REBUILD_PROMPT.md
```

One paragraph: which files are deliberately kept out of git, how to restore them, and the batch replay
and manifest verification scripts.
