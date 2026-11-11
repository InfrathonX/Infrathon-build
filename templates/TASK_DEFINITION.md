# <Task name>: <one line on what the agent optimizes>

Category: <training | inference>, <operator | module | system> level. Start: <scratch (empty) | seed (a working baseline)>.

<!--
A task is τ = (W, P, E, R, T): the workspace the agent starts from, the public tests it can run,
the private tests it cannot see, the resources, and the time budget. Fill in every section. Where
you leave something open, the builder decides and tells you what it decided and why.
Put supporting files in attachments/: public/ for files the agent may see, hidden/ for Judge-only
files. Never put hidden test data in public/.
-->

## 1. Task (W)

What the agent builds or optimizes, and why it is hard. The interface it implements: function
signatures, inputs and outputs, what it may change and what is fixed.

What the agent starts from, and how long it has.

Rules: what is forbidden (for example: no precomputed answers, no weights someone else trained, no
behavior that differs when not measured). Each rule becomes a Judge check or an integrity-review rule.

## 2. Evaluation protocol

How one test case is run, end to end, the same for public and private tests: inputs, the exact
procedure, what is measured, tolerances.

## 3. Public tests (P)

| workload | source | selection | cases | sizes |
|---|---|---|---:|---|

Tiers the agent can run (for example a one-case smoke test, a quick subset, the full public suite).

## 4. Private tests (E)

### 4.1 Cases

| workload | source | selection | cases | sizes |
|---|---|---|---:|---|

How private cases are drawn, and how they stay disjoint from public ones. Seeds, and what may not be
redistributed (the builder rebuilds such data from its sources and freezes it).

### 4.2 Metric

The per-case metric, how it is aggregated into one score, and its direction. What an invalid
submission scores.

### 4.3 Checks

Every condition a submission must meet to be valid: correctness, cost or time limits, memory,
side effects, forbidden content, integrity review.

### 4.4 Feedback

What each submission returns to the agent, and what it must not reveal.

## 5. Resources (R)

| environment | allocation |
|---|---|
| Work | GPUs, CPUs, memory |
| Judge | GPUs, CPUs, memory |

Software versions, and what must not be installed. How long one Judge run may take.

## 6. Time (T)

The agent's budget, and any limit on how often it may submit.

## 7. Baselines

| baseline | definition | expected result |
|---|---|---|

The builder runs each one through the real Judge and checks the result against this column.

## 8. Attachments

| file | what it is |
|---|---|

## 9. Sources

| what | url, with a revision where there is one |
|---|---|
