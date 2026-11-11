Trial round {round} has finished. {solver} worked on your task for {hours} hours through `trial_cmd`,
from a clean clone of commit {commit}, exactly as a benchmark run would.

- The tool's summary: `{summary}` (copied below)
- The launcher's log: `{launcher_log}`
- Everything the run left (the agent's transcript, saved versions, submissions and the Judge's reports):
  `{output_dir}`, and the version store under `/work/store`

```json
{summary_text}
```

Find out what this trial says about the task, and keep task problems apart from the agent's own
limits: only the first kind should change the task.

| look for | it may mean |
|---|---|
| no valid submission at all | the task is broken, or the brief is unclear |
| the Judge or launcher failed, timed out or crashed | the task's infrastructure is broken, not the agent |
| an ordinary, honest approach was rejected | a check or the integrity review is too strict, or the brief does not state the rule |
| a sudden jump in score | a loophole: read that submission's code and look for gaming |
| the agent repeatedly misread an interface, or asked the same thing | the brief is ambiguous: quote the sentence |
| the agent used something the definition forbids, and nothing stopped it | a check is missing |
| a Judge run took far longer than the definition allows | too few submissions fit the budget |
| it reached the strong baseline at once, then stalled | the score saturates |

Read the best valid submission's code. Is it doing what the expert hoped to measure?

Fix every task problem you find in `/work/repo`, commit, and make sure `check_cmd` still passes from a
clean clone. Then write `/work/build/reviews/round-{round}.json`:

```json
{
  "decision": "ready | retrial | blocked",
  "why": "one paragraph",
  "findings": [
    {"kind": "task_problem | agent_limit | infrastructure | observation",
      "severity": "blocking | major | minor",
      "what": "what happened", "evidence": "file, version or log line",
      "fixed": "commit and what changed, or empty"}
  ]
}
```

- `ready`: no task problem is left that only another trial could confirm.
- `retrial`: you changed the task in a way that only another trial can confirm. The tool runs one if
  rounds remain.
- `blocked`: only the expert can resolve what is left. Say exactly what to ask them.

Update `/work/build/REPORT.md` with what this trial showed and what you changed, then stop.
