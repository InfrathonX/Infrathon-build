You are building a new benchmark task for Infrathon from an expert's definition.

Read `/guide/README.md` first. It describes this machine, what an Infrathon task is made of, the
rules, and the handoff file the tool reads. Then read the expert's definition in
`/input/TASK_DEFINITION.md` and everything under `/input/attachments/`.

`/work/repo` is an Infrathon checkout with a few example tasks. Add the new task there, in the same
framework, and commit it. `/work/build` is yours for notes and the handoff.

## What to deliver

When you stop, all of this must be true:

1. The task, with its rebuild layer and reference solutions, its launcher and its registry entries are
   committed on `main` in `/work/repo`.
2. `/work/build/TASK.json` tells the tool how to check the task and how to run a trial (see the guide).
3. You ran the task's `check_cmd` yourself, from a clean clone of your HEAD, and it passed. It builds the
   images, runs the public tests, and puts the starting workspace and the expert's baselines through
   the real Judge. Their scores match what the expert expects, or the report explains why not.
4. `/work/build/REPORT.md` is written for the expert: what you built, every decision you made for them
   with its reason, what you measured, and what is not done or not enforced.

All GPUs on this machine are yours while you work. Test on them as you go.

## What happens next

After you stop, the tool runs `check_cmd` again from a clean clone. If it fails, you get the log and
fix it. Then it runs a trial: a different agent attempts your task through `trial_cmd` for a few
hours, as a real benchmark run would. You will then see what happened, and fix what the trial shows is
wrong with the task.

Work until the task is complete. Do not stop early to ask; record decisions in the report instead.
