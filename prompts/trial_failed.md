The tool started trial round {round} with `trial_cmd`, and the run never got going: {why}. This
does not count as a trial round.

It ran exactly as a maintainer would: in the clean clone of commit {commit} where `check_cmd` had just
passed, so the images and data that `check_cmd` built are there. The launcher's log is `{log}`. Its
last lines:

```text
{tail}
```

Find the cause, fix it in `/work/repo`, commit, and update `/work/build/TASK.json` if a command
changed. Make sure `check_cmd` still passes from a fresh clean clone, and that `trial_cmd` gets past
its start in that same clone. Then stop: the tool checks again and starts the trial.
