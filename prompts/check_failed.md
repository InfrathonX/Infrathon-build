The tool ran your `check_cmd` from a clean clone of commit {commit} and it failed: {why}

The full log is `{log}`. Its last lines:

```text
{tail}
```

A clean clone has only what is committed: no untracked files, no `.venv` until `uv sync` makes one, and
no state from your own runs except what lives under `/work/assets`, `/work/hf` and `/work/store`.

Find the cause, fix it in `/work/repo`, commit, and update `/work/build/TASK.json` if a command changed.
Run `check_cmd` yourself from a fresh clean clone and see it pass, then stop.
