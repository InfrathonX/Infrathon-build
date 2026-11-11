# The machine: a sandbox

You work in a sandbox of your own.

- **You are a non-root user** without sudo. Install tools with `uv`, or run them in a container. Your
  home is `/work/home`.
- **Docker** is yours alone: it starts empty, and nothing else shares it. Builds, GPU containers and
  everything the harness does work. Bind mounts must come from under `/work` or `/tmp`. Privileged
  containers, host PID, extra kernel capabilities and unconfined security profiles are refused; no
  task needs them. Check `df -h /work` before pulling or generating anything large.
- **GPUs:** `nvidia-smi` shows them. Docker's default runtime is NVIDIA, so containers select GPUs with
  `-e NVIDIA_VISIBLE_DEVICES=<uuid>`, exactly as the harness does. All of them are yours while you work.
  The tool stops your containers when you stop, and runs its checks and trials on the same GPUs.
- **Network:** outbound HTTP and HTTPS work, from your shell and from containers. Other ports do not,
  so use `https://` git remotes. Some addresses may be blocked by policy and return 403 or a
  certificate error. If that happens, move on.
- **Disk is limited.** The run pauses if the host runs low. Delete images and outputs you no longer
  need, but never `docker system prune -a`: base images are slow to pull again.
- **Python:** `/work/repo/.venv` has the framework installed. Run its tests with
  `.venv/bin/python -m pytest -q` from `/work/repo`.
- **The integrity review** (an LLM reading each submission against the brief) needs credentials that
  only the tool's checks and trials have. Test with `SFORGE_LLM_JUDGE_MODE=off`; the tool runs
  `check_cmd` and trials with the review on.
