# The machine: a shared host

You work directly on a host machine, as the user who started the tool, in the run's directory
`/work`. Other people's jobs may run on this machine too, so the rules below protect them as well
as you.

- **Stay inside `/work`.** Write only there and in `/tmp`; read the expert's input and this guide
  where the table says. Your home is `/work/home`, so configuration you write (git, tools) stays in
  the run. Do not change anything in the user's own home or elsewhere on the machine.
- **Docker** is the host's daemon, shared with other jobs. Build, run and remove your own images and
  containers freely; leave every other image, container, volume and network alone, and never run
  `docker system prune` or remove images you did not build.
- **GPUs:** this run's GPUs are **{gpus}** (as `nvidia-smi` numbers them). Use only those: set
  `CUDA_VISIBLE_DEVICES` for processes you start, and select them by UUID with
  `-e NVIDIA_VISIBLE_DEVICES=<uuid>` for containers, exactly as the harness does. Other GPUs belong
  to other jobs. The tool stops your processes when you stop, and runs its checks and trials on the
  same GPUs.
- **Network:** ordinary internet access.
- **Disk is shared and limited.** The run pauses if the host runs low. Delete images and outputs you
  no longer need.
- **Python:** `/work/repo/.venv` has the framework installed. Run its tests with
  `.venv/bin/python -m pytest -q` from `/work/repo`.
- **The integrity review** (an LLM reading each submission against the brief) needs credentials that
  the tool passes only to its checks and trials. Test with `SFORGE_LLM_JUDGE_MODE=off`; the tool runs
  `check_cmd` and trials with the review on.
