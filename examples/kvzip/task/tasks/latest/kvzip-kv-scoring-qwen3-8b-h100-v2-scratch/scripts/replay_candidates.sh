#!/usr/bin/env bash
# Re-score a closed run's saved candidates through the Judge (sforge replay-candidates), or each
# retained scorers/scorer_vNNN once (--per-version: sforge replay-versions; the Judge honors the
# host's candidate_version).  Holds the Judge GPUs' locks for the whole serial batch.
#
#   replay_candidates.sh --origin-run RUN_ID --measurement-set NAME --version-store-root DIR \
#       --judge-gpus G1,...,G7 [--per-version] [--candidate vN]... [--from-v N] [--to-v N]
#       [--every N] [--repeat N] [--resume] [--replay-run-id ID] [--timeout S] [--dry-run] [--json]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
origin_run=""; measurement_set=""; store=""; judge_gpus="${JUDGE_GPUS:-}"; per_version=0; dry_run=0
pass=(); selection=()
while (( $# )); do
  case "$1" in
    --origin-run|--source-run-id) origin_run="$2"; shift 2 ;;
    --measurement-set) measurement_set="$2"; shift 2 ;;
    --version-store-root) store="$2"; shift 2 ;;
    --judge-gpus) judge_gpus="$2"; shift 2 ;;
    --per-version) per_version=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --json) pass+=(--json); shift ;;
    --replay-run-id|--timeout|--repeat) pass+=("$1" "$2"); shift 2 ;;
    --candidate|--from-v|--to-v|--every) selection+=("$1" "$2"); shift 2 ;;
    --resume) selection+=(--resume); shift ;;
    -h|--help) sed -n 2,8p "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "${origin_run}" && -n "${measurement_set}" && -n "${store}" ]] \
  || die "--origin-run, --measurement-set and --version-store-root are required"
for value in "${origin_run}" "${measurement_set}"; do
  [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$ ]] || die "unsafe identifier: ${value}"
done
subcommand=replay-candidates
if (( per_version )); then
  subcommand=replay-versions
  (( ${#selection[@]} == 0 )) || die "--per-version does not take candidate selection flags"
fi
cmd=("${PYTHON_BIN}" "${SFORGE_ENTRY}" --log-dir "${LOGS_DIR}" --tasks-dir "${TASKS_DIR}" "${subcommand}"
     --backend docker --task "${TASK_ID}" --run-id "${origin_run}" --measurement-set "${measurement_set}"
     --version-store-root "${store}" --judge-cpu-limit 56 --judge-mem-limit 88g "${pass[@]}" "${selection[@]}")
(( ! dry_run )) || exec "${cmd[@]}" --dry-run
[[ -n "${judge_gpus}" ]] || die "pass --judge-gpus (the seven Judge GPUs)"
uuids=()
IFS=, read -r -a sels <<<"${judge_gpus}"
fd=40
for sel in "${sels[@]}"; do
  u="$(gpu_uuid "${sel}")"; uuids+=("${u}"); gpu_lock "${fd}" "${u}"; fd=$((fd + 1)); assert_gpu_idle "${u}" "Judge"
done
export KVZIP_JUDGE_GPU_UUIDS="$(IFS=,; echo "${uuids[*]}")"
unset SFORGE_JUDGE_EXTRA_ENV
exec "${cmd[@]}"
