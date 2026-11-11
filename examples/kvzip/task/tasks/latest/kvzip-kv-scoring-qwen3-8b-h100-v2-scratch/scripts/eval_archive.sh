#!/usr/bin/env bash
# Score one submission archive through the real Judge path (`sforge eval`: the integrity
# review, then the Judge container on the Judge GPUs).
#
#   eval_archive.sh --judge-gpus G1,...,G7 [--run-id ID] [--skip-integrity-review]
#                   [--reference-baseline fastkvzip] ARCHIVE
#
# --reference-baseline fastkvzip is the trusted host channel for the Fast KVzip baseline: the
# Judge copies the released gate into the scorer's tree and reports the weights check as fired
# instead of rejecting.  It reaches the Judge only through the private evaluation context.
# The report lands in ${LOGS_DIR}/runs/<run id>/<task id>/submissions/manual-N/report.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
judge_gpus="${JUDGE_GPUS:-}"; run_id=""; skip_review=(); reference=""; archive=""
while (( $# )); do
  case "$1" in
    --judge-gpus) judge_gpus="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --skip-integrity-review) skip_review=(--skip-integrity-review); shift ;;
    --reference-baseline) reference="$2"; shift 2 ;;
    -h|--help) sed -n 2,13p "$0"; exit 0 ;;
    *) [[ -z "${archive}" ]] || die "unexpected argument $1"; archive="$1"; shift ;;
  esac
done
[[ -f "${archive}" ]] || die "archive not found: ${archive}"
[[ -n "${judge_gpus}" ]] || die "set --judge-gpus (or JUDGE_GPUS) to the Judge GPUs (indices or UUIDs)"
[[ -z "${reference}" || "${reference}" == fastkvzip ]] || die "--reference-baseline takes fastkvzip"
docker image inspect "${JUDGE_IMAGE}" >/dev/null 2>&1 || die "missing ${JUDGE_IMAGE}; run scripts/build_images.sh"
uuids=()
IFS=, read -r -a sels <<<"${judge_gpus}"
fd=40
for sel in "${sels[@]}"; do
  u="$(gpu_uuid "${sel}")"; uuids+=("${u}")
  gpu_lock "${fd}" "${u}"; fd=$((fd + 1))
  assert_gpu_idle "${u}" "Judge"
done
export KVZIP_JUDGE_GPU_UUIDS="$(IFS=,; echo "${uuids[*]}")"
run_id="${run_id:-eval-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe run id ${run_id}"
if [[ -n "${reference}" ]]; then
  export SFORGE_JUDGE_EXTRA_ENV="SFORGE_PRIVATE_CONTEXT_KVZIP_REFERENCE_BASELINE=${reference}"
else
  unset SFORGE_JUDGE_EXTRA_ENV
fi
mkdir -p "${LOGS_DIR}"
"${PYTHON_BIN}" "${SFORGE_ENTRY}" --log-dir "${LOGS_DIR}" --tasks-dir "${TASKS_DIR}" eval --backend docker \
  --task "${TASK_ID}" --archive "${archive}" --run-id "${run_id}" --timeout 3600 \
  --judge-cpu-limit 56 --judge-mem-limit 88g "${skip_review[@]}" --json
