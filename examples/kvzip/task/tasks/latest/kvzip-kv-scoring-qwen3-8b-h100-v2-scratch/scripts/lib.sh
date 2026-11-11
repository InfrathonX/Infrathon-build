#!/usr/bin/env bash
# Shared paths and helpers for this task's scripts.  Sourced, not executed.
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
find_repository_root() {
  local candidate="${TASK_ROOT}"
  while [[ "${candidate}" != "/" ]]; do
    if [[ -f "${candidate}/pyproject.toml" && -d "${candidate}/sforge" ]]; then
      printf '%s\n' "${candidate}"; return 0
    fi
    candidate="$(dirname "${candidate}")"
  done
  echo "ERROR: no repository root above ${TASK_ROOT}" >&2; exit 1
}
REPOSITORY_ROOT="${REPOSITORY_ROOT:-$(find_repository_root)}"
TASK_DIRECTORY="$(basename "${TASK_ROOT}")"
TASK_ID="kvzip_kv_scoring_qwen3_8b_h100_scratch_v2"
BENCHMARK_NAME="kvzip_kv_scoring"
PYTHON_BIN="${PYTHON_BIN:-${REPOSITORY_ROOT}/.venv/bin/python}"
SFORGE_ENTRY="${TASK_ROOT}/scripts/sforge_kvzip_entry.py"
TASKS_DIR="${TASK_ROOT}/tasks"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPOSITORY_ROOT}/outputs}"
LOGS_DIR="${LOGS_DIR:-${OUTPUT_ROOT}/${TASK_DIRECTORY}}"
BUILD_DIR="${LOGS_DIR}/build"

DEPS_IMAGE="${BENCHMARK_NAME}.deps.${TASK_ID}:local"
WORK_IMAGE="${BENCHMARK_NAME}.work.${TASK_ID}:local"
JUDGE_PUBLIC_IMAGE="${BENCHMARK_NAME}.judge-public.${TASK_ID}:local"
JUDGE_IMAGE="${BENCHMARK_NAME}.judge.${TASK_ID}:local"

# Inputs (assets/manifest.json; scripts/infrathon/assets.py fetch puts them here).
INFRATHON_ASSETS_ROOT="${INFRATHON_ASSETS_ROOT:-${HOME}/.cache/infrathon/assets}"
ASSET_DIR="${INFRATHON_ASSETS_ROOT}/${TASK_DIRECTORY}"
KVZIP_MODEL_DIR="${KVZIP_MODEL_DIR:-${ASSET_DIR}/public/models/qwen3-8b}"
GATES_DIR="${ASSET_DIR}/public/fast-kvzip"
ESSAYS_DIR="${ASSET_DIR}/public/kvzip/data/needle/PaulGrahamEssays"
# Generated suites (scripts/build_suites.sh; identities frozen in judge/suite_manifest.json).
SUITE_DIR="${ASSET_DIR}/generated"
PUBLIC_SUITE="${SUITE_DIR}/public/public_suite.jsonl.gz"
HIDDEN_SUITE="${SUITE_DIR}/judge/hidden_suite.jsonl.gz"
export INFRATHON_ASSETS_ROOT KVZIP_MODEL_DIR SFORGE_TASKS_DIR="${TASKS_DIR}" SFORGE_LOG_DIR="${LOGS_DIR}"

die() { echo "ERROR: $*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

hub_cache() {
  "${PYTHON_BIN}" "${REPOSITORY_ROOT}/scripts/infrathon/assets.py" hub-cache
}

# GPU index or UUID -> UUID (nvidia-smi numbering).
gpu_uuid() {
  local sel="$1" uuid
  if [[ "${sel}" =~ ^GPU- ]]; then uuid="${sel}"; else
    [[ "${sel}" =~ ^[0-9]+$ ]] || die "GPU selector must be an index or a UUID: ${sel}"
    uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${sel}" 2>/dev/null | tr -d '[:space:]')"
  fi
  [[ "${uuid}" =~ ^GPU-[0-9a-fA-F-]+$ ]] || die "no GPU for selector ${sel}"
  printf '%s\n' "${uuid}"
}

gpu_lock() {  # gpu_lock FD UUID: hold the host-wide lock of one GPU for this process
  local fd="$1" uuid="$2"
  eval "exec ${fd}>\"/tmp/sforge-gpu-${uuid}.lock\""
  flock -n "${fd}" || die "GPU ${uuid} is in use by another run (lock /tmp/sforge-gpu-${uuid}.lock)"
}

assert_gpu_idle() {
  local uuid="$1" role="$2" free
  free="$(nvidia-smi --id="${uuid}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free >= 70000 )) || die "${role} GPU ${uuid} has only ${free} MiB free"
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -Fxq "${uuid}"; then
    die "${role} GPU ${uuid} already runs a compute process"
  fi
}

wait_for_http() {
  local url="$1" attempts="${2:-60}"
  for _ in $(seq 1 "${attempts}"); do
    curl --connect-timeout 2 --max-time 5 -fsS "${url}" >/dev/null 2>&1 && return 0
    sleep 1
  done
  die "service did not become healthy: ${url}"
}

image_source_digest() {
  ( cd "${TASK_ROOT}" && find images judge/suite_manifest.json -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
      | LC_ALL=C sort -z | xargs -0 sha256sum ) | sha256sum | awk '{print $1}'
}

register_version_store() {  # register this task in SFORGE_VERSION_STORE_ROOT (idempotent)
  local store="${SFORGE_VERSION_STORE_ROOT:-}"
  [[ -n "${store}" ]] || return 0
  [[ -f "${store}/.sforge-version-store.json" ]] || die "SFORGE_VERSION_STORE_ROOT=${store} is not a version store"
  local reg="${store}/tasks/${TASK_DIRECTORY}/task.json"
  if [[ ! -f "${reg}" ]]; then
    mkdir -p "$(dirname "${reg}")"
    printf '{"schema_version": 1, "task_directory": "%s", "task_id": "%s"}\n' "${TASK_DIRECTORY}" "${TASK_ID}" > "${reg}"
    echo "registered ${TASK_ID} in version store ${store}"
  fi
}
