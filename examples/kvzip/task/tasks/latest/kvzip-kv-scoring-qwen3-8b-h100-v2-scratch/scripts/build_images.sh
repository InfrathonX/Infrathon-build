#!/usr/bin/env bash
# Build this task's images from source (idempotent; Docker's layer cache makes a rebuild quick):
#   1. fetch and verify the pinned inputs (assets/fetch.sh)
#   2. the dependency image, then the suites (scripts/build_suites.sh, frozen identities)
#   3. Work image (with the public suite), public Judge, Judge holdout layer (hidden suite, gate)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_command docker
"${PYTHON_BIN}" "${TASK_ROOT}/scripts/validate_layout.py" "${TASK_ROOT}"
register_version_store
"${TASK_ROOT}/assets/fetch.sh"
mkdir -p "${BUILD_DIR}"
docker build --pull=false --target deps -t "${DEPS_IMAGE}" "${TASK_ROOT}/images/work"
"${TASK_ROOT}/scripts/build_suites.sh"
docker build --pull=false --target work --build-context "public_suite=${SUITE_DIR}/public" \
  -t "${WORK_IMAGE}" "${TASK_ROOT}/images/work"
docker build --pull=false --build-arg "WORK_IMAGE=${WORK_IMAGE}" -t "${JUDGE_PUBLIC_IMAGE}" "${TASK_ROOT}/images/judge"
private_ctx="${BUILD_DIR}/private-context"
rm -rf "${private_ctx}"
mkdir -p "${private_ctx}" && chmod 700 "${private_ctx}"
trap 'rm -rf "${private_ctx}"' EXIT
cp "${HIDDEN_SUITE}" "${TASK_ROOT}/judge/suite_manifest.json" "${GATES_DIR}/qwen3-8b/q4_dim16_sink16.pt" "${private_ctx}/"
docker build --pull=false -f "${TASK_ROOT}/images/judge/Dockerfile.private" \
  --build-arg "JUDGE_PUBLIC_IMAGE=${JUDGE_PUBLIC_IMAGE}" --build-context "private=${private_ctx}" \
  -t "${JUDGE_IMAGE}" "${TASK_ROOT}/images/judge"
work_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
cat > "${BUILD_DIR}/image-manifest.json" <<JSON
{"schema_version": 1, "source_digest": "$(image_source_digest)", "work": {"name": "${WORK_IMAGE}", "id": "${work_id}"},
 "judge": {"name": "${JUDGE_IMAGE}", "id": "${judge_id}"}}
JSON
echo "built ${WORK_IMAGE} (${work_id:7:12}) and ${JUDGE_IMAGE} (${judge_id:7:12})"
