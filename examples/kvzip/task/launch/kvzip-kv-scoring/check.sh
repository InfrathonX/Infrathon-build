#!/usr/bin/env bash
set -euo pipefail

# Qualification check for kvzip-kv-scoring-qwen3-8b-h100-v2-scratch (exits 0 only if all pass):
#   1. build: pinned inputs, suites (frozen identities), Work and Judge images
#   2. task tests and the framework tests that cover the task
#   3. public tests in the Work image on the Work GPU: the starting stub is rejected,
#      KNorm passes every check
#   4. the starting workspace and both baselines through `sforge eval` (integrity review as
#      configured by SFORGE_LLM_JUDGE_*, then the Judge on the seven Judge GPUs):
#        starting stub -> invalid (score 0)
#        KNorm        -> valid; far below Fast KVzip; retrieval group near zero
#        Fast KVzip   -> valid (trusted reference: weights check fires, review skipped);
#                        high, and below 100 at r = 0.1
# Usage: check.sh [--work-gpu GPU] [--judge-gpus G1,...,G7]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/kvzip-kv-scoring-qwen3-8b-h100-v2-scratch"
work_gpu="${WORK_GPU:-}"; judge_gpus="${JUDGE_GPUS:-}"
while (( $# )); do
  case "$1" in
    --work-gpu) work_gpu="$2"; shift 2 ;;
    --judge-gpus) judge_gpus="$2"; shift 2 ;;
    -h|--help) sed -n 4,16p "$0"; exit 0 ;;
    *) launch_die "unknown argument: $1" ;;
  esac
done
launch_load_config ""
[[ -n "${work_gpu}" ]] || launch_die "pass --work-gpu or set WORK_GPU (the Work GPU's index or UUID)"
export WORK_GPU="${work_gpu}" JUDGE_GPUS="${judge_gpus}"
launch_require_judge_gpus 7
source "${TASK_ROOT}/scripts/lib.sh"
stamp="$(date -u +%Y%m%d-%H%M%S)"
CHECK_DIR="${LOGS_DIR}/check-${stamp}"
mkdir -p "${CHECK_DIR}"
results=()
note() { echo "== $*"; results+=("$*"); }
fail() { echo "CHECK FAILED: $*" >&2; printf '%s\n' "${results[@]}" "FAILED: $*" > "${CHECK_DIR}/summary.txt"; exit 1; }

echo "== 1. build (logs in ${CHECK_DIR})"
register_version_store
"${TASK_ROOT}/scripts/build_images.sh" > "${CHECK_DIR}/build.log" 2>&1 || { tail -30 "${CHECK_DIR}/build.log"; fail "image build"; }
note "build: images and suites current ($(jq -r .source_digest "${BUILD_DIR}/image-manifest.json" | cut -c1-12))"

echo "== 2. tests"
( cd "${REPOSITORY_ROOT}" && "${PYTHON_BIN}" -m pytest -q -p no:cacheprovider tests/test_infrathon_assets.py \
    tests/test_infrathon_registry.py tests/test_infrathon_image_record.py \
    tests/sforge/harness/test_latest_task_store_integration.py \
    tests/sforge/harness/test_llm_judge_task_policies.py tests/sforge/harness/test_capability_envelope.py \
    > "${CHECK_DIR}/framework-tests.log" 2>&1 ) || { tail -30 "${CHECK_DIR}/framework-tests.log"; fail "framework tests"; }
docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=void -e HOME=/tmp -e KVZIP_SUITE_DIR=/suites \
  -v "${TASK_ROOT}:/task:ro" -v "${SUITE_DIR}:/suites:ro" -v "${GATES_DIR}:/gates:ro" \
  --entrypoint python3 "${JUDGE_PUBLIC_IMAGE}" -B -m pytest -q -p no:cacheprovider /task/tests \
  > "${CHECK_DIR}/task-tests.log" 2>&1 || { tail -40 "${CHECK_DIR}/task-tests.log"; fail "task tests"; }
note "tests: framework $(tail -1 "${CHECK_DIR}/framework-tests.log"); task $(tail -1 "${CHECK_DIR}/task-tests.log")"

echo "== 3. public tests in the Work image"
work_uuid="$(gpu_uuid "${work_gpu}")"
exec 39>"/tmp/sforge-gpu-${work_uuid}.lock"; flock -n 39 || fail "Work GPU ${work_uuid} is locked by another run"
public() {  # public <log> <dir> <tier> <--submission|--scorer-dir>
  docker run --rm -e "NVIDIA_VISIBLE_DEVICES=${work_uuid}" --shm-size 8g -v "${KVZIP_MODEL_DIR}:/opt/models/qwen3-8b:ro" \
    -v "$2:/sub:ro" "${WORK_IMAGE}" /workspace/public_tests/run_public_tests.sh --tier "$3" "$4" /sub \
    --show-answers 0 > "$1" 2>&1
}
if public "${CHECK_DIR}/public-stub.log" "${TASK_ROOT}/images/work/workdir/kvzip_lab" one --submission; then
  fail "public tests accepted the starting stub"
fi
grep -q "NotImplementedError" "${CHECK_DIR}/public-stub.log" || fail "public tests: stub not rejected for NotImplementedError"
note "public tests, starting stub: rejected ($(grep -m1 '^INVALID' "${CHECK_DIR}/public-stub.log"))"
public "${CHECK_DIR}/public-knorm.log" "${TASK_ROOT}/baselines/knorm" quick --scorer-dir || { tail -20 "${CHECK_DIR}/public-knorm.log"; fail "public tests on KNorm"; }
note "public tests, KNorm quick tier: $(grep -m1 '^PUBLIC SCORE' "${CHECK_DIR}/public-knorm.log"); $(grep -m1 '^CHECKS' "${CHECK_DIR}/public-knorm.log")"
exec 39>&-

echo "== 4. the Judge"
for b in stub knorm fastkvzip; do
  src="${TASK_ROOT}/baselines/${b}"; [[ "${b}" == stub ]] && src="${TASK_ROOT}/images/work/workdir/kvzip_lab"
  "${PYTHON_BIN}" "${TASK_ROOT}/scripts/make_archive.py" "${src}" "${CHECK_DIR}/${b}.tar.gz" >/dev/null
done
judge() {  # judge <name> [eval_archive flags]
  local name="$1" t0="${SECONDS}"; shift
  "${TASK_ROOT}/scripts/eval_archive.sh" --judge-gpus "${judge_gpus}" --run-id "check-${stamp}-${name}" "$@" \
    "${CHECK_DIR}/${name}.tar.gz" > "${CHECK_DIR}/judge-${name}.log" 2>&1 || { tail -30 "${CHECK_DIR}/judge-${name}.log"; fail "Judge run ${name}"; }
  echo "$(( SECONDS - t0 ))" > "${CHECK_DIR}/seconds-${name}"
  local report
  report="$(ls -1d "${LOGS_DIR}/runs/check-${stamp}-${name}/${TASK_ID}/submissions/"*/report.json | tail -1)"
  [[ -f "${report}" ]] || fail "no report for ${name}"
  cp "${report}" "${CHECK_DIR}/report-${name}.json"
  echo "${report}"
}
judge stub >/dev/null
judge knorm >/dev/null
judge fastkvzip --reference-baseline fastkvzip --skip-integrity-review >/dev/null
"${PYTHON_BIN}" - "${CHECK_DIR}" <<'PY' || fail "Judge results do not match the definition's expectations (see above)"
import json, sys
d = sys.argv[1]
r = {n: json.load(open(f"{d}/report-{n}.json")) for n in ("stub", "knorm", "fastkvzip")}
m = {n: (r[n].get("metrics") or {}) for n in r}
def show(n):
    x = m[n]
    print(f"   {n:10s} valid={r[n]['valid']} score={r[n]['score']} status={x.get('evaluation_status')} "
          f"reason={x.get('rejection_reason')!r}")
    if x.get("ratio_scores"):
        print(f"              ratios={x['ratio_scores']} groups={x['group_scores']} overhead={x.get('overhead')}")
        print(f"              checks={x.get('checks')}")
    print(f"              wall time {open(f'{d}/seconds-{n}').read().strip()} s (review, Judge container, evaluation)")
for n in r:
    show(n)
problems = []
if r["stub"]["valid"] or (r["stub"]["score"] or 0) != 0:
    problems.append("starting stub must be invalid with score 0")
if m["stub"].get("evaluation_status") == "integrity_rejected":
    print("   note: the integrity review stopped the starting stub before the Judge")
elif "NotImplementedError" not in (m["stub"].get("rejection_reason") or ""):
    problems.append("starting stub must be rejected for its NotImplementedError")
for n in ("knorm", "fastkvzip"):
    if not r[n]["valid"]:
        problems.append(f"{n} must be valid")
if not problems:
    k, f = r["knorm"]["score"], r["fastkvzip"]["score"]
    if not f - k >= 30:
        problems.append(f"KNorm ({k:.2f}) must be far below Fast KVzip ({f:.2f}): expected a gap of at least 30")
    if not m["knorm"]["group_scores"]["retrieval"] <= 5:
        problems.append("KNorm's retrieval group score must be near zero (<= 5)")
    if not f >= 75:
        problems.append(f"Fast KVzip must score high (>= 75), got {f:.2f}")
    if not m["fastkvzip"]["ratio_scores"]["0.1"] < 100:
        problems.append("Fast KVzip must be below 100 at r = 0.1")
    if m["fastkvzip"]["checks"].get("weights") != "fired_reference_baseline":
        problems.append("the weights check must fire on the released gates")
for p in problems:
    print("   PROBLEM:", p)
sys.exit(1 if problems else 0)
PY
note "Judge: stub invalid; KNorm $(jq -r .score "${CHECK_DIR}/report-knorm.json"); Fast KVzip $(jq -r .score "${CHECK_DIR}/report-fastkvzip.json") -- as expected"
printf '%s\n' "${results[@]}" "PASSED" | tee "${CHECK_DIR}/summary.txt"
