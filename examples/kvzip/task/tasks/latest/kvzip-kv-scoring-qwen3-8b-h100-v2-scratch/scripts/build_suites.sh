#!/usr/bin/env bash
# Build the public and hidden suites from the pinned inputs (assets/fetch.sh) inside the
# task's dependency image, and verify them against judge/suite_manifest.json.
# Idempotent: suites already present with the frozen identities are kept.
#   --refreeze   write the identities of a new build into judge/suite_manifest.json
#                (a new suite identity; never needed for an ordinary rebuild)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
refreeze=0
[[ "${1:-}" == "--refreeze" ]] && refreeze=1
MANIFEST="${TASK_ROOT}/judge/suite_manifest.json"

suite_sha() {
  "${PYTHON_BIN}" -c 'import gzip,hashlib,sys; print(hashlib.sha256(gzip.open(sys.argv[1]).read()).hexdigest())' "$1"
}
frozen() { "${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["sha256"])' "${MANIFEST}" "$1"; }

if (( ! refreeze )) && [[ -f "${PUBLIC_SUITE}" && -f "${HIDDEN_SUITE}" ]] \
   && [[ "$(suite_sha "${PUBLIC_SUITE}")" == "$(frozen public)" && "$(suite_sha "${HIDDEN_SUITE}")" == "$(frozen hidden)" ]]; then
  echo "suites: present and matching judge/suite_manifest.json"
  exit 0
fi
docker image inspect "${DEPS_IMAGE}" >/dev/null 2>&1 || die "missing ${DEPS_IMAGE}; run scripts/build_images.sh"
for input in "${KVZIP_MODEL_DIR}/tokenizer.json" "${ESSAYS_DIR}"; do
  [[ -e "${input}" ]] || die "missing input ${input}; run assets/fetch.sh"
done
HUB="$(hub_cache)"
tmp="$(mktemp -d "${ASSET_DIR}/.suites.XXXXXX")"
trap 'rm -rf "${tmp}"' EXIT
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
  -v "${TASK_ROOT}/judge:/task/judge:ro" -v "${TASK_ROOT}/images/work/runtime:/task/images/work/runtime:ro" \
  -v "${KVZIP_MODEL_DIR}:/inputs/model:ro" -v "${HUB}:/inputs/hub:ro" -v "${ESSAYS_DIR}:/inputs/essays:ro" \
  -v "${tmp}:/out" --entrypoint python3 "${DEPS_IMAGE}" -B /task/judge/suite_builder/build_suites.py \
  --tokenizer /inputs/model --hf-cache /inputs/hub --essays /inputs/essays \
  --needles /task/judge/private/needles.json --out /out
pub="$(suite_sha "${tmp}/public/public_suite.jsonl.gz")"
hid="$(suite_sha "${tmp}/judge/hidden_suite.jsonl.gz")"
if (( refreeze )); then
  "${PYTHON_BIN}" - "${tmp}/judge/build_report.json" "${MANIFEST}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
out = {k: {"sha256": v["sha256"], "rows": v["rows"], "questions": v["questions"]} for k, v in report.items()}
out["note"] = ("sha256 of the uncompressed JSON lines written by judge/suite_builder/build_suites.py; "
               "a different value is a different suite identity")
open(sys.argv[2], "w").write(json.dumps(out, indent=2, sort_keys=True) + "\n")
PY
  echo "froze public ${pub} hidden ${hid} into judge/suite_manifest.json"
else
  [[ "${pub}" == "$(frozen public)" ]] || die "public suite ${pub} differs from judge/suite_manifest.json"
  [[ "${hid}" == "$(frozen hidden)" ]] || die "hidden suite ${hid} differs from judge/suite_manifest.json"
fi
mkdir -p "${SUITE_DIR}/public" "${SUITE_DIR}/judge"
chmod 700 "${SUITE_DIR}/judge"
install -m 644 "${tmp}/public/public_suite.jsonl.gz" "${PUBLIC_SUITE}"
install -m 600 "${tmp}/judge/hidden_suite.jsonl.gz" "${HIDDEN_SUITE}"
install -m 600 "${tmp}/judge/build_report.json" "${SUITE_DIR}/judge/build_report.json"
echo "suites: public ${pub:0:12} hidden ${hid:0:12} (matching judge/suite_manifest.json)"
