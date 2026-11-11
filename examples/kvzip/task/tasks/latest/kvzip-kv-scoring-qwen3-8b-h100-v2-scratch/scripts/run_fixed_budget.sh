#!/usr/bin/env bash
# One benchmark run of this task (launch/kvzip-kv-scoring/run.sh delegates here):
#   the agent works in the Work container (one GPU) for TASK_TIMEOUT seconds; every
#   sforge-submit goes through the Judge server (Controller: rate limit, integrity review)
#   to a Judge container on the seven Judge GPUs; when the budget ends the workspace is
#   frozen as final_archive.tar.gz and evaluated once more the same way.
#
# Environment (launch/kvzip-kv-scoring/run.sh sets these from its flags):
#   MODEL, REASONING_EFFORT, TASK_TIMEOUT, RUN_ID, MLSYSFORGE_AGENT (codex|claude-code)
#   WORK_GPU (index or UUID), JUDGE_GPUS (seven indices or UUIDs), JUDGE_PORT
#   CODEX_AUTH_JSON_PATH | MODEL_API_BASE_URL + MODEL_API_KEY_FILE | CLAUDE_OAUTH_TOKEN_FILE
#   SFORGE_AGENT_RUNTIME_DIR, SFORGE_VERSION_STORE_ROOT, SFORGE_LLM_JUDGE_* (docs/LLM_JUDGE.md)
#   INFRATHON_ASSETS_ROOT (where assets/fetch.sh put the model)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
for c in curl docker flock jq nvidia-smi ss; do require_command "${c}"; done

AGENT_NAME="${MLSYSFORGE_AGENT:-codex}"
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
TASK_TIMEOUT="${TASK_TIMEOUT:-86400}"
EVAL_INTERVAL="${EVAL_INTERVAL:-300}"
FINAL_EVAL_TIMEOUT="${FINAL_EVAL_TIMEOUT:-3600}"
JUDGE_PORT="${JUDGE_PORT:-18761}"
WORK_GPU="${WORK_GPU:-}"
JUDGE_GPUS="${JUDGE_GPUS:-}"
AUTH_JSON="${CODEX_AUTH_JSON_PATH:-}"
CLAUDE_TOKEN_FILE="${CLAUDE_OAUTH_TOKEN_FILE:-}"
MODEL_API_BASE_URL="${MODEL_API_BASE_URL:-}"
MODEL_API_KEY_FILE="${MODEL_API_KEY_FILE:-}"
MODEL_FORWARD_PORT=3131
JUDGE_GPU_COUNT=7

[[ -n "${MODEL}" ]] || die "set MODEL (launch/kvzip-kv-scoring/run.sh --model)"
[[ "${MODEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] || die "unsafe MODEL: ${MODEL}"
[[ "${TASK_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || die "TASK_TIMEOUT must be a positive integer"
[[ "${JUDGE_PORT}" =~ ^[0-9]+$ ]] && (( JUDGE_PORT >= 1024 && JUDGE_PORT <= 65535 )) \
  && [[ "${JUDGE_PORT}" != 3128 && "${JUDGE_PORT}" != 3130 && "${JUDGE_PORT}" != "${MODEL_FORWARD_PORT}" ]] \
  || die "invalid JUDGE_PORT ${JUDGE_PORT}"
[[ -n "${WORK_GPU}" ]] || die "set WORK_GPU (--work-gpu) to the Work GPU's index or UUID"
[[ -n "${JUDGE_GPUS}" ]] || die "set JUDGE_GPUS (--judge-gpus) to the ${JUDGE_GPU_COUNT} Judge GPUs, comma separated"
case "${AGENT_NAME}" in
  codex)
    if [[ -n "${MODEL_API_BASE_URL}" ]]; then
      [[ "${MODEL_API_BASE_URL}" =~ ^http://([^/:[:space:]]+):([0-9]+)(/.*)?$ ]] \
        || die "MODEL_API_BASE_URL must be http://HOST:PORT[/path] (relayed as raw TCP by the sidecar)"
      model_api_host="${BASH_REMATCH[1]}"; model_api_port="${BASH_REMATCH[2]}"; model_api_path="${BASH_REMATCH[3]:-}"
      [[ -f "${MODEL_API_KEY_FILE}" ]] || die "set MODEL_API_KEY_FILE with MODEL_API_BASE_URL"
    else
      [[ -f "${AUTH_JSON}" ]] || die "set CODEX_AUTH_JSON_PATH (--auth) to the Codex auth.json"
    fi ;;
  claude-code) [[ -f "${CLAUDE_TOKEN_FILE}" ]] || die "set CLAUDE_OAUTH_TOKEN_FILE (--claude-token)" ;;
  *) die "MLSYSFORGE_AGENT must be codex or claude-code" ;;
esac
RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}"
[[ -f "${RUNTIME_DIR}/.sforge-agent-runtime.json" ]] \
  || die "set SFORGE_AGENT_RUNTIME_DIR to an agent runtime (launch/build-agent-runtime.sh --agent ${AGENT_NAME})"
runtime_agent="$(jq -er '.agent' "${RUNTIME_DIR}/.sforge-agent-runtime.json")"
[[ "${runtime_agent}" == "${AGENT_NAME}" ]] || die "agent runtime ${RUNTIME_DIR} is ${runtime_agent}, not ${AGENT_NAME}"

# Images built from this source and the model on the host. The build record lives in this
# checkout's outputs/, so a fresh checkout (or a changed source) builds first: build_images.sh
# is idempotent and takes seconds when Docker already holds the layers (e.g. after check.sh
# ran in another checkout of the same commit).
manifest="${BUILD_DIR}/image-manifest.json"
images_current() {
  local role name
  [[ -s "${manifest}" ]] && [[ "$(jq -er .source_digest "${manifest}")" == "$(image_source_digest)" ]] || return 1
  for role in work judge; do
    name="$(jq -er ".${role}.name" "${manifest}")"
    [[ "$(docker image inspect --format '{{.Id}}' "${name}" 2>/dev/null)" == "$(jq -er ".${role}.id" "${manifest}")" ]] \
      || return 1
  done
}
if ! images_current; then
  if [[ "${SFORGE_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
    echo "Images: not yet built from this source (a real run builds them first)"
  else
    mkdir -p "${BUILD_DIR}"
    echo "Building the images from this source (scripts/build_images.sh; log ${BUILD_DIR}/build.log)"
    "${TASK_ROOT}/scripts/build_images.sh" > "${BUILD_DIR}/build.log" 2>&1 \
      || { tail -30 "${BUILD_DIR}/build.log" >&2; die "image build failed (${BUILD_DIR}/build.log)"; }
    images_current || die "the images still do not match this source after the build"
  fi
fi
[[ -f "${KVZIP_MODEL_DIR}/model.safetensors.index.json" ]] || die "model missing at ${KVZIP_MODEL_DIR}; run assets/fetch.sh"

# The integrity review (docs/LLM_JUDGE.md) must be ready unless explicitly switched off.
review_summary="$(cd "${REPOSITORY_ROOT}" && "${PYTHON_BIN}" -B -c '
from sforge.harness.llm_judge.config import load_llm_judge_config
c = load_llm_judge_config()
c.check_ready()
print(f"{c.mode} {c.engine} {c.agent} {c.model}")' 2>&1)" || die "integrity review is not ready: ${review_summary##*$'\n'}"
echo "Integrity review: ${review_summary}"

# GPUs: one for Work, seven for the Judge, all distinct.
work_uuid="$(gpu_uuid "${WORK_GPU}")"
IFS=, read -r -a judge_sel <<<"${JUDGE_GPUS}"
(( ${#judge_sel[@]} == JUDGE_GPU_COUNT )) || die "the Judge uses exactly ${JUDGE_GPU_COUNT} GPUs; got ${#judge_sel[@]}"
judge_uuids=()
for sel in "${judge_sel[@]}"; do judge_uuids+=("$(gpu_uuid "${sel}")"); done
all="$(printf '%s\n' "${work_uuid}" "${judge_uuids[@]}" | sort -u | wc -l)"
(( all == JUDGE_GPU_COUNT + 1 )) || die "Work and Judge GPUs must be ${JUDGE_GPU_COUNT}+1 distinct GPUs"
KVZIP_JUDGE_GPU_UUIDS="$(IFS=,; echo "${judge_uuids[*]}")"
export KVZIP_WORK_GPU_UUID="${work_uuid}" KVZIP_JUDGE_GPU_UUIDS

model_slug="$(tr 'A-Z' 'a-z' <<<"${MODEL}" | sed -E 's/[^a-z0-9.]+/-/g; s/^-+//; s/-+$//')"
RUN_ID="${RUN_ID:-kvzip-kv-scoring-${model_slug}-${REASONING_EFFORT}-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe RUN_ID: ${RUN_ID}"
RUN_ROOT="${LOGS_DIR}/runs/${RUN_ID}"
TASK_RUN_ROOT="${RUN_ROOT}/${TASK_ID}"

if [[ "${SFORGE_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run: ${RUN_ID}"
  echo "  agent ${AGENT_NAME} ${MODEL} effort=${REASONING_EFFORT} runtime ${RUNTIME_DIR}"
  echo "  budget ${TASK_TIMEOUT}s; submission cooldown ${EVAL_INTERVAL}s; final Judge timeout ${FINAL_EVAL_TIMEOUT}s"
  echo "  Work GPU ${work_uuid}; Judge GPUs ${KVZIP_JUDGE_GPU_UUIDS}; Judge port ${JUDGE_PORT}"
  echo "  logs ${RUN_ROOT}"
  [[ ! -e "${RUN_ROOT}" ]] || die "run directory already exists: ${RUN_ROOT}"
  exit 0
fi

fd=40
for u in "${work_uuid}" "${judge_uuids[@]}"; do gpu_lock "${fd}" "${u}"; fd=$((fd + 1)); done
assert_gpu_idle "${work_uuid}" "Work"
for u in "${judge_uuids[@]}"; do assert_gpu_idle "${u}" "Judge"; done
[[ ! -e "${RUN_ROOT}" ]] || die "run directory already exists: ${RUN_ROOT}"
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${JUDGE_PORT}$"; then die "Judge port ${JUDGE_PORT} is in use"; fi
register_version_store
mkdir -p "${RUN_ROOT}"

judge_pid=""; network=""; proxy_name=""
cleanup() {
  [[ -n "${judge_pid}" ]] && kill "${judge_pid}" 2>/dev/null && wait "${judge_pid}" 2>/dev/null || true
  docker rm -f "sforge.run.${TASK_ID}.${RUN_ID}" >/dev/null 2>&1 || true
  [[ -n "${proxy_name}" ]] && docker rm -f "${proxy_name}" >/dev/null 2>&1 || true
  [[ -n "${network}" ]] && docker network rm "${network}" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export SFORGE_ADMIN_SECRET
SFORGE_ADMIN_SECRET="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_urlsafe(48))')"
unset SFORGE_JUDGE_EXTRA_ENV
bridge_gateway="$(docker network inspect bridge | jq -er '.[0].IPAM.Config[0].Gateway')"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${bridge_gateway},127.0.0.1,localhost" no_proxy="${bridge_gateway},127.0.0.1,localhost"
"${PYTHON_BIN}" "${SFORGE_ENTRY}" --log-dir "${LOGS_DIR}" --tasks-dir "${TASKS_DIR}" \
  serve --host "${bridge_gateway}" --host 127.0.0.1 --port "${JUDGE_PORT}" >"${RUN_ROOT}/judge-server.log" 2>&1 &
judge_pid="$!"
wait_for_http "http://127.0.0.1:${JUDGE_PORT}/openapi.json" 120

# Work sits on an internal network; its only exit is a CONNECT sidecar that tunnels to public
# hosts on 80/443 (internet: true in the task JSON), refuses internal addresses, and relays the
# Judge port.  SForge's own iptables isolation is not used.
network="${RUN_ID}-internal"
proxy_name="${RUN_ID}-proxy"
docker network create --driver bridge --internal --label "infrathon.run_id=${RUN_ID}" "${network}" >/dev/null
model_forward=()
[[ -n "${MODEL_API_BASE_URL}" ]] && model_forward=(--model-forward "${MODEL_FORWARD_PORT}:${model_api_host}:${model_api_port}")
docker run --detach --rm --name "${proxy_name}" --network bridge --env NVIDIA_VISIBLE_DEVICES=void \
  --volume "${SCRIPT_DIR}/restricted_proxy.py:/opt/restricted_proxy.py:ro" --entrypoint python3 \
  "${WORK_IMAGE}" /opt/restricted_proxy.py --allow-public-egress \
  --evaluation-upstream-host "${bridge_gateway}" --evaluation-upstream-port "${JUDGE_PORT}" \
  "${model_forward[@]}" >/dev/null
docker network connect --alias runtime-proxy "${network}" "${proxy_name}"
proxy_ip="$(docker inspect "${proxy_name}" | jq -er --arg n "${network}" '.[0].NetworkSettings.Networks[$n].IPAddress')"
export NO_PROXY="${proxy_ip},${NO_PROXY}" no_proxy="${proxy_ip},${NO_PROXY}"
probe() { docker run --rm --network "${network}" --env NVIDIA_VISIBLE_DEVICES=void --entrypoint "$1" "${WORK_IMAGE}" "${@:2}"; }
for _ in $(seq 1 30); do
  probe curl --connect-timeout 2 --max-time 3 -fsS "http://${proxy_ip}:3128/health" >/dev/null 2>&1 && break
  sleep 1
done
probe curl --connect-timeout 2 --max-time 5 -fsS "http://${proxy_ip}:3130/openapi.json" >/dev/null \
  || die "the Judge relay on the sidecar does not reach the Judge server"
if probe curl --connect-timeout 2 --max-time 4 -fsS https://example.com >/dev/null 2>&1; then
  die "the internal Work network unexpectedly has direct egress"
fi
probe python3 -c 'import socket,sys
p=sys.argv[1]
def connect(host, expected):
    with socket.create_connection((p,3128),5) as s:
        s.sendall(f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n".encode())
        line=s.recv(256).split(b"\r\n",1)[0]
        if expected not in line: raise SystemExit(f"{host}: expected {expected!r}, got {line!r}")
connect("github.com", b" 200 ")
connect(sys.argv[2], b" 403 ")' "${proxy_ip}" "${bridge_gateway}" || die "CONNECT sidecar preflight failed"
echo "Network: internal ${network}, CONNECT sidecar ${proxy_ip} (public 80/443 only), Judge relay :3130"

export SFORGE_ISOLATED_DOCKER_NETWORK="${network}" MLSYSFORGE_WORK_INTERNET=true
export SFORGE_HTTP_PROXY="http://${proxy_ip}:3128" SFORGE_HTTPS_PROXY="http://${proxy_ip}:3128"
export SFORGE_NO_PROXY="${proxy_ip},127.0.0.1,localhost"
export SFORGE_AGENT_RUNTIME_DIR="${RUNTIME_DIR}" SFORGE_CODEX_REASONING_EFFORT="${REASONING_EFFORT}"
export SFORGE_CODEX_CLI_VERSION="$(jq -er '.version' "${RUNTIME_DIR}/.sforge-agent-runtime.json")"
unset SFORGE_AGENT_API_KEY SFORGE_AGENT_API_BASE_URL OPENAI_API_KEY OPENAI_BASE_URL CODEX_API_KEY
if [[ -n "${MODEL_API_BASE_URL}" ]]; then
  export SFORGE_AGENT_API_BASE_URL="http://${proxy_ip}:${MODEL_FORWARD_PORT}${model_api_path}"
  SFORGE_AGENT_API_KEY="$(tr -d '[:space:]' < "${MODEL_API_KEY_FILE}")"; export SFORGE_AGENT_API_KEY
elif [[ "${AGENT_NAME}" == claude-code ]]; then
  export MLSYSFORGE_CLAUDE_OAUTH_TOKEN_FILE="${CLAUDE_TOKEN_FILE}"
else
  export SFORGE_CODEX_AUTH_JSON_PATH="${AUTH_JSON}"
fi
export MLSYSFORGE_AGENT="${AGENT_NAME}" MLSYSFORGE_AGENT_MODEL="${MODEL}" MLSYSFORGE_AGENT_EFFORT="${REASONING_EFFORT}"
export MLSYSFORGE_AGENT_BUDGET_SECONDS="${TASK_TIMEOUT}" MLSYSFORGE_RUN_LOG_ROOT="${RUN_ROOT}"

jq -n --arg run_id "${RUN_ID}" --arg task_id "${TASK_ID}" --arg agent "${AGENT_NAME}" --arg model "${MODEL}" \
  --arg effort "${REASONING_EFFORT}" --argjson timeout "${TASK_TIMEOUT}" --argjson cooldown "${EVAL_INTERVAL}" \
  --arg work_gpu "${work_uuid}" --arg judge_gpus "${KVZIP_JUDGE_GPU_UUIDS}" \
  --arg work_image "$(jq -er .work.id "${manifest}")" --arg judge_image "$(jq -er .judge.id "${manifest}")" \
  --arg started "$(date -u +%FT%TZ)" \
  '{schema_version: 1, run_id: $run_id, task_id: $task_id, agent: $agent, model: $model, effort: $effort,
    timeout_seconds: $timeout, submission_cooldown_seconds: $cooldown, scheduled_auto_evaluation: false,
    work_gpu: $work_gpu, judge_gpus: ($judge_gpus | split(",")), images: {work: $work_image, judge: $judge_image},
    started_at: $started}' >"${RUN_ROOT}/experiment-manifest.json"

shim=()
if [[ "${AGENT_NAME}" == claude-code ]]; then
  shim=(-P -c '
import runpy, sys
entry = sys.argv[1]
sys.argv = [entry, *sys.argv[2:]]
module = runpy.run_path(entry, run_name="sforge_task_entry")
from sforge.harness.agent.claude_local import install_from_environment
install_from_environment(module["_task_internet"], sys.argv)
module["main"]()
')
fi
echo "Starting ${RUN_ID}: ${AGENT_NAME} ${MODEL} (${REASONING_EFFORT}), ${TASK_TIMEOUT}s; logs ${RUN_ROOT}"
set +e
"${PYTHON_BIN}" "${shim[@]}" "${SFORGE_ENTRY}" --log-dir "${LOGS_DIR}" --tasks-dir "${TASKS_DIR}" run \
  --backend docker --task "${TASK_ID}" --agent "${AGENT_NAME}" --model "${MODEL}" --timeout "${TASK_TIMEOUT}" \
  --eval-interval "${EVAL_INTERVAL}" --disable-auto-eval --submission-cooldown "${EVAL_INTERVAL}" \
  --judge-url "http://${proxy_ip}:3130" --enable-internet --work-cpu-limit 8 --work-mem-limit 32g \
  --judge-cpu-limit 56 --judge-mem-limit 88g --run-id "${RUN_ID}" "$@" 2>&1 | tee "${RUN_ROOT}/sforge-run.log"
run_rc="${PIPESTATUS[0]}"
set -e
kill "${judge_pid}" 2>/dev/null; wait "${judge_pid}" 2>/dev/null || true; judge_pid=""
docker rm -f "sforge.run.${TASK_ID}.${RUN_ID}" >/dev/null 2>&1 || true

final_archive="${TASK_RUN_ROOT}/final_archive.tar.gz"
final_rc=1
if [[ -s "${final_archive}" ]]; then
  # The workspace frozen at the deadline, judged once more like a submission (review + Judge).
  set +e
  "${PYTHON_BIN}" "${SFORGE_ENTRY}" --log-dir "${LOGS_DIR}" --tasks-dir "${TASKS_DIR}" eval --backend docker \
    --task "${TASK_ID}" --archive "${final_archive}" --run-id "${RUN_ID}" --timeout "${FINAL_EVAL_TIMEOUT}" \
    --judge-cpu-limit 56 --judge-mem-limit 88g --json >"${RUN_ROOT}/final-eval.log" 2>&1
  final_rc="$?"
  set -e
else
  echo "WARNING: no final_archive.tar.gz; the run kept only its submissions" >&2
fi
"${PYTHON_BIN}" - "${TASK_RUN_ROOT}" "${RUN_ROOT}/completion.json" "${run_rc}" "${final_rc}" <<'PY'
import json, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
reports = []
for path in sorted(root.glob("submissions/*/report.json")):
    try:
        r = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    reports.append({"round": path.parent.name, "valid": r.get("valid"), "score": r.get("score"),
                    "status": (r.get("metrics") or {}).get("evaluation_status")})
valid = [r for r in reports if r["valid"] is True and isinstance(r["score"], (int, float))]
best = max(valid, key=lambda r: r["score"]) if valid else None
summary = {"schema_version": 1, "sforge_run_exit": int(sys.argv[3]), "final_eval_exit": int(sys.argv[4]),
           "submissions": reports, "best_valid": best}
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"best_valid": best, "submissions": len(reports)}))
PY
(( run_rc == 0 )) || die "SForge run failed with exit ${run_rc}"
