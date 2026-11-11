#!/usr/bin/env bash
set -euo pipefail

# Unified experiment entrypoint for kvzip-kv-scoring-qwen3-8b-h100-v2-scratch.
# Eight GPUs per run: Work on one (--work-gpu), the Judge on seven (--judge-gpus);
# the Judge measures while Work keeps going.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/kvzip-kv-scoring-qwen3-8b-h100-v2-scratch"
ENTRY="${TASK_ROOT}/scripts/run_fixed_budget.sh"

usage() {
  cat <<'USAGE'
Usage:
  run.sh --model MODEL [options] [-- extra]

Options:
  --agent NAME        codex (default) or claude-code
  --model NAME        A full gpt-*/claude-* model ID, or any ID the MODEL_API_BASE_URL endpoint serves
  --effort LEVEL      Reasoning effort (max|ultra|xhigh|high|medium|low); default max
  --hours N           Agent budget in hours (sets TASK_TIMEOUT=N*3600; default 24 h)
  --timeout SECONDS   Agent budget in seconds (overrides --hours)
  --run-id ID         Explicit run ID (default: task-derived)
  --auth FILE         Codex auth.json path (sets CODEX_AUTH_JSON_PATH)
  --claude-token FILE 0600 file with one `claude setup-token` OAuth token (claude-code)
  --work-gpu GPU      GPU of the Work container, index or UUID (or WORK_GPU; required)
  --judge-gpus LIST   The seven Judge GPUs, comma-separated indices or UUIDs (or JUDGE_GPUS; required)
  --judge-port PORT   Host port of the Judge server (default 18761)
  --dry-run           Resolve and check the plan, print it and exit; no lock, container or GPU work
  --tmux              Run detached in a new tmux session
  --config FILE       Launch config (.env or .json); see launch/mlsysforge.env.example
  -- ...              Everything after -- is passed to `sforge run`

Before the first run: tasks/latest/kvzip-kv-scoring-qwen3-8b-h100-v2-scratch/scripts/build_images.sh
(fetches the pinned inputs into INFRATHON_ASSETS_ROOT, builds the suites and the images).
The integrity review must be configured (SFORGE_LLM_JUDGE_*, docs/LLM_JUDGE.md).
USAGE
}

agent="codex"; claude_token=""; model=""; effort=""; timeout_seconds=""; hours=""; run_id=""; auth=""
work_gpu=""; judge_gpus=""; judge_port=""; config_file=""; use_tmux=0; dry_run=0; passthrough=()
while (( $# )); do
  case "$1" in
    --agent)   [[ $# -ge 2 ]] || launch_die "--agent requires a value"; agent="$2"; shift 2 ;;
    --claude-token) [[ $# -ge 2 ]] || launch_die "--claude-token requires a value"; claude_token="$2"; shift 2 ;;
    --model)   [[ $# -ge 2 ]] || launch_die "--model requires a value"; model="$2"; shift 2 ;;
    --effort)  [[ $# -ge 2 ]] || launch_die "--effort requires a value"; effort="$2"; shift 2 ;;
    --hours)   [[ $# -ge 2 ]] || launch_die "--hours requires a value"; hours="$2"; shift 2 ;;
    --timeout) [[ $# -ge 2 ]] || launch_die "--timeout requires a value"; timeout_seconds="$2"; shift 2 ;;
    --run-id)  [[ $# -ge 2 ]] || launch_die "--run-id requires a value"; run_id="$2"; shift 2 ;;
    --auth)    [[ $# -ge 2 ]] || launch_die "--auth requires a value"; auth="$2"; shift 2 ;;
    --work-gpu) [[ $# -ge 2 ]] || launch_die "--work-gpu requires a value"; work_gpu="$2"; shift 2 ;;
    --judge-gpus) [[ $# -ge 2 ]] || launch_die "--judge-gpus requires a value"; judge_gpus="$2"; shift 2 ;;
    --judge-port) [[ $# -ge 2 ]] || launch_die "--judge-port requires a value"; judge_port="$2"; shift 2 ;;
    --config)  [[ $# -ge 2 ]] || launch_die "--config requires a value"; config_file="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --tmux)    use_tmux=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --)        shift; passthrough=("$@"); break ;;
    *)         usage >&2; launch_die "unknown argument: $1" ;;
  esac
done

launch_load_config "${config_file}"
[[ -x "${ENTRY}" ]] || launch_die "task entry script is missing: ${ENTRY}"
export WORK_GPU="${work_gpu:-${WORK_GPU:-}}"
export JUDGE_GPUS="${judge_gpus:-${JUDGE_GPUS:-}}"
[[ -n "${WORK_GPU}" ]] || launch_die "pass --work-gpu or set WORK_GPU (the Work GPU's index or UUID)"
launch_require_judge_gpus 7
if [[ -n "${judge_port}" ]]; then
  [[ "${judge_port}" =~ ^[0-9]+$ ]] || launch_die "--judge-port must be an integer"
  export JUDGE_PORT="$((10#${judge_port}))"
fi
export JUDGE_PORT="${JUDGE_PORT:-18761}"
[[ -n "${model}" ]] && { MODEL="$(resolve_model_id "${model}")"; export MODEL; }
effort="${effort:-${REASONING_EFFORT:-max}}"
validate_reasoning_effort "${effort}"
export REASONING_EFFORT="${effort}"
if [[ -n "${timeout_seconds}" ]]; then
  [[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] || launch_die "invalid --timeout"
  export TASK_TIMEOUT="${timeout_seconds}"
elif [[ -n "${hours}" ]]; then
  [[ "${hours}" =~ ^[1-9][0-9]*$ ]] || launch_die "invalid --hours"
  export TASK_TIMEOUT="$(( hours * 3600 ))"
elif [[ -z "${TASK_TIMEOUT:-}" ]]; then
  export TASK_TIMEOUT="86400"
fi
[[ -z "${run_id}" ]] || export RUN_ID="${run_id}"
case "${agent}" in
  codex)
    export MLSYSFORGE_AGENT=codex
    if [[ -n "${auth}" ]]; then
      [[ -f "${auth}" ]] || launch_die "auth file does not exist: ${auth}"
      export CODEX_AUTH_JSON_PATH="${auth}"
    fi
    [[ -n "${CODEX_AUTH_JSON_PATH:-}" || -n "${MODEL_API_BASE_URL:-}" ]] \
      || launch_die "codex needs --auth FILE (or CODEX_AUTH_JSON_PATH, or MODEL_API_BASE_URL)"
    export CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-${CODEX_CLI_VERSION_PIN}}"
    export_agent_runtime_dir ;;
  claude-code)
    export MLSYSFORGE_AGENT=claude-code  # --auth (a Codex credential) is not used
    if [[ -n "${claude_token}" ]]; then
      [[ -f "${claude_token}" ]] || launch_die "Claude token file does not exist: ${claude_token}"
      export CLAUDE_OAUTH_TOKEN_FILE="${claude_token}"
    fi
    [[ -n "${CLAUDE_OAUTH_TOKEN_FILE:-}" ]] || launch_die "claude-code needs --claude-token FILE (or CLAUDE_OAUTH_TOKEN_FILE)"
    export_claude_runtime_dir ;;
  *) launch_die "--agent takes codex or claude-code" ;;
esac
assert_container_reachable_proxy --entry "${ENTRY}"
export_version_store_root --task-root "${TASK_ROOT}"
(( ! dry_run )) || export SFORGE_LAUNCH_DRY_RUN=1

if (( use_tmux && ! dry_run )); then
  stamp="$(date -u +%Y%m%d-%H%M%S)"
  run_in_tmux "kvzip-kv-scoring-${stamp}" "${SCRIPT_DIR}/logs/run-${stamp}.log" "${ENTRY}" "${passthrough[@]}"
else
  exec "${ENTRY}" "${passthrough[@]}"
fi
