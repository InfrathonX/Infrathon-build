#!/usr/bin/env bash
# Run the planted submissions (tests/planted/README.md) through the real Judge and compare each
# outcome with the check it was written to trigger.  Integrity review skipped (these are
# deliberate control archives).  Usage: run_planted.sh --judge-gpus G1,...,G7 [case ...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
judge_gpus="${JUDGE_GPUS:-}"; cases=()
while (( $# )); do
  case "$1" in
    --judge-gpus) judge_gpus="$2"; shift 2 ;;
    *) cases+=("$1"); shift ;;
  esac
done
[[ -n "${judge_gpus}" ]] || die "pass --judge-gpus"
declare -A expect=(
  [bad_admission]="check:admission" [released_gates_fp8]="check:weights" [init_raises]="check:scorer_init"
  [bad_dtype]="check:output" [nan_output]="check:output" [modifies_inputs]="check:inputs_unchanged"
  [global_rng]="check:rng_unchanged" [spawns_process]="check:single_gpu" [other_gpus]="check:single_gpu"
  [slow]="check:cost_per_context" [memory_hog]="check:memory" [stateful]="check:stateless"
  [probe_hidden_files]="reason:PermissionError" [probe_network]="reason:OSError"
  [probe_frozen_between_calls]="reason:KeyError" [forged_reply]="reason:sent an unexpected reply"
  [leaky_exception_name]="reason:raised an exception of a type outside Python, PyTorch, NumPy and Triton"
  [leaky_shape]="reason:score shape is not [36, 8, context tokens]"
  [outside_pytorch_memory]="reason:held more than 19 GiB of device memory")
(( ${#cases[@]} )) || cases=("${!expect[@]}")
stamp="$(date -u +%Y%m%d-%H%M%S)"
work="${LOGS_DIR}/planted-${stamp}"
mkdir -p "${work}"
build_case() {  # build_case NAME -> archive path on stdout
  local name="$1" dir="${work}/src-${name}"
  case "${name}" in
    bad_admission)
      mkdir -p "${dir}"; cp "${TASK_ROOT}/baselines/knorm/scorer.py" "${dir}/"
      printf '\x7fELF\x02\x01\x01' > "${dir}/fast_kernels.so" ;;
    released_gates_fp8)
      mkdir -p "${dir}"; cp "${TASK_ROOT}/baselines/knorm/scorer.py" "${dir}/"
      docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=void --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "${GATES_DIR}:/gates:ro" -v "${dir}:/out" --entrypoint python3 "${DEPS_IMAGE}" -c '
import numpy as np, torch
from safetensors.torch import save_file
safe = [(np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"), np.ndarray, np.dtype,
        np.dtypes.Float64DType, np.dtypes.Float32DType]
with torch.serialization.safe_globals(safe):
    state = torch.load("/gates/qwen3-8b/q4_dim16_sink16.pt", weights_only=True, map_location="cpu")["module"]
out = {}
for i, layer in enumerate(state):
    for k in ("q_proj.weight", "k_proj.weight"):
        w = layer[k].float(); s = w.abs().amax(1, keepdim=True) / 448
        out[f"g{i}_{k[0]}"] = (w / s).to(torch.float8_e4m3fn); out[f"g{i}_{k[0]}_scale"] = s
save_file(out, "/out/my_trained_gate.safetensors")' ;;
    *) dir="${TASK_ROOT}/tests/planted/${name}" ;;
  esac
  "${PYTHON_BIN}" "${TASK_ROOT}/scripts/make_archive.py" "${dir}" "${work}/${name}.tar.gz" >/dev/null
  echo "${work}/${name}.tar.gz"
}
summary="${work}/summary.txt"
: > "${summary}"
for name in "${cases[@]}"; do
  [[ -n "${expect[${name}]:-}" ]] || die "unknown case ${name}"
  archive="$(build_case "${name}")"
  t0=$(date +%s)
  "${TASK_ROOT}/scripts/eval_archive.sh" --judge-gpus "${judge_gpus}" --run-id "planted-${stamp}-${name//_/-}" \
    --skip-integrity-review "${archive}" > "${work}/${name}.log" 2>&1 || true
  report="$(ls -1 "${LOGS_DIR}/runs/planted-${stamp}-${name//_/-}/${TASK_ID}/submissions/"*/report.json 2>/dev/null | tail -1)"
  verdict="$("${PYTHON_BIN}" - "${report:-/nonexistent}" "${expect[${name}]}" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except OSError:
    print("NO-REPORT"); sys.exit()
m = r.get("metrics") or {}
kind, want = sys.argv[2].split(":", 1)
reason = m.get("rejection_reason") or ""
if kind == "check":
    ok = (not r.get("valid")) and (m.get("checks") or {}).get(want) == "fail"
else:
    ok = (not r.get("valid")) and want in reason
ok = ok and "LEAK" not in reason and "7777" not in reason  # the leaky cases' data never reaches the agent
print(("PASS" if ok else "UNEXPECTED") + f" valid={r.get('valid')} reason={reason!r}")
PY
)"
  printf '%-28s %-34s %4ss  %s\n' "${name}" "${expect[${name}]}" "$(( $(date +%s) - t0 ))" "${verdict}" | tee -a "${summary}"
done
echo "summary: ${summary}"
