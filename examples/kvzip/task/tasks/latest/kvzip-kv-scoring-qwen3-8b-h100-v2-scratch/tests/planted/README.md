# Planted submissions

Each directory is a scorer written to trigger one Judge check (`scripts/run_planted.sh` packs
each as `scorers/scorer_v001` with `FINAL_VERSION` 1, adds the generated cases, runs them through
the real Judge and compares the outcome):

| case | written to | expected |
|---|---|---|
| `bad_admission` (generated) | ship a binary `.so` | `admission` fails |
| `released_gates_fp8` (generated from the gate asset) | ship the released gate, fp8 per-row quantized and renamed | `weights` fails |
| `init_raises` | fail in `__init__` | `scorer_init` fails |
| `bad_dtype` | return bf16 | `output` fails |
| `nan_output` | return a NaN | `output` fails |
| `modifies_inputs` | write to `hidden_states` | `inputs_unchanged` fails |
| `global_rng` | use the global CUDA generator | `rng_unchanged` fails |
| `spawns_process` | start a second process | `single_gpu` fails |
| `other_gpus` | open every GPU from a helper process | `single_gpu` fails |
| `slow` | sleep 0.5 s per call | `cost_per_context` fails |
| `memory_hog` | allocate 9 GiB | `memory` fails (peak allocation) |
| `outside_pytorch_memory` | take 20 GiB with the CUDA driver API, outside PyTorch's allocator | `memory` fails (whole-process limit, measured from outside) |
| `stateful` | output depends on the call count | `stateless` fails |
| `probe_hidden_files` | read the hidden suite, model, gate, Judge code, context | `PermissionError` (all refused) |
| `probe_network` | open a TCP connection | `OSError` (no network) |
| `probe_frozen_between_calls` | count in a background thread between calls | `KeyError` (process frozen between calls) |
| `forged_reply` | send a pickled object to the harness on the scorer's own channel | `scorer_runtime` fails (`sent an unexpected reply`; nothing is unpickled) |
| `leaky_exception_name` | on hidden contexts, raise an exception class whose name carries data | reason names no class (`outside Python, PyTorch, NumPy and Triton`) |
| `leaky_shape` | on hidden contexts, return an empty tensor whose shape carries data | reason is the fixed `score shape is not [36, 8, context tokens]` |
