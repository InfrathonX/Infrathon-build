"""Detect the released Fast KVzip gates in a submission, renamed, reshaped, cast or quantized.

Every tensor in every weight file of the submission is read (safetensors, torch
weights_only, numpy without pickles).  Each tensor contributes candidate rows of
4096 values:
  * its flat order cut into rows (any reshape that keeps element order), aligned at the
    start and at the end of the tensor;
  * its rows and, when its second-to-last dimension is at least 4096, its columns: the first
    and the last 4096 values of each (a copy padded with extra inputs, such as 4098-wide
    gate layers, still lines up at one end);
  * for 8-bit tensors, the bytes as signed and unsigned integers, as the two 4-bit nibble
    streams (packed int4, unsigned and two's complement) and as fp8 (e4m3, e5m2); for 16-bit
    integer tensors, the bits as fp16 and bf16.
A candidate row matches a released projection row (q_proj/k_proj, 23,040 rows of the
hidden size) when their rank (Spearman) correlation is at least ROW_R.  Rank correlation
keeps casting, rounding, per-row or per-group scaling and int4/int8/fp8 quantization far
above the threshold, while two vectors that merely share Qwen3's few massive-activation
coordinates (for example a mean hidden state, whose Pearson correlation with some
released rows exceeds 0.9 for that reason alone) stay far below it.  The submission is
rejected when at least MIN_MATCHED_ROWS distinct released rows are matched by at least
MIN_MATCHED_ROWS distinct candidate rows (a few vectors matching many released rows are
not a copy).
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".npy", ".npz"}
WIDTH = 4096
ROW_R = 0.8
MIN_MATCHED_ROWS = 64
MAX_TOTAL_ELEMENTS = 1 << 31  # refuse decompression bombs


class UnreadableWeights(Exception):
    pass


# numpy containers torch.load(weights_only=True) may rebuild (no code execution); the released
# gate file itself stores its training curves this way.
SAFE_NUMPY = [(np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"), np.ndarray, np.dtype,
              np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Float16DType, np.dtypes.Int64DType,
              np.dtypes.Int32DType, np.dtypes.Int8DType, np.dtypes.UInt8DType, np.dtypes.BoolDType]


def load_released(path: str | Path) -> torch.Tensor:
    with torch.serialization.safe_globals(SAFE_NUMPY):
        state = torch.load(path, weights_only=True, map_location="cpu")["module"]
    rows = [layer[name].float() for layer in state for name in ("q_proj.weight", "k_proj.weight")]
    return torch.cat(rows, 0)  # [36 * 640, 4096]


def _tensors_from_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)
        return
    if suffix == ".npy":
        yield "array", torch.from_numpy(np.array(np.load(path, allow_pickle=False)))
        return
    if suffix == ".npz":
        with zipfile.ZipFile(path) as zf:
            if sum(i.file_size for i in zf.infolist()) > 4 * MAX_TOTAL_ELEMENTS:
                raise UnreadableWeights(f"{path.name}: archive expands beyond the size limit")
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                yield key, torch.from_numpy(np.array(data[key]))
        return
    # .pt / .pth / .bin: torch's restricted unpickler only
    try:
        with torch.serialization.safe_globals(SAFE_NUMPY):
            obj = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    except Exception as exc:  # noqa: BLE001
        try:
            from safetensors import safe_open

            with safe_open(str(path), framework="pt") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)
            return
        except Exception:
            raise UnreadableWeights(f"{path.name}: not loadable with torch.load(weights_only=True) "
                                    f"or safetensors ({type(exc).__name__})") from None
    stack = [("", obj)]
    while stack:
        prefix, item = stack.pop()
        if isinstance(item, torch.Tensor):
            yield prefix, item
        elif isinstance(item, dict):
            stack.extend((f"{prefix}.{k}", v) for k, v in item.items())
        elif isinstance(item, (list, tuple)):
            stack.extend((f"{prefix}[{i}]", v) for i, v in enumerate(item))
        elif isinstance(item, np.ndarray) and item.dtype != object:
            yield prefix, torch.from_numpy(np.array(item))


def _views(t: torch.Tensor):
    """(flat float values, shape or None) interpretations of one tensor."""
    if t.dtype in (torch.uint8, torch.int8):
        raw = t.flatten().contiguous()
        u = raw.view(torch.uint8)
        lo, hi = (u & 0xF).to(torch.int16), (u >> 4).to(torch.int16)
        yield raw.to(torch.int16).float(), t.shape                    # the integers themselves
        yield (u.to(torch.int16) if t.dtype == torch.int8 else u.view(torch.int8).to(torch.int16)).float(), t.shape
        for a, b in ((lo, hi), (hi, lo)):                              # packed int4, both nibble orders
            nib = torch.stack([a, b], 1).flatten()
            yield nib.float(), None                                    # unsigned / offset encoding
            yield torch.where(nib >= 8, nib - 16, nib).float(), None   # two's complement
        for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):           # raw fp8 bytes
            yield u.view(fp8).float(), t.shape
        return
    if t.dtype in (torch.int16, torch.uint16):                         # raw 16-bit floats?
        raw = t.flatten().contiguous().view(torch.int16)
        yield raw.float(), t.shape
        yield raw.view(torch.float16).float(), t.shape
        yield raw.view(torch.bfloat16).float(), t.shape
        return
    if t.is_complex() or t.dtype == torch.bool:
        return
    yield t.flatten().float(), t.shape


def _rows(flat: torch.Tensor, shape):
    """Candidate [rows, 4096] views: flat order (start- and end-aligned), row and column windows."""
    n = flat.numel()
    k = n // WIDTH
    if k:
        yield flat[: k * WIDTH].view(k, WIDTH)
        if n % WIDTH:
            yield flat[n - k * WIDTH:].view(k, WIDTH)
    if shape is None or len(shape) < 2:
        return
    mat = flat.view(*shape)
    mat = mat.reshape(-1, shape[-1]) if len(shape) > 2 else mat
    cols = shape[-1]
    if cols > WIDTH:                                   # rows wider than a hidden vector
        yield mat[:, :WIDTH]
        yield mat[:, cols - WIDTH:]
    if shape[-2] >= WIDTH and cols != WIDTH:           # columns as rows (transposed storage)
        m = flat.view(*shape).transpose(-1, -2).reshape(-1, shape[-2])
        yield m[:, :WIDTH]
        if shape[-2] > WIDTH:
            yield m[:, shape[-2] - WIDTH:]


def _ranks(rows: torch.Tensor) -> torch.Tensor:
    """Average ranks within each row (ties share their mean rank)."""
    rows = rows.contiguous()
    ordered, _ = rows.sort(1)
    lo = torch.searchsorted(ordered, rows, right=False).float()
    hi = torch.searchsorted(ordered, rows, right=True).float()
    return (lo + hi - 1) / 2


def _normalize(rows: torch.Tensor) -> torch.Tensor:
    rows = rows - rows.mean(1, keepdim=True)
    return rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)


def scan(submission_dir: str | Path, released: torch.Tensor, device: str = "cpu") -> dict:
    root = Path(submission_dir)
    ref = _normalize(_ranks(released.to(device)))
    matched = torch.zeros(ref.shape[0], dtype=torch.bool, device=device)
    files, elements, max_r, candidates = [], 0, 0.0, 0
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES):
        files.append(str(path.relative_to(root)))
        for _, tensor in _tensors_from_file(path):
            elements += tensor.numel()
            if elements > MAX_TOTAL_ELEMENTS:
                raise UnreadableWeights("weight files hold more elements than the scan allows")
            for flat, shape in _views(tensor.detach().cpu()):
                for rows in _rows(flat.to(device), shape):
                    for chunk in rows.split(8192):
                        chunk = torch.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
                        chunk = chunk[(chunk != chunk[:, :1]).any(1)]  # constant rows carry no signal
                        if not len(chunk):
                            continue
                        r = (_normalize(_ranks(chunk)) @ ref.T).abs()
                        max_r = max(max_r, float(r.max()))
                        hit = r >= ROW_R
                        matched |= hit.any(0)
                        candidates += int(hit.any(1).sum())
    n = int(matched.sum())
    return {"files": files, "elements": elements, "matched_released_rows": n, "matching_candidate_rows": candidates,
            "released_rows": int(ref.shape[0]), "max_row_correlation": round(max_r, 4),
            "uses_released_gates": n >= MIN_MATCHED_ROWS and candidates >= MIN_MATCHED_ROWS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--released", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    try:
        result = scan(args.submission, load_released(args.released), args.device)
    except UnreadableWeights as exc:
        result = {"unreadable": str(exc)}
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
