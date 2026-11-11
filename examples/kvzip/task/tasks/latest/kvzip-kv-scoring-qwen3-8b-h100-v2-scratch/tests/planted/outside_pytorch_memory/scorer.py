import ctypes

import torch


class Scorer:
    """KNorm, plus 20 GiB of device memory taken with the CUDA driver API, outside PyTorch's
    allocator (and its statistics): only the outside measurement of the process can see it."""

    def __init__(self, config):
        cuda = ctypes.CDLL("libcuda.so.1")
        self.hidden_buffer = ctypes.c_uint64()
        rc = cuda.cuMemAlloc_v2(ctypes.byref(self.hidden_buffer), ctypes.c_size_t(20 << 30))
        if rc != 0:
            raise RuntimeError(f"cuMemAlloc failed: {rc}")

    def score(self, states):
        keys = states.keys[:, :, states.context_start:, :]
        return -torch.linalg.vector_norm(keys, dim=-1, dtype=torch.float32)
