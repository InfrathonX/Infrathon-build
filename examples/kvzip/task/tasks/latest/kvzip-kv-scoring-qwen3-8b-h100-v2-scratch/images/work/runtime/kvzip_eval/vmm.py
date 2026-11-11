"""Device buffers shared with the scorer process.

Allocated with the CUDA driver's virtual-memory API and exported as a POSIX file
descriptor, so a process of another user can map them (legacy CUDA IPC
handles refuse that).  Mapped memory is not part of the importer's torch
allocator statistics.
"""
import ctypes, torch

_LIB = None


class _Lazy:
    """libcuda is loaded on first use (it exists only where a GPU driver is mounted)."""

    def __getattr__(self, name):
        global _LIB
        if _LIB is None:
            _LIB = ctypes.CDLL("libcuda.so.1")
        return getattr(_LIB, name)


_cu = _Lazy()
CUresult = ctypes.c_int
class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]
class _AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]
class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int),
                ("location", CUmemLocation), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", _AllocFlags)]
class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]

PINNED, POSIX_FD, LOC_DEVICE, RW, GRAN_MIN = 1, 1, 1, 3, 0

def _check(r, what):
    if r != 0:
        s = ctypes.c_char_p()
        _cu.cuGetErrorString(r, ctypes.byref(s))
        raise RuntimeError(f"{what} failed: {r} {s.value}")

def _prop(dev):
    p = CUmemAllocationProp()
    p.type = PINNED; p.requestedHandleTypes = POSIX_FD
    p.location.type = LOC_DEVICE; p.location.id = dev
    return p

def granularity(dev=0):
    g = ctypes.c_size_t()
    _check(_cu.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(_prop(dev)), GRAN_MIN), "granularity")
    return g.value

class _CAI:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3, "strides": None}

class SharedBuffer:
    """Owner side: create(nbytes); importer side: open(fd, nbytes)."""
    def __init__(self, handle, ptr, size, dev, fd=None):
        self.handle, self.ptr, self.size, self.dev, self.fd = handle, ptr, size, dev, fd
        self.tensor = torch.as_tensor(_CAI(ptr, size), device=f"cuda:{dev}")

    @staticmethod
    def _map(handle, size, dev):
        ptr = ctypes.c_uint64()
        _check(_cu.cuMemAddressReserve(ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_size_t(0), ctypes.c_uint64(0), ctypes.c_uint64(0)), "reserve")
        _check(_cu.cuMemMap(ptr, ctypes.c_size_t(size), ctypes.c_size_t(0), handle, ctypes.c_uint64(0)), "map")
        acc = CUmemAccessDesc(); acc.location.type = LOC_DEVICE; acc.location.id = dev; acc.flags = RW
        _check(_cu.cuMemSetAccess(ptr, ctypes.c_size_t(size), ctypes.byref(acc), ctypes.c_size_t(1)), "setaccess")
        return ptr.value

    @classmethod
    def create(cls, nbytes, dev=0):
        torch.cuda.init(); torch.zeros(1, device=f"cuda:{dev}")
        g = granularity(dev)
        size = (nbytes + g - 1) // g * g
        handle = ctypes.c_uint64()
        _check(_cu.cuMemCreate(ctypes.byref(handle), ctypes.c_size_t(size), ctypes.byref(_prop(dev)), ctypes.c_uint64(0)), "create")
        ptr = cls._map(handle, size, dev)
        fd = ctypes.c_int()
        _check(_cu.cuMemExportToShareableHandle(ctypes.byref(fd), handle, POSIX_FD, ctypes.c_uint64(0)), "export")
        return cls(handle, ptr, size, dev, fd.value)

    @classmethod
    def open(cls, fd, size, dev=0):
        torch.cuda.init(); torch.zeros(1, device=f"cuda:{dev}")
        handle = ctypes.c_uint64()
        _check(_cu.cuMemImportFromShareableHandle(ctypes.byref(handle), ctypes.c_void_p(fd), POSIX_FD), "import")
        ptr = cls._map(handle, size, dev)
        return cls(handle, ptr, size, dev)
