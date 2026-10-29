"""Host wrapper for libw6tp.so (w6tp.cu): exact top-p masking of fp32 logits in place, one thread-block cluster per
row. apply_top_p(logits, p) masks exactly the tokens the sort-based definition masks (descending logit, ties by index,
keep while the mass before a token is < p), with order-independent 46-bit fixed-point mass sums."""
from __future__ import annotations

import ctypes
import os

import torch

_LIB = None
MASK_VALUE = float("-inf")


class _Args(ctypes.Structure):
    _fields_ = [("logits", ctypes.c_void_p), ("ld", ctypes.c_longlong), ("p", ctypes.c_void_p), ("B", ctypes.c_int),
                ("V", ctypes.c_int), ("slice", ctypes.c_int), ("mask_value", ctypes.c_float), ("src", ctypes.c_void_p),
                ("src_ld", ctypes.c_longlong), ("src_bf16", ctypes.c_int)]


def lib():
    global _LIB
    if _LIB is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libw6tp.so")
        L = ctypes.CDLL(path)
        if L.w6tp_abi() != 3 or L.w6tp_args_size() != ctypes.sizeof(_Args):
            raise RuntimeError("libw6tp ABI mismatch")
        L.w6tp_launch.argtypes = [ctypes.POINTER(_Args), ctypes.c_int, ctypes.c_void_p]
        L.w6tp_launch.restype = ctypes.c_int
        _LIB = L
    return _LIB


_MAXC: dict[tuple[int, int], int] = {}


def max_clusters(C: int, V: int) -> int:
    """Clusters of C CTAs the GPU holds at once (cudaOccupancyMaxActiveClusters: GPC placement, not just SM count)."""
    k = (C, V)
    if k not in _MAXC:
        lib().w6tp_max_clusters.argtypes = [ctypes.c_int, ctypes.c_int]
        _MAXC[k] = int(lib().w6tp_max_clusters(C, V))
    return _MAXC[k]


def cluster_size(rows: int, num_sms: int, V: int = 129280) -> int:
    """CTAs per row: the largest of 16 / 8 / 6 / 5 whose clusters all fit on the GPU at once (one wave, by the
    occupancy query), else 4 (fewer, longer CTAs beat a second wave; GB300 bench at 24-48 rows)."""
    for c in (16, 8, 6, 5):
        if rows * c <= num_sms and rows <= max_clusters(c, V):
            return c
    return 4


_SMS: dict[int, int] = {}


def apply_top_p(logits: torch.Tensor, p: torch.Tensor, mask_value: float = MASK_VALUE,
                out: torch.Tensor | None = None) -> torch.Tensor:
    """Top-p mask by per-row p [B] (fp32). Without `out`: logits [B, V] fp32 (unit stride along V) is masked in place
    and returned. With `out` (fp32 [B, V], unit stride): logits may be bf16 or fp32 and is left untouched; out receives
    logits.float() with the masked entries set to mask_value (the copy apply_sampling_params makes, fused)."""
    assert logits.ndim == 2 and logits.stride(1) == 1 and logits.is_cuda
    assert p.dtype == torch.float32 and p.shape == (logits.shape[0],) and p.is_contiguous()
    B, V = logits.shape
    if B == 0:
        return logits if out is None else out
    if out is None:
        assert logits.dtype == torch.float32
        dst, src, src_ld, src_bf16 = logits, None, 0, 0
    else:
        assert out.dtype == torch.float32 and out.shape == logits.shape and out.stride(1) == 1
        assert logits.dtype in (torch.bfloat16, torch.float32)
        dst, src, src_ld, src_bf16 = out, logits.data_ptr(), logits.stride(0), int(logits.dtype == torch.bfloat16)
    dev = logits.device.index
    if dev not in _SMS:
        _SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    a = _Args(dst.data_ptr(), dst.stride(0), p.data_ptr(), B, V, 0, mask_value, src, src_ld, src_bf16)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    C = cluster_size(B, _SMS[dev], V)
    rc = lib().w6tp_launch(ctypes.byref(a), C, stream)
    if rc != 0 and C != 8:  # a cluster shape the GPU cannot place: launch errors are not sticky, retry portable 8
        rc = lib().w6tp_launch(ctypes.byref(a), 8, stream)
    if rc != 0:
        raise RuntimeError(f"w6tp launch failed (code {rc})")
    return dst
