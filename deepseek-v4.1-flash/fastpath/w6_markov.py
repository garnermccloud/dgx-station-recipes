"""Host side of W6's one-launch DSpark Markov loop (libw6mk.so, see w6mk.cu).

FusedMarkovLoop(w1, w2, max_num_reqs, num_steps).run(...) replaces the whole sequential loop of
DSparkSpeculator._sample_sequential after the base logits: anchor embedding + per step (bias GEMV, add, q-cache store,
Gumbel draw, next embedding). One cooperative launch: one CTA per SM, w2 slices kept in shared memory and registers
after the first step. Persistent buffers (slots, counters) have fixed addresses, so the launch replays in a CUDA graph.
"""
from __future__ import annotations

import ctypes
import os

import torch

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libw6mk.so")
RANK = 256
MAX_R = 1  # larger batches route to W5's loop (the kernel is slower from R = 4)
RES_ROWS = 384  # resident w2 rows per SM (384 x 576 B = 216 KB)


class _Args(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_void_p), ("base_s0", ctypes.c_longlong), ("base_s1", ctypes.c_longlong),
        ("input_ids", ctypes.c_void_p), ("anchor_idx", ctypes.c_void_p), ("idx_mapping", ctypes.c_void_p),
        ("sample_pos", ctypes.c_void_p), ("temp", ctypes.c_void_p), ("seeds", ctypes.c_void_p),
        ("cache", ctypes.c_void_p), ("cache_s0", ctypes.c_longlong), ("cache_s1", ctypes.c_longlong),
        ("draft_tokens", ctypes.c_void_p), ("dt_s0", ctypes.c_longlong),
        ("w1", ctypes.c_void_p), ("w2", ctypes.c_void_p), ("slots", ctypes.c_void_p), ("counters", ctypes.c_void_p),
        ("trace", ctypes.c_void_p),
        ("R", ctypes.c_int), ("S", ctypes.c_int), ("V", ctypes.c_int), ("rows_per_cta", ctypes.c_int),
        ("res_rows", ctypes.c_int), ("dbg", ctypes.c_int), ("n_active", ctypes.c_void_p),
    ]


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        L = ctypes.CDLL(LIB_PATH)
        if L.w6mk_abi() != 4 or L.w6mk_args_size() != ctypes.sizeof(_Args):
            raise RuntimeError("libw6mk ABI mismatch")
        L.w6mk_launch.argtypes = [ctypes.POINTER(_Args), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.w6mk_launch.restype = ctypes.c_int
        L.w6mk_ctas_per_sm.argtypes = [ctypes.c_int]
        L.w6mk_ctas_per_sm.restype = ctypes.c_int

        _LIB = L
    return _LIB


class FusedMarkovLoop:
    """Persistent state of the fused loop for one drafter (fixed addresses for CUDA-graph replay)."""

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, max_num_reqs: int, num_steps: int):
        assert w1.dtype == w2.dtype == torch.bfloat16 and w1.shape == w2.shape and w1.shape[1] == RANK
        assert w1.is_contiguous() and w2.is_contiguous()
        L = lib()
        self.w1, self.w2 = w1, w2
        self.V = w2.shape[0]
        self.num_steps = num_steps
        self.max_reqs = min(max_num_reqs, L.w6mk_max_reqs())
        self.max_batch = min(self.max_reqs, MAX_R)  # the largest batch run() takes
        dev = w2.device
        self.grid = torch.cuda.get_device_properties(dev).multi_processor_count
        self.rows_per_cta = -(-self.V // self.grid)
        tile = L.w6mk_tile()
        self.res_rows = min(RES_ROWS, self.rows_per_cta) // tile * tile
        self.smem_bytes = self.res_rows * L.w6mk_res_row_bytes()
        fit = L.w6mk_ctas_per_sm(self.smem_bytes)
        if fit < 1:
            raise RuntimeError(f"fused Markov kernel does not fit on an SM (occupancy {fit})")
        self.slots = torch.zeros(self.max_reqs * num_steps, dtype=torch.int64, device=dev)
        # steps the next launches run (device int read by the kernel, so CUDA-graph replays follow it); set_active
        # changes it in stream order. Default: every step.
        self.n_active = torch.full((1,), num_steps, dtype=torch.int32, device=dev)
        self._n_active_host = num_steps
        self.counters = torch.zeros(num_steps + 1, dtype=torch.int32, device=dev)

    def set_active(self, n: int) -> None:
        """Run only draft steps 0..n-1 from the next launch (or graph replay) on; stream-ordered, a no-op when n is
        already set. The caller guarantees that the steps' consumers read only the first n draft tokens / q rows."""
        n = max(1, min(int(n), self.num_steps))
        if n != self._n_active_host:
            self.n_active.fill_(n)
            self._n_active_host = n

    def supports(self, num_reqs: int) -> bool:
        return 1 <= num_reqs <= self.max_batch

    def run(self, base_logits, input_ids, anchor_idx, idx_mapping, sample_pos, temperature, seeds, logits_cache,
            draft_tokens, trace: torch.Tensor | None = None) -> None:
        """base_logits [R, S, V] bf16; idx_mapping / sample_pos [R * S] in (req, step) order; logits_cache
        [max_reqs, K >= S, V] bf16; draft_tokens [max_reqs, K >= S] int64 (rows 0..R-1, columns 0..S-1 written)."""
        R, S, V = base_logits.shape
        assert S == self.num_steps and V == self.V and self.supports(R)
        assert base_logits.dtype == torch.bfloat16 and base_logits.stride(2) == 1
        assert logits_cache.dtype == torch.bfloat16 and logits_cache.stride(2) == 1 and logits_cache.size(-1) >= V
        assert input_ids.dtype == torch.int32 and anchor_idx.dtype == torch.int64
        assert idx_mapping.dtype == torch.int32 and sample_pos.dtype == torch.int64
        assert temperature.dtype == torch.float32 and seeds.dtype == torch.int64 and draft_tokens.dtype == torch.int64
        a = _Args(
            base_logits.data_ptr(), base_logits.stride(0), base_logits.stride(1),
            input_ids.data_ptr(), anchor_idx.data_ptr(), idx_mapping.data_ptr(), sample_pos.data_ptr(),
            temperature.data_ptr(), seeds.data_ptr(),
            logits_cache.data_ptr(), logits_cache.stride(0), logits_cache.stride(1),
            draft_tokens.data_ptr(), draft_tokens.stride(0),
            self.w1.data_ptr(), self.w2.data_ptr(), self.slots.data_ptr(), self.counters.data_ptr(),
            trace.data_ptr() if trace is not None else None,
            R, S, V, self.rows_per_cta, self.res_rows, 0,
            self.n_active.data_ptr(),
        )
        stream = torch.cuda.current_stream().cuda_stream
        rc = lib().w6mk_launch(ctypes.byref(a), self.grid, self.smem_bytes, ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"fused Markov launch failed (code {rc})")
