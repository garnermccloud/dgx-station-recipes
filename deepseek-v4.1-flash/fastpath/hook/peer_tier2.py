"""Experimental peer tier v2: compacted cold rows for any batch size (decode and prefill).

A sidecar on the RTX PRO 6000 holds every layer's cold experts in VRAM and runs them with b12x's
SM120 fused MoE. Per MoE layer the GB300:

1. packs only the tokens that have a cold route (activations, local cold ids, route weights) into a
   shared pinned host buffer and publishes the row count with a sequence number (release, system
   scope);
2. runs its hot experts;
3. waits (acquire, system scope) for the peer to finish that sequence, then adds the peer's rows back
   to their tokens.

Every step is a device kernel with no host sync, so CUDA graphs capture it. A layer with no cold rows
skips the wait; the peer still acknowledges the sequence. The engine opens the sidecar's segment
without O_CREAT, so a missing sidecar is a boot failure, not a silent cold-route drop.
"""

from __future__ import annotations

import mmap
import os

import torch
import triton
import triton.language as tl

HIDDEN = 5120
TOPK = 6
MAX_ROWS = 8192
# Routing: buckets up to this size are copied whole inside the peer's CUDA graphs. The sidecar imports
# this constant, so the two processes cannot disagree.
SMALL_ROWS = 64
HEADER = 64          # int64 words
SLOT = 4096          # byte offset of the header
X_OFF = 8192
XS_OFF = X_OFF + MAX_ROWS * HIDDEN
IDS_OFF = XS_OFF + MAX_ROWS * (HIDDEN // 32)
W_OFF = IDS_OFF + MAX_ROWS * TOPK * 4
OUT_OFF = W_OFF + MAX_ROWS * TOPK * 4
TOTAL_BYTES = (OUT_OFF + MAX_ROWS * HIDDEN * 2 + 4095) // 4096 * 4096
PATH = "/dev/shm/vllm_peer_tier2"
# Global words: [0] published seq, [1] completed seq, [2] timeouts.
# Header: [0] seq, [1] layer, [2] rows, [3] tokens.


def open_shared(create: bool) -> mmap.mmap:
    # Open an existing buffer without O_CREAT: fs.protected_regular refuses
    # O_CREAT on another user's file in sticky /dev/shm, even for root.
    try:
        fd = os.open(PATH, os.O_RDWR)
    except FileNotFoundError:
        if not create:
            raise
        fd = os.open(PATH, os.O_RDWR | os.O_CREAT, 0o666)
    if os.fstat(fd).st_size < TOTAL_BYTES:
        os.ftruncate(fd, TOTAL_BYTES)
    buf = mmap.mmap(fd, TOTAL_BYTES)
    os.close(fd)
    return buf


def register(buf: mmap.mmap, device: torch.device) -> tuple[torch.Tensor, int]:
    """Pin the mapping in this process's CUDA context; return (host view, device ptr)."""
    from cuda.bindings import runtime as rt

    host = torch.frombuffer(buf, dtype=torch.uint8)
    pointer = host.data_ptr()
    flags = rt.cudaHostRegisterMapped | rt.cudaHostRegisterPortable
    with torch.cuda.device(device):
        (error,) = rt.cudaHostRegister(pointer, TOTAL_BYTES, flags)
        if error != rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostRegister failed: {error}")
        error, dptr = rt.cudaHostGetDevicePointer(pointer, 0)
        if error != rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {error}")
    return host, int(dptr)


class _Bytes:
    def __init__(self, pointer: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,), "typestr": "|u1", "data": (pointer, False), "version": 3,
        }


def device_view(dptr: int, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(_Bytes(dptr, TOTAL_BYTES), device=device)


@triton.jit
def _pack(X, XS, IDS, WTS, HAS, POS, PX, PXS, PIDS, PW,
          H: tl.constexpr, HS: tl.constexpr, K: tl.constexpr,
          BLOCK: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr):
    """Copy token t's row to shared row POS[t] if it has a cold route."""
    t = tl.program_id(0).to(tl.int64)
    if tl.load(HAS + t) == 0:
        return
    r = tl.load(POS + t).to(tl.int64)
    for off in tl.static_range(0, H, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < H
        tl.store(PX + r * H + cols, tl.load(X + t * H + cols, mask=mask), mask=mask)
    s = tl.arange(0, BLOCK_S)
    tl.store(PXS + r * HS + s, tl.load(XS + t * HS + s, mask=s < HS), mask=s < HS)
    k = tl.arange(0, BLOCK_K)
    tl.store(PIDS + r * K + k, tl.load(IDS + t * K + k, mask=k < K), mask=k < K)
    tl.store(PW + r * K + k, tl.load(WTS + t * K + k, mask=k < K), mask=k < K)


@triton.jit(do_not_specialize=["LAYER", "T"])
def _publish(WORDS, HEADER_PTR, SEQ, LAYER, ROWS, T, PIDS, K: tl.constexpr, PAD: tl.constexpr):
    """Mask unused rows below PAD, advance the device sequence, write the header, publish with release.

    The peer copies whole buckets of up to PAD rows inside its CUDA graphs, so rows past ROWS must
    carry masked routes; otherwise stale ids make it read experts no live row routes to.
    """
    rows = tl.load(ROWS).to(tl.int64)
    if PAD > 0:
        lane = tl.arange(0, 512)
        row, col = lane // 8, lane % 8
        tl.store(PIDS + row * K + col, tl.full((512,), -1, tl.int32),
                 mask=(row >= rows) & (row < PAD) & (col < K))
        tl.debug_barrier()
    seq = tl.load(SEQ) + 1
    tl.store(SEQ, seq)
    tl.store(HEADER_PTR + 1, LAYER.to(tl.int64))
    tl.store(HEADER_PTR + 2, rows)
    tl.store(HEADER_PTR + 3, T.to(tl.int64))
    tl.store(HEADER_PTR + 0, seq)
    tl.atomic_xchg(WORDS, seq, sem="release", scope="sys")


@triton.jit
def _trap(dep):
    return tl.inline_asm_elementwise("trap; mov.u32 $0, 0;", "=r,r", [dep], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def _wait(WORDS, SEQ, ROWS, OK, TIMEOUT: tl.constexpr):
    """Spin (acquire) until the peer completed this sequence. A lost peer traps; it never drops cold rows."""
    if tl.load(ROWS) == 0:
        tl.store(OK, 0)
        return
    seq = tl.load(SEQ)
    spins = 0
    done = tl.atomic_add(WORDS + 1, 0, sem="acquire", scope="sys")
    while (done < seq) & (spins < TIMEOUT):
        done = tl.atomic_add(WORDS + 1, 0, sem="acquire", scope="sys")
        spins += 1
    if done < seq:
        tl.atomic_add(WORDS + 2, 1, sem="relaxed", scope="sys")
        _trap(seq)
    tl.store(OK, 1)


@triton.jit
def _scatter_add(Y, OUT, HAS, POS, OK, H: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    if (tl.load(OK) == 0) | (tl.load(HAS + t) == 0):
        return
    r = tl.load(POS + t).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    y = tl.load(Y + t * H + cols, mask=mask).to(tl.float32)
    o = tl.load(OUT + r * H + cols, mask=mask).to(tl.float32)
    tl.store(Y + t * H + cols, (y + o).to(Y.dtype.element_ty), mask=mask)


class PeerTier2:
    """GB300 side of the v2 peer tier; one instance serves every layer of the process."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.buf = open_shared(create=False)  # the sidecar must have created it
        self.host, dptr = register(self.buf, device)
        self.host[:X_OFF].zero_()
        shared = device_view(dptr, device)
        self.words = shared[:64].view(torch.int64)
        self.header = shared[SLOT:SLOT + HEADER * 8].view(torch.int64)
        self.px = shared[X_OFF:XS_OFF]
        self.pxs = shared[XS_OFF:IDS_OFF]
        self.pids = shared[IDS_OFF:W_OFF].view(torch.int32)
        self.pw = shared[W_OFF:OUT_OFF].view(torch.float32)
        self.pout = shared[OUT_OFF:OUT_OFF + MAX_ROWS * HIDDEN * 2].view(torch.bfloat16)
        self.seq = torch.zeros(1, dtype=torch.int64, device=device)
        self.ok = torch.zeros(1, dtype=torch.int32, device=device)
        self.max_rows = MAX_ROWS
        self.timeout = 20_000_000

    def send(self, x_quant, x_scale, cold_ids, weights, layer: int):
        """Pack tokens with a cold route and publish; returns state for finish()."""
        tokens, topk = cold_ids.shape
        if tokens > MAX_ROWS or topk != TOPK:
            raise ValueError(f"peer tier v2 handles up to {MAX_ROWS} tokens of top-{TOPK} routes")
        has = (cold_ids >= 0).any(dim=1)
        pos = torch.cumsum(has, 0, dtype=torch.int32) - 1
        rows = has.sum(dtype=torch.int32).reshape(1)
        has = has.to(torch.int8)
        _pack[(tokens,)](
            x_quant.view(torch.uint8), x_scale.view(torch.uint8), cold_ids.contiguous(),
            weights.float().contiguous(), has, pos, self.px, self.pxs, self.pids, self.pw,
            H=HIDDEN, HS=HIDDEN // 32, K=TOPK, BLOCK=1024, BLOCK_S=256, BLOCK_K=8)
        _publish[(1,)](self.words, self.header, self.seq, layer, rows, tokens, self.pids, K=TOPK, PAD=SMALL_ROWS)
        return has, pos, rows

    def finish(self, output, has, pos, rows) -> None:
        _wait[(1,)](self.words, self.seq, rows, self.ok, TIMEOUT=self.timeout)
        _scatter_add[(output.shape[0], triton.cdiv(HIDDEN, 1024))](
            output, self.pout, has, pos, self.ok, H=HIDDEN, BLOCK=1024)
