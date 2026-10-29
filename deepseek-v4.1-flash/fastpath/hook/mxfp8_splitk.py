"""Split-K MXFP8 dense GEMM for decode-sized M (1..64) on SM100/SM103, in Triton.

out[M, N] (bf16) = X[M, K] (e4m3, e8m0 per 32) @ W[N, K]^T (e4m3, e8m0 per 32)

Operands stay exactly as vLLM's FlashInferCutedslMxfp8LinearKernel stores them: W is [N, K] row-major (the layer
keeps its transpose view [K, N]), and both scale tensors are in the 128x4-swizzled ue8m0 layout (rows padded to
128, K/32 padded to 4). The scales are read in place through the swizzle formula, so no weight is copied.

Swap-AB: each program multiplies a BLOCK_N=128 slab of weight rows by all M tokens (padded to BLOCK_M >= 16) with
tl.dot_scaled, which lowers to the tcgen05 block-scaled MMA. K is split SPLIT_K ways so N/128 * SPLIT_K programs
cover the 152 SMs; each writes an fp32 partial, and the last program to arrive for an N slab (device ticket, reset
by that program) sums the partials in split order and writes bf16, so results are deterministic and the kernel is
CUDA-graph safe.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sf_offset(row, kb, KB_PAD: tl.constexpr):
    # 128x4 swizzle: [row // 128][kb // 4] tiles of 512 B, inside: (row % 32) * 16 + ((row % 128) // 32) * 4 + kb % 4
    return ((row // 128) * (KB_PAD // 4) + kb // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + kb % 4


@triton.jit
def _mxfp8_splitk(X, XS, W, WS, OUT, PART, TICKET, M, N,
                  K: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    KB_PAD: tl.constexpr = (K // 32 + 3) // 4 * 4
    K_PER: tl.constexpr = K // SPLIT_K
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BLOCK_M)
    nmask = rn < N
    mmask = rm < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    k0 = pid_k * K_PER
    for kk in range(0, K_PER, BLOCK_K):
        rk = k0 + kk + tl.arange(0, BLOCK_K)
        kb = (k0 + kk) // 32 + tl.arange(0, BLOCK_K // 32)
        w = tl.load(W + rn[:, None].to(tl.int64) * K + rk[None, :], mask=nmask[:, None])
        ws = tl.load(WS + _sf_offset(rn[:, None], kb[None, :], KB_PAD), mask=nmask[:, None], other=127)
        x = tl.load(X + rm[:, None] * K + rk[None, :], mask=mmask[:, None])  # padded token rows only feed masked outputs
        xs = tl.load(XS + _sf_offset(rm[:, None], kb[None, :], KB_PAD), mask=mmask[:, None], other=127)
        acc = tl.dot_scaled(w, ws, "e4m3", tl.trans(x), xs, "e4m3", acc)
    if SPLIT_K == 1:
        tl.store(OUT + rm[None, :] * N + rn[:, None], acc.to(tl.bfloat16), mask=nmask[:, None] & mmask[None, :])
    else:
        slab = (pid_k * tl.num_programs(0) + pid_n).to(tl.int64) * (BLOCK_N * BLOCK_M)
        offs = tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + rm[None, :]
        tl.store(PART + slab + offs, acc)
        arrived = tl.atomic_add(TICKET + pid_n, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            total = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                sl = (s * tl.num_programs(0) + pid_n).to(tl.int64) * (BLOCK_N * BLOCK_M)
                total += tl.load(PART + sl + offs, cache_modifier=".cg")
            tl.store(OUT + rm[None, :] * N + rn[:, None], total.to(tl.bfloat16),
                     mask=nmask[:, None] & mmask[None, :])
            tl.store(TICKET + pid_n, 0)


_ws: dict = {}


def _workspace(dev, n_slabs: int, split_k: int, block_m: int):
    key = (dev, n_slabs, split_k, block_m)
    buf = _ws.get(key)
    if buf is None:
        part = torch.empty(split_k * n_slabs * 128 * block_m, dtype=torch.float32, device=dev)
        ticket = torch.zeros(n_slabs, dtype=torch.int32, device=dev)
        buf = _ws[key] = (part, ticket)
    return buf


def pick_split(n: int, k: int, sms: int = 152, block_k: int = 256) -> int:
    """Smallest power-of-two split that puts at least ~one program per SM, keeping >= 2 K blocks per program."""
    slabs = triton.cdiv(n, 128)
    split = 1
    while slabs * split < sms and k // (split * 2) >= 2 * block_k and split < 16:
        split *= 2
    return split


def mxfp8_gemm(x, x_sf, w_nk, w_sf, n: int, k: int, out=None, split_k: int | None = None,
               block_k: int = 256, num_warps: int = 4, num_stages: int = 4):
    """x: [M, K] e4m3; x_sf / w_sf: 128x4-swizzled ue8m0 (uint8 or e8m0 view); w_nk: [N, K] e4m3 row-major."""
    m = x.shape[0]
    if m > 64 or k % block_k:
        raise ValueError(f"mxfp8_gemm handles M <= 64 and K % {block_k} == 0 (got M={m}, K={k})")
    block_m = max(16, triton.next_power_of_2(m))
    split = split_k or pick_split(n, k, block_k=block_k)
    while (k // split) % block_k:
        split //= 2
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    slabs = triton.cdiv(n, 128)
    part, ticket = _workspace(x.device, slabs, split, block_m)
    _mxfp8_splitk[(slabs, split)](
        x, x_sf.view(torch.uint8), w_nk, w_sf.view(torch.uint8), out, part, ticket, m, n,
        K=k, SPLIT_K=split, BLOCK_N=128, BLOCK_M=block_m, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages)
    return out
