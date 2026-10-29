"""Quantized full-vocabulary copies of the DSpark draft LM head and their small-M weight-only GEMMs (W5).

The drafter's head is the target's BF16 lm_head (shared tensor, 1.32 GB read for 5 rows per step). These copies keep
all 129,280 rows (no vocabulary truncation) and quantize only the weights; activations stay BF16:

  mxfp8  e4m3 values, one ue8m0 scale per 32 along K        5120 + 160 B/row   = 682.6 MB
  mxfp4  e2m1 values (2 per byte), ue8m0 scale per 32         2560 + 160 B/row   = 351.6 MB
  nvfp4  e2m1 values, e4m3 scale per 16, one fp32 global      2560 + 320 B/row   = 372.3 MB

Scales are chosen by rounding up (2^ceil(log2(amax / max_code)), FlashMLA's output-cast rule) so no value
saturates. The GEMM is one Triton kernel: each program streams BLOCK_N weight rows through K in BLOCK_K steps
(pipelined), dequantizes them to BF16 in registers (exact: an e4m3 / e2m1 code times a power of two, or times an e4m3
scale for nvfp4, fits BF16's mantissa) and multiplies the tile by the BF16 activations on tensor cores with fp32
accumulation (swap-AB: the vocab is the MMA M side, the <= 16 draft rows the N side). The nvfp4 global scale is
applied to the fp32 accumulator. Output BF16, like the production head.

The draft distribution changes by the weight-quantization noise only -- a DRAFT-ONLY change (acceptance only):
the drafter samples from, and caches for the rejection sampler, the logits this head produces.
"""
from __future__ import annotations

import torch

import triton
import triton.language as tl

BLOCK_N = 128
BLOCK_K = 128
STAGES = 3
WARPS = 8
FMT_ID = {"mxfp8": 0, "mxfp4": 1, "nvfp4": 2, "bf16": 3}
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_up_pow2_exp(amax: torch.Tensor, max_code: float) -> torch.Tensor:
    """Smallest e with amax / 2^e <= max_code (FlashMLA's ceil rule); amax clamped to avoid -inf."""
    return torch.ceil(torch.log2(torch.clamp(amax, min=1e-30) / max_code)).clamp(-127, 127)


def _e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round |x| <= 6 to the nearest e2m1 code (ties to the even code), sign in bit 3."""
    mags = torch.tensor(E2M1, device=x.device)
    a = x.abs()
    idx = torch.argmin((a.unsqueeze(-1) - mags).abs() * 4 + (torch.arange(8, device=x.device) % 2) * 1e-6, dim=-1)
    return (idx | ((x < 0).to(idx.dtype) << 3)).to(torch.uint8)


def quantize_rows(w: torch.Tensor, fmt: str):
    """Quantize a [n, K] BF16 slab (one GPU chunk); returns (codes, scales, amax-for-global)."""
    n, k = w.shape
    x = w.float()
    if fmt == "mxfp8":
        blocks = x.view(n, k // 32, 32)
        e = _round_up_pow2_exp(blocks.abs().amax(-1), 448.0)
        q = (blocks / torch.exp2(e).unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8).view(n, k)
        return q, (e + 127).to(torch.uint8)
    if fmt == "mxfp4":
        blocks = x.view(n, k // 32, 32)
        e = _round_up_pow2_exp(blocks.abs().amax(-1), 6.0)
        codes = _e2m1_codes(blocks / torch.exp2(e).unsqueeze(-1)).view(n, k)
        return (codes[:, 0::2] | (codes[:, 1::2] << 4)), (e + 127).to(torch.uint8)
    raise ValueError(fmt)


def quantize_nvfp4_rows(w: torch.Tensor, global_scale: float):
    n, k = w.shape
    blocks = w.float().view(n, k // 16, 16)
    s = (blocks.abs().amax(-1) / 6.0 / global_scale)
    s = torch.clamp(s, max=448.0)  # e4m3fn has no inf: a cast above 448 is NaN
    s8 = s.to(torch.float8_e4m3fn)
    # round the block scale UP to the next e4m3 value so no code saturates (never past 448 = code 0x7E)
    under = (s8.float() < s) & (s8.view(torch.uint8) < 0x7E)
    s8 = torch.where(under, (s8.view(torch.uint8) + 1).view(torch.float8_e4m3fn), s8)
    sf = torch.clamp(s8.float(), min=2.0**-9) * global_scale
    codes = _e2m1_codes(blocks / sf.unsqueeze(-1)).view(n, k)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)), s8.view(torch.uint8)


class QuantHead:
    """A weight-only quantized copy of an [V, K] BF16 head, built in GPU chunks (no full-size temporary).

    `cols` (int32 [V], optional) scatters row r's logit to output column cols[r] (hybrid heads: several heads over
    disjoint row sets of one vocabulary write one [M, vocab] logits tensor)."""

    def __init__(self, weight: torch.Tensor, fmt: str, chunk: int | None = None, cols: torch.Tensor | None = None,
                 device=None, rows: torch.Tensor | None = None):
        """`rows` (int64, optional): build from weight[rows] chunk by chunk (no full-size gathered temporary)."""
        assert fmt in FMT_ID and weight.dim() == 2 and weight.shape[1] % 128 == 0
        v = weight.shape[0] if rows is None else rows.numel()
        k = weight.shape[1]
        self.fmt, self.v, self.k = fmt, v, k
        dev = device or weight.device
        self.cols = cols
        if fmt == "bf16":
            self.codes = (weight if rows is None else weight.index_select(0, rows)).to(dev).contiguous()
            self.scales = self.codes  # unused
            self.global_scale = 1.0
            self.nbytes = self.codes.numel() * 2
            return
        if fmt == "nvfp4":
            lo, hi = torch.aminmax(weight)  # no full-size |w| temporary
            self.global_scale = max(-float(lo), float(hi)) / (448.0 * 6.0)
        else:
            self.global_scale = 1.0
        qk = k if fmt == "mxfp8" else k // 2
        sk = k // 16 if fmt == "nvfp4" else k // 32
        self.codes = torch.empty(v, qk, dtype=torch.uint8, device=dev)
        self.scales = torch.empty(v, sk, dtype=torch.uint8, device=dev)
        # small chunks: the e2m1 rounding builds an [n, K, 8] fp32 temporary
        chunk = chunk or (2048 if fmt == "mxfp8" else 256)
        for r in range(0, v, chunk):
            w = weight[r:r + chunk] if rows is None else weight.index_select(0, rows[r:r + chunk])
            if w.device != dev:
                w = w.to(dev)
            q, s = quantize_nvfp4_rows(w, self.global_scale) if fmt == "nvfp4" else quantize_rows(w, fmt)
            self.codes[r:r + q.shape[0]] = q
            self.scales[r:r + s.shape[0]] = s
        self.nbytes = self.codes.numel() + self.scales.numel()

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None, n_virtual: int | None = None):
        """x [M, K] BF16 -> [M, V] BF16 (n_virtual > V: time a virtual head that re-reads rows modulo V)."""
        m, k = x.shape
        assert k == self.k and x.dtype == torch.bfloat16 and m <= 128
        n = n_virtual or self.v
        if out is None:
            out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
        m_pad = max(16, triton.next_power_of_2(m))
        cols = self.cols if self.cols is not None else self.scales  # dummy pointer when unused
        _wq_gemm_kernel[(triton.cdiv(n, BLOCK_N),)](
            x, x.stride(0), self.codes, self.codes.stride(0), self.scales, self.scales.stride(0), out, out.stride(0),
            cols, m, n, self.v, self.global_scale,
            K=k, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, M_PAD=m_pad, FMT=FMT_ID[self.fmt], STAGES=STAGES,
            HAS_COLS=self.cols is not None, num_warps=WARPS,
        )
        return out


@triton.jit
def _e2m1_to_f32(nib):
    """e2m1 code (4 bits, sign in bit 3) -> fp32."""
    mag = nib & 7
    e = mag >> 1
    m = mag & 1
    v = tl.where(e == 0, m.to(tl.float32) * 0.5, (1.0 + m.to(tl.float32) * 0.5) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where((nib & 8) != 0, -v, v)


@triton.jit
def _wq_gemm_kernel(x_ptr, x_stride, q_ptr, q_stride, s_ptr, s_stride, out_ptr, out_stride, cols_ptr, M, N, N_PHYS,
                    global_scale, K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, M_PAD: tl.constexpr,
                    FMT: tl.constexpr, STAGES: tl.constexpr, HAS_COLS: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < N
    prow = (rows % N_PHYS).to(tl.int64)
    ms = tl.arange(0, M_PAD)
    m_ok = ms < M
    acc = tl.zeros((BLOCK_N, M_PAD), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K, num_stages=STAGES):
        kk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + ms[None, :] * x_stride + kk[:, None], mask=m_ok[None, :], other=0.0)  # [BLOCK_K, M_PAD]
        if FMT == 3:  # bf16 rows (exact)
            w = tl.load(q_ptr + prow[:, None] * q_stride + kk[None, :]).to(tl.float32)
        elif FMT == 0:  # mxfp8
            c = tl.load(q_ptr + prow[:, None] * q_stride + kk[None, :])
            sc = tl.load(s_ptr + prow[:, None] * s_stride + (k0 // 32 + tl.arange(0, BLOCK_K // 32))[None, :])
            w = c.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            w = tl.reshape(w, (BLOCK_N, BLOCK_K // 32, 32)) * tl.exp2(sc.to(tl.float32) - 127.0)[:, :, None]
        else:
            c = tl.load(q_ptr + prow[:, None] * q_stride + (k0 // 2 + tl.arange(0, BLOCK_K // 2))[None, :])
            lo = _e2m1_to_f32(c & 15)
            hi = _e2m1_to_f32(c >> 4)
            w = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K))
            if FMT == 1:  # mxfp4: ue8m0 per 32
                sc = tl.load(s_ptr + prow[:, None] * s_stride + (k0 // 32 + tl.arange(0, BLOCK_K // 32))[None, :])
                w = tl.reshape(w, (BLOCK_N, BLOCK_K // 32, 32)) * tl.exp2(sc.to(tl.float32) - 127.0)[:, :, None]
            else:  # nvfp4: e4m3 per 16 (global scale on the accumulator)
                sc = tl.load(s_ptr + prow[:, None] * s_stride + (k0 // 16 + tl.arange(0, BLOCK_K // 16))[None, :])
                w = tl.reshape(w, (BLOCK_N, BLOCK_K // 16, 16)) * sc.to(tl.float8e4nv, bitcast=True).to(tl.float32)[:, :, None]
        w = tl.reshape(w, (BLOCK_N, BLOCK_K)).to(tl.bfloat16)
        acc = tl.dot(w, x, acc)
    if FMT == 2:
        acc = acc * global_scale
    if HAS_COLS:
        ocol = tl.load(cols_ptr + prow, mask=row_ok, other=0).to(tl.int64)
    else:
        ocol = rows.to(tl.int64)
    tl.store(out_ptr + ms[None, :] * out_stride + ocol[:, None], acc.to(tl.bfloat16),
             mask=row_ok[:, None] & m_ok[None, :])


# ---- W-quantized x A8 path: hardware block-scaled MMA (tcgen05 f8f6f4), no dequantization in registers ----------
@triton.jit
def _act_mxfp8_kernel(x_ptr, x_stride, q_ptr, s_ptr, M, K: tl.constexpr, BLOCK_K: tl.constexpr):
    """x [M, K] bf16 -> e4m3 [M_PAD, K] + ue8m0 [M_PAD, K/32] (round-up scales); rows >= M zero."""
    m = tl.program_id(0)
    kb = tl.program_id(1)
    j = tl.arange(0, 32)
    for i in tl.static_range(BLOCK_K // 32):
        k0 = kb * BLOCK_K + i * 32
        v = tl.load(x_ptr + m * x_stride + k0 + j, mask=m < M, other=0.0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(v), axis=0), 1e-30)
        e = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / 448.0)), -127.0), 127.0)
        q = (v * tl.exp2(-e)).to(tl.float8e4nv)
        tl.store(q_ptr + m * K + k0 + j, q.to(tl.uint8, bitcast=True))
        tl.store(s_ptr + m * (K // 32) + k0 // 32, (e + 127.0).to(tl.uint8))


@triton.jit
def _wq_scaled_kernel(xq_ptr, xs_ptr, q_ptr, q_stride, s_ptr, s_stride, out_ptr, out_stride, cols_ptr, M, N, N_PHYS,
                      K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, M_PAD: tl.constexpr,
                      W_FP4: tl.constexpr, STAGES: tl.constexpr, HAS_COLS: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < N
    prow = (rows % N_PHYS).to(tl.int64)
    ms = tl.arange(0, M_PAD)
    acc = tl.zeros((BLOCK_N, M_PAD), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K, num_stages=STAGES):
        kb = k0 // 32 + tl.arange(0, BLOCK_K // 32)
        if W_FP4:
            w = tl.load(q_ptr + prow[:, None] * q_stride + (k0 // 2 + tl.arange(0, BLOCK_K // 2))[None, :])
        else:
            w = tl.load(q_ptr + prow[:, None] * q_stride + (k0 + tl.arange(0, BLOCK_K))[None, :])
        ws = tl.load(s_ptr + prow[:, None] * s_stride + kb[None, :])
        x = tl.load(xq_ptr + ms[:, None] * K + (k0 + tl.arange(0, BLOCK_K))[None, :])  # [M_PAD, BK] e4m3 bits
        xs = tl.load(xs_ptr + ms[:, None] * (K // 32) + kb[None, :])
        if W_FP4:
            acc = tl.dot_scaled(w, ws, "e2m1", tl.trans(x), xs, "e4m3", acc)
        else:
            acc = tl.dot_scaled(w, ws, "e4m3", tl.trans(x), xs, "e4m3", acc)
    if HAS_COLS:
        ocol = tl.load(cols_ptr + prow, mask=row_ok, other=0).to(tl.int64)
    else:
        ocol = rows.to(tl.int64)
    tl.store(out_ptr + ms[None, :] * out_stride + ocol[:, None], acc.to(tl.bfloat16),
             mask=row_ok[:, None] & (ms < M)[None, :])


class QuantHeadA8(QuantHead):
    """mxfp8 / mxfp4 weights x MXFP8 activations on the block-scaled MMA (activation quantized per call)."""

    def __init__(self, weight, fmt, **kw):
        assert fmt in ("mxfp8", "mxfp4")
        super().__init__(weight, fmt, **kw)
        self._xq = None

    def __call__(self, x, out=None, n_virtual=None):
        m, k = x.shape
        m_pad = max(16, triton.next_power_of_2(m))
        if self._xq is None or self._xq.shape[0] < m_pad:
            self._xq = torch.zeros(m_pad, k, dtype=torch.uint8, device=x.device)
            self._xs = torch.full((m_pad, k // 32), 127, dtype=torch.uint8, device=x.device)
        _act_mxfp8_kernel[(m_pad, k // 512)](x, x.stride(0), self._xq, self._xs, m, K=k, BLOCK_K=512)
        n = n_virtual or self.v
        if out is None:
            out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
        cols = self.cols if self.cols is not None else self.scales
        _wq_scaled_kernel[(triton.cdiv(n, BLOCK_N),)](
            self._xq, self._xs, self.codes, self.codes.stride(0), self.scales, self.scales.stride(0), out,
            out.stride(0), cols, m, n, self.v,
            K=k, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, M_PAD=m_pad, W_FP4=self.fmt == "mxfp4", STAGES=STAGES,
            HAS_COLS=self.cols is not None, num_warps=WARPS,
        )
        return out


# ---- TMA-fed variant: weight (and scale) tiles through host tensor descriptors (no per-element address tensors) ----
@triton.jit
def _wq_scaled_tma_kernel(xq_desc, xs_ptr, w_desc, s_desc, s_ptr, s_stride, out_ptr, out_stride, cols_ptr, M, N,
                          N_PHYS, K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, M_PAD: tl.constexpr,
                          W_FP4: tl.constexpr, STAGES: tl.constexpr, HAS_COLS: tl.constexpr, SCALE_TMA: tl.constexpr,
                          WS: tl.constexpr):
    """Scales with SCALE_TMA: a [K/BK, V*BK/32] re-layout, one contiguous [1, BN*BK/32] box per (row tile, k step)."""
    pid = tl.program_id(0)
    row0 = (pid * BLOCK_N) % N_PHYS
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    prow = (row0 + tl.arange(0, BLOCK_N)).to(tl.int64)
    ms = tl.arange(0, M_PAD)
    acc = tl.zeros((BLOCK_N, M_PAD), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K, num_stages=STAGES, warp_specialize=WS):
        if W_FP4:
            w = w_desc.load([row0, k0 // 2])
        else:
            w = w_desc.load([row0, k0])
        if SCALE_TMA:
            ws = tl.reshape(s_desc.load([k0 // BLOCK_K, row0 * (BLOCK_K // 32)]), (BLOCK_N, BLOCK_K // 32))
        else:
            kb = k0 // 32 + tl.arange(0, BLOCK_K // 32)
            ws = tl.load(s_ptr + prow[:, None] * s_stride + kb[None, :])
        x = xq_desc.load([0, k0])
        xs = tl.load(xs_ptr + ms[:, None] * (K // 32) + (k0 // 32 + tl.arange(0, BLOCK_K // 32))[None, :])
        if W_FP4:
            acc = tl.dot_scaled(w, ws, "e2m1", tl.trans(x), xs, "e4m3", acc)
        else:
            acc = tl.dot_scaled(w, ws, "e4m3", tl.trans(x), xs, "e4m3", acc)
    if HAS_COLS:
        ocol = tl.load(cols_ptr + prow, mask=rows < N, other=0).to(tl.int64)
    else:
        ocol = rows.to(tl.int64)
    tl.store(out_ptr + ms[None, :] * out_stride + ocol[:, None], acc.to(tl.bfloat16),
             mask=(rows < N)[:, None] & (ms < M)[None, :])


SCALE_TMA = True
WS = False


class QuantHeadTMA(QuantHeadA8):
    """QuantHeadA8 with the weight stream on TMA (host descriptors: fixed addresses, CUDA-graph safe)."""

    def _scales_by_kstep(self, bk: int) -> torch.Tensor:
        """[V, K/32] -> [K/BK, V*BK/32]: each (row tile, k step) scale block is one contiguous box."""
        cache = self.__dict__.setdefault("_s_kstep", {})
        if bk not in cache:
            v, sk = self.scales.shape
            cache[bk] = self.scales.view(v, sk // (bk // 32), bk // 32).permute(1, 0, 2).contiguous().view(
                sk // (bk // 32), v * (bk // 32))
        return cache[bk]

    def __call__(self, x, out=None, n_virtual=None):
        from triton.tools.tensor_descriptor import TensorDescriptor

        m, k = x.shape
        m_pad = max(16, triton.next_power_of_2(m))
        if self._xq is None or self._xq.shape[0] != m_pad:
            self._xq = torch.zeros(m_pad, k, dtype=torch.uint8, device=x.device)
            self._xs = torch.full((m_pad, k // 32), 127, dtype=torch.uint8, device=x.device)
            self._desc = None
        key = (BLOCK_N, BLOCK_K, m_pad, SCALE_TMA)
        if getattr(self, "_desc", None) is None or self._desc[0] != key:
            bkb = BLOCK_K // 2 if self.fmt == "mxfp4" else BLOCK_K
            s_desc = (TensorDescriptor.from_tensor(self._scales_by_kstep(BLOCK_K), [1, BLOCK_N * BLOCK_K // 32])
                      if SCALE_TMA else self.scales)
            self._desc = (key, TensorDescriptor.from_tensor(self.codes, [BLOCK_N, bkb]),
                          TensorDescriptor.from_tensor(self._xq, [m_pad, BLOCK_K]), s_desc)
        _, w_desc, x_desc, s_desc = self._desc
        _act_mxfp8_kernel[(m_pad, k // 512)](x, x.stride(0), self._xq, self._xs, m, K=k, BLOCK_K=512)
        n = n_virtual or self.v
        if out is None:
            out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
        cols = self.cols if self.cols is not None else self.scales
        _wq_scaled_tma_kernel[(triton.cdiv(n, BLOCK_N),)](
            x_desc, self._xs, w_desc, s_desc, self.scales, self.scales.stride(0), out, out.stride(0), cols, m, n,
            self.v, K=k, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, M_PAD=m_pad, W_FP4=self.fmt == "mxfp4", STAGES=STAGES,
            HAS_COLS=self.cols is not None, SCALE_TMA=SCALE_TMA, WS=WS, num_warps=WARPS,
        )
        return out


# ---- Tiled layout: each CTA's whole weight stream is one contiguous range (sequential DRAM, full-burst TMA) ---------
@triton.jit
def _wq_tiled_kernel(xq_desc, xs_ptr, w_desc, s_desc, out_ptr, out_stride, cols_ptr, M, N,
                     K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, M_PAD: tl.constexpr,
                     W_FP4: tl.constexpr, STAGES: tl.constexpr, HAS_COLS: tl.constexpr):
    """codes [NT*KT*BN, BKB]: tile (n, k) = rows [(n*KT + k)*BN, +BN); scales [NT*KT, BN*BK/32] likewise."""
    pid = tl.program_id(0)
    KT: tl.constexpr = K // BLOCK_K
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    ms = tl.arange(0, M_PAD)
    acc = tl.zeros((BLOCK_N, M_PAD), dtype=tl.float32)
    for ks in tl.range(0, KT, num_stages=STAGES):
        w = w_desc.load([(pid * KT + ks) * BLOCK_N, 0])
        ws = tl.reshape(s_desc.load([pid * KT + ks, 0]), (BLOCK_N, BLOCK_K // 32))
        x = xq_desc.load([0, ks * BLOCK_K])
        xs = tl.load(xs_ptr + ms[:, None] * (K // 32) + (ks * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32))[None, :])
        if W_FP4:
            acc = tl.dot_scaled(w, ws, "e2m1", tl.trans(x), xs, "e4m3", acc)
        else:
            acc = tl.dot_scaled(w, ws, "e4m3", tl.trans(x), xs, "e4m3", acc)
    row_ok = rows < N
    if HAS_COLS:
        ocol = tl.load(cols_ptr + rows, mask=row_ok, other=0).to(tl.int64)
    else:
        ocol = rows.to(tl.int64)
    tl.store(out_ptr + ms[None, :] * out_stride + ocol[:, None], acc.to(tl.bfloat16),
             mask=row_ok[:, None] & (ms < M)[None, :])


def tile_codes_scales(codes: torch.Tensor, scales: torch.Tensor, bn: int, bk: int, fp4: bool):
    """Row-major codes [V, Kq] / ue8m0 scales [V, K/32] -> the tiled layout (rows padded to a multiple of bn)."""
    v, kq = codes.shape
    k = kq * 2 if fp4 else kq
    bkb = bk // 2 if fp4 else bk
    nt, kt = triton.cdiv(v, bn), k // bk
    pad = nt * bn - v
    if pad:
        codes = torch.cat([codes, codes.new_zeros(pad, kq)])
        scales = torch.cat([scales, scales.new_full((pad, scales.shape[1]), 127)])
    tc = codes.view(nt, bn, kt, bkb).permute(0, 2, 1, 3).contiguous().view(nt * kt * bn, bkb)
    ts = scales.view(nt, bn, kt, bk // 32).permute(0, 2, 1, 3).contiguous().view(nt * kt, bn * bk // 32)
    return tc, ts


class QuantHeadTiled(QuantHeadA8):
    """mxfp4 / mxfp8 x MXFP8 activations on the block-scaled MMA, weights in the CTA-contiguous tiled layout.

    Same MMA and accumulation order as QuantHeadTMA / DeepGEMM's sm100_fp8_fp4_gemm_1d1d (outputs bitwise equal in
    bench/qhead_dg.py); only where the bytes live differs. `keep_rowmajor=False` frees the row-major copy."""

    def __init__(self, weight, fmt, bn: int = 128, bk: int = 128, keep_rowmajor: bool = False, **kw):
        super().__init__(weight, fmt, **kw)
        self.bn, self.bk = bn, bk
        self.tcodes, self.tscales = tile_codes_scales(self.codes, self.scales, bn, bk, fmt == "mxfp4")
        if not keep_rowmajor:
            self.codes = self.scales = None
        self.nbytes = self.v * (self.k // 2 if fmt == "mxfp4" else self.k) + self.v * self.k // 32
        self._desc = None

    def __call__(self, x, out=None, n_virtual=None):
        from triton.tools.tensor_descriptor import TensorDescriptor

        assert n_virtual is None
        m, k = x.shape
        m_pad = max(16, triton.next_power_of_2(m))
        if self._xq is None or self._xq.shape[0] != m_pad:
            self._xq = torch.zeros(m_pad, k, dtype=torch.uint8, device=x.device)
            self._xs = torch.full((m_pad, k // 32), 127, dtype=torch.uint8, device=x.device)
            self._desc = None
        if self._desc is None:
            bkb = self.bk // 2 if self.fmt == "mxfp4" else self.bk
            self._desc = (TensorDescriptor.from_tensor(self.tcodes, [self.bn, bkb]),
                          TensorDescriptor.from_tensor(self.tscales, [1, self.bn * self.bk // 32]),
                          TensorDescriptor.from_tensor(self._xq, [m_pad, self.bk]))
        w_desc, s_desc, x_desc = self._desc
        _act_mxfp8_kernel[(m_pad, k // 512)](x, x.stride(0), self._xq, self._xs, m, K=k, BLOCK_K=512)
        if out is None:
            out = torch.empty(m, self.v, dtype=torch.bfloat16, device=x.device)
        cols = self.cols if self.cols is not None else self.tscales
        _wq_tiled_kernel[(triton.cdiv(self.v, self.bn),)](
            x_desc, self._xs, w_desc, s_desc, out, out.stride(0), cols, m, self.v,
            K=k, BLOCK_N=self.bn, BLOCK_K=self.bk, M_PAD=m_pad, W_FP4=self.fmt == "mxfp4", STAGES=STAGES,
            HAS_COLS=self.cols is not None, num_warps=WARPS,
        )
        return out
