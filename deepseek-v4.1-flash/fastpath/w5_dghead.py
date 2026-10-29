"""Quantized full-vocabulary DSpark draft head on DeepGEMM's sm100 block-scaled GEMM (W5).

Weights: a load-time copy of the BF16 lm_head in MXFP8 (e4m3, ue8m0 per 32: 682.6 MB) or MXFP4 (e2m1, ue8m0 per 32:
351.6 MB), every vocabulary row kept. Activations: MXFP8 per 32 (the block-scaled MMA needs FP8 on both sides). With
`two_term`, each activation row is split x ~= hi + lo (hi = mxfp8(x), lo = mxfp8(x - hi)) and the GEMM runs 2M rows
against the same weight bytes (M <= 8 stays inside one 16-row MMA tile, so the weight stream is unchanged); the two fp32
partial logits are summed and rounded to BF16 once. On real drafter rows that is indistinguishable from BF16
activations (bench/qhead_drift.py: mxfp8 A8x2 KL 2.15e-3 = mxfp8 A16; A8 alone 3.31e-3).

GEMM = vllm.third_party.deep_gemm.fp8_fp4_gemm_nt (sm100_fp8_fp4_gemm_1d1d, the kernel production already runs for
its MXFP8 dense layers), recipe (1, 32) on both operands, fp32 accumulation. Its output is bitwise equal to the
Triton block-scaled kernel in w5_qhead (bench/qhead_dg.py), at 125 us (mxfp8) / 79 us (mxfp4) vs 222 us for the
cuBLAS BF16 head (full shape, L2 cold).

Scale layout DeepGEMM wants for recipe (1, 32) on sm100 (probed with transform_sf_into_required_layout,
bench/dg_probe.py): int32 [mn, K/128], column-major with the mn stride rounded up to a multiple of 4, byte j of word
(r, c) = the ue8m0 exponent of row r's block 4c + j. Both operands are written in that layout directly (weights once
at load, activations by the quantization kernel), so no per-call transform kernel runs.
"""
from __future__ import annotations

import torch

import triton
import triton.language as tl

TWO_TERM_MAX_M = 64  # beyond this the fp32 partial-logit buffer grows large; one-term (A8) from there


def _align4(n: int) -> int:
    return (n + 3) // 4 * 4


def pack_ue8m0_colmajor(e: torch.Tensor) -> torch.Tensor:
    """ue8m0 exponents uint8 [mn, K/32] -> DeepGEMM's packed int32 [mn, K/128] column-major (mn stride aligned to 4)."""
    mn, kb = e.shape
    words = e.contiguous().view(torch.int32)  # [mn, K/128]: byte j of word c = block 4c + j (little endian)
    buf = torch.zeros(kb // 4, _align4(mn), dtype=torch.int32, device=e.device)
    buf[:, :mn] = words.t()
    return buf.t()[:mn]


@triton.jit
def _act_mxfp8_dg_kernel(x_ptr, x_stride, q_ptr, s_ptr, s_stride, M, K: tl.constexpr, TWO_TERM: tl.constexpr):
    """Row m of x (bf16) -> e4m3 codes q[m] (+ q[M + m] = the residual's codes if TWO_TERM) and packed ue8m0 scales
    s[word c, row] (int32, 4 consecutive 32-blocks per word), round-up scale rule as w5_qhead._act_mxfp8_kernel."""
    m = tl.program_id(0)
    c = tl.program_id(1)  # one int32 scale word = 128 elements
    k = c * 128 + tl.arange(0, 128)
    v = tl.load(x_ptr + m * x_stride + k).to(tl.float32)
    b = tl.reshape(v, (4, 32))
    amax = tl.maximum(tl.max(tl.abs(b), axis=1), 1e-30)
    e = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / 448.0)), -127.0), 127.0)
    hi = (b * tl.exp2(-e)[:, None]).to(tl.float8e4nv)
    tl.store(q_ptr + m * K + k, tl.reshape(hi, (128,)).to(tl.uint8, bitcast=True))
    sh = tl.arange(0, 4) * 8
    word = tl.sum((e + 127.0).to(tl.int32) << sh, axis=0)
    tl.store(s_ptr + c * s_stride + m, word)
    if TWO_TERM:
        r = b - hi.to(tl.float32) * tl.exp2(e)[:, None]
        ra = tl.maximum(tl.max(tl.abs(r), axis=1), 1e-30)
        e2 = tl.minimum(tl.maximum(tl.ceil(tl.log2(ra / 448.0)), -127.0), 127.0)
        lo = (r * tl.exp2(-e2)[:, None]).to(tl.float8e4nv)
        tl.store(q_ptr + (M + m) * K + k, tl.reshape(lo, (128,)).to(tl.uint8, bitcast=True))
        word2 = tl.sum((e2 + 127.0).to(tl.int32) << sh, axis=0)
        tl.store(s_ptr + c * s_stride + M + m, word2)


@triton.jit
def _sum_two_term_kernel(d_ptr, out_ptr, out_stride, M, N, BLOCK: tl.constexpr):
    """out[m, :] = bf16(d[m, :] + d[M + m, :]) (fp32 partial logits of hi and lo activations)."""
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < N
    a = tl.load(d_ptr + m * N + offs, mask=ok)
    b = tl.load(d_ptr + (M + m) * N + offs, mask=ok)
    tl.store(out_ptr + m * out_stride + offs, (a + b).to(tl.bfloat16), mask=ok)


class DGHead:
    """[M, 5120] BF16 -> [M, V] BF16 draft logits through a quantized copy of the head (see module docstring)."""

    def __init__(self, weight: torch.Tensor, fmt: str, two_term: bool = True, chunk: int = 2048):
        import vllm.third_party.deep_gemm as dg

        from w5_qhead import quantize_rows

        self._gemm = dg.fp8_fp4_gemm_nt
        assert fmt in ("mxfp8", "mxfp4") and weight.dim() == 2 and weight.shape[1] % 128 == 0
        v, k = weight.shape
        dev = weight.device
        self.fmt, self.v, self.k, self.two_term = fmt, v, k, two_term
        qk = k if fmt == "mxfp8" else k // 2
        codes = torch.empty(v, qk, dtype=torch.uint8, device=dev)
        exps = torch.empty(v, k // 32, dtype=torch.uint8, device=dev)
        step = chunk if fmt == "mxfp8" else 256  # e2m1 rounding builds an [n, K, 8] temporary
        for r in range(0, v, step):
            q, e = quantize_rows(weight[r:r + step], fmt)
            codes[r:r + q.shape[0]] = q
            exps[r:r + e.shape[0]] = e
        self.codes = codes.view(torch.float8_e4m3fn) if fmt == "mxfp8" else codes.view(torch.int8)
        self.sfb = pack_ue8m0_colmajor(exps)
        del exps
        self.nbytes = codes.numel() + self.sfb.numel() * 4

    def gemm_rows(self, m: int) -> int:
        return 2 * m if (self.two_term and m <= TWO_TERM_MAX_M) else m

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        assert k == self.k and x.dtype == torch.bfloat16
        x = x.contiguous()
        m2 = self.gemm_rows(m)
        two = m2 != m
        xq = torch.empty(m2, k, dtype=torch.float8_e4m3fn, device=x.device)
        sbuf = torch.empty(k // 128, _align4(m2), dtype=torch.int32, device=x.device)
        _act_mxfp8_dg_kernel[(m, k // 128)](x, x.stride(0), xq.view(torch.uint8), sbuf, sbuf.stride(0), m, K=k,
                                            TWO_TERM=two, num_warps=1)
        sfa = sbuf.t()[:m2]
        out = torch.empty(m, self.v, dtype=torch.bfloat16, device=x.device)
        if not two:
            self._gemm((xq, sfa), (self.codes, self.sfb), out, recipe_a=(1, 32), recipe_b=(1, 32))
            return out
        d = torch.empty(m2, self.v, dtype=torch.float32, device=x.device)
        self._gemm((xq, sfa), (self.codes, self.sfb), d, recipe_a=(1, 32), recipe_b=(1, 32))
        _sum_two_term_kernel[(m, triton.cdiv(self.v, 4096))](d, out, out.stride(0), m, self.v, BLOCK=4096,
                                                              num_warps=4)
        return out

    def warmup(self, max_m: int) -> None:
        """Compile (DeepGEMM JIT) every GEMM shape the drafter can hit, outside any CUDA-graph capture."""
        x = torch.zeros(max_m, self.k, dtype=torch.bfloat16, device=self.codes.device)
        for m in range(1, max_m + 1):
            self(x[:m])
        torch.cuda.synchronize()
