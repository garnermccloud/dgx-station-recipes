// W4 mHC boundary math, bit-exact against the production kernels (vLLM af7f9488, TileLang 0.1.12, DeepGEMM,
// FlashInfer 0.7.0 cute-dsl MXFP8, Triton prepare_megamoe_inputs). Header-only, device functions only.
//
// Exactness notes (each verified bit for bit by tests/probe_exact.py and tests/test_boundary.py):
//  * mhc_post (TileLang): v = fma(post_i, x, comb[0][i]*res0); v = fma(comb[j][i], res_j, v) j=1..3; bf16 RN.
//    (nvcc contracts `c*d + a0*b0` into fma(c, d, a0*b0); the other association differs in ~1e-5 of elements.)
//  * collapse (TileLang "input" split): o = fma(pre0, r0, +0); o = fma(pre_j, r_j, o); bf16 RN.
//  * RMSNorm sum of squares: TileLang's 64 collapse threads; virtual thread v owns positions
//    p = hh*512 + v*8 + e (hh in {0,1}, e in 0..7) of every 1024-wide block b; per position an fma chain over
//    b = 0..4; per thread s = sum over rv=0..15 of sp[(rv&1)*8 + (rv>>1)] (sequential fadd); then a butterfly over
//    the 64 threads (xor 32 via smem, then 16, 8, 4, 2, 1). rs = rsqrtf(s / 5120 + 1e-20) (IEEE div).
//    out = bf16( (float(c) * rs) * w ).
//  * mix statistics (TileLang "stats" split, see stats_token): bit-exact given DeepGEMM's split partials.
//  * DeepGEMM tf32_hc_prenorm_gemm: its B operand (hc_fn) is used rounded to tf32 with round-to-nearest-even;
//    a tensor-core tf32 chain (tcgen05 kind::tf32 M=64 K=8, or mma.sync m16n8k8) over the same 16 splits x 160
//    K=8 steps, on hc_fn pre-rounded RNE (tf32_rne below), reproduces it bit for bit. Its sum of squares:
//    sqrsum_split below.
#pragma once

#include <cstdint>
#include <cuda_bf16.h>

namespace w4 {

constexpr int kHidden = 5120;
constexpr int kHc = 4;
constexpr int kMix = 24;
constexpr int kVt = 64;           // TileLang collapse threads (virtual threads of the norm reduction)
constexpr float kNormEps = 1e-20f;  // rms_norm_eps (also the hc-mix rms eps)
constexpr float kHcEps = 1e-6f;
constexpr int kSinkhornIters = 20;

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f2bf(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float bfbits2f(uint32_t lo16) { return __uint_as_float(lo16 << 16); }

// Unpack 8 bf16 (one uint4) to fp32.
__device__ __forceinline__ void unpack8(const uint4& u, float (&f)[8]) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);  // .x = a (low), .y = b (high)
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  return make_uint4(pack2(f[0], f[1]), pack2(f[2], f[3]), pack2(f[4], f[5]), pack2(f[6], f[7]));
}

// ------------------------------------------------------------------------------------------------ mhc_post
// comb_col[j] = comb[t][j][i] (residual j feeding stream i).
__device__ __forceinline__ float post_elem(float post_i, float x, const float (&comb_col)[4], const float (&r)[4]) {
  float v = __fmaf_rn(post_i, x, __fmul_rn(comb_col[0], r[0]));
  v = __fmaf_rn(comb_col[1], r[1], v);
  v = __fmaf_rn(comb_col[2], r[2], v);
  v = __fmaf_rn(comb_col[3], r[3], v);
  return v;
}

// ------------------------------------------------------------------------------------------------ collapse
__device__ __forceinline__ float collapse_elem(const float (&pre)[4], const float (&rn)[4]) {
  float o = __fmaf_rn(pre[0], rn[0], 0.f);
  o = __fmaf_rn(pre[1], rn[1], o);
  o = __fmaf_rn(pre[2], rn[2], o);
  o = __fmaf_rn(pre[3], rn[3], o);
  return o;
}

// Per-thread partial of TileLang's sum of squares from the 16 per-position chains sp[hh*8 + e].
__device__ __forceinline__ float vt_partial(const float (&sp)[16]) {
  float s = 0.f;
#pragma unroll
  for (int rv = 0; rv < 16; ++rv) s = __fadd_rn(s, sp[(rv & 1) * 8 + (rv >> 1)]);
  return s;
}

// Butterfly over the 64 virtual-thread partials, called by a full warp: lane l passes s_l and s_{l+32}.
// Every lane returns the total (bit-identical to every TileLang collapse thread's value).
__device__ __forceinline__ float vt_butterfly(float s_lo, float s_hi) {
  float a = __fadd_rn(s_lo, s_hi);  // offset 32 (TileLang: x + red[t ^ 32])
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, off));
  return a;
}

__device__ __forceinline__ float norm_rsqrt(float sumsq) {
  return rsqrtf(__fadd_rn(__fdiv_rn(sumsq, 5120.f), kNormEps));
}
__device__ __forceinline__ float norm_apply(float c_bf16_as_f32, float rs, float w) {
  return __fmul_rn(__fmul_rn(c_bf16_as_f32, rs), w);
}

// Column of virtual thread v, half hh, element e in block b.
__host__ __device__ constexpr int vt_col(int b, int hh, int v, int e) { return b * 1024 + hh * 512 + v * 8 + e; }

// ------------------------------------------------------------------------------------------------ stats
// The TileLang "stats" split for one token, run by 16 lanes of a warp (lanes 0-15 or 16-31; both halves of
// the warp must call it, possibly with valid = false). mul/sqr: DeepGEMM split partials ([ns][T][24], [ns][T]).
// Writes post[4], comb[16], pre[4] (the next pre-mix). Bit-exact: same sequential split sums, the same
// expf/IEEE-division sequence, and TileLang's AllReduce butterflies (row: xor 2, 1; column: xor 8, 4).
template <int NS>
__device__ __forceinline__ void stats_token(const float* __restrict__ mul, const float* __restrict__ sqr,
                                            int T, int t, bool valid, const float* __restrict__ scale,
                                            const float* __restrict__ base, float* __restrict__ post_out,
                                            float* __restrict__ comb_out, float* __restrict__ pre_out) {
  const int hl = threadIdx.x & 15;
  const int tt = valid ? t : 0;
  // All split partials are loaded up front (independent loads), then summed in split order.
  float sq[NS], mp[NS], mq[NS], mc[NS];
  const int jp = hl & 3, jq = (hl & 3) + 4, jc = hl + 8;
#pragma unroll
  for (int s = 0; s < NS; ++s) {
    const float* row = mul + ((size_t)s * T + tt) * kMix;
    sq[s] = sqr[s * T + tt];
    mp[s] = row[jp];
    mq[s] = row[jq];
    mc[s] = row[jc];
  }
  float rms = 0.f;
#pragma unroll
  for (int s = 0; s < NS; ++s) rms = __fadd_rn(rms, sq[s]);
  rms = rsqrtf(__fadd_rn(__fdiv_rn(rms, 20480.f), kNormEps));
  auto mix = [&](const float (&v)[NS]) {
    float m = 0.f;
#pragma unroll
    for (int s = 0; s < NS; ++s) m = __fadd_rn(m, v[s]);
    return __fmul_rn(m, rms);
  };
  const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
  if (valid && hl < 4) {
    pre_out[t * 4 + hl] = __fadd_rn(
        __fdiv_rn(1.f, __fadd_rn(1.f, expf(__fsub_rn(0.f, __fmaf_rn(mix(mp), s0, base[hl]))))), kHcEps);
    post_out[t * 4 + hl] = __fmul_rn(
        __fdiv_rn(1.f, __fadd_rn(1.f, expf(__fsub_rn(0.f, __fmaf_rn(mix(mq), s1, base[hl + 4]))))), 2.f);
  }
  float cm = __fmaf_rn(mix(mc), s2, base[hl + 8]);
  float mx = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, 2));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
  cm = expf(__fsub_rn(cm, mx));
  float rsum = __fadd_rn(cm, __shfl_xor_sync(0xffffffffu, cm, 2));
  rsum = __fadd_rn(rsum, __shfl_xor_sync(0xffffffffu, rsum, 1));
  cm = __fadd_rn(__fdiv_rn(cm, rsum), kHcEps);
  float csum = __fadd_rn(cm, __shfl_xor_sync(0xffffffffu, cm, 8));
  csum = __fadd_rn(csum, __shfl_xor_sync(0xffffffffu, csum, 4));
  cm = __fdiv_rn(cm, __fadd_rn(csum, kHcEps));
#pragma unroll 1
  for (int it = 0; it < kSinkhornIters - 1; ++it) {
    rsum = __fadd_rn(cm, __shfl_xor_sync(0xffffffffu, cm, 2));
    rsum = __fadd_rn(rsum, __shfl_xor_sync(0xffffffffu, rsum, 1));
    cm = __fdiv_rn(cm, __fadd_rn(rsum, kHcEps));
    csum = __fadd_rn(cm, __shfl_xor_sync(0xffffffffu, cm, 8));
    csum = __fadd_rn(csum, __shfl_xor_sync(0xffffffffu, csum, 4));
    cm = __fdiv_rn(cm, __fadd_rn(csum, kHcEps));
  }
  if (valid) comb_out[t * 16 + hl] = cm;
}

// The stats in two steps. (1) stats_sums: every thread of the CTA turns some of the (token, mix) split sums into
// scaled mixes m[t][j] = (sum_s mul[s][t][j]) * rms[t], rms[t] = rsqrtf((sum_s sqr[s][t]) / 20480 + eps), with
// TileLang's sequential split order (bit-identical). (2) stats_from_sums: 16 lanes per token, 2 tokens
// per warp: sigmoid pre/post and Sinkhorn. TileLang's row butterfly (xor 2, then 1 over the 4 columns) is the value
// (v0 + v2) + (v1 + v3) in every lane, computed locally; its column butterfly (lanes e, e^8, e^4) becomes xor 2,
// then xor 1 over the 4 row lanes.
template <int NS>
__device__ __forceinline__ void stats_sums(const float* __restrict__ mul, const float* __restrict__ sqr, int T, int t0,
                                           int nt, float* __restrict__ m_out /* [nt][25]: 24 mixes + rms */, int tid,
                                           int nthr) {
  for (int i = tid; i < nt * 25; i += nthr) {
    const int t = t0 + i / 25, j = i % 25;
    float v[NS];
    if (j < 24) {
#pragma unroll
      for (int s = 0; s < NS; ++s) v[s] = __ldcg(mul + ((size_t)s * T + t) * kMix + j);
    } else {
#pragma unroll
      for (int s = 0; s < NS; ++s) v[s] = __ldcg(sqr + s * T + t);
    }
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < NS; ++s) acc = __fadd_rn(acc, v[s]);
    m_out[i] = acc;  // raw sums; scaled below once rms is known
  }
}

// m: [nt][25] raw split sums from stats_sums (index 24 = sum of squares). 16 lanes per token (lanes 0-15 or 16-31,
// element e = lane & 15 of the 4x4 comb, as TileLang's stats threads); every lane of the warp must call.
__device__ __forceinline__ void stats_from_sums(const float* __restrict__ m, int g, int t, bool valid,
                                                const float* __restrict__ scale, const float* __restrict__ base,
                                                float* __restrict__ post_out, float* __restrict__ comb_out,
                                                float* __restrict__ pre_out) {
  const int hl = threadIdx.x & 15, h4 = hl & 3;
  const float* row = m + (valid ? g : 0) * 25;
  const float rms = rsqrtf(__fadd_rn(__fdiv_rn(row[24], 20480.f), kNormEps));
  const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
  // pre / post sigmoids in every lane (lanes 4..15 repeat lane hl & 3's values; only lanes 0..3 store): no branch
  // around them, so the scheduler overlaps them with the softmax start below. The arithmetic is unchanged.
  const float pre_v = __fadd_rn(
      __fdiv_rn(1.f, __fadd_rn(1.f, expf(__fsub_rn(0.f, __fmaf_rn(__fmul_rn(row[h4], rms), s0, base[h4]))))),
      kHcEps);
  const float post_v = __fmul_rn(
      __fdiv_rn(1.f, __fadd_rn(1.f, expf(__fsub_rn(0.f, __fmaf_rn(__fmul_rn(row[h4 + 4], rms), s1, base[h4 + 4]))))),
      2.f);
  float cm = __fmaf_rn(__fmul_rn(row[hl + 8], rms), s2, base[hl + 8]);
  float mx = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, 2));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
  cm = expf(__fsub_rn(cm, mx));
  float rsum = __fadd_rn(cm, __shfl_xor_sync(0xffffffffu, cm, 2));
  rsum = __fadd_rn(rsum, __shfl_xor_sync(0xffffffffu, rsum, 1));
  cm = __fadd_rn(__fdiv_rn(cm, rsum), kHcEps);
  float csum = __fadd_rn(cm, __shfl_xor_sync(0xffffffffu, cm, 8));
  csum = __fadd_rn(csum, __shfl_xor_sync(0xffffffffu, csum, 4));
  cm = __fdiv_rn(cm, __fadd_rn(csum, kHcEps));
#pragma unroll  // fully unrolled: no loop overhead on the dependent chain
  for (int it = 0; it < kSinkhornIters - 1; ++it) {
    rsum = __fadd_rn(cm, __shfl_xor_sync(0xffffffffu, cm, 2));
    rsum = __fadd_rn(rsum, __shfl_xor_sync(0xffffffffu, rsum, 1));
    cm = __fdiv_rn(cm, __fadd_rn(rsum, kHcEps));
    csum = __fadd_rn(cm, __shfl_xor_sync(0xffffffffu, cm, 8));
    csum = __fadd_rn(csum, __shfl_xor_sync(0xffffffffu, csum, 4));
    cm = __fdiv_rn(cm, __fadd_rn(csum, kHcEps));
  }
  if (valid && hl < 4) {
    pre_out[t * 4 + hl] = pre_v;
    post_out[t * 4 + hl] = post_v;
  }
  if (valid) comb_out[t * 16 + hl] = cm;
}

// ------------------------------------------------------------------------------------------------ tf32
__host__ __device__ inline uint32_t tf32_rne_bits(uint32_t u) {
  return (u + 0xfffu + ((u >> 13) & 1u)) & 0xffffe000u;
}

// DeepGEMM's per-split sum of squares for one row: lane quad q (0..3) of a 4-lane group, values x[k] for the
// split's k range in 64-wide stages; fma2 lanes x (k = s*64 + i*8 + 2q) and y (k+1); then (x + y) and a
// 4-lane butterfly (xor 2, then 1). `row` points at the split's first element. Called by 4 consecutive lanes.
__device__ __forceinline__ float sqrsum_split(const __nv_bfloat16* __restrict__ row, int kps, int q) {
  float sx = 0.f, sy = 0.f;
  for (int s = 0; s < kps / 64; ++s)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(row + s * 64 + i * 8 + 2 * q);
      const float a = __low2float(v), b = __high2float(v);
      sx = __fmaf_rn(a, a, sx);
      sy = __fmaf_rn(b, b, sy);
    }
  float v = __fadd_rn(sx, sy);
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return v;
}

// ------------------------------------------------------------------------------------------------ MXFP8
// FlashInfer cute-dsl mxfp8_quantize (and peer_fused2's in-kernel quantizer): per 32 bf16 values,
// e = ue8m0(amax * fl32(1/448)) [ceil, subnormal rule, clamp 254, 0 for amax <= 0], inv = 2^(127-e) (0 if e == 0),
// q = cvt.rn.satfinite.e4m3(clamp(x * inv, +-448)).
__device__ __forceinline__ uint32_t fi_ue8m0(float v) {
  if (!(v > 0.f)) return 0u;
  const uint32_t bits = __float_as_uint(v);
  const uint32_t ex = (bits >> 23) & 0xffu, mant = bits & 0x7fffffu;
  uint32_t bump = mant != 0u ? 1u : 0u;
  if (ex == 0u && mant <= 0x400000u) bump = 0u;
  const uint32_t r = ex + bump;
  return r > 254u ? 254u : r;
}
__device__ __forceinline__ float fi_inv_scale(uint32_t e) {
  if (e == 0u) return 0.f;
  int ne = 254 - (int)e;
  ne = ne < 0 ? 0 : ne;
  return __uint_as_float((uint32_t)ne << 23);
}
__device__ __forceinline__ uint16_t fp8x2_e4m3(float a, float b) {  // a -> low byte
  a = fmaxf(fminf(a, 448.f), -448.f);
  b = fmaxf(fminf(b, 448.f), -448.f);
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(b), "f"(a));
  return r;
}
// amax of 8 bf16 values exactly as FlashInfer (bf16 max of |x| is exact).
__device__ __forceinline__ float amax8(const float (&f)[8]) {
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) m = fmaxf(m, fabsf(f[i]));
  return m;
}
constexpr float kInvE4m3Max = 1.0f / 448.0f;

// 8 values -> 8 fp8 bytes (uint2) with the given inverse scale.
__device__ __forceinline__ uint2 fi_quant8(const float (&f)[8], float inv) {
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // FlashInfer multiplies first (mul.f32), then clamps; fp8x2_e4m3 clamps.
    w[i] = fp8x2_e4m3(__fmul_rn(f[2 * i], inv), __fmul_rn(f[2 * i + 1], inv));
  }
  return make_uint2(w[0] | (w[1] << 16), w[2] | (w[3] << 16));
}

// FlashInfer 128x4 swizzled scale-factor offset (padded_cols = K/32 rounded up to 4).
__host__ __device__ constexpr int sf_swizzled_128x4(int row, int col, int padded_cols) {
  return (col % 4) + (col / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + (row / 128) * (128 * padded_cols);
}

// ------------------------------------------------------------------------------------------------ MegaMoE prep
// Triton _prepare_megamoe_inputs_kernel: per 32 values amax (fp32 of bf16, max with 1e-4), scale = amax / 448
// (div.full.f32), exp = biased exponent + (mantissa != 0), clamp [1, 254]; values * div.full(1, 2^(exp-127)).
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ uint32_t prep_exp(float amax) {
  amax = fmaxf(amax, 1.0e-4f);
  const uint32_t bits = __float_as_uint(div_full(amax, 448.0f));
  uint32_t e = ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1u : 0u);
  e = e < 1u ? 1u : (e > 254u ? 254u : e);
  return e;
}
__device__ __forceinline__ float prep_inv(uint32_t e) { return div_full(1.0f, __uint_as_float(e << 23)); }
__device__ __forceinline__ uint2 prep_quant8(const float (&f)[8], float inv) {
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(__fmul_rn(f[2 * i + 1], inv)), "f"(__fmul_rn(f[2 * i], inv)));
    w[i] = r;
  }
  return make_uint2(w[0] | (w[1] << 16), w[2] | (w[3] << 16));
}
// Row of shared_x_sf (MN-major, permuted per MegaMoE BLOCK_M) for token t.
__host__ __device__ constexpr int prep_shared_row(int t, int sbm) {
  return (t / sbm) * (((sbm + 127) / 128) * 128) + ((t % sbm) / 128) * 128 + ((t % sbm) % 32) * 4 +
         ((t % sbm) % 128) / 32;
}

}  // namespace w4
