// W4 router math, bit-exact against vLLM af7f9488's decode router for 5 <= T <= 16:
//   GateLinear -> LLBf16SplitK (cute-dsl, tile 16x16x256, split_k 6 ranks round-robin over 20 K tiles, 4 MMA warps,
//   warp w owns k-blocks w, w+4, w+8, w+12 of each tile, mma.sync m16n8k16 bf16 -> f32), warp partials and rank
//   partials summed sequentially from +0;
//   -> dsv4_topk (Triton): sqrtsoftplus via ex2.approx / libdevice logf (ftz) / sqrt.approx.ftz, bias (bias_vl for
//   image-sentinel tokens), NaN -> -1e30, 6 argmax rounds (lowest id on ties), weights renormalized by
//   div.full(1.5, ((w0+w4)+w2) + ((w1+w5)+w3)).
#pragma once

#include <cstdint>
#include <cuda_bf16.h>

namespace w4 {

constexpr int kExperts = 384;
constexpr int kTopk = 6;
constexpr int kRouterTileK = 256;
constexpr int kRouterKTiles = 5120 / kRouterTileK;  // 20
constexpr int kRouterSplitK = 6;
constexpr int kImageSentinelLo = 129264;

__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// libdevice __nv_logf as Triton links it (ftz), instruction for instruction (see dsv4_topk PTX).
__device__ __forceinline__ float triton_logf(float x) {
  float r;
  asm("{\n"
      " .reg .pred p_small, p_fin;\n"
      " .reg .f32 xx, eadj, fe, f, q, t;\n"
      " .reg .b32 ix, e, m;\n"
      " setp.lt.f32 p_small, %1, 0f00800000;\n"
      " mul.f32 t, %1, 0f4B000000;\n"
      " selp.f32 xx, t, %1, p_small;\n"
      " selp.f32 eadj, 0fC1B80000, 0f00000000, p_small;\n"
      " mov.b32 ix, xx;\n"
      " add.s32 e, ix, -1059760811;\n"
      " and.b32 e, e, -8388608;\n"
      " sub.s32 m, ix, e;\n"
      " cvt.rn.f32.s32 fe, e;\n"
      " fma.rn.ftz.f32 fe, fe, 0f34000000, eadj;\n"
      " mov.b32 f, m;\n"
      " add.f32 f, f, 0fBF800000;\n"
      " fma.rn.ftz.f32 q, 0fBE055027, f, 0f3E1039F6;\n"
      " fma.rn.ftz.f32 q, q, f, 0fBDF8CDCC;\n"
      " fma.rn.ftz.f32 q, q, f, 0f3E0F2955;\n"
      " fma.rn.ftz.f32 q, q, f, 0fBE2AD8B9;\n"
      " fma.rn.ftz.f32 q, q, f, 0f3E4CED0B;\n"
      " fma.rn.ftz.f32 q, q, f, 0fBE7FFF22;\n"
      " fma.rn.ftz.f32 q, q, f, 0f3EAAAA78;\n"
      " fma.rn.ftz.f32 q, q, f, 0fBF000000;\n"
      " mul.f32 t, f, q;\n"
      " fma.rn.ftz.f32 t, t, f, f;\n"
      " fma.rn.ftz.f32 %0, fe, 0f3F317218, t;\n"
      " setp.lt.u32 p_fin, ix, 2139095040;\n"
      " @!p_fin fma.rn.ftz.f32 %0, xx, 0f7F800000, 0f7F800000;\n"
      "}\n"
      : "=f"(r)
      : "f"(x));
  return r;
}

// sqrt(softplus(l)) as dsv4_topk computes it.
__device__ __forceinline__ float sqrt_softplus(float l) {
  float e, s;
  asm("{\n .reg .f32 t;\n mul.f32 t, %1, 0f3FB8AA3B;\n ex2.approx.f32 %0, t;\n}\n" : "=f"(e) : "f"(l));
  const float lg = triton_logf(__fadd_rn(e, 1.0f));
  const float v = l > 20.0f ? l : lg;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(s) : "f"(v));
  return s;
}

__device__ __forceinline__ float div_full_f32(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// Top-6 of one token, one warp. lane l holds logits of experts l + 32 i (i < 12). Returns (in every lane) the
// 6 expert ids and renormalized weights in slots 0..5.
__device__ __forceinline__ void topk6_warp(const float (&logit)[12], const float* __restrict__ bias, bool image,
                                           const float* __restrict__ bias_vl, int (&ids)[6], float (&wts)[6],
                                           float routed_scale) {
  const int lane = threadIdx.x & 31;
  float w[12], cur[12];
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    const int e = lane + 32 * i;
    w[i] = sqrt_softplus(logit[i]);
    const float b = image ? bias_vl[e] : bias[e];
    float c = __fadd_rn(w[i], b);
    cur[i] = c == c ? c : -1e30f;
  }
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    float bv = cur[0];
    int bi = 0;
#pragma unroll
    for (int i = 1; i < 12; ++i)
      if (cur[i] > bv) bv = cur[i], bi = i;
    int be = lane + 32 * bi;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
      const int oe = __shfl_xor_sync(0xffffffffu, be, off);
      if (ov > bv || (ov == bv && oe < be)) bv = ov, be = oe;
    }
    ids[k] = be;
    const int owner = be & 31, slot = be >> 5;
    float mine = 0.f;
#pragma unroll
    for (int i = 0; i < 12; ++i)
      if (i == slot && lane == owner) mine = w[i], cur[i] = -__int_as_float(0x7f800000);
    wts[k] = __shfl_sync(0xffffffffu, mine, owner);
  }
  const float ws = __fadd_rn(__fadd_rn(__fadd_rn(wts[0], wts[4]), wts[2]), __fadd_rn(__fadd_rn(wts[1], wts[5]), wts[3]));
  const float f = div_full_f32(routed_scale, ws > 0.f ? ws : 1.0f);
#pragma unroll
  for (int k = 0; k < 6; ++k) wts[k] = __fmul_rn(f, wts[k]);
}

}  // namespace w4
