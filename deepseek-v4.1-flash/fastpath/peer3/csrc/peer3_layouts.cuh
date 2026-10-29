// Weight readers for the RTX kernel. A lane reads 4 rows (n8i = 0..3 -> row r8 + 8 * n8i of its 32-row group) x
// 8 consecutive K values (k = kt * 128 + k32 * 32 + cgrp * 8) as one 32-bit word per row (element i in nibble i),
// plus the 4 rows' E8M0 scales of that 32-K block (byte n8i).
#pragma once
#include "peer3_rtx.cuh"

namespace peer3 {

struct W4 {
  uint4 w;
  uint32_t s;
};

__device__ __forceinline__ uint32_t ldg_nc_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ldg_nc_u8(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u8 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

// fmt 0: checkpoint layout as the sidecar stacks it. w13 [E][4608][2560] rows = [w1 (gate) 0..2303; w3 (up)],
// s13 [E][4608][160]; w2 [E][5120][1152], s2 [E][5120][72]. Low nibble = even k.
__device__ __forceinline__ W4 load13_ckpt(const LayerWeights& L, int e, int mat, int g, int kt, int k32, int r8,
                                          int cgrp) {
  W4 r;
  const int k = kt * 128 + k32 * 32 + cgrp * 8;
  uint32_t w[4], s = 0;
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const size_t row = size_t(e) * 2 * kInter + (mat ? kInter : 0) + g * 32 + n * 8 + r8;
    w[n] = ldg_nc_u32(L.w13 + row * (kHidden / 2) + k / 2);
    s |= ldg_nc_u8(L.s13 + row * (kHidden / 32) + k / 32) << (8 * n);
  }
  r.w = make_uint4(w[0], w[1], w[2], w[3]);
  r.s = s;
  return r;
}

__device__ __forceinline__ W4 load2_ckpt(const LayerWeights& L, int e, int g, int kt, int k32, int r8, int cgrp) {
  W4 r;
  const int k = kt * 128 + k32 * 32 + cgrp * 8;
  uint32_t w[4], s = 0;
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const size_t row = size_t(e) * kHidden + g * 32 + n * 8 + r8;
    w[n] = ldg_nc_u32(L.w2 + row * (kInter / 2) + k / 2);
    s |= ldg_nc_u8(L.s2 + row * (kInter / 32) + k / 32) << (8 * n);
  }
  r.w = make_uint4(w[0], w[1], w[2], w[3]);
  r.s = s;
  return r;
}

// Weights are streamed once per call: no L1 allocation, L2 evict-first.
__device__ __forceinline__ uint4 ldg_nc_v4(const void* p) {
  uint4 v;
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p), "l"(policy));
  return v;
}

// fmt 1: b12x w4a8_mx prepared ("rp", N256/K128 tiles), verified bit for bit against b12x prepare_weights
// (b12x-spec/spec.py). For R rows, K reduction, NT = R/256, KT = K/128 and the w13 row rotation by I:
//   p = (row - rot) mod R, nt = p >> 8, q = p & 255 (n8c = q >> 5, n8i = (q >> 3) & 3, r8 = q & 7)
//   word  = ((e*NT + nt)*KT + kt)*4096 + (k32<<10 | n8c<<7 | r8<<4 | cgrp<<2 | n8i), nibble = k & 7
//   scale = ((e*NT + nt)*KT + kt)*1024 + (q >> 3)*32 + (q & 7)*4 + kb
// After the rotation, prepared rows [0, I) are w3 (up) and [I, 2I) are w1 (gate).
__device__ __forceinline__ W4 load_rp(const uint8_t* w, const uint8_t* sc, int tile, int n8c, int k32, int r8,
                                      int cgrp) {
  W4 r;
  r.w = ldg_nc_v4(w + (size_t(tile) * 4096 + (k32 << 10 | n8c << 7 | r8 << 4 | cgrp << 2)) * 4);
  const uint8_t* sb = sc + size_t(tile) * 1024 + n8c * 128 + r8 * 4;
  uint32_t s = 0;
#pragma unroll
  for (int n = 0; n < 4; ++n) s |= ((ldg_nc_u32(sb + n * 32) >> (8 * k32)) & 0xFF) << (8 * n);
  r.s = s;
  return r;
}

__device__ __forceinline__ W4 load13(const LayerWeights& L, int e, int mat, int g, int kt, int k32, int r8, int cgrp) {
  if (L.fmt == 1) {
    const int nt = (mat == 0 ? kInter / 256 : 0) + (g >> 3);  // gate rows sit after the up rows
    return load_rp(L.w13, L.s13, (e * (2 * kInter / 256) + nt) * kKt1 + kt, g & 7, k32, r8, cgrp);
  }
  return load13_ckpt(L, e, mat, g, kt, k32, r8, cgrp);
}
__device__ __forceinline__ W4 load2(const LayerWeights& L, int e, int g, int kt, int k32, int r8, int cgrp) {
  if (L.fmt == 1) return load_rp(L.w2, L.s2, (e * (kHidden / 256) + (g >> 3)) * kKt2 + kt, g & 7, k32, r8, cgrp);
  return load2_ckpt(L, e, g, kt, k32, r8, cgrp);
}

}  // namespace peer3
