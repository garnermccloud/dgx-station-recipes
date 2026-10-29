// Shared device helpers of W6's fused DSpark Markov kernels: production's draft Gumbel draw (bit for bit), the
// packed winner key, and the tensor-core bias GEMV tile (m16n8k16, interleaved K mapping).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace w6mk_common {

constexpr int kRank = 256;  // dspark_markov_rank: one row = 512 B
constexpr int kTile = 16;   // rows per MMA tile

// ---------------------------------------------------------------- production's draft Gumbel draw, bit for bit
__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint32_t rotl32(uint32_t v, int s) { return (v << s) | (v >> (32 - s)); }
__device__ __forceinline__ uint32_t mm3_mix(uint32_t h, uint32_t key) {
  key *= 0xCC9E2D51u;
  key = rotl32(key, 15);
  key *= 0x1B873593u;
  h ^= key;
  h = rotl32(h, 13);
  return h * 5u + 0xE6546B64u;
}
__device__ __forceinline__ uint32_t mm3_fmix(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  return h ^ (h >> 16);
}
// gumbel.murmur3_hash32(seed, pos, offset, domain=0)
__device__ __forceinline__ uint32_t mm3_hash(long long seed, long long pos, uint32_t offset) {
  uint32_t h = 0;
  h = mm3_mix(h, static_cast<uint32_t>(seed & 0xFFFFFFFFll));
  h = mm3_mix(h, static_cast<uint32_t>((seed >> 32) & 0xFFFFFFFFll));
  h = mm3_mix(h, static_cast<uint32_t>(pos & 0xFFFFFFFFll));
  h = mm3_mix(h, offset);
  return mm3_fmix(h ^ 16u);
}
__device__ __forceinline__ float fdiv_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// gumbel.murmur3_uniform32: hi16 * 2^-16 + (lo16 + 0.5) * 2^-32, as Triton emits it (add, mul, fma)
__device__ __forceinline__ float mm3_uniform(long long seed, long long pos, uint32_t token) {
  const uint32_t r = mm3_hash(seed, pos, token);
  const float hi = static_cast<float>(static_cast<int>(r >> 16));
  const float lo = static_cast<float>(static_cast<int>(r & 0xFFFFu));
  return __fmaf_rn(hi, 1.52587890625e-05f, __fmul_rn(__fadd_rn(lo, 0.5f), 2.3283064365386963e-10f));
}
// gumbel._log1p_neg_stable(u) = log(1 - u)
__device__ __forceinline__ float log1p_neg_stable(float v) {
  float p = 0.125f;
  p = __fmaf_rn(v, p, 0.142857149f);  // 1/7
  p = __fmaf_rn(v, p, 0.166666672f);  // 1/6
  p = __fmaf_rn(v, p, 0.200000003f);  // 1/5
  p = __fmaf_rn(v, p, 0.25f);
  p = __fmaf_rn(v, p, 0.333333343f);  // 1/3
  p = __fmaf_rn(v, p, 0.5f);
  p = __fmaf_rn(v, p, 1.0f);
  const float series = __fmul_rn(-v, p);
  const float direct = logf(fmaxf(__fsub_rn(1.0f, v), 5.960464477539063e-08f));
  return v < 0.25f ? series : direct;
}
// value gumbel_noised_argmax maximizes for one token (IS_DRAFTING, APPLY_TEMPERATURE, fp32)
__device__ __forceinline__ float noised(float logit, float temp, long long seed, long long pos, uint32_t token) {
  if (temp == 0.0f) return logit;
  const float x = fdiv_full(logit, temp);
  const float u = mm3_uniform(seed, pos + (1ll << 30), token);
  const float g = -logf(-log1p_neg_stable(u));
  return __fadd_rn(x, g);
}
// W5's packed winner key: high 32 = order-preserving value (NaN on top, -0 == +0), low 31 = 2^31 - 1 - token
__device__ __forceinline__ unsigned long long pack(float v, int token) {
  uint32_t b = __float_as_uint(v == 0.0f ? 0.0f : v);
  uint32_t o = (b >> 31) ? ~b : (b | 0x80000000u);
  if (v != v) o = 0xFFFFFFFFu;
  return (static_cast<unsigned long long>(o) << 31) | static_cast<unsigned long long>(2147483647 - token);
}
__device__ __forceinline__ int unpack_token(unsigned long long k) { return 2147483647 - static_cast<int>(k & 2147483647ull); }

// ---------------------------------------------------------------- tensor-core GEMV
// m16n8k16 fragment ownership (lane = 4 g + t): A rows g, g + 8, logical k columns {2t, 2t+1, 2t+8, 2t+9} per k-block;
// B column g (= request), same k columns; C rows g, g + 8, columns 2t, 2t + 1. Logical (k-block kb, column c) maps to
// the physical k = 8 (4 (kb >> 1) + t) + 4 (kb & 1) + {0, 1} for c in {2t, 2t+1} and + {2, 3} for c in {2t+8, 2t+9}:
// at piece i lane t reads the 16 B chunk 4 i + t of a row, so the 4 lanes of a row read 64 contiguous bytes.
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// One 16-row tile: lo = row g's 128 B piece (8 x 16 B), hi = row g + 8's; e = this lane's embedding piece for
// request g (zeros past the batch). Returns C fragments (dots for rows g, g + 8 x requests 2t, 2t + 1).
__device__ __forceinline__ void tile_mma(float (&c)[4], const uint4 (&lo)[8], const uint4 (&hi)[8],
                                         const uint4 (&e)[8]) {
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // two independent accumulator chains, summed at the end
  c[0] = c[1] = c[2] = c[3] = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {  // k-blocks 2i, 2i + 1 live in the i-th 16 B piece
    mma16816(c, lo[i].x, hi[i].x, lo[i].y, hi[i].y, e[i].x, e[i].y);
    mma16816(d, lo[i].z, hi[i].z, lo[i].w, hi[i].w, e[i].z, e[i].w);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) c[j] = __fadd_rn(c[j], d[j]);
}
// Row tail for one (row, request): bias, logits, q-cache entry, noised value -> packed key.
__device__ __noinline__ unsigned long long row_key(float dot, __nv_bfloat16 b, int token, __nv_bfloat16* cache_row,
                                                   float temp, long long seed, long long pos) {
  const __nv_bfloat16 bias = __float2bfloat16_rn(dot);
  const __nv_bfloat16 logit = __float2bfloat16_rn(__fadd_rn(__bfloat162float(b), __bfloat162float(bias)));
  if (cache_row != nullptr) cache_row[token] = logit;
  return pack(noised(__bfloat162float(logit), temp, seed, pos, token), token);
}


}  // namespace w6mk_common
