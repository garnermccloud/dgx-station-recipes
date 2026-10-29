// Device math of DeepGEMM's bf16 Mega gate (sm100_bf16_mega_gate_impl<5120, 384, 16, 12, 256, 1, 8, 3, .., 6,
// SqrtSoftplus, full tile, bias, image mask>), the router production runs for 16 < T: scoring and the warp top-k /
// weight epilogue, copied from deep_gemm/epilogue/sm100_mega_gate.cuh (built without fast math, as DeepGEMM's JIT).
// The epilogue is templated on the expert waves (E / 128) and top-k as DeepGEMM's is: the DSpark drafter's gate is
// sm100_bf16_mega_gate_impl<5120, 128, 16, 12, 256, 1, 8, 1, .., 3, ..> (same 8 K splits, one expert wave, top-3).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace w4::mgate {

constexpr int kExperts = 384, kTopk = 6, kSplitK = 8, kKPerSplit = 5120 / kSplitK;  // 640
constexpr int kWaves = kExperts / 128;                                               // 3
#define W4_MGATE_NEG_INF (-__int_as_float(0x7f800000))

__device__ __forceinline__ void sqrt_softplus4(float (&v)[4]) {
  float e[4], s[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) e[i] = expf(v[i]);
#pragma unroll
  for (int i = 0; i < 4; ++i) s[i] = log1pf(e[i]);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = sqrtf(v[i] > 20.0f ? v[i] : s[i]);
}

__device__ __forceinline__ float redux_max_f32(float v) {
  float r;
  asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(r) : "f"(v));
  return r;
}
__device__ __forceinline__ void warp_reduce_best(const float& score, int& expert_idx) {
  const float best_score = redux_max_f32(score);
  const uint32_t tied = score == best_score && expert_idx >= 0 ? static_cast<uint32_t>(expert_idx) : 0xffffffffu;
  expert_idx = static_cast<int>(__reduce_min_sync(0xffffffffu, tied));
}
__device__ __forceinline__ void compare_swap_best_first(float& ls, int& li, float& rs, int& ri) {
  if (rs > ls || (rs == ls && ri < li)) {
    const float s = ls;
    const int i = li;
    ls = rs, li = ri;
    rs = s, ri = i;
  }
}
__device__ __forceinline__ uint32_t sort4_best_first(float& s0, float& s1, float& s2, float& s3) {
  int i0 = 0, i1 = 1, i2 = 2, i3 = 3;
  compare_swap_best_first(s0, i0, s1, i1);
  compare_swap_best_first(s2, i2, s3, i3);
  compare_swap_best_first(s0, i0, s2, i2);
  compare_swap_best_first(s1, i1, s3, i3);
  compare_swap_best_first(s1, i1, s2, i2);
  return static_cast<uint32_t>(i0 | i1 << 2 | i2 << 4 | i3 << 6);
}
// select_warp_topk (expert_base_idx 0): lane k < K ends with the k-th pick.
template <int W = kWaves, int K = kTopk>
__device__ __forceinline__ void select_warp_topk(float (&scores)[W * 4], int lane, int& selected) {
  uint32_t perm[W];
#pragma unroll
  for (int w = 0; w < W; ++w) perm[w] = sort4_best_first(scores[w * 4], scores[w * 4 + 1], scores[w * 4 + 2], scores[w * 4 + 3]);
  uint32_t cursors = 0;
#pragma unroll
  for (int out = 0; out < K; ++out) {
    float best = W4_MGATE_NEG_INF;
    int best_idx = -1;
    uint32_t best_wave = 0;
#pragma unroll
    for (int w = 0; w < W; ++w) {
      const uint32_t c = cursors >> (w * 3) & 7;
      const float cand = c == 4 ? W4_MGATE_NEG_INF
                         : c & 2 ? (c & 1 ? scores[w * 4 + 3] : scores[w * 4 + 2])
                                 : (c & 1 ? scores[w * 4 + 1] : scores[w * 4 + 0]);
      const uint32_t off = perm[w] >> (c * 2) & 3;
      const int idx = static_cast<int>(w * 128 + lane * 4 + off);
      if (cand > best) {
        best = cand;
        best_idx = idx;
        best_wave = w;
      }
    }
    const int local = best_idx;
    warp_reduce_best(best, best_idx);
    if (local == best_idx) cursors += 1u << (best_wave * 3);
    if (lane == out) selected = best_idx;
  }
}
template <int W = kWaves>
__device__ __forceinline__ float select_warp_expert_value(const float (&values)[W * 4], int selected) {
  const int vi = selected >= 0 ? selected / 128 * 4 + selected % 4 : -1;
  const int src = selected >= 0 ? selected % 128 / 4 : 0;
  float r = 0.0f;
#pragma unroll
  for (int i = 0; i < W * 4; ++i) {
    const float x = __shfl_sync(0xffffffffu, values[i], src);
    if (vi == i) r = x;
  }
  return r;
}

// One warp per token: `logit(e)` returns the split-summed GEMM output for expert e (lane-local experts
// w * 128 + 4 lane + v), bias the [W * 128] score bias. Lane k < K returns pick k (expert id, final weight).
template <int W = kWaves, int K = kTopk, class Logit>
__device__ __forceinline__ void route_token(Logit logit, const float* bias, float routed_scale, int lane, int& id,
                                            float& weight) {
  static_assert(K <= 8, "DeepGEMM sums the picks with warp_reduce_sum<8> only for top-k <= 8");
  float sc[W * 4], unb[W * 4];
#pragma unroll
  for (int w = 0; w < W; ++w) {
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = logit(w * 128 + lane * 4 + i);
    sqrt_softplus4(v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      unb[w * 4 + i] = v[i];
      sc[w * 4 + i] = v[i] + bias[w * 128 + lane * 4 + i];
    }
  }
  int sel = -1;
  select_warp_topk<W, K>(sc, lane, sel);
  if (lane < K) sel = min(max(sel, 0), W * 128 - 1);
  const float s = select_warp_expert_value<W>(unb, sel);
  // warp_reduce_sum<8> (xor 4, 2, 1) + 1e-20: DeepGEMM's kSumWidth is 8 for any top-k <= 8
  float sum = s;
  sum = sum + __shfl_xor_sync(0xffffffffu, sum, 4);
  sum = sum + __shfl_xor_sync(0xffffffffu, sum, 2);
  sum = sum + __shfl_xor_sync(0xffffffffu, sum, 1);
  sum = sum + 1e-20f;
  id = sel;
  weight = s / sum * routed_scale;
}

}  // namespace w4::mgate
