// W6 v11: the MegaMoE fork v5 lane-parallel combine as a standalone kernel (no peer3 fold), for consumers of a call
// run with skip_combine = 1 that need y (W4's _finish_now fallback). Same arithmetic per element as the fork: fp32
// accumulator from 0, present slots in k order (routed 0..kTopk-1 where topk_idx >= 0, then the shared slot), one
// add.rn.f32.bf16 per element, then __float22bfloat162_rn. Build: build_combine.sh (sm_100a + sm_103a).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int kHidden = 5120, kPiece = 128, kPieces = kHidden / kPiece, kTopk = 6;

__device__ __forceinline__ void acc_bf16x2(float2& a, uint32_t b) {
  const uint16_t lo = static_cast<uint16_t>(b & 0xFFFFu), hi = static_cast<uint16_t>(b >> 16);
  asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.x) : "h"(lo));
  asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.y) : "h"(hi));
}

__global__ void w6_combine_kernel(uint8_t* __restrict__ y, const uint8_t* __restrict__ slot_base,
                                  unsigned long long slot_stride, unsigned long long token_stride,
                                  const int64_t* __restrict__ topk_idx, int num_tokens, int has_shared) {
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5;
  for (int item = warp; item < num_tokens * kPieces; item += nwarps) {
    const int token = item / kPieces, piece = item % kPieces;
    const int stored = lane < kTopk ? static_cast<int>(topk_idx[token * kTopk + lane]) : (has_shared && lane == kTopk ? kTopk : -1);
    const uint32_t mask = __ballot_sync(0xffffffffu, stored >= 0);
    const uint32_t byte_off = piece * kPiece * 2u + lane * 8u;
    uint2 v[kTopk + 1];
#pragma unroll
    for (int k = 0; k <= kTopk; ++k) {
      v[k] = make_uint2(0u, 0u);
      if ((mask >> k) & 1u)
        v[k] = __ldcg(reinterpret_cast<const uint2*>(slot_base + k * slot_stride + token * token_stride + byte_off));
    }
    float2 r0 = {0.f, 0.f}, r1 = {0.f, 0.f};
#pragma unroll
    for (int k = 0; k <= kTopk; ++k) {
      if ((mask >> k) & 1u) {
        acc_bf16x2(r0, v[k].x);
        acc_bf16x2(r1, v[k].y);
      }
    }
    const __nv_bfloat162 o0 = __float22bfloat162_rn(r0), o1 = __float22bfloat162_rn(r1);
    uint2 o;
    o.x = *reinterpret_cast<const uint32_t*>(&o0);
    o.y = *reinterpret_cast<const uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(y + static_cast<uint64_t>(token) * (kHidden * 2) + byte_off) = o;
  }
}
}  // namespace

extern "C" int w6_combine_launch(void* y, const void* slot_base, unsigned long long slot_stride,
                                 unsigned long long token_stride, const void* topk_idx, int num_tokens, int has_shared,
                                 int grid, void* stream) {
  if (num_tokens <= 0) return 0;
  w6_combine_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
      static_cast<uint8_t*>(y), static_cast<const uint8_t*>(slot_base), slot_stride, token_stride,
      static_cast<const int64_t*>(topk_idx), num_tokens, has_shared);
  return (int)cudaGetLastError();
}
