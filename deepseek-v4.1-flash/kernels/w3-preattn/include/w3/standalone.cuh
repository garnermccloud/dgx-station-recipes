// Standalone (PDL) kernel shell for one or two W3 stages: the same per-stage task loop the persistent kernel runs.
// Unified task space [A tasks][B tasks]; CTA b runs unified tasks b, b+G, ... Smem: [ring slots][stage smem]
// [ring barriers][tmem slot]. Warp 8 lane 0 produces weights for all of this CTA's tasks (stage order), then
// triggers launch_dependents; compute warps run consume() per task.
#pragma once
#include "w3/mxf8_core.cuh"

namespace w3 {

struct NoStage {
  struct Params {
    int dummy;
  };
  static constexpr bool kHasWeights = false;
  static constexpr int kTraceId = 4;
  __host__ __device__ static int tasks(const Params&) { return 0; }
  __device__ static void produce(const Params&, int, Ring&) {}
  template <int N, class Sync>
  __device__ static void consume(const Params&, int, int, bool, Sync&, Ring&, uint8_t*, uint32_t) {}
};

template <class A, class B>
struct StagePair {
  typename A::Params a;
  typename B::Params b;
};

template <class S>
__device__ __forceinline__ const CUtensorMap* stage_tmap(const typename S::Params& p) {
  if constexpr (S::kHasWeights) return &p.tm_w;
  else return nullptr;
}

__host__ __device__ constexpr uint32_t standalone_smem(uint32_t nslots, uint32_t stage_bytes) {
  return nslots * kSlotBytes + ((stage_bytes + 1023) / 1024) * 1024 + 2 * nslots * 8 + 64;
}

template <class A, class B, int N>
__global__ void __launch_bounds__(kThreads, 1) stage_kernel(const __grid_constant__ StagePair<A, B> pp, int nslots,
                                                             uint32_t stage_bytes, int trigger) {
  extern __shared__ __align__(1024) uint8_t smem[];
  trace(A::kTraceId, 0);
  uint8_t* stage_smem = smem + nslots * kSlotBytes;
  uint64_t* bars = reinterpret_cast<uint64_t*>(stage_smem + ((stage_bytes + 1023) / 1024) * 1024);
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(bars + 2 * nslots);
  Ring ring = Ring::init(smem, bars, nslots, kSlotBytes);
  const int na = A::tasks(pp.a), nb = B::tasks(pp.b), nt = na + nb;
  bool any_mma = false;
  for (int u = blockIdx.x; u < nt; u += gridDim.x) any_mma |= (u < na) ? A::kHasWeights : B::kHasWeights;
  if (warp_id() == 0 && any_mma) {
    tmem_alloc(tmem_slot, kTmemCols);
    tmem_relinquish();
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = any_mma ? *tmem_slot : 0u;
  trace(A::kTraceId, 1);
  StandaloneSync sync;
  if (trigger == 1) griddep_launch_dependents();
  if (warp_id() == kProducerWarp) {
    if (lane_id() == 0) {
      if constexpr (A::kHasWeights) prefetch_tmap(stage_tmap<A>(pp.a));
      if constexpr (B::kHasWeights) prefetch_tmap(stage_tmap<B>(pp.b));
      for (int u = blockIdx.x; u < nt; u += gridDim.x) {
        if (u < na) A::produce(pp.a, u, ring);
        else B::produce(pp.b, u - na, ring);
      }
      if (g_trace_enabled()) trace_any(A::kTraceId, 14);
    }
    __syncwarp();
    if (trigger >= 0) griddep_launch_dependents();  // -1: dependents launch at exit
  } else {
    int ia = 0, ib = 0;
    for (int u = blockIdx.x; u < nt; u += gridDim.x) {
      const bool last = u + static_cast<int>(gridDim.x) >= nt;
      if (u < na) {
        const bool last_a = u + static_cast<int>(gridDim.x) >= na;
        A::template consume<N, StandaloneSync>(pp.a, u, ia++, last_a, sync, ring, stage_smem, tmem);
      } else {
        B::template consume<N, StandaloneSync>(pp.b, u - na, ib++, last, sync, ring, stage_smem, tmem);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  trace(A::kTraceId, 15);
  if (warp_id() == 0 && any_mma) tmem_dealloc(tmem, kTmemCols);
}

}  // namespace w3
