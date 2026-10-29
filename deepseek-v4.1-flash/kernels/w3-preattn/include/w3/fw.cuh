// Minimal stand-in for the W8 persistent-kernel framework (CONTRACT.md v1): 256 compute threads + 1 producer warp
// (warp 8), a TMA weight ring fed by the producer, StandaloneSync = griddepcontrol.wait, run_tasks = the per-stage
// task loop (CTA b runs tasks b, b+G, ...). The W3 stages are written against this interface so they fold into the
// W8 kernel by swapping this header for <w8/...>.
#pragma once
#include "w3/ptx.cuh"

namespace w3 {

// Debug timeline (tests only): when g_trace != nullptr, thread 0 of each CTA records %globaltimer at event slots.
#ifdef W3_TRACE
__device__ unsigned long long* g_trace;
__device__ __forceinline__ void trace(int kid, int ev) {
  if (g_trace && threadIdx.x == 0) g_trace[(kid * 256 + blockIdx.x) * 16 + ev] = globaltimer();
}
__device__ __forceinline__ void trace_any(int kid, int ev) {
  if (g_trace) g_trace[(kid * 256 + blockIdx.x) * 16 + ev] = globaltimer();
}
__device__ __forceinline__ bool g_trace_enabled() { return g_trace != nullptr; }
#else
__device__ __forceinline__ bool g_trace_enabled() { return false; }
__device__ __forceinline__ void trace(int, int) {}
__device__ __forceinline__ void trace_any(int, int) {}
#endif

constexpr int kComputeThreads = 256;
constexpr int kProducerWarp = 8;
constexpr int kThreads = 288;

__device__ __forceinline__ void compute_sync() { named_bar_sync(1, kComputeThreads); }

// Producer/consumer weight ring: n slots of slot_bytes in shared memory, full/empty mbarriers.
// The producer (one lane of warp 8) and the MMA-issuing consumer thread each keep their own running index, so the
// same Ring object (copied by value into each role) works across all tasks and stages of a CTA.
struct Ring {
  uint8_t* base;
  uint64_t* full;
  uint64_t* empty;
  uint32_t nslots;
  uint32_t slot_bytes;
  uint32_t idx;

  __device__ static Ring init(uint8_t* base, uint64_t* bars, uint32_t nslots, uint32_t slot_bytes) {
    Ring r{base, bars, bars + nslots, nslots, slot_bytes, 0};
    if (threadIdx.x == 0) {
      for (uint32_t i = 0; i < nslots; ++i) {
        mbar_init(&r.full[i], 1);
        mbar_init(&r.empty[i], 1);
      }
      fence_mbar_init();
    }
    return r;
  }
  // producer side
  __device__ uint8_t* acquire(uint32_t tx_bytes) {
    const uint32_t s = idx % nslots;
    if (idx >= nslots) mbar_wait(&empty[s], ((idx / nslots) - 1) & 1);
    mbar_arrive_expect_tx(&full[s], tx_bytes);
    return base + s * slot_bytes;
  }
  __device__ uint64_t* bar() const { return &full[idx % nslots]; }
  __device__ void advance() { ++idx; }
  // consumer side (MMA-issuing thread)
  __device__ uint8_t* wait_full() {
    const uint32_t s = idx % nslots;
    mbar_wait(&full[s], (idx / nslots) & 1);
    return base + s * slot_bytes;
  }
  __device__ uint64_t* empty_bar() const { return &empty[idx % nslots]; }
};

// Standalone (PDL) mode: every dependency is the previous kernel launch.
struct StandaloneSync {
  __device__ void wait() { griddep_wait(); }
  __device__ void wait_extern() { griddep_wait(); }
};

// Per-CTA scratch every stage may use: one TMEM allocation and a few mbarriers.
struct CtaCtx {
  uint32_t tmem;
  uint64_t* done;  // tcgen05.commit target for "MMAs of this task complete"
  uint32_t done_phase;
};

}  // namespace w3
