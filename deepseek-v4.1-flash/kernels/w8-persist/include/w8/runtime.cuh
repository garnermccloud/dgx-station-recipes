// W8 persistent-kernel framework: CTA shape, the weight ring, sync policies, task placement, launch prologue/epilogue.
//
// Execution model (CONTRACT.md): G = #SMs CTAs, one per SM, all co-resident (cooperative launch attribute). CTA b runs
// task t of a stage placed at `offset` iff t == (b - offset) mod G (+ k*G). Stages run in a fixed order on every CTA, a
// task waits only on tasks of earlier stages, so progress is guaranteed by induction over the stage order.
#pragma once
#include "sync.cuh"
#include "umma.cuh"

namespace w8 {

#ifndef W8_COMPUTE_WARPS
#define W8_COMPUTE_WARPS 8  // CONTRACT v1 shape; GEMM-heavy kernels build with 16 (two consumer groups)
#endif
constexpr int kComputeWarps = W8_COMPUTE_WARPS;
constexpr int kComputeThreads = kComputeWarps * 32;
constexpr int kProducerWarp = kComputeWarps;
constexpr int kThreads = kComputeThreads + 32;
constexpr uint32_t kBarCompute = 1;  // named barrier of the 256 compute threads

__device__ __forceinline__ void compute_sync() { bar_sync(kBarCompute, kComputeThreads); }

// ---------------------------------------------------------------- task placement
__host__ __device__ constexpr int first_task(int cta, int offset, int G) { return ((cta - offset) % G + G) % G; }

// ---------------------------------------------------------------- weight ring (per CTA, in SMEM)
// Producer: one lane of the producer warp walks the CTA's tiles in consumption order. Consumer: the thread that reads a
// slot (the MMA issuer) releases it with tcgen05.commit (or a plain arrive) on the slot's empty barrier.
template <int NSLOT_, int SLOT_BYTES_>
struct Ring {
  static constexpr int NSLOT = NSLOT_;
  static constexpr int SLOT_BYTES = SLOT_BYTES_;
  uint8_t* base;
  uint64_t* full;   // [NSLOT], count 1 (+tx)
  uint64_t* empty;  // [NSLOT], count empty_count
  uint32_t empty_count = 1;

  __device__ uint8_t* slot(uint32_t it) const { return base + (it % NSLOT) * SLOT_BYTES; }
  __device__ uint64_t* full_bar(uint32_t it) const { return &full[it % NSLOT]; }
  __device__ uint64_t* empty_bar(uint32_t it) const { return &empty[it % NSLOT]; }
  // producer side
  __device__ uint8_t* acquire(uint32_t it, Diag* diag) const {
    if (it >= static_cast<uint32_t>(NSLOT)) mbar_wait(empty_bar(it), ((it / NSLOT) - 1) & 1, diag, 0xE0);
    return slot(it);
  }
  __device__ void submit(uint32_t it, uint32_t bytes) const { mbar_arrive_expect_tx(full_bar(it), bytes); }
  // consumer side
  __device__ uint8_t* wait_full(uint32_t it, Diag* diag) const {
    mbar_wait(full_bar(it), (it / NSLOT) & 1, diag, 0xE1);
    return slot(it);
  }
  // full: the producer's expect_tx arrival (+ tx bytes); empty: one arrival per releasing consumer (1 for a
  // tcgen05.commit, the number of consumer warps for mma.sync consumers).
  __device__ void init_bars(uint32_t full_count = 1) const {
    for (int i = 0; i < NSLOT; ++i) {
      mbar_init(&full[i], full_count);
      mbar_init(&empty[i], empty_count);
    }
  }
  // Release a slot from one thread of a stage that read it with plain loads (all its readers synchronized first):
  // one arrive carrying the ring's full count, whatever the ring's own consumers usually arrive with.
  __device__ void release(uint32_t it) const { mbar_arrive_cnt(empty_bar(it), empty_count); }
};

// ---------------------------------------------------------------- sync policies
// Persistent: hand-offs are tagged flags in this width's scratch arena.
struct PersistSync {
  uint32_t tag;
  Diag* diag;
  // all compute threads: block until flags[lo, hi) == tag
  __device__ void wait(const Dep& d, uint32_t site, bool async_reader = false) const {
    for (int i = d.lo + static_cast<int>(threadIdx.x); i < d.hi; i += kComputeThreads)
      spin_until_eq(d.flags + static_cast<size_t>(i) * d.stride, tag, diag, site, static_cast<uint32_t>(i));
    if (async_reader) fence_proxy_async_global();
    compute_sync();
  }
  // one thread (any) waits; for warps that then issue async copies themselves
  __device__ void wait_one_thread(const Dep& d, uint32_t site) const {
    for (int i = d.lo; i < d.hi; ++i) spin_until_eq(d.flags + static_cast<size_t>(i) * d.stride, tag, diag, site, static_cast<uint32_t>(i));
    fence_proxy_async_global();
  }
  // a warp waits (lane i polls flags lo+i, lo+i+32, ...)
  __device__ void wait_warp(const Dep& d, uint32_t site) const {
    for (int i = d.lo + static_cast<int>(lane_id()); i < d.hi; i += 32)
      spin_until_eq(d.flags + static_cast<size_t>(i) * d.stride, tag, diag, site, static_cast<uint32_t>(i));
    __syncwarp();
    fence_proxy_async_global();
  }
  __device__ void wait_extern() const {}  // the prologue ran griddepcontrol.wait on every thread
  // all compute threads: publish flags[i] = tag after this CTA's prior writes
  __device__ void publish(uint32_t* flags, int i, int stride = 1) const {
    compute_sync();
    if (threadIdx.x == 0) {
      fence_acq_rel_gpu();
      st_relaxed_gpu(flags + static_cast<size_t>(i) * stride, tag);
    }
  }
  // publish from a subset of threads that already synchronized among themselves (e.g. one epilogue warp group)
  __device__ void publish_one_thread(uint32_t* flags, int i, int stride = 1) const {
    fence_acq_rel_gpu();
    st_relaxed_gpu(flags + static_cast<size_t>(i) * stride, tag);
  }
};

// Standalone (one stage per launch, PDL-chained): every input comes from an earlier launch.
struct StandaloneSync {
  uint32_t tag = 1;
  Diag* diag = nullptr;
  __device__ void wait(const Dep&, uint32_t, bool = false) const { pdl_wait(); compute_sync(); }
  __device__ void wait_one_thread(const Dep&, uint32_t) const { pdl_wait(); }
  __device__ void wait_warp(const Dep&, uint32_t) const { pdl_wait(); }
  __device__ void wait_extern() const { pdl_wait(); }
  __device__ void publish(uint32_t*, int, int = 1) const {}
  __device__ void publish_one_thread(uint32_t*, int, int = 1) const {}
};

// ---------------------------------------------------------------- launch prologue / epilogue
// Every thread: griddepcontrol.wait (inputs of earlier launches visible), then the tag. Weights may be fetched before.
__device__ __forceinline__ uint32_t read_tag(const Epoch* ep) {
  pdl_wait();
  return ld_relaxed_gpu(&ep->epoch);
}

// All threads of the CTA call this after their last access of this launch's hand-offs. The last CTA bumps the epoch
// and zeroes the launch-local counters (reset[0 .. nreset), e.g. pacing counters) for the next launch.
__device__ __forceinline__ void epoch_arrive(Epoch* ep, uint32_t tag, uint32_t* reset = nullptr, int nreset = 0,
                                             int reset_stride = 1) {
  __syncthreads();
  if (threadIdx.x == 0) {
    // the atom's release (cumulative over what the barrier made visible to this thread) orders the CTA's accesses
    const uint32_t old = atom_add_acq_rel_gpu(&ep->arrive, 1u);
    if (old == gridDim.x - 1) {
      for (int i = 0; i < nreset; ++i) st_relaxed_gpu(reset + i * reset_stride, 0u);
      st_relaxed_gpu(&ep->arrive, 0u);
      st_relaxed_gpu(&ep->epoch, next_tag(tag));
    }
  }
}

}  // namespace w8
