// W8 persistent-kernel framework: epoch tags, bounded spins, flags, tagged mailboxes, the launch epoch.
//
// A hand-off word is "ready" iff it equals the launch's tag. Tags are the launch epoch (a device counter bumped by the
// last CTA of every launch), so nothing is ever cleared: a word from an earlier launch carries an older tag.
#pragma once
#include <cstdio>
#include "ptx.cuh"

#ifndef W8_SPIN_TIMEOUT_NS
#define W8_SPIN_TIMEOUT_NS 500000000ull  // 0.5 s: far above any legitimate wait, far below a watchdog
#endif

namespace w8 {

// Diagnostics of a timed-out spin, written before __trap(). Point it at host-mapped memory to read it after the trap.
struct Diag {
  uint32_t hit;       // 1 once something timed out
  uint32_t site;      // caller-chosen site id (stage id << 8 | wait kind)
  uint32_t cta;
  uint32_t task;
  uint32_t want;
  uint32_t seen;
  uint32_t tid;
  uint32_t pad;
};

static __device__ __noinline__ void spin_fail(Diag* diag, uint32_t site, uint32_t task, uint32_t want, uint32_t seen) {
  if (diag != nullptr) {
    diag->site = site;
    diag->cta = blockIdx.x;
    diag->task = task;
    diag->want = want;
    diag->seen = seen;
    diag->tid = threadIdx.x;
    fence_acq_rel_sys();
    diag->hit = 1;
    fence_acq_rel_sys();
  }
  printf("W8 spin timeout: site 0x%x cta %d tid %d task %u want %u seen %u\n", site, blockIdx.x, threadIdx.x, task, want,
         seen);
  __trap();
}

#ifndef W8_SPIN_BACKOFF_NS
#define W8_SPIN_BACKOFF_NS 64  // pollers sleep between polls so they don't steal issue slots from co-resident warps
#endif

// Spin until *p == want (acquire). Bounded: traps after W8_SPIN_TIMEOUT_NS.
__device__ __forceinline__ void spin_until_eq(const uint32_t* p, uint32_t want, Diag* diag, uint32_t site, uint32_t task) {
  uint32_t v = ld_acquire_gpu(p);
  if (v == want) return;
  const uint64_t t0 = globaltimer();
  uint32_t n = 0;
  while (true) {
    __nanosleep(W8_SPIN_BACKOFF_NS);
    v = ld_acquire_gpu(p);
    if (v == want) return;
    if ((++n & 15) == 0 && globaltimer() - t0 > W8_SPIN_TIMEOUT_NS) spin_fail(diag, site, task, want, v);
  }
}

// One thread waits until all n (multiple of 4, 16-byte aligned) flags hold `want`: every round loads all of them at
// once (relaxed: for pacing, e.g. a producer that holds back prefetch, not for data ordering), then sleeps `sleep_ns`.
__device__ __forceinline__ void wait_all_paced(const uint32_t* flags, int n, uint32_t want, uint32_t sleep_ns, Diag* diag,
                                               uint32_t site) {
  const uint64_t t0 = globaltimer();
  while (true) {
    bool all = true;
    for (int i = 0; i < n; i += 4) {
      uint32_t a, b, c, d;
      asm volatile("ld.relaxed.gpu.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(flags + i) : "memory");
      all = all && a == want && b == want && c == want && d == want;
    }
    if (all) return;
    __nanosleep(sleep_ns);
    if (globaltimer() - t0 > W8_SPIN_TIMEOUT_NS) spin_fail(diag, site, 0, want, 0);
  }
}

// Bounded mbarrier wait (an mbarrier never completes if a TMA faulted or a producer died).
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity, Diag* diag = nullptr, uint32_t site = 0xffff) {
  if (mbar_try_wait(b, parity)) return;
  const uint64_t t0 = globaltimer();
  while (!mbar_try_wait(b, parity)) {
    if (globaltimer() - t0 > W8_SPIN_TIMEOUT_NS) spin_fail(diag, site, 0, parity, 0);
  }
}
// The whole warp waits, with warp-uniform control flow (a vote per round), so code after it stays warp-uniform (the
// warp-uniform tcgen05 issue in umma.cuh).
__device__ __forceinline__ void mbar_wait_warp(uint64_t* b, uint32_t parity, Diag* diag = nullptr, uint32_t site = 0xffff) {
  if (__all_sync(0xffffffffu, mbar_try_wait(b, parity))) return;
  const uint64_t t0 = globaltimer();
  while (!__all_sync(0xffffffffu, mbar_try_wait(b, parity))) {
    if (globaltimer() - t0 > W8_SPIN_TIMEOUT_NS) spin_fail(diag, site, 0, parity, 0);
  }
}

// ---------------------------------------------------------------- tagged mailbox: (value, tag) in one 64-bit store
__device__ __forceinline__ void mbox_put(uint64_t* slot, uint32_t value, uint32_t tag) {
  st_relaxed_gpu_u64(slot, (static_cast<uint64_t>(tag) << 32) | value);
}
__device__ __forceinline__ void mbox_put_f32(uint64_t* slot, float value, uint32_t tag) {
  mbox_put(slot, __float_as_uint(value), tag);
}
__device__ __forceinline__ uint32_t mbox_get(const uint64_t* slot, uint32_t tag, Diag* diag, uint32_t site) {
  uint64_t v = ld_acquire_gpu_u64(slot);
  if (static_cast<uint32_t>(v >> 32) == tag) return static_cast<uint32_t>(v);
  const uint64_t t0 = globaltimer();
  uint32_t n = 0;
  while (true) {
    v = ld_acquire_gpu_u64(slot);
    if (static_cast<uint32_t>(v >> 32) == tag) return static_cast<uint32_t>(v);
    if ((++n & 63) == 0 && globaltimer() - t0 > W8_SPIN_TIMEOUT_NS)
      spin_fail(diag, site, 0, tag, static_cast<uint32_t>(v >> 32));
  }
}

// ---------------------------------------------------------------- dependencies
// The producer tasks [lo, hi) of a stage whose flags a consumer needs.
struct Dep {
  const uint32_t* flags;
  int lo;
  int hi;
  int stride = 1;  // words between consecutive flags (32 = one 128 B line each, for flags many CTAs poll)
};

// ---------------------------------------------------------------- the launch epoch
// One per process (all widths, all layers share it). epoch starts at 1 (0 never matches a fresh flag).
struct Epoch {
  uint32_t epoch;
  uint32_t arrive;
  uint32_t pad[30];
};

__host__ __device__ constexpr uint32_t next_tag(uint32_t t) { return t + 1 == 0 ? 1u : t + 1; }

}  // namespace w8
