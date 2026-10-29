// W8: epoch-tagged hand-off words. A 64-bit word {payload (low 32 bits), launch tag (high 32 bits)} is written with a
// relaxed gpu-scope store and read with a relaxed gpu-scope load; aligned 64-bit accesses are single-copy atomic, so a
// reader that sees its launch's tag also sees that launch's payload. The consumer polls the data itself: a hand-off
// costs one L2 round trip, with no flag, no release fence and no separate data load.
// Vector forms (.v2.b64) are two such words; each element is atomic on its own and carries its own tag.
#pragma once
#include "sync.cuh"

namespace w8 {

__device__ __forceinline__ uint64_t tw(uint32_t payload, uint32_t tag) {
  return (static_cast<uint64_t>(tag) << 32) | payload;
}
__device__ __forceinline__ uint32_t tw_payload(uint64_t w) { return static_cast<uint32_t>(w); }
__device__ __forceinline__ bool tw_ok(uint64_t w, uint32_t tag) { return static_cast<uint32_t>(w >> 32) == tag; }

__device__ __forceinline__ void st_tw2(uint64_t* p, uint64_t a, uint64_t b) {
  asm volatile("st.relaxed.gpu.global.v2.b64 [%0], {%1, %2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ void st_tw(uint64_t* p, uint64_t a) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(a) : "memory");
}
__device__ __forceinline__ void ld_tw2(const uint64_t* p, uint64_t& a, uint64_t& b) {
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}

__device__ __forceinline__ uint64_t ld_tw(const uint64_t* p) {
  uint64_t a;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(a) : "l"(p) : "memory");
  return a;
}

// Warp-collective wait for one sentinel word per producer (lane k polls sentinel(k), in parallel): consumers that
// arrive early wait here at the cost of one poller per producer, then fetch everything with one poll_tw2 round.
// The sentinel is any word the producer writes; words of the same producer that land later are re-polled by poll_tw2.
template <class Sent>
__device__ __forceinline__ void wait_tw(Sent&& sentinel, int nsent, uint32_t tag, Diag* diag, uint32_t site) {
  if (tag != 0)
    for (int k = static_cast<int>(lane_id()); k < nsent; k += 32) {
      const uint64_t* w = sentinel(k);
      if (tw_ok(ld_tw(w), tag)) continue;
      const uint64_t t0 = globaltimer();
      uint32_t rounds = 0;
      uint64_t a;
      do {
        __nanosleep(W8_SPIN_BACKOFF_NS);
        a = ld_tw(w);
        if ((++rounds & 15) == 0 && !tw_ok(a, tag) && globaltimer() - t0 > W8_SPIN_TIMEOUT_NS)
          spin_fail(diag, site, static_cast<uint32_t>(k), tag, static_cast<uint32_t>(a >> 32));
      } while (!tw_ok(a, tag));
    }
  __syncwarp();
}

// Warp-collective (every lane calls, n may differ per lane): poll n (<= N <= 32) word pairs addr(0 .. n) until every
// word carries `tag`. Each round issues all of a lane's outstanding loads before checking any, so a ready set costs
// one round trip; while words are missing only one lane per warp polls (one of its missing words, with backoff), so
// early consumers do not flood L2. Bounded: traps after W8_SPIN_TIMEOUT_NS. tag == 0 accepts anything (stand-alone
// tests that preload the hand-off words).
template <int N, class Addr>
__device__ __forceinline__ void poll_tw2(Addr&& addr, int n, uint64_t (&v)[N][2], uint32_t tag, Diag* diag,
                                         uint32_t site) {
  static_assert(N >= 1 && N <= 32, "pending mask");
  uint32_t pend = n >= 32 ? 0xffffffffu : ((1u << n) - 1u);
  uint64_t t0 = 0;
  while (true) {
#pragma unroll
    for (int i = 0; i < N; ++i)
      if (pend & (1u << i)) ld_tw2(addr(i), v[i][0], v[i][1]);
#pragma unroll
    for (int i = 0; i < N; ++i)
      if ((pend & (1u << i)) && (tag == 0 || (tw_ok(v[i][0], tag) && tw_ok(v[i][1], tag)))) pend &= ~(1u << i);
    const uint32_t waiting = __ballot_sync(0xffffffffu, pend != 0);
    if (waiting == 0) return;
    if (t0 == 0) t0 = globaltimer();
    if (lane_id() == static_cast<uint32_t>(__ffs(waiting) - 1)) {
      const uint64_t* w = nullptr;
#pragma unroll
      for (int i = 0; i < N; ++i)
        if (w == nullptr && (pend & (1u << i))) w = addr(i);
      uint32_t rounds = 0;
      while (true) {
        __nanosleep(W8_SPIN_BACKOFF_NS);
        uint64_t a, b;
        ld_tw2(w, a, b);
        if (tw_ok(a, tag) && tw_ok(b, tag)) break;
        if ((++rounds & 15) == 0 && globaltimer() - t0 > W8_SPIN_TIMEOUT_NS)
          spin_fail(diag, site, 0, tag, static_cast<uint32_t>((tw_ok(a, tag) ? b : a) >> 32));
      }
    }
    __syncwarp();
  }
}

}  // namespace w8
