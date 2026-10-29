// Minimal stand-in for the W8 persistent-kernel framework API (CONTRACT.md v1), so W1's stages compile and run
// standalone before the real `include/w8/*.cuh` exists: compute_sync, Dep, Ring, the epoch-tagged task flags and a
// two-stage standalone runner. Swap for the real headers when they land; the stages only use the contract surface.
#pragma once
#include <cstdint>

namespace w8 {

constexpr int kComputeThreads = 256;
constexpr int kProducerWarp = 8;
constexpr int kBlockThreads = 288;

__device__ __forceinline__ void compute_sync() { asm volatile("bar.sync 1, 256;" ::: "memory"); }

struct Ring {};  // W1's stages stream no weights

// Flags: one uint32 per (stage, task); ready iff == the launch tag. Never cleared.
__device__ __forceinline__ uint32_t ld_acquire(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_relaxed(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// Bounded spin (0.5 s), then trap: a wedged GPU is never acceptable.
__device__ __forceinline__ void spin_until_eq(const uint32_t* p, uint32_t tag) {
    if (ld_acquire(p) == tag) return;
    const uint64_t t0 = globaltimer();
    while (ld_acquire(p) != tag) {
        if (globaltimer() - t0 > 500000000ull) {
#ifdef W1_DEBUG_TIMEOUT
            printf("w8 spin timeout cta %d tid %d flag %p want %u have %u\n", blockIdx.x, threadIdx.x, p, tag, ld_acquire(p));
#endif
            __trap();
        }
        __nanosleep(32);
    }
}

struct Dep {
    const uint32_t* flags;
    int lo, hi;  // tasks [lo, hi) of the producer stage
};

// Standalone sync: wait_extern = griddepcontrol.wait (once); wait(Dep) polls the producer stage's task flags.
struct StandaloneSync {
    uint32_t tag = 0;
    bool waited = false;
    __device__ __forceinline__ void wait_extern() {
        if (!waited) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
            waited = true;
        }
    }
    __device__ __forceinline__ void wait(Dep d, int stride = 1) {
        wait_extern();
        const int lane = threadIdx.x & 31;
        // warp 0 polls; everyone else waits at the barrier the caller places after this
        if (threadIdx.x < 32)
            for (int i = d.lo + lane * stride; i < d.hi; i += 32 * stride) spin_until_eq(d.flags + i, tag);
    }
};

}  // namespace w8
