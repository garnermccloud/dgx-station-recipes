// SM100 (sm_103a) PTX helpers for the W1 decode attention: mbarriers, tcgen05 MMA / TMEM access, UMMA smem
// descriptors, conversions. Header-only, no globals.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace w1 {

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---------------------------------------------------------------- mbarrier
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ uint32_t mbar_try_wait(uint64_t* bar, uint32_t phase) {
    uint32_t ok;
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
        : "=r"(ok)
        : "r"(smem_u32(bar)), "r"(phase)
        : "memory");
    return ok;
}
#ifdef W1_DEBUG_TIMEOUT
__device__ __forceinline__ void mbar_wait_dbg(uint64_t* bar, uint32_t phase, int line) {
    uint64_t t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    while (!mbar_try_wait(bar, phase)) {
        uint64_t t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > 2000000000ull) {
            printf("w1 mbar timeout line %d cta %d tid %d phase %u\n", line, blockIdx.x, threadIdx.x, phase);
            __trap();
        }
    }
}
#define mbar_wait(b, ph) mbar_wait_dbg((b), (ph), __LINE__)
#else
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n .reg .pred p;\n W1_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W1_WAIT_%=;\n}" ::"r"(
            smem_u32(bar)),
        "r"(phase)
        : "memory");
}
#endif
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}

// ---------------------------------------------------------------- proxies / fences
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }

// ---------------------------------------------------------------- TMA bulk copy (1D)
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
                 "l"(src), "r"(bytes), "r"(smem_u32(bar))
                 : "memory");
}

// ---------------------------------------------------------------- cp.async (16 B, global -> smem, no registers)
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
}
// named-barrier OR-reduction over `count` threads
__device__ __forceinline__ bool bar_red_or(int id, int count, bool v) {
    uint32_t r;
    asm volatile("{\n .reg .pred p, q;\n setp.ne.u32 p, %1, 0;\n bar.red.or.pred q, %2, %3, p;\n selp.u32 %0, 1, 0, q;\n}"
                 : "=r"(r)
                 : "r"((uint32_t)v), "r"(id), "r"(count)
                 : "memory");
    return r != 0;
}

// ---------------------------------------------------------------- tcgen05
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

// D[tmem] (+)= A[smem desc] x B[smem desc]^T, kind::f16 (bf16 in, fp32 accumulate), cta_group::1.
__device__ __forceinline__ void tc_mma_ss(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
                                          uint32_t accumulate) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
        " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(d_tmem),
        "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate)
        : "memory");
}
// Arrive on `bar` once all previously issued tcgen05.mma of this thread have completed.
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar))
                 : "memory");
}
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// 32 lanes x 32 bit, N consecutive columns: thread t of the warp gets lane (taddr.lane + t), columns col..col+N-1.
template <int N>
__device__ __forceinline__ void tc_ld_32x32b(uint32_t taddr, uint32_t (&r)[N]);
template <>
__device__ __forceinline__ void tc_ld_32x32b<8>(uint32_t taddr, uint32_t (&r)[8]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "r"(taddr));
}
template <>
__device__ __forceinline__ void tc_ld_32x32b<16>(uint32_t taddr, uint32_t (&r)[16]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
          "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
        : "r"(taddr));
}
template <int N>
__device__ __forceinline__ void tc_st_32x32b(uint32_t taddr, const uint32_t (&r)[N]);
template <>
__device__ __forceinline__ void tc_st_32x32b<8>(uint32_t taddr, const uint32_t (&r)[8]) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(taddr), "r"(r[0]),
                 "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7])
                 : "memory");
}
template <>
__device__ __forceinline__ void tc_st_32x32b<16>(uint32_t taddr, const uint32_t (&r)[16]) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(
            taddr),
        "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
        "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
        : "memory");
}

// TMEM allocation (standalone kernels only; the persistent framework owns it there).
__device__ __forceinline__ void tc_alloc(uint32_t* smem_dst, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(smem_dst)),
                 "r"(ncols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_dealloc(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}

// ---------------------------------------------------------------- UMMA descriptors
// Shared-memory matrix descriptor, SWIZZLE_128B, version 1 (Blackwell). Byte offsets.
__device__ __forceinline__ uint64_t umma_desc_sw128(uint32_t smem_addr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
    uint64_t d = 0;
    d |= (uint64_t)((smem_addr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;  // version
    d |= (uint64_t)2 << 61;  // SWIZZLE_128B
    return d;
}
// Instruction descriptor kind::f16: bf16 x bf16 -> fp32. a_mn / b_mn: operand is MN-major.
__host__ __device__ constexpr uint32_t umma_idesc_bf16(int M, int N, bool a_mn, bool b_mn) {
    return (1u << 4) | (1u << 7) | (1u << 10) | ((a_mn ? 1u : 0u) << 15) | ((b_mn ? 1u : 0u) << 16) |
           ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

// Byte offset of element (row, k) of a SW128 K-major bf16 tile whose atom-columns (64 elements) are `col_stride`
// bytes apart (rows contiguous at 128 B; the 16 B chunk index is XORed with row % 8).
__device__ __forceinline__ uint32_t sw128_off(uint32_t row, uint32_t k, uint32_t col_stride) {
    return (k >> 6) * col_stride + row * 128u + ((((k & 63u) >> 3) ^ (row & 7u)) << 4) + ((k & 7u) << 1);
}

// ---------------------------------------------------------------- conversions
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ uint32_t f16x2_to_bf16x2(uint32_t h2) {
    __half2 h = *reinterpret_cast<__half2*>(&h2);
    float2 f = __half22float2(h);
    return pack_bf16x2(f.x, f.y);
}
// 4 e4m3 in a word (lowest byte = first element) -> 2 x f16x2, exact (e4m3 is a subset of f16)
__device__ __forceinline__ void e4m3x4_to_f16x2x2(uint32_t w, uint32_t& h0, uint32_t& h1) {
    asm("{\n .reg .b16 a, b;\n mov.b32 {a, b}, %2;\n cvt.rn.f16x2.e4m3x2 %0, a;\n cvt.rn.f16x2.e4m3x2 %1, b;\n}"
        : "=r"(h0), "=r"(h1)
        : "r"(w));
}
// 8 e2m1 in a word (lowest nibble = first element) -> 4 x f16x2, exact
__device__ __forceinline__ void e2m1x8_to_f16x2x4(uint32_t w, uint32_t (&h)[4]) {
    asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %4;\n"
        " cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        " cvt.rn.f16x2.e2m1x2 %2, b2;\n cvt.rn.f16x2.e2m1x2 %3, b3;\n}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(w));
}

// CUDA 13.2+ (PTX ISA 9.2) converts e4m3x2 / e2m1x2 straight to bf16x2 in one instruction; older toolchains go
// through f16 -> f32. Both are exact for every finite code.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define W1_NATIVE_CVT 1
#else
#define W1_NATIVE_CVT 0
#endif

// 4 e4m3 (lowest byte first) -> 2 x bf16x2
__device__ __forceinline__ void e4m3x4_to_bf16x2x2(uint32_t w, uint32_t& b0, uint32_t& b1) {
#if W1_NATIVE_CVT
    asm("{\n .reg .b16 a, b;\n mov.b32 {a, b}, %2;\n cvt.rn.bf16x2.e4m3x2 %0, a;\n cvt.rn.bf16x2.e4m3x2 %1, b;\n}"
        : "=r"(b0), "=r"(b1)
        : "r"(w));
#else
    uint32_t h0, h1;
    e4m3x4_to_f16x2x2(w, h0, h1);
    b0 = f16x2_to_bf16x2(h0);
    b1 = f16x2_to_bf16x2(h1);
#endif
}
// 8 e2m1 (lowest nibble first) -> 4 x bf16x2
__device__ __forceinline__ void e2m1x8_to_bf16x2x4(uint32_t w, uint32_t (&b)[4]) {
#if W1_NATIVE_CVT
    asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %4;\n"
        " cvt.rn.bf16x2.e2m1x2 %0, b0;\n cvt.rn.bf16x2.e2m1x2 %1, b1;\n"
        " cvt.rn.bf16x2.e2m1x2 %2, b2;\n cvt.rn.bf16x2.e2m1x2 %3, b3;\n}"
        : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
        : "r"(w));
#else
    uint32_t h[4];
    e2m1x8_to_f16x2x4(w, h);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = f16x2_to_bf16x2(h[i]);
#endif
}

// ue8m0 -> bf16x2 (s, s): 2^(s-127)
__device__ __forceinline__ uint32_t ue8m0_to_bf16x2(uint32_t s) {
    uint32_t r;
    uint16_t packed = (uint16_t)(s * 0x0101u);
    asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(r) : "h"(packed));
    return r;
}
__device__ __forceinline__ uint32_t hmul2_bf16(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ float ex2_approx(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
// two floats -> e4m3x2 (low byte = first), round-to-nearest, satfinite
__device__ __forceinline__ uint16_t f32x2_to_e4m3x2(float lo, float hi) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}


// ---------------------------------------------------------------- thread-block clusters (DSMEM)
__device__ __forceinline__ uint32_t cluster_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
// shared::cluster address of `local` (this CTA's smem) in CTA `rank` of the cluster
__device__ __forceinline__ uint32_t mapa(const void* local, uint32_t rank) {
    uint32_t a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(smem_u32(local)), "r"(rank));
    return a;
}
__device__ __forceinline__ void cluster_sync_all() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// 16 B into a remote CTA's smem, completing `bytes` on its mbarrier (both shared::cluster addresses)
__device__ __forceinline__ void st_async_v4(uint32_t raddr, uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                                            uint32_t rbar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(
                     raddr),
                 "r"(a), "r"(b), "r"(c), "r"(d), "r"(rbar)
                 : "memory");
}
__device__ __forceinline__ void st_async_b32(uint32_t raddr, uint32_t a, uint32_t rbar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(raddr), "r"(a),
                 "r"(rbar)
                 : "memory");
}
// bulk copy of this CTA's smem into a remote CTA's smem, completing on its mbarrier
__device__ __forceinline__ void bulk_s2r(uint32_t rdst, const void* src, uint32_t bytes, uint32_t rbar) {
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                     rdst),
                 "r"(smem_u32(src)), "r"(bytes), "r"(rbar)
                 : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote(uint32_t rbar) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(rbar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_local(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
// parity wait that also acquires at cluster scope (data written by the partner CTA)
#ifdef W1_DEBUG_TIMEOUT
__device__ __forceinline__ void mbar_wait_cluster_dbg(uint64_t* bar, uint32_t phase, int line) {
    uint64_t t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    while (true) {
        uint32_t ok;
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
            " selp.u32 %0, 1, 0, p;\n}"
            : "=r"(ok)
            : "r"(smem_u32(bar)), "r"(phase)
            : "memory");
        if (ok) return;
        uint64_t t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > 2000000000ull) {
            printf("w1 cluster mbar timeout line %d cta %d tid %d phase %u\n", line, blockIdx.x, threadIdx.x, phase);
            __trap();
        }
    }
}
#define mbar_wait_cluster(b, ph) mbar_wait_cluster_dbg((b), (ph), __LINE__)
#else
__device__ __forceinline__ void mbar_wait_cluster(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n .reg .pred p;\n W1_WAITC_%=:\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
        " @!p bra W1_WAITC_%=;\n}" ::"r"(smem_u32(bar)),
        "r"(phase)
        : "memory");
}
#endif
// commit this thread's MMAs to the mbarrier at the same offset in every CTA of `mask`
__device__ __forceinline__ void tc_commit_mc(uint64_t* bar, uint16_t mask) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
            smem_u32(bar)),
        "h"(mask)
        : "memory");
}

// ---------------------------------------------------------------- PDL
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

}  // namespace w1
