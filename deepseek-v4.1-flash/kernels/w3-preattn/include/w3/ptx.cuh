// Minimal SM100/SM103 PTX wrappers (mbarrier, TMA, tcgen05, PDL, conversions) for the W3 pre-attention stages.
// Header-only, no globals. Descriptor bit layouts follow CUTLASS cute/arch/mma_sm100_desc.hpp.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace w3 {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t lane_id() { return threadIdx.x & 31; }
__device__ __forceinline__ uint32_t warp_id() { return threadIdx.x >> 5; }

// ------------------------------------------------------------------ mbarrier
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(ok)
      : "r"(smem_u32(bar)), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  while (!mbar_try_wait(bar, parity)) {
  }
}

// ------------------------------------------------------------------ TMA / bulk copies
__device__ __forceinline__ void prefetch_tmap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(tmap)) : "memory");
}
// 2D tiled TMA load, coordinates in elements (c0 = innermost).
__device__ __forceinline__ void tma_load_2d(void* dst, const void* tmap, uint64_t* bar, int c0, int c1,
                                            uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(policy)
      : "memory");
}
// 1D bulk copy global -> shared (bytes % 16 == 0, both 16 B aligned).
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar,
                                          uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(src)), "r"(bytes), "r"(smem_u32(bar)), "l"(policy)
      : "memory");
}
// Bulk prefetch of [src, src+bytes) into L2 (bytes % 16 == 0); no smem, no barrier.
__device__ __forceinline__ void prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(reinterpret_cast<uint64_t>(src)), "r"(bytes) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// Generic-proxy smem writes -> visible to the async proxy (tcgen05.mma / tcgen05.cp operands).
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

// ------------------------------------------------------------------ PDL
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// ------------------------------------------------------------------ named barriers
__device__ __forceinline__ void named_bar_sync(uint32_t id, uint32_t nthreads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}

// ------------------------------------------------------------------ tcgen05
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, uint32_t ncols) {  // whole warp
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)),
               "r"(ncols)
               : "memory");
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {  // whole warp
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

// UMMA shared-memory matrix descriptor (SM100 version 1).
__device__ __forceinline__ uint64_t smem_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((saddr >> 4) & 0x3FFF);
  d |= static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16;
  d |= static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32;
  d |= static_cast<uint64_t>(1) << 46;  // version = 1
  d |= static_cast<uint64_t>(layout & 7) << 61;
  return d;
}
constexpr uint32_t kLayoutSw128 = 2;
constexpr uint32_t kLayoutNone = 0;

// Instruction descriptor, kind::mxf8f6f4 block-scaled, E4M3 x E4M3 -> F32, UE8M0 scales, both K-major.
__host__ __device__ constexpr uint32_t idesc_mxf8(uint32_t m, uint32_t n, uint32_t sfa_id, uint32_t sfb_id) {
  return (sfb_id << 4) | ((n >> 3) << 17) | (1u << 23) | ((m >> 4) << 24) | (sfa_id << 29);
}
// Instruction descriptor, kind::f16, BF16 x BF16 -> F32, both K-major.
__host__ __device__ constexpr uint32_t idesc_bf16(uint32_t m, uint32_t n) {
  return (1u << 4) /*c f32*/ | (1u << 7) /*a bf16*/ | (1u << 10) /*b bf16*/ | ((n >> 3) << 17) | ((m >> 4) << 24);
}

__device__ __forceinline__ void mma_mxf8(uint32_t tmem_d, uint64_t adesc, uint64_t bdesc, uint32_t idesc,
                                         uint32_t tmem_sfa, uint32_t tmem_sfb, uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}" ::"r"(tmem_d),
      "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate), "r"(tmem_sfa), "r"(tmem_sfb)
      : "memory");
}
__device__ __forceinline__ void mma_bf16(uint32_t tmem_d, uint64_t adesc, uint64_t bdesc, uint32_t idesc,
                                         uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(tmem_d),
      "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate)
      : "memory");
}
// smem (32 rows x 128 bit, core matrices of 8 x 16 B) -> TMEM, replicated to the 4 lane quadrants.
__device__ __forceinline__ void utccp_32x128b_warpx4(uint32_t taddr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void tmem_ld_32x32b_x8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(taddr)
               : "memory");
}
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

// ------------------------------------------------------------------ conversions
// FlashInfer float_to_ue8m0_fast: ceil-exponent UE8M0 of a non-negative float, 0 for <= 0, clamp 254.
__device__ __forceinline__ uint32_t float_to_ue8m0(float v) {
  if (!(v > 0.f)) return 0u;
  uint32_t bits = __float_as_uint(v);
  uint32_t e = (bits >> 23) & 255u, man = bits & 0x7FFFFFu;
  uint32_t bump = (man != 0u) && !((e == 0u) && (man <= 0x400000u));
  uint32_t r = e + bump;
  return r > 254u ? 254u : r;
}
// 2^(127 - e) as float bits, 0 for e == 0 (FlashInfer ue8m0_to_inv_scale_fast).
__device__ __forceinline__ float ue8m0_inv_scale(uint32_t e) {
  if (e == 0u) return 0.f;
  int ne = 254 - static_cast<int>(e);
  ne = ne < 0 ? 0 : ne;
  return __uint_as_float(static_cast<uint32_t>(ne) << 23);
}
__device__ __forceinline__ uint16_t cvt_e4m3x2(float lo, float hi) {  // lo -> low byte
  uint16_t r;
  asm("{\n\t.reg .b16 t;\n\tcvt.rn.satfinite.e4m3x2.f32 t, %2, %1;\n\tmov.b16 %0, t;\n\t}" : "=h"(r) : "f"(lo), "f"(hi));
  return r;
}
__device__ __forceinline__ uint8_t cvt_e2m1x2(float lo, float hi) {  // lo -> low nibble
  uint16_t r;
  asm("{\n\t.reg .b8 t;\n\tcvt.rn.satfinite.e2m1x2.f32 t, %2, %1;\n\tcvt.u16.u8 %0, t;\n\t}" : "=h"(r) : "f"(lo), "f"(hi));
  return static_cast<uint8_t>(r);
}
__device__ __forceinline__ float bf16_round(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// Byte offset of (row, k) inside a K-major SWIZZLE_128B tile whose rows are 128 B (one swizzle atom along K).
__host__ __device__ __forceinline__ uint32_t sw128_offset(uint32_t row, uint32_t kbyte) {
  return (row >> 3) * 1024u + (row & 7u) * 128u + ((((kbyte >> 4) ^ (row & 7u)) & 7u) << 4) + (kbyte & 15u);
}

}  // namespace w3
