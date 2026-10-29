// W8 persistent-kernel framework: PTX primitives for sm_103a (memory ordering, mbarriers, bulk/TMA copies, timers).
#pragma once
#include <cstdint>
#include <cuda.h>

namespace w8 {

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint32_t warp_id() { return threadIdx.x >> 5; }
__device__ __forceinline__ uint32_t lane_id() { return threadIdx.x & 31; }

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// ---------------------------------------------------------------- named barriers
__device__ __forceinline__ void bar_sync(uint32_t id, uint32_t nthreads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}

// ---------------------------------------------------------------- PDL
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// ---------------------------------------------------------------- fences
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_acq_rel_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
// generic-proxy writes to SMEM -> visible to the async proxy (tcgen05.mma / tcgen05.cp / TMA store)
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// generic-proxy writes to global (by any CTA, acquired) -> visible to this thread's async-proxy reads (TMA / bulk)
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
__device__ __forceinline__ void fence_mbarrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }

// ---------------------------------------------------------------- global loads/stores with explicit scope
__device__ __forceinline__ uint32_t ld_acquire_gpu(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint64_t ld_acquire_gpu_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_gpu(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_relaxed_gpu_u64(uint64_t* p, uint64_t v) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint32_t atom_add_acq_rel_gpu(uint32_t* p, uint32_t v) {
  uint32_t old;
  asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
// L2-coherent (L1-bypassing) vector load for activations written by other CTAs in this launch
__device__ __forceinline__ uint4 ld_cg_v4(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ld_cg_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ float ld_cg_f32(const void* p) { return __uint_as_float(ld_cg_u32(p)); }
// four independent L2 loads issued back to back (one round trip, not four)
__device__ __forceinline__ void ld_cg_u32x4(const void* p0, const void* p1, const void* p2, const void* p3, uint32_t (&v)[4]) {
  asm volatile(
      "ld.global.cg.u32 %0, [%4];\n ld.global.cg.u32 %1, [%5];\n ld.global.cg.u32 %2, [%6];\n ld.global.cg.u32 %3, [%7];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
      : "l"(p0), "l"(p1), "l"(p2), "l"(p3));
}

// ---------------------------------------------------------------- mbarriers (CTA-local)
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cnt(uint64_t* b, uint32_t count) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, 0x989680;\n selp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok)
      : "r"(smem_u32(b)), "r"(parity)
      : "memory");
  return ok != 0;
}

// ---------------------------------------------------------------- bulk / TMA copies (global -> this CTA's SMEM)
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void bulk_g2s_hint(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int32_t c0, int32_t c1, uint64_t* bar,
                                            uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], "
      "%5;" ::"r"(smem_u32(dst)),
      "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d(void* dst, const CUtensorMap* map, int32_t c0, int32_t c1, int32_t c2,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d_hint(void* dst, const CUtensorMap* map, int32_t c0, int32_t c1, int32_t c2,
                                                 uint64_t* bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], "
      "[%2], %6;" ::"r"(smem_u32(dst)),
      "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_prefetch_3d(const CUtensorMap* map, int32_t c0, int32_t c1, int32_t c2) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global [%0, {%1, %2, %3}];" ::"l"(map), "r"(c0), "r"(c1), "r"(c2)
               : "memory");
}
__device__ __forceinline__ void tma_load_4d(void* dst, const CUtensorMap* map, int32_t c0, int32_t c1, int32_t c2,
                                            int32_t c3, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
      : "memory");
}
// L2 prefetch of a contiguous global range (bytes: a multiple of 16)
__device__ __forceinline__ void bulk_prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
// L2 prefetch of one 2D TMA box (a hint: lands in L2, writes nothing)
__device__ __forceinline__ void tma_prefetch_2d(const CUtensorMap* map, int32_t c0, int32_t c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];" ::"l"(map), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void prefetch_tmap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t policy_evict_normal() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void prefetch_l2_bulk(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

// ---------------------------------------------------------------- numerics helpers
__device__ __forceinline__ float bf16_bits_to_f32(uint32_t b16) { return __uint_as_float(b16 << 16); }
__device__ __forceinline__ uint32_t f32_to_bf16_bits(float f) {  // RNE, same as __float2bfloat16_rn
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
  return r;
}
__device__ __forceinline__ float bf16_round(float f) { return bf16_bits_to_f32(f32_to_bf16_bits(f)); }

}  // namespace w8
