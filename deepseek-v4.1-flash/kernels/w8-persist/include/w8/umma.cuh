// W8 persistent-kernel framework: tcgen05 (UMMA) helpers for block-scaled MXFP8 on sm_103a.
// Descriptor bit layouts follow cute/arch/mma_sm100_desc.hpp (SmemDescriptor, InstrDescriptorBlockScaled).
#pragma once
#include "ptx.cuh"

namespace w8::umma {

// ---------------------------------------------------------------- TMEM allocation (one warp)
__device__ __forceinline__ void tmem_alloc(uint32_t* smem_dst, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(smem_dst)), "r"(ncols)
               : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void fence_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

// ---------------------------------------------------------------- SMEM matrix descriptors
// K-major operand tile with 128-byte rows (K = 128 FP8 values), TMA SWIZZLE_128B, 8-row atoms of 1024 B.
__device__ __forceinline__ uint64_t sdesc_k128_sw128(const void* tile) {
  const uint32_t a = smem_u32(tile);
  uint64_t d = 0;
  d |= static_cast<uint64_t>((a >> 4) & 0x3fff);        // start address
  d |= static_cast<uint64_t>(0) << 16;                   // LBO (unused for this layout)
  d |= static_cast<uint64_t>((1024 >> 4) & 0x3fff) << 32;  // SBO: 8 rows x 128 B
  d |= static_cast<uint64_t>(1) << 46;                   // version (sm100)
  d |= static_cast<uint64_t>(2) << 61;                   // SWIZZLE_128B
  return d;
}
// Scale-factor source for tcgen05.cp 32x128b.warpx4: 32 rows x 16 B, no swizzle (SBO = 8 rows x 16 B).
__device__ __forceinline__ uint64_t sdesc_sf(const void* p) {
  const uint32_t a = smem_u32(p);
  uint64_t d = 0;
  d |= static_cast<uint64_t>((a >> 4) & 0x3fff);
  d |= static_cast<uint64_t>((128 >> 4) & 0x3fff) << 32;
  d |= static_cast<uint64_t>(1) << 46;
  return d;
}

// ---------------------------------------------------------------- instruction descriptor
// kind::mxf8f6f4.block_scale, A = B = E4M3, K-major both, scales UE8M0, fp32 accumulate (implicit for this kind).
__host__ __device__ constexpr uint32_t idesc_mxf8(uint32_t M, uint32_t N, uint32_t sfa_id = 0, uint32_t sfb_id = 0) {
  return (sfb_id & 3u) << 4         // b_sf_id
         | (0u << 7)                 // a_format E4M3
         | (0u << 10)                // b_format E4M3
         | (0u << 15) | (0u << 16)   // K-major A, B
         | ((N >> 3) & 63u) << 17    // n_dim
         | (1u << 23)                // scale format E8M0
         | ((M >> 4) & 31u) << 24    // m_dim
         | (sfa_id & 3u) << 29;      // a_sf_id
}

// D[tmem_d] (+)= A[desc_a] x B[desc_b], scales from TMEM. Issued by one thread.
__device__ __forceinline__ void mma_mxf8(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
                                         uint32_t tmem_sfa, uint32_t tmem_sfb, uint32_t accumulate) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n}" ::"r"(tmem_d),
      "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate), "r"(tmem_sfa), "r"(tmem_sfb)
      : "memory");
}
// Arrive on an mbarrier once all prior tcgen05 ops of this thread complete.
__device__ __forceinline__ void commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar))
               : "memory");
}
// SMEM (32 rows x 16 B, replicated to the 4 lane quadrants) -> TMEM 4 columns. One thread.
__device__ __forceinline__ void cp_sf_32x128b_x4(uint32_t tmem_col_addr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(tmem_col_addr), "l"(sdesc) : "memory");
}

// ---------------------------------------------------------------- warp-uniform issue (the whole warp calls; one lane
// is elected inside the asm). Issuing from a divergent single-thread branch makes the compiler wrap every tcgen05 op in
// an elect loop with R2UR.BROADCAST of each operand: measured (tests/mma_place_probe.cu) 88-127 clk per dependent
// MMA vs 48 clk issued this way, i.e. the chain was issue-bound. Operands must be warp-uniform.
__device__ __forceinline__ void mma_mxf8_w(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
                                           uint32_t tmem_sfa, uint32_t tmem_sfb, uint32_t accumulate) {
  asm volatile(
      "{\n .reg .pred p, e;\n setp.ne.b32 p, %4, 0;\n elect.sync _|e, 0xffffffff;\n"
      " @e tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n}" ::"r"(tmem_d),
      "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate), "r"(tmem_sfa), "r"(tmem_sfb)
      : "memory");
}
// The same with each SMEM descriptor given as its low word (address >> 4 plus offsets) and a shared constant high word.
__device__ __forceinline__ void mma_mxf8_w2(uint32_t tmem_d, uint32_t a_lo, uint32_t b_lo, uint32_t hi, uint32_t idesc,
                                            uint32_t tmem_sfa, uint32_t tmem_sfb, uint32_t accumulate) {
  asm volatile(
      "{\n .reg .pred p, e;\n .reg .b64 da, db;\n setp.ne.b32 p, %4, 0;\n mov.b64 da, {%1, %7};\n mov.b64 db, {%2, %7};\n"
      " elect.sync _|e, 0xffffffff;\n"
      " @e tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], da, db, %3, [%5], [%6], p;\n}" ::"r"(tmem_d),
      "r"(a_lo), "r"(b_lo), "r"(idesc), "r"(accumulate), "r"(tmem_sfa), "r"(tmem_sfb), "r"(hi)
      : "memory");
}
__device__ __forceinline__ void cp_sf_32x128b_x4_w(uint32_t tmem_col_addr, uint64_t sdesc) {
  asm volatile("{\n .reg .pred e;\n elect.sync _|e, 0xffffffff;\n @e tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n}"
               ::"r"(tmem_col_addr), "l"(sdesc) : "memory");
}
__device__ __forceinline__ void commit_w(uint64_t* bar) {
  asm volatile("{\n .reg .pred e;\n elect.sync _|e, 0xffffffff;\n"
               " @e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n}" ::"r"(smem_u32(bar))
               : "memory");
}

// ---------------------------------------------------------------- TMEM -> registers (whole warp, its 32-lane quadrant)
template <int N>
__device__ __forceinline__ void ld_32x32b(uint32_t taddr, uint32_t (&r)[N]);

template <>
__device__ __forceinline__ void ld_32x32b<8>(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(taddr));
}
template <>
__device__ __forceinline__ void ld_32x32b<16>(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr));
}
template <>
__device__ __forceinline__ void ld_32x32b<32>(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
}
__device__ __forceinline__ void ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

}  // namespace w8::umma
