// Shared tcgen05 block-scaled MXFP8 GEMM pieces for the W3 stages.
//
// Math contract (bitwise parity with FlashInfer's MXFP8 GEMMs, measured in tests/test_probe_mma.py):
//   one tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale per 32-element K block, M = 128 weight rows (A, K-major,
//   SW128 via TMA), N = 8/16/24/32 tokens (B, K-major SW128 in smem), UE8M0 scales in TMEM via
//   tcgen05.cp.32x128b.warpx4 straight from the F8_128x4 swizzled scale tensors, sequential accumulation over the
//   task's K range in fp32 TMEM.
#pragma once
#include "w3/fw.cuh"

namespace w3 {

constexpr uint32_t kChunkBytes = 16384;          // 128 rows x 128 B of A
constexpr uint32_t kSlotBytes = 17408;           // A chunk + its 512 B SFA atom, 1024 B aligned
constexpr uint32_t kSfaOffset = kChunkBytes;     // SFA atom inside a slot
// TMEM: two accumulators of up to 48 columns (double buffer / split partials), then 4 rotating SFA+SFB column sets
// (consecutive chunks never rewrite the scale columns an in-flight MMA still reads).
constexpr uint32_t kTmemAccStride = 48;
__host__ __device__ constexpr uint32_t tmem_acc(int b) { return static_cast<uint32_t>(b) * kTmemAccStride; }
constexpr uint32_t kTmemSfBase = 96;
constexpr uint32_t kTmemCols = 128;

__host__ __device__ constexpr uint32_t b_chunk_bytes(int n_mma) { return static_cast<uint32_t>((n_mma + 7) / 8) * 1024u; }

// Producer: weight rows [rt*128, +128), K chunks [kc0, kc1) (128 K each), with their SFA atoms.
__device__ __forceinline__ void produce_a_chunks(const CUtensorMap* tm, const uint8_t* wsf, int k_groups, int rt, int kc0,
                                                 int kc1, Ring& ring, uint64_t pol) {
  for (int c = kc0; c < kc1; ++c) {
    uint8_t* slot = ring.acquire(kChunkBytes + 512);
    tma_load_2d(slot, tm, ring.bar(), c * 128, rt * 128, pol);
    bulk_load(slot + kSfaOffset, wsf + (static_cast<size_t>(rt) * k_groups + c) * 512, 512, ring.bar(), pol);
    ring.advance();
  }
}

// L2 prefetch of a weight row tile's K range: rows [rt*128, +128) x bytes [kc0*128, kc1*128) of a row-major [n, K]
// matrix (one bulk prefetch per row; the full-K case is one contiguous block).
__device__ __forceinline__ void prefetch_a_rows_l2(const uint8_t* w, int K, int rt, int kc0, int kc1) {
  if (kc0 == 0 && kc1 * 128 == K) {
    const uint8_t* base = w + static_cast<size_t>(rt) * 128 * K;
    const uint32_t total = 128u * K;
    for (uint32_t off = 0; off < total; off += 32768u) prefetch_l2(base + off, min(32768u, total - off));
    return;
  }
  for (int r = 0; r < 128; ++r) prefetch_l2(w + (static_cast<size_t>(rt) * 128 + r) * K + kc0 * 128, (kc1 - kc0) * 128);
}

// Consumer MMA issue (one thread): chunks [kc0, kc1) popped from the ring, B chunk c at sB + (c - b0) * bstride.
template <int N>
__device__ __forceinline__ void issue_mmas(Ring& ring, const uint8_t* sB, const uint8_t* sSFB, int kc0, int kc1, int b0,
                                           uint32_t tmem, uint32_t acc_col, uint64_t* done, int trace_id = -1) {
  constexpr uint32_t bstride = b_chunk_bytes(N);
  for (int c = kc0; c < kc1; ++c) {
    if (trace_id >= 0 && c - kc0 < 4) trace_any(trace_id, 8 + 2 * (c - kc0));
    uint8_t* slot = ring.wait_full();
    if (trace_id >= 0 && c - kc0 < 4) trace_any(trace_id, 9 + 2 * (c - kc0));
    tc_fence_after();
    const uint32_t sf = kTmemSfBase + (ring.idx & 3) * 8;
    utccp_32x128b_warpx4(tmem + sf, smem_desc(smem_u32(slot + kSfaOffset), 0, 128, kLayoutNone));
    utccp_32x128b_warpx4(tmem + sf + 4, smem_desc(smem_u32(sSFB + (c - b0) * 512), 0, 128, kLayoutNone));
    const uint32_t a0 = smem_u32(slot), bb = smem_u32(sB + (c - b0) * bstride);
#pragma unroll
    for (int k = 0; k < 4; ++k)
      mma_mxf8(tmem + acc_col, smem_desc(a0 + k * 32, 0, 1024, kLayoutSw128), smem_desc(bb + k * 32, 0, 1024, kLayoutSw128),
               idesc_mxf8(128, N, k, k), tmem + sf, tmem + sf + 4, (c != kc0) || (k != 0));
    tc_commit(ring.empty_bar());
    ring.advance();
  }
  tc_commit(done);
}

// Load an fp8 activation [T, K] (row-major, row stride ld bytes) K chunks [kc0, kc1) into SW128 B layout + copy its
// F8_128x4 scale atoms (rows padded to 128, k-groups of 4 x 32), all 256 compute threads.
template <int N>
__device__ __forceinline__ void load_b_fp8(const uint8_t* x, int ld, const uint8_t* xsf, int T, int kc0, int kc1, uint8_t* sB,
                                           uint8_t* sSFB) {
  constexpr uint32_t bstride = b_chunk_bytes(N);
  const int nkc = kc1 - kc0;
  const int items = nkc * N * 8;
  for (int it = threadIdx.x; it < items; it += kComputeThreads) {
    const int j = it & 7, n = (it >> 3) % N, c = it / (N * 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (n < T) v = *reinterpret_cast<const uint4*>(x + static_cast<size_t>(n) * ld + (kc0 + c) * 128 + j * 16);
    *reinterpret_cast<uint4*>(sB + c * bstride + sw128_offset(n, j * 16)) = v;
  }
  for (int it = threadIdx.x; it < nkc * 32; it += kComputeThreads) {
    const int c = it / 32, q = it % 32;
    *reinterpret_cast<uint4*>(sSFB + c * 512 + q * 16) =
        *reinterpret_cast<const uint4*>(xsf + static_cast<size_t>(kc0 + c) * 512 + q * 16);
  }
}

// Epilogue read: this thread's accumulator row (lane quadrant warp%4) columns [c0, c0+8).
__device__ __forceinline__ void acc_row8(uint32_t tmem, uint32_t col, float (&v)[8]) {
  uint32_t r[8];
  tmem_ld_32x32b_x8(tmem + (static_cast<uint32_t>((warp_id() & 3) * 32) << 16) + col, r);
  tmem_ld_wait();
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __uint_as_float(r[i]);
}

}  // namespace w3
