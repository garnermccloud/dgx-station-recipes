// W7 L2 weight-prefetch stage, embeddable in any kernel (W1 attention, W3 pre-attention, W4 boundary, W8 persistent).
//
// Call from ONE thread per CTA, early in the kernel (before griddepcontrol.wait: it only reads an immutable table and
// touches immutable weights):
//     if (threadIdx.x == 0) w7::prefetch_stage(table, nranges, blockIdx.x, gridDim.x);
// Each CTA issues cp.async.bulk.prefetch.L2 for its share of the 16 KiB chunks of the (address, bytes) ranges, in table
// order (earliest-consumed first). Cost per CTA: ceil(total / 16 KiB / ncta) bulk-prefetch instructions; the SM's TMA
// issues ~16 KiB per ~62 ns, so 82 MB over 152 CTAs is ~34 instructions (~2 us of TMA time) per SM. The hint lands in
// L2 (the address's home-die L2); it never writes memory, so it cannot change any result.
#pragma once
#include <stdint.h>

namespace w7 {

constexpr int64_t kStageChunk = 16384;

__device__ __forceinline__ void prefetch_stage(const int64_t* __restrict__ table, int nranges, int64_t worker,
                                               int64_t workers) {
  int64_t phase = 0;
  for (int row = 0; row < nranges; ++row) {
    const int64_t base = table[2 * row], nbytes = table[2 * row + 1];
    if (base == 0 || nbytes <= 0) continue;
    const int64_t chunks = (nbytes + kStageChunk - 1) / kStageChunk;
    int64_t index = worker - phase % workers;
    if (index < 0) index += workers;
    for (; index < chunks; index += workers) {
      const int64_t offset = index * kStageChunk;
      int64_t size = nbytes - offset;
      if (size > kStageChunk) size = kStageChunk;
      size &= ~int64_t(15);
      if (size > 0) {
        const char* a = (const char*)((base + offset) & ~int64_t(15));
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a), "r"((uint32_t)size) : "memory");
      }
    }
    phase += chunks;
  }
}

}  // namespace w7
