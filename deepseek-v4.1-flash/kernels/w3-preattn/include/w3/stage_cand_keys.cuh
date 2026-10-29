// Stage cand_keys (first half of the long-row candidate selection): block scores -> order-preserving uint32 keys
// in global scratch + a per-chunk histogram of the keys' top 12 bits. Task = (row, chunk of kChunk blocks); tasks
// past a row's valid prefix exit at once. Stage cand (with p.chunk_hist set) then selects from the keys.
#pragma once
#include "w3/stage_cand.cuh"

namespace w3 {

struct CandKeysStage {
  static constexpr int kChunk = 4096;  // blocks per task
  static constexpr int kBins = 4096;   // top 12 key bits
  struct Params {
    const float* logits;
    long long ld;
    int width;
    const int* row_end;
    int block_size;   // 8
    uint32_t* gkeys;  // [rows, gkeys_stride]
    int gkeys_stride;
    uint32_t* hist;   // [rows, nchunks, kBins]
    int nchunks;
    int rows;
  };
  static constexpr int kTraceId = 4;
  static constexpr bool kHasWeights = false;
  static constexpr const char* kName = "cand_keys";
  __host__ __device__ static int tasks(const Params& p) { return p.rows * p.nchunks; }
  __host__ __device__ static uint32_t smem_bytes(int, int) { return kBins * 4 + 64; }
  __device__ static void produce(const Params&, int, Ring&) {}

  template <int N, class Sync>
  __device__ static void consume(const Params& p, int task, int iter, bool, Sync& sync, Ring&, uint8_t* smem, uint32_t) {
    uint32_t* h = reinterpret_cast<uint32_t*>(smem);
    const int r = task / p.nchunks, ch = task % p.nchunks;
    const int t = threadIdx.x;
    if (iter == 0) sync.wait();
    const int end = p.row_end[r];
    const int nvalid = end > 0 ? min((end + p.block_size - 1) / p.block_size, (p.width + p.block_size - 1) / p.block_size) : 0;
    const int b_lo = ch * kChunk, b_hi = min(b_lo + kChunk, nvalid);
    uint32_t* hrow = p.hist + (static_cast<long long>(r) * p.nchunks + ch) * kBins;
    for (int i = t; i < kBins; i += kComputeThreads) h[i] = 0;
    compute_sync();
    if (b_lo < b_hi) {
      const float* row = p.logits + static_cast<long long>(r) * p.ld;
      uint32_t* gk = p.gkeys + static_cast<long long>(r) * p.gkeys_stride;
      const int half = t & 1;
      constexpr int U = 8;
      for (int b0 = b_lo; b0 < b_hi; b0 += 128 * U) {
        float4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const int b = b0 + u * 128 + (t >> 1);
          const int c = b * 8 + half * 4;
          const float ninf = __int_as_float(0xff800000);
          v[u] = make_float4(ninf, ninf, ninf, ninf);
          if (b < b_hi) {
            if (c + 4 <= end && c + 4 <= p.width) {
              v[u] = *reinterpret_cast<const float4*>(row + c);
            } else {
              for (int i = 0; i < 4; ++i)
                if (c + i < end && c + i < p.width) (&v[u].x)[i] = row[c + i];
            }
          }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const int b = b0 + u * 128 + (t >> 1);
          float m = CandStage::nanmax(CandStage::nanmax(v[u].x, v[u].y), CandStage::nanmax(v[u].z, v[u].w));
          m = CandStage::nanmax(m, __shfl_xor_sync(0xffffffffu, m, 1));
          if (half == 0 && b < b_hi) {
            const uint32_t k = (b == nvalid - 1) ? CandStage::fkey(__int_as_float(0x7f800000)) : CandStage::fkey(m);
            gk[b] = k;
            atomicAdd(&h[k >> 20], 1u);
          }
        }
      }
    }
    compute_sync();
    for (int i = t; i < kBins; i += kComputeThreads) hrow[i] = h[i];
    compute_sync();
  }
};

}  // namespace w3
