// Stage kt: the attention's compressed-KV key table for one token (compute_global_topk_indices_and_lens) plus the
// int32 positions forward_mqa casts. Integer-exact. Task = token; T tasks; reads only step metadata and the
// index source's top-k buffer (written by an earlier layer), so it runs at t = 0 next to the wqkv GEMM.
#pragma once
#include "w3/fw.cuh"

namespace w3 {

struct KtStage {
  struct Params {
    const int* topk;          // [T, topk_stride] request-local compressed positions (-1 = none)
    int topk_stride;
    int topk_n;               // 512
    const int* token_to_req;  // [T]
    const int* block_table;   // [reqs, bt_stride] kv-source compressed cache pages
    int bt_stride;
    int block_size;           // compressed slots per page (kernel block 128 / ratio)
    const bool* is_valid;     // [T]
    const long long* positions;  // [T] int64
    int* gidx;                // [T, topk] global slot ids
    int* glens;               // [T]
    int* pos32;               // [T] (may be null)
    int T;
  };
  static constexpr int kTraceId = 3;
  static constexpr bool kHasWeights = false;
  static constexpr const char* kName = "kt";
  __host__ __device__ static int tasks(const Params& p) { return p.T; }
  __host__ __device__ static uint32_t smem_bytes(int, int) { return 64; }

  __device__ static void produce(const Params&, int, Ring&) {}
  template <int N, class Sync>
  __device__ static void consume(const Params& p, int n, int, bool, Sync&, Ring&, uint8_t* smem, uint32_t) {
    int* red = reinterpret_cast<int*>(smem);
    if (p.topk == nullptr) {  // sliding-window-only layer: positions only
      if (threadIdx.x == 0 && p.pos32) p.pos32[n] = static_cast<int>(p.positions[n]);
      return;
    }
    const int req = p.token_to_req[n];
    int cnt = 0;
    for (int i = threadIdx.x; i < p.topk_n; i += kComputeThreads) {
      const int local = p.topk[static_cast<size_t>(n) * p.topk_stride + i];
      int slot = -1;
      if (local >= 0) {
        const int blk = p.block_table[static_cast<size_t>(req) * p.bt_stride + local / p.block_size];
        slot = blk * p.block_size + local % p.block_size;
        ++cnt;
      }
      p.gidx[static_cast<size_t>(n) * p.topk_n + i] = slot;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
    if (lane_id() == 0) red[warp_id()] = cnt;
    compute_sync();
    if (threadIdx.x == 0) {
      int tot = 0;
      for (int w = 0; w < 8; ++w) tot += red[w];
      p.glens[n] = p.is_valid[n] ? tot : 0;
      if (p.pos32) p.pos32[n] = static_cast<int>(p.positions[n]);
    }
    compute_sync();
  }
};

}  // namespace w3
