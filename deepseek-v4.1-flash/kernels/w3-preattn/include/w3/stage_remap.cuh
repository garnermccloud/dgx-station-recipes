// Stage remap: sparse-logits top-k columns -> request-local compressed positions (the shared topk_indices_buffer
// later layers read) -> the attention key table (global slots + lens), one launch. Replaces, for candidate-consumer
// indexer layers (24/28/32/36), _sparse_topk_remap_kernel + positions.to(int32) +
// _compute_global_topk_indices_and_lens (and makes the topk_indices_buffer[:T] = -1 fill redundant: every column of
// rows [0, T) is written). Integer-exact. Task = token.
#pragma once
#include "w3/fw.cuh"

namespace w3 {

struct RemapStage {
  struct Params {
    const int* cols;            // [T, col_stride] DeepSelect sparse-column top-k (-1 / out-of-range = none)
    int col_stride;
    const int* sparse_idx;      // [T, si_stride] sparse block ids backing the logits
    int si_stride;
    const int* row_ks;          // [T] (zeros on the paged path)
    int width;                  // logits columns = num sparse blocks * sbk
    int sbk;                    // sparse block size (8)
    int topk_n;                 // 512
    int* topk_out;              // [T, topk_stride] request-local positions
    int topk_stride;
    const int* token_to_req;    // [T]
    const int* block_table;     // [reqs, bt_stride]
    int bt_stride;
    int block_size;             // compressed slots per page
    const bool* is_valid;       // [T]
    int* gidx;                  // [T, topk_n]
    int* glens;                 // [T]
    const long long* positions; // [T]
    int* pos32;                 // [T] or null
    int T;
  };
  static constexpr int kTraceId = 4;
  static constexpr bool kHasWeights = false;
  static constexpr const char* kName = "remap";
  __host__ __device__ static int tasks(const Params& p) { return p.T; }
  __host__ __device__ static uint32_t smem_bytes(int, int) { return 64; }
  __device__ static void produce(const Params&, int, Ring&) {}

  template <int N, class Sync>
  __device__ static void consume(const Params& p, int n, int iter, bool, Sync& sync, Ring&, uint8_t* smem, uint32_t) {
    int* red = reinterpret_cast<int*>(smem);
    const int req = p.token_to_req[n];  // step metadata: before the wait
    const int ks_base = p.row_ks[n] / p.sbk;
    if (iter == 0) sync.wait();
    int cnt = 0;
    for (int i = threadIdx.x; i < p.topk_n; i += kComputeThreads) {
      const int c = p.cols[static_cast<size_t>(n) * p.col_stride + i];
      const bool valid = c >= 0 && c < p.width;
      const int cc = valid ? c : 0;
      const int blk = p.sparse_idx[static_cast<size_t>(n) * p.si_stride + cc / p.sbk];
      const int pos = valid ? (blk - ks_base) * p.sbk + cc % p.sbk : -1;
      p.topk_out[static_cast<size_t>(n) * p.topk_stride + i] = pos;
      int slot = -1;
      if (pos >= 0) {
        const int pg = p.block_table[static_cast<size_t>(req) * p.bt_stride + pos / p.block_size];
        slot = pg * p.block_size + pos % p.block_size;
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
