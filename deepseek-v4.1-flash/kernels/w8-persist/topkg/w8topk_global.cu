// libw8topkg.so: the indexer's decode top-k (vLLM's cooperative_topk, its kernel body compiled unchanged from
// src/w8topk/cooperative_topk.cuh) fused with the key-table translation that follows it on the compressor + indexer
// layers (vLLM compute_global_topk_indices_and_lens: local compressed index -> global KV slot via the block table,
// per-token valid count, 0 for padding tokens). One launch instead of two; plain C ABI for ctypes.
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#if defined(W8TG_SORT) && W8TG_SORT && !defined(W8TG_NO_DET_TIES)
#define W8TG_DET_TIES 1  // the determinism build also takes boundary ties by lowest index (topk_histogram_4096.cuh)
#endif
#include "w8topk/cooperative_topk.cuh"

namespace ct = vllm::cooperative;
namespace hist4096 = vllm::topk_histogram_4096;

namespace w8tg {

struct Global {
  int32_t* gidx;               // out [rows][TopK] global slot ids (-1 invalid)
  int32_t* glens;              // out [rows] valid count (0 for padding tokens)
  const int32_t* token_to_req; // [rows]
  const int32_t* block_table;  // [reqs][bt_stride] (int32)
  int64_t bt_stride;
  int32_t block_size;          // compressed entries per block (flashmla block_size / compress ratio)
  const bool* is_valid;        // [rows]
};

#ifndef W8TG_SORT
#define W8TG_SORT 0
#endif

#if W8TG_SORT
// Ascending emission of the row's TopK selected indices (all distinct, in [0, len), len > TopK) through a bitmap in smem:
// clear len / 32 words, atomicOr each index's bit, block-wide exclusive scan of the words' popcounts (each thread owns
// a contiguous run of words), then each thread emits its words' set bits in order. O(len / 32 + TopK), no sort.
__device__ __forceinline__ void bitmap_emit(const int32_t* row_out, uint32_t len, uint32_t* bm, uint32_t* dst,
                                            uint32_t* s_ws, uint32_t tx) {
  constexpr uint32_t BS = hist4096::kBlockSize;
  const uint32_t nwords = (len + 31) / 32, per = (nwords + BS - 1) / BS;
  for (uint32_t w = tx; w < nwords; w += BS) bm[w] = 0;
  for (uint32_t i = tx; i < 512; i += BS) dst[i] = 0xFFFFFFFFu;  // stays -1 if a row ever selects fewer than 512
  __syncthreads();
  for (uint32_t i = tx; i < 512; i += BS) {
    const int32_t v = __ldcg(row_out + i);
    if (v >= 0) atomicOr(&bm[v >> 5], 1u << (v & 31));
  }
  __syncthreads();
  const uint32_t w0 = tx * per, w1 = min(w0 + per, nwords);
  uint32_t local = 0;
  for (uint32_t w = w0; w < w1; ++w) local += __popc(bm[w]);
  const uint32_t lane = tx & 31, warp = tx >> 5;
  uint32_t x = local;
#pragma unroll
  for (uint32_t o = 1; o < 32; o <<= 1) {
    const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) s_ws[warp] = x;
  __syncthreads();
  if (warp == 0) {
    uint32_t v = lane < BS / 32 ? s_ws[lane] : 0;
#pragma unroll
    for (uint32_t o = 1; o < 32; o <<= 1) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, v, o);
      if (lane >= o) v += y;
    }
    if (lane < BS / 32) s_ws[lane] = v;
  }
  __syncthreads();
  uint32_t pos = x - local + (warp > 0 ? s_ws[warp - 1] : 0);
  for (uint32_t w = w0; w < w1; ++w) {
    uint32_t bits = bm[w];
    while (bits) {
      const uint32_t b = __ffs(bits) - 1;
      dst[pos++] = w * 32 + b;
      bits &= bits - 1;
    }
  }
  __syncthreads();
}
#endif

// compute_global_topk_indices_and_lens on row blockIdx.x, by cluster rank 0, once every rank's top-k writes landed.
// W8TG_SORT=1 (determinism build): rank 0 first sorts the row's selected request-local indices ascending (invalid -1
// last) and rewrites the top-k row, so both outputs are independent of the top-k body's atomicAdd arrival order.
template <uint32_t TopK>
__device__ __forceinline__ void translate_row(const ct::CooperativeTopKParams<TopK>& p, const Global& g) {
  cooperative_groups::this_cluster().sync();  // release / acquire at cluster scope: the peers' output writes
  if (blockIdx.y != 0) return;
  __shared__ int s_cnt[hist4096::kBlockSize / 32];
  const uint32_t row = blockIdx.x, tx = threadIdx.x;
  int32_t* row_out = p.output + static_cast<size_t>(row) * TopK;
#if W8TG_SORT
  // Rows longer than TopK: the TopK selected indices re-emitted ascending (bitmap_emit) in the top-k body's dynamic
  // smem, dead after the cluster sync (no peer reads it any more). Rows of len <= TopK are already iota (+ -1 padding).
  extern __shared__ __align__(16) uint8_t w8tg_dyn[];
  __shared__ uint32_t s_ws[hist4096::kBlockSize / 32];
  uint32_t* s_key = reinterpret_cast<uint32_t*>(w8tg_dyn);
  uint32_t* s_bm = s_key + TopK;
  const int32_t len_s = p.lengths[row];
  const uint32_t len = len_s > 0 ? static_cast<uint32_t>(len_s) : 0u;
  const bool sorted_row = len > TopK;
  if (sorted_row) {
    static_assert(TopK == 512, "bitmap_emit is written for the 512-wide decode top-k");
    bitmap_emit(row_out, len, s_bm, s_key, s_ws, tx);
    for (uint32_t i = tx; i < TopK; i += hist4096::kBlockSize)
      row_out[i] = s_key[i] == 0xFFFFFFFFu ? -1 : static_cast<int32_t>(s_key[i]);
  }
#endif
  const int32_t* out = row_out;
  const int32_t req = g.token_to_req[row];
  const int32_t* bt = g.block_table + static_cast<int64_t>(req) * g.bt_stride;
  int cnt = 0;
  for (uint32_t i = tx; i < TopK; i += hist4096::kBlockSize) {
#if W8TG_SORT
    const int32_t local = sorted_row ? (s_key[i] == 0xFFFFFFFFu ? -1 : static_cast<int32_t>(s_key[i])) : __ldcg(out + i);
#else
    const int32_t local = __ldcg(out + i);
#endif
    int32_t slot = -1;
    if (local >= 0) {
      slot = bt[local / g.block_size] * g.block_size + local % g.block_size;
      ++cnt;
    }
    g.gidx[static_cast<size_t>(row) * TopK + i] = slot;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
  if ((tx & 31) == 0) s_cnt[tx >> 5] = cnt;
  __syncthreads();
  if (tx == 0) {
    int total = 0;
    for (int w = 0; w < static_cast<int>(hist4096::kBlockSize / 32); ++w) total += s_cnt[w];
    g.glens[row] = g.is_valid[row] ? total : 0;
  }
}

template <uint32_t TopK>
__global__ void __launch_bounds__(hist4096::kBlockSize, 1) __cluster_dims__(1, 4, 1)
    topk_global_cs4(ct::CooperativeTopKParams<TopK> params, Global g) {
  ct::cooperative_topk_body<TopK, 4>(params);
  translate_row<TopK>(params, g);
}
template <uint32_t TopK>
__global__ void __launch_bounds__(hist4096::kBlockSize, 1) __cluster_dims__(1, 2, 1)
    topk_global_cs2(ct::CooperativeTopKParams<TopK> params, Global g) {
  ct::cooperative_topk_body<TopK, 2>(params);
  translate_row<TopK>(params, g);
}
template <uint32_t TopK>
__global__ void __launch_bounds__(hist4096::kBlockSize, 1) __cluster_dims__(1, 8, 1)
    topk_global_cs8(ct::CooperativeTopKParams<TopK> params, Global g) {
  ct::cooperative_topk_body<TopK, 8>(params);
  translate_row<TopK>(params, g);
}
template <uint32_t TopK>
__global__ void __launch_bounds__(hist4096::kBlockSize, 1) __cluster_dims__(1, 16, 1)
    topk_global_cs16(ct::CooperativeTopKParams<TopK> params, Global g) {
  ct::cooperative_topk_body<TopK, 16>(params);
  translate_row<TopK>(params, g);
}

// vLLM's launch_cooperative_cluster with the fused kernels
template <uint32_t TopK, uint32_t CS>
int launch(ct::CooperativeTopKParams<TopK>& params, const Global& g, size_t smem, cudaStream_t stream) {
  auto kernel = []() {
    if constexpr (CS == 2) return &topk_global_cs2<TopK>;
    else if constexpr (CS == 16) return &topk_global_cs16<TopK>;
    else if constexpr (CS == 8) return &topk_global_cs8<TopK>;
    else return &topk_global_cs4<TopK>;
  }();
  if constexpr (CS > 8) cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(params.num_rows, CS);
  cfg.blockDim = dim3(hist4096::kBlockSize);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {1, CS, 1};
  cfg.numAttrs = 1;
  cfg.attrs = attrs;
  return cudaLaunchKernelEx(&cfg, kernel, params, g) == cudaSuccess ? 0 : 5;
}

}  // namespace w8tg

extern "C" {

struct W8tgArgs {
  const float* logits;
  int64_t stride;          // logits.stride(0) (multiple of 4)
  const int32_t* lengths;  // [rows] per-row lengths (seq_lens flattened, as cooperative_topk takes them)
  int32_t* topk;           // out [rows][512] request-local indices (topk_indices_buffer rows)
  void* workspace;         // >= rows * kTieWsPerRow * sizeof(Tie) bytes (vLLM RADIX_TOPK_WORKSPACE_SIZE suffices)
  int32_t* gidx;
  int32_t* glens;
  const int32_t* token_to_req;
  const int32_t* block_table;
  int64_t bt_stride;
  int64_t block_size;
  const bool* is_valid;
  int64_t rows;
};

// 0: sizeof(W8tgArgs), 1: offsetof(rows), 2: max rows
long long w8tg_abi(int which) {
  switch (which) {
    case 0: return sizeof(W8tgArgs);
    case 1: return offsetof(W8tgArgs, rows);
    case 2: return 64;
    default: return -1;
  }
}

// 0 ok; 1 shape; 5 launch
int w8tg_launch(const W8tgArgs* a, void* stream) {
  if (a->rows < 1 || a->rows > 64 || a->stride % 4 != 0) return 1;
  ct::CooperativeTopKParams<512> params;
  params.input = a->logits;
  params.output = a->topk;
  params.lengths = a->lengths;
  params.num_rows = static_cast<uint32_t>(a->rows);
  params.stride = static_cast<uint32_t>(a->stride);
  params.tie_ws = reinterpret_cast<hist4096::Tie*>(a->workspace);
  w8tg::Global g{a->gidx, a->glens, a->token_to_req, a->block_table, a->bt_stride, static_cast<int32_t>(a->block_size),
                 a->is_valid};
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  int dev = 0, major = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  // vLLM's cluster-size choice by row count (launch_cooperative_topk_impl)
  if (a->rows <= 4 && major >= 10) return w8tg::launch<512, 16>(params, g, ct::kSmemSize8, st);
  if (a->rows <= 8) return w8tg::launch<512, 8>(params, g, ct::kSmemSize8, st);
  if (a->rows <= 33) return w8tg::launch<512, 4>(params, g, ct::kSmemSize4, st);
  return w8tg::launch<512, 2>(params, g, ct::kSmemSize2, st);
}

}  // extern "C"
