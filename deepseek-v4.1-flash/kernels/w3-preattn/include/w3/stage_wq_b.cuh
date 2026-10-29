// Stage wq_b: q = qr_q @ wq_b^T (MXFP8, 1280 -> 32768, rows pre-permuted for mega attention), bf16 out.
// Task = one 128-row tile of wq_b over the full K = 1280 (10 chunks); 256 tasks. Production (FlashInfer LL GEMM at
// M <= 8, cute-dsl persistent above) runs split_k = 1 for this shape, so sequential K inside the task is bitwise.
// Reads: qr_q (fp8 [T, 1280]) + its F8_128x4 scales, written by stage qkv (previous launch / earlier stage).
#pragma once
#include "w3/mxf8_core.cuh"

namespace w3 {

struct WqBStage {
  struct Params {
    alignas(64) CUtensorMap tm_w;  // wq_b weight [32768, 1280] uint8, box 128 x 128 B, SW128
    const uint8_t* w_sf;           // F8_128x4 swizzled weight scales
    const uint8_t* xq;             // [T, 1280] e4m3 activations
    const uint8_t* x_sf;           // F8_128x4 swizzled activation scales (rows padded to 128)
    __nv_bfloat16* out;            // [T, n_out]
    int T;
    int n_out;   // 32768
    int k_groups;  // K / 128 = 10
  };
  static constexpr int kTraceId = 2;
  static constexpr bool kHasWeights = true;
  static constexpr const char* kName = "wq_b";
  __host__ __device__ static int tasks(const Params& p) { return p.n_out / 128; }
  __host__ __device__ static constexpr uint32_t smem_bytes(int n_mma, int k_groups) {
    return k_groups * b_chunk_bytes(n_mma) + k_groups * 512 + 64;
  }

  __device__ static void produce(const Params& p, int task, Ring& ring) {
    produce_a_chunks(&p.tm_w, p.w_sf, p.k_groups, task, 0, p.k_groups, ring, policy_evict_first());
  }

  // Tasks of a CTA are software-pipelined by one: consume(task i) issues task i's MMAs into accumulator i % 2 and
  // then drains task i-1's accumulator, so the tensor pipe never idles on an epilogue; the CTA's last task drains
  // itself too.
  template <int N, class Sync>
  __device__ static void consume(const Params& p, int task, int iter, bool last, Sync& sync, Ring& ring, uint8_t* smem,
                                 uint32_t tmem) {
    uint8_t* sB = smem;
    uint8_t* sSFB = sB + p.k_groups * b_chunk_bytes(N);
    uint64_t* done = reinterpret_cast<uint64_t*>(sSFB + p.k_groups * 512);  // [2]
    if (iter == 0) {
      if (threadIdx.x == 0) {
        mbar_init(&done[0], 1);
        mbar_init(&done[1], 1);
        fence_mbar_init();
      }
      sync.wait();
      trace(kTraceId, 2);
      load_b_fp8<N>(p.xq, p.k_groups * 128, p.x_sf, p.T, 0, p.k_groups, sB, sSFB);
      fence_proxy_async_smem();
      compute_sync();
      trace(kTraceId, 3);
    }
    if (threadIdx.x == 0) issue_mmas<N>(ring, sB, sSFB, 0, p.k_groups, 0, tmem, tmem_acc(iter & 1), &done[iter & 1], iter == 0 ? kTraceId : -1);
    trace(kTraceId, 4 + (iter & 1));
    if (iter > 0) drain<N>(p, task - static_cast<int>(gridDim.x), iter - 1, tmem, done);
    if (last) drain<N>(p, task, iter, tmem, done);
    trace(kTraceId, 6 + (iter & 1));
  }

  template <int N>
  __device__ static void drain(const Params& p, int task, int iter, uint32_t tmem, uint64_t* done) {
    mbar_wait(&done[iter & 1], (iter >> 1) & 1);
    tc_fence_after();
    const int w = warp_id();
    const int row = task * 128 + (w & 3) * 32 + lane_id();
#pragma unroll
    for (int g = 0; g < N / 8; ++g) {
      if ((g & 1) != (w >> 2)) continue;
      float v[8];
      acc_row8(tmem, tmem_acc(iter & 1) + g * 8, v);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int n = g * 8 + i;
        if (n < p.T) p.out[static_cast<size_t>(n) * p.n_out + row] = __float2bfloat16_rn(v[i]);
      }
    }
    tc_fence_before();
    compute_sync();
  }
};

}  // namespace w3
