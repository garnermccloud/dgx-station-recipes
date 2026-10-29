// Stage wqkv (partials): qr_kv partial[s] = mxfp8(x)[:, Ks] @ [wq_a; wkv]^T[:, Ks]  (5120 -> 1792, fp32 out).
//
// Task = (split s, 128-row tile rt) over the split's K range; 14 x nsplit tasks. The split partition mirrors the
// production tactic at this token count (FlashInfer LL GEMM cluster split-K: bucket 8 -> split 2 at K = 2560,
// bucket 4 -> 4 even, bucket 2 -> 4 uneven [0,1536,3072,4608,5120], bucket 1 -> 2), each split accumulated
// sequentially; stage qkv sums the partials in the production order. The activation is quantized in-task to MXFP8
// exactly as flashinfer.mxfp8_quantize (cute-dsl) does (amax * fp32(1/448) -> UE8M0 ceil, x * 2^(127-e) -> e4m3 RN
// satfinite), straight into the SW128 B operand.
#pragma once
#include "w3/mxf8_core.cuh"

namespace w3 {

// Quantize x[n, kb*32 .. +32) (bf16) for kb in [kc0*4, kc1*4) and n < T into the SW128 B layout + SFB atoms.
template <int N>
__device__ __forceinline__ void quantize_b_bf16(const __nv_bfloat16* x, int ld, int T, int kc0, int kc1, uint8_t* sB,
                                                uint8_t* sSFB) {
  constexpr uint32_t bstride = b_chunk_bytes(N);
  const int nkc = kc1 - kc0;
  // zero the scale atoms (rows >= T must read as UE8M0 0) and the padding token rows of B
  for (int i = threadIdx.x; i < nkc * 32; i += kComputeThreads)
    reinterpret_cast<uint4*>(sSFB)[i] = make_uint4(0, 0, 0, 0);
  for (int i = threadIdx.x; i < nkc * (N - T) * 8; i += kComputeThreads) {
    const int j = i & 7, n = T + (i >> 3) % (N - T), c = i / ((N - T) * 8);
    *reinterpret_cast<uint4*>(sB + c * bstride + sw128_offset(n, j * 16)) = make_uint4(0, 0, 0, 0);
  }
  compute_sync();
  const int nblk = T * nkc * 4;
  for (int b = threadIdx.x; b < nblk; b += kComputeThreads) {
    const int n = b % T, kbl = b / T;  // local k-block
    const int kb = kc0 * 4 + kbl;
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(n) * ld + kb * 32);
    uint4 v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = src[i];
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(v);
    float f[32];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float2 p = __bfloat1622float2(h[i]);
      f[2 * i] = p.x;
      f[2 * i + 1] = p.y;
      amax = fmaxf(amax, fmaxf(fabsf(p.x), fabsf(p.y)));
    }
    const uint32_t e = float_to_ue8m0(amax * 0.002232142857142857f);
    const float inv = ue8m0_inv_scale(e);
    uint32_t q[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t lo = cvt_e4m3x2(f[4 * i] * inv, f[4 * i + 1] * inv);
      const uint32_t hi = cvt_e4m3x2(f[4 * i + 2] * inv, f[4 * i + 3] * inv);
      q[i] = lo | (hi << 16);
    }
    const int c = kbl >> 2, kin = (kbl & 3) * 32;
    uint8_t* row = sB + c * bstride;
    *reinterpret_cast<uint4*>(row + sw128_offset(n, kin)) = make_uint4(q[0], q[1], q[2], q[3]);
    *reinterpret_cast<uint4*>(row + sw128_offset(n, kin + 16)) = make_uint4(q[4], q[5], q[6], q[7]);
    sSFB[c * 512 + (n & 31) * 16 + ((n >> 5) & 3) * 4 + (kbl & 3)] = static_cast<uint8_t>(e);
  }
}

struct WqkvStage {
  struct Params {
    alignas(64) CUtensorMap tm_w;  // fused [wq_a; wkv] [1792, 5120] uint8, box 128 x 128 B, SW128
    const uint8_t* w_sf;           // F8_128x4 weight scales
    const uint8_t* w;              // raw weight pointer (L2 prefetch)
    const __nv_bfloat16* x;        // [T, 5120] attention input
    float* part;                   // [nsplit][T][1792]
    int T;
    int n_out;     // 1792
    int k_groups;  // 40
    int nsplit;
    int kb[5];     // split boundaries in 128-element chunks
    int max_chunks;  // max over splits of kb[s+1] - kb[s] (sizes the B buffer)
    int l2_prefetch;
  };
  static constexpr int kTraceId = 0;
  static constexpr bool kHasWeights = true;
  static constexpr const char* kName = "wqkv";
  __host__ __device__ static int tasks(const Params& p) { return (p.n_out / 128) * p.nsplit; }
  __host__ __device__ static uint32_t smem_bytes(int n_mma, int max_chunks) {
    return max_chunks * b_chunk_bytes(n_mma) + max_chunks * 512 + 64;
  }
  __device__ static void produce(const Params& p, int task, Ring& ring) {
    const int rt = task % (p.n_out / 128), s = task / (p.n_out / 128);
    if (p.l2_prefetch) prefetch_a_rows_l2(p.w, p.k_groups * 128, rt, p.kb[s], p.kb[s + 1]);
    produce_a_chunks(&p.tm_w, p.w_sf, p.k_groups, rt, p.kb[s], p.kb[s + 1], ring, policy_evict_first());
  }
  template <int N, class Sync>
  __device__ static void consume(const Params& p, int task, int iter, bool /*last*/, Sync& sync, Ring& ring,
                                 uint8_t* smem, uint32_t tmem) {
    const int tiles = p.n_out / 128;
    const int rt = task % tiles, s = task / tiles;
    const int kc0 = p.kb[s], kc1 = p.kb[s + 1];
    uint8_t* sB = smem;
    uint8_t* sSFB = sB + p.max_chunks * b_chunk_bytes(N);
    uint64_t* done = reinterpret_cast<uint64_t*>(sSFB + p.max_chunks * 512);
    if (iter == 0 && threadIdx.x == 0) {
      mbar_init(done, 1);
      fence_mbar_init();
    }
    if (iter == 0) sync.wait();
    trace(kTraceId, 2);
    quantize_b_bf16<N>(p.x, p.k_groups * 128, p.T, kc0, kc1, sB, sSFB);
    fence_proxy_async_smem();
    compute_sync();
    trace(kTraceId, 3);
    if (threadIdx.x == 0) issue_mmas<N>(ring, sB, sSFB, kc0, kc1, kc0, tmem, tmem_acc(0), done);
    trace(kTraceId, 4);
    mbar_wait(done, iter & 1);
    trace(kTraceId, 5);
    tc_fence_after();
    const int w = warp_id();
    const int row = rt * 128 + (w & 3) * 32 + lane_id();
#pragma unroll
    for (int g = 0; g < N / 8; ++g) {
      if ((g & 1) != (w >> 2)) continue;
      float v[8];
      acc_row8(tmem, tmem_acc(0) + g * 8, v);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int n = g * 8 + i;
        if (n < p.T) p.part[(static_cast<size_t>(s) * p.T + n) * p.n_out + row] = v[i];
      }
    }
    tc_fence_before();
    compute_sync();
  }
};

}  // namespace w3
