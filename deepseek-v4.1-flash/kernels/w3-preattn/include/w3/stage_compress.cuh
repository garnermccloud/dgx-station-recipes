// Stage compress: the compressor's state save + group pooling + RMSNorm (vLLM af7f9488
// _fused_save_compress_norm_kernel), bitwise. Same task space as the Triton grid: ratio 2 has one request task per
// request (pools the group its chunk's first token closes with the ring row, then saves the chunk's last rows to the
// ring) followed by one task per pair of packed tokens (the group ending inside the pair); ratio 1 one task per token.
//
// Faster than the Triton kernel by parallelism, not by different math: the ring tail rows (<= ring size, 8) are
// loaded at once by all 256 threads instead of one row per sequential loop iteration, and the norm weight is fetched
// before griddepcontrol.wait. Arithmetic replicates the Triton PTX op for op (sm_103a, num_warps 4):
//   softmax over the 2 rows: m = max.f32(s0, s1), e_i = ex2.approx(mul.rn(s_i - m, log2 e)), den = e0 + e1,
//   w_i = div.full(e_i, den), pooled = fma(w0, kv0, w1 * kv1)
//   sum of squares: thread v of 128 owns columns 4v..4v+3: s = x1*x1; s = fma(x0,x0,s); s = fma(x2,x2,s);
//   s = s + x3*x3; warp butterfly 16..1; warps (w0 + w2) + (w1 + w3)
//   r = rsqrt.approx.ftz(div.full(sum, 512) + eps); y = bf16_rn((x * r) * w)
#pragma once
#include <cuda_bf16.h>

#include "w3/fw.cuh"

namespace w3 {

struct CompressStage {
  struct Params {
    const float* raw;              // kv_score [T, 512 * ratio] fp32
    long long raw_stride;
    const long long* positions;    // [T]
    float* state;                  // ratio 2: ring [blocks, ring, 1024] fp32
    long long state_stride;
    long long state_row_stride;
    int ring;                      // rows per ring block (8)
    const long long* slots;        // [num_tokens] ring slots (ratio 2) or main-cache slots (ratio 1); < 0 skip
    const int* qsl;                // ratio 2: [num_reqs + 1]
    const int* req_ids;            // ratio 2: [T]
    const __nv_bfloat16* norm_w;   // [512]
    __nv_bfloat16* latent;         // [T, 512]
    int num_tokens;
    int num_reqs;                  // 0 for ratio 1
    int ratio;                     // 1 or 2
    float eps;
  };
  static constexpr int kTraceId = 0;
  static constexpr bool kHasWeights = false;
  static constexpr const char* kName = "compress";
  static constexpr int kMaxRing = 8;
  __host__ __device__ static int tasks(const Params& p) {
    return p.num_reqs + (p.num_tokens + p.ratio - 1) / p.ratio;
  }
  __host__ __device__ static uint32_t smem_bytes(int, int) { return 64; }
  __device__ static void produce(const Params&, int, Ring&) {}

  __device__ static float ex2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  }
  __device__ static float div_full(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
  }
  __device__ static float rsqrt_approx_ftz(float x) {
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  }

  // softmax-pool two [kv | score] rows over 4 columns (row 0 = older row)
  __device__ static float4 pool2(float4 kv0, float4 s0, float4 kv1, float4 s1) {
    const float a0[4] = {kv0.x, kv0.y, kv0.z, kv0.w}, b0[4] = {s0.x, s0.y, s0.z, s0.w};
    const float a1[4] = {kv1.x, kv1.y, kv1.z, kv1.w}, b1[4] = {s1.x, s1.y, s1.z, s1.w};
    float o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float m = fmaxf(b0[j], b1[j]);
      const float e0 = ex2_approx(__fmul_rn(__fsub_rn(b0[j], m), 1.4426950408889634f));
      const float e1 = ex2_approx(__fmul_rn(__fsub_rn(b1[j], m), 1.4426950408889634f));
      const float den = __fadd_rn(e0, e1);
      o[j] = __fmaf_rn(div_full(e0, den), a0[j], __fmul_rn(div_full(e1, den), a1[j]));
    }
    return make_float4(o[0], o[1], o[2], o[3]);
  }

  // RMSNorm of the 512-column pooled row held by threads 0..127 (4 columns each) and its bf16 store; all 256
  // compute threads call this (barriers), `active` = this CTA has a row (uniform).
  __device__ static void norm_store(const Params& p, float4 x, const float (&w)[4], int row, float* part) {
    const int t = threadIdx.x;
    float s = 0.f;
    if (t < 128) {
      s = __fmul_rn(x.y, x.y);
      s = __fmaf_rn(x.x, x.x, s);
      s = __fmaf_rn(x.z, x.z, s);
      s = __fadd_rn(s, __fmul_rn(x.w, x.w));
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
      if (lane_id() == 0) part[warp_id()] = s;
    }
    compute_sync();
    if (t < 128) {
      const float total = __fadd_rn(__fadd_rn(part[0], part[2]), __fadd_rn(part[1], part[3]));
      const float r = rsqrt_approx_ftz(__fadd_rn(div_full(total, 512.f), p.eps));
      __nv_bfloat162 lo = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(x.x, r), w[0]), __fmul_rn(__fmul_rn(x.y, r), w[1]));
      __nv_bfloat162 hi = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(x.z, r), w[2]), __fmul_rn(__fmul_rn(x.w, r), w[3]));
      uint2 v;
      v.x = *reinterpret_cast<uint32_t*>(&lo);
      v.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(p.latent + static_cast<long long>(row) * 512 + 4 * t) = v;
    }
    compute_sync();
  }

  template <int N, class Sync>
  __device__ static void consume(const Params& p, int task, int iter, bool, Sync& sync, Ring&, uint8_t* smem, uint32_t) {
    float* part = reinterpret_cast<float*>(smem);
    const int t = threadIdx.x;
    float w[4] = {0.f, 0.f, 0.f, 0.f};  // the norm weight is a constant: fetched before the dependency wait
    if (t < 128) {
      const uint2 wv = *reinterpret_cast<const uint2*>(p.norm_w + 4 * t);
      const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&wv.x);
      const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&wv.y);
      w[0] = __low2float(lo);
      w[1] = __high2float(lo);
      w[2] = __low2float(hi);
      w[3] = __high2float(hi);
    }
    if (iter == 0) sync.wait();
    const int c4 = 4 * t;  // threads < 128: columns [c4, c4 + 4)
    if (p.ratio == 2 && task < p.num_reqs) {
      // ---- request task: the group the chunk's first token closes (ring row + raw row), then the ring tail store
      const int start = p.qsl[task], end = p.qsl[task + 1];
      if (start >= end) return;
      const long long slot0 = p.slots[start];
      if (slot0 < 0) return;
      const long long pos = p.positions[start];
      const bool closes = (pos + 1) % 2 == 0;
      const int nrows = min(end - start, p.ring);
      // every load first: the ring row the pooling reads, the raw tail rows, their slots
      float4 kv0, s0, kv1, s1;
      if (closes && t < 128) {
        const float* prev = p.state + (slot0 / p.ring) * p.state_stride + ((pos - 1) % p.ring) * p.state_row_stride;
        const float* cur = p.raw + static_cast<long long>(start) * p.raw_stride;
        kv0 = *reinterpret_cast<const float4*>(prev + c4);
        s0 = *reinterpret_cast<const float4*>(prev + 512 + c4);
        kv1 = *reinterpret_cast<const float4*>(cur + c4);
        s1 = *reinterpret_cast<const float4*>(cur + 512 + c4);
      }
      float4 tail[kMaxRing];
      long long tslot[kMaxRing];
#pragma unroll
      for (int k = 0; k < kMaxRing; ++k) {
        tslot[k] = -1;
        if (k < nrows) {
          const int token = end - nrows + k;
          tslot[k] = p.slots[token];
          tail[k] = *reinterpret_cast<const float4*>(p.raw + static_cast<long long>(token) * p.raw_stride + c4);
        }
      }
      float4 pooled = make_float4(0.f, 0.f, 0.f, 0.f);
      if (closes && t < 128) pooled = pool2(kv0, s0, kv1, s1);
      compute_sync();  // every ring read is done (its values consumed above) before the tail store overwrites rows
#pragma unroll
      for (int k = 0; k < kMaxRing; ++k)
        if (k < nrows && tslot[k] >= 0)
          *reinterpret_cast<float4*>(p.state + (tslot[k] / p.ring) * p.state_stride +
                                     (tslot[k] % p.ring) * p.state_row_stride + c4) = tail[k];
      if (closes) norm_store(p, pooled, w, start, part);
      return;
    }
    // ---- group task: ratio 1 one token; ratio 2 the token of the pair that ends a group
    int tk = (task - p.num_reqs) * p.ratio;
    if (p.ratio == 2) tk += static_cast<int>((p.positions[tk] + 1) % 2);
    const long long slot = tk < p.num_tokens ? p.slots[tk] : -1;
    if (slot < 0) return;
    float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
    if (p.ratio == 1) {
      if (t < 128) x = *reinterpret_cast<const float4*>(p.raw + static_cast<long long>(tk) * p.raw_stride + c4);
    } else {
      if (tk == p.qsl[p.req_ids[tk]]) return;  // a chunk's first token belongs to its request task
      if (t < 128) {
        const float* r0 = p.raw + static_cast<long long>(tk - 1) * p.raw_stride;
        const float* r1 = r0 + p.raw_stride;
        x = pool2(*reinterpret_cast<const float4*>(r0 + c4), *reinterpret_cast<const float4*>(r0 + 512 + c4),
                  *reinterpret_cast<const float4*>(r1 + c4), *reinterpret_cast<const float4*>(r1 + 512 + c4));
      }
    }
    norm_store(p, x, w, tk, part);
  }
};

}  // namespace w3
