// Stage qkv: per token, the production split-K reduce of the wqkv partials -> bf16 qr_kv, then
//   q_norm (RMSNorm 1280) -> bf16 -> MXFP8 (qr_q + F8_128x4 scales)          [vLLM _q_kv_norm_quant_kernel]
//   kv_norm (RMSNorm 512) -> bf16 kv                                           [same kernel]
//   kv: GPT-J RoPE on dims 448.. -> bf16 -> MXFP8 (UE8M0 per 32) -> SWA record  [fused_deepseek_v4_qnorm_rope_kv_rope_
//                                                                                quant_insert, kv_mxfp8, pad_to = 0]
// Bitwise parity: the RMS reductions replicate the Triton kernel's order (num_warps 8, 8 contiguous elements per
// thread summed sequentially, butterfly 16..1 per warp, butterfly 4..1 over the 8 warp partials, div.full.f32 by the
// width, rsqrt.approx.ftz), the KV record replicates the CUDA kernel lane for lane.
// Task = token; T tasks, no weights.
#pragma once
#include <cuda_fp8.h>
#include "w3/fw.cuh"

namespace w3 {

struct QkvStage {
  struct Params {
    const float* part;  // [nsplit][T][1792] fp32 split partials, or nullptr to read qr_kv_in
    int nsplit;
    const __nv_bfloat16* qr_kv_in;  // [T, ld_in] bf16 GEMM output (production fused_wqa_wkv) when part == nullptr
    int ld_in;
    int T;
    const __nv_bfloat16* q_w;   // [1280]
    const __nv_bfloat16* kv_w;  // [512]
    float eps;
    uint8_t* qx8;               // [T, 1280] e4m3
    uint8_t* qx8_sf;            // F8_128x4, 128 padded rows x 40 groups (rows >= T stay 0)
    __nv_bfloat16* qr_kv_out;   // optional [T, 1792] (bf16 GEMM output, diagnostics)
    __nv_bfloat16* kv_out;      // optional [T, 512] normed kv
    // SWA cache insert
    uint8_t* swa_cache;
    long long swa_block_stride;
    const long long* slot_mapping;  // [num_insert] int64, -1 = skip
    int num_insert;                 // slot_mapping length (<= T; rows past it are padding)
    int zero_pad_sf;                // task 0 zero-fills the scale rows [T, pad128(T)) (fresh, uninitialized buffer)
    int split_roles;                // 1: tasks [0,T) = q path, [T,2T) = kv path (two CTAs per token, lower latency)
    const long long* positions;     // [T] int64
    const float* cos_sin;           // [max_pos, 64] fp32
    int swa_block_size;             // 32
  };
  static constexpr int kTraceId = 1;
  static constexpr bool kHasWeights = false;
  static constexpr const char* kName = "qkv";
  __host__ __device__ static int tasks(const Params& p) { return p.split_roles ? 2 * p.T : p.T; }
  __host__ __device__ static uint32_t smem_bytes(int, int) { return 512 * 2 + 64 * 4; }

  // Triton-order RMS of the 8 values this thread holds (threads past the width hold zeros).
  __device__ static float rms_rcp(const float (&v)[8], float* red, float width, float eps) {
    float s = v[0] * v[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) s = fmaf(v[i], v[i], s);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, o);
    if (lane_id() == 0) red[warp_id()] = s;
    compute_sync();
    if (threadIdx.x < 32) {
      float t = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
#pragma unroll
      for (int o = 4; o > 0; o >>= 1) t = t + __shfl_xor_sync(0xffffffffu, t, o);
      if (threadIdx.x == 0) red[8] = t;
    }
    compute_sync();
    const float tot = red[8];
    float mean;
    asm("div.full.f32 %0, %1, %2;" : "=f"(mean) : "f"(tot), "f"(width));
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(mean + eps));
    compute_sync();  // red[] reused by the next reduction
    return r;
  }

  __device__ static void produce(const Params&, int, Ring&) {}
  template <int N, class Sync>
  __device__ static void consume(const Params& p, int task, int iter, bool, Sync& sync, Ring&, uint8_t* smem, uint32_t) {
    __nv_bfloat16* skv = reinterpret_cast<__nv_bfloat16*>(smem);
    float* red = reinterpret_cast<float*>(smem + 1024);
    const int t = threadIdx.x;
    const int n = p.split_roles ? task % p.T : task;
    const bool do_q = !p.split_roles || task < p.T;   // CTA-uniform
    const bool do_kv = !p.split_roles || task >= p.T;
    const bool has_q = do_q && t < 160, has_kv = do_kv && t < 64;
    // Constants first (they do not depend on the previous launch): norm weights, slot, position, RoPE row.
    uint4 qw4 = make_uint4(0, 0, 0, 0), kvw4 = make_uint4(0, 0, 0, 0);
    if (has_q) qw4 = *reinterpret_cast<const uint4*>(p.q_w + 8 * t);
    if (has_kv) kvw4 = *reinterpret_cast<const uint4*>(p.kv_w + 8 * t);
    long long slot = -1, pos = 0;
    float4 rc0 = make_float4(0, 0, 0, 0), rc1 = rc0, rs0 = rc0, rs1 = rc0;
    if (do_q && n == 0 && p.zero_pad_sf) {
      const int padT = ((p.T + 127) / 128) * 128;
      for (int i = t; i < (padT - p.T) * 40; i += kComputeThreads) {
        const int rr = p.T + i / 40, g = i % 40;
        p.qx8_sf[(rr / 128) * (128 * 40) + (g / 4) * 512 + (rr % 32) * 16 + ((rr % 128) / 32) * 4 + (g % 4)] = 0;
      }
    }
    if (do_kv && warp_id() == 0) {
      slot = n < p.num_insert ? p.slot_mapping[n] : -1;
      pos = p.positions[n];
      const int L = lane_id();
      if (L >= 28) {
        const float* cp = p.cos_sin + pos * 64;
        const int hb = (L * 16 - 448) >> 1;
        rc0 = *reinterpret_cast<const float4*>(cp + hb);
        rc1 = *reinterpret_cast<const float4*>(cp + hb + 4);
        rs0 = *reinterpret_cast<const float4*>(cp + 32 + hb);
        rs1 = *reinterpret_cast<const float4*>(cp + 32 + hb + 4);
      }
    }
    const __nv_bfloat16* qw = reinterpret_cast<const __nv_bfloat16*>(&qw4);
    const __nv_bfloat16* kvw = reinterpret_cast<const __nv_bfloat16*>(&kvw4);
    trace(kTraceId, 2);
    if (iter == 0) sync.wait();
    trace(kTraceId, 3);
    // ---- split-K reduce, production order: shard owner r first, then the other ranks ascending.
    const int shard = 8 / p.nsplit;
    const int owner = min(n / (shard > 0 ? shard : 1), p.nsplit - 1);
    float q[8], kv[8];
    auto gather = [&](int col0, float (&o)[8]) {
      const float* base = p.part + static_cast<size_t>(n) * 1792 + col0;
      const size_t sstride = static_cast<size_t>(p.T) * 1792;
      float4 a = *reinterpret_cast<const float4*>(base + owner * sstride);
      float4 b = *reinterpret_cast<const float4*>(base + owner * sstride + 4);
      float acc[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      for (int s = 0; s < p.nsplit; ++s) {
        if (s == owner) continue;
        float4 c = *reinterpret_cast<const float4*>(base + s * sstride);
        float4 d = *reinterpret_cast<const float4*>(base + s * sstride + 4);
        acc[0] += c.x; acc[1] += c.y; acc[2] += c.z; acc[3] += c.w;
        acc[4] += d.x; acc[5] += d.y; acc[6] += d.z; acc[7] += d.w;
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) o[i] = bf16_round(acc[i]);
    };
#pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = kv[i] = 0.f;
    auto gather_bf16 = [&](int col0, float (&o)[8]) {
      const uint4 v = *reinterpret_cast<const uint4*>(p.qr_kv_in + static_cast<size_t>(n) * p.ld_in + col0);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        o[2 * i] = f.x;
        o[2 * i + 1] = f.y;
      }
    };
    if (p.part) {
      if (has_q) gather(8 * t, q);
      if (has_kv) gather(1280 + 8 * t, kv);
    } else {
      if (has_q) gather_bf16(8 * t, q);
      if (has_kv) gather_bf16(1280 + 8 * t, kv);
    }
    if (p.qr_kv_out) {
      __nv_bfloat16* o = p.qr_kv_out + static_cast<size_t>(n) * 1792;
      if (has_q)
        for (int i = 0; i < 8; ++i) o[8 * t + i] = __float2bfloat16_rn(q[i]);
      if (has_kv)
        for (int i = 0; i < 8; ++i) o[1280 + 8 * t + i] = __float2bfloat16_rn(kv[i]);
    }
    trace(kTraceId, 4);
    // ---- q_norm -> bf16 -> MXFP8
    const float rq = do_q ? rms_rcp(q, red, 1280.f, p.eps) : 0.f;
    if (has_q) {
      float y[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) y[i] = bf16_round((q[i] * rq) * __bfloat162float(qw[i]));
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(y[i]));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
      const uint32_t e = float_to_ue8m0(amax * 0.002232142857142857f);
      const float inv = ue8m0_inv_scale(e);
      uint32_t w0 = cvt_e4m3x2(y[0] * inv, y[1] * inv) | (static_cast<uint32_t>(cvt_e4m3x2(y[2] * inv, y[3] * inv)) << 16);
      uint32_t w1 = cvt_e4m3x2(y[4] * inv, y[5] * inv) | (static_cast<uint32_t>(cvt_e4m3x2(y[6] * inv, y[7] * inv)) << 16);
      *reinterpret_cast<uint2*>(p.qx8 + static_cast<size_t>(n) * 1280 + 8 * t) = make_uint2(w0, w1);
      if ((t & 3) == 0) {
        const int g = t >> 2;  // 32-element group
        const int off = (n / 128) * (128 * 40) + (g / 4) * 512 + (n % 32) * 16 + ((n % 128) / 32) * 4 + (g % 4);
        p.qx8_sf[off] = static_cast<uint8_t>(e);
      }
    }
    // ---- kv_norm -> bf16
    if (!do_kv) return;
    const float rk = rms_rcp(kv, red, 512.f, p.eps);
    if (has_kv) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const __nv_bfloat16 b = __float2bfloat16_rn((kv[i] * rk) * __bfloat162float(kvw[i]));
        skv[8 * t + i] = b;
        if (p.kv_out) p.kv_out[static_cast<size_t>(n) * 512 + 8 * t + i] = b;
      }
    }
    compute_sync();
    trace(kTraceId, 5);
    // ---- SWA record insert: warp 0, lane L owns dims [16L, 16L+16) (fused insert kernel, KV slot branch)
    if (warp_id() == 0) {
      const int L = lane_id();
      if (slot >= 0) {
        const int dim_base = L * 16;
        float e[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) e[i] = __bfloat162float(skv[dim_base + i]);
        if (dim_base >= 448) {
          const float ca[8] = {rc0.x, rc0.y, rc0.z, rc0.w, rc1.x, rc1.y, rc1.z, rc1.w};
          const float sa[8] = {rs0.x, rs0.y, rs0.z, rs0.w, rs1.x, rs1.y, rs1.z, rs1.w};
#pragma unroll
          for (int q2 = 0; q2 < 8; ++q2) {
            const float xe = e[2 * q2], xo = e[2 * q2 + 1];
            e[2 * q2] = xe * ca[q2] - xo * sa[q2];
            e[2 * q2 + 1] = xe * sa[q2] + xo * ca[q2];
          }
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) e[i] = bf16_round(e[i]);
        float lmax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) lmax = fmaxf(lmax, fabsf(e[i]));
        const float absmax = fmaxf(fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 1)), 1e-4f);
        const float exponent = ceilf(log2f(absmax / 448.0f));
        const float inv_scale = exp2f(-exponent);
        uint8_t ob[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          float sc = fminf(fmaxf(e[i] * inv_scale, -448.0f), 448.0f);
          ob[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(sc, __NV_SATFINITE, __NV_E4M3));
        }
        const long long blk = slot / p.swa_block_size, pib = slot % p.swa_block_size;
        uint8_t* bb = p.swa_cache + blk * p.swa_block_stride;
        *reinterpret_cast<uint4*>(bb + pib * 512 + dim_base) = *reinterpret_cast<const uint4*>(ob);
        if ((L & 1) == 0) {
          const float enc = fmaxf(fminf(exponent + 127.0f, 255.0f), 0.0f);
          bb[static_cast<long long>(p.swa_block_size) * 512 + pib * 16 + (L >> 1)] = static_cast<uint8_t>(enc);
        }
      }
    }
    compute_sync();
  }
};

}  // namespace w3
