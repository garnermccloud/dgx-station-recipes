// W4 mHC seam + next-coefficient stages for 16 < S (production's DeepGEMM Mega mHC, sm100_mega_mhc_impl<5120, 40,
// 152, true, true, false, 0>, shifted, bf16 output) for the W8 persistent kernel (CONTRACT.md v1). Bit-exact replica
// of src/csrc/mega.cu (tests/test_mega.py vs production) as cluster-free stages; tests/test_mega_stages.py.
//
//   MegaSeamPost  40 tasks (128-column splits j), all S tokens: [cold add] + post -> residual_out (fmul, then fma in
//                 input-route order), x1 = bf16(sum_o pv[o] * pre[o]) (fma from 0) -> scratch, and the split's
//                 partial of sum(x1^2) in DeepGEMM's Post-worker order -> scratch part[S][40].
//   MegaSeamNorm  40 tasks (same splits). Waits on all MegaSeamPost tasks. rs = rsqrtf(fma(sum of the 40 partials in
//                 split order, 1/5120, eps)) * scale; y = bf16((x1 * rs) * w) -> layer input + its quantized forms
//                 (fi_q / prep_* / W8's MXFP8 image, seam.cuh seam_chunk_store).
//   MegaMixGemm   40 * ceil(S/16) tasks (split s, 16-token tile): the split's tf32 GEMM in DeepGEMM's issue order
//                 (2 blocks x 8 K=8 atoms x 4 routes; hc_fn tf32-RNE image through 2 ring slots, pv = residual_out
//                 in 1 borrowed slot) and the split's sum of pv^2 -> scratch.
//   MegaMixStats  ceil(S/16) tasks: DeepGEMM's Mix epilogue (warp per 2 tokens, split sums, rsqrtf, __expf sigmoid,
//                 rcp.approx Sinkhorn) -> post / comb / pre.
// Scratch per width S: x1 [S][5120] bf16 = 10240 S B; part [S][40] f32; gemm_part [40][S][24] f32; hc_part [40][S].
#pragma once

#include <cstdint>
#include <cuda_bf16.h>

#include "w8/runtime.cuh"
#include "mhc_core.cuh"
#include "params.h"
#include "seam.cuh"
#include "stage_seam.cuh"

namespace w4::stage {

constexpr int kMgSplits = 40, kMgSplitW = 128, kMgOut = 24, kMgR = 4;
constexpr int kMgAStride = kMgSplitW + 4;                    // fp32 row stride of the mix operands (conflict-free)
constexpr int kMgBSlotBytes = 2 * kMgOut * kMgAStride * 4;   // 25,344: 2 routes of hc_fn per ring slot
constexpr int kMgASlotBytes = kMgR * 16 * kMgAStride * 4;    // 33,792: pv of a 16-token tile, 4 routes
constexpr float kMgInvHidden = 1.0f / 5120.0f;
constexpr float kMgInvHcHidden = 1.0f / (4.0f * 5120.0f);

__device__ __forceinline__ float mg_bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float mg_bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t mg_bf2(float a, float b) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ float mg_fast_rcp(float x) {
  float r;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// Post for 8 hidden elements h.. of token t from x values xf (bf16 values as floats): pv[o][e] (bf16-rounded, as
// floats), residual_out written. coef: post[4] | comb[16] (input route i, output o at i * 4 + o).
__device__ __forceinline__ void mega_post8f(const SeamParams& p, int t, int h, const float (&xf)[8], const float* coef,
                                            float (&pv)[kMgR][8]) {
  uint4 rv[kMgR];
#pragma unroll
  for (int i = 0; i < kMgR; ++i) rv[i] = __ldg(reinterpret_cast<const uint4*>(p.residual + ((size_t)t * kMgR + i) * kHidden + h));
#pragma unroll
  for (int o = 0; o < kMgR; ++o) {
    uint32_t ow[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float v0 = __fmul_rn(xf[2 * j], coef[o]), v1 = __fmul_rn(xf[2 * j + 1], coef[o]);
#pragma unroll
      for (int i = 0; i < kMgR; ++i) {
        const uint32_t rw = (&rv[i].x)[j];
        v0 = __fmaf_rn(mg_bf_lo(rw), coef[4 + i * kMgR + o], v0);
        v1 = __fmaf_rn(mg_bf_hi(rw), coef[4 + i * kMgR + o], v1);
      }
      ow[j] = mg_bf2(v0, v1);
      pv[o][2 * j] = mg_bf_lo(ow[j]);
      pv[o][2 * j + 1] = mg_bf_hi(ow[j]);
    }
    if (p.residual_out != nullptr)
      *reinterpret_cast<uint4*>(p.residual_out + ((size_t)t * kMgR + o) * kHidden + h) =
          make_uint4(ow[0], ow[1], ow[2], ow[3]);
  }
}

// ------------------------------------------------------------------------------------------------ MegaSeamPost
struct MegaSeamPost {
  struct Params {
    SeamParams seam;      // x, residual, post, comb, pre, residual_out, cold_* / p3_* (fused finish); T = S
    __nv_bfloat16* x1;    // scratch [S][5120]
    float* part;          // scratch [S][40]
    w8::Dep dep_x;        // producer tasks of x (flags), or {nullptr, 0, 0}
  };
  static constexpr const char* kName = "mega_seam_post";
  __host__ __device__ static constexpr int tasks(int) { return kMgSplits; }
  __host__ __device__ static constexpr uint32_t smem_bytes(int S) { return (uint32_t)S * (24 * 4 + kMgSplitW * 2) + 64; }
  __host__ __device__ static constexpr uint32_t tmem_cols(int) { return 0; }
  static size_t scratch_bytes(int S) { return (size_t)S * kHidden * 2 + (size_t)S * kMgSplits * 4; }

  template <class Sync, class X8 = SeamXBf16>
  __device__ static void consume(const Params& p, int S, int j, Sync& sync, uint8_t* smem, const X8& x8 = X8{}) {
    float* coef = reinterpret_cast<float*>(smem);                                 // [S][24]
    __nv_bfloat16* x1s = reinterpret_cast<__nv_bfloat16*>(smem + S * 24 * 4);     // [S][128]
    const int tid = threadIdx.x, lane = tid & 31, h0 = j * kMgSplitW;
    sync.wait_extern();
    if (p.dep_x.flags != nullptr) sync.wait(p.dep_x, 0x420);
    const bool cold_ok = p3_finish_wait(p.seam, S, j, smem + S * (24 * 4 + kMgSplitW * 2));
    for (int i = tid; i < S * 24; i += w8::kComputeThreads) {
      const int g = i / 24, k = i % 24;
      coef[i] = k < 4 ? p.seam.post[g * 4 + k] : k < 20 ? p.seam.comb[g * 16 + k - 4] : p.seam.pre[g * 4 + k - 20];
    }
    w8::compute_sync();
    // Phase A: post (residual_out) and x1, 8 hidden elements per item.
    for (int item = tid; item < S * (kMgSplitW / 8); item += w8::kComputeThreads) {
      const int g = item / (kMgSplitW / 8), c8 = item % (kMgSplitW / 8), h = h0 + c8 * 8;
      float xf[8];
      x8(p.seam, g, h, xf);
      if (cold_ok && p.seam.cold_has[g] != 0) {  // peer v3 finish: out rows are token-indexed host memory
        uint4 o;
        asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(o.x), "=r"(o.y), "=r"(o.z), "=r"(o.w)
                     : "l"(p.seam.p3_seg + kP3OffOut + ((size_t)g * kHidden + h) * 2));
        float cf[8];
        unpack8(o, cf);
#pragma unroll
        for (int e = 0; e < 8; ++e) xf[e] = bf2f(f2bf(__fadd_rn(xf[e], cf[e])));
        if (p.seam.p3_y != nullptr) *reinterpret_cast<uint4*>(p.seam.p3_y + (size_t)g * kHidden + h) = pack8(xf);
      }
      float pv[kMgR][8];
      mega_post8f(p.seam, g, h, xf, coef + g * 24, pv);
      uint32_t xw[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int o = 0; o < kMgR; ++o) {
          a0 = __fmaf_rn(pv[o][2 * q], coef[g * 24 + 20 + o], a0);
          a1 = __fmaf_rn(pv[o][2 * q + 1], coef[g * 24 + 20 + o], a1);
        }
        xw[q] = mg_bf2(a0, a1);
      }
      const uint4 xv = make_uint4(xw[0], xw[1], xw[2], xw[3]);
      *reinterpret_cast<uint4*>(x1s + g * kMgSplitW + c8 * 8) = xv;
      *reinterpret_cast<uint4*>(p.x1 + (size_t)g * kHidden + h) = xv;
    }
    w8::compute_sync();
    // Phase B: the split's partial of sum(x1^2) in DeepGEMM's order. Item (g, w, cp) = 8 g + 4 w + cp.
    for (int base = 0; base < S * 8; base += w8::kComputeThreads) {
      const int item = base + tid;
      const int cp = item & 3, w = (item >> 2) & 1, g = item >> 3;
      const bool active = g < S;
      float ax = 0.f, ay = 0.f;
      if (active) {
#pragma unroll
        for (int kb = 0; kb < 2; ++kb)
#pragma unroll
          for (int uu = 0; uu < 4; ++uu) {
            const int hl = kb * 64 + (w + 2 * uu) * 8 + 2 * cp;
            const uint32_t xx = *reinterpret_cast<const uint32_t*>(x1s + g * kMgSplitW + hl);
            ax = __fmaf_rn(mg_bf_lo(xx), mg_bf_lo(xx), ax);
            ay = __fmaf_rn(mg_bf_hi(xx), mg_bf_hi(xx), ay);
          }
      }
      float v = __fadd_rn(ax, ay);
      v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
      v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
      const float other = __shfl_xor_sync(0xffffffffu, v, 4);
      const float s = __fadd_rn(__fadd_rn(0.f, w == 0 ? v : other), w == 0 ? other : v);
      if (active && (lane & 7) == 0) p.part[g * kMgSplits + j] = s;
    }
  }
};

// ------------------------------------------------------------------------------------------------ MegaSeamNorm
struct MegaSeamNorm {
  struct Params {
    SeamParams seam;           // norm_w, layer_input, fi_*, prep_*; T = S
    const __nv_bfloat16* x1;   // MegaSeamPost scratch
    const float* part;
    float rms_eps, rms_scale;  // production: norm_eps, 1.0
    w8::Dep dep_post;          // all 40 MegaSeamPost tasks
    MxImage img{};             // W8's SW128 MXFP8 image of the layer input, or q == null
  };
  static constexpr const char* kName = "mega_seam_norm";
  __host__ __device__ static constexpr int tasks(int) { return kMgSplits; }
  __host__ __device__ static constexpr uint32_t smem_bytes(int S) { return (uint32_t)S * 4 + 64; }
  __host__ __device__ static constexpr uint32_t tmem_cols(int) { return 0; }
  static size_t scratch_bytes(int) { return 0; }

  template <class Sync>
  __device__ static void consume(const Params& p, int S, int j, Sync& sync, uint8_t* smem) {
    float* rs = reinterpret_cast<float*>(smem);
    const int tid = threadIdx.x, h0 = j * kMgSplitW;
    sync.wait(p.dep_post, 0x421);
    for (int g = tid; g < S; g += w8::kComputeThreads) {
      float v[kMgSplits];
#pragma unroll
      for (int s = 0; s < kMgSplits; ++s) v[s] = __ldcg(p.part + g * kMgSplits + s);
      float sum = 0.f;
#pragma unroll
      for (int s = 0; s < kMgSplits; ++s) sum = __fadd_rn(sum, v[s]);
      rs[g] = __fmul_rn(rsqrtf(__fmaf_rn(sum, kMgInvHidden, p.rms_eps)), p.rms_scale);
    }
    w8::compute_sync();
    // y = bf16((x1 * rs) * w), 8 hidden elements per item; 4 consecutive lanes = one 32-column MXFP8 block
    for (int base = 0; base < S * (kMgSplitW / 8); base += w8::kComputeThreads) {
      const int item = base + tid;
      const bool active = item < S * (kMgSplitW / 8);
      const int g = active ? item / (kMgSplitW / 8) : 0, c8 = item % (kMgSplitW / 8), h = h0 + c8 * 8;
      const uint4 xv = __ldcg(reinterpret_cast<const uint4*>(p.x1 + (size_t)g * kHidden + h));
      const uint4 wv = ld_nc_v4(p.seam.norm_w + h);
      float y[8];
      const float r = rs[g];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t a = (&xv.x)[q], b = (&wv.x)[q];
        const uint32_t yw = mg_bf2(__fmul_rn(__fmul_rn(mg_bf_lo(a), r), mg_bf_lo(b)),
                                   __fmul_rn(__fmul_rn(mg_bf_hi(a), r), mg_bf_hi(b)));
        y[2 * q] = mg_bf_lo(yw), y[2 * q + 1] = mg_bf_hi(yw);
      }
      seam_chunk_store(p.seam, g, h, y, active, p.img);
    }
    if (p.img.q != nullptr)  // the image's padding rows [S, npad) of this split: zero values and scales
      for (int item = tid; item < (p.img.npad - S) * (kMgSplitW / 8); item += w8::kComputeThreads) {
        const int t = S + item / (kMgSplitW / 8), col = h0 + (item % (kMgSplitW / 8)) * 8;
        *reinterpret_cast<uint2*>(p.img.q + mx_image_off(t, col, p.img.npad)) = make_uint2(0, 0);
        if ((col & 31) == 0) p.img.sf[mx_image_sf_off(t, col)] = 0;
      }
    if (j == 0) seam_zero_sf_padding(p.seam, 0, 1, tid, w8::kComputeThreads);
  }
};

// ------------------------------------------------------------------------------------------------ MegaMixGemm
struct MegaMixGemm {
  struct Params {
    const __nv_bfloat16* rn;  // residual_out [S][4][5120] (MegaSeamPost)
    const float* fn_img;      // w4b.boundary.mega_fn_image: [40 splits][4 routes][24][132] tf32-RNE hc_fn
    float* gemm_part;         // scratch [40][S][24]
    float* hc_part;           // scratch [40][S]
    w8::Dep dep_post;         // all 40 MegaSeamPost tasks
  };
  static constexpr const char* kName = "mega_mix_gemm";
  // task = mt * 40 + split; ring slots in order B0 (routes 0, 1), A (borrowed, 0 tx), B1 (routes 2, 3)
  __host__ __device__ static constexpr int tasks(int S) { return kMgSplits * ((S + 15) / 16); }
  __host__ __device__ static constexpr uint32_t smem_bytes(int) { return 0; }
  __host__ __device__ static constexpr uint32_t tmem_cols(int) { return 0; }
  static size_t scratch_bytes(int S) { return (size_t)kMgSplits * S * (kMgOut + 1) * 4; }

  template <class RingT>
  __device__ static void produce(const Params& p, int, int task, const RingT& ring, uint32_t& prod_it, w8::Diag* diag) {
    static_assert(RingT::SLOT_BYTES >= kMgASlotBytes, "MegaMixGemm needs ring slots of >= 33,792 B");
    const int s = task % kMgSplits;
    for (int q = 0; q < 3; ++q, ++prod_it) {
      uint8_t* slot = ring.acquire(prod_it, diag);
      if (q == 1) {
        ring.submit(prod_it, 0);
        continue;
      }
      const int h = q == 0 ? 0 : 1;
      w8::bulk_g2s(slot, p.fn_img + ((size_t)s * kMgR + 2 * h) * kMgOut * kMgAStride, kMgBSlotBytes,
                   ring.full_bar(prod_it));
      ring.submit(prod_it, kMgBSlotBytes);
    }
  }

  template <class Sync, class RingT>
  __device__ static void consume(const Params& p, int S, int task, Sync& sync, const RingT& ring, uint32_t& ring_it,
                                 uint8_t*, w8::Diag* diag) {
    const int s = task % kMgSplits, mt = task / kMgSplits, t0 = mt * 16, tn = min(16, S - t0);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, hs = s * kMgSplitW;
    float* A = reinterpret_cast<float*>(ring.wait_full(ring_it + 1, diag));  // [route][16][kMgAStride]
    sync.wait(p.dep_post, 0x422);
    for (int item = tid; item < kMgR * 16 * (kMgSplitW / 8); item += w8::kComputeThreads) {
      const int o = item / (16 * (kMgSplitW / 8)), r = (item / (kMgSplitW / 8)) % 16, c8 = item % (kMgSplitW / 8);
      float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      if (r < tn) unpack8(__ldcg(reinterpret_cast<const uint4*>(p.rn + ((size_t)(t0 + r) * kMgR + o) * kHidden + hs + c8 * 8)), v);
      float* dst = A + (o * 16 + r) * kMgAStride + c8 * 8;
      *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
      *reinterpret_cast<float4*>(dst + 4) = make_float4(v[4], v[5], v[6], v[7]);
    }
    w8::compute_sync();
    if (warp < 3) {
      // the split's GEMM, n-atom `warp`: (block kb, atom u, route o) in DeepGEMM's issue order
      const float* B0 = reinterpret_cast<const float*>(ring.wait_full(ring_it + 0, diag));  // routes 0, 1
      const float* B1 = reinterpret_cast<const float*>(ring.wait_full(ring_it + 2, diag));  // routes 2, 3
      const int g = lane >> 2, tig = lane & 3, na = warp;
      float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kb = 0; kb < 2; ++kb)
#pragma unroll
        for (int u = 0; u < 8; ++u)
#pragma unroll
          for (int o = 0; o < kMgR; ++o) {
            const int k0 = kb * 64 + u * 8;
            const float* a = A + (o * 16) * kMgAStride + k0;
            const float* b = (o < 2 ? B0 : B1) + ((o & 1) * kMgOut + na * 8 + g) * kMgAStride + k0;
            mma_tf32_m16n8k8(c, make_float4(a[g * kMgAStride + tig], a[(g + 8) * kMgAStride + tig],
                                            a[g * kMgAStride + tig + 4], a[(g + 8) * kMgAStride + tig + 4]),
                             make_float2(b[tig], b[tig + 4]));
          }
      const int col = na * 8 + 2 * tig;
      if (g < tn)
        *reinterpret_cast<float2*>(p.gemm_part + ((size_t)s * S + t0 + g) * kMgOut + col) = make_float2(c[0], c[1]);
      if (g + 8 < tn)
        *reinterpret_cast<float2*>(p.gemm_part + ((size_t)s * S + t0 + g + 8) * kMgOut + col) = make_float2(c[2], c[3]);
    } else if (warp < 7) {
      // the split's sum of pv^2 as the two Post workers accumulate it; item (row, worker, cp) = 8 row + 4 w + cp
      const int item = tid - 96, cp = item & 3, w = (item >> 2) & 1, r = item >> 3;
      float ax = 0.f, ay = 0.f;
#pragma unroll
      for (int kb = 0; kb < 2; ++kb)
#pragma unroll
        for (int uu = 0; uu < 4; ++uu)
#pragma unroll
          for (int o = 0; o < kMgR; ++o) {
            const float* a = A + (o * 16 + r) * kMgAStride + kb * 64 + (w + 2 * uu) * 8 + 2 * cp;
            ax = __fmaf_rn(a[0], a[0], ax);
            ay = __fmaf_rn(a[1], a[1], ay);
          }
      float v = __fadd_rn(ax, ay);
      v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
      v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
      const float other = __shfl_xor_sync(0xffffffffu, v, 4);
      const float sum = __fadd_rn(__fadd_rn(0.f, w == 0 ? v : other), w == 0 ? other : v);
      if ((item & 7) == 0 && r < tn) p.hc_part[(size_t)s * S + t0 + r] = sum;
    }
    w8::fence_proxy_async_smem();  // A was written through the generic proxy; its slot may next take a TMA write
    w8::compute_sync();
    if (tid == 0) ring.release(ring_it), ring.release(ring_it + 1), ring.release(ring_it + 2);
    ring_it += 3;
  }
};

// ------------------------------------------------------------------------------------------------ MegaMixStats
struct MegaMixStats {
  struct Params {
    const float* gemm_part;  // [40][S][24]
    const float* hc_part;    // [40][S]
    const float* scale;      // [3]
    const float* base;       // [24]
    float* post_out;         // [S][4]
    float* comb_out;         // [S][16]
    float* pre_out;          // [S][4]
    float hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps;
    int sinkhorn_iters;
    w8::Dep dep_gemm;        // MegaMixGemm tasks of all tiles (task mt * 40 + split)
  };
  static constexpr const char* kName = "mega_mix_stats";
  __host__ __device__ static constexpr int tasks(int S) { return (S + 15) / 16; }
  __host__ __device__ static constexpr uint32_t smem_bytes(int) { return 0; }
  __host__ __device__ static constexpr uint32_t tmem_cols(int) { return 0; }
  static size_t scratch_bytes(int) { return 0; }

  // DeepGEMM epilogue::mega_mhc::run_mix_task (shifted), one warp, two tokens interleaved (mega.cu mega_mix_tokens2)
  __device__ static void tokens2(const Params& p, int S, const int (&t)[2], const bool (&valid)[2], int lane) {
    constexpr int kFirstCombLane = 2 * kMgR;
    const bool is_hc_output_lane = lane < kMgOut;
    const bool is_pre_lane = lane < kMgR;
    const bool is_pre_or_post_lane = lane < kFirstCombLane;
    const bool is_comb_lane = is_hc_output_lane && !is_pre_or_post_lane;
    const int safe = lane < kMgOut - 1 ? lane : kMgOut - 1;
    float gp[2][kMgSplits], hp0[2], hp1[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int s = 0; s < kMgSplits; ++s) gp[j][s] = __ldcg(p.gemm_part + ((size_t)s * S + t[j]) * kMgOut + safe);
      hp0[j] = __ldcg(p.hc_part + (size_t)lane * S + t[j]);
      hp1[j] = lane + 32 < kMgSplits ? __ldcg(p.hc_part + (size_t)(lane + 32) * S + t[j]) : 0.f;
    }
    float mix_value[2], value[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float hc_norm_sqr_sum = 0.0f;
      hc_norm_sqr_sum += hp0[j];
      if (lane + 32 < kMgSplits) hc_norm_sqr_sum += hp1[j];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        hc_norm_sqr_sum = hc_norm_sqr_sum + __shfl_xor_sync(0xffffffffu, hc_norm_sqr_sum, off);
      float hc_output = 0.0f;
#pragma unroll
      for (int s = 0; s < kMgSplits; ++s) hc_output += gp[j][s];
      hc_output = is_hc_output_lane ? hc_output : 0.0f;
      const float hc_rms_scale = rsqrtf(hc_norm_sqr_sum * kMgInvHcHidden + p.hc_norm_eps);
      mix_value[j] = 0.0f;
      if (is_hc_output_lane) {
        const int mix_scale_idx = is_pre_lane ? 0 : is_pre_or_post_lane ? 1 : 2;
        const float affine_value = hc_output * hc_rms_scale * p.scale[mix_scale_idx] + p.base[lane];
        if (is_pre_or_post_lane) {
          const float sigmoid = mg_fast_rcp(1.0f + __expf(-affine_value));
          mix_value[j] = is_pre_lane ? sigmoid + p.hc_pre_eps : sigmoid * p.hc_post_scale;
        } else {
          mix_value[j] = affine_value;
        }
      }
    }
    __syncwarp();
    auto wsum4 = [](float v) {
      v = v + __shfl_xor_sync(0xffffffffu, v, 2);
      return v + __shfl_xor_sync(0xffffffffu, v, 1);
    };
    auto wmax4 = [](float v) {
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
      return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    };
    auto col_sum = [](float e) {
      float s = e + __shfl_xor_sync(0xffffffffu, e, kMgR);
      s += __shfl_xor_sync(0xffffffffu, s, 6 * kMgR);
      return s;
    };
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      value[j] = is_comb_lane ? mix_value[j] : 0.0f;
      const float row_max = wmax4(value[j]);
      value[j] = __expf(value[j] - row_max);
      const float row_sum = wsum4(value[j]);
      value[j] = value[j] * mg_fast_rcp(row_sum) + p.sinkhorn_eps;
      value[j] *= mg_fast_rcp(col_sum(value[j]) + p.sinkhorn_eps);
    }
#pragma unroll 1
    for (int it = 1; it < p.sinkhorn_iters; ++it) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float row_sum = wsum4(value[j]);
        value[j] *= mg_fast_rcp(row_sum + p.sinkhorn_eps);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) value[j] *= mg_fast_rcp(col_sum(value[j]) + p.sinkhorn_eps);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      if (!valid[j] || !is_hc_output_lane) continue;
      if (is_pre_lane) p.pre_out[t[j] * kMgR + lane] = mix_value[j];
      else if (is_comb_lane) p.comb_out[t[j] * kMgR * kMgR + lane - kFirstCombLane] = value[j];
      else p.post_out[t[j] * kMgR + (lane & (kMgR - 1))] = mix_value[j];
    }
  }

  template <class Sync>
  __device__ static void consume(const Params& p, int S, int mt, Sync& sync, uint8_t*) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t0 = mt * 16, tn = min(16, S - t0);
    sync.wait(w8::Dep{p.dep_gemm.flags, mt * kMgSplits, mt * kMgSplits + kMgSplits}, 0x423);
    const int half = (tn + 1) / 2;
    if (warp < half) {  // warp w: tokens w and w + half of the tile
      const int r1 = warp + half;
      const int tt[2] = {t0 + warp, t0 + (r1 < tn ? r1 : warp)};
      const bool ok[2] = {true, r1 < tn};
      tokens2(p, S, tt, ok, lane);
    }
  }
};

}  // namespace w4::stage
