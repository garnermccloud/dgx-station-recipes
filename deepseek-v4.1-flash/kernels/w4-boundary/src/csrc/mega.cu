// W4 mega boundary (T > 16): bit-exact replica of DeepGEMM's sm100_mega_mhc_impl<5120, 40, 152, true, true, false, 0>,
// the kernel production's mhc_shifted_post_pre runs for 16 < T (Mega mHC, shifted, bf16 output), as two launches:
//
//   mega_seam (main stream): residual_new = bf16(x * post[o] + sum_i r[i] * comb[i][o]) (fmul, then fma in i order),
//     x1 = bf16(sum_o pv[o] * pre[o]) (fma from 0 in o order), and y = bf16((x1 * rms) * w) with
//     rms = rsqrtf(S * (1/5120) + eps) * scale. S reproduces DeepGEMM's order: per 128-wide K split, per Post worker
//     (atoms u = w, w + 2, .. of each of the split's two 64-wide blocks) and per column pair cp, two chains (pair
//     halves) of fma(x1, x1, acc); then acc.x + acc.y, a butterfly over cp (xor 2, xor 1), (0 + W0) + W1 per split,
//     and the 40 split partials summed in split order. A cluster of 8 CTAs owns 4 tokens; CTA r owns splits
//     [5r, 5r + 5) and pushes its partials to every CTA of the cluster (st.async + mbarrier).
//   mega_mix (side stream): per (split, 16-token tile) the split's tf32 GEMM as DeepGEMM issues it (2 blocks x 8 K=8
//     atoms x 4 routes, accumulated in that order; mma.sync m16n8k8 on hc_fn rounded to tf32 RNE, as tcgen05 sees it),
//     the split's sum of pv^2 in the Post workers' order, then the last CTA of a tile runs DeepGEMM's Mix epilogue
//     (warp per token, split sums, rsqrtf, __expf sigmoid, rcp.approx Sinkhorn) verbatim.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "w4/params.h"

namespace {

using w4::MegaParams;

constexpr int kH = 5120, kR = 4, kSplits = 40, kSplitW = 128, kOut = 24;
constexpr int kCS = 8;                         // seam cluster
constexpr int kSplitsPerCta = kSplits / kCS;   // 5
constexpr int kHPerCta = kH / kCS;             // 640
constexpr int kG = 4;                          // tokens per seam cluster
constexpr int kThreads = 256;
constexpr float kInvHidden = 1.0f / static_cast<float>(kH);
constexpr float kInvHcHidden = 1.0f / static_cast<float>(kR * kH);

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// debug stamps of mega_mix: CTA (0, 0) phases 0-4, the last arriver of tile 0 phases 5-6
#define MSTAMP(i) \
  if (p.dbg != nullptr && threadIdx.x == 0 && blockIdx.y == 0 && ((i) >= 5 || blockIdx.x == 0)) p.dbg[i] = gtimer();
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t bf2(float a, float b) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}
// The tensor cores see hc_fn rounded to tf32 with round-to-nearest-even (verified against Mega mHC); mma.sync would
// truncate the raw fp32 bits, so B is rounded when it is staged.
__device__ __forceinline__ float tf32_rne(float f) {
  const uint32_t u = __float_as_uint(f);
  return __uint_as_float((u + 0xfffu + ((u >> 13) & 1u)) & 0xffffe000u);
}
__device__ __forceinline__ float fast_rcp(float x) {
  float r;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// Post for 8 hidden elements of one token: pv[o][e] (bf16-rounded, as float); writes residual_new.
__device__ __forceinline__ void mega_post8(const MegaParams& p, int t, int h, const float* post, const float* comb,
                                           float (&pv)[kR][8]) {
  const uint4 xv = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)t * kH + h));
  uint4 rv[kR];
#pragma unroll
  for (int i = 0; i < kR; ++i) rv[i] = __ldg(reinterpret_cast<const uint4*>(p.residual + ((size_t)t * kR + i) * kH + h));
  const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
  for (int o = 0; o < kR; ++o) {
    uint32_t ow[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float v0 = __fmul_rn(bf_lo(xw[j]), post[o]), v1 = __fmul_rn(bf_hi(xw[j]), post[o]);
#pragma unroll
      for (int i = 0; i < kR; ++i) {
        const uint32_t rw = (&rv[i].x)[j];
        v0 = __fmaf_rn(bf_lo(rw), comb[i * kR + o], v0);
        v1 = __fmaf_rn(bf_hi(rw), comb[i * kR + o], v1);
      }
      ow[j] = bf2(v0, v1);
      pv[o][2 * j] = bf_lo(ow[j]);
      pv[o][2 * j + 1] = bf_hi(ow[j]);
    }
    if (p.residual_out != nullptr)
      *reinterpret_cast<uint4*>(p.residual_out + ((size_t)t * kR + o) * kH + h) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
  }
}

// ------------------------------------------------------------------------------------------------ mega_seam
template <int G>  // tokens per cluster: 4, or 2 at T <= 16 (every phase-A item in one round: decode, the DSpark drafter)
__global__ void __launch_bounds__(kThreads, 1) mega_seam_kernel(const MegaParams p) {
  __shared__ __align__(16) __nv_bfloat16 x1_sh[G][kHPerCta];
  __shared__ __align__(16) __nv_bfloat16 w_sh[kHPerCta];
  __shared__ float coef[G][24];  // post 4 | comb 16 | pre 4
  __shared__ float part[G][kSplits];
  __shared__ __align__(8) uint64_t bar;
  const uint32_t rank = cluster_rank();
  const int t0 = (blockIdx.x / kCS) * G, tg = min(G, p.T - t0);
  const int tid = threadIdx.x, lane = tid & 31;
  const int h0 = (int)rank * kHPerCta;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar)));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bar)), "r"(tg * kSplits * 4));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  if (p.l2_prefetch != nullptr && tid < 4) {  // this CTA's slice of the next kernel's weights, in 4 bulk prefetches
    const long long n = (long long)gridDim.x * 4, slice = ((p.l2_prefetch_bytes + n - 1) / n + 15) / 16 * 16;
    const long long off = ((long long)blockIdx.x * 4 + tid) * slice;
    const long long len = min(slice, p.l2_prefetch_bytes - off);
    if (len > 0)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.l2_prefetch + off), "r"((uint32_t)len) : "memory");
  }
  for (int i = tid; i < kHPerCta / 8; i += kThreads)
    *reinterpret_cast<uint4*>(&w_sh[i * 8]) = __ldg(reinterpret_cast<const uint4*>(p.norm_w + h0 + i * 8));
  pdl_launch();
  pdl_wait();
  if (tid < G * 24) {
    const int g = tid / 24, j = tid % 24;
    float v = 0.f;
    if (g < tg) v = j < 4 ? p.post[(t0 + g) * 4 + j] : j < 20 ? p.comb[(t0 + g) * 16 + j - 4] : p.pre[(t0 + g) * 4 + j - 20];
    coef[g][j] = v;
  }
  __syncthreads();
  // Phase A: post (residual_new) and x1, 8 hidden elements per item.
  for (int item = tid; item < tg * (kHPerCta / 8); item += kThreads) {
    const int g = item / (kHPerCta / 8), c8 = item % (kHPerCta / 8);
    float pv[kR][8];
    mega_post8(p, t0 + g, h0 + c8 * 8, &coef[g][0], &coef[g][4], pv);
    uint32_t xw[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float a0 = 0.f, a1 = 0.f;
#pragma unroll
      for (int o = 0; o < kR; ++o) {
        a0 = __fmaf_rn(pv[o][2 * j], coef[g][20 + o], a0);
        a1 = __fmaf_rn(pv[o][2 * j + 1], coef[g][20 + o], a1);
      }
      xw[j] = bf2(a0, a1);
    }
    *reinterpret_cast<uint4*>(&x1_sh[g][c8 * 8]) = make_uint4(xw[0], xw[1], xw[2], xw[3]);
  }
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  // Phase B: split partials of sum(x1^2) in DeepGEMM's order. Item (g, j, w, cp): lane = 8 * (..) + 4 w + cp.
  for (int base = 0; base < G * kSplitsPerCta * 8; base += kThreads) {
    const int item = base + tid;
    const int cp = item & 3, w = (item >> 2) & 1, gj = item >> 3, g = gj / kSplitsPerCta, j = gj % kSplitsPerCta;
    const bool active = g < tg;
    float ax = 0.f, ay = 0.f;
    if (active) {
#pragma unroll
      for (int kb = 0; kb < 2; ++kb)
#pragma unroll
        for (int uu = 0; uu < 4; ++uu) {
          const int hl = j * kSplitW + kb * 64 + (w + 2 * uu) * 8 + 2 * cp;
          const uint32_t xx = *reinterpret_cast<const uint32_t*>(&x1_sh[g][hl]);
          ax = __fmaf_rn(bf_lo(xx), bf_lo(xx), ax);
          ay = __fmaf_rn(bf_hi(xx), bf_hi(xx), ay);
        }
    }
    float v = __fadd_rn(ax, ay);
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
    const float other = __shfl_xor_sync(0xffffffffu, v, 4);
    const float s = __fadd_rn(__fadd_rn(0.f, w == 0 ? v : other), w == 0 ? other : v);
    if (active && (lane & 7) == 0) {
      const uint32_t la = smem_u32(&part[g][(int)rank * kSplitsPerCta + j]), lb = smem_u32(&bar);
#pragma unroll
      for (int r = 0; r < kCS; ++r) {
        uint32_t ra, rb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(la), "r"(r));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(lb), "r"(r));
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(ra),
                     "r"(__float_as_uint(s)), "r"(rb)
                     : "memory");
      }
    }
  }
  asm volatile(
      "{\n .reg .pred p;\n W4G_WAIT%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
      " @!p bra W4G_WAIT%=;\n}\n" ::"r"(smem_u32(&bar))
      : "memory");
  // Phase C: y = bf16((x1 * rms) * w).
  for (int item = tid; item < tg * (kHPerCta / 8); item += kThreads) {
    const int g = item / (kHPerCta / 8), c8 = item % (kHPerCta / 8);
    float S = 0.f;
#pragma unroll 8
    for (int s = 0; s < kSplits; ++s) S = __fadd_rn(S, part[g][s]);
    const float rs = __fmul_rn(rsqrtf(__fmaf_rn(S, kInvHidden, p.rms_eps)), p.rms_scale);
    const uint4 xv = *reinterpret_cast<const uint4*>(&x1_sh[g][c8 * 8]);
    const uint4 wv = *reinterpret_cast<const uint4*>(&w_sh[c8 * 8]);
    uint32_t yw[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t a = (&xv.x)[j], b = (&wv.x)[j];
      yw[j] = bf2(__fmul_rn(__fmul_rn(bf_lo(a), rs), bf_lo(b)), __fmul_rn(__fmul_rn(bf_hi(a), rs), bf_hi(b)));
    }
    *reinterpret_cast<uint4*>(p.y_out + (size_t)(t0 + g) * kH + h0 + c8 * 8) = make_uint4(yw[0], yw[1], yw[2], yw[3]);
  }
}

// ------------------------------------------------------------------------------------------------ mega_mix
constexpr int kAStride = kSplitW + 4;  // fp32 row stride in smem: conflict-free fragment loads
constexpr int kMixSmem = (kR * 16 * kAStride + kR * kOut * kAStride) * 4;

__device__ __forceinline__ void mma_tf32(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float wsum4(float v) {  // DeepGEMM warp_reduce_sum<4>
  v = v + __shfl_xor_sync(0xffffffffu, v, 2);
  return v + __shfl_xor_sync(0xffffffffu, v, 1);
}
__device__ __forceinline__ float wmax4(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
}

// DeepGEMM epilogue::mega_mhc::run_mix_task (shifted) from the split partials: one warp, two tokens interleaved
// (every operation is per token; the two chains only share issue slots). Token slots with valid == false compute
// on a clamped token and store nothing.
// The comb mixes go to cmix[t % 16][16] for mega_sinkhorn.
__device__ void mega_mix_tokens2(const MegaParams& p, const int (&t)[2], const bool (&valid)[2], int lane,
                                 float* cmix) {
  constexpr int kFirstCombLane = 2 * kR;
  const bool is_hc_output_lane = lane < kOut;
  const bool is_pre_lane = lane < kR;
  const bool is_pre_or_post_lane = lane < kFirstCombLane;
  const bool is_comb_lane = is_hc_output_lane && !is_pre_or_post_lane;
  const int safe = lane < kOut - 1 ? lane : kOut - 1;
  float gp[2][kSplits], hp0[2], hp1[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
#pragma unroll
    for (int s = 0; s < kSplits; ++s) gp[j][s] = __ldcg(p.gemm_part + ((size_t)s * p.T + t[j]) * kOut + safe);
    hp0[j] = __ldcg(p.hc_part + (size_t)lane * p.T + t[j]);
    hp1[j] = lane + 32 < kSplits ? __ldcg(p.hc_part + (size_t)(lane + 32) * p.T + t[j]) : 0.f;
  }
  float mix_value[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    float hc_norm_sqr_sum = 0.0f;
    hc_norm_sqr_sum += hp0[j];
    if (lane + 32 < kSplits) hc_norm_sqr_sum += hp1[j];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      hc_norm_sqr_sum = hc_norm_sqr_sum + __shfl_xor_sync(0xffffffffu, hc_norm_sqr_sum, off);
    float hc_output = 0.0f;
#pragma unroll
    for (int s = 0; s < kSplits; ++s) hc_output += gp[j][s];
    hc_output = is_hc_output_lane ? hc_output : 0.0f;
    const float hc_rms_scale = rsqrtf(hc_norm_sqr_sum * kInvHcHidden + p.hc_norm_eps);
    mix_value[j] = 0.0f;
    if (is_hc_output_lane) {
      const int mix_scale_idx = is_pre_lane ? 0 : is_pre_or_post_lane ? 1 : 2;
      const float affine_value = hc_output * hc_rms_scale * p.scale[mix_scale_idx] + p.base[lane];
      if (is_pre_or_post_lane) {
        const float sigmoid = fast_rcp(1.0f + __expf(-affine_value));
        mix_value[j] = is_pre_lane ? sigmoid + p.hc_pre_eps : sigmoid * p.hc_post_scale;
      } else {
        mix_value[j] = affine_value;
      }
    }
  }
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    if (!valid[j] || !is_hc_output_lane) continue;
    if (is_pre_lane) p.pre_out[t[j] * kR + lane] = mix_value[j];
    else if (is_comb_lane) cmix[(t[j] % 16) * kR * kR + lane - kFirstCombLane] = mix_value[j];
    else p.post_out[t[j] * kR + (lane & (kR - 1))] = mix_value[j];
  }
}

// DeepGEMM's Sinkhorn (run_mix_task) of one token's comb mix, in one thread. DeepGEMM runs it on the 16 comb lanes
// (lane 8 + 4 r + c), where every lane's row butterfly (xor 2, then 1) yields (v[r][0] + v[r][2]) + (v[r][1] + v[r][3])
// and its column butterfly (xor 4, then 24) (v[0][c] + v[1][c]) + (v[2][c] + v[3][c]) (fp addition commutes): the
// same values without the shuffle latency, one token per lane.
__device__ __forceinline__ void mega_sinkhorn(float (&m)[kR * kR], float eps, int iters) {
  auto row_sum = [&](int r) { return (m[4 * r] + m[4 * r + 2]) + (m[4 * r + 1] + m[4 * r + 3]); };
  auto col_norm = [&]() {
#pragma unroll
    for (int c = 0; c < kR; ++c) {
      const float inv = fast_rcp(((m[c] + m[4 + c]) + (m[8 + c] + m[12 + c])) + eps);
#pragma unroll
      for (int r = 0; r < kR; ++r) m[4 * r + c] *= inv;
    }
  };
#pragma unroll
  for (int r = 0; r < kR; ++r) {
    const float row_max = fmaxf(fmaxf(m[4 * r], m[4 * r + 2]), fmaxf(m[4 * r + 1], m[4 * r + 3]));
#pragma unroll
    for (int c = 0; c < kR; ++c) m[4 * r + c] = __expf(m[4 * r + c] - row_max);
    const float inv = fast_rcp(row_sum(r));
#pragma unroll
    for (int c = 0; c < kR; ++c) m[4 * r + c] = m[4 * r + c] * inv + eps;
  }
  col_norm();
#pragma unroll 1
  for (int it = 1; it < iters; ++it) {
#pragma unroll
    for (int r = 0; r < kR; ++r) {
      const float inv = fast_rcp(row_sum(r) + eps);
#pragma unroll
      for (int c = 0; c < kR; ++c) m[4 * r + c] *= inv;
    }
    col_norm();
  }
}

__global__ void __launch_bounds__(kThreads, 2) mega_mix_kernel(const MegaParams p) {
  extern __shared__ __align__(16) float msm[];
  float* A = msm;                          // [route][16 rows][kAStride] pv (bf16-exact fp32)
  float* B = msm + kR * 16 * kAStride;     // [route][24 n][kAStride] fn (raw fp32)
  __shared__ float coef[16][20];
  __shared__ float cmix[16][kR * kR];  // the tile's comb mixes, for the per-lane Sinkhorn
  __shared__ int last;
  const int s = blockIdx.x, mt = blockIdx.y, t0 = mt * 16, tn = min(16, p.T - t0);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int hs = s * kSplitW;
  MSTAMP(0);
  // B (weights) before the dependency wait: fn[n][route * 5120 + hs + k], 24 x 4 rows of 128 floats.
  {
    constexpr int kBIter = kR * kOut * (kSplitW / 4) / kThreads;  // 12
    float4 f[kBIter];
#pragma unroll
    for (int j = 0; j < kBIter; ++j) {
      const int i = tid + j * kThreads, row = i / (kSplitW / 4), c4 = i % (kSplitW / 4), r = row / kOut, n = row % kOut;
      f[j] = __ldg(reinterpret_cast<const float4*>(p.fn + (size_t)n * (kR * kH) + r * kH + hs + c4 * 4));
    }
#pragma unroll
    for (int j = 0; j < kBIter; ++j) {
      const int i = tid + j * kThreads, row = i / (kSplitW / 4), c4 = i % (kSplitW / 4), r = row / kOut, n = row % kOut;
      *reinterpret_cast<float4*>(&B[(r * kOut + n) * kAStride + c4 * 4]) =
          make_float4(tf32_rne(f[j].x), tf32_rne(f[j].y), tf32_rne(f[j].z), tf32_rne(f[j].w));
    }
  }
  // no early launch_dependents: the side branch's last kernel releases a joined PDL kernel only at its exit
  pdl_wait();
  MSTAMP(1);
  for (int i = tid; i < 16 * 20; i += kThreads) {
    const int g = i / 20, j = i % 20;
    coef[g][j] = g < tn ? (j < 4 ? p.post[(t0 + g) * 4 + j] : p.comb[(t0 + g) * 16 + j - 4]) : 0.f;
  }
  __syncthreads();
  for (int item = tid; item < 16 * (kSplitW / 8); item += kThreads) {
    const int r = item / (kSplitW / 8), c8 = item % (kSplitW / 8);
    float pv[kR][8];
    if (r < tn) {
      MegaParams q = p;
      q.residual_out = nullptr;
      mega_post8(q, t0 + r, hs + c8 * 8, &coef[r][0], &coef[r][4], pv);
    } else {
#pragma unroll
      for (int o = 0; o < kR; ++o)
#pragma unroll
        for (int e = 0; e < 8; ++e) pv[o][e] = 0.f;
    }
#pragma unroll
    for (int o = 0; o < kR; ++o) {
      float* dst = &A[(o * 16 + r) * kAStride + c8 * 8];
      *reinterpret_cast<float4*>(dst) = make_float4(pv[o][0], pv[o][1], pv[o][2], pv[o][3]);
      *reinterpret_cast<float4*>(dst + 4) = make_float4(pv[o][4], pv[o][5], pv[o][6], pv[o][7]);
    }
  }
  __syncthreads();
  MSTAMP(2);
  if (warp < 3) {
    // the split's GEMM, n-atom `warp`: (block kb, atom u, route o) in DeepGEMM's issue order
    const int g = lane >> 2, tig = lane & 3, na = warp;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kb = 0; kb < 2; ++kb)
#pragma unroll
      for (int u = 0; u < 8; ++u)
#pragma unroll
        for (int o = 0; o < kR; ++o) {
          const int k0 = kb * 64 + u * 8;
          const float* a = &A[(o * 16) * kAStride + k0];
          const float* b = &B[(o * kOut + na * 8 + g) * kAStride + k0];
          mma_tf32(c, __float_as_uint(a[g * kAStride + tig]), __float_as_uint(a[(g + 8) * kAStride + tig]),
                   __float_as_uint(a[g * kAStride + tig + 4]), __float_as_uint(a[(g + 8) * kAStride + tig + 4]),
                   __float_as_uint(b[tig]), __float_as_uint(b[tig + 4]));
        }
    const int col = na * 8 + 2 * tig;
    if (g < tn) {
      p.gemm_part[((size_t)s * p.T + t0 + g) * kOut + col] = c[0];
      p.gemm_part[((size_t)s * p.T + t0 + g) * kOut + col + 1] = c[1];
    }
    if (g + 8 < tn) {
      p.gemm_part[((size_t)s * p.T + t0 + g + 8) * kOut + col] = c[2];
      p.gemm_part[((size_t)s * p.T + t0 + g + 8) * kOut + col + 1] = c[3];
    }
  } else if (warp < 7) {
    // the split's sum of pv^2 as the two Post workers accumulate it; item (row, worker, cp) = 8 row + 4 w + cp
    const int item = tid - 96, cp = item & 3, w = (item >> 2) & 1, r = item >> 3;
    float ax = 0.f, ay = 0.f;
#pragma unroll
    for (int kb = 0; kb < 2; ++kb)
#pragma unroll
      for (int uu = 0; uu < 4; ++uu)
#pragma unroll
        for (int o = 0; o < kR; ++o) {
          const float* a = &A[(o * 16 + r) * kAStride + kb * 64 + (w + 2 * uu) * 8 + 2 * cp];
          ax = __fmaf_rn(a[0], a[0], ax);
          ay = __fmaf_rn(a[1], a[1], ay);
        }
    float v = __fadd_rn(ax, ay);
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
    const float other = __shfl_xor_sync(0xffffffffu, v, 4);
    const float sum = __fadd_rn(__fadd_rn(0.f, w == 0 ? v : other), w == 0 ? other : v);
    if ((item & 7) == 0 && r < tn) p.hc_part[(size_t)s * p.T + t0 + r] = sum;
  }
  __syncthreads();
  MSTAMP(3);
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const bool l = atomicAdd(p.counters + mt, 1u) == (uint32_t)(kSplits - 1);
    if (l) __threadfence();
    last = l;
  }
  __syncthreads();
  if (!last) return;
  MSTAMP(5);
  const int half = (tn + 1) / 2;
  if (warp < half) {  // warp w: tokens w and w + half of the tile
    const int r1 = warp + half;
    const int tt[2] = {t0 + warp, t0 + (r1 < tn ? r1 : warp)};
    const bool ok[2] = {true, r1 < tn};
    mega_mix_tokens2(p, tt, ok, lane, &cmix[0][0]);
  }
  __syncthreads();
  if (warp == 0 && lane < tn) {  // one token per lane
    float m[kR * kR];
#pragma unroll
    for (int i = 0; i < kR * kR; ++i) m[i] = cmix[lane][i];
    mega_sinkhorn(m, p.sinkhorn_eps, p.sinkhorn_iters);
    float4* dst = reinterpret_cast<float4*>(p.comb_out + (size_t)(t0 + lane) * kR * kR);
#pragma unroll
    for (int q = 0; q < kR; ++q) dst[q] = make_float4(m[4 * q], m[4 * q + 1], m[4 * q + 2], m[4 * q + 3]);
  }
  __syncthreads();
  MSTAMP(6);
  if (tid == 0) p.counters[mt] = 0u;
}

}  // namespace

extern "C" int w4_mega_seam_launch(const MegaParams* p, int pdl, cudaStream_t stream) {
  const int g = p->T <= 16 ? 2 : kG;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(((p->T + g - 1) / g) * kCS);
  cfg.blockDim = dim3(kThreads);
  cfg.stream = stream;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = kCS, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = pdl;
  cfg.attrs = at;
  cfg.numAttrs = 2;
  return g == 2 ? (int)cudaLaunchKernelEx(&cfg, mega_seam_kernel<2>, *p)
                : (int)cudaLaunchKernelEx(&cfg, mega_seam_kernel<kG>, *p);
}

extern "C" int w4_mega_mix_launch(const MegaParams* p, int pdl, cudaStream_t stream) {
  static bool attr = false;
  if (!attr) {
    const cudaError_t e = cudaFuncSetAttribute(mega_mix_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMixSmem);
    if (e != cudaSuccess) return (int)e;
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kSplits, (p->T + 15) / 16);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = kMixSmem;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  return (int)cudaLaunchKernelEx(&cfg, mega_mix_kernel, *p);
}
