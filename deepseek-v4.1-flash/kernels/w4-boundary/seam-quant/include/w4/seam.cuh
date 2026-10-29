// W4 seam: the mHC boundary between two sublayers, as device stages.
//
// A slice owns TileLang virtual threads [v0, v0 + VPC) of the norm reduction, i.e. the columns
// vt_col(b, hh, v, e) for b < 5, hh < 2, e < 8 (40 chunks of 8 columns per VPC = 4). Work granules:
//   chunk  (g, b, hh, vl): one 8-column chunk of one token            -> seam_chunk_post, seam_chunk_norm
//   vthread (g, hh, vl):   the per-position fma chains over b = 0..4  -> seam_vt_chain (pair hh = 0/1 adjacent)
// Phase order: chunk_post (all chunks) -> vt_chain -> [reduce the 64 partials of each token: vt_butterfly] ->
// chunk_norm (all chunks). The c (collapse) values of the slice live in shared memory between phases.
#pragma once

#include <cstdint>
#include <cuda_bf16.h>

#include "mhc_core.cuh"
#include "params.h"

namespace w4 {

__device__ __forceinline__ uint4 ld_nc_v4(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ld_v4(const void* p) { return *reinterpret_cast<const uint4*>(p); }

// Per-token mixes, loaded once per chunk thread (L1 broadcast).
struct TokenMix {
  float post[4], comb[16], pre[4];
};
__device__ __forceinline__ void load_token_mix(const SeamParams& p, int t, TokenMix& m) {
  const float4 a = *reinterpret_cast<const float4*>(p.post + t * 4);
  m.post[0] = a.x, m.post[1] = a.y, m.post[2] = a.z, m.post[3] = a.w;
  const float4 b = *reinterpret_cast<const float4*>(p.pre + t * 4);
  m.pre[0] = b.x, m.pre[1] = b.y, m.pre[2] = b.z, m.pre[3] = b.w;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float4 c = *reinterpret_cast<const float4*>(p.comb + t * 16 + q * 4);
    m.comb[q * 4 + 0] = c.x, m.comb[q * 4 + 1] = c.y, m.comb[q * 4 + 2] = c.z, m.comb[q * 4 + 3] = c.w;
  }
}

// The sublayer output x of a chunk: x8(p, t, col, xf) writes the 8 values of x[t][col, col + 8) as floats. Each
// value must be what production's bf16 x holds (bf16-representable) for the seam to stay bitwise. Default: load the
// bf16 x from p.x. W8 / W6 pass their own (fixed-order sums of tagged partials) to fold a reduce into the seam.
struct SeamXBf16 {
  __device__ __forceinline__ void operator()(const SeamParams& p, int t, int col, float (&xf)[8]) const {
    unpack8(ld_v4(p.x + (size_t)t * kHidden + col), xf);
  }
};

// MegaMoE's combine (W6 fork v5, single rank) done in the seam: per element acc = 0.f, then for each present slot k
// in order (routed 0..5 where cmb_topk[t][k] >= 0, the shared slot 6 always) acc = add.rn.f32.bf16(slot, acc) (DeepGEMM
// ptx::accumulate), then bf16 RN. Bitwise v5's y.
__device__ __forceinline__ float acc_bf16(float acc, uint32_t bits16) {
  float r;
  const unsigned short h = (unsigned short)bits16;
  asm("add.rn.f32.bf16 %0, %1, %2;" : "=f"(r) : "h"(h), "f"(acc));
  return r;
}
struct SeamXCombine {
  __device__ __forceinline__ void operator()(const SeamParams& p, int t, int col, float (&xf)[8]) const {
    const long long* ids = p.cmb_topk + (size_t)t * 6;
    const uint8_t* base = p.cmb_slots + (size_t)t * p.cmb_tok_stride + (size_t)col * 2;
    bool on[7];
#pragma unroll
    for (int k = 0; k < 6; ++k) on[k] = ids[k] >= 0;
    on[6] = true;
    uint4 v[7];
#pragma unroll
    for (int k = 0; k < 7; ++k) v[k] = on[k] ? ld_v4(base + (size_t)k * p.cmb_slot_stride) : make_uint4(0, 0, 0, 0);
    float acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
    for (int k = 0; k < 7; ++k) {
      if (!on[k]) continue;
      const uint32_t w[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[2 * j] = acc_bf16(acc[2 * j], w[j] & 0xFFFFu);
        acc[2 * j + 1] = acc_bf16(acc[2 * j + 1], w[j] >> 16);
      }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) xf[e] = bf2f(f2bf(acc[e]));
  }
};

// [cold add] + mHC post + collapse of one 8-column chunk at column `col` of token t.
// Stores residual_out (and aux); returns the collapse output c (8 bf16, packed).
// peer tier v3 out rows (W6 peer3_proto.cuh kOffOut: page-aligned after the scales)
constexpr uint64_t kP3OffOut = ((4096 + 64 * 128 + 64 * 5120 + 64 * 160 + 4095) / 4096) * 4096;

template <class X8 = SeamXBf16>
__device__ __forceinline__ uint4 seam_chunk_post(const SeamParams& p, int t, int col, const TokenMix& m,
                                                 bool cold_ok = false, const X8& x8 = X8{},
                                                 const uint8_t* rows = nullptr) {  // rows: the out rows' base (null: p3_seg)
  const size_t H = kHidden;
  uint4 rv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) rv[j] = ld_v4(p.residual + ((size_t)t * 4 + j) * H + col);
  float xf[8];
  x8(p, t, col, xf);
  bool finished = false;
  if (p.p3_seg != nullptr) {
    if (cold_ok && p.cold_has[t] != 0) {  // peer v3: out rows are token-indexed host memory written by the RTX
      uint4 o;
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(o.x), "=r"(o.y), "=r"(o.z), "=r"(o.w)
                   : "l"((rows != nullptr ? rows : p.p3_seg) + kP3OffOut + ((size_t)t * H + col) * 2));
      float cf[8];
      unpack8(o, cf);
#pragma unroll
      for (int e = 0; e < 8; ++e) xf[e] = bf2f(f2bf(__fadd_rn(xf[e], cf[e])));
      finished = true;
    }
    // the finished y: cold tokens' rows, and with the combine here every token (MegaMoE wrote no y)
    if (p.p3_y != nullptr && (finished || p.cmb_slots != nullptr))
      *reinterpret_cast<uint4*>(p.p3_y + (size_t)t * H + col) = pack8(xf);
  } else if (p.cold_rows != nullptr && *p.cold_rows != 0 && p.cold_has[t] != 0) {
    float cf[8];
    unpack8(ld_v4(p.cold_out + (size_t)p.cold_pos[t] * H + col), cf);
#pragma unroll
    for (int e = 0; e < 8; ++e) xf[e] = bf2f(f2bf(__fadd_rn(xf[e], cf[e])));
  }
  float r[4][8];
#pragma unroll
  for (int j = 0; j < 4; ++j) unpack8(rv[j], r[j]);
  float rn[4][8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float rr[4] = {r[0][e], r[1][e], r[2][e], r[3][e]};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float cc[4] = {m.comb[0 * 4 + i], m.comb[1 * 4 + i], m.comb[2 * 4 + i], m.comb[3 * 4 + i]};
      rn[i][e] = bf2f(f2bf(post_elem(m.post[i], xf[e], cc, rr)));
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(p.residual_out + ((size_t)t * 4 + i) * H + col) = pack8(rn[i]);
  if (p.aux_out != nullptr) {
    float a[8];
#pragma unroll
    for (int e = 0; e < 8; ++e)
      a[e] = __fdiv_rn(__fadd_rn(__fadd_rn(__fadd_rn(rn[0][e], rn[1][e]), rn[2][e]), rn[3][e]), 4.f);
    *reinterpret_cast<uint4*>(p.aux_out + (size_t)t * H + col) = pack8(a);
  }
  float c[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float rr[4] = {rn[0][e], rn[1][e], rn[2][e], rn[3][e]};
    c[e] = collapse_elem(m.pre, rr);
  }
  return pack8(c);
}

// The 8 per-position fma chains of one (token, hh, virtual thread) over its 5 chunks c[b], b = 0..4.
// The hh = 0 / hh = 1 threads of a virtual thread must be adjacent lanes (xor 1); returns the virtual
// thread's partial sum in the hh == 0 lane.
__device__ __forceinline__ float seam_vt_chain(const uint4 (&c)[5]) {
  float sp[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) sp[e] = 0.f;
#pragma unroll
  for (int b = 0; b < 5; ++b) {
    float f[8];
    unpack8(c[b], f);
#pragma unroll
    for (int e = 0; e < 8; ++e) sp[e] = __fmaf_rn(f[e], f[e], sp[e]);
  }
  float all[16];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    all[e] = sp[e];
    all[8 + e] = __shfl_xor_sync(0xffffffffu, sp[e], 1);
  }
  return vt_partial(all);
}

// W8's SW128 MXFP8 activation image of the layer input (FlashInfer's MXFP8 recipe, as fi_q): values
// [K block of 128 columns][npad rows][128 B], byte (t, k) of a block at (t / 8) * 1024 + (t % 8) * 128 +
// (((k / 16) ^ (t % 8)) * 16) + k % 16; scales: one 512-B atom per K block, token t's 4 UE8M0 32-column sub-block
// scales at byte t * 16 + sub. q == null: none.
struct MxImage {
  uint8_t* q = nullptr;
  uint8_t* sf = nullptr;
  int npad = 16;  // 16 (S <= 16) or 32
};
__device__ __forceinline__ size_t mx_image_off(int t, int col, int npad) {
  const int kb = col >> 7, k = col & 127;
  return (size_t)kb * npad * 128 + (t >> 3) * 1024 + (t & 7) * 128 + ((((k >> 4) ^ (t & 7)) & 7) << 4) + (k & 15);
}
__device__ __forceinline__ size_t mx_image_sf_off(int t, int col) { return (size_t)(col >> 7) * 512 + t * 16 + ((col >> 5) & 3); }

// Write one 8-column chunk of the layer input li (bf16 values as floats) and its quantized forms. The 4 chunks of a
// 32-column MXFP8 block must be lanes l, l^1, l^2, l^3 of one warp and call together.
__device__ __forceinline__ void seam_chunk_store(const SeamParams& p, int t, int col, const float (&li)[8], bool active,
                                                 const MxImage& img = MxImage{}) {
  const size_t H = kHidden;
  if (active && p.layer_input != nullptr) *reinterpret_cast<uint4*>(p.layer_input + (size_t)t * H + col) = pack8(li);
  if (p.fi_q == nullptr && p.prep_x == nullptr && img.q == nullptr) return;
  float am = amax8(li);
  const unsigned chunk_mask = __activemask();
  am = fmaxf(am, __shfl_xor_sync(chunk_mask, am, 1));
  am = fmaxf(am, __shfl_xor_sync(chunk_mask, am, 2));
  const bool lead = active && (col & 31) == 0;
  if (p.fi_q != nullptr) {
    const uint32_t e8 = fi_ue8m0(__fmul_rn(am, kInvE4m3Max));
    if (active) *reinterpret_cast<uint2*>(p.fi_q + (size_t)t * H + col) = fi_quant8(li, fi_inv_scale(e8));
    if (lead) {
      if (p.fi_sf_swz != nullptr) p.fi_sf_swz[sf_swizzled_128x4(t, col / 32, 160)] = (uint8_t)e8;
      if (p.fi_sf_lin != nullptr) p.fi_sf_lin[t * 160 + col / 32] = (uint8_t)e8;
    }
  }
  if (img.q != nullptr) {
    const uint32_t e8 = fi_ue8m0(__fmul_rn(am, kInvE4m3Max));
    if (active) *reinterpret_cast<uint2*>(img.q + mx_image_off(t, col, img.npad)) = fi_quant8(li, fi_inv_scale(e8));
    if (lead) img.sf[mx_image_sf_off(t, col)] = (uint8_t)e8;
  }
  if (p.prep_x != nullptr) {
    const uint32_t pe = prep_exp(am);
    if (active) *reinterpret_cast<uint2*>(p.prep_x + (size_t)t * H + col) = prep_quant8(li, prep_inv(pe));
    if (lead) {
      const int grp = col / 128, byte = (col % 128) / 32;
      reinterpret_cast<uint8_t*>(p.prep_sf)[((size_t)t * (H / 128) + grp) * 4 + byte] = (uint8_t)pe;
      if (p.prep_shared_sf != nullptr) {
        const int row = prep_shared_row(t, p.shared_block_m);
        reinterpret_cast<uint8_t*>(p.prep_shared_sf)[((size_t)row * (H / 128) + grp) * 4 + byte] = (uint8_t)pe;
      }
    }
  }
}

// Normalize one chunk (TileLang's RMSNorm: li = bf16(c * rs * w)) and write li and its quantized forms (same lane
// grouping as seam_chunk_store).
__device__ __forceinline__ void seam_chunk_norm(const SeamParams& p, int t, int col, uint4 cpk, float rs,
                                                const float (&w)[8], bool active, const MxImage& img = MxImage{}) {
  float c[8], li[8];
  unpack8(cpk, c);
#pragma unroll
  for (int e = 0; e < 8; ++e) li[e] = bf2f(f2bf(norm_apply(c[e], rs, w[e])));
  seam_chunk_store(p, t, col, li, active, img);
}


// Zero the swizzled scale bytes of padding rows [T, roundup(T, 128)) (FlashInfer leaves them zero),
// part `part` of `parts`.
__device__ __forceinline__ void seam_zero_sf_padding(const SeamParams& p, int part, int parts, int tid, int nthr) {
  if (p.fi_sf_swz == nullptr || p.sf_pad_ready) return;
  const int rows = ((p.T + 127) / 128) * 128;
  const int n = (rows - p.T) * 40;  // 4-byte groups (4 scale columns are contiguous)
  for (int i = part * nthr + tid; i < n; i += parts * nthr) {
    const int r = p.T + i / 40, cg = i % 40;
    *reinterpret_cast<uint32_t*>(p.fi_sf_swz + sf_swizzled_128x4(r, cg * 4, 160)) = 0u;
  }
}

}  // namespace w4
