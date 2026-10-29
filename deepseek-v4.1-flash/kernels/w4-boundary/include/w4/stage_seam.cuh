// W4 mHC seam + next-coefficient stages for the W8 persistent kernel (CONTRACT.md v1).
//
// Bit-exact against production (vLLM af7f9488): mhc_post_tilelang, mhc_pre_big_fuse_with_norm_tilelang "input"
// (collapse + RMSNorm) and "stats" (sigmoid + Sinkhorn) splits, DeepGEMM sm100_tf32_hc_prenorm_gemm (16 splits),
// FlashInfer cute-dsl mxfp8_quantize (swizzled 128x4 and linear scales) and Triton prepare_megamoe_inputs (x / x_sf
// / shared_x_sf). Evidence: w4-boundary tests/test_boundary.py (576 checks, layers 3/20/38, T 1..48, all equal).
//
// Stages (S = width, 1..64; 256 compute threads; no clusters, no intra-stage waits, no data atomics):
//   SeamPost  16 tasks: task r = TileLang norm virtual threads [4r, 4r+4) = 320 columns x 4 streams, all S tokens.
//             [cold add] + mHC post -> residual_out; collapse c -> scratch; per-virtual-thread partial sums of c^2
//             -> scratch. Reads x (dep_x: e.g. W8's wo_b reduce tasks) and residual/post/comb/pre (earlier launch).
//   SeamNorm  16 tasks (same slices). Waits on all 16 SeamPost tasks. Reduces the 64 partials of every token in
//             TileLang's butterfly order, normalizes its slice, writes layer_input (bf16) and the quantized forms.
//   MixGemm   16 * ceil(S/16) tasks: task = (split s, 16-token tile). Waits on all SeamPost tasks (reads
//             residual_out). 160 dependent tf32 mma.sync m16n8k8 per n-atom (3 chains on warps 0-2), B = hc_fn
//             tf32-RNE fragment image through the weight ring (4 slots x 30 KB per task, produce()), A (80 KB of tf32
//             fragments) in 3 borrowed ring slots (7 slots per task: needs a ring of >= 6 x 33,792 B), sums of squares
//             from warp 0's A fragments (same fma chains). Writes the 16 split partials (scratch).
//   MixStats  ceil(S/64) tasks: 2 tokens per warp. Waits on all MixGemm tasks. Writes post/comb/pre (production).
//   Peer3Advance  1 task (Block 2, seam.p3_seg set): waits on all SeamPost tasks, advances peer3's call counter.
// Block 2 extras: SeamPost runs the fused peer3 finish (p3_finish_wait + the cold-row add) when seam.p3_seg is set;
// SeamNorm writes W8's SW128 MXFP8 image of the layer input when img.q is set (seam.cuh MxImage).
//
// Scratch (bytes, per width S): c [S][16][40] uint4 = 10240 S; vtp [S][64] f32 = 256 S; mul [16][S][24] f32 = 1536 S;
// sqr [16][S] f32 = 64 S.  Shared memory (stage area): SeamPost 640 S + 64, SeamNorm 256 S + 64, MixGemm 0, MixStats
// 6400.
#pragma once

#include <cstdint>
#include <cuda_bf16.h>

#include "w8/runtime.cuh"
#include "mhc_core.cuh"
#include "params.h"
#include "seam.cuh"

namespace w4::stage {

static_assert(w8::kComputeWarps == 8, "w4 stages are laid out for the 256-thread (8 compute warp) CTA");
#define W4SEAM_STAMP(dbg, task, j) \
  if ((dbg) != nullptr && threadIdx.x == 0) (dbg)[(task) * 8 + (j)] = w8::globaltimer();

constexpr int kSlices = 16;          // seam tasks: 64 TileLang virtual threads / 4
constexpr int kVPC = kVt / kSlices;  // 4 virtual threads per slice
constexpr int kChunks = 5 * 2 * kVPC;  // 40 chunks of 8 columns per token per slice
constexpr int kMixSplits = 16, kMixKps = 1280, kMixSteps = 160;
constexpr int kMixChunkSteps = 40;                       // steps per ring slot
constexpr int kMixSlotBytes = 3 * kMixChunkSteps * 256;  // 30720: [n-atom 3][40 steps][32 lanes][float2]
constexpr int kMixSlots = kMixSteps / kMixChunkSteps;    // 4 per task

__device__ __forceinline__ int slice_chunk_col(int r, int ci) {
  const int vl = ci % kVPC, hh = (ci / kVPC) & 1, b = ci / (2 * kVPC);
  return vt_col(b, hh, r * kVPC + vl, 0);
}

// Fused peer tier v3 finish (W6 peer3 W4_INTEGRATION.md section 6, as w4_seam's): with seam.p3_seg set, the cold
// rows of the MoE call that produced x are added in seam_chunk_post. Thread 0 of task 0 always, and of the other
// tasks when a token is cold, waits (ld.acquire.sys, at most p3_spins polls) for the RTX's acknowledgement of
// tag = (p3_state[0], p3_layer, S); task 0 records peer3's GbStats (single writer, plain stores). Returns whether
// the cold rows may be added (all compute threads; `flag` is 4 B of SMEM). The call counter is advanced by
// Peer3Advance once every SeamPost task is done.
__device__ __forceinline__ bool p3_finish_wait(const SeamParams& p, int S, int task, uint8_t* flag) {
  if (p.p3_seg == nullptr) return false;
  if (threadIdx.x == 0) {
    const uint32_t tag = (p.p3_state[0] << 12) | ((uint32_t)p.p3_layer << 6) | (uint32_t)(S - 1);
    bool need = task == 0;
    for (int t = 0; t < S; ++t) need |= p.cold_has[t] != 0;
    int ok = 0;
    if (need) {
      const unsigned long long w0 = w8::globaltimer();
      long long spins = 0;
      unsigned long long done;
      asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(done) : "l"(p.p3_seg + 128) : "memory");
      while (done != tag && spins < p.p3_spins) {
        asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(done) : "l"(p.p3_seg + 128) : "memory");
        ++spins;
      }
      ok = done == tag;
      if (task == 0 && p.p3_stats != nullptr) {  // GbStats: calls, waited, wait_ns, timeouts, hist[16] (+ last ns, tag)
        const unsigned long long ns = w8::globaltimer() - w0;
        p.p3_stats[0] += 1ull;
        p.p3_stats[2] += ns;
        if (!ok) p.p3_stats[3] += 1ull;
        int bkt = 0;
        for (unsigned long long us = ns / 1000; us > 0 && bkt < 15; us >>= 1) ++bkt;
        p.p3_stats[4 + bkt] += 1ull;
        if (p.p3_stats_words >= 22) p.p3_stats[20] = ns, p.p3_stats[21] = tag;
      }
    }
    *reinterpret_cast<volatile int*>(flag) = ok;
  }
  w8::compute_sync();
  return *reinterpret_cast<volatile int*>(flag) != 0;
}

// ------------------------------------------------------------------------------------------------ SeamPost
struct SeamPost {
  struct Params {
    SeamParams seam;      // x, residual, post, comb, pre, cold_*, residual_out, aux_out; T = S
    uint4* c;             // scratch [S][16][40]
    float* vtp;           // scratch [S][64]
    w8::Dep dep_x;        // producer tasks of x (flags); {nullptr, 0, 0} if x comes from an earlier launch
  };
  static constexpr const char* kName = "seam_post";
  __host__ __device__ static constexpr int tasks(int) { return kSlices; }
  __host__ __device__ static constexpr uint32_t smem_bytes(int S) { return (uint32_t)S * kChunks * 16 + 64; }
  __host__ __device__ static constexpr uint32_t tmem_cols(int) { return 0; }
  static size_t scratch_bytes(int S) { return (size_t)S * kSlices * kChunks * 16 + (size_t)S * kVt * 4; }
  // seam.p3_seg set (Block 2: the attention seam after a peer-tier MoE call): fused finish, see p3_finish_wait.

  // x8: the source of x (SeamXBf16 = load p.seam.x; see seam.cuh). A functor that waits on its own tagged inputs
  // runs with dep_x = {nullptr, 0, 0}.
  template <class Sync, class X8 = SeamXBf16>
  __device__ static void consume(const Params& p, int S, int r, Sync& sync, uint8_t* smem, const X8& x8 = X8{}) {
    uint4* c_sh = reinterpret_cast<uint4*>(smem);  // [S][40]
    const int tid = threadIdx.x;
    sync.wait_extern();
    if (p.dep_x.flags != nullptr) sync.wait(p.dep_x, 0x400);
    const bool cold_ok = p3_finish_wait(p.seam, S, r, smem + S * kChunks * 16);
    for (int item = tid; item < S * kChunks; item += w8::kComputeThreads) {
      const int g = item / kChunks, ci = item % kChunks;
      TokenMix m;
      load_token_mix(p.seam, g, m);
      const uint4 c = seam_chunk_post(p.seam, g, slice_chunk_col(r, ci), m, cold_ok, x8);
      c_sh[item] = c;
      p.c[((size_t)g * kSlices + r) * kChunks + ci] = c;
    }
    w8::compute_sync();
    for (int base = 0; base < S * kVPC * 2; base += w8::kComputeThreads) {
      const int item = base + tid;
      const bool active = item < S * kVPC * 2;
      const int hh = item & 1, vl = (item >> 1) % kVPC, g = active ? (item >> 1) / kVPC : 0;
      uint4 c[5];
#pragma unroll
      for (int b = 0; b < 5; ++b) c[b] = c_sh[g * kChunks + (b * 2 + hh) * kVPC + vl];
      const float s = seam_vt_chain(c);
      if (active && hh == 0) p.vtp[g * kVt + r * kVPC + vl] = s;
    }
  }
};

// ------------------------------------------------------------------------------------------------ SeamNorm
struct SeamNorm {
  struct Params {
    SeamParams seam;      // norm_w, layer_input, fi_*, prep_*; T = S
    const uint4* c;       // scratch from SeamPost
    const float* vtp;
    w8::Dep dep_post;     // all 16 SeamPost tasks
    MxImage img{};        // W8's SW128 MXFP8 image of the layer input (seam.cuh MxImage), or q == null
  };
  static constexpr const char* kName = "seam_norm";
  __host__ __device__ static constexpr int tasks(int) { return kSlices; }
  __host__ __device__ static constexpr uint32_t smem_bytes(int S) { return (uint32_t)S * 4 + 64; }
  __host__ __device__ static constexpr uint32_t tmem_cols(int) { return 0; }
  static size_t scratch_bytes(int) { return 0; }

  template <class Sync>
  __device__ static void consume(const Params& p, int S, int r, Sync& sync, uint8_t* smem) {
    float* rs_sh = reinterpret_cast<float*>(smem);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    sync.wait(p.dep_post, 0x401);
    for (int g = warp; g < S; g += w8::kComputeWarps) {
      const float tot = vt_butterfly(__ldcg(p.vtp + g * kVt + lane), __ldcg(p.vtp + g * kVt + lane + 32));
      if (lane == 0) rs_sh[g] = norm_rsqrt(tot);
    }
    w8::compute_sync();
    for (int base = 0; base < S * kChunks; base += w8::kComputeThreads) {
      const int item = base + tid;
      const bool active = item < S * kChunks;
      const int g = active ? item / kChunks : 0, ci = item % kChunks;
      const int col = slice_chunk_col(r, ci);
      float wf[8];
      unpack8(ld_nc_v4(p.seam.norm_w + col), wf);
      const uint4 c = active ? __ldcg(p.c + ((size_t)g * kSlices + r) * kChunks + ci) : make_uint4(0, 0, 0, 0);
      seam_chunk_norm(p.seam, g, col, c, rs_sh[g], wf, active, p.img);
    }
    if (p.img.q != nullptr)  // the image's padding rows [S, npad) of this slice: zero values and scales
      for (int item = tid; item < (p.img.npad - S) * kChunks; item += w8::kComputeThreads) {
        const int t = S + item / kChunks, col = slice_chunk_col(r, item % kChunks);
        *reinterpret_cast<uint2*>(p.img.q + mx_image_off(t, col, p.img.npad)) = make_uint2(0, 0);
        if ((col & 31) == 0) p.img.sf[mx_image_sf_off(t, col)] = 0;
      }
    if (r == 0) seam_zero_sf_padding(p.seam, 0, 1, tid, w8::kComputeThreads);
  }
};

// ------------------------------------------------------------------------------------------------ MixGemm
// not volatile: the next steps' SMEM operand loads may be scheduled ahead of this one (the accumulator dependency
// keeps the chain's order); a volatile mma put every step's LDS on the critical path (64 vs 20 cycles per step)
__device__ __forceinline__ void mma_tf32_m16n8k8(float (&c)[4], const float4& a, const float2& b) {
  asm(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(__float_as_uint(a.x)), "r"(__float_as_uint(a.y)), "r"(__float_as_uint(a.z)), "r"(__float_as_uint(a.w)),
        "r"(__float_as_uint(b.x)), "r"(__float_as_uint(b.y)));
}

struct MixGemm {
  struct Params {
    const __nv_bfloat16* rn;  // residual_out [S][4*5120]
    const uint8_t* fn_ring;   // hc_fn image (w4b.boundary.mix_fn_ring_image): [split][slot 4][30720 B]
    float* mul;               // scratch [16][S][24]
    float* sqr;               // scratch [16][S]
    w8::Dep dep_post;         // all 16 SeamPost tasks
    unsigned long long* dbg = nullptr;  // tests: phase stamps [task][8] (globaltimer)
  };
  static constexpr const char* kName = "mix_gemm";
  // A (the tile's tf32 fragments, 160 steps x 512 B = 80 KB) lives in 3 borrowed ring slots (66 steps each), B (hc_fn)
  // in 4 weight slots: 7 ring slots per task, in the order B0 A0 A1 A2 B1 B2 B3. B3 reuses B0's slot on a 6-slot
  // ring (B0 is released after steps 0..39), so the task is deadlock-free on any ring of >= 6 slots of >= 33,792 B.
  static constexpr int kASteps = 66, kASlots = 3, kTaskSlots = kMixSlots + kASlots;
  __host__ __device__ static constexpr int a_pos(int j) { return 1 + j; }
  __host__ __device__ static constexpr int b_pos(int s) { return s == 0 ? 0 : kASlots + s; }
  __host__ __device__ static constexpr int tasks(int S) { return kMixSplits * ((S + 15) / 16); }
  __host__ __device__ static constexpr uint32_t smem_bytes(int) { return 0; }
  __host__ __device__ static constexpr uint32_t tmem_cols(int) { return 0; }
  static size_t scratch_bytes(int S) { return (size_t)kMixSplits * S * (24 + 1) * 4; }

  // Producer (one lane of the producer warp): this task's 7 slots in consumption order (A slots: 0 tx).
  template <class RingT>
  __device__ static void produce(const Params& p, int, int task, const RingT& ring, uint32_t& prod_it, w8::Diag* diag) {
    static_assert(RingT::NSLOT >= 6 && RingT::SLOT_BYTES >= kASteps * 512, "MixGemm needs >= 6 slots of 33,792 B");
    const int split = task % kMixSplits;
    for (int q = 0; q < kTaskSlots; ++q, ++prod_it) {
      uint8_t* slot = ring.acquire(prod_it, diag);
      if (q >= a_pos(0) && q <= a_pos(kASlots - 1)) {
        ring.submit(prod_it, 0);
        continue;
      }
      const int s = q == 0 ? 0 : q - kASlots;
      w8::bulk_g2s(slot, p.fn_ring + ((size_t)split * kMixSlots + s) * kMixSlotBytes, kMixSlotBytes,
                   ring.full_bar(prod_it));
      ring.submit(prod_it, kMixSlotBytes);
    }
  }

  template <class Sync, class RingT>
  __device__ static void consume(const Params& p, int S, int task, Sync& sync, const RingT& ring, uint32_t& ring_it,
                                 uint8_t*, w8::Diag* diag) {
    const int split = task % kMixSplits, mt = task / kMixSplits;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int rows = min(16, S - mt * 16);
    // A step st: float4 [32] at slot[st / 66] + (st % 66) * 512, lane slot l at l ^ (st & 31)
    uint8_t* a0 = const_cast<uint8_t*>(ring.wait_full(ring_it + a_pos(0), diag));
    uint8_t* a1 = const_cast<uint8_t*>(ring.wait_full(ring_it + a_pos(1), diag));
    uint8_t* a2 = const_cast<uint8_t*>(ring.wait_full(ring_it + a_pos(2), diag));
    auto astep = [&](int st) {
      return reinterpret_cast<float4*>((st < kASteps ? a0 : st < 2 * kASteps ? a1 : a2) +
                                       (st - (st < kASteps ? 0 : st < 2 * kASteps ? kASteps : 2 * kASteps)) * 512);
    };
    W4SEAM_STAMP(p.dbg, task, 0);
    sync.wait(p.dep_post, 0x402);
    W4SEAM_STAMP(p.dbg, task, 1);
    {
      const int g = warp;  // 8 warps = 8 row pairs (g, g + 8)
      const __nv_bfloat16* row_lo = p.rn + (size_t)(mt * 16 + g) * 4 * kHidden + split * kMixKps;
      const __nv_bfloat16* row_hi = p.rn + (size_t)(mt * 16 + g + 8) * 4 * kHidden + split * kMixKps;
      uint4 lo[5], hi[5];
#pragma unroll
      for (int j = 0; j < 5; ++j) {
        const int st = lane + 32 * j;
        lo[j] = g < rows ? __ldcg(reinterpret_cast<const uint4*>(row_lo + st * 8)) : make_uint4(0, 0, 0, 0);
        hi[j] = g + 8 < rows ? __ldcg(reinterpret_cast<const uint4*>(row_hi + st * 8)) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int j = 0; j < 5; ++j) {
        const int st = lane + 32 * j;
        float a[8], b[8];
        unpack8(lo[j], a);
        unpack8(hi[j], b);
        float4* dst = astep(st);
#pragma unroll
        for (int tig = 0; tig < 4; ++tig)
          dst[(g * 4 + tig) ^ (st & 31)] = make_float4(a[tig], b[tig], a[tig + 4], b[tig + 4]);
      }
    }
    w8::compute_sync();
    W4SEAM_STAMP(p.dbg, task, 2);
    if (warp < 3) {
      // n-atom na = warp: 160 dependent tf32 mma, operands prefetched kPf steps ahead within each B slot. Warp 0 also
      // runs the sums of squares from its A fragments: lane (g, tig) holds A[g][tig], A[g+8][tig], A[g][tig+4],
      // A[g+8][tig+4] of every step, i.e. the per-(row, element) fma chains, combined below in production's order.
      constexpr int kPf = 4;
      const int na = warp;
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;  // rows g / g + 8, elements tig / tig + 4
#pragma unroll 1
      for (int s = 0; s < kMixSlots; ++s) {
        const float2* b2 =
            reinterpret_cast<const float2*>(ring.wait_full(ring_it + b_pos(s), diag)) + na * kMixChunkSteps * 32;
        if (p.dbg != nullptr && tid == 0 && s > 0) p.dbg[task * 8 + 4 + s] = w8::globaltimer();
        float4 ra[kPf];
        float2 rb[kPf];
#pragma unroll
        for (int d = 0; d < kPf; ++d) {
          const int st = s * kMixChunkSteps + d;
          ra[d] = astep(st)[lane ^ (st & 31)];
          rb[d] = b2[d * 32 + lane];
        }
#pragma unroll
        for (int i = 0; i < kMixChunkSteps; ++i) {
          const float4 av = ra[i % kPf];
          const float2 bv = rb[i % kPf];
          if (i + kPf < kMixChunkSteps) {
            const int st = s * kMixChunkSteps + i + kPf;
            ra[i % kPf] = astep(st)[lane ^ (st & 31)];
            rb[i % kPf] = b2[(i + kPf) * 32 + lane];
          }
          mma_tf32_m16n8k8(c, av, bv);
          if (warp == 0) {
            q0 = __fmaf_rn(av.x, av.x, q0);
            q1 = __fmaf_rn(av.y, av.y, q1);
            q2 = __fmaf_rn(av.z, av.z, q2);
            q3 = __fmaf_rn(av.w, av.w, q3);
          }
        }
        w8::bar_sync(2, 96);  // the 3 chain warps are done with this slot
        if (tid == 0) ring.release(ring_it + b_pos(s));
      }
      W4SEAM_STAMP(p.dbg, task, 3);
      const int g = lane >> 2, tig = lane & 3, n0 = na * 8 + 2 * tig;
      if (g < rows)
        *reinterpret_cast<float2*>(p.mul + ((size_t)split * S + mt * 16 + g) * kMix + n0) = make_float2(c[0], c[1]);
      if (g + 8 < rows)
        *reinterpret_cast<float2*>(p.mul + ((size_t)split * S + mt * 16 + g + 8) * kMix + n0) =
            make_float2(c[2], c[3]);
      if (warp == 0) {
        // row r: v_q = s_{2q} + s_{2q+1} (chains of elements 2q, 2q + 1), total = ((v0 + v2) + (v1 + v3))
        auto row_total = [&](float lo, float hi) {  // lo = s_tig, hi = s_{tig+4} of one row
          const float x = __fadd_rn(lo, __shfl_xor_sync(0xffffffffu, lo, 1));  // tig 0: v0, tig 2: v1
          const float y = __fadd_rn(hi, __shfl_xor_sync(0xffffffffu, hi, 1));  // tig 0: v2, tig 2: v3
          const float t = __fadd_rn(x, y);                                      // tig 0: v0 + v2, tig 2: v1 + v3
          return __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, 2));
        };
        const float r0 = row_total(q0, q2), r1 = row_total(q1, q3);
        if (tig == 0 && g < rows) p.sqr[split * S + mt * 16 + g] = r0;
        if (tig == 0 && g + 8 < rows) p.sqr[split * S + mt * 16 + g + 8] = r1;
      }
    }
    // A was written with generic stores: order them before the slots' next (async-proxy) writers
    w8::fence_proxy_async_smem();
    w8::compute_sync();
    if (tid == 0)
      for (int j = 0; j < kASlots; ++j) ring.release(ring_it + a_pos(j));
    ring_it += kTaskSlots;
    W4SEAM_STAMP(p.dbg, task, 4);
  }
};

// ------------------------------------------------------------------------------------------------ MixStats
struct MixStats {
  struct Params {
    const float* mul;
    const float* sqr;
    const float* scale;  // hc_*_scale [3]
    const float* base;   // hc_*_base [24]
    float* post_out;     // [S][4]
    float* comb_out;     // [S][16]
    float* pre_out;      // [S][4]
    w8::Dep dep_gemm;    // all MixGemm tasks
    unsigned long long* dbg = nullptr;  // tests: phase stamps [task][8] (globaltimer)
  };
  static constexpr const char* kName = "mix_stats";
  __host__ __device__ static constexpr int tasks(int S) { return (S + 63) / 64; }
  __host__ __device__ static constexpr uint32_t smem_bytes(int) { return 64 * 25 * 4; }
  __host__ __device__ static constexpr uint32_t tmem_cols(int) { return 0; }
  static size_t scratch_bytes(int) { return 0; }

  template <class Sync>
  __device__ static void consume(const Params& p, int S, int task, Sync& sync, uint8_t* smem) {
    W4SEAM_STAMP(p.dbg, task, 0);
    sync.wait(p.dep_gemm, 0x403);
    W4SEAM_STAMP(p.dbg, task, 1);
    float* msum = reinterpret_cast<float*>(smem);
    const int t0 = task * 64, nt = min(64, S - t0);
    stats_sums<kMixSplits>(p.mul, p.sqr, S, t0, nt, msum, threadIdx.x, w8::kComputeThreads);
    w8::compute_sync();
    W4SEAM_STAMP(p.dbg, task, 2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int g0 = warp * 2; g0 < nt; g0 += w8::kComputeWarps * 2) {
      const int g = g0 + (lane >> 4);
      stats_from_sums(msum, g, t0 + g, g < nt, p.scale, p.base, p.post_out, p.comb_out, p.pre_out);
    }
    W4SEAM_STAMP(p.dbg, task, 3);
  }
};

// ------------------------------------------------------------------------------------------------ Peer3Advance
// One task: after every SeamPost task (all fused-finish tags read, all cold rows added), advances peer3's call
// counter (1..15, as the finish's last CTA does). Only with seam.p3_seg set.
struct Peer3Advance {
  struct Params {
    uint32_t* p3_state;  // [2] peer3 GB300 state: [0] call counter
    w8::Dep dep_post;    // all 16 SeamPost tasks
  };
  static constexpr const char* kName = "peer3_advance";
  __host__ __device__ static constexpr int tasks(int) { return 1; }
  __host__ __device__ static constexpr uint32_t smem_bytes(int) { return 0; }
  __host__ __device__ static constexpr uint32_t tmem_cols(int) { return 0; }
  static size_t scratch_bytes(int) { return 0; }

  template <class Sync>
  __device__ static void consume(const Params& p, int, int, Sync& sync, uint8_t*) {
    sync.wait(p.dep_post, 0x404);
    if (threadIdx.x == 0) p.p3_state[0] = p.p3_state[0] >= 15 ? 1u : p.p3_state[0] + 1u;
  }
};

}  // namespace w4::stage
