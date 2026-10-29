// W4 MoE router as W8-contract stages (CONTRACT.md v1): no clusters, no DSMEM, no data atomics.
//
// Bit-exact against production's router at every width (vLLM af7f9488 + the ds41 hooks):
//   S <= 4        GateLinear -> ll_bf16_gemm's LLBf16Dotprod (bs 128, 8-wide vectors) -> dsv4_topk
//   5 <= S <= 16  GateLinear -> ll_bf16_gemm's LLBf16SplitK (6 K ranks of 256-wide tiles, 16-wide K blocks) -> dsv4_topk
//   17 <= S <= 48 DeepGEMM bf16_mega_gate (8 K splits of 640, K16 steps, splits summed from split 0, precise
//                 sqrt-softplus, weights s / (sum + 1e-20) * scale)
// then the hot/cold split, MegaMoE's topk_idx / topk_weights (padding masked) and the peer tier v3 publish
// (W6 peer3_proto.cuh), byte for byte what peer3's send writes. MegaMoE's x / x_sf / shared_x_sf and the cold rows'
// MXFP8 come from SeamNorm (SeamParams prep_* and fi_q / fi_sf_lin), not from these stages.
//
// Stages (S = 1..48, 256 compute threads, stage area <= 13 KB; larger operands ride borrowed ring slots):
//   RouterGate   (tile, K rank) tasks; each writes its rank's partial logits as W8 tagged words (w8/tagged.cuh).
//                It waits on the SeamPost tasks (dep_post) and normalizes its own li columns from SeamPost's c / vtp
//                exactly as SeamNorm does (no SeamNorm hop); weights are fetched before the wait.
//                Weights: one swizzled tensor-map TMA per slot (RouterGateMaps; a cp.async.bulk costs ~55 ns to issue).
//                The ring's slots must be 1024-B aligned in SMEM (128-B swizzle).
//                S <= 4:   96 tasks = 24 tiles of 16 experts x 4 ranks (production's dotprod warp vw): 2 weight
//                          slots (8 experts each), the rank's li columns {1024 i + 256 vw + [0, 256)} in the stage
//                          area; thread = (vt = 32 vw + lane, experts 2w, 2w + 1), chains in order, butterfly 16..1.
//                S 5..16:  72 tasks = 12 tiles of 32 experts x 6 ranks: 2 weight slots, li rank slice in 1 borrowed slot.
//                S 17..48: 96 tasks = 12 tiles of 32 experts x 8 splits: 2 weight slots, li slice in ceil(S/16)
//                          borrowed slots; li from the Mega seam (MegaSeamPost's x1 + split partials, dep_mega_post).
//   RouterTopk   S tasks, one token each: polls the token's rank partials (tagged), sums them in production order,
//                scores all 384 experts, selects the top-6, computes the routing weights, writes topk ids / weights,
//                hot ids, MegaMoE topk_idx / topk_weights (plain and/or tagged) and has[t], then sends the token from
//                warp 0 (v3 record: 6 weights and the MXFP8 row if any route is cold, __syncwarp, one st.release.sys of
//                word0). It acquires SeamNorm's flags (dep_norm: fi_q / fi_sf_lin) and prefetches the row first.
// RouterTopk reads RouterGate's output as tagged words polled with relaxed loads: no flag or release fence between
// them (RouterGate need not publish flags).
//
// Scratch (per width S): partials [24][8][S][16] u64 = 24576 S B. Tagged scratch must never hold a word carrying a
// future launch's tag: zero it once at allocation (tags only grow).
#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#include "w8/runtime.cuh"
#include "w8/tagged.cuh"
#include "mgate_core.cuh"
#include "mhc_core.cuh"
#include "route_core.cuh"
#include "seam.cuh"
#include "stage_mega.cuh"
#include "stage_seam.cuh"

namespace w4::stage {

static_assert(w8::kComputeWarps == 8, "w4 router stages are laid out for the 256-thread (8 compute warp) CTA");

constexpr int kDotMaxS = 4, kLLMaxS = 16, kMegaMaxS = 48;
constexpr int kTiles = kExperts / 16;  // 24 tiles of 16 experts
constexpr int kDotRanks = 4;
constexpr int kDotSliceE = 5 * 256;            // a dot rank's li columns per token
constexpr int kDotARowE = kDotSliceE + 8;      // staged li row (elements)
constexpr int kLLRowB = 4 * 256 * 2 + 16;      // LL rank: its <= 4 tiles of 256 per row (16 rows = 33,024 B)
constexpr int kMgRowB = 640 * 2 + 16;          // Mega split: 640 per row (16 rows = 20,736 B; 24 li rows = 31,104 B)
constexpr int kGateSlotBytes = 16 * kLLRowB;   // the largest slot any task fills (the LL li slice)
constexpr uint64_t kP3OffRec = 4096, kP3RecBytes = 128;
constexpr uint64_t kP3OffXq = kP3OffRec + 64 * kP3RecBytes;
constexpr uint64_t kP3OffXs = kP3OffXq + 64ull * 5120;

__host__ __device__ constexpr int gate_ranks(int S) {
  return S <= kDotMaxS ? kDotRanks : S <= kLLMaxS ? kRouterSplitK : mgate::kSplitK;
}
// LLBf16SplitK: tile j of rank z is z + 6 j (j < ntiles(z))
__host__ __device__ constexpr int ll_ntiles(int z) { return (kRouterKTiles - z + kRouterSplitK - 1) / kRouterSplitK; }

// mma.sync m16n8k16 bf16 without `volatile`: independent chains may be scheduled together
__device__ __forceinline__ void rt_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Order-preserving u32 key of a score (+0 == -0); NaN is mapped below every finite score by the callers.
__device__ __forceinline__ uint32_t rt_score_key(float v) {
  const uint32_t b = __float_as_uint(__fadd_rn(v, 0.0f));
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float rt_dot8(float acc, uint4 a, uint4 b) {  // in element order; products are exact
  const uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc = __fmaf_rn(__uint_as_float(av[i] << 16), __uint_as_float(bv[i] << 16), acc);
    acc = __fmaf_rn(__uint_as_float(av[i] & 0xffff0000u), __uint_as_float(bv[i] & 0xffff0000u), acc);
  }
  return acc;
}
__device__ __forceinline__ uint32_t rt_lds32(const void* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ void rt_cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(w8::smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void rt_cp_wait() { asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory"); }

// RouterGate's weight tensor maps (host: stage_router_maps.h encode_router_gate_maps): one TMA per ring slot
struct RouterGateMaps {
  CUtensorMap dot;   // S <= 4: [i 5][c 4][row 8][64 cols] per slot (20,480 B)
  CUtensorMap ll4;   // S 5..16, ranks 0, 1: [j 4][c 4][row 16][64] (32,768 B)
  CUtensorMap ll3;   // ranks 2..5: [j 3][c 4][row 16][64] (24,576 B)
  CUtensorMap mega;  // S 17..24: [c 10][row 16][64] (20,480 B)
};
__device__ __forceinline__ void rt_tma(void* dst, const CUtensorMap* map, int c0, int c1, int c2, int c3, int c4,
                                       uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6, %7}], [%2];" ::"r"(
          w8::smem_u32(dst)),
      "l"(map), "r"(w8::smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4)
      : "memory");
}
__device__ __forceinline__ void rt_tma4(void* dst, const CUtensorMap* map, int c0, int c1, int c2, int c3, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];" ::"r"(
          w8::smem_u32(dst)),
      "l"(map), "r"(w8::smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
      : "memory");
}
// Byte offset of byte b (< 128) of 128-B line L in a 128-B-swizzled box (1024-B aligned): 16-B chunk ^= L % 8
__device__ __forceinline__ uint32_t rt_swz(int L, int b) { return L * 128 + ((((b >> 4) ^ L) & 7) << 4) + (b & 15); }

struct RouterCommon {
  const float* bias;           // [384] e_score_correction_bias
  const float* bias_vl;        // [384] image bias, or null
  const long long* input_ids;  // [S] (S <= 16 image detection with bias_vl), or null
  const bool* image_mask;      // [S] MegaGateRoutingMetadata.image_token_mask (S > 16), or null
  uint64_t* part;              // scratch [24][8][S][16] tagged rank partials (f32 bits)
  unsigned long long* dbg = nullptr;  // tests: per-task phase stamps [task][8]
};
#define W4RT_STAMP(c, task, j) \
  if ((c).dbg != nullptr && threadIdx.x == 0) (c).dbg[(task) * 8 + (j)] = w8::globaltimer();
#define W4RT_CLOCK(c, task, j) \
  if ((c).dbg != nullptr && threadIdx.x == 0) (c).dbg[(task) * 8 + (j)] = clock64();

__device__ __forceinline__ bool rt_image(const RouterCommon& c, int S, int t) {
  if (S > kLLMaxS) return c.image_mask != nullptr && c.image_mask[t];
  if (c.bias_vl == nullptr || c.input_ids == nullptr) return false;
  const long long id = c.input_ids[t];
  return id >= kImageSentinelLo && id < kImageSentinelLo + 5;
}
__device__ __forceinline__ uint64_t* rt_part(const RouterCommon& c, int S, int nt, int z, int t, int e) {
  return c.part + ((size_t)(nt * 8 + z) * S + t) * 16 + e;
}
__device__ __forceinline__ uint64_t rt_twf(float v, uint32_t tag) { return w8::tw(__float_as_uint(v), tag); }

// Poll the R (<= 8) tagged rank-partial pairs of (t, experts e, e + 1) of tile nt until all carry `tag` (tag 0
// accepts anything); every round reloads all of them with unconditional loads (the values stay in registers: a
// conditionally loaded array lands in local memory and serializes the loads). Bounded: traps after the W8 timeout.
__device__ __forceinline__ void rt_poll_parts(const RouterCommon& c, int S, int nt, int t, int e, int R, uint32_t tag,
                                              w8::Diag* diag, uint32_t site, float (&v)[8][2]) {
  uint64_t w[8][2];
  uint64_t t0 = 0;
  uint32_t rounds = 0;
  while (true) {
#pragma unroll
    for (int z = 0; z < 8; ++z) w8::ld_tw2(rt_part(c, S, nt, z < R ? z : 0, t, e), w[z][0], w[z][1]);
    bool ok = true;
#pragma unroll
    for (int z = 0; z < 8; ++z)
      ok = ok && (z >= R || tag == 0 || (w8::tw_ok(w[z][0], tag) && w8::tw_ok(w[z][1], tag)));
    if (ok) break;
    if (t0 == 0) t0 = w8::globaltimer();
    __nanosleep(W8_SPIN_BACKOFF_NS);
    if ((++rounds & 15) == 0 && w8::globaltimer() - t0 > W8_SPIN_TIMEOUT_NS)
      w8::spin_fail(diag, site, (uint32_t)(nt * 16 + e), tag, (uint32_t)(w[0][0] >> 32));
  }
#pragma unroll
  for (int z = 0; z < 8; ++z)
    v[z][0] = __uint_as_float(w8::tw_payload(w[z][0])), v[z][1] = __uint_as_float(w8::tw_payload(w[z][1]));
}

// ------------------------------------------------------------------------------------------------ RouterGate
struct RouterGate {
  struct Params {
    RouterCommon c;
    RouterGateMaps maps;          // over gate_w [384][5120] (stage_router_maps.h)
    const uint4* seam_c;          // SeamPost's collapse scratch [S][16][40] (8 bf16 per chunk)
    const float* seam_vtp;        // SeamPost's per-virtual-thread sums of squares [S][64]
    const __nv_bfloat16* norm_w;  // [5120] the FFN RMSNorm weight
    w8::Dep dep_post;             // the SeamPost tasks (S <= 16)
    // S > 16: the layer input comes from the Mega seam (stage_mega.cuh): x1 and the 40 split partials
    const __nv_bfloat16* mega_x1 = nullptr;  // [S][5120] MegaSeamPost scratch
    const float* mega_part = nullptr;        // [S][40]
    float rms_eps = 0.f, rms_scale = 1.f;    // production: norm_eps, 1.0
    w8::Dep dep_mega_post{};                 // the 40 MegaSeamPost tasks
  };
  static constexpr const char* kName = "router_gate";
  static constexpr int kLLTiles = kTiles / 2;  // S 5..16 and 17..24: tiles of 32 experts
  __host__ __device__ static constexpr int tasks(int S) {
    return S <= kDotMaxS ? kTiles * kDotRanks : S <= kLLMaxS ? kLLTiles * kRouterSplitK : kLLTiles * mgate::kSplitK;
  }
  // ring slots per task, in production order: dot W0 W1 | LL W0 W1 A | Mega W0 W1 A0 .. A(ceil(S/16) - 1)
  // (A = borrowed, 0 tx; a Mega A slot holds 16 li rows)
  __host__ __device__ static constexpr int mega_aslots(int S) { return (S + 15) / 16; }
  __host__ __device__ static constexpr int slots(int S) { return S <= kDotMaxS ? 2 : S <= kLLMaxS ? 3 : 2 + mega_aslots(S); }
  // [0, 128): rs[S]; then dot: li [4][kDotARowE] bf16 + norm_w slice [160][16 B]; LL: chain partials
  // [8 warps][2][16][8] f32 + norm_w slice [128][16 B]; Mega: rs in [0, 256), norm_w slice [80][16 B]
  __host__ __device__ static constexpr uint32_t smem_bytes(int S) {
    return S <= kDotMaxS ? 128u + (uint32_t)kDotMaxS * kDotARowE * 2 + 160 * 16
           : S <= kLLMaxS ? 128u + 8u * 2 * 16 * 8 * 4 + 128 * 16
                          : 256u + 80u * 16;  // Mega: rs[S <= 48] in [0, 256)
  }
  __host__ __device__ static constexpr uint32_t tmem_cols(int) { return 0; }
  static size_t scratch_bytes(int S) { return (size_t)kTiles * 8 * S * 16 * 8; }

  template <class RingT>
  __device__ static void produce(const Params& p, int S, int task, const RingT& ring, uint32_t& prod_it, w8::Diag* diag) {
    static_assert(RingT::SLOT_BYTES >= kGateSlotBytes && RingT::SLOT_BYTES % 1024 == 0,
                  "RouterGate needs ring slots of >= 33,024 B, 1024-B aligned (128-B swizzled TMA)");
    w8::prefetch_l2_bulk(p.norm_w, 5120 * 2);  // the consumer's li normalization reads it after the SeamPost wait
    if (S <= kDotMaxS) {
      const int nt = task / kDotRanks, vw = task % kDotRanks;
      for (int h = 0; h < 2; ++h, ++prod_it) {
        uint8_t* slot = ring.acquire(prod_it, diag);
        rt_tma(slot, &p.maps.dot, 0, nt * 16 + h * 8, 0, vw, 0, ring.full_bar(prod_it));
        ring.submit(prod_it, 8 * 5 * 512);
      }
      return;
    }
    const int nt = task / (S <= kLLMaxS ? kRouterSplitK : mgate::kSplitK);
    const int z = task % (S <= kLLMaxS ? kRouterSplitK : mgate::kSplitK);
    for (int h = 0; h < 2; ++h, ++prod_it) {
      uint8_t* slot = ring.acquire(prod_it, diag);
      if (S <= kLLMaxS) {
        rt_tma(slot, z < 2 ? &p.maps.ll4 : &p.maps.ll3, 0, nt * 32 + h * 16, 0, z, 0, ring.full_bar(prod_it));
        ring.submit(prod_it, 16 * ll_ntiles(z) * 512);
      } else {
        rt_tma4(slot, &p.maps.mega, 0, nt * 32 + h * 16, 0, z, ring.full_bar(prod_it));
        ring.submit(prod_it, 16 * 1280);
      }
    }
    for (int q = S <= kLLMaxS ? 1 : mega_aslots(S); q > 0; --q, ++prod_it) {  // the li slice's borrowed slot(s)
      ring.acquire(prod_it, diag);
      ring.submit(prod_it, 0);
    }
  }

  template <class Sync, class RingT>
  __device__ static void consume(const Params& p, int S, int task, Sync& sync, const RingT& ring, uint32_t& ring_it,
                                 uint8_t* smem, w8::Diag* diag) {
    W4RT_STAMP(p.c, task, 0);
    if (S <= kDotMaxS)
      dot_task(p, S, task, sync, ring, ring_it, smem, diag);
    else if (S <= kLLMaxS)
      ll_task(p, S, task, sync, ring, ring_it, smem, diag);
    else
      mega_task(p, S, task, sync, ring, ring_it, smem, diag);
    W4RT_STAMP(p.c, task, 4);
  }

  // The FFN layer input of the columns col(cc) (cc < NCH 16-B chunks of 8 columns; ok(cc) false: skipped) of rows
  // < S, normalized here exactly as SeamNorm does it (TileLang butterfly of the 64 virtual-thread sums -> rs;
  // li = bf16(c * rs * w)) from SeamPost's outputs: no SeamNorm hop. SeamPost's c chunks land in A[r * rowb + cc * 16]
  // and the norm_w chunks in wsm by cp.async (one round trip, no register arrays), then are normalized in place.
  template <int NCH, class Sync, class Col, class Ok>
  __device__ static void stage_li(const Params& p, int S, Sync& sync, uint8_t* A, int rowb, Col&& col, Ok&& ok,
                                  uint8_t* wsm, uint8_t* smem, int task) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, n = S * NCH;
    float* rs = reinterpret_cast<float*>(smem);
    sync.wait(p.dep_post, 0x410);
    W4RT_STAMP(p.c, task, 5);
    for (int i = tid; i < n; i += w8::kComputeThreads) {
      const int r = i / NCH, cc = i % NCH;
      if (!ok(cc)) continue;
      const int cl = col(cc), v = (cl >> 3) & 63, ci = (cl >> 10) * 8 + ((cl >> 9) & 1) * 4 + (v & 3);  // SeamPost's
      rt_cp16(A + r * rowb + cc * 16, p.seam_c + ((size_t)r * kSlices + (v >> 2)) * kChunks + ci);       // chunk
    }
    for (int cc = tid; cc < NCH; cc += w8::kComputeThreads)
      if (ok(cc)) rt_cp16(wsm + cc * 16, p.norm_w + col(cc));
    for (int g = warp; g < S; g += w8::kComputeWarps) {
      const float tot = vt_butterfly(__ldcg(p.seam_vtp + g * kVt + lane), __ldcg(p.seam_vtp + g * kVt + lane + 32));
      if (lane == 0) rs[g] = norm_rsqrt(tot);
    }
    rt_cp_wait();
    w8::compute_sync();
    W4RT_STAMP(p.c, task, 6);
    for (int i = tid; i < n; i += w8::kComputeThreads) {
      const int r = i / NCH, cc = i % NCH;
      if (!ok(cc)) continue;
      uint4* a = reinterpret_cast<uint4*>(A + r * rowb + cc * 16);
      float c8[8], w8v[8], li[8];
      unpack8(*a, c8);
      unpack8(*reinterpret_cast<const uint4*>(wsm + cc * 16), w8v);
      const float rr = rs[r];
#pragma unroll
      for (int e = 0; e < 8; ++e) li[e] = bf2f(f2bf(norm_apply(c8[e], rr, w8v[e])));
      *a = pack8(li);
    }
    w8::compute_sync();
  }

  // ---------------------------------------------------------------- S <= 4: LLBf16Dotprod rank vw of a tile
  // Thread (warp w, lane): vt = 32 vw + lane, experts 2w and 2w + 1 of the tile; chain over k = 1024 i + 8 vt + j
  // (i < 5, j < 8) in order, butterfly 16..1. RouterTopk adds the 4 ranks from +0 (production's warp sums).
  template <class Sync, class RingT>
  __device__ static void dot_task(const Params& p, int S, int task, Sync& sync, const RingT& ring, uint32_t& ring_it,
                                  uint8_t* smem, w8::Diag* diag) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nt = task / kDotRanks, vw = task % kDotRanks;
    __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem + 128);  // [S][kDotARowE]
    const uint8_t* W = ring.wait_full(ring_it + (warp >> 2), diag);  // [i][c][row 8][64], swizzled
    const int r0 = (2 * warp) & 7;
    W4RT_STAMP(p.c, task, 1);
    stage_li<160>(p, S, sync, reinterpret_cast<uint8_t*>(sA), kDotARowE * 2,
                  [&](int c) { return (c >> 5) * 1024 + vw * 256 + (c & 31) * 8; }, [](int) { return true; },
                  smem + 128 + kDotMaxS * kDotARowE * 2, smem, task);
    W4RT_STAMP(p.c, task, 2);
    float c0[kDotMaxS], c1[kDotMaxS];
#pragma unroll
    for (int t = 0; t < kDotMaxS; ++t) c0[t] = c1[t] = 0.f;
#pragma unroll 1
    for (int i = 0; i < 5; ++i) {
      const int col = i * 256 + lane * 8, L0 = (i * 4 + (lane >> 3)) * 8 + r0;
      const uint4 b0 = *reinterpret_cast<const uint4*>(W + rt_swz(L0, (lane & 7) * 16));
      const uint4 b1 = *reinterpret_cast<const uint4*>(W + rt_swz(L0 + 1, (lane & 7) * 16));
#pragma unroll
      for (int t = 0; t < kDotMaxS; ++t)
        if (t < S) {
          const uint4 a = *reinterpret_cast<const uint4*>(sA + t * kDotARowE + col);
          c0[t] = rt_dot8(c0[t], a, b0);
          c1[t] = rt_dot8(c1[t], a, b1);
        }
    }
#pragma unroll
    for (int t = 0; t < kDotMaxS; ++t)
      if (t < S)
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          c0[t] = __fadd_rn(c0[t], __shfl_xor_sync(0xffffffffu, c0[t], off));
          c1[t] = __fadd_rn(c1[t], __shfl_xor_sync(0xffffffffu, c1[t], off));
        }
    W4RT_STAMP(p.c, task, 3);
#pragma unroll
    for (int t = 0; t < kDotMaxS; ++t)  // every lane holds the sums; lane t stores token t's pair
      if (lane == t && t < S)
        w8::st_tw2(rt_part(p.c, S, nt, vw, t, 2 * warp), rt_twf(c0[t], sync.tag), rt_twf(c1[t], sync.tag));
    w8::compute_sync();
    if (tid == 0) ring.release(ring_it), ring.release(ring_it + 1);
    ring_it += 2;
  }

  // ---------------------------------------------------------------- S 5..16: LLBf16SplitK rank z of a 32-expert tile
  // chains (w4 = warp & 3, n-atoms 2 nh and 2 nh + 1, nh = warp >> 2): k blocks w4 + 4 ki of each of the rank's
  // tiles, in order; the rank partial is the 4 w4 chains in order from +0, times 1
  template <class Sync, class RingT>
  __device__ static void ll_task(const Params& p, int S, int task, Sync& sync, const RingT& ring, uint32_t& ring_it,
                                 uint8_t* smem, w8::Diag* diag) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int nt = task / kRouterSplitK, z = task % kRouterSplitK, nj = ll_ntiles(z);
    const int w4 = warp & 3, nh = warp >> 2;
    const uint8_t* W = ring.wait_full(ring_it + nh, diag);  // n-atoms 2 nh, 2 nh + 1 = rows [0, 16) of slot nh
    uint8_t* A = ring.wait_full(ring_it + 2, diag);         // [row < S][kLLRowB]
    float* red = reinterpret_cast<float*>(smem + 128);      // [8 warps][2][16 rows][8 experts]
    W4RT_STAMP(p.c, task, 1);
    stage_li<128>(p, S, sync, A, kLLRowB,
                  [&](int c) { return (z + kRouterSplitK * (c >> 5)) * kRouterTileK + (c & 31) * 8; },
                  [&](int c) { return (c >> 5) < nj; }, smem + 128 + 8 * 2 * 16 * 8 * 4, smem, task);
    W4RT_STAMP(p.c, task, 2);
    const bool ok0 = g < S, ok1 = g + 8 < S;
    const uint8_t* a0 = A + g * kLLRowB;
    const uint8_t* a1 = A + (g + 8) * kLLRowB;
    float c[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll 1
    for (int j = 0; j < nj; ++j)
#pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        const int kq = w4 + 4 * ki;  // 16-wide K block of the tile: 64-col chunk kq / 4, bytes (kq % 4) * 32 + 4 tig
        const int kb = (j * 256 + kq * 16 + 2 * tig) * 2, L = (j * 4 + (kq >> 2)) * 16 + g, bb = (kq & 3) * 32 + 4 * tig;
        const uint32_t a[4] = {ok0 ? rt_lds32(a0 + kb) : 0u, ok1 ? rt_lds32(a1 + kb) : 0u,
                               ok0 ? rt_lds32(a0 + kb + 16) : 0u, ok1 ? rt_lds32(a1 + kb + 16) : 0u};
        rt_mma(c[0], a, rt_lds32(W + rt_swz(L, bb)), rt_lds32(W + rt_swz(L, bb + 16)));
        rt_mma(c[1], a, rt_lds32(W + rt_swz(L + 8, bb)), rt_lds32(W + rt_swz(L + 8, bb + 16)));
      }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      float* r = red + (warp * 2 + u) * 128;  // [16 rows][8 experts] of this chain
      r[g * 8 + 2 * tig] = c[u][0], r[g * 8 + 2 * tig + 1] = c[u][1];
      r[(g + 8) * 8 + 2 * tig] = c[u][2], r[(g + 8) * 8 + 2 * tig + 1] = c[u][3];
    }
    w8::fence_proxy_async_smem();  // A was written through the generic proxy; its slot may next take a TMA write
    w8::compute_sync();
    if (tid == 0) ring.release(ring_it), ring.release(ring_it + 1), ring.release(ring_it + 2);
    ring_it += 3;
    W4RT_STAMP(p.c, task, 3);
    // (t, expert pair): 16 pairs per token over the 32 experts
    if (tid < S * 16) {
      const int t = tid >> 4, pp = tid & 15, na = pp >> 2, e = (pp & 3) * 2;  // n-atom na = 2 nh' + u'
      float v[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < 4; ++q) s = __fadd_rn(s, red[(((na >> 1) * 4 + q) * 2 + (na & 1)) * 128 + t * 8 + e + u]);
        v[u] = __fmul_rn(s, 1.0f);
      }
      w8::st_tw2(rt_part(p.c, S, nt * 2 + (na >> 1), z, t, (na & 1) * 8 + e), rt_twf(v[0], sync.tag),
                 rt_twf(v[1], sync.tag));
    }
  }

  // ---------------------------------------------------------------- S 17..48: Mega gate split z of a 32-expert tile
  // The layer input of the split's 640 columns from the Mega seam: x1 rows (cp.async) normalized in place,
  // y = bf16((x1 * rs) * w), rs = rsqrtf(fma(sum of the 40 split partials in order, 1/5120, eps)) * scale.
  // Row r lives in borrowed slot r / 16, row r % 16 (stride kMgRowB).
  template <class Sync, class RowP>
  __device__ static void stage_li_mega(const Params& p, int S, Sync& sync, RowP&& rowp, int z, uint8_t* wsm,
                                       uint8_t* smem, int task) {
    constexpr int kNch = mgate::kKPerSplit / 8;  // 80
    const int tid = threadIdx.x, n = S * kNch, c0 = z * mgate::kKPerSplit;
    float* rs = reinterpret_cast<float*>(smem);
    sync.wait(p.dep_mega_post, 0x417);
    W4RT_STAMP(p.c, task, 5);
    for (int i = tid; i < n; i += w8::kComputeThreads) {
      const int r = i / kNch, cc = i % kNch;
      rt_cp16(rowp(r) + cc * 16, p.mega_x1 + (size_t)r * 5120 + c0 + cc * 8);
    }
    for (int cc = tid; cc < kNch; cc += w8::kComputeThreads) rt_cp16(wsm + cc * 16, p.norm_w + c0 + cc * 8);
    for (int g = tid; g < S; g += w8::kComputeThreads) {
      float v[kMgSplits];
#pragma unroll
      for (int s = 0; s < kMgSplits; ++s) v[s] = __ldcg(p.mega_part + g * kMgSplits + s);
      float sum = 0.f;
#pragma unroll
      for (int s = 0; s < kMgSplits; ++s) sum = __fadd_rn(sum, v[s]);
      rs[g] = __fmul_rn(rsqrtf(__fmaf_rn(sum, kMgInvHidden, p.rms_eps)), p.rms_scale);
    }
    rt_cp_wait();
    w8::compute_sync();
    W4RT_STAMP(p.c, task, 6);
    for (int i = tid; i < n; i += w8::kComputeThreads) {
      const int r = i / kNch, cc = i % kNch;
      uint4* a = reinterpret_cast<uint4*>(rowp(r) + cc * 16);
      const uint4 xv = *a, wv = *reinterpret_cast<const uint4*>(wsm + cc * 16);
      const float rr = rs[r];
      uint32_t yw[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t xa = (&xv.x)[q], wb = (&wv.x)[q];
        yw[q] = mg_bf2(__fmul_rn(__fmul_rn(mg_bf_lo(xa), rr), mg_bf_lo(wb)),
                       __fmul_rn(__fmul_rn(mg_bf_hi(xa), rr), mg_bf_hi(wb)));
      }
      *a = make_uint4(yw[0], yw[1], yw[2], yw[3]);
    }
    w8::compute_sync();
  }

  // chains (n-atom na = warp & 3, 16-token tiles warp >> 2 and, for S > 32, warp >> 2 + 2): the split's 40 K16 steps
  // from zero
  template <class Sync, class RingT>
  __device__ static void mega_task(const Params& p, int S, int task, Sync& sync, const RingT& ring, uint32_t& ring_it,
                                   uint8_t* smem, w8::Diag* diag) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int nt = task / mgate::kSplitK, z = task % mgate::kSplitK, nA = mega_aslots(S);
    const int na = warp & 3;
    const uint8_t* W = ring.wait_full(ring_it + (na >> 1), diag);
    uint8_t* A0 = ring.wait_full(ring_it + 2, diag);
    uint8_t* A1 = nA > 1 ? ring.wait_full(ring_it + 3, diag) : A0;
    uint8_t* A2 = nA > 2 ? ring.wait_full(ring_it + 4, diag) : A0;
    auto rowp = [&](int r) { return (r < 16 ? A0 : r < 32 ? A1 : A2) + (r & 15) * kMgRowB; };
    W4RT_STAMP(p.c, task, 1);
    stage_li_mega(p, S, sync, rowp, z, smem + 256, smem, task);
    W4RT_STAMP(p.c, task, 2);
    const int row = (na & 1) * 8 + g;  // [c 10][row 16][64], swizzled
    const int nt16 = nt * 2 + (na >> 1), e = (na & 1) * 8 + 2 * tig;
#pragma unroll 1
    for (int mt = warp >> 2; mt < nA; mt += 2) {
      const int r0 = mt * 16 + g, r1 = r0 + 8;
      const bool ok0 = r0 < S, ok1 = r1 < S;
      const uint8_t* a0 = rowp(r0);
      const uint8_t* a1 = rowp(r1);
      float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int st = 0; st < mgate::kKPerSplit / 16; ++st) {
        const int kb = (st * 16 + 2 * tig) * 2, L = (st >> 2) * 16 + row, bb = (st & 3) * 32 + 4 * tig;
        const uint32_t a[4] = {ok0 ? rt_lds32(a0 + kb) : 0u, ok1 ? rt_lds32(a1 + kb) : 0u,
                               ok0 ? rt_lds32(a0 + kb + 16) : 0u, ok1 ? rt_lds32(a1 + kb + 16) : 0u};
        rt_mma(c, a, rt_lds32(W + rt_swz(L, bb)), rt_lds32(W + rt_swz(L, bb + 16)));
      }
      if (ok0) w8::st_tw2(rt_part(p.c, S, nt16, z, r0, e), rt_twf(c[0], sync.tag), rt_twf(c[1], sync.tag));
      if (ok1) w8::st_tw2(rt_part(p.c, S, nt16, z, r1, e), rt_twf(c[2], sync.tag), rt_twf(c[3], sync.tag));
    }
    W4RT_STAMP(p.c, task, 3);
    w8::fence_proxy_async_smem();
    w8::compute_sync();
    if (tid == 0)
      for (int q = 0; q < 2 + nA; ++q) ring.release(ring_it + q);
    ring_it += 2 + nA;
  }
};

// ------------------------------------------------------------------------------------------------ RouterTopk
// One token per task. Polls the token's 384 x R rank partials (tagged pairs, one L2 round trip), sums them in
// production order, scores every expert into SMEM; warp 0 selects the top-6 (score desc, lowest id on ties: lane l
// holds experts l + 32 i), computes the routing weights, writes topk ids / weights, hot ids, MegaMoE topk_idx (plain
// and/or tagged) / topk_weights and has[t], then sends the token (W6's send_token_warp pattern: the row's MXFP8 bytes
// from SeamNorm, __syncwarp, one st.release.sys of word0, cumulative over the warp's writes).
struct RouterTopk {
  struct Params {
    RouterCommon c;
    float* logits;             // [S][384] (tests), or null
    const int32_t* row_map;    // [384] hot row, or -(cold row + 1)
    const bool* is_padding;    // [S] or null (MegaMoE staging masks padded rows)
    float routed_scale;        // 1.5
    long long* topk_ids;       // [S][6] or null
    float* topk_w;             // [S][6] or null
    long long* hot_ids;        // [S][6] or null
    long long* mega_idx;       // [S][6] MegaMoE topk_idx, or null
    uint64_t* mega_idx_tw;     // [S][6] the same as W8 tagged words {payload (u32)topk_idx, launch tag}, or null
    float* mega_w;             // [S][6] MegaMoE topk_weights, or null
    uint64_t* mega_w_tw;       // [S][6] the same as tagged words {payload f32 bits, launch tag}, or null
    int8_t* has;               // [S] a route of the token is cold (the finish reads it)
    const uint8_t* fi_q;       // [S][5120] FlashInfer MXFP8 of li (SeamNorm fi_q)
    const uint8_t* fi_sf_lin;  // [S][160] its ue8m0 scales (SeamNorm fi_sf_lin)
    uint8_t* p3_seg;           // device view of the host-mapped v3 segment, or null (no send)
    const uint32_t* p3_state;  // [2] peer3 state; [0] = this call's counter
    int layer;
    w8::Dep dep_norm;          // the SeamNorm tasks (fi_q / fi_sf_lin); the partials are polled (tagged)
  };
  static constexpr const char* kName = "router_topk";
  __host__ __device__ static constexpr int tasks(int S) { return S; }
  __host__ __device__ static constexpr uint32_t smem_bytes(int) { return 5 * kExperts * 4; }
  __host__ __device__ static constexpr uint32_t tmem_cols(int) { return 0; }
  static size_t scratch_bytes(int) { return 0; }

  template <class Sync>
  __device__ static void consume(const Params& p, int S, int t, Sync& sync, uint8_t* smem) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, R = gate_ranks(S);
    int32_t* rmap = reinterpret_cast<int32_t*>(smem);  // constants, staged before the waits
    float* sb = reinterpret_cast<float*>(smem + kExperts * 4);
    float* sbv = sb + kExperts;
    uint32_t* skey = reinterpret_cast<uint32_t*>(sbv + kExperts);  // [384] score keys
    float* swr = reinterpret_cast<float*>(skey + kExperts);         // [384] raw weights
    for (int i = tid; i < kExperts; i += w8::kComputeThreads) {
      rmap[i] = __ldg(p.row_map + i);
      sb[i] = __ldg(p.c.bias + i);
      sbv[i] = p.c.bias_vl != nullptr ? __ldg(p.c.bias_vl + i) : 0.f;
    }
    const bool img = rt_image(p.c, S, t);
    const bool pad = p.is_padding != nullptr && p.is_padding[t];
    W4RT_STAMP(p.c, 200 + t, 0);
    sync.wait(p.dep_norm, 0x411);  // SeamNorm's rows visible (acquire); its compute_sync publishes the constants
    uint4 row[10];
    uint2 sf = make_uint2(0, 0);
    if (warp == 0 && p.p3_seg != nullptr) {  // prefetch the token's MXFP8 row and scales (sent if a route is cold)
      const uint4* src = reinterpret_cast<const uint4*>(p.fi_q + (size_t)t * 5120);
#pragma unroll
      for (int i = 0; i < 10; ++i) row[i] = __ldcg(src + lane + 32 * i);
      if (lane < 20) sf = __ldcg(reinterpret_cast<const uint2*>(p.fi_sf_lin + (size_t)t * 160) + lane);
    }
    // thread q < 192: experts 2q, 2q + 1 (tile q / 8, pair q % 8): the R rank partials, polled as tagged pairs
    {
      const bool item = tid < kExperts / 2;
      const int q = item ? tid : 0, nt = q >> 3, ep = q & 7;
      float v[8][2];  // threads >= 192 re-read pair 0 with tag 0 (accept anything): unconditional, in registers
      rt_poll_parts(p.c, S, nt, t, 2 * ep, R, item ? sync.tag : 0u, sync.diag, 0x413, v);
      if (item) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          float lg;
          if (S <= kLLMaxS) {  // dotprod warp sums / LLBf16SplitK rank partials: in order from +0
            lg = 0.f;
#pragma unroll
            for (int z = 0; z < 8; ++z)
              if (z < R) lg = __fadd_rn(lg, v[z][u]);
          } else {  // Mega gate: split 0 + split 1 + ...
            lg = v[0][u];
#pragma unroll
            for (int z = 1; z < mgate::kSplitK; ++z) lg = __fadd_rn(lg, v[z][u]);
          }
          const int e = 2 * q + u;
          if (p.logits != nullptr) p.logits[(size_t)t * kExperts + e] = lg;
          const float wraw = S <= kLLMaxS ? sqrt_softplus(lg)                          // dsv4_topk's recipe
                                          : sqrtf(lg > 20.0f ? lg : log1pf(expf(lg)));  // Mega gate's precise one
          const float sc = __fadd_rn(wraw, img ? sbv[e] : sb[e]);
          skey[e] = rt_score_key(sc == sc ? sc : -1e30f);
          swr[e] = wraw;
        }
      }
    }
    W4RT_STAMP(p.c, 200 + t, 1);
    w8::compute_sync();
    if (warp != 0) return;
    // top-6: lane l holds experts l + 32 i (i < 12); per round the best (key, id) of each lane, the warp's max key,
    // the lowest id among the lanes at that key; the winner (lane id & 31) drops the expert
    uint32_t key[12];
#pragma unroll
    for (int i = 0; i < 12; ++i) key[i] = skey[lane + 32 * i];
    float s_k = 0.f;
    int e_k = 0;
#pragma unroll 1
    for (int r = 0; r < kTopk; ++r) {
      uint32_t bk = key[0];
      int bi = 0;
#pragma unroll
      for (int i = 1; i < 12; ++i)
        if (key[i] > bk) bk = key[i], bi = i;
      const uint32_t mk = __reduce_max_sync(0xffffffffu, bk);
      const int mid = (int)__reduce_min_sync(0xffffffffu, bk == mk ? (uint32_t)(lane + 32 * bi) : 0xffffffffu);
      if (lane == r) s_k = swr[mid], e_k = mid;
      const bool win = lane == (mid & 31);
#pragma unroll
      for (int i = 0; i < 12; ++i) key[i] = win && i == (mid >> 5) ? 0u : key[i];  // select: keys stay in registers
    }
    W4RT_STAMP(p.c, 200 + t, 2);
    float wt;
    if (S <= kLLMaxS) {
      // dsv4_topk: ((w0 + w4) + w2) + ((w1 + w5) + w3), routed_scale / sum (div.full), times each weight
      const float w0 = __shfl_sync(0xffffffffu, s_k, 0), w1 = __shfl_sync(0xffffffffu, s_k, 1);
      const float w2 = __shfl_sync(0xffffffffu, s_k, 2), w3 = __shfl_sync(0xffffffffu, s_k, 3);
      const float w4 = __shfl_sync(0xffffffffu, s_k, 4), w5 = __shfl_sync(0xffffffffu, s_k, 5);
      const float ws = __fadd_rn(__fadd_rn(__fadd_rn(w0, w4), w2), __fadd_rn(__fadd_rn(w1, w5), w3));
      const float f = div_full_f32(p.routed_scale, ws > 0.f ? ws : 1.0f);
      wt = __fmul_rn(f, s_k);
    } else {
      // DeepGEMM store_topk_token: warp_reduce_sum<8> (xor 4, 2, 1) + 1e-20, then s / sum * scale
      float sum = s_k;
      sum = sum + __shfl_xor_sync(0xffffffffu, sum, 4);
      sum = sum + __shfl_xor_sync(0xffffffffu, sum, 2);
      sum = sum + __shfl_xor_sync(0xffffffffu, sum, 1);
      sum = sum + 1e-20f;
      wt = s_k / sum * p.routed_scale;
    }
    int cold = -1;
    if (lane < kTopk) {
      const int rm = rmap[e_k];
      cold = rm < 0 ? -rm - 1 : -1;
      const int hot = rm >= 0 ? rm : -1;
      const int o = t * kTopk + lane;
      if (p.mega_idx_tw != nullptr) w8::st_tw(p.mega_idx_tw + o, w8::tw((uint32_t)(pad ? -1 : hot), sync.tag));
      if (p.mega_w_tw != nullptr) w8::st_tw(p.mega_w_tw + o, rt_twf(pad ? 0.f : wt, sync.tag));
      if (p.mega_idx != nullptr) p.mega_idx[o] = pad ? -1 : hot;
      if (p.mega_w != nullptr) p.mega_w[o] = pad ? 0.f : wt;
      if (p.topk_ids != nullptr) p.topk_ids[o] = e_k;
      if (p.topk_w != nullptr) p.topk_w[o] = wt;
      if (p.hot_ids != nullptr) p.hot_ids[o] = hot;
    }
    const bool any = __ballot_sync(0xffffffffu, lane < kTopk && cold >= 0) != 0;
    if (lane == 0) p.has[t] = any;
    W4RT_STAMP(p.c, 200 + t, 3);
    if (p.p3_seg == nullptr) return;
    uint64_t ids48 = 0;
#pragma unroll
    for (int k = 0; k < kTopk; ++k) {
      const int c = __shfl_sync(0xffffffffu, cold, k);
      ids48 |= (uint64_t)(c >= 0 ? (uint32_t)c & 0xFFu : 0xFFu) << (8 * k);
    }
    uint8_t* rec = p.p3_seg + kP3OffRec + (size_t)t * kP3RecBytes;
    if (any) {
      if (lane < kTopk) reinterpret_cast<float*>(rec + 8)[lane] = wt;
      uint4* dst = reinterpret_cast<uint4*>(p.p3_seg + kP3OffXq + (size_t)t * 5120);
#pragma unroll
      for (int i = 0; i < 10; ++i) dst[lane + 32 * i] = row[i];
      if (lane < 20) reinterpret_cast<uint2*>(p.p3_seg + kP3OffXs + (size_t)t * 160)[lane] = sf;
    }
    __syncwarp();
    if (lane == 0) {  // one release of word0, cumulative over the warp's writes (W6 section 6)
      const uint32_t tag = (p.p3_state[0] << 12) | ((uint32_t)p.layer << 6) | (uint32_t)(S - 1);
      asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(rec), "l"(((uint64_t)tag << 48) | ids48) : "memory");
    }
    W4RT_STAMP(p.c, 200 + t, 4);
  }
};

}  // namespace w4::stage
