// W4 router: gate GEMM (bit-exact vs ll_bf16_gemm: LLBf16Dotprod for T <= 4, LLBf16SplitK for 5 <= T <= 16) +
// dsv4_topk-exact top-6 + hot/cold split + MegaMoE staging + an optional peer-v2 cold-row send, in one launch.
//
// Grid (6 ranks, 24 n-tiles of 16 experts), cluster (6, 1, 1), 256 threads; cluster nt owns experts [16 nt, 16 nt + 16).
//   T >= 5: CTA (z, nt) runs LLBf16SplitK's rank-z chains on warps 0-3.
//   T <= 4: the cluster's 48 warps run the 64 (expert, dotprod warp) tasks of LLBf16Dotprod (bs 128, 8-wide vectors:
//           virtual thread v = 32 w + lane sums k = 1024 i + 8 v + j, i < 5, j < 8, in that order; butterfly 16..1).
// The partials (split-k ranks, or dotprod warps) meet in the cluster leader (rank 0) through DSMEM (st.async +
// mbarrier) and are summed there in the stock order. Each leader then
//   - quantizes K tile nt of li (staged in shared memory with the A operand) for MegaMoE (prepare_megamoe recipe) and
//     for the sidecar (FlashInfer recipe); with a v3 send it writes that slice of EVERY token's MXFP8 row at once
//     (rows are indexed by token; the sidecar reads only the rows its records mark),
//   - turns its 16 logits per token into sqrt-softplus weights and biased scores and keeps its local top-6
//     (score desc, id asc): the global top-6 is the top-6 of these 24 x 6 candidates, so selection is unchanged,
//   - stores them in the candidate mailbox (no counter: each word starts as the sentinel 0xFFFFFFFF, which no
//     candidate word can be, and its reader polls the words themselves; two mailboxes alternate by launch parity,
//     sync[2], and each leader re-arms its slots of the other one for the next launch).
// Leader 0 polls the mailbox, merges the candidates (one warp per token, redux argmax), routes and writes the
// outputs. v3 send: the other leaders arrive on sync[1] (release) after their row slices; leader 0 acquires all 23
// arrivals (by then long done) and publishes each token's record (release, sys). Peer v2 (rows by cold position):
// every leader merges, copies its slice of each cold row, arrives on sync[1]; the last one publishes.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "w4/mhc_core.cuh"
#include "w4/params.h"
#include "w4/route_core.cuh"

namespace {

using namespace w4;

constexpr int kThreads = 256;
constexpr int kRowPad = kRouterTileK + 8;  // bf16 row stride in smem: conflict-free fragment loads
constexpr int kMaxTiles = 4;               // ceil(20 / 6)
constexpr int kClusters = kExperts / 16;   // 24
constexpr int kDotMaxT = 4;                // ll_bf16 takes the dotprod kernel for M <= 4
constexpr int kDotWarps = 4;               // LLBf16Dotprod: bs 128 = 4 warps per expert
constexpr int kDotTiles = 5120 / 1024;     // 128 threads x 8-wide vectors per K tile
constexpr int kDotTasks = 16 * kDotWarps;  // per cluster
constexpr int kDotBytes = kDotTasks * kDotMaxT * 4;
// peer tier v3 segment layout (W6 peer3_proto.cuh; w4_abi(9..11) exposes these for the boot-time check)
constexpr uint64_t kP3OffRec = 4096, kP3RecBytes = 128;
constexpr uint64_t kP3OffXq = kP3OffRec + 64 * kP3RecBytes;
constexpr uint64_t kP3OffXs = kP3OffXq + 64ull * 5120;
constexpr uint64_t kP3OffReady = 1152;  // u32 host ready word (COPY_AHEAD): the router arms 0, the RTX sets 1 after done

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint32_t ld_acquire(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release(uint32_t* p, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_relaxed(const void* p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed(void* p, uint32_t v) {
  asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
constexpr uint32_t kBoxEmpty = 0xFFFFFFFFu;            // mailbox sentinel (never a candidate's score, weight or id)
constexpr int kBoxWords = 16 * kClusters * kTopk;          // one mailbox (T <= 16), per array: 2304 words
// Order-preserving map of a float score to u32 (larger score -> larger key); +0 and -0 are made equal first.
__device__ __forceinline__ uint32_t score_key(float v) {
  const uint32_t b = __float_as_uint(__fadd_rn(v, 0.0f));
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ uint4 ld_stream16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
// acc += a * b over 8 bf16 pairs, in element order (bf16 x bf16 products are exact in fp32, so fma == mul + add).
__device__ __forceinline__ float dot8_seq(float acc, uint4 a, uint4 b) {
  const uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc = __fmaf_rn(__uint_as_float(av[i] << 16), __uint_as_float(bv[i] << 16), acc);
    acc = __fmaf_rn(__uint_as_float(av[i] & 0xffff0000u), __uint_as_float(bv[i] & 0xffff0000u), acc);
  }
  return acc;
}
__device__ __forceinline__ void push_f32(uint32_t local_dst, uint32_t local_bar, float v) {
  uint32_t dst, dbar;
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(dst) : "r"(local_dst));
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(dbar) : "r"(local_bar));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(dst),
               "r"(__float_as_uint(v)), "r"(dbar)
               : "memory");
}
#define RSTAMP(i) \
  if (p.dbg != nullptr && threadIdx.x == 0 && ((i) >= 4 || (blockIdx.x == 0 && blockIdx.y == 0))) p.dbg[i] = gtimer();
// Per-CTA stamps (debug only, when the host sets dbg[7] = 1): dbg[8 + 4 cta + j].
#define CSTAMP(j) \
  if (p.dbg != nullptr && threadIdx.x == 0 && p.dbg[7] == 1)                                         \
    p.dbg[8 + 4 * (blockIdx.y * kRouterSplitK + blockIdx.x) + (j)] = gtimer();

__global__ void __launch_bounds__(kThreads, 2) router_kernel(const RouterParams p) {  // 2 CTAs/SM: all 24 clusters co-resident
  __shared__ __align__(16) __nv_bfloat16 sW[kMaxTiles][16][kRowPad];
  __shared__ __align__(16) __nv_bfloat16 sA[kMaxTiles][16][kRowPad];
  __shared__ float red[4][256];
  __shared__ float part[kRouterSplitK][256];
  __shared__ float logit_sh[16][16];
  __shared__ __align__(16) uint8_t sq[16][kRouterTileK];  // FlashInfer MXFP8 of this leader's K tile, per token
  __shared__ uint8_t ssf[16][kRouterTileK / 32];
  __shared__ __align__(16) int32_t c_rowmap[kExperts];
  __shared__ __align__(16) float c_bias[16], c_bvl[16];
  __shared__ __align__(8) uint64_t bar;
  __shared__ int flag;
  __shared__ int8_t s_has[16];
  __shared__ int32_t s_pos[16];
  __shared__ int32_t s_cold[16 * kTopk];
  __shared__ float s_wts[16 * kTopk];
  __shared__ int s_sel_id[16][kTopk];
  __shared__ float s_sel_w[16][kTopk];
  __shared__ __align__(16) __nv_bfloat16 sQ[16][kRouterTileK];  // leader: K tile nt of li (rows < T)
  __shared__ float s_cv[16][kTopk], s_cw[16][kTopk];            // leader: its local top-6 (score, weight, id)
  __shared__ int s_ci[16][kTopk];
  __shared__ int8_t s_img[16], s_pad[16];                        // leader: per-token image flag / padding flag
  __shared__ uint32_t s_epoch;                                   // leader: this launch's mailbox counter (sync[2])
  __shared__ uint32_t s_p3ctr;                                   // leader 0: the peer3 call counter (record tag)
  const int z = blockIdx.x, nt = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t rank = cluster_rank();
  const int ntiles = (kRouterKTiles - z + kRouterSplitK - 1) / kRouterSplitK;
  const int T = p.T;
  const bool dot = T <= kDotMaxT;  // launch-uniform
  RSTAMP(0);
  CSTAMP(0);
  if (rank == 0 && tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar)));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bar)),
                 "r"(dot ? kDotBytes : kRouterSplitK * 256 * 4));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  // Weights before the dependency wait. Dotprod tasks of warp wi = 8 z + w: q = wi and q = wi + 48 (< 64), expert
  // 16 nt + q / 4, dotprod warp q % 4 (the same for both, so they share the activation loads).
  const int dwi = z * 8 + warp, dvw = dwi & 3, dvt = dvw * 32 + lane;
  const bool dtask1 = dwi + 48 < kDotTasks;
  uint4 dW[2][kDotTiles];
  if (dot) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int col = (dwi + 48 * q) >> 2;
#pragma unroll
      for (int i = 0; i < kDotTiles; ++i)
        dW[q][i] = (q == 0 || dtask1) ? ld_stream16(p.gate_w + (size_t)(nt * 16 + col) * 5120 + i * 1024 + dvt * 8)
                                      : make_uint4(0, 0, 0, 0);
    }
  } else {
    for (int i = tid; i < ntiles * 16 * (kRouterTileK / 8); i += kThreads) {
      const int j = i / (16 * 32), rem = i % (16 * 32), n = rem / 32, c8 = rem % 32;
      const int k = (z + j * kRouterSplitK) * kRouterTileK + c8 * 8;
      cp_async16(&sW[j][n][c8 * 8], p.gate_w + (size_t)(nt * 16 + n) * 5120 + k);
    }
  }
  if (rank == 0) {  // constants the leader needs after the wait
    if (tid < kExperts / 4) cp_async16(&c_rowmap[tid * 4], p.row_map + tid * 4);
    if (tid >= 96 && tid < 100) cp_async16(&c_bias[(tid - 96) * 4], p.bias + nt * 16 + (tid - 96) * 4);
    if (p.bias_vl != nullptr && tid >= 100 && tid < 104)
      cp_async16(&c_bvl[(tid - 100) * 4], p.bias_vl + nt * 16 + (tid - 100) * 4);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  // MegaMoE's prologue may overlap us. With the FFN-side mix as our PDL dependent (trigger_late, M3) the trigger
  // comes after our own dependency wait: the mix reads the seam's output and has no wait of its own.
  if (!p.trigger_late) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (p.trigger_late) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  RSTAMP(1);
  CSTAMP(1);
  // Leader: the step's inputs it needs later (off the routing path), and its quantization tile with the A operand.
  const bool need_q = rank == 0 && nt < kRouterKTiles && (p.prep_x != nullptr || p.peer_words != nullptr ||
                                                          p.p3_seg != nullptr || p.quant_cold);
  // The scalar loads go to registers (a load blocks only at its first use) and reach shared memory in the leader
  // section, so they overlap the A load and the MMA instead of gating the A phase's barrier.
  long long r_id = 0;
  bool r_pad = false;
  uint32_t r_word = 0;
  if (rank == 0) {
    if (tid < T) {
      if (p.bias_vl != nullptr) r_id = p.input_ids[tid];
      if (p.is_padding != nullptr) r_pad = p.is_padding[tid];
    }
    if (tid == 32) r_word = ld_relaxed(p.sync + 2);
    if (tid == 64 && nt == 0 && p.p3_state != nullptr) r_word = p.p3_state[0];
    if (need_q)
      for (int i = tid; i < T * (kRouterTileK / 8); i += kThreads)
        cp_async16(&sQ[i / 32][(i % 32) * 8], p.li + (size_t)(i / 32) * 5120 + nt * kRouterTileK + (i % 32) * 8);
  }
  float dacc[2][kDotMaxT];
  if (dot) {
    // LLBf16Dotprod's per-thread chains (each in k order; the tiles stream through), then its warp butterfly (all
    // lanes end with the warp sum).
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
      for (int t = 0; t < kDotMaxT; ++t) dacc[q][t] = 0.f;
#pragma unroll
    for (int i = 0; i < kDotTiles; ++i) {
      uint4 a[kDotMaxT];
#pragma unroll
      for (int t = 0; t < kDotMaxT; ++t)
        a[t] = t < T ? __ldcg(reinterpret_cast<const uint4*>(p.li + (size_t)t * 5120 + i * 1024 + dvt * 8))
                     : make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int q = 0; q < 2; ++q)
#pragma unroll
        for (int t = 0; t < kDotMaxT; ++t) dacc[q][t] = dot8_seq(dacc[q][t], a[t], dW[q][i]);
    }
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
      for (int t = 0; t < kDotMaxT; ++t)
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          dacc[q][t] = __fadd_rn(dacc[q][t], __shfl_xor_sync(0xffffffffu, dacc[q][t], off));
  } else {
    for (int i = tid; i < ntiles * 16 * (kRouterTileK / 8); i += kThreads) {
      const int j = i / (16 * 32), rem = i % (16 * 32), r = rem / 32, c8 = rem % 32;
      const int k = (z + j * kRouterSplitK) * kRouterTileK + c8 * 8;
      if (r < T)
        cp_async16(&sA[j][r][c8 * 8], p.li + (size_t)r * 5120 + k);
      else
        *reinterpret_cast<uint4*>(&sA[j][r][c8 * 8]) = make_uint4(0, 0, 0, 0);
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  RSTAMP(2);
  CSTAMP(2);
  // LLBf16SplitK chains: warp w (< 4), k-blocks w + 4 ki of each of this rank's tiles, 2 n-atoms of 8 experts.
  if (!dot && warp < 4) {
    const int g = lane >> 2, tig = lane & 3;
    float c[2][4] = {};
    for (int j = 0; j < ntiles; ++j) {
#pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        const int k0 = (warp + 4 * ki) * 16;
        uint32_t a[4];
        a[0] = *reinterpret_cast<const uint32_t*>(&sA[j][g][k0 + 2 * tig]);
        a[1] = *reinterpret_cast<const uint32_t*>(&sA[j][g + 8][k0 + 2 * tig]);
        a[2] = *reinterpret_cast<const uint32_t*>(&sA[j][g][k0 + 2 * tig + 8]);
        a[3] = *reinterpret_cast<const uint32_t*>(&sA[j][g + 8][k0 + 2 * tig + 8]);
#pragma unroll
        for (int na = 0; na < 2; ++na) {
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sW[j][na * 8 + g][k0 + 2 * tig]);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sW[j][na * 8 + g][k0 + 2 * tig + 8]);
          mma_bf16_16816(c[na], a, b0, b1);
        }
      }
    }
#pragma unroll
    for (int na = 0; na < 2; ++na) {
      const int col = na * 8 + 2 * tig;
      red[warp][g * 16 + col] = c[na][0];
      red[warp][g * 16 + col + 1] = c[na][1];
      red[warp][(g + 8) * 16 + col] = c[na][2];
      red[warp][(g + 8) * 16 + col + 1] = c[na][3];
    }
  }
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");  // the leader's mbarrier is initialized
  if (dot) {
    // Dotprod warp partials -> leader part[warp][16 t + expert]; lane t < 4 pushes token t (zero past T).
    if (lane < kDotMaxT) {
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        if (q == 1 && !dtask1) break;
        float v = dacc[q][0];
#pragma unroll
        for (int t = 1; t < kDotMaxT; ++t) v = lane == t ? dacc[q][t] : v;
        const int col = (dwi + 48 * q) >> 2;
        push_f32(smem_u32(&part[dvw][lane * 16 + col]), smem_u32(&bar), lane < T ? v : 0.f);
      }
    }
  } else {
    // Warp partials -> this rank's partial (sequential from +0), pushed to the leader (st.async + complete_tx).
    const int e = tid;  // 256 elements, one per thread
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) s = __fadd_rn(s, red[w][e]);
    s = __fmul_rn(s, 1.0f);
    push_f32(smem_u32(&part[z][e]), smem_u32(&bar), s);
  }
  if (rank != 0) {
    CSTAMP(3);
    return;
  }
  // ---------------------------------------------------------------- cluster leader
  if (tid < 16) {
    s_img[tid] = p.bias_vl != nullptr && tid < T && r_id >= kImageSentinelLo && r_id < kImageSentinelLo + 5;
    s_pad[tid] = r_pad;
  }
  if (tid == 32) s_epoch = r_word;
  if (tid == 64) s_p3ctr = r_word;
  // While the rank partials arrive: quantize K tile nt of li (MegaMoE staging; FlashInfer MXFP8 for the sidecar).
  const bool send = p.peer_words != nullptr || p.p3_seg != nullptr;
  const bool v3 = p.p3_seg != nullptr;
  if (need_q) {
    for (int base = 0; base < T * 32; base += kThreads) {
      const int item = base + tid;
      const bool active = item < T * 32;
      const int t = active ? item / 32 : 0, c8 = item % 32;
      const int col = nt * kRouterTileK + c8 * 8;
      float f[8];
      unpack8(active ? *reinterpret_cast<const uint4*>(&sQ[t][c8 * 8]) : make_uint4(0, 0, 0, 0), f);
      float am = amax8(f);
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
      const bool lead = active && (c8 & 3) == 0;
      if (p.prep_x != nullptr) {
        const uint32_t pe = prep_exp(am);
        if (active) *reinterpret_cast<uint2*>(p.prep_x + (size_t)t * 5120 + col) = prep_quant8(f, prep_inv(pe));
        if (lead) {
          const int grp = col / 128, byte = (col % 128) / 32;
          reinterpret_cast<uint8_t*>(p.prep_sf)[((size_t)t * p.prep_sf_sm + (size_t)grp * p.prep_sf_sk) * 4 + byte] =
              (uint8_t)pe;
          if (p.prep_shared_sf != nullptr) {
            const int row = prep_shared_row(t, p.shared_block_m);
            reinterpret_cast<uint8_t*>(
                p.prep_shared_sf)[((size_t)row * p.prep_ssf_sm + (size_t)grp * p.prep_ssf_sk) * 4 + byte] = (uint8_t)pe;
          }
        }
      }
      if (send || p.quant_cold) {
        const uint32_t e8 = fi_ue8m0(__fmul_rn(am, kInvE4m3Max));
        const uint2 q = fi_quant8(f, fi_inv_scale(e8));
        if (active) {
          *reinterpret_cast<uint2*>(&sq[t][c8 * 8]) = q;
          if (p.quant_cold && p.cold_q != nullptr) *reinterpret_cast<uint2*>(p.cold_q + (size_t)t * 5120 + col) = q;
          if (v3) *reinterpret_cast<uint2*>(p.p3_seg + kP3OffXq + (size_t)t * 5120 + col) = q;  // every token's row
        }
        if (lead) {
          ssf[t][c8 / 4] = (uint8_t)e8;
          if (p.quant_cold && p.cold_sf != nullptr) p.cold_sf[t * 160 + col / 32] = (uint8_t)e8;
          if (v3) p.p3_seg[kP3OffXs + (size_t)t * 160 + col / 32] = (uint8_t)e8;
        }
      }
    }
  }
  asm volatile(
      "{\n .reg .pred p;\n W4R_WAIT%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
      " @!p bra W4R_WAIT%=;\n}\n" ::"r"(smem_u32(&bar))
      : "memory");
  RSTAMP(3);
  CSTAMP(3);
  {
    const int e = tid, r = e / 16, col = e % 16;
    float s = 0.f;
    if (dot) {  // LLBf16Dotprod: thread 0 adds the 4 warp sums in order, from +0
      if (r < kDotMaxT)
#pragma unroll
        for (int w = 0; w < kDotWarps; ++w) s = __fadd_rn(s, part[w][e]);
    } else {
#pragma unroll
      for (int zz = 0; zz < kRouterSplitK; ++zz) s = __fadd_rn(s, part[zz][e]);
    }
    logit_sh[r][col] = s;
    if (p.logits != nullptr && r < T) p.logits[r * kExperts + nt * 16 + col] = s;
  }
  __syncthreads();
  // Local top-6 of this leader's 16 experts (score desc, id asc, as dsv4_topk selects): warp w takes tokens w and
  // w + 8, lane c < 16 one expert; argmax by redux over the order-preserving score key.
  for (int t = warp; t < T; t += kThreads / 32) {
    const bool image = s_img[t];
    const int c = lane & 15, e = nt * 16 + c;
    const float wv = sqrt_softplus(logit_sh[t][c]);
    const float v = __fadd_rn(wv, image ? c_bvl[c] : c_bias[c]);
    const float cur = v == v ? v : -1e30f;
    uint32_t key = lane < 16 ? score_key(cur) : 0u;
#pragma unroll
    for (int k = 0; k < kTopk; ++k) {
      const uint32_t mk = __reduce_max_sync(0xffffffffu, key);
      const int win = __ffs(__ballot_sync(0xffffffffu, key == mk)) - 1;  // lowest lane = lowest expert id
      if (lane == win) {
        s_cv[t][k] = cur;
        s_cw[t][k] = wv;
        s_ci[t][k] = e;
        key = 0u;
      }
    }
  }
  __syncthreads();
  // Into this launch's mailbox (no fence: the words carry their own validity; the outputs reach MegaMoE by kernel
  // completion, the v3 rows reach the sidecar through the sync[1] arrivals below). Then re-arm this leader's slots
  // of the other mailbox (read again two launches from now, after this kernel completed).
  const uint32_t epoch = s_epoch;
  const size_t box = (size_t)(epoch & 1u) * kBoxWords, rearm = (size_t)((epoch & 1u) ^ 1u) * kBoxWords;
  uint32_t* const bv = reinterpret_cast<uint32_t*>(p.cand_v);
  uint32_t* const bw = reinterpret_cast<uint32_t*>(p.cand_w);
  uint32_t* const bi = reinterpret_cast<uint32_t*>(p.cand_i);
  if (tid < T * kTopk) {
    const int t = tid / kTopk, k = tid % kTopk;
    const size_t o = box + ((size_t)t * kClusters + nt) * kTopk + k;
    const float w = s_cw[t][k];
    st_relaxed(bv + o, __float_as_uint(s_cv[t][k]));         // never NaN (cur above): never the sentinel
    st_relaxed(bw + o, w == w ? __float_as_uint(w) : 0x7FFFFFFFu);  // a NaN weight goes canonical
    st_relaxed(bi + o, (uint32_t)s_ci[t][k]);                 // < 384
  }
  if (tid < 16 * kTopk) {
    const size_t o = rearm + ((size_t)(tid / kTopk) * kClusters + nt) * kTopk + tid % kTopk;
    bv[o] = kBoxEmpty, bw[o] = kBoxEmpty, bi[o] = kBoxEmpty;
  }
  const bool writer = nt == 0;
  const bool all_merge = send && !v3;  // peer v2 (rows by cold position): every leader needs the routing
  if (!writer && !all_merge) {
    if (v3 && tid == 0) {  // arrive after this leader's row slices (every thread's, ordered before by the barrier)
      if (p.lean_fence) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p.sync + 1) : "memory");
      } else {
        asm volatile("fence.acq_rel.sys;" ::: "memory");
        atomicAdd(p.sync + 1, 1u);
      }
    }
    return;
  }
  // Leader 0 (peer v2: every leader) merges the 24 x 6 candidates of every token (warp per token); leader 0 writes
  // the outputs. Lane l < 24 polls leader l's 6 candidates of the warp's tokens in the mailbox until none of its words
  // is the sentinel; they are in selection order (score desc, id asc), so the global top-6 is a 24-way merge of these
  // lists: per round a redux max of the list heads, a ballot of the lanes holding it (a redux min of their ids only
  // on an exact score tie), and the winner pops its head. Both tokens of a warp are merged in the same loop so their
  // chains interleave.
  uint32_t keys[2][kTopk];
  int idss[2][kTopk];
  float wvs[2][kTopk];
  {
    uint32_t rv[2][kTopk], rw[2][kTopk], ri[2][kTopk];
    const unsigned long long t0 = gtimer();
    for (;;) {
      bool ok = true;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int t = warp + j * (kThreads / 32);
        if (lane < kClusters && t < T) {
          const size_t o = box + ((size_t)t * kClusters + lane) * kTopk;
#pragma unroll
          for (int i = 0; i < kTopk; ++i) {
            rv[j][i] = ld_relaxed(bv + o + i);
            rw[j][i] = ld_relaxed(bw + o + i);
            ri[j][i] = ld_relaxed(bi + o + i);
            ok &= rv[j][i] != kBoxEmpty && rw[j][i] != kBoxEmpty && ri[j][i] != kBoxEmpty;
          }
        }
      }
      if (__all_sync(0xffffffffu, ok)) break;
      // a lost leader must not wedge the GPU; the bound is long: time descheduled by another context counts too
      if (gtimer() - t0 > 20000000000ull) __trap();
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const bool ok = lane < kClusters && warp + j * (kThreads / 32) < T;
#pragma unroll
      for (int i = 0; i < kTopk; ++i) {
        keys[j][i] = ok ? score_key(__uint_as_float(rv[j][i])) : 0u;
        idss[j][i] = ok ? (int)ri[j][i] : 0x7fffffff;
        wvs[j][i] = ok ? __uint_as_float(rw[j][i]) : 0.f;
      }
    }
  }
  RSTAMP(4);
  // v3: the other leaders' arrivals (sync[1], after their row slices), read relaxed now so the load overlaps the
  // merge; checked (and acquired) before the records.
  uint32_t arrived = v3 && tid == 0 ? ld_relaxed(p.sync + 1) : 0u;
#pragma unroll
  for (int k = 0; k < kTopk; ++k) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int t = warp + j * (kThreads / 32);
      if (t < T) {  // warp-uniform
        const uint32_t bk = keys[j][0];
        const int bi = idss[j][0];
        const uint32_t mk = __reduce_max_sync(0xffffffffu, bk);
        const uint32_t eq = __ballot_sync(0xffffffffu, bk == mk);
        bool mine;
        if (__popc(eq) == 1) {
          mine = (eq >> lane) & 1u;
        } else {
          const int mi = (int)__reduce_min_sync(0xffffffffu, bk == mk ? (uint32_t)bi : 0xffffffffu);
          mine = bk == mk && bi == mi;
        }
        if (mine) {
          s_sel_id[t][k] = bi;
          s_sel_w[t][k] = wvs[j][0];
#pragma unroll
          for (int i = 0; i < kTopk - 1; ++i) keys[j][i] = keys[j][i + 1], idss[j][i] = idss[j][i + 1],
                                              wvs[j][i] = wvs[j][i + 1];
          keys[j][kTopk - 1] = 0u, idss[j][kTopk - 1] = 0x7fffffff;
        }
      }
    }
  }
  __syncwarp();
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int t = warp + j * (kThreads / 32);
    if (t < T && lane < kTopk) {
      float w6[6];
#pragma unroll
      for (int k = 0; k < kTopk; ++k) w6[k] = s_sel_w[t][k];
      const float ws = __fadd_rn(__fadd_rn(__fadd_rn(w6[0], w6[4]), w6[2]), __fadd_rn(__fadd_rn(w6[1], w6[5]), w6[3]));
      const float f = div_full_f32(p.routed_scale, ws > 0.f ? ws : 1.0f);
      const int e = s_sel_id[t][lane];
      const float wt = __fmul_rn(f, s_sel_w[t][lane]);
      const int rm = c_rowmap[e];
      const int cold = rm < 0 ? -rm - 1 : -1;
      s_cold[t * kTopk + lane] = cold;
      s_wts[t * kTopk + lane] = wt;
      if (writer) {
        const int hot = rm >= 0 ? rm : -1;
        const bool pad = s_pad[t];
        if (p.topk_ids != nullptr) p.topk_ids[t * kTopk + lane] = e;
        if (p.topk_w != nullptr) p.topk_w[t * kTopk + lane] = wt;
        if (p.hot_ids != nullptr) p.hot_ids[t * kTopk + lane] = hot;
        p.mega_idx[t * kTopk + lane] = pad ? -1 : hot;
        p.mega_w[t * kTopk + lane] = pad ? 0.f : wt;
        if (p.cold_ids != nullptr) p.cold_ids[t * kTopk + lane] = cold;
      }
    }
  }
  __syncthreads();
  if (writer && tid == 0 && p.mega_ctl != nullptr) *p.mega_ctl = p.mega_ctl_val;  // MegaMoE reads it after us
  if (warp == 0) {  // has / pos / rows: lane t is token t; pos = cold tokens before t (minus one when t has none)
    bool h = false;
    if (lane < T)
#pragma unroll
      for (int k = 0; k < kTopk; ++k) h |= s_cold[lane * kTopk + k] >= 0;
    const uint32_t m = __ballot_sync(0xffffffffu, h);
    const int before = __popc(m & ((1u << lane) - 1u));
    const int rows = __popc(m);
    if (lane < T) {
      s_has[lane] = h;
      s_pos[lane] = h ? before : before - 1;
      if (writer) p.has[lane] = h, p.pos[lane] = s_pos[lane];
    }
    if (lane == 0) {
      if (writer) *p.rows = rows;
      flag = rows;
    }
  }
  __syncthreads();
  RSTAMP(6);
  const int rows = (int)flag;
  if (v3) {
    // v3, leader 0 alone: each cold token's 6 route weights; then the other leaders' arrivals (acquire: their row
    // slices, written long before), and every token's record (release, sys).
    uint8_t* seg = p.p3_seg;
    for (int i = tid; i < T * kTopk; i += kThreads)
      if (s_has[i / kTopk])
        reinterpret_cast<float*>(seg + kP3OffRec + (i / kTopk) * kP3RecBytes + 8)[i % kTopk] = s_wts[i];
    if (tid == 0) {
      const unsigned long long t0 = gtimer();
      while (arrived < (uint32_t)(kClusters - 1)) {
        if (gtimer() - t0 > 20000000000ull) __trap();  // as above: a lost leader must not wedge the GPU
        arrived = ld_relaxed(p.sync + 1);
      }
      asm volatile("fence.acq_rel.gpu;" ::: "memory");  // acquire (relaxed load + fence): their row slices
      if (!p.lean_fence) asm volatile("fence.acq_rel.sys;" ::: "memory");
    }
    __syncthreads();
    if (warp == 0) {
      if (lane < T) {  // token `lane` publishes its record
        uint64_t ids48 = 0;
#pragma unroll
        for (int k = 0; k < kTopk; ++k) {
          const int c = s_cold[lane * kTopk + k];
          ids48 |= (uint64_t)(c >= 0 ? (uint32_t)c & 0xFFu : 0xFFu) << (8 * k);
        }
        const uint32_t tag = (s_p3ctr << 12) | ((uint32_t)p.layer << 6) | (uint32_t)(T - 1);
        // arm the RTX's host ready word (COPY_AHEAD: the RTX sets it after this call's done tag; ordered before the
        // records by their release). Unused by sidecars without it.
        if (lane == 0)
          asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p.p3_seg + kP3OffReady), "r"(0u) : "memory");
        if (!p.lean_fence) asm volatile("fence.acq_rel.sys;" ::: "memory");
        asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(p.p3_seg + kP3OffRec + lane * kP3RecBytes),
                     "l"(((uint64_t)tag << 48) | ids48)
                     : "memory");
      }
      if (lane == 0) {
        p.sync[1] = 0u;
        p.sync[2] = epoch + 1u;
      }
    }
  } else if (send) {
    // peer v2: every leader copies its slice of each cold row (by cold position), arrives; the last one publishes.
    if (writer) {
      if (p.peer_pad > 0)
        for (int i = tid; i < p.peer_pad * kTopk; i += kThreads)
          if (i / kTopk >= rows) p.peer_ids[i] = -1;
      for (int i = tid; i < T * kTopk; i += kThreads) {
        const int t = i / kTopk;
        if (s_has[t]) {
          p.peer_ids[s_pos[t] * kTopk + i % kTopk] = s_cold[i];
          p.peer_w[s_pos[t] * kTopk + i % kTopk] = s_wts[i];
        }
      }
    }
    if (nt < kRouterKTiles) {
      // 16 B per item: token t, chunk c of this leader's 256-byte slice of the row (c == 16: its 8 scale bytes).
      for (int i = tid; i < T * 17; i += kThreads) {
        const int t = i / 17, c = i % 17;
        if (!s_has[t]) continue;
        const int r = s_pos[t];
        if (c < 16)
          *reinterpret_cast<uint4*>(p.peer_x + (size_t)r * 5120 + nt * kRouterTileK + c * 16) =
              *reinterpret_cast<const uint4*>(&sq[t][c * 16]);
        else
          *reinterpret_cast<uint2*>(p.peer_xs + (size_t)r * 160 + nt * 8) =
              *reinterpret_cast<const uint2*>(&ssf[t][0]);
      }
    }
    if (!p.lean_fence) asm volatile("fence.acq_rel.sys;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
      bool last;
      if (p.lean_fence) {
        // release: this CTA's payload (ordered before by the barrier); acquire: every earlier arrival's payload
        uint32_t old;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p.sync + 1) : "memory");
        last = old == kClusters - 1;
      } else {
        last = atomicAdd(p.sync + 1, 1u) == kClusters - 1;
        // Every leader fenced its payload at system scope before arriving; the last one acquires all arrivals
        // here, so its releases below order the whole payload.
        if (last) asm volatile("fence.acq_rel.sys;" ::: "memory");
      }
      if (last) {
        const long long seq = *p.peer_seq + 1;
        *p.peer_seq = seq;
        volatile long long* h = p.peer_header;
        h[1] = p.layer;
        h[2] = rows;
        h[3] = T;
        h[0] = seq;
        asm volatile("atom.exch.release.sys.global.b64 _, [%0], %1;" ::"l"(p.peer_words), "l"(seq) : "memory");
        p.sync[1] = 0u;
      }
      if (writer) p.sync[2] = epoch + 1u;  // every leader read the counter before its mailbox store
    }
  } else if (tid == 0) {
    p.sync[2] = epoch + 1u;
  }
  RSTAMP(5);
}

}  // namespace

extern "C" int w4_router_launch(const RouterParams* p, int pdl, cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kRouterSplitK, kExperts / 16);
  cfg.blockDim = dim3(kThreads);
  cfg.stream = stream;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = kRouterSplitK, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = pdl;
  cfg.attrs = at;
  cfg.numAttrs = 2;
  return (int)cudaLaunchKernelEx(&cfg, router_kernel, *p);
}
