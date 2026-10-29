// W8 standalone wo_a + MXFP8 quantize (one PDL launch), bitwise vs production:
//   z = DeepGEMM fp8_einsum("bhr,hdr->bhd", attn_out, wo_a, recipe (1, 1, 32))   [one accumulation chain per 128-row
//       weight tile over the whole K = 4096: measured bitwise at S = 1..16, incl. cancellation-heavy inputs]
//   (zq, sf) = vLLM's mxfp8_e4m3_quantize(z.flatten(1), is_sf_swizzled_layout=True)   [FlashInfer recipe, W4 fi_*]
// so the stock wo_b linear consumes (zq, sf) as a QuantizedActivation and the separate quantize launch is gone.
//
// SPLIT = 1 (bitwise): 64 CTAs, CTA = one 128-row tile of one group (rows g * 1024 + t * 128): McChains<1> (32 K
// blocks). SPLIT = 1, C = 2 (oproj-sk2's in-CTA split-K: the same 64 CTAs, two 16-K-block chains interleaved by ONE
// issuer warp (McChains::issue_pair; two issuer warps hung intermittently in this configuration), z = p0 + p1: the
// fixed fp32 order of SPLIT = 2, C = 1, without a second CTA per tile). SPLIT = 2 (the deterministic split-K of the exactness policy: fixed fp32 order, not bitwise): 64 clusters of
// 2 CTAs, rank r runs K blocks [16 r, 16 r + 16) of the tile as C chains (C = 2: two interleaved 8-K-block chains,
// parts q = 2 r, 2 r + 1, summed in the CTA: s_r = p_2r + p_2r+1); rank 1 pushes its fp32 s_1 into rank 0's SMEM
// (st.async + mbarrier), rank 0 forms z = bf16(s_0 + s_1) and quantizes. The first 32 CTAs to start (a ticket) also
// L2-prefetch 1/32 of the whole weight each, so CTAs that only start after the attention kernel find wo_a in L2.
// Prologue before
// griddepcontrol.wait: the tile's weight scales and as many weight slots as the ring holds (the rest L2-prefetched),
// so wo_a streams while the attention kernel (the predecessor) still runs. launch_dependents fires once the prologue is
// issued, so wo_b's own prologue can start on the free SMs. Epilogue: warp w of warps 4..7 holds rows 32 w .. + 32 of
// the tile = one 32-feature MXFP8 block per token: bf16 round, warp amax, ue8m0, e4m3.
#pragma once
#include <cuda_bf16.h>
#include "w8/mcgemm.cuh"
#include "w4/mhc_core.cuh"

namespace w8::woa {

constexpr int kGroups = 8, kOLora = 1024, kK = 4096, kKb = kK / 128, kTiles = kGroups * kOLora / 128;  // 64
constexpr int kZ = kGroups * kOLora;  // 8192
constexpr int kSmemMax = 232448;

struct Params {
  CUtensorMap tm_w;    // wo_a weight [8 * 1024][4096] as 3D {128 B, rows, K blocks}, box {128, 128, 2}, SW128
  CUtensorMap tm_x;    // attention output fp8 [S][8][4096] as 4D {128 B, token, K block, group}, box {128, NPAD, 32, 1}
  const void* w_base;  // the weight itself (SPLIT 2: L2 prefetch)
  const uint8_t* wsf;  // wo_a scale atoms, tile-major: (tile, kb) at (tile * 32 + kb) * 512 (re-laid out at boot)
  const uint32_t* xs;  // attention output scales: word (t, g, kb) at g * xs_sg + kb * xs_skb + t
  int xs_sg, xs_skb;
  uint8_t* zq;           // out fp8 [S][8192]
  uint8_t* zsf;          // out FlashInfer 128x4-swizzled ue8m0 [ceil(S / 128) * 128][256] (pad rows stay 0)
  __nv_bfloat16* z;      // optional out bf16 [S][8][1024] (tests)
  float* z32;            // optional out: the fp32 accumulator before the bf16 round [S][8][1024] (error reports)
  Diag* diag;
  unsigned long long* tl;  // optional [CTAs][4]: entry, weights issued, act ready, done
  uint32_t prologue;       // bit 0: no L2-tail prefetch (ring slots only); bit 1: no weight stream before
                           // griddepcontrol.wait (the producer waits too); bit 2: 2 ring slots before the wait, the
                           // rest after (attention-contention variants)
  uint32_t* ticket;        // SPLIT 2: u32 start counter (k % grid = start order in this launch; never reset)
};

template <int S, int SPLIT = 1, int C = 1>
struct Layout {
  static constexpr int NPAD = S <= 8 ? 8 : 16;
  static constexpr int AS = NPAD * 16;
  static constexpr int KB = kKb / SPLIT;  // K blocks of one CTA
  static constexpr int KC = KB / C;       // K blocks of one chain
  static_assert(KC % 2 == 0 && C <= 2, "chains");
  using Gemm = McChains<C, NPAD, AS>;
  static constexpr int ACT = 0;
  static constexpr int ACT_SF = Gemm::act_bytes(KB);
  static constexpr int WSF = (ACT_SF + Gemm::act_sf_bytes(KB) + 127) / 128 * 128;
  static constexpr int RECV = WSF + Gemm::wsf_bytes(KB);  // SPLIT 2, rank 0: rank 1's partial [128][NPAD] fp32
  static constexpr int STAGE = RECV + (SPLIT > 1 ? 128 * NPAD * 4 : 0);
  static constexpr int BARS_OFF = kSmemMax - 512;
  static constexpr int NSLOT_FIT = (BARS_OFF - STAGE) / Gemm::SLOT_BYTES;
  static constexpr int NSLOT = NSLOT_FIT < 6 ? NSLOT_FIT : 6;
  static_assert(NSLOT >= 3, "ring");
  using RingT = Ring<NSLOT, Gemm::SLOT_BYTES>;
  static constexpr int STAGE_OFF = NSLOT * Gemm::SLOT_BYTES;
  static_assert(STAGE_OFF + STAGE <= BARS_OFF, "smem");
};

// FlashInfer F8_128x4 swizzle (prod_ref.sf_swizzled_index): row = token, col = 32-feature block of 256
__device__ __forceinline__ int sf_swz(int row, int col) {
  return (col & 3) + (col >> 2) * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4 + (row >> 7) * 128 * 256;
}

__device__ __forceinline__ uint32_t woa_cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void woa_cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

template <int S, int SPLIT = 1, int C = 1>
__global__ void __launch_bounds__(kThreads, 1) woa_kernel(const __grid_constant__ Params p) {
  using L = Layout<S, SPLIT, C>;
  using Gemm = typename L::Gemm;
  constexpr int NPAD = L::NPAD, AS = L::AS, KB = L::KB;
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tile = blockIdx.x / SPLIT, g = tile / 8;
  const uint32_t rank = SPLIT > 1 ? woa_cluster_rank() : 0u;
  const int kb0 = static_cast<int>(rank) * KB;
  if (p.tl != nullptr && threadIdx.x == 0) p.tl[blockIdx.x * 4] = globaltimer();
  typename L::RingT ring;
  ring.base = smem;
  ring.empty_count = 1;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + L::BARS_OFF);
  ring.full = bars;
  ring.empty = bars + 8;
  uint64_t* act_bar = bars + 16;
  uint64_t* recv_bar = act_bar + 1;
  auto* gb = reinterpret_cast<McBars<2>*>(recv_bar + 1);
  auto* wsb = reinterpret_cast<McWsf*>(gb + 1);
  uint32_t* tmem_ptr = reinterpret_cast<uint32_t*>(wsb + 1);
  uint8_t* stage = smem + L::STAGE_OFF;
  uint8_t* act = stage + L::ACT;
  uint8_t* act_sf = stage + L::ACT_SF;
  if (warp_id() == 0) umma::tmem_alloc(tmem_ptr, kMcTmemCols);
  if (threadIdx.x == 32) {
    ring.init_bars(1);
    mbar_init(act_bar, 1);
    mbar_init(recv_bar, 1);
    gb->init();
    wsb->init();
    if (SPLIT > 1 && rank == 0) mbar_arrive_expect_tx(recv_bar, 128 * NPAD * 4);
    fence_mbarrier_init();
  }
  __syncthreads();
  if (SPLIT > 1) woa_cluster_sync();  // the partner's barriers exist before any push
  umma::fence_after_sync();
  const uint32_t tmem = *tmem_ptr;
  McChain ch[C];
#pragma unroll
  for (int c = 0; c < C; ++c) ch[c] = McChain{tile * 128, kb0 + c * L::KC, L::KC, c * L::KC};
  if (warp_id() == kProducerWarp) {
    if (lane_id() == 0) {
      // ---- prologue: weights only (immutable), before griddepcontrol.wait
      prefetch_tmap(&p.tm_w);
      prefetch_tmap(&p.tm_x);
      if (p.prologue & 2u) pdl_wait();  // variant: stream nothing while the predecessor (attention) runs
      uint32_t it = 0, n_wsf = 0;
      Gemm::produce(ring, it, *wsb, n_wsf, stage + L::WSF, &p.tm_w, ch, policy_evict_first(), p.diag,
                    [&](uint8_t* dst, int c, int bytes, uint64_t* bar) {
                      bulk_g2s(dst, p.wsf + (static_cast<size_t>(tile) * kKb + ch[c].kb0) * kMcSfBytes, bytes, bar);
                    }, (p.prologue & 1u) ? (1 << 30) : L::NSLOT - 1,
                    [&](int k) {  // bit 2: only 2 ring slots (64 KB) before griddepcontrol.wait, the rest after
                      if ((p.prologue & 4u) && k == 2) pdl_wait();
                    });
      if (p.tl != nullptr) p.tl[blockIdx.x * 4 + 1] = globaltimer();
    }
    __syncwarp();
  } else {
    if (SPLIT > 1 && threadIdx.x == 0 && p.ticket != nullptr) {
      // the first 32 CTAs to start L2-prefetch the whole weight (1/32 each): the CTAs that only start once the
      // attention kernel leaves its SMs find their tiles in L2
      const uint32_t k = atomicAdd(p.ticket, 1u) % gridDim.x;  // start order within this launch
      if (k < 32) {
        const uint8_t* base = reinterpret_cast<const uint8_t*>(p.w_base);
        const size_t part = (static_cast<size_t>(kTiles) * 128 * kK) / 32;
        for (size_t o = 0; o < part; o += 16384) bulk_prefetch_l2(base + k * part + o, 16384);
      }
    }
    pdl_launch_dependents();  // wo_b's prologue may start on the free SMs
    pdl_wait();               // the attention output is complete
    const int tid = threadIdx.x;
    // ---- activations: this CTA's K range of the group's [S][4096] fp8 via one 4D TMA (issued first, so it overlaps
    // the threads' scale gather)
    if (tid == 0) {
      mbar_arrive_expect_tx(act_bar, KB * NPAD * 128);
      tma_load_4d(act, &p.tm_x, 0, 0, kb0, g, act_bar);
    }
    for (int i = tid; i < KB * S; i += kComputeThreads) {
      const int kb = i / S, t = i % S;
      *reinterpret_cast<uint32_t*>(act_sf + kb * AS + t * 16) = ld_cg_u32(p.xs + g * p.xs_sg + (kb0 + kb) * p.xs_skb + t);
    }
    fence_proxy_async_smem();
    compute_sync();
    McCounters cn;
    float own[NPAD];
    uint32_t rrecv = 0, rbar = 0;
    if (SPLIT > 1) {
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rrecv) : "r"(smem_u32(stage + L::RECV)), "r"(0));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(smem_u32(recv_bar)), "r"(0));
    }
    Gemm::template run<SPLIT == 1 && C == 2>(ring, gb, cn, *wsb, stage + L::WSF, act_bar, 0, act, act_sf, ch, tmem, p.diag,
              [&](int c, int row, const float (&acc)[NPAD]) {
                if (c == 0 && row == 0 && p.tl != nullptr) p.tl[blockIdx.x * 4 + 2] = globaltimer();
                if (c == 0) {
#pragma unroll
                  for (int i = 0; i < NPAD; ++i) own[i] = acc[i];
                } else {
#pragma unroll
                  for (int i = 0; i < NPAD; ++i) own[i] = own[i] + acc[i];  // s_r = p_2r + p_2r+1
                }
                if (c != C - 1) return;
                if (SPLIT > 1 && rank == 1) {
#pragma unroll
                  for (int i = 0; i < NPAD; i += 4)
                    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
                                 ::"r"(rrecv + static_cast<uint32_t>((row * NPAD + i) * 4)), "f"(own[i]), "f"(own[i + 1]),
                                 "f"(own[i + 2]), "f"(own[i + 3]), "r"(rbar) : "memory");
                }
              });
    if (rank == 0 && tid >= 128) {
      // ---- z = bf16(p0 [+ p1]), the MXFP8 quantize (one warp = one 32-feature block per token)
      const int row = tid - 128, f = tile * 128 + row;
      if (SPLIT > 1) {
        mbar_wait(recv_bar, 0, p.diag, 0x7A0);
        const float* recv = reinterpret_cast<const float*>(stage + L::RECV);
#pragma unroll
        for (int i = 0; i < NPAD; ++i) own[i] = own[i] + recv[row * NPAD + i];
      }
#pragma unroll
      for (int i = 0; i < S; ++i) {
        if (p.z32 != nullptr) p.z32[static_cast<size_t>(i) * kZ + f] = own[i];
        const float v = bf16_round(own[i]);
        if (p.z != nullptr) p.z[static_cast<size_t>(i) * kZ + f] = __float2bfloat16_rn(v);
        float am = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        const uint32_t e8 = w4::fi_ue8m0(__fmul_rn(am, w4::kInvE4m3Max));
        p.zq[static_cast<size_t>(i) * kZ + f] = static_cast<uint8_t>(w4::fp8x2_e4m3(__fmul_rn(v, w4::fi_inv_scale(e8)), 0.f) & 0xffu);
        if ((row & 31) == 0) p.zsf[sf_swz(i, f >> 5)] = static_cast<uint8_t>(e8);
      }
    }
    if (p.tl != nullptr && tid == 0) p.tl[blockIdx.x * 4 + 3] = globaltimer();
    compute_sync();
    if (warp_id() == 0) umma::tmem_dealloc(tmem, kMcTmemCols);
  }
  __syncthreads();
  if (SPLIT > 1) {
    woa_cluster_sync();  // rank 0's SMEM outlives rank 1's pushes
  }
}

}  // namespace w8::woa
