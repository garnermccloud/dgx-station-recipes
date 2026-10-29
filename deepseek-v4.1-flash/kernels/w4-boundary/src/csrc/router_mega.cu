// W4 router for 16 < T <= 64: bit-exact replica of production's T > 16 MoE glue, i.e. DeepGEMM's bf16 Mega gate
// (sm100_bf16_mega_gate_impl: 8 K splits of 640 accumulated in K16 steps, splits summed from split 0, precise
// sqrt-softplus, bias / image bias, warp top-6 by (score desc, id asc), weights s / (sum + 1e-20) * 1.5), then the
// hot/cold split, MegaMoE staging (prepare_megamoe recipe) and the peer cold-row publish (v2 route_send2 or v3).
//
//   K1 grid (8 splits, 24 expert tiles of 16), cluster (8, 1, 1) = one tile's splits, 256 threads, 2 CTAs/SM:
//      the split GEMM (mma.sync m16n8k16, warp = 16-token tile x 8-expert atom) into shared memory; rank r then sums
//      tokens [8r, 8r + 8) over the 8 ranks' partials in split order (DSMEM), scores them and keeps the tile's
//      top-6 per token as merge candidates.
//   K2 one cluster of 16 CTAs, PDL after K1 (which releases it once li is complete): MegaMoE staging before the
//      wait, then the merge (warp per token), has[] exchanged through DSMEM, the cold-row slices, rank 0 publishes.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "w4/mgate_core.cuh"
#include "w4/mhc_core.cuh"
#include "w4/params.h"
#include "w4/route_core.cuh"

namespace {

using namespace w4;

constexpr int kThreads = 256;
constexpr int kMaxT = 64;
constexpr int kTiles = kExperts / 16;  // 24
constexpr int kKS = mgate::kKPerSplit;  // 640
constexpr int kRow = kKS + 8;           // bf16 smem row stride (conflict-free fragments)
constexpr int kK1Smem = (16 + kMaxT) * kRow * 2;
constexpr uint64_t kP3OffRec = 4096, kP3RecBytes = 128;
constexpr uint64_t kP3OffXq = kP3OffRec + 64 * kP3RecBytes;
constexpr uint64_t kP3OffXs = kP3OffXq + 64ull * 5120;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ uint32_t score_key(float v) {
  const uint32_t b = __float_as_uint(__fadd_rn(v, 0.0f));
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ bool token_image(const RouterParams& p, int t) { return p.image_mask != nullptr && p.image_mask[t]; }

// MegaMoE staging of K tile `tile` (256 columns) for all T tokens (prepare_megamoe_inputs' x / x_sf / shared_x_sf).
__device__ void stage_tile(const RouterParams& p, int tile, int tid) {
  const int T = p.T;
  for (int base = 0; base < T * 32; base += kThreads) {
    const int item = base + tid;
    const bool active = item < T * 32;
    const int t = active ? item / 32 : 0, c8 = item % 32;
    const int col = tile * kRouterTileK + c8 * 8;
    float f[8];
    unpack8(active ? *reinterpret_cast<const uint4*>(p.li + (size_t)t * 5120 + col) : make_uint4(0, 0, 0, 0), f);
    float am = amax8(f);
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    const uint32_t pe = prep_exp(am);
    if (active) *reinterpret_cast<uint2*>(p.prep_x + (size_t)t * 5120 + col) = prep_quant8(f, prep_inv(pe));
    if (active && (c8 & 3) == 0) {
      const int grp = col / 128, byte = (col % 128) / 32;
      reinterpret_cast<uint8_t*>(p.prep_sf)[((size_t)t * p.prep_sf_sm + (size_t)grp * p.prep_sf_sk) * 4 + byte] = (uint8_t)pe;
      if (p.prep_shared_sf != nullptr) {
        const int row = prep_shared_row(t, p.shared_block_m);
        reinterpret_cast<uint8_t*>(p.prep_shared_sf)[((size_t)row * p.prep_ssf_sm + (size_t)grp * p.prep_ssf_sk) * 4 + byte] =
            (uint8_t)pe;
      }
    }
  }
}

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// debug: dbg[8 + 8 * cta + j] per K1 CTA (cta = 8 * tile + split)
#define K2STAMP(j) \
  if (p.dbg != nullptr && threadIdx.x == 0) p.dbg[8 + 8 * 192 + 8 * cluster_rank() + (j)] = gtimer();
#define KSTAMP(j) \
  if (p.dbg != nullptr && threadIdx.x == 0) p.dbg[8 + 8 * (blockIdx.y * 8 + blockIdx.x) + (j)] = gtimer();
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float ld_dsmem(const float* local, uint32_t rank) {
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(smem_u32(local)), "r"(rank));
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void st_dsmem_u32(uint32_t* local, uint32_t rank, uint32_t v) {
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(smem_u32(local)), "r"(rank));
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

// Scratch after the split partials in mg_part: final weights [T][6] and cold-local ids [T][6] of the merged picks.
__device__ __forceinline__ float* pick_w(const RouterParams& p) { return p.mg_part + (size_t)mgate::kSplitK * kMaxT * kExperts; }
__device__ __forceinline__ int* pick_cold(const RouterParams& p) {
  return reinterpret_cast<int*>(pick_w(p) + kMaxT * kTopk);
}

// K1: cluster of the 8 K splits of one 16-expert tile.
__global__ void __launch_bounds__(kThreads, 2) mgate_k1(const RouterParams p) {
  extern __shared__ __align__(16) __nv_bfloat16 k1s[];
  __nv_bfloat16* sW = k1s;              // [16][kRow]
  __nv_bfloat16* sA = k1s + 16 * kRow;  // [kMaxT][kRow]
  __shared__ float part[kMaxT][16];     // this split's partial logits of the tile
  __shared__ uint32_t last_flag;
  const uint32_t s = cluster_rank();
  const int nt = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int T = p.T, mtn = (T + 15) / 16, k0 = (int)s * kKS;
  KSTAMP(0);
  for (int i = tid; i < 16 * (kKS / 8); i += kThreads) {
    const int n = i / (kKS / 8), c8 = i % (kKS / 8);
    cp_async16(&sW[n * kRow + c8 * 8], p.gate_w + (size_t)(nt * 16 + n) * 5120 + k0 + c8 * 8);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  // only now (li is complete): K2's pre-wait part (MegaMoE staging of li) may start beside this kernel
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  KSTAMP(1);
  for (int i = tid; i < mtn * 16 * (kKS / 8); i += kThreads) {
    const int r = i / (kKS / 8), c8 = i % (kKS / 8);
    if (r < T)
      cp_async16(&sA[r * kRow + c8 * 8], p.li + (size_t)r * 5120 + k0 + c8 * 8);
    else
      *reinterpret_cast<uint4*>(&sA[r * kRow + c8 * 8]) = make_uint4(0, 0, 0, 0);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  KSTAMP(2);
  {  // the split's chain: warp = (16-token tile, 8-expert atom), 40 K16 steps from zero
    const int mt = warp >> 1, na = warp & 1, g = lane >> 2, tig = lane & 3;
    if (mt < mtn) {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      const __nv_bfloat16* a0 = sA + (mt * 16 + g) * kRow;
      const __nv_bfloat16* a1 = a0 + 8 * kRow;
      const __nv_bfloat16* b = sW + (na * 8 + g) * kRow;
#pragma unroll 8
      for (int st = 0; st < kKS / 16; ++st) {
        const int kk = st * 16 + 2 * tig;
        const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(a0 + kk), *reinterpret_cast<const uint32_t*>(a1 + kk),
                               *reinterpret_cast<const uint32_t*>(a0 + kk + 8),
                               *reinterpret_cast<const uint32_t*>(a1 + kk + 8)};
        mma_bf16_16816(c, a, *reinterpret_cast<const uint32_t*>(b + kk), *reinterpret_cast<const uint32_t*>(b + kk + 8));
      }
      const int t0 = mt * 16 + g, e = na * 8 + 2 * tig;
      part[t0][e] = c[0], part[t0][e + 1] = c[1];
      part[t0 + 8][e] = c[2], part[t0 + 8][e + 1] = c[3];
    }
  }
  KSTAMP(3);
  cluster_sync();  // every split's partials are in its shared memory
  KSTAMP(4);
  // Rank r scores tokens [8r, 8r + 8): split sum in split order (DSMEM), precise sqrt-softplus, bias, tile top-6.
  {
    const int t = (int)s * 8 + warp;
    if (t < T) {
      const int c = lane & 15, e = nt * 16 + c;
      float v = ld_dsmem(&part[t][c], 0);
#pragma unroll
      for (int k = 1; k < mgate::kSplitK; ++k) v = __fadd_rn(v, ld_dsmem(&part[t][c], k));
      if (p.logits != nullptr && lane < 16) p.logits[(size_t)t * kExperts + e] = v;
      const float unb = sqrtf(v > 20.0f ? v : log1pf(expf(v)));
      const float* bias = token_image(p, t) ? p.bias_vl : p.bias;
      const float sc = __fadd_rn(unb, bias[e]);
      uint32_t key = lane < 16 ? score_key(sc == sc ? sc : -1e30f) : 0u;
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        const uint32_t mk = __reduce_max_sync(0xffffffffu, key);
        const int win = __ffs(__ballot_sync(0xffffffffu, key == mk)) - 1;
        if (lane == win) {
          const size_t o = ((size_t)t * kTiles + nt) * kTopk + k;
          p.cand_v[o] = sc;
          p.cand_w[o] = unb;
          p.cand_i[o] = e;
          key = 0u;
        }
      }
    }
  }
  KSTAMP(5);
  cluster_sync();  // the other ranks' partial reads are done before any CTA of the cluster exits
}

// K2: one cluster of 16 CTAs. Before the dependency wait: MegaMoE staging of K tiles c and c + 16. Then warp w of
// CTA c merges token 8c + w (24 x 6 candidates; DeepGEMM's weights) and writes its routing outputs; has[] meets in
// every CTA through DSMEM; each CTA quantizes its K tiles of every cold row into the peer buffer; rank 0 publishes.
constexpr int kK2 = 16;
__global__ void __launch_bounds__(kThreads, 1) mgate_k2(const RouterParams p) {
  __shared__ int8_t s_has[kMaxT];
  __shared__ int32_t s_pos[kMaxT];
  __shared__ int s_rows;
  const uint32_t c = cluster_rank();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int T = p.T;
  const bool writer = c == 0;
  K2STAMP(0);
  if (p.prep_x != nullptr)
    for (int tile = (int)c; tile < kRouterKTiles; tile += kK2) stage_tile(p, tile, tid);
  K2STAMP(1);
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");  // peers may now write into our s_has
  K2STAMP(2);
  const int t = (int)c * 8 + warp;
  if (t < T) {
    uint32_t keys[kTopk];
    int idss[kTopk];
    float wvs[kTopk];
    const bool ok = lane < kTiles;
    const size_t o = ((size_t)t * kTiles + (ok ? lane : 0)) * kTopk;
#pragma unroll
    for (int i = 0; i < kTopk; ++i) {
      keys[i] = ok ? score_key(__ldcg(p.cand_v + o + i)) : 0u;
      idss[i] = ok ? __ldcg(p.cand_i + o + i) : 0x7fffffff;
      wvs[i] = ok ? __ldcg(p.cand_w + o + i) : 0.f;
    }
    float s_k = 0.f;
    int e_k = 0;
#pragma unroll
    for (int k = 0; k < kTopk; ++k) {
      const uint32_t mk = __reduce_max_sync(0xffffffffu, keys[0]);
      const uint32_t eq = __ballot_sync(0xffffffffu, keys[0] == mk);
      bool mine;
      if (__popc(eq) == 1) {
        mine = (eq >> lane) & 1u;
      } else {
        const int mi = (int)__reduce_min_sync(0xffffffffu, keys[0] == mk ? (uint32_t)idss[0] : 0xffffffffu);
        mine = keys[0] == mk && idss[0] == mi;
      }
      const int src = __ffs(__ballot_sync(0xffffffffu, mine)) - 1;
      const int pid = __shfl_sync(0xffffffffu, idss[0], src);
      const float pw = __shfl_sync(0xffffffffu, wvs[0], src);
      if (lane == k) s_k = pw, e_k = pid;
      if (mine) {
#pragma unroll
        for (int i = 0; i < kTopk - 1; ++i) keys[i] = keys[i + 1], idss[i] = idss[i + 1], wvs[i] = wvs[i + 1];
        keys[kTopk - 1] = 0u, idss[kTopk - 1] = 0x7fffffff;
      }
    }
    // store_topk_token: warp_reduce_sum<8> (xor 4, 2, 1) of the picks' unbiased scores + 1e-20; s / sum * scale
    float sum = s_k;
    sum = sum + __shfl_xor_sync(0xffffffffu, sum, 4);
    sum = sum + __shfl_xor_sync(0xffffffffu, sum, 2);
    sum = sum + __shfl_xor_sync(0xffffffffu, sum, 1);
    sum = sum + 1e-20f;
    int cold = -1;
    if (lane < kTopk) {
      const float wt = s_k / sum * p.routed_scale;
      const int rm = p.row_map[e_k];
      cold = rm < 0 ? -rm - 1 : -1;
      const int hot = rm >= 0 ? rm : -1;
      const bool pad = p.is_padding != nullptr && p.is_padding[t];
      if (p.topk_ids != nullptr) p.topk_ids[t * kTopk + lane] = e_k;
      if (p.topk_w != nullptr) p.topk_w[t * kTopk + lane] = wt;
      if (p.hot_ids != nullptr) p.hot_ids[t * kTopk + lane] = hot;
      p.mega_idx[t * kTopk + lane] = pad ? -1 : hot;
      p.mega_w[t * kTopk + lane] = pad ? 0.f : wt;
      if (p.cold_ids != nullptr) p.cold_ids[t * kTopk + lane] = cold;
      pick_w(p)[t * kTopk + lane] = wt;
      pick_cold(p)[t * kTopk + lane] = cold;
    }
    const bool h = __ballot_sync(0xffffffffu, lane < kTopk && cold >= 0) != 0;
    if (lane < kK2) {
      uint32_t a;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(smem_u32(&s_has[t])), "r"(lane));
      asm volatile("st.shared::cluster.u8 [%0], %1;" ::"r"(a), "h"((unsigned short)h) : "memory");
    }
    if (writer && lane == 0) p.has[t] = h;  // (CTA 0 owns tokens 0..7 only; the others' has[] written below)
  }
  K2STAMP(3);
  cluster_sync();  // every token's has[] is in every CTA; picks are in global memory
  K2STAMP(4);
  if (warp == 0) {
    const bool h0 = lane < T && s_has[lane] != 0, h1 = lane + 32 < T && s_has[lane + 32] != 0;
    const uint32_t m0 = __ballot_sync(0xffffffffu, h0), m1 = __ballot_sync(0xffffffffu, h1);
    const uint32_t lt = (1u << lane) - 1u;
    const int b0 = __popc(m0 & lt), b1 = __popc(m0) + __popc(m1 & lt);
    if (lane < T) {
      s_pos[lane] = h0 ? b0 : b0 - 1;
      if (writer) p.has[lane] = h0, p.pos[lane] = s_pos[lane];
    }
    if (lane + 32 < T) {
      s_pos[lane + 32] = h1 ? b1 : b1 - 1;
      if (writer) p.has[lane + 32] = h1, p.pos[lane + 32] = s_pos[lane + 32];
    }
    if (lane == 0) {
      s_rows = __popc(m0) + __popc(m1);
      if (writer) *p.rows = s_rows;
    }
  }
  __syncthreads();
  const int rows = s_rows;
  const bool v3 = p.p3_seg != nullptr, send = v3 || p.peer_words != nullptr;
  if (send) {
    const float* pw = pick_w(p);
    const int* pc = pick_cold(p);
    if (writer) {
      if (v3) {
        for (int i = tid; i < T * kTopk; i += kThreads)
          if (s_has[i / kTopk])
            reinterpret_cast<float*>(p.p3_seg + kP3OffRec + (i / kTopk) * kP3RecBytes + 8)[i % kTopk] = __ldcg(pw + i);
      } else {
        if (p.peer_pad > 0)
          for (int i = tid; i < p.peer_pad * kTopk; i += kThreads)
            if (i / kTopk >= rows) p.peer_ids[i] = -1;
        for (int i = tid; i < T * kTopk; i += kThreads) {
          const int tt = i / kTopk;
          if (s_has[tt]) {
            p.peer_ids[s_pos[tt] * kTopk + i % kTopk] = __ldcg(pc + i);
            p.peer_w[s_pos[tt] * kTopk + i % kTopk] = __ldcg(pw + i);
          }
        }
      }
    }
    // this CTA's K tiles of every cold row, FlashInfer MXFP8 (route_send2's recipe), 8 values per item
    for (int tile = (int)c; tile < kRouterKTiles; tile += kK2)
      for (int base = 0; base < T * 32; base += kThreads) {
        const int item = base + tid;
        const bool active = item < T * 32;
        const int tt = active ? item / 32 : 0, c8 = item % 32;
        const bool cold = active && s_has[tt];  // uniform per 4-lane group
        const int col = tile * kRouterTileK + c8 * 8;
        float f[8];
        unpack8(cold ? *reinterpret_cast<const uint4*>(p.li + (size_t)tt * 5120 + col) : make_uint4(0, 0, 0, 0), f);
        float am = amax8(f);
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
        if (!cold) continue;
        const uint32_t e8 = fi_ue8m0(__fmul_rn(am, kInvE4m3Max));
        const uint2 q = fi_quant8(f, fi_inv_scale(e8));
        uint8_t* xq = v3 ? p.p3_seg + kP3OffXq + (size_t)tt * 5120 : p.peer_x + (size_t)s_pos[tt] * 5120;
        uint8_t* xs = v3 ? p.p3_seg + kP3OffXs + (size_t)tt * 160 : p.peer_xs + (size_t)s_pos[tt] * 160;
        *reinterpret_cast<uint2*>(xq + col) = q;
        if ((c8 & 3) == 0) xs[col / 32] = (uint8_t)e8;
      }
    K2STAMP(5);
    if (!p.lean_fence) asm volatile("fence.acq_rel.sys;" ::: "memory");
  }
  K2STAMP(6);
  cluster_sync();  // every CTA's payload is fenced at system scope
  K2STAMP(7);
  if (!writer || !send) return;
  if (!p.lean_fence) asm volatile("fence.acq_rel.sys;" ::: "memory");
  if (v3) {
    const int* pc = pick_cold(p);
    for (int tt = tid; tt < T; tt += kThreads) {
      uint64_t ids48 = 0;
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        const int cc = __ldcg(pc + tt * kTopk + k);
        ids48 |= (uint64_t)(cc >= 0 ? (uint32_t)cc & 0xFFu : 0xFFu) << (8 * k);
      }
      const uint32_t tag = (p.p3_state[0] << 12) | ((uint32_t)p.layer << 6) | (uint32_t)(T - 1);
      asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(p.p3_seg + kP3OffRec + tt * kP3RecBytes),
                   "l"(((uint64_t)tag << 48) | ids48)
                   : "memory");
    }
  } else if (tid == 0) {
    const long long seq = *p.peer_seq + 1;
    *p.peer_seq = seq;
    volatile long long* h = p.peer_header;
    h[1] = p.layer;
    h[2] = rows;
    h[3] = T;
    h[0] = seq;
    asm volatile("atom.exch.release.sys.global.b64 _, [%0], %1;" ::"l"(p.peer_words), "l"(seq) : "memory");
  }
}

}  // namespace

extern "C" int w4_router_mega_launch(const RouterParams* p, int pdl, cudaStream_t stream) {
  static bool attr = false;
  if (!attr) {
    const cudaError_t e = cudaFuncSetAttribute(mgate_k1, cudaFuncAttributeMaxDynamicSharedMemorySize, kK1Smem);
    if (e != cudaSuccess) return (int)e;
    attr = true;
  }
  if (p->T <= 16 || p->T > kMaxT) return (int)cudaErrorInvalidValue;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl;
  at[1].id = cudaLaunchAttributeClusterDimension;
  at[1].val.clusterDim.x = mgate::kSplitK, at[1].val.clusterDim.y = 1, at[1].val.clusterDim.z = 1;
  cudaLaunchConfig_t c1 = {};
  c1.gridDim = dim3(mgate::kSplitK, kTiles);
  c1.blockDim = dim3(kThreads);
  c1.dynamicSmemBytes = kK1Smem;
  c1.stream = stream;
  c1.attrs = at;
  c1.numAttrs = 2;
  cudaError_t e = cudaLaunchKernelEx(&c1, mgate_k1, *p);
  if (e != cudaSuccess) return (int)e;
  static bool attr2 = false;
  if (!attr2) {
    const cudaError_t e2 = cudaFuncSetAttribute(mgate_k2, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (e2 != cudaSuccess) return (int)e2;
    attr2 = true;
  }
  cudaLaunchAttribute at2[2];
  at2[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at2[0].val.programmaticStreamSerializationAllowed = 1;
  at2[1].id = cudaLaunchAttributeClusterDimension;
  at2[1].val.clusterDim.x = kK2, at2[1].val.clusterDim.y = 1, at2[1].val.clusterDim.z = 1;
  cudaLaunchConfig_t c2 = {};
  c2.gridDim = dim3(kK2);
  c2.blockDim = dim3(kThreads);
  c2.stream = stream;
  c2.attrs = at2;
  c2.numAttrs = 2;
  return (int)cudaLaunchKernelEx(&c2, mgate_k2, *p);
}
