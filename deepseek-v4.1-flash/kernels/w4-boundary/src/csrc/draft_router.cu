// W4 DSpark drafter router, 2 <= T <= 64: one launch, bit-exact replica of production's drafter MoE glue
// (DeepseekV4MoE.forward of the 128-expert top-3 gate, which crosses to the Mega gate above T = 1):
//   bf16_mega_gate = DeepGEMM sm100_bf16_mega_gate_impl<5120, 128, 16, 12, 256, 1, 8 splits, 1 group, .., 3,
//     SqrtSoftplus, full tile, bias, image mask>: 8 K splits of 640 accumulated in K16 steps from zero, the splits
//     summed from split 0, then the copied epilogue (mgate_core.cuh, one expert wave, top-3);
//   prepare_megamoe_inputs: x / x_sf / shared_x_sf staging of li, top-k rows masked by is_padding.
//
// grid (16, ceil(T / 16)), cluster (16, 1, 1) = one 16-token block, 512 threads, 1 CTA per SM; launched without PDL
// (resident early beside the seam, the cluster slows the side-stream mix).
//   CTA rank = 8 h + g owns experts [16 g, 16 g + 16) over K half h (splits 4h .. 4h + 3). It stages its 16 gate
//   half-rows (80 KB) and the block's li half-rows with TMA bulk copies (one per row). Warp w < 8 runs split
//   4h + w / 2 for expert atom w % 2 (40 K16 steps from zero, mma.sync m16n8k16, operands in shared memory); warps
//   8-15 meanwhile load the bias vectors and stage the MegaMoE K tiles of the CTA's half from the shared li copy.
//   Split sums in split order: half-0 CTAs sum splits 0..3; after a cluster barrier, each half-1 CTA reads its
//   partner's sum (DSMEM) and adds splits 4..7. After a second barrier, warp w < 8 of CTA r < 2 routes token 8r + w:
//   lane l reads experts 4l .. 4l + 3 (DeepGEMM's lane layout) from CTA 8 + l / 4, runs the epilogue, and lanes < 3
//   write the staged top-k rows.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "w4/mgate_core.cuh"
#include "w4/mhc_core.cuh"
#include "w4/params.h"
#include "w4/route_core.cuh"

namespace {

using w4::DraftRouterParams;

constexpr int kE = 128, kTopkD = 3, kH = 5120, kSplits = 8, kKS = kH / kSplits;  // 640
constexpr int kCtas = 16, kGroups = 8, kEPerGroup = kE / kGroups;                // 16 experts per CTA
constexpr int kHalfK = kH / 2, kSplitsPerHalf = kSplits / 2;                    // 2560, 4
constexpr int kThreads = 512, kMmaThreads = 256, kBlockT = 16, kMaxT = 64;  // warps 0-7: GEMM; 8-15: staging
constexpr int kRow = kHalfK + 8;  // bf16 row stride in shared memory (1284 words: conflict-free fragment reads)
constexpr int kTileK = 256, kTilesPerHalf = kHalfK / kTileK;  // 10 MegaMoE staging tiles per half
constexpr int kSmem = (kEPerGroup + kBlockT) * kRow * 2;      // 164,352 B
static_assert(kSplits == w4::mgate::kSplitK && kKS == w4::mgate::kKPerSplit, "the Mega gate's K split");

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t dsmem(const void* local, uint32_t rank) {
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(smem_u32(local)), "r"(rank));
  return a;
}
__device__ __forceinline__ float ld_dsmem_f32(const float* local, uint32_t rank) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(dsmem(local, rank)) : "memory");
  return v;
}
__device__ __forceinline__ float4 ld_dsmem_f4(const float* local, uint32_t rank) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "r"(dsmem(local, rank))
               : "memory");
  return v;
}
__device__ __forceinline__ uint32_t lds32(const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ void mma_bar_sync() { asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads) : "memory"); }
__device__ __forceinline__ void mbar_init(uint64_t* bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(bar)));
}
__device__ __forceinline__ void mbar_expect(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n W4D_WAIT%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra W4D_WAIT%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
// one TMA bulk copy global -> this CTA's shared memory, completing on `bar`
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// debug stamps: dbg[8 * (cluster block * 16 + rank) + j]
#define DSTAMP(j) \
  if (p.dbg != nullptr && threadIdx.x == 0) p.dbg[8 * (blockIdx.y * kCtas + rank) + (j)] = gtimer();

// MegaMoE staging of K tile `tile` of this half (256 columns at kHalfK * h + 256 tile) for the block's nt tokens,
// from the shared li copy (prepare_megamoe_inputs' x / x_sf / shared_x_sf; the router_mega.cu recipe).
__device__ void stage_tile(const DraftRouterParams& p, const __nv_bfloat16* sA, int h, int tile, int t0, int nt,
                           int tid) {  // tid: 0 .. kThreads - kMmaThreads - 1
  for (int base = 0; base < nt * 32; base += kThreads - kMmaThreads) {
    const int item = base + tid;
    const bool active = item < nt * 32;
    const int tl = active ? item / 32 : 0, c8 = item % 32;
    const int lc = tile * kTileK + c8 * 8, col = h * kHalfK + lc, t = t0 + tl;
    float f[8];
    w4::unpack8(active ? *reinterpret_cast<const uint4*>(sA + tl * kRow + lc) : make_uint4(0, 0, 0, 0), f);
    float am = w4::amax8(f);
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    const uint32_t pe = w4::prep_exp(am);
    if (active) *reinterpret_cast<uint2*>(p.prep_x + (size_t)t * kH + col) = w4::prep_quant8(f, w4::prep_inv(pe));
    if (active && (c8 & 3) == 0) {
      const int grp = col / 128, byte = (col % 128) / 32;
      reinterpret_cast<uint8_t*>(p.prep_sf)[((size_t)t * p.prep_sf_sm + (size_t)grp * p.prep_sf_sk) * 4 + byte] = (uint8_t)pe;
      if (p.prep_shared_sf != nullptr) {
        const int row = w4::prep_shared_row(t, p.shared_block_m);
        reinterpret_cast<uint8_t*>(p.prep_shared_sf)[((size_t)row * p.prep_ssf_sm + (size_t)grp * p.prep_ssf_sk) * 4 + byte] =
            (uint8_t)pe;
      }
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1) draft_router(const DraftRouterParams p) {
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  __nv_bfloat16* sW = smem;                   // [16 experts][kRow]: this CTA's gate half-rows
  __nv_bfloat16* sA = smem + kEPerGroup * kRow;  // [16 tokens][kRow]: the block's li half-rows
  __shared__ float part[kSplitsPerHalf][kBlockT][kEPerGroup];   // split partial logits
  __shared__ __align__(16) float ssum[kBlockT][kEPerGroup];     // half 0: splits 0..3; half 1: all 8 (final)
  __shared__ __align__(16) float sbias[kE], sbvl[kE];
  __shared__ __align__(8) uint64_t bars[2];                     // [0] gate rows, [1] li rows (one phase each)
  const uint32_t rank = cluster_rank();
  const int g = (int)rank % kGroups, h = (int)rank / kGroups;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int t0 = blockIdx.y * kBlockT, nt = min(kBlockT, p.T - t0);
  DSTAMP(0);
  if (tid == 0) {
    mbar_init(&bars[0]);
    mbar_init(&bars[1]);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {  // the 16 gate half-rows: one bulk copy each (weights, before the dependency wait)
    if (lane == 0) mbar_expect(&bars[0], kEPerGroup * kHalfK * 2);
    __syncwarp();
    if (lane < kEPerGroup)
      bulk_g2s(&sW[lane * kRow], p.gate_w + (size_t)(g * kEPerGroup + lane) * kH + h * kHalfK, kHalfK * 2, &bars[0]);
  }
  // MegaMoE (our dependent) may launch now: its prologue overlaps us, its own dependency wait still waits for this
  // grid's completion (and memory flush) before it reads the staging we write
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");  // li (the previous kernel's output) is complete
  DSTAMP(1);
  if (warp == 0) {  // the block's li half-rows: one bulk copy per token
    if (lane == 0) mbar_expect(&bars[1], nt * kHalfK * 2);
    __syncwarp();
    if (lane < nt) bulk_g2s(&sA[lane * kRow], p.li + (size_t)(t0 + lane) * kH + h * kHalfK, kHalfK * 2, &bars[1]);
  }
  if (tid < kMmaThreads) {
    for (int i = tid; i < (kBlockT - nt) * (kHalfK / 8); i += kMmaThreads) {  // rows past the block's tokens: zero
      const int r = nt + i / (kHalfK / 8), c8 = i % (kHalfK / 8);
      *reinterpret_cast<uint4*>(&sA[r * kRow + c8 * 8]) = make_uint4(0, 0, 0, 0);
    }
    mbar_wait(&bars[0], 0);
    mbar_wait(&bars[1], 0);
    mma_bar_sync();
    DSTAMP(2);
    {  // split sl = warp / 2 of this half, expert atom na = warp % 2: 40 K16 steps from zero
      const int sl = warp >> 1, na = warp & 1, gq = lane >> 2, tig = lane & 3;
      const __nv_bfloat16* a0 = sA + gq * kRow + sl * kKS + 2 * tig;
      const __nv_bfloat16* a1 = a0 + 8 * kRow;
      const __nv_bfloat16* b = sW + (na * 8 + gq) * kRow + sl * kKS + 2 * tig;
      float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
      for (int st = 0; st < kKS / 16; ++st) {
        const int kk = st * 16;
        const uint32_t a[4] = {lds32(a0 + kk), lds32(a1 + kk), lds32(a0 + kk + 8), lds32(a1 + kk + 8)};
        w4::mma_bf16_16816(c, a, lds32(b + kk), lds32(b + kk + 8));
      }
      const int e = na * 8 + 2 * tig;
      part[sl][gq][e] = c[0], part[sl][gq][e + 1] = c[1];
      part[sl][gq + 8][e] = c[2], part[sl][gq + 8][e + 1] = c[3];
    }
    mma_bar_sync();
    DSTAMP(3);
    if (h == 0) {  // the Mega gate's split sum, first half: split 0, then += splits 1..3
      const int t = tid / kEPerGroup, e = tid % kEPerGroup;  // 256 threads = 16 tokens x 16 experts
      float v = part[0][t][e];
#pragma unroll
      for (int s = 1; s < kSplitsPerHalf; ++s) v = __fadd_rn(v, part[s][t][e]);
      ssum[t][e] = v;
    }
  } else {  // staging warps: the bias vectors, then MegaMoE K tiles of this half from the shared li copy
    const int st = tid - kMmaThreads;
    if (st < kE) {
      sbias[st] = p.bias[st];
      sbvl[st] = p.bias_vl != nullptr ? p.bias_vl[st] : 0.0f;
    }
    mbar_wait(&bars[1], 0);
    for (int tile = g; tile < kTilesPerHalf; tile += kGroups) stage_tile(p, sA, h, tile, t0, nt, st);
  }
  cluster_sync();  // the half-0 sums are visible cluster-wide
  DSTAMP(4);
  if (h == 1 && tid < kMmaThreads) {  // second half: the partner's splits 0..3, then += splits 4..7
    const int t = tid / kEPerGroup, e = tid % kEPerGroup;
    float v = ld_dsmem_f32(&ssum[t][e], (uint32_t)g);
#pragma unroll
    for (int s = 0; s < kSplitsPerHalf; ++s) v = __fadd_rn(v, part[s][t][e]);
    ssum[t][e] = v;
    if (p.logits != nullptr && t < nt) p.logits[(size_t)(t0 + t) * kE + g * kEPerGroup + e] = v;
  }
  cluster_sync();  // the final logits (half-1 CTAs) are visible cluster-wide
  DSTAMP(5);
  if (rank < kBlockT / 8 && warp < 8) {
    const int tl = (int)rank * 8 + warp;
    if (tl < nt) {
      const int t = t0 + tl;
      const float4 v = ld_dsmem_f4(&ssum[tl][(lane & 3) * 4], (uint32_t)(kGroups + (lane >> 2)));
      const float vals[4] = {v.x, v.y, v.z, v.w};
      const bool image = p.image_mask != nullptr && p.image_mask[t];
      int id;
      float wt;
      w4::mgate::route_token<1, kTopkD>([&](int e) { return vals[e & 3]; }, image ? sbvl : sbias, p.routed_scale,
                                         lane, id, wt);
      if (lane < kTopkD) {
        const bool pad = p.is_padding != nullptr && p.is_padding[t];
        p.topk_idx[t * kTopkD + lane] = pad ? -1ll : (long long)id;
        p.topk_w[t * kTopkD + lane] = pad ? 0.0f : wt;
      }
    }
  }
  DSTAMP(6);
  cluster_sync();  // the routing CTAs' DSMEM reads are done before any CTA exits
}

}  // namespace

extern "C" int w4_draft_router_launch(const DraftRouterParams* p, int pdl, cudaStream_t stream) {
  static bool attr = false;
  if (!attr) {
    cudaError_t e = cudaFuncSetAttribute(draft_router, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    if (e == cudaSuccess) e = cudaFuncSetAttribute(draft_router, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (e != cudaSuccess) return (int)e;
    attr = true;
  }
  if (p->T < 1 || p->T > kMaxT || (p->image_mask != nullptr && p->bias_vl == nullptr)) return (int)cudaErrorInvalidValue;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl;
  at[1].id = cudaLaunchAttributeClusterDimension;
  at[1].val.clusterDim.x = kCtas, at[1].val.clusterDim.y = 1, at[1].val.clusterDim.z = 1;
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3(kCtas, (p->T + kBlockT - 1) / kBlockT);
  c.blockDim = dim3(kThreads);
  c.dynamicSmemBytes = kSmem;
  c.stream = stream;
  c.attrs = at;
  c.numAttrs = 2;
  return (int)cudaLaunchKernelEx(&c, draft_router, *p);
}
