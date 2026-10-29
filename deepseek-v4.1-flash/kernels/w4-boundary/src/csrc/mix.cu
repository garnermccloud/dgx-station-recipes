// W4 mix: the next sublayer's mHC coefficients from residual_new, bit-exact against DeepGEMM's
// sm100_tf32_hc_prenorm_gemm (16 K splits) + TileLang's "stats" split.
//
// One CTA per (K split, 16-token tile). Each split's GEMM is a chain of 160 dependent tf32 MMAs (K = 8) per
// output, which fixes the latency floor; the chain is issued as mma.sync m16n8k8 (bit-identical to DeepGEMM's
// tcgen05 kind::tf32 sequence on hc_fn pre-rounded RNE, and ~4x lower dependent latency than M=64 tcgen05.mma
// at this shape: 11 ns vs 40-58 ns per step on GB300). One warp runs the 3 n-atom chains interleaved, reading
// fragment-ordered shared memory (one ld.shared.v4 of A and three ld.shared.v2 of B per step):
//   B image (host, mix_fn_image): [split][slot 4][n-atom 3][step 40][lane 32] float2 (b0, b1) of tf32-RNE hc_fn
//   A image (in kernel):          [step 160][lane 32] float4 (a0..a3) of the 16 rows of the token tile.
// The last CTA of a token tile runs the TileLang stats (sigmoid + Sinkhorn).
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "w4/mhc_core.cuh"
#include "w4/params.h"

namespace {

using namespace w4;

constexpr int kThreads = 256;
constexpr int kSplits = 16, kKps = 4 * kHidden / kSplits;  // 1280
constexpr int kSteps = kKps / 8;                             // 160
constexpr int kBBytes = 3 * kSteps * 32 * 8;                 // 122880
constexpr int kABytes = kSteps * 32 * 16;                    // 81920
constexpr int kSmem = kBBytes + kABytes + 64;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n W4M_WAIT%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra W4M_WAIT%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mma_tf32(float (&c)[4], const float4& a, const float2& b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(__float_as_uint(a.x)), "r"(__float_as_uint(a.y)), "r"(__float_as_uint(a.z)), "r"(__float_as_uint(a.w)),
        "r"(__float_as_uint(b.x)), "r"(__float_as_uint(b.y)));
}

#define MSTAMP(i) \
  if (p.dbg != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) p.dbg[i] = gtimer();

__global__ void __launch_bounds__(kThreads, 1) mix_kernel(const MixParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const float2* sB = reinterpret_cast<const float2*>(smem);  // [3][160][32]
  float* sA = reinterpret_cast<float*>(smem + kBBytes);       // [160][32][4]
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem + kBBytes + kABytes);
  __shared__ uint32_t last_flag;
  const int split = blockIdx.x, mt = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // pdl_mode 2: release MegaMoE now. It fires once every CTA of this grid has executed it, i.e. once all of them are
  // resident (and the router, launched before us, is resident too), so MegaMoE's early CTAs only take free SMs; they
  // do nothing but set-up before their own griddepcontrol.wait, which returns after this grid completes (and this grid
  // completes only after the router has: the wait at the end).
  if (p.pdl_mode == 2) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  MSTAMP(0);
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(bar)));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // Weights before the dependency wait: this split's fragment-ordered B image, one bulk copy.
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(kBBytes));
    const uint8_t* src = reinterpret_cast<const uint8_t*>(p.fn_img) + (size_t)split * kBBytes;
    constexpr int kChunk = 30720;
#pragma unroll
    for (int c = 0; c < kBBytes / kChunk; ++c)
      asm volatile(
          "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
              smem_u32(smem + c * kChunk)),
          "l"(src + c * kChunk), "r"(kChunk), "r"(smem_u32(bar))
          : "memory");
  }
  if (p.l2_prefetch != nullptr && mt == 0 && tid == 32) {  // this split's slab of the next mix's image, into L2
    const long long off = (long long)split * kBBytes;
    if (off < p.l2_prefetch_bytes) {
      const long long n = p.l2_prefetch_bytes - off < kBBytes ? p.l2_prefetch_bytes - off : kBBytes;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.l2_prefetch + off), "r"((uint32_t)n)
                   : "memory");
    }
  }
  // Side-stream launch: wait for the boundary that wrote rn. PDL mode (M3, the router's dependent): the router triggers
  // this kernel only after its own dependency wait returned, so rn is complete already; the wait is at the end.
  if (!p.pdl_mode) asm volatile("griddepcontrol.wait;" ::: "memory");
  // No early griddepcontrol.launch_dependents: a joined or following PDL kernel (MegaMoE) must launch only when this
  // kernel exits.
  MSTAMP(1);
  const int rows = min(16, p.T - mt * 16);
  // A image: element (r, k) of the tile goes to step k/8, lane (r%8)*4 + k%4, register (k%8 >= 4)*2 + (r >= 8);
  // within a step, lane slot l is stored at l ^ (step & 31) so the converter's stores (32 steps per warp
  // instruction) are bank-conflict free. Warp w converts row pair (w, w + 8); lane = step % 32.
  {
    const int g = warp;  // 8 warps = 8 row pairs
    uint4 lo[5], hi[5];
    const __nv_bfloat16* row_lo = p.rn + (size_t)(mt * 16 + g) * 4 * kHidden + split * kKps;
    const __nv_bfloat16* row_hi = p.rn + (size_t)(mt * 16 + g + 8) * 4 * kHidden + split * kKps;
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
      float4* dst = reinterpret_cast<float4*>(sA) + st * 32;
#pragma unroll
      for (int tig = 0; tig < 4; ++tig)
        dst[(g * 4 + tig) ^ (st & 31)] = make_float4(a[tig], b[tig], a[tig + 4], b[tig + 4]);
    }
  }
  __syncthreads();
  MSTAMP(2);
  if (warp < 3) {
    // One n-atom chain per warp (warps 0-2 sit on different SM sub-partitions): 160 dependent mma.sync.
    const int na = warp;
    mbar_wait(smem_u32(bar), 0);
    MSTAMP(6);  // the B image has landed (warp 0)
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    const float4* a4 = reinterpret_cast<const float4*>(sA);
    // B image = 4 slots of 40 steps: [slot][n-atom][step in slot][lane] (the W8 ring layout).
    for (int s = 0; s < 4; ++s) {
      const float2* b2 = sB + ((size_t)s * 3 + na) * 40 * 32;
#pragma unroll 8
      for (int i = 0; i < 40; ++i) {
        const int st = s * 40 + i;
        mma_tf32(c, a4[st * 32 + (lane ^ (st & 31))], b2[i * 32 + lane]);
      }
    }
    const int g = lane >> 2, tig = lane & 3, n0 = na * 8 + 2 * tig;
    if (g < rows) *reinterpret_cast<float2*>(p.mul + ((size_t)split * p.T + mt * 16 + g) * kMix + n0) =
        make_float2(c[0], c[1]);
    if (g + 8 < rows) *reinterpret_cast<float2*>(p.mul + ((size_t)split * p.T + mt * 16 + g + 8) * kMix + n0) =
        make_float2(c[2], c[3]);
  } else if (warp == 3) {
    // DeepGEMM's per-split sum of squares, one warp on its own SM sub-partition, reading the chain's A image a float4
    // slot per lane and step (conflict-free, the chain's own access pattern). Lane (g, tig) holds rows g / g + 8 at
    // columns tig / tig + 4 of every step: four fma(v, v, acc) chains in step order. Column c = 2q is DeepGEMM's
    // x-lane of quad lane q, c = 2q + 1 its y-lane: v(q) = x + y joins lanes tig ^ 1, and its butterfly over q
    // (xor 2, then 1) is (v0 + v2) + (v1 + v3): columns 0 / 4 (q 0 / 2) and 2 / 6 (q 1 / 3) meet in lanes tig 0 / 2.
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const float4* a4 = reinterpret_cast<const float4*>(sA);
#pragma unroll 8
    for (int st = 0; st < kSteps; ++st) {
      const float4 f = a4[st * 32 + (lane ^ (st & 31))];
      acc[0] = __fmaf_rn(f.x, f.x, acc[0]);  // row g,     column tig
      acc[1] = __fmaf_rn(f.y, f.y, acc[1]);  // row g + 8, column tig
      acc[2] = __fmaf_rn(f.z, f.z, acc[2]);  // row g,     column tig + 4
      acc[3] = __fmaf_rn(f.w, f.w, acc[3]);  // row g + 8, column tig + 4
    }
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __fadd_rn(acc[i], __shfl_xor_sync(0xffffffffu, acc[i], 1));  // x + y
    const float lo = __fadd_rn(v[0], v[2]), hi = __fadd_rn(v[1], v[3]);  // rows g / g + 8: v(q) + v(q + 2)
    const float lo_t = __fadd_rn(lo, __shfl_xor_sync(0xffffffffu, lo, 2));
    const float hi_t = __fadd_rn(hi, __shfl_xor_sync(0xffffffffu, hi, 2));
    const int g = lane >> 2, tig = lane & 3;
    if (tig == 0) {
      if (g < rows) p.sqr[split * p.T + mt * 16 + g] = lo_t;
      if (g + 8 < rows) p.sqr[split * p.T + mt * 16 + g + 8] = hi_t;
    }
  }
  MSTAMP(3);
  // The last split of this token tile computes the stats. The barrier orders the CTA's partial stores before thread
  // 0's acq_rel arrival (release is cumulative), and the winner's acquire before its barrier: no per-thread fences.
  __syncthreads();
  if (tid == 0) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p.counters + mt) : "memory");
    last_flag = old == kSplits - 1;
  }
  __syncthreads();
  if (!last_flag) return;
  if (p.dbg != nullptr && tid == 0) {
    p.dbg[4] = gtimer();
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    p.dbg[7] = smid;  // the last CTA's SM (where the stats tail runs)
  }
  // Stats: split sums by all threads, then 16 lanes per token, 2 tokens per warp.
  float* msum = reinterpret_cast<float*>(smem);  // reuse the B image region: [16 tokens][25]
  stats_sums<kSplits>(p.mul, p.sqr, p.T, mt * 16, rows, msum, tid, kThreads);
  __syncthreads();
  {
    const int g = warp * 2 + (lane >> 4);  // 8 warps x 2 tokens = the 16-token tile
    stats_from_sums(msum, g, mt * 16 + g, g < rows, p.scale, p.base, p.post_out, p.comb_out, p.pre_out);
  }
  if (tid == 0) p.counters[mt] = 0u;
  if (p.dbg != nullptr && tid == 0) p.dbg[5] = gtimer();
  // PDL mode: this grid must not complete before the router's (MegaMoE reads the router's staging after us).
  if (p.pdl_mode != 0) asm volatile("griddepcontrol.wait;" ::: "memory");
}

}  // namespace

extern "C" int w4_mix_launch(const MixParams* p, int pdl, cudaStream_t stream) {
  cudaFuncSetAttribute(mix_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kSplits, (p->T + 15) / 16);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = kSmem;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  return (int)cudaLaunchKernelEx(&cfg, mix_kernel, *p);
}
