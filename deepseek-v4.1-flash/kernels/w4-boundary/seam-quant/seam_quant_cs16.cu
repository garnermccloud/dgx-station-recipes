// lib_seam_quant_cs16.so: the W4 seam of src/csrc/boundary.cu for quant_producer (cluster 16, gsize 1, T 6/8 only).
//
//  w4_seam:  one launch = [cold-row add] + mHC post + collapse + exact RMSNorm + layer input (bf16) + MXFP8 forms.
//            A cluster of CS CTAs per token group; CTA r owns TileLang virtual threads [r*64/CS, (r+1)*64/CS).
//            The 64 per-token partial sums of squares are exchanged through DSMEM.
//  (w4_mix, the next sublayer's coefficients, is in mix.cu.)
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "w4/mhc_core.cuh"
#include "w4/params.h"
#include "w4/seam.cuh"

namespace {

using namespace w4;

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ float ld_dsmem_f32(const float* local, uint32_t rank) {
  uint32_t la = (uint32_t)__cvta_generic_to_shared(local), ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(la), "r"(rank));
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");
  return v;
}
__device__ __forceinline__ int ld_dsmem_i32(const int* local, uint32_t rank) {
  uint32_t la = (uint32_t)__cvta_generic_to_shared(local), ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(la), "r"(rank));
  int v;
  asm volatile("ld.shared::cluster.s32 %0, [%1];" : "=r"(v) : "r"(ra) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define STAMP(i) \
  if (p.dbg != nullptr && blockIdx.x == 0 && threadIdx.x == 0) p.dbg[i] = gtimer();
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

constexpr int kSeamThreads = 256;
constexpr int kGmax = 32;  // max tokens per cluster (smem slots)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_async_f32(uint32_t remote, float v, uint32_t remote_bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(remote),
               "r"(__float_as_uint(v)), "r"(remote_bar)
               : "memory");
}
__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n W4_WAIT%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra W4_WAIT%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}

// peer3 segment control words this kernel uses: the RTX's done tag, and (COPY_AHEAD) the host ready word the RTX sets
// after the done tag; in the device mirror, the copied tag (+128) and the chain's ready word (+136).
constexpr uint64_t kP3OffDone = 128, kP3OffReady = 1152, kMirrorReady = 136;

// Thread 0: wait for the RTX's acknowledgement of this call's tag (bounded by p3_spins host polls). Returns 2 when the
// device mirror holds this call's tag and rows (the copy-engine chain finished: L2 reads), 1 when the host segment
// does, 0 on timeout (a dead RTX: then also release the chain's stream wait on the host ready word, so the graph's
// replay still completes). CTA 0 records the wait.
__device__ __forceinline__ int p3_wait_ack(const SeamParams& p) {
  const uint32_t tag = (p.p3_state[0] << 12) | ((uint32_t)p.p3_layer << 6) | (uint32_t)(p.T - 1);
  const unsigned long long w0 = gtimer();
  auto mirror_ok = [&]() {
    uint32_t rd;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(rd) : "l"(p.p3_mirror + kMirrorReady) : "memory");
    if (rd == 0) return false;
    unsigned long long mt;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(mt) : "l"(p.p3_mirror + kP3OffDone) : "memory");
    return mt == tag;  // a stale chain (a mismatched tag) never counts
  };
  int how = p.p3_mirror != nullptr && mirror_ok() ? 2 : 0;
  if (!how) {
    long long spins = 0;
    for (;;) {
      unsigned long long done;
      asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(done) : "l"(p.p3_seg + kP3OffDone) : "memory");
      if (done == tag) {
        how = 1;
        break;
      }
      if (p.p3_mirror != nullptr && mirror_ok()) {
        how = 2;
        break;
      }
      if (++spins >= p.p3_spins) break;
    }
    if (!how && p.p3_mirror != nullptr)
      asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p.p3_seg + kP3OffReady), "r"(1u) : "memory");
  }
  if (blockIdx.x == 0 && p.p3_stats != nullptr) {  // peer3 GbStats: calls, waited, wait_ns, timeouts, hist[16]
    const unsigned long long ns = gtimer() - w0;
    atomicAdd(p.p3_stats + 0, 1ull);
    atomicAdd(p.p3_stats + 2, ns);
    if (!how) atomicAdd(p.p3_stats + 3, 1ull);
    int bkt = 0;
    for (unsigned long long us = ns / 1000; us > 0 && bkt < 15; us >>= 1) ++bkt;
    atomicAdd(p.p3_stats + 4 + bkt, 1ull);
    if (p.p3_stats_words >= 22) p.p3_stats[20] = ns, p.p3_stats[21] = tag;
  }
  return how;
}
// Thread 0 at the CTA's end: CTA 0 waits for the ack if it has not yet, then the arrival; the last CTA clears the
// mirror's ready word and advances the call counter. (The counter is read by tag at the start of p3_wait_ack, before
// any CTA's arrival.)
__device__ __forceinline__ void p3_finish_arrive(const SeamParams& p, int waited, uint32_t nclusters) {
  if (blockIdx.x == 0 && waited < 0) p3_wait_ack(p);
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p.p3_state + 1) : "memory");
  if (old == nclusters - 1) {
    if (p.p3_mirror != nullptr) {
      *reinterpret_cast<volatile uint32_t*>(p.p3_mirror + kMirrorReady) = 0u;
      // release this call's copy chain in any case (it may not have seen the RTX's word yet): every chain then ends
      // with its call's seam, so none can lag into the next call and the join at the forward's end always completes.
      // The next router arms the word again before its records; a chain that passes this way copies final data.
      asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p.p3_seg + kP3OffReady), "r"(1u) : "memory");
    }
    p.p3_state[1] = 0u;
    p.p3_state[0] = p.p3_state[0] >= 15 ? 1u : p.p3_state[0] + 1u;
    __threadfence();
  }
}

template <int CS>
__global__ void __launch_bounds__(kSeamThreads, 1) seam_kernel(const SeamParams p, int gsize) {
  constexpr int VPC = kVt / CS;
  constexpr int CHUNKS = 5 * 2 * VPC;  // 8-column chunks per token owned by this CTA
  __shared__ uint4 c_sh[kGmax][CHUNKS];
  __shared__ uint4 w_sh[CHUNKS];          // norm weights (bf16 x 8) of the chunks
  __shared__ float part_all[kGmax][kVt];  // every token's 64 virtual-thread partials, pushed by all CTAs
  __shared__ float rs_sh[kGmax];
  __shared__ __align__(8) uint64_t bar;
  const uint32_t rank = cluster_rank();
  const int cid = blockIdx.x / CS;
  const int t0 = cid * gsize;
  const int tg = min(gsize, p.T - t0);
  const int tid = threadIdx.x;
  STAMP(0);
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar)));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bar)),
                 "r"(tg * kVt * 4));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __shared__ int p3_ok;  // 2: rows in the device mirror, 1: in the host segment, 0: timed out, -1: not waited yet
  cluster_arrive();  // peers may push into our smem only after this (matched by the wait before the push)
  auto chunk_col = [&](int ci) {
    const int vl = ci % VPC, hh = (ci / VPC) & 1, b = ci / (2 * VPC);
    return vt_col(b, hh, (int)rank * VPC + vl, 0);
  };
  // Weights: asynchronous copies that land while phase A runs.
  for (int ci = tid; ci < CHUNKS; ci += kSeamThreads) cp_async16(&w_sh[ci], p.norm_w + chunk_col(ci));
  asm volatile("cp.async.commit_group;" ::: "memory");
  STAMP(1);
  pdl_launch();  // at the start: the next kernel (e.g. a weight-streaming GEMM) may launch early
  pdl_wait();
  STAMP(2);
  STAMP(9);
  // Fused peer-v3 finish. A CTA with a cold token waits for the RTX's acknowledgement before phase A (it adds the
  // RTX's rows); CTA 0 without one waits at the end instead. Every cluster's leader arrives on p3_state[1] at its end;
  // the last cluster advances the call counter (read by the next layer's router).
  if (p.p3_seg != nullptr && tid == 0) {
    bool cold = false;
    for (int g = 0; g < tg; ++g) cold |= p.cold_has[t0 + g] != 0;
    p3_ok = cold ? p3_wait_ack(p) : -1;
  }
  if (p.p3_seg != nullptr) __syncthreads();
  STAMP(10);
  const bool cold_ok = p.p3_seg != nullptr && p3_ok >= 1;
  const uint8_t* const p3_rows = p.p3_seg != nullptr && p3_ok == 2 ? p.p3_mirror : p.p3_seg;
  // Phase A: post + collapse, one chunk per item.
  for (int item = tid; item < tg * CHUNKS; item += kSeamThreads) {
    const int g = item / CHUNKS, ci = item % CHUNKS;
    TokenMix m;
    load_token_mix(p, t0 + g, m);
    c_sh[g][ci] = p.cmb_slots != nullptr  // launch-uniform
                      ? seam_chunk_post(p, t0 + g, chunk_col(ci), m, cold_ok, SeamXCombine{}, p3_rows)
                      : seam_chunk_post(p, t0 + g, chunk_col(ci), m, cold_ok, SeamXBf16{}, p3_rows);
  }
  __syncthreads();
  STAMP(3);
  // Post-only boundary (an Engram injection or the final collapse comes next): the streams are all there is.
  if (p.layer_input == nullptr && p.fi_q == nullptr && p.prep_x == nullptr) {  // launch-uniform
    asm volatile("cp.async.wait_all;" ::: "memory");  // the norm weights' copies (unused here)
    cluster_wait();
    if (p.p3_seg != nullptr) {
      cluster_arrive(); cluster_wait();
      if (tid == 0 && rank == 0) { STAMP(11); p3_finish_arrive(p, p3_ok, gridDim.x / CS); STAMP(12); }
    }
    return;
  }
  cluster_wait();
  // Phase B: per-position chains, then push each virtual-thread partial to every CTA of the cluster.
  for (int base = 0; base < tg * VPC * 2; base += kSeamThreads) {
    const int item = base + tid;
    const bool active = item < tg * VPC * 2;
    const int hh = item & 1, vl = (item >> 1) % VPC, g = active ? (item >> 1) / VPC : 0;
    uint4 c[5];
#pragma unroll
    for (int b = 0; b < 5; ++b) c[b] = c_sh[g][(b * 2 + hh) * VPC + vl];
    const float s = seam_vt_chain(c);
    if (active && hh == 0) {
      const uint32_t la = smem_u32(&part_all[g][rank * VPC + vl]), lb = smem_u32(&bar);
#pragma unroll
      for (int r = 0; r < CS; ++r) st_async_f32(mapa(la, r), s, mapa(lb, r));
    }
  }
  STAMP(4);
  mbar_wait_parity(smem_u32(&bar), 0);
  STAMP(5);
  // Phase C: every CTA reduces the 64 partials of each of its tokens in TileLang's order.
  const int warp = tid >> 5, lane = tid & 31;
  for (int gg = warp; gg < tg; gg += kSeamThreads / 32) {
    const float tot = vt_butterfly(part_all[gg][lane], part_all[gg][lane + 32]);
    if (lane == 0) rs_sh[gg] = norm_rsqrt(tot);
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  STAMP(6);
  // Phase D: normalize + quantize, one chunk per item (MXFP8 blocks = 4 consecutive lanes).
  for (int base = 0; base < tg * CHUNKS; base += kSeamThreads) {
    const int item = base + tid;
    const bool active = item < tg * CHUNKS;
    const int g = active ? item / CHUNKS : 0, ci = item % CHUNKS;
    if (active) {  // CS16/Gsize1: ten complete four-lane quant groups.
      float w[8];
      unpack8(w_sh[ci], w);
      seam_chunk_norm(p, t0 + g, chunk_col(ci), c_sh[g][ci], rs_sh[g], w, true);
    }
  }
  STAMP(7);
  seam_zero_sf_padding(p, blockIdx.x, gridDim.x, tid, kSeamThreads);
  if (p.p3_seg != nullptr) {
    cluster_arrive(); cluster_wait();
    if (tid == 0 && rank == 0) { STAMP(11); p3_finish_arrive(p, p3_ok, gridDim.x / CS); STAMP(12); }
  }
  STAMP(8);
}

}  // namespace

extern "C" {

int w4_seam_launch(const SeamParams* p, int cluster, int gsize, int pdl, cudaStream_t stream) {
  if (cluster != 16 || gsize != 1 || (p->T != 6 && p->T != 8) ||
      !p->layer_input || !p->fi_q || !p->fi_sf_swz || p->prep_x || p->fi_sf_lin)
    return 9410;
  const int nclusters = (p->T + gsize - 1) / gsize;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(cluster * nclusters);
  cfg.blockDim = dim3(kSeamThreads);
  cfg.stream = stream;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = cluster, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = pdl;
  cfg.attrs = at;
  cfg.numAttrs = 2;
  cudaError_t e;
  if (cluster == 16) {
    cudaFuncSetAttribute(seam_kernel<16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    e = cudaLaunchKernelEx(&cfg, seam_kernel<16>, *p, gsize);

  } else {
    return -1;
  }
  return (int)e;
}


}  // extern "C"

// Layout checks for the ctypes mirrors in fastpath/fastpath_boundary.py.
#include <cstddef>
extern "C" int w4_abi(int which) {
  switch (which) {
    case 0: return (int)sizeof(w4::SeamParams);
    case 1: return (int)offsetof(w4::SeamParams, dbg);
    case 2: return (int)sizeof(w4::MixParams);
    case 3: return (int)offsetof(w4::MixParams, dbg);
    case 4: return (int)sizeof(w4::RouterParams);
    case 5: return (int)offsetof(w4::RouterParams, dbg);
    case 6: return (int)offsetof(w4::RouterParams, peer_pad);
    case 7: return (int)offsetof(w4::RouterParams, quant_cold);
    case 8: return (int)offsetof(w4::RouterParams, p3_seg);
    // peer tier v3 segment offsets the router writes (checked against peer3's extension at boot)
    case 9: return 4096;                // kOffRec
    case 10: return 4096 + 64 * 128;    // kOffXq
    case 11: return 4096 + 64 * 128 + 64 * 5120;  // kOffXs
    case 12: return (int)sizeof(w4::MegaParams);
    case 13: return (int)offsetof(w4::MegaParams, dbg);
    case 14: return (int)w4::kP3OffOut;
    case 15: return (int)offsetof(w4::SeamParams, p3_seg);
    case 16: return (int)sizeof(w4::DraftRouterParams);
    case 17: return (int)offsetof(w4::DraftRouterParams, dbg);
    case 18: return (int)sizeof(w4::HeadParams);
    case 19: return (int)offsetof(w4::HeadParams, dbg);
    case 20: return (int)offsetof(w4::MixParams, pdl_mode);
    case 21: return (int)offsetof(w4::RouterParams, trigger_late);
    default: return -1;
  }
}
