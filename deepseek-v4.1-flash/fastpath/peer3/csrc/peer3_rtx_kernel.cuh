// Peer tier v3 RTX persistent kernel body (design in peer3_rtx.cuh).
#pragma once
#include "peer3_rtx.cuh"

namespace peer3 {

// Cross-CTA hand-offs follow the cooperative-groups / CUTLASS semaphore pattern: the CTA's writes, __syncthreads,
// then one release (or acq_rel atomic) by thread 0, which is cumulative over the barrier-ordered writes; the reader's
// thread 0 acquires, __syncthreads, and every thread reads with __ldcg (L2). No per-thread fences.
// ---------------------------------------------------------------------------------------------------------------
// Helpers. Device-memory loads of data produced in this call use __ldcg (L2, not L1; not volatile, so the compiler
// can batch them). Host-segment loads use ld.volatile (never cached). No printf anywhere (a stack frame makes the
// driver reserve local memory for every resident thread), and no cluster-scoped PTX (on sm_120 it made the launch
// reserve ~3.75 GB of device memory).
// ---------------------------------------------------------------------------------------------------------------
__device__ char* g_trap_seg;
__device__ __forceinline__ void trap_if_late(uint64_t t0, uint64_t limit, int where) {
  if (limit != 0 && globaltimer() - t0 > limit) {
    if (g_trap_seg != nullptr)
      st_release_sys_u64(g_trap_seg + kOffRtxStats + 56,
                         (uint64_t(where) << 32) | (blockIdx.x << 16) | threadIdx.x | (1ull << 63));
    __trap();
  }
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t ld_volatile_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint64_t ld_relaxed_gpu_u64(const void* p) {
  uint64_t v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_gpu_u64(void* p, uint64_t v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// old = *p; *p = old >= lim ? 0 : old + 1 (a counter that wraps by itself after lim + 1 arrivals)
__device__ __forceinline__ uint32_t atom_inc_acq_rel_gpu(uint32_t* p, uint32_t lim) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(lim) : "memory");
  return old;
}
// One thread: spin until *f == want (acquire).
__device__ __forceinline__ void wait_flag(const uint32_t* f, uint32_t want, uint64_t limit, int where) {
  const uint64_t t0 = globaltimer();
  while (ld_acquire_gpu_u32(f) != want) trap_if_late(t0, limit, where);
}
// Phase events for benchmarks (PEER3_EVENTS builds only; the atomics cost a little on the critical path).
__device__ __forceinline__ void mark(RtxWork* W, int ev) {
#ifdef PEER3_EVENTS
  if (threadIdx.x == 0) W->cta_ev[blockIdx.x][ev] = static_cast<unsigned long long>(globaltimer());  // plain store
#endif
}

// 8 FP4 (one 32-bit word, element i in nibble i) as fp32 values times 2^-14, order (0,4),(1,5),(2,6),(3,7).
__device__ __forceinline__ void decode_fp4x8(uint32_t w, float (&v)[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t s = w >> (4 * i);
    const uint32_t h = ((s & 0x00070007u) << 9) | ((s & 0x00080008u) << 12);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
    v[2 * i + 0] = f.x;
    v[2 * i + 1] = f.y;
  }
}
// 8 bf16 (16 bytes) as fp32 in the same (0,4),(1,5),(2,6),(3,7) order.
__device__ __forceinline__ void bf16x8_to_f32_pairs(const uint4 u, float (&x)[8]) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
  float e[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    e[2 * j + 0] = __uint_as_float(w[j] << 16);
    e[2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    x[2 * i + 0] = e[i];
    x[2 * i + 1] = e[i + 4];
  }
}
// sum_i w_i * x_i in a fixed order; the products are exact in fp32.
__device__ __forceinline__ float dot8(const float (&w)[8], const float (&x)[8]) {
  float d = w[0] * x[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) d = __fmaf_rn(w[i], x[i], d);
  return d;
}
// Exact d * 2^(s - 127) * 2^14 as two power-of-two multiplies (no exponent overflow for s <= 254).
__device__ __forceinline__ float apply_scale(float d, uint32_t s) {
  const uint32_t s1 = s < 240u ? s : 240u;
  return __fmul_rn(__fmul_rn(d, __uint_as_float((s1 + 14u) << 23)), __uint_as_float((s - s1 + 127u) << 23));
}
// MegaMoE's UE8M0 scale for e4m3 (deep_gemm common/math.cuh get_ue8m0_sf_exp, fp32 input).
__device__ __forceinline__ uint32_t ue8m0_sf_exp_e4m3(float amax) {
  const uint32_t rounded = (__float_as_uint(amax) + 0x7FFFFFu - 0x600000u) >> 23;
  return (rounded > 113u ? rounded : 113u) - 8u;
}
__device__ __forceinline__ float fast_rcp(float x) {
  float r;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// 16 e4m3 bytes times an exact power of two -> 16 bf16 (exact for normal results; the sidecar _dequant values).
__device__ __forceinline__ void e4m3x16_to_bf16(const uint4 q, float sc, uint4 (&out)[2]) {
  const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
  uint32_t o[8];
#pragma unroll
  for (int b = 0; b < 4; ++b)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t((qw[b] >> (16 * h)) & 0xFFFF), __NV_E4M3);
      const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
      const __nv_bfloat162 r = __floats2bfloat162_rn(__fmul_rn(f.x, sc), __fmul_rn(f.y, sc));
      o[b * 2 + h] = *reinterpret_cast<const uint32_t*>(&r);
    }
  out[0] = make_uint4(o[0], o[1], o[2], o[3]);
  out[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

// ---------------------------------------------------------------------------------------------------------------
// Shared memory and the tile pipeline (cp.async.bulk + mbarrier on sm_90+, cooperative loads otherwise)
// ---------------------------------------------------------------------------------------------------------------
struct alignas(128) Stage {
  uint8_t w[2][kTileBytes];
  uint8_t sf[2][kSfBytes];
};

struct alignas(128) Smem {
  Stage st[kStages];
  union {  // the chunk's MXFP8 operand rows for this K slice: e4m3 bytes + one UE8M0 per 32
    struct {
      uint8_t q[kMT][kHidden / 2];      // FC1 activations (ks1 >= 2 keeps a slice within half of K)
      uint8_t sf[kMT][kHidden / 64];
    } x;
    struct {
      uint8_t q[kMT][kInter];           // FC2 intermediate rows
      uint8_t sf[kMT][kInter / 32];
    } a;
  } in;
  uint64_t bar[kStages];
  // The call, as every CTA rebuilds it from the T word0s (deterministic, so all CTAs agree)
  uint64_t ids[kMaxTokens];
  uint16_t exp_id[kMaxExperts];
  uint16_t exp_start[kMaxExperts + 1];
  uint8_t pair_token[kMaxPairs];
  uint8_t pair_slot[kMaxPairs];
  int16_t pair_of[kMaxTokens][kTopk];
  uint8_t rows[kMaxTokens];
  int16_t slot_of[kMaxExperts];
  uint16_t count[kMaxExperts];
  uint32_t tag, flag, n_exp, n_pairs, ks1, ks2, n_rows;
  uint32_t hot_bits[kNumExperts / 32];  // statistics: distinct hot experts of the call (CTA 0)
  uint32_t cbits[kExpWords];            // the call's distinct cold experts (plan_experts)
};

// The stages form a ring: tiles are consumed in the order they were issued, whatever item they belong to.
struct Pipe {
  uint32_t parity;  // bit s = parity of stage s's next completion (a bitmask, so it stays in a register)
  uint32_t issue, wait;  // next stage to fill / to consume
};

__device__ __forceinline__ void pipe_issue(Smem& sm, Pipe& pp, const uint8_t* wa, const uint8_t* sa,
                                           const uint8_t* wb, const uint8_t* sb) {
  const int s = int(pp.issue);
  pp.issue = (pp.issue + 1) % kStages;
  const uint8_t* srcs[4] = {wa, sa, wb, sb};
  uint8_t* dsts[4] = {sm.st[s].w[0], sm.st[s].sf[0], sm.st[s].w[1], sm.st[s].sf[1]};
  const int n = wb != nullptr ? 4 : 2;
#if __CUDA_ARCH__ >= 900 && !defined(PEER3_NO_BULK)
  if (threadIdx.x == 0) {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    const uint32_t bytes = (kTileBytes + kSfBytes) * (n / 2);
    const uint32_t bar = smem_u32(&sm.bar[s]);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (i >= n) break;
      const uint32_t size = (i & 1) ? kSfBytes : kTileBytes;
      asm volatile(
          "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
          ::"r"(smem_u32(dsts[i])), "l"(srcs[i]), "r"(size), "r"(bar), "l"(policy)
          : "memory");
    }
  }
#else
  // Pre-Hopper fallback (single-GPU logic tests): synchronous cooperative copy.
  for (int i = 0; i < n; ++i) {
    const int size = (i & 1) ? kSfBytes : kTileBytes;
    for (int c = threadIdx.x; c < size / 16; c += blockDim.x)
      reinterpret_cast<uint4*>(dsts[i])[c] = __ldg(reinterpret_cast<const uint4*>(srcs[i]) + c);
  }
#endif
}

// Returns the stage that now holds the oldest issued tiles.
__device__ __forceinline__ int pipe_wait(Smem& sm, Pipe& pp) {
  const int s = int(pp.wait);
  pp.wait = (pp.wait + 1) % kStages;
#if __CUDA_ARCH__ >= 900 && !defined(PEER3_NO_BULK)
  const uint32_t parity = (pp.parity >> s) & 1u;
  const uint32_t bar = smem_u32(&sm.bar[s]);
  asm volatile(
      "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra DONE;\n\tbra LAB_WAIT;\n\tDONE:\n\t}" ::"r"(bar),
      "r"(parity)
      : "memory");
#else
  __syncthreads();
#endif
  pp.parity ^= 1u << s;
  return s;
}

// One 16 KB tile (256 rows x 128 K) of the prepared layout against the chunk's MXFP8 rows (<= kMT tokens).
// Lane (gid = lane >> 2, tig = lane & 3) of warp n8c holds, per K32 step, rows n8c*32 + n*8 + gid (n = 0..3) at
// K offsets 8*tig .. 8*tig+7 (one 16-byte load). Results land in the m16n8 accumulator layout: acc[mb][0..3] =
// rows (n8c*32 + 2mb*8 + gid, +8) x tokens (2tig, 2tig+1), mb = 0, 1 (the warp's two 16-row blocks).
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define PEER3_QMMA 1
// SM120 block-scaled QMMA: A = weights (e2m1 in 8-bit containers, nibble in bits 5:2), B = activations (e4m3),
// UE8M0 scale per 32 K for every A row and B column, f32 accumulate. The lane's 8 consecutive K (8tig..8tig+7)
// feed MMA k {4tig..4tig+3} and {16+4tig..16+4tig+3}, for A and B alike, so the products pair up exactly.
__device__ __forceinline__ void e2m1x8_to_qmma(uint32_t w, uint32_t& lo, uint32_t& hi) {
  uint32_t a, b;
  asm("{\n\t.reg .b32 s, t;\n\tshl.b32 s, %2, 2;\n\tshr.u32 t, %2, 2;\n\t"
      "prmt.b32 %0, s, t, 0x5140;\n\tprmt.b32 %1, s, t, 0x7362;\n\t}"
      : "=r"(a), "=r"(b) : "r"(w));
  lo = a & 0x3C3C3C3Cu;
  hi = b & 0x3C3C3C3Cu;
}
template <int KB>
__device__ __forceinline__ void qmma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %13}, {%11}, {%12, %13};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "n"(KB), "n"(0));
}
template <int KB>
__device__ __forceinline__ void tile_k32(const uint8_t* wt, uint32_t sfa0, uint32_t sfa1, uint2 bx, uint32_t sfb,
                                         int n8c, int gid, int tig, float (&acc)[2][4]) {
  const uint4 wv = *reinterpret_cast<const uint4*>(wt + 4 * (KB << 10 | n8c << 7 | gid << 4 | tig << 2));
  uint32_t a0, a1, a2, a3;
  e2m1x8_to_qmma(wv.x, a0, a2);
  e2m1x8_to_qmma(wv.y, a1, a3);
  qmma<KB>(acc[0], a0, a1, a2, a3, bx.x, bx.y, sfa0, sfb);
  e2m1x8_to_qmma(wv.z, a0, a2);
  e2m1x8_to_qmma(wv.w, a1, a3);
  qmma<KB>(acc[1], a0, a1, a2, a3, bx.x, bx.y, sfa1, sfb);
}
// `q` / `sf`: the operand rows of this tile's 128 K for token j at q + j*q_stride, sf + j*sf_stride (4 bytes).
__device__ __forceinline__ void tile_math(const uint8_t* wt, const uint8_t* sft, const uint8_t* q, int q_stride,
                                          const uint8_t* sf, int sf_stride, int mt, float (&acc)[2][4]) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  const bool live = gid < mt;
  // A scales: the lane supplies row gid (tig even) or gid + 8 (tig odd) of each 16-row block, all 4 K32 bytes.
  const uint8_t* sa = sft + (warp * 4 + (tig & 1)) * 32 + gid * 4;
  const uint32_t sfa0 = *reinterpret_cast<const uint32_t*>(sa);
  const uint32_t sfa1 = *reinterpret_cast<const uint32_t*>(sa + 64);
  const uint32_t sfb = live ? *reinterpret_cast<const uint32_t*>(sf + gid * sf_stride) : 0x7F7F7F7Fu;
  const uint8_t* xr = q + gid * q_stride + 8 * tig;
  const uint2 z = make_uint2(0, 0);
  tile_k32<0>(wt, sfa0, sfa1, live ? *reinterpret_cast<const uint2*>(xr) : z, sfb, warp, gid, tig, acc);
  tile_k32<1>(wt, sfa0, sfa1, live ? *reinterpret_cast<const uint2*>(xr + 32) : z, sfb, warp, gid, tig, acc);
  tile_k32<2>(wt, sfa0, sfa1, live ? *reinterpret_cast<const uint2*>(xr + 64) : z, sfb, warp, gid, tig, acc);
  tile_k32<3>(wt, sfa0, sfa1, live ? *reinterpret_cast<const uint2*>(xr + 96) : z, sfb, warp, gid, tig, acc);
}
#else
// CUDA-core fallback (pre-SM120 logic tests): each lane computes its own m16n8 accumulator slots over the tile's
// 128 K directly (rows gid / gid + 8 of both 16-row blocks, tokens 2tig and 2tig + 1). Slow, but needs no scratch.
__device__ __forceinline__ void tile_math(const uint8_t* wt, const uint8_t* sft, const uint8_t* q, int q_stride,
                                          const uint8_t* sf, int sf_stride, int mt, float (&acc)[2][4]) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
  for (int mb = 0; mb < 2; ++mb)
    for (int h = 0; h < 2; ++h) {  // row gid (h = 0) or gid + 8 (h = 1) of the block
      const int n = 2 * mb + h;
      for (int c = 0; c < 2; ++c) {  // token 2tig + c
        const int j = 2 * tig + c;
        if (j >= mt) continue;
        float sum = 0.f;
        for (int k32 = 0; k32 < 4; ++k32) {
          const uint32_t ws = sft[warp * 128 + n * 32 + gid * 4 + k32];
          const uint32_t xs = sf[j * sf_stride + k32];
          float blk = 0.f;
          for (int cg = 0; cg < 4; ++cg) {
            const uint32_t w = *reinterpret_cast<const uint32_t*>(wt + 4 * (k32 << 10 | warp << 7 | gid << 4 | cg << 2 | n));
            float wf[8];
            decode_fp4x8(w, wf);  // values * 2^-14, order (0,4),(1,5),(2,6),(3,7)
            const uint8_t* xr = q + j * q_stride + k32 * 32 + 8 * cg;
            float xv[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(xr[i] | (xr[i + 4] << 8)), __NV_E4M3);
              const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
              xv[2 * i] = f.x;
              xv[2 * i + 1] = f.y;
            }
            blk = __fadd_rn(blk, dot8(wf, xv));
          }
          float d = apply_scale(blk, ws);                                 // * 2^(ws - 127) * 2^14
          d = __fmul_rn(d, __uint_as_float((xs > 0 ? xs : 1u) << 23));  // * 2^(xs - 127)
          sum = __fadd_rn(sum, d);
        }
        acc[mb][h * 2 + c] = __fadd_rn(acc[mb][h * 2 + c], sum);
      }
    }
}
#endif

// Lane's accumulator slots -> fp32 partial rows: dst(token j, row q) for q in the warp's 32 rows.
template <typename F>
__device__ __forceinline__ void store_acc(const float (&acc)[2][4], int mt, F&& dst) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int mb = 0; mb < 2; ++mb) {
    const int q0 = warp * 32 + 2 * mb * 8 + gid;
    if (2 * tig < mt) {
      dst(2 * tig, q0) = acc[mb][0];
      dst(2 * tig, q0 + 8) = acc[mb][2];
    }
    if (2 * tig + 1 < mt) {
      dst(2 * tig + 1, q0) = acc[mb][1];
      dst(2 * tig + 1, q0 + 8) = acc[mb][3];
    }
  }
}

// MegaMoE SwiGLU + MXFP8 for one (pair, channel); warp-cooperative over 32 consecutive channels.
__device__ __forceinline__ void swiglu_quant_store(const RtxParams& p, int pair, int ch, float gate, float up,
                                                   float w) {
  __nv_bfloat16 gb = __float2bfloat16_rn(gate);
  __nv_bfloat16 ub = __float2bfloat16_rn(up);
  gb = __hmin(gb, __float2bfloat16_rn(kSwigluLimit));
  ub = __hmax(ub, __float2bfloat16_rn(-kSwigluLimit));
  ub = __hmin(ub, __float2bfloat16_rn(kSwigluLimit));
  const float gf = __bfloat162float(gb);
  const float den = __fadd_rn(1.0f, __expf(-gf));
  const float gated = __fmul_rn(gf, fast_rcp(den));
  const float act = __fmul_rn(__fmul_rn(gated, __bfloat162float(ub)), w);
  float amax = fabsf(act);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const uint32_t sf = ue8m0_sf_exp_e4m3(amax);
  const float inv = __uint_as_float((254u - sf) << 23);
  p.inter[size_t(pair) * kInter + ch] = __nv_cvt_float_to_fp8(__fmul_rn(act, inv), __NV_SATFINITE, __NV_E4M3);
  if ((threadIdx.x & 31) == 0) p.inter_sf[size_t(pair) * (kInter / 32) + ch / 32] = uint8_t(sf);
}

// ---------------------------------------------------------------------------------------------------------------
// Work item geometry
// ---------------------------------------------------------------------------------------------------------------
struct Fc1Item {
  int pe, tp, ksi, e, p0, p1, kt0, nkt;
  size_t tg0, tu0;  // first gate / up tile of the slice
};
__device__ __forceinline__ Fc1Item fc1_geom(const Smem& sm, int item, int ks1) {
  Fc1Item f;
  f.ksi = item % ks1;
  f.tp = (item / ks1) % kUpTiles;
  f.pe = item / (ks1 * kUpTiles);
  f.e = sm.exp_id[f.pe];
  f.p0 = sm.exp_start[f.pe];
  f.p1 = sm.exp_start[f.pe + 1];
  f.kt0 = f.ksi * kKt1 / ks1;
  f.nkt = (f.ksi + 1) * kKt1 / ks1 - f.kt0;
  const size_t texp = size_t(f.e) * (2 * kUpTiles) * kKt1;
  f.tg0 = texp + size_t(f.tp + kUpTiles) * kKt1 + f.kt0;  // gate rows: prepared tiles 9..17
  f.tu0 = texp + size_t(f.tp) * kKt1 + f.kt0;             // up rows: prepared tiles 0..8
  return f;
}
__device__ __forceinline__ void fc1_issue(Smem& sm, Pipe& pp, const LayerWeights& L, const Fc1Item& f, int u) {
  pipe_issue(sm, pp, L.w13 + (f.tg0 + u) * kTileBytes, L.s13 + (f.tg0 + u) * kSfBytes,
             L.w13 + (f.tu0 + u) * kTileBytes, L.s13 + (f.tu0 + u) * kSfBytes);
}

struct Fc2Item {
  int pe, nt, ksi, e, p0, p1, kt0, nkt;
  size_t t0;
};
__device__ __forceinline__ Fc2Item fc2_geom(const Smem& sm, int item, int ks2) {
  Fc2Item f;
  f.ksi = item % ks2;
  f.nt = (item / ks2) % kOutTiles;
  f.pe = item / (ks2 * kOutTiles);
  f.e = sm.exp_id[f.pe];
  f.p0 = sm.exp_start[f.pe];
  f.p1 = sm.exp_start[f.pe + 1];
  f.kt0 = f.ksi * kKt2 / ks2;
  f.nkt = (f.ksi + 1) * kKt2 / ks2 - f.kt0;
  f.t0 = (size_t(f.e) * kOutTiles + f.nt) * kKt2 + f.kt0;
  return f;
}
__device__ __forceinline__ void fc2_issue(Smem& sm, Pipe& pp, const LayerWeights& L, const Fc2Item& f, int u) {
  pipe_issue(sm, pp, L.w2 + (f.t0 + u) * kTileBytes, L.s2 + (f.t0 + u) * kSfBytes, nullptr, nullptr);
}

// ---------------------------------------------------------------------------------------------------------------
// FC1 item. Partials -> part1[(ksi*9 + tp)*n_pairs + pair][mat*256 + row] (mat 0 = gate, 1 = up). `issued` = the
// first stages were issued already (only for the first token chunk). After the item's last tile the CTA issues
// `next2` (its first FC2 item, or nullptr) so that FC2 weights stream during the FC1 epilogue.
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void fc1_item(const RtxParams& p, Smem& sm, Pipe& pp, const LayerWeights& L,
                                         uint32_t seq, const Fc1Item f, int n_pairs, int ks1, int n_stage_ctas,
                                         bool issued, bool prefetch2, const Fc2Item next2) {
  RtxWork* W = p.work;
  for (int c0 = f.p0; c0 < f.p1; c0 += kMT) {
    const int mt = min(kMT, f.p1 - c0);
    const bool last_chunk = c0 + kMT >= f.p1;
    if (!issued || c0 != f.p0)
      for (int u = 0; u < min(kStages, f.nkt); ++u) fc1_issue(sm, pp, L, f, u);
    if (c0 == f.p0) mark(W, kEvXWaitIn);
    if (int(threadIdx.x) < n_stage_ctas) wait_flag(&W->xready[threadIdx.x].v, seq, p.wait_limit_ns, 2);  // rows staged
    if (c0 == f.p0) mark(W, kEvXWait);
    __syncthreads();
    // The chunk tokens' MXFP8 rows of this K slice: e4m3 bytes and their UE8M0 scales, from the staged copy.
    const int kq = f.nkt * 128 / 16, ksf = f.nkt * 4 / 4;  // 16-byte chunks of values / 4-byte words of scales
    for (int idx = threadIdx.x; idx < mt * (kq + ksf); idx += blockDim.x) {
      const int j = idx / (kq + ksf), c = idx % (kq + ksf);
      const int tok = sm.pair_token[c0 + j];
      if (c < kq)
        *reinterpret_cast<uint4*>(&sm.in.x.q[j][c * 16]) =
            __ldcg(reinterpret_cast<const uint4*>(p.xstage + size_t(tok) * kHidden + f.kt0 * 128 + c * 16));
      else
        *reinterpret_cast<uint32_t*>(&sm.in.x.sf[j][(c - kq) * 4]) = __ldcg(reinterpret_cast<const uint32_t*>(
            p.xstage + size_t(kMaxTokens) * kHidden + size_t(tok) * kScalesPerRow + f.kt0 * 4 + (c - kq) * 4));
    }
    __syncthreads();
    if (c0 == f.p0) mark(W, kEvXSlice);
    float acc[2][2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int b = 0; b < 2; ++b)
#pragma unroll
        for (int n = 0; n < 4; ++n) acc[m][b][n] = 0.f;
    for (int u = 0; u < f.nkt; ++u) {
      const int s = pipe_wait(sm, pp);
      if (u == 0 && c0 == f.p0) mark(W, kEvFirstTile);
#ifndef PEER3_BISECT_NO_MATH
      tile_math(sm.st[s].w[0], sm.st[s].sf[0], &sm.in.x.q[0][u * 128], kHidden / 2, &sm.in.x.sf[0][u * 4],
                kHidden / 64, mt, acc[0]);
      tile_math(sm.st[s].w[1], sm.st[s].sf[1], &sm.in.x.q[0][u * 128], kHidden / 2, &sm.in.x.sf[0][u * 4],
                kHidden / 64, mt, acc[1]);
#endif
      __syncthreads();
      if (u + kStages < f.nkt) {
        fc1_issue(sm, pp, L, f, u + kStages);
      } else if (last_chunk && prefetch2) {  // the stage is free for good: start FC2's tiles
        const int v = u + kStages - f.nkt;  // 0, 1, ...
        if (v < min(kStages, next2.nkt)) fc2_issue(sm, pp, L, next2, v);
      }
    }
    float* part = p.part1 + size_t(f.ksi * kUpTiles + f.tp) * n_pairs * 512;
#pragma unroll
    for (int m = 0; m < 2; ++m)
      store_acc(acc[m], mt, [&](int j, int q) -> float& { return part[size_t(c0 + j) * 512 + m * 256 + q]; });
  }
  mark(W, kEvFc1);
  __syncthreads();
  if (threadIdx.x == 0) {
    sm.flag = atom_inc_acq_rel_gpu(&W->fc1_count[f.pe * kUpTiles + f.tp], uint32_t(ks1 - 1)) == uint32_t(ks1 - 1);
  }
  __syncthreads();
  if (!sm.flag) return;
  // Last K slice of (pe, tp): sum the slices in order 0..ks1-1, SwiGLU, quantize.
  if (threadIdx.x == 0) wait_flag(&W->w_ready.v, seq, p.wait_limit_ns, 8);
  __syncthreads();
  const int c = threadIdx.x;  // channel within the tile pair
  for (int pr = f.p0; pr < f.p1; ++pr) {
    float g[kKt1 / 2], uu[kKt1 / 2];
#pragma unroll
    for (int s = 0; s < kKt1 / 2; ++s)
      if (s < ks1) {
        const float* part = p.part1 + (size_t(s * kUpTiles + f.tp) * n_pairs + pr) * 512;
        g[s] = __ldcg(part + c);
        uu[s] = __ldcg(part + 256 + c);
      }
    float gsum = g[0], usum = uu[0];
#pragma unroll
    for (int s = 1; s < kKt1 / 2; ++s)
      if (s < ks1) {
        gsum = __fadd_rn(gsum, g[s]);
        usum = __fadd_rn(usum, uu[s]);
      }
    swiglu_quant_store(p, pr, f.tp * 256 + c, gsum, usum, __ldcg(&W->pair_w[pr]));
  }
  __syncthreads();
  if (threadIdx.x == 0) st_release_gpu_u32(&W->fc1_tp[f.pe].tp[f.tp], seq);
  mark(W, kEvFc1Epi);
}

// ---------------------------------------------------------------------------------------------------------------
// FC2 item; the last item of a column tile finalizes its 256 columns.
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void fc2_finalize(const RtxParams& p, const Smem& sm, int nt, int n_pairs, int ks2,
                                             uint32_t tag, uint32_t seq) {
  RtxWork* W = p.work;
  mark(W, kEvFinStart);
  const int c = threadIdx.x;
  __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(p.seg + kOffOut);
  const int n_rows = int(sm.n_rows);
  for (int r = 0; r < n_rows; ++r) {
    const int t = sm.rows[r];
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < kTopk; ++k) {
      const int pr = sm.pair_of[t][k];
      if (pr < 0) continue;
      float v[kKt2 / 2];
#pragma unroll
      for (int q = 0; q < kKt2 / 2; ++q)
        if (q < ks2) v[q] = __ldcg(p.part2 + (size_t(q * kOutTiles + nt) * n_pairs + pr) * 256 + c);
      float a = v[0];
#pragma unroll
      for (int q = 1; q < kKt2 / 2; ++q)
        if (q < ks2) a = __fadd_rn(a, v[q]);
      s = __fadd_rn(s, __bfloat162float(__float2bfloat16_rn(a)));
    }
    out[size_t(t) * kHidden + nt * 256 + c] = __float2bfloat16_rn(s);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // The system fence flushes the CTA's rows (cumulative over the barrier) before the arrival; done is then a plain
    // system-scope store (a release would be a second fence on the call's critical path).
    fence_sys();
    mark(W, kEvFinFence);
    const bool last = atom_inc_acq_rel_gpu(&W->done_count.v, uint32_t(kOutTiles - 1)) == uint32_t(kOutTiles - 1);
    if (last) {  // every finalizer fenced its rows before its arrival
      st_relaxed_sys_u64(p.seg + kOffDone, tag);
      st_release_sys_u64(p.seg + kOffReady, 1);  // d1: after the done tag (the release orders it)
      st_release_gpu_u32(&W->fin_seq.v, seq);  // the call is complete: idle CTAs may prefetch the next layer
      mark(W, kEvDone);
    }
  }
}

__device__ __forceinline__ void fc2_item(const RtxParams& p, Smem& sm, Pipe& pp, const LayerWeights& L,
                                         const Fc2Item f, int n_exp, int n_pairs, int ks2, uint32_t tag, uint32_t seq,
                                         bool issued) {
  RtxWork* W = p.work;
  for (int c0 = f.p0; c0 < f.p1; c0 += kMT) {
    const int mt = min(kMT, f.p1 - c0);
    if (!issued || c0 != f.p0)
      for (int u = 0; u < min(kStages, f.nkt); ++u) fc2_issue(sm, pp, L, f, u);
    {  // only the FC1 tile pairs this K slice reads: channels [kt0, kt0 + nkt) * 128
      const int tp0 = f.kt0 / 2, ntp = (f.kt0 + f.nkt - 1) / 2 - tp0 + 1;
      if (int(threadIdx.x) < ntp) wait_flag(&W->fc1_tp[f.pe].tp[tp0 + threadIdx.x], seq, p.wait_limit_ns, 7);
    }
    mark(W, kEvFc2Wait);
    __syncthreads();
    // The chunk pairs' intermediate rows of this K slice (e4m3 + UE8M0, as the FC1 epilogue wrote them).
    const int kq = f.nkt * 128 / 16, ksf = f.nkt;
    for (int idx = threadIdx.x; idx < mt * (kq + ksf); idx += blockDim.x) {
      const int j = idx / (kq + ksf), c = idx % (kq + ksf);
      const int pr = c0 + j;
      if (c < kq)
        *reinterpret_cast<uint4*>(&sm.in.a.q[j][c * 16]) =
            __ldcg(reinterpret_cast<const uint4*>(p.inter + size_t(pr) * kInter + f.kt0 * 128 + c * 16));
      else
        *reinterpret_cast<uint32_t*>(&sm.in.a.sf[j][(c - kq) * 4]) = __ldcg(
            reinterpret_cast<const uint32_t*>(p.inter_sf + size_t(pr) * (kInter / 32) + f.kt0 * 4 + (c - kq) * 4));
    }
    __syncthreads();
    float acc[2][4];
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int n = 0; n < 4; ++n) acc[b][n] = 0.f;
    for (int u = 0; u < f.nkt; ++u) {
      const int s = pipe_wait(sm, pp);
#ifndef PEER3_BISECT_NO_MATH
      tile_math(sm.st[s].w[0], sm.st[s].sf[0], &sm.in.a.q[0][u * 128], kInter, &sm.in.a.sf[0][u * 4], kInter / 32,
                mt, acc);
#endif
      __syncthreads();
      if (u + kStages < f.nkt) fc2_issue(sm, pp, L, f, u + kStages);
    }
    float* part = p.part2 + size_t(f.ksi * kOutTiles + f.nt) * n_pairs * 256;
    store_acc(acc, mt, [&](int j, int q) -> float& { return part[size_t(c0 + j) * 256 + q]; });
  }
  mark(W, kEvFc2);
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t lim = uint32_t(n_exp * ks2 - 1);
    sm.flag = atom_inc_acq_rel_gpu(&W->fc2_count[f.nt].v, lim) == lim;
  }
  __syncthreads();
  if (sm.flag) fc2_finalize(p, sm, f.nt, n_pairs, ks2, tag, seq);
}

// ---------------------------------------------------------------------------------------------------------------
// Row staging: the cold tokens' MXFP8 rows (e4m3 values, then their UE8M0 scales) copied from the host segment into
// device memory, one 16-byte chunk per thread over CTAs 1..nst.
// ---------------------------------------------------------------------------------------------------------------
// Stager `part` of num_stagers(n_rows) (CTAs 0 ..): its chunks of the cold rows, all host loads issued before the
// device stores (PCIe latency once per kStageInFlight chunks).
__device__ __forceinline__ void stage_rows(const RtxParams& p, const uint8_t* rows, int n_rows, int part,
                                           uint32_t seq) {
  RtxWork* W = p.work;
  const int nst = num_stagers(n_rows), n = n_rows * kRowChunks;
  for (int i0 = part * blockDim.x + threadIdx.x; i0 < n; i0 += nst * blockDim.x * kStageInFlight) {
    uint4 v[kStageInFlight];
#pragma unroll
    for (int j = 0; j < kStageInFlight; ++j) {
      const int i = i0 + j * nst * int(blockDim.x);
      if (i >= n) break;
      const int r = i / kRowChunks, c = i % kRowChunks, t = rows[r];
      v[j] = c < kHidden / 16 ? ld_volatile_v4(p.seg + kOffXq + size_t(t) * kHidden + c * 16)
                              : ld_volatile_v4(p.seg + kOffXs + size_t(t) * kScalesPerRow + (c - kHidden / 16) * 16);
    }
#pragma unroll
    for (int j = 0; j < kStageInFlight; ++j) {
      const int i = i0 + j * nst * int(blockDim.x);
      if (i >= n) break;
      const int r = i / kRowChunks, c = i % kRowChunks, t = rows[r];
      uint8_t* dst = c < kHidden / 16 ? p.xstage + size_t(t) * kHidden + c * 16
                                      : p.xstage + size_t(kMaxTokens) * kHidden + size_t(t) * kScalesPerRow + (c - kHidden / 16) * 16;
      *reinterpret_cast<uint4*>(dst) = v[j];
    }
  }
  mark(W, kEvXLoads);
  __syncthreads();
  if (threadIdx.x == 0) st_release_gpu_u32(&W->xready[part].v, seq);
  mark(W, kEvX);
}

// ---------------------------------------------------------------------------------------------------------------
// CTA 0, one thread: the plan from the T word0s in shared memory (weights are read after the release).
// ---------------------------------------------------------------------------------------------------------------
// Warp 0 of CTA 0, after the call was acknowledged (off the call's critical path): one statistics record into the
// host ring. Distinct hot experts come from the optional global ids in each token record (peer3::post_route_ids),
// counted only when every token's tag copy matches this call. Fields first, a system fence, then seq, so the
// sidecar's reader never pairs a new seq with old fields.
__device__ __forceinline__ void post_stats(const RtxParams& p, Smem& sm, uint32_t tag, int T, int n_rows, int n_pairs,
                                           int n_exp, bool timed) {
  RtxWork* W = p.work;
  const int lane = threadIdx.x & 31;
  if (lane < kNumExperts / 32) sm.hot_bits[lane] = 0;
  __syncwarp();
  bool known = true;
  for (int t = lane; t < T; t += 32) {
    const uint4 g = ld_volatile_v4(p.seg + kOffRec + size_t(t) * kRecBytes + kRecGids);
    if ((g.w & 0xFFFFu) != tag) {
      known = false;
      continue;
    }
    const uint32_t gw[3] = {g.x, g.y, g.z};
#pragma unroll
    for (int k = 0; k < kTopk; ++k) {
      if (word0_id(sm.ids[t], k) != 0xFFu) continue;  // cold (or not routed through the hot tier)
      const int e = int(int16_t(gw[k >> 1] >> (16 * (k & 1))));
      if (e >= 0 && e < kNumExperts) atomicOr(&sm.hot_bits[e >> 5], 1u << (e & 31));
    }
  }
  known = __all_sync(0xffffffffu, known);
  __syncwarp();
  const int n_hot = __reduce_add_sync(0xffffffffu, lane < kNumExperts / 32 ? __popc(sm.hot_bits[lane]) : 0);
  if (lane == 0) {
    const uint32_t seq = ++W->stat_seq;
    char* rec = p.seg + kOffStatRing + size_t(seq % kStatRing) * kStatRecBytes;
    const uint64_t busy = timed ? globaltimer() - W->call0 : 0;
    *reinterpret_cast<uint32_t*>(rec + 8) = tag;
    *reinterpret_cast<uint32_t*>(rec + 12) = uint32_t(n_rows) | (uint32_t(n_pairs) << 16);
    *reinterpret_cast<uint32_t*>(rec + 16) = uint32_t(n_exp) | (uint32_t(known ? n_hot : 0xFFFF) << 16);
    *reinterpret_cast<uint32_t*>(rec + 20) = uint32_t(busy > 0xFFFFFFFFull ? 0xFFFFFFFFull : busy);
    unsigned long long prev;  // the GB300's forwarded wait of its previous call (record 0, written before word0)
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(prev) : "l"(p.seg + kOffRec + kRecPrevWait));
    const bool fresh = uint32_t(prev >> 48) == tag;  // carried by this call (not left over from an earlier one)
    const uint32_t wait_ns = fresh ? uint32_t(prev & 0x7FFFFFFFull) : 0u;
    const uint32_t prev_tag = fresh ? uint32_t(prev >> 32) & 0xFFFFu : 0u;
    *reinterpret_cast<uint32_t*>(rec + 24) = wait_ns;
    *reinterpret_cast<uint32_t*>(rec + 28) = prev_tag;
    __threadfence_system();
    *reinterpret_cast<volatile unsigned long long*>(rec) = seq;
  }
  __syncwarp();
}

// ---------------------------------------------------------------------------------------------------------------
// Next-layer L2 prefetch. Layer calls come in order, and between two calls the RTX idles while the GB300 runs the
// rest of the layer; that idle DRAM time pulls the next layer's most popular cold experts into L2 (128 MB), so a
// call that picks one streams it from L2. Popularity is learned online: CTA 0 counts, per layer, the calls touching
// each cold expert and keeps the top kPrefetchMax. Results are unaffected (only cache state changes).
// ---------------------------------------------------------------------------------------------------------------
// Warp 0 of CTA 0, after a call with work: count its experts and refresh the layer's top list.
__device__ __forceinline__ void learn_popularity(const RtxParams& p, const Smem& sm, int layer, int n_exp) {
  RtxWork* W = p.work;
  const int lane = threadIdx.x & 31;
  const int n_cold = p.layers[layer].n_cold;
  uint32_t* pop = W->pop[layer];
  for (int e = lane; e < n_exp; e += 32) pop[sm.exp_id[e]] += 1;  // distinct within a call: no conflicts
  __syncwarp();
  const uint32_t calls = __shfl_sync(0xffffffffu, lane == 0 ? ++W->pop_calls[layer] : 0u, 0);
  if (calls % kPopHalveEvery == 0) {
    for (int e = lane; e < n_cold; e += 32) pop[e] >>= 1;
    __syncwarp();
  }
  uint32_t taken = 0;  // lane-local bitmask over this lane's entries e = lane + 32 i (n_cold <= 160)
  int n = 0;
  for (; n < kPrefetchMax; ++n) {
    uint32_t best = 0;
    int arg = -1;
    for (int i = 0; lane + 32 * i < n_cold; ++i)
      if (!(taken >> i & 1u) && pop[lane + 32 * i] > best) {
        best = pop[lane + 32 * i];
        arg = lane + 32 * i;
      }
    // warp argmax, ties to the lowest expert id (deterministic)
    uint64_t key = (uint64_t(best) << 32) | uint32_t(arg < 0 ? 0xFFFFFFFFu : ~uint32_t(arg));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const uint64_t other = __shfl_xor_sync(0xffffffffu, key, o);
      key = other > key ? other : key;
    }
    if ((key >> 32) == 0) break;
    const int win = int(~uint32_t(key));
    if ((win & 31) == lane) taken |= 1u << (win >> 5);
    if (lane == 0) W->top[layer][n] = uint16_t(win);
  }
  if (lane == 0) W->top_n[layer] = uint32_t(n);
}

// Every CTA but 0 (the poller), after a call with work: once the call's done is out, issue this CTA's share of
// L2 prefetches for the next layer's top experts. An expert's weights are four contiguous regions (w13 tiles, their
// scales, w2 tiles, their scales), cut into 64 KB pieces; piece j of expert e belongs to CTA 1 + (e*kPieces + j) %
// (G - 1). Stops early if the next call has already been released. Benchmarks (bench_discard) first drop the next
// layer's weights from L2 (discard.global.L2, read-only data), each CTA its own pieces, so only prefetched experts
// can hit.
constexpr uint64_t kPiece = 65536;
constexpr uint64_t kRegW13 = uint64_t(2 * kUpTiles) * kKt1 * kTileBytes, kRegS13 = uint64_t(2 * kUpTiles) * kKt1 * kSfBytes;
constexpr uint64_t kRegW2 = uint64_t(kOutTiles) * kKt2 * kTileBytes, kRegS2 = uint64_t(kOutTiles) * kKt2 * kSfBytes;
constexpr uint64_t kPcW13 = (kRegW13 + kPiece - 1) / kPiece, kPcS13 = (kRegS13 + kPiece - 1) / kPiece;
constexpr uint64_t kPcW2 = (kRegW2 + kPiece - 1) / kPiece, kPcS2 = (kRegS2 + kPiece - 1) / kPiece;
constexpr int kPieces = int(kPcW13 + kPcS13 + kPcW2 + kPcS2);  // 288 per expert

// Piece j of expert e: its address and length.
__device__ __forceinline__ const uint8_t* expert_piece(const LayerWeights& L, int e, int j, uint32_t& bytes) {
  uint64_t q = uint64_t(j);
  const uint8_t* base;
  uint64_t len;
  if (q < kPcW13) { base = L.w13 + uint64_t(e) * kRegW13; len = kRegW13; }
  else if ((q -= kPcW13) < kPcS13) { base = L.s13 + uint64_t(e) * kRegS13; len = kRegS13; }
  else if ((q -= kPcS13) < kPcW2) { base = L.w2 + uint64_t(e) * kRegW2; len = kRegW2; }
  else { q -= kPcW2; base = L.s2 + uint64_t(e) * kRegS2; len = kRegS2; }
  const uint64_t off = q * kPiece;
  bytes = uint32_t(min(kPiece, len - off));
  return base + off;
}

__device__ __forceinline__ uint32_t call_seq_now(const RtxWork* W) {  // seq of the newest call block this CTA reads
  return uint32_t(ld_relaxed_gpu_u64(&W->call[blockIdx.x % kCallCopies][0]) >> 32);
}
__device__ __forceinline__ void prefetch_next_layer(const RtxParams& p, uint32_t tag, uint32_t seq) {
#if __CUDA_ARCH__ >= 900 && !defined(PEER3_NO_BULK)
  RtxWork* W = p.work;
  __shared__ int go;
  if (threadIdx.x == 0) {
    const uint64_t t0 = globaltimer();
    uint32_t f;
    while ((f = ld_acquire_gpu_u32(&W->fin_seq.v)) != seq && call_seq_now(W) == seq)
      trap_if_late(t0, p.wait_limit_ns, 11);
    go = f == seq && call_seq_now(W) == seq;
  }
  __syncthreads();
  if (!go) return;
  int next = int(tag_layer(tag)) + 1;
  if (next >= kMaxLayers || p.layers[next].n_cold == 0) next = 0;
  if (p.hint_k > 0 && next >= p.hint_min_layer) return;  // the GB300's hint for that layer is better informed
  const LayerWeights L = p.layers[next];
  const int G1 = int(gridDim.x) - 1, me = int(blockIdx.x) - 1;
  if (p.bench_discard) {
    for (int e = 0; e < L.n_cold; ++e) {
      const int j0 = ((me - e * kPieces) % G1 + G1) % G1;  // this CTA's first piece of expert e
      for (int j = j0; j < kPieces; j += G1) {
        uint32_t bytes;
        const uint8_t* a = expert_piece(L, e, j, bytes);
        for (uint32_t o = threadIdx.x * 128u; o < bytes; o += blockDim.x * 128u)
          asm volatile("discard.global.L2 [%0], 128;" ::"l"(a + o) : "memory");
      }
    }
    __syncthreads();
  }
  const int k = min(p.prefetch_k, int(__ldcg(&W->top_n[next])));
  for (int t = 0; t < k; ++t) {
    const int e = __ldcg(&W->top[next][t]);
    const int j0 = ((me - e * kPieces) % G1 + G1) % G1;
    for (int j = j0 + int(threadIdx.x) * G1; j < kPieces; j += int(blockDim.x) * G1) {
      uint32_t bytes;
      const uint8_t* a = expert_piece(L, e, j, bytes);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a), "r"(bytes) : "memory");
    }
  }
#endif
}

// Warp 0: the rows list (tokens with a cold pick, ascending) from sm.ids.
__device__ __forceinline__ void build_rows(Smem& sm, int T) {
  const int lane = threadIdx.x & 31;
  int n_rows = 0;
  for (int t0 = 0; t0 < T; t0 += 32) {
    const int t = t0 + lane;
    bool cold = false;
    if (t < T) {
#pragma unroll
      for (int k = 0; k < kTopk; ++k) cold |= word0_id(sm.ids[t], k) < uint32_t(kMaxExperts);
    }
    const uint32_t m = __ballot_sync(0xffffffffu, cold);
    if (cold) sm.rows[n_rows + __popc(m & ((1u << lane) - 1u))] = uint8_t(t);
    n_rows += __popc(m);
  }
  if (lane == 0) sm.n_rows = n_rows;
}

// Warp 0, part 1 of the plan (fast, so the weight tiles can start streaming): the call's distinct cold experts in
// ascending id order (a 5-word bitmap), the number of cold pairs and the K splits. Deterministic, so every CTA
// builds the same plan.
__device__ void plan_experts(const RtxParams& p, Smem& sm, int T, int G) {
  const int lane = threadIdx.x & 31;
  const int n_ent = T * kTopk;
  if (lane < kExpWords) sm.cbits[lane] = 0u;
  __syncwarp();
  int n_valid = 0;
  for (int i0 = 0; i0 < n_ent; i0 += 32) {
    const int i = i0 + lane;
    const uint32_t c = i < n_ent ? word0_id(sm.ids[i / kTopk], i % kTopk) : 0xFFu;
    const bool valid = c < uint32_t(kMaxExperts);
    if (valid) atomicOr(&sm.cbits[c >> 5], 1u << (c & 31));
    n_valid += __popc(__ballot_sync(0xffffffffu, valid));
  }
  __syncwarp();
  const uint32_t w = lane < kExpWords ? sm.cbits[lane] : 0u;
  const int cnt = __popc(w);
  int incl = cnt;
#pragma unroll
  for (int o = 1; o < 8; o <<= 1) {
    const int u = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += u;
  }
  const int n_exp = __shfl_sync(0xffffffffu, incl, 7);
  if (lane < kExpWords) {
    uint32_t x = w;
    for (int e = incl - cnt; x != 0u; ++e) {
      const int c = lane * 32 + __ffs(x) - 1;
      x &= x - 1u;
      sm.exp_id[e] = uint16_t(c);
      sm.slot_of[c] = int16_t(e);
    }
  }
  if (lane == 0) {
    // FC1: about half the grid (its tiles are 2x FC2's); FC2: about the whole grid, assigned from the far end so
    // the CTAs without an FC1 item start streaming FC2 at once.
    const int start = n_valid;
    const int t1 = p.target_items > 0 ? p.target_items : G / 2, t2 = G;
    int ks1 = n_exp ? (2 * t1 + kUpTiles * n_exp) / (2 * kUpTiles * n_exp) : 2;
    int ks2 = n_exp ? (2 * t2 + kOutTiles * n_exp) / (2 * kOutTiles * n_exp) : 1;
    ks1 = max(2, min(ks1, kKt1 / 2));  // >= 2 keeps a slice's activations within sm.in.x
    ks2 = max(1, min(ks2, kKt2 / 2));
    if (start > 0) {
      ks1 = max(2, min(ks1, p.part1_rows / start));
      ks2 = max(1, min(ks2, p.part2_rows / start));
    }
    sm.n_exp = n_exp;
    sm.n_pairs = start;
    sm.ks1 = ks1;
    sm.ks2 = ks2;
  }
  __syncwarp();
}

// Warp 0, part 2: (token, slot) pairs grouped by expert in (token, slot) order. Entries i = t*6 + k go through in
// chunks of 32 (lane = i % 32); match_any ranks equal experts inside a chunk.
__device__ void plan_pairs(Smem& sm, int T) {
  const int lane = threadIdx.x & 31;
  const uint32_t lt = (1u << lane) - 1u;
  const int n_ent = T * kTopk, n_exp = int(sm.n_exp);
  for (int e = lane; e < n_exp; e += 32) sm.count[e] = 0;
  __syncwarp();
  for (int i0 = 0; i0 < n_ent; i0 += 32) {
    const int i = i0 + lane;
    const uint32_t c = i < n_ent ? word0_id(sm.ids[i / kTopk], i % kTopk) : 0xFFu;
    const bool valid = c < uint32_t(kMaxExperts);
    const uint32_t vmask = __ballot_sync(0xffffffffu, valid);
    const uint32_t same = __match_any_sync(0xffffffffu, valid ? c : 0x100u + lane) & vmask;
    const bool leader = valid && (same & lt) == 0;
    int slot = -1, rank = 0;
    if (valid) {
      slot = sm.slot_of[c];
      rank = sm.count[slot] + __popc(same & lt);
    }
    __syncwarp();
    if (leader) sm.count[slot] += __popc(same);
    if (i < n_ent) sm.pair_of[i / kTopk][i % kTopk] = valid ? int16_t(rank) : int16_t(-1);  // rank for now
    __syncwarp();
  }
  // Exclusive prefix sum of the per-expert counts -> exp_start.
  int carry = 0;
  for (int e0 = 0; e0 < n_exp; e0 += 32) {
    const int e = e0 + lane;
    const int v = e < n_exp ? sm.count[e] : 0;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += u;
    }
    if (e < n_exp) sm.exp_start[e] = uint16_t(carry + incl - v);
    carry += __shfl_sync(0xffffffffu, incl, 31);
  }
  if (lane == 0) sm.exp_start[n_exp] = uint16_t(carry);
  __syncwarp();
  // Pass 2: pair index = expert start + rank.
  for (int i = lane; i < n_ent; i += 32) {
    const int t = i / kTopk, k = i % kTopk;
    const int rank = sm.pair_of[t][k];
    if (rank < 0) continue;
    const int pr = sm.exp_start[sm.slot_of[word0_id(sm.ids[t], k)]] + rank;
    sm.pair_token[pr] = uint8_t(t);
    sm.pair_slot[pr] = uint8_t(k);
    sm.pair_of[t][k] = int16_t(pr);
  }
  __syncwarp();
}

// ---------------------------------------------------------------------------------------------------------------
// The call block (RtxWork::call): CTA 0 publishes, every other CTA reads its copy. No fence on either side.
// ---------------------------------------------------------------------------------------------------------------
constexpr uint64_t kIds48 = 0xFFFFFFFFFFFFull;
// CTA 0, all threads: every copy's header and all kMaxTokens slot words (slots >= T carry no cold pick).
__device__ __forceinline__ void publish_call(RtxWork* W, const Smem& sm, uint32_t seq, uint32_t tag) {
  const int T = int(tag_tokens(tag));
  const uint64_t s16 = uint64_t(seq & 0xFFFFu) << 48;
  constexpr int kWords = 1 + kMaxTokens;
  for (int i = threadIdx.x; i < kCallCopies * kWords; i += blockDim.x) {
    const int c = i / kWords, w = i % kWords;
    const uint64_t v = w == 0 ? (uint64_t(seq) << 32 | tag) : s16 | ((w - 1 < T ? sm.ids[w - 1] : ~0ull) & kIds48);
    st_relaxed_gpu_u64(&W->call[c][w], v);
  }
}
// CTA 0, all threads: the exit block (header seq | tag 0) on every copy.
__device__ __forceinline__ void publish_exit(RtxWork* W, uint32_t seq) {
  if (int(threadIdx.x) < kCallCopies) st_relaxed_gpu_u64(&W->call[threadIdx.x][0], uint64_t(seq) << 32);
}
// CTA b > 0, one thread: its share of hint slot x's expert (64 KB pieces; piece j belongs to CTA 1 + (x * kPieces +
// j) % (G - 1)) into L2. Hint only: results never depend on it.
__device__ __forceinline__ int hint_count(const RtxParams& p, uint64_t hint) {
  return (int(hint >> 32) & 63) < kMaxLayers ? min(int(hint >> 38) & 7, p.hint_k) : 0;
}
__device__ __forceinline__ void prefetch_hint(const RtxParams& p, uint64_t hint, int x) {
#if __CUDA_ARCH__ >= 900 && !defined(PEER3_NO_BULK)
  const LayerWeights L = p.layers[int(hint >> 32) & 63];
  const int e = int(hint >> (8 * x)) & 0xFF;
  if (e >= L.n_cold) return;
  const int G1 = int(gridDim.x) - 1, me = int(blockIdx.x) - 1;
  for (int j = ((me - x * kPieces) % G1 + G1) % G1; j < kPieces; j += G1) {
    uint32_t bytes;
    const uint8_t* a = expert_piece(L, e, j, bytes);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a), "r"(bytes) : "memory");
  }
#endif
}

// Warp 0 of CTA b > 0: wait for call `want` in copy b % kCallCopies; on return sm.flag = 1 (exit) or sm.tag and
// sm.ids[0 .. T) hold the call. Lanes 0..15 poll the header's line, which also holds slots 0..14, so a call of
// T <= 15 needs one round trip once its line has landed; a slot word counts only once it carries want's low bits.
// Hinted experts go out one at a time, best first, p.hint_stagger_ns apart (about one expert's DRAM time), so the
// likeliest expert lands first and a call that arrives early meets at most one expert's prefetch traffic.
__device__ __forceinline__ void read_call(const RtxParams& p, Smem& sm, uint32_t want, uint64_t& hint_seen) {
  uint64_t hint_t0 = 0;
  int hint_next = 0, hint_n = 0;
  const int lane = threadIdx.x & 31;
  const unsigned long long* blk = p.work->call[blockIdx.x % kCallCopies];
  const uint32_t want16 = want & 0xFFFFu;
  const uint64_t idle0 = globaltimer();
  for (;;) {
    const uint64_t a = lane < 16 ? ld_relaxed_gpu_u64(blk + lane) : lane == 16 ? ld_relaxed_gpu_u64(blk + kHintWord) : 0ull;
    const uint64_t h = __shfl_sync(0xffffffffu, a, 0);
    const uint64_t hint = __shfl_sync(0xffffffffu, a, 16);
    if (hint != hint_seen) {  // a new hint while idle
      hint_seen = hint;
      hint_t0 = globaltimer();
      hint_next = 0;
      hint_n = p.hint_k > 0 ? hint_count(p, hint) : 0;
    }
    if (hint_next < hint_n && globaltimer() - hint_t0 >= uint64_t(hint_next) * p.hint_stagger_ns) {
      if (lane == 0) prefetch_hint(p, hint_seen, hint_next);
      ++hint_next;
    }
    if (uint32_t(h >> 32) == want) {
      const uint32_t tag = uint32_t(h) & 0xFFFFu;
      if (tag == 0) {
        if (lane == 0) sm.flag = 1;
        return;
      }
      const int T = int(tag_tokens(tag));
      uint64_t s0 = __shfl_down_sync(0xffffffffu, a, 1);  // slot lane = word lane + 1
      if (lane >= 15 && lane < T) s0 = ld_relaxed_gpu_u64(blk + 1 + lane);
      const uint64_t s1 = lane + 32 < T ? ld_relaxed_gpu_u64(blk + 33 + lane) : 0ull;
      const bool ok = (lane >= T || word0_tag(s0) == want16) && (lane + 32 >= T || word0_tag(s1) == want16);
      if (__all_sync(0xffffffffu, ok)) {
        if (lane < T) sm.ids[lane] = s0;
        if (lane + 32 < T) sm.ids[lane + 32] = s1;
        if (lane == 0) {
          sm.tag = tag;
          sm.flag = 0;
        }
        return;
      }
    }
    if (p.idle_limit_ns != 0) trap_if_late(idle0, p.idle_limit_ns + 2000000000ull, 0);
  }
}

__global__ void __launch_bounds__(kThreads, 1) rtx_persistent_kernel(RtxParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  RtxWork* W = p.work;
  const int G = gridDim.x;
  Pipe pp{};
  if (blockIdx.x == 0 && threadIdx.x == 0) g_trap_seg = p.seg;
#if __CUDA_ARCH__ >= 900 && !defined(PEER3_NO_BULK)
  if (threadIdx.x == 0) {
    for (int s = 0; s < kStages; ++s)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&sm.bar[s])) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
#endif
  uint32_t seq = call_seq_now(W);  // the last call (or exit) released; every copy agrees between launches
  // hints: CTA 0 forwards the segment's word when it changes; the others act on their copy when it changes
  uint64_t hint_fwd = 0, hint_seen = 0;
  if (blockIdx.x == 0) {
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(hint_fwd) : "l"(p.seg + kOffHint));
  } else {
    hint_seen = ld_relaxed_gpu_u64(&W->call[blockIdx.x % kCallCopies][kHintWord]);
  }
  for (;;) {
    if (blockIdx.x == 0) {
      // ---- CTA 0: detect and read the call; it acknowledges calls without cold picks by itself ----
      // Warp 0 polls every record's word0 at once (lane l: slots l and l + 32), so the call is complete one PCIe round
      // trip after its last record lands; slot 0's tag gives T.
      if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        const uint32_t last = W->last_tag;
        const uint64_t idle0 = globaltimer();
        uint32_t tag = 0, exit_now = 0;
        int n = 0;
        uint64_t w0, w1;
        for (;;) {
          w0 = ld_acquire_sys_u64(p.seg + kOffRec + size_t(lane) * kRecBytes);
          w1 = ld_acquire_sys_u64(p.seg + kOffRec + size_t(lane + 32) * kRecBytes);
          if (p.hint_k > 0) {  // the GB300's prefetch hint: forward a new one to every copy of the call block
            uint64_t hw = 0;
            if (lane == 0) asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(hw) : "l"(p.seg + kOffHint));
            hw = __shfl_sync(0xffffffffu, hw, 0);
            if (hw != hint_fwd) {
              hint_fwd = hw;
              if (lane < kCallCopies) st_relaxed_gpu_u64(&W->call[lane][kHintWord], hw);
            }
          }
          const uint32_t t0 = word0_tag(__shfl_sync(0xffffffffu, w0, 0));
          if (t0 != 0 && t0 != last) {
            const int T = int(tag_tokens(t0));
            const bool ok = (lane >= T || word0_tag(w0) == t0) && (lane + 32 >= T || word0_tag(w1) == t0);
            if (__all_sync(0xffffffffu, ok)) {
              tag = t0;
              break;
            }
          }
          if ((++n & 15) == 0) {
            bool stop = false;
            if (lane == 0)
              stop = ld_acquire_sys_u64(p.seg + kOffCtrl) != 0 ||
                     (p.idle_limit_ns != 0 && globaltimer() - idle0 > p.idle_limit_ns);
            if (__shfl_sync(0xffffffffu, stop, 0)) {
              exit_now = 1;
              break;
            }
          }
        }
        bool cold = false;
        if (!exit_now) {
          const int T = int(tag_tokens(tag));
          if (lane < T) sm.ids[lane] = w0;
          if (lane + 32 < T) sm.ids[lane + 32] = w1;
#pragma unroll
          for (int k = 0; k < kTopk; ++k)
            cold |= (lane < T && word0_id(w0, k) < uint32_t(kMaxExperts)) ||
                    (lane + 32 < T && word0_id(w1, k) < uint32_t(kMaxExperts));
        }
        cold = __any_sync(0xffffffffu, cold);
        if (lane == 0) {
          sm.tag = tag;
          sm.flag = exit_now ? 1u : cold ? 2u : 0u;
        }
      }
      __syncthreads();
      const uint32_t tag = sm.tag, mode = sm.flag;
      if (mode == 1) {
        publish_exit(W, seq + 1);
        if (threadIdx.x == 0) st_release_sys_u64(p.seg + kOffRtxState, 0);  // the sidecar relaunches on demand
        break;
      }
      const int T = int(tag_tokens(tag));
      if (mode == 0) {  // no cold pick: acknowledge alone
        if (threadIdx.x == 0) {
          st_relaxed_sys_u64(p.seg + kOffDone, tag);  // no data to order: no fence
          st_release_sys_u64(p.seg + kOffReady, 1);   // d1: after the done tag (the release orders it)
          W->last_tag = tag;
          W->stats[0] += 1;
        }
        if (threadIdx.x < 32) post_stats(p, sm, tag, T, 0, 0, 0, false);
        __syncthreads();
        continue;
      }
      ++seq;
      if (threadIdx.x == 0) W->call0 = globaltimer();
      publish_call(W, sm, seq, tag);
      if (threadIdx.x < 32) build_rows(sm, T);
      __syncthreads();
      mark(W, kEvPub);
      stage_rows(p, sm.rows, sm.n_rows, 0, seq);  // CTA 0 is stager 0: the cold rows start crossing PCIe at once
    } else {
      if (threadIdx.x < 32) read_call(p, sm, seq + 1, hint_seen);
      __syncthreads();
      if (sm.flag) break;
      ++seq;
      if (threadIdx.x < 32) build_rows(sm, int(tag_tokens(sm.tag)));
      __syncthreads();
      mark(W, kEvWake);
      if (int(blockIdx.x) < num_stagers(int(sm.n_rows))) stage_rows(p, sm.rows, sm.n_rows, int(blockIdx.x), seq);
    }
    const uint32_t tag = sm.tag;
    const int T = int(tag_tokens(tag));
    if (threadIdx.x < 32) plan_experts(p, sm, T, G);
    __syncthreads();
    if (blockIdx.x == 0) mark(W, kEvPlan);
    const int n_exp = int(sm.n_exp), n_pairs = int(sm.n_pairs);
    const int ks1 = int(sm.ks1), ks2 = int(sm.ks2);
    const LayerWeights L = p.layers[tag_layer(tag)];
    const uint64_t call0 = globaltimer();
    const int items1 = n_exp * kUpTiles * ks1, items2 = n_exp * kOutTiles * ks2;
    const int my1 = blockIdx.x == 0 ? G - 1 : int(blockIdx.x) - 1;  // FC1 item i on CTA i + 1: CTA 0 stages first
    const int my2 = G - 1 - int(blockIdx.x);  // FC2 items run from the far end of the grid

    // 1. weights of the first FC1 item (or, for CTAs without one, the first FC2 item) start streaming: their
    // addresses need only the expert list, so they go out before the pairs are planned.
    const bool has1 = my1 < items1, has2 = my2 < items2;
    Fc1Item f1{};
    Fc2Item f2{};
    if (has2) f2 = fc2_geom(sm, my2, ks2);
    bool f2_issued = false;
    if (has1) {
      f1 = fc1_geom(sm, my1, ks1);
      for (int u = 0; u < min(kStages, f1.nkt); ++u) fc1_issue(sm, pp, L, f1, u);
      mark(W, kEvIssue);
    } else if (has2) {
      for (int u = 0; u < min(kStages, f2.nkt); ++u) fc2_issue(sm, pp, L, f2, u);
      mark(W, kEvIssue2);
      f2_issued = true;
    }
    // every FC1 item's weight tiles into L2 now: DRAM streams from the call's start, before the rows (x) arrive
    if (threadIdx.x == 0) {
      for (int it = my1; it < items1; it += G) {
        const Fc1Item f = fc1_geom(sm, it, ks1);
        const uint32_t n = uint32_t(f.nkt);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(L.w13 + f.tg0 * kTileBytes), "r"(n * kTileBytes) : "memory");
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(L.w13 + f.tu0 * kTileBytes), "r"(n * kTileBytes) : "memory");
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(L.s13 + f.tg0 * kSfBytes), "r"(n * kSfBytes) : "memory");
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(L.s13 + f.tu0 * kSfBytes), "r"(n * kSfBytes) : "memory");
      }
    }
    if (threadIdx.x < 32) plan_pairs(sm, T);
    __syncthreads();
    mark(W, kEvPlanAll);
    const int n_rows = int(sm.n_rows);
    const int n_stage_ctas = num_stagers(n_rows);
    if (has1) f1 = fc1_geom(sm, my1, ks1);  // now with the expert's pairs
    if (has2) f2 = fc2_geom(sm, my2, ks2);
    // The last CTA reads the route weights from the host segment (only the FC1 epilogues need them).
    if (int(blockIdx.x) == G - 1) {
      for (int pr = threadIdx.x; pr < n_pairs; pr += blockDim.x) {
        const int t = sm.pair_token[pr], k = sm.pair_slot[pr];
        float wv;
        asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(wv) : "l"(p.seg + kOffRec + size_t(t) * kRecBytes + 8 + 4 * k));
        W->pair_w[pr] = wv;
      }
          __syncthreads();
      if (threadIdx.x == 0) st_release_gpu_u32(&W->w_ready.v, seq);
    }
    // 3. FC1 items; the last one prefetches the first FC2 item's tiles (unless they are in flight already).
    for (int it = my1; it < items1; it += G) {
      const bool first = it == my1, last = it + G >= items1;
      const Fc1Item f = first ? f1 : fc1_geom(sm, it, ks1);
      const bool pre2 = last && has2 && !f2_issued;
      fc1_item(p, sm, pp, L, seq, f, n_pairs, ks1, n_stage_ctas, first, pre2, f2);
      if (pre2) f2_issued = true;
    }
    const uint64_t t_fc1 = globaltimer();
    // 4. FC2 items, item i on CTA G-1-i (then every G-th).
    for (int it = my2; it < items2; it += G) {
      const bool first = it == my2;
      fc2_item(p, sm, pp, L, first ? f2 : fc2_geom(sm, it, ks2), n_exp, n_pairs, ks2, tag, seq, first && f2_issued);
    }

    // ---- end of call: every CTA arrives; CTA 0 waits, so the next call never overwrites a live one ----
    __syncthreads();
    if (threadIdx.x == 0) {
      if (atom_inc_acq_rel_gpu(&W->end_count.v, uint32_t(G - 1)) == uint32_t(G - 1))
        st_release_gpu_u32(&W->end_seq.v, seq);
      if (blockIdx.x == 0) {
        wait_flag(&W->end_seq.v, seq, p.wait_limit_ns, 6);
        W->last_tag = tag;
        W->stats[0] += 1;
        W->stats[1] += 1;
        W->stats[2] += n_pairs;
        W->stats[3] += n_exp;
        W->stats[4] += globaltimer() - call0;
        W->stats[6] += t_fc1 - call0;
      }
    }
#ifdef PEER3_EVENTS
    // CTA 0, warp 0 (off the call's critical path): earliest / latest of every event over the CTAs that hit it
    if (blockIdx.x == 0 && threadIdx.x < 32) {
      const unsigned long long c0 = W->call0;
      for (int i = 0; i < kNumEv; ++i) {
        unsigned long long mx = 0, mn = ~0ull;
        for (int b = threadIdx.x; b < G; b += 32) {
          const unsigned long long e = __ldcg(&W->cta_ev[b][i]);
          if (e > c0) mx = max(mx, e), mn = min(mn, e);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
          mn = min(mn, __shfl_xor_sync(0xffffffffu, mn, o));
        }
        if (threadIdx.x == 0 && mx > c0) W->ev_sum[i] += mx - c0, W->ev_min_sum[i] += mn - c0;
      }
    }
#endif
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x < 32) post_stats(p, sm, tag, T, n_rows, n_pairs, n_exp, true);
    if (p.flush != nullptr) {  // benchmark mode: evict this call's weights from L2 while idle
      uint32_t acc = 0;
      const size_t n = p.flush_bytes / 16;
      for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(G) * blockDim.x) {
        const uint4 v = __ldcg(p.flush + i);
        acc ^= v.x ^ v.w;
      }
      if (acc == 0x9E3779B9u) sm.flag = acc;
      __syncthreads();
    }
    if (p.prefetch_k > 0 || p.bench_discard) {
      if (blockIdx.x == 0) {
        if (threadIdx.x < 32) learn_popularity(p, sm, int(tag_layer(tag)), n_exp);
      } else {
        prefetch_next_layer(p, tag, seq);
      }
    }
  }
}

}  // namespace peer3
