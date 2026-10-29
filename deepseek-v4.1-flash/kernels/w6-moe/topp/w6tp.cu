// W6: exact top-p masking for the verify / sampling path (vLLM apply_top_k_top_p with top_p only), one thread-block
// cluster per row.
//
// Semantics (the definition vLLM's sort-based reference implements): sort a row's tokens by logit, descending, ties by
// lower index first; token j is kept iff the probability mass of the tokens before it is < p (so the kept set is the
// smallest prefix with mass >= p; at least one token). Masked logits become mask_value, kept ones are left untouched.
// Rows with p >= 1 and rows without a finite logit are left as they are. With a source tensor (bf16 or fp32) the
// kernel also performs the fp32 copy apply_sampling_params would make, writing every output element.
//
// Arithmetic: masses are expf(x - M) (M = row max) rounded to 46-bit fixed point (an exact token mass <= 2^46) and
// summed as 64-bit integers, so every sum is exact and independent of order; the only rounding is expf's (<= 2 ulp)
// and the fixed point's 2^-47 per token, versus fp32 partial sums of tl.exp values in the Triton kernel. The boundary
// test compares an exact integer prefix with p * Z in fp64.
//
// Per row, a cluster of C CTAs splits the vocabulary into C contiguous slices held in shared memory:
//   1. row max (cluster all-gather);
//   2. level 0: mass histogram over 512 linear buckets of d = M - x in [0, 33) (expf(-33) * 2^46 < 1); cluster totals
//      by reduce-scatter + all-gather over distributed shared memory, so every CTA finds the same boundary bucket;
//   3. the key range of that bucket's tokens (cluster min / max), then up to four 9-bit levels over the
//      order-preserving fp32 keys relative to the range's minimum, until the boundary key is exact;
//   4. the boundary value's duplicates: keep the first ceil((p Z - mass above) / mass) by index;
//   5. mask pass over the slice.
// Histograms (v5): kCopies shared copies of three 16-bit limbs per bucket in 32-bit words, filled by native 32-bit
// shared atomic adds (a 64-bit shared atomic is a CAS loop on this GPU); merged per CTA into exact 64-bit sums
// (lo + mid * 2^16 + hi * 2^32; every limb sum stays below 2^32), double-buffered across levels, so one cluster.sync
// per level; no global scratch, no inter-cluster dependence (no co-residency requirement). Cluster size C is any
// 4..16 (C need not divide the 512 buckets: CTA r owns buckets [r P, min(512, (r + 1) P)), P = ceil(512 / C)). All
// sums are exact integers, so the kept set does not depend on C or on the summation order: bit-identical to v4.
//   nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -Xcompiler -fPIC -shared w6tp.cu -o libw6tp.so
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cg = cooperative_groups;

namespace w6tp {
struct Args {
  float* logits;     // [B, V] fp32 output, row stride ld (elements): masked in place, or written in full from src
  long long ld;
  const float* p;    // [B] top-p per row
  int B, V;
  int slice;         // tokens per CTA: ceil(V / C) rounded up to a multiple of 4
  float mask_value;  // -inf in vLLM
  const void* src;   // optional [B, V] source (bf16 or fp32, row stride src_ld): fused fp32 copy + top-p
  long long src_ld;
  int src_bf16;
};
}  // namespace w6tp
using w6tp::Args;

namespace {

constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kBits = 9;
constexpr int kBuckets = 1 << kBits;  // == kThreads: one bucket per thread in the merge / select steps
constexpr float kDMax = 33.0f;
constexpr int kMaxCluster = 16;
constexpr int kMaxIters = 64;         // slice / kThreads for C >= 4 (tie path bookkeeping)
constexpr int kCopies = 4;            // shared histogram copies (warp w adds into copy w % kCopies)
constexpr int kMaxOwn = (kBuckets + 3) / 4;   // buckets owned per CTA (C >= 4)
constexpr int kRedMax = 528;                  // >= max over C in 4..16 of C * ceil(kBuckets / C)
static_assert(kBuckets == kThreads, "one bucket per thread");

__device__ __forceinline__ uint32_t okey(float x) {  // order-preserving; NaN lowest, -0 == +0
  if (x != x) return 0u;
  const uint32_t b = __float_as_uint(x == 0.0f ? 0.0f : x);
  return (b >> 31) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float okey_inv(uint32_t k) {
  return __uint_as_float((k >> 31) ? (k & 0x7FFFFFFFu) : ~k);
}
// fixed-point mass of a token with logit x; d = M - x
__device__ __forceinline__ unsigned long long fmass(float d) {
  return __float2ull_rn(expf(-d) * 0x1p46f);
}
__device__ __forceinline__ int lin_bucket(float d) { return min(kBuckets - 1, static_cast<int>(d * (kBuckets / kDMax))); }
__device__ __forceinline__ bool finite_or_inf_neg(float x) { return x > -__int_as_float(0x7f800000); }

#ifdef TP_TRACE  // tests: globaltimer marks of row 0, CTA 0, thread 0 at each phase (w6tp_trace reads them)
__device__ long long g_trace[32];
#define TMARK(i)                                                                      \
  do {                                                                                \
    if (blockIdx.x == 0 && threadIdx.x == 0) {                                        \
      long long t_;                                                                   \
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));                          \
      g_trace[i] = t_;                                                                \
    }                                                                                 \
  } while (0)
#else
#define TMARK(i) \
  do {           \
  } while (0)
#endif

template <typename T, typename Op>
__device__ __forceinline__ T warp_reduce(T v, Op op) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) v = op(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  return v;
}

struct Shared {
  uint32_t whist[kCopies][3][kBuckets];        // limb histograms of the current level (16-bit limbs of the mass)
  unsigned long long hist[2][kBuckets];        // per-CTA histograms (read by the cluster), double-buffered by level
  unsigned long long scan_w[kWarps];
  float redf[kWarps];
  uint32_t redk[2][kWarps];
  unsigned long long red[kRedMax];             // reduce-scatter: peer partials of the buckets this CTA owns
  unsigned long long own[kMaxOwn];             // reduce-scatter: totals of the owned buckets
  unsigned long long tot[kBuckets];            // all-gathered cluster totals of the current level
  int dupc[kMaxIters * kWarps];                // tie path: duplicates per (iteration, warp)
  unsigned long long cta_dups;                 // tie path: duplicates of x* in this CTA's slice
  float cta_max;
  uint32_t cta_kmin, cta_kmax;
  int sel;                       // selected bucket of the current level
  unsigned long long sel_above;  // mass strictly above the selected bucket
  unsigned long long sel_mass;   // mass of the selected bucket
  unsigned long long z;          // row mass
};

// Block-wide exclusive scan of one u64 per thread (thread order). Returns the exclusive prefix; *total = sum.
__device__ unsigned long long block_excl_scan(unsigned long long v, unsigned long long* total, Shared& sh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  unsigned long long inc = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned long long t = __shfl_up_sync(0xFFFFFFFFu, inc, o);
    if (lane >= o) inc += t;
  }
  if (lane == 31) sh.scan_w[warp] = inc;
  __syncthreads();
  unsigned long long base = 0, all = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    const unsigned long long s = sh.scan_w[w];
    base += w < warp ? s : 0ull;
    all += s;
  }
  __syncthreads();  // scan_w is reused by the next call
  *total = all;
  return base + inc - v;
}

// Histogram add of a mass f < 2^47 into copy wh (three 16-bit limbs, native 32-bit shared atomic adds: RED.ADD).
// A copy takes the elements of kWarps / kCopies warps, <= 4 * 32 * kMaxIters = 8192 of them, so each limb sum stays
// below 8192 * 2^16 = 2^29.
__device__ __forceinline__ void warp_hist_add(uint32_t* wh, int bucket, unsigned long long f) {
  if (bucket >= 0) {
    atomicAdd(wh + bucket, static_cast<uint32_t>(f & 0xFFFFu));
    atomicAdd(wh + kBuckets + bucket, static_cast<uint32_t>((f >> 16) & 0xFFFFu));
    const uint32_t hi = static_cast<uint32_t>(f >> 32);
    if (hi) atomicAdd(wh + 2 * kBuckets + bucket, hi);
  }
}

// Merge the limb histograms into exact u64 sums in hist[buf] (one bucket per thread) and clear them for the next level.
__device__ __forceinline__ void merge_warp_hists(Shared& sh, int buf) {
  __syncthreads();
  unsigned long long lo = 0, mid = 0, hi = 0;
#pragma unroll
  for (int c = 0; c < kCopies; ++c) {
    lo += sh.whist[c][0][threadIdx.x];
    mid += sh.whist[c][1][threadIdx.x];
    hi += sh.whist[c][2][threadIdx.x];
    sh.whist[c][0][threadIdx.x] = 0;
    sh.whist[c][1][threadIdx.x] = 0;
    sh.whist[c][2][threadIdx.x] = 0;
  }
  sh.hist[buf][threadIdx.x] = lo + (mid << 16) + (hi << 32);
}

// Cluster totals of hist[buf] by reduce-scatter + all-gather: CTA `rank` owns <= ceil(kBuckets / C) buckets, sums
// them over the C peers (~one remote load per thread) and writes the totals into every peer's tot[] (~one remote
// store per thread); after a cluster.sync each CTA scans tot[] in rank order (rank r is bucket rank_to_bucket(r),
// rank 0 = most probable) and selects the bucket whose exclusive mass prefix (plus `above`) is < T and inclusive one
// >= T, identically everywhere. The caller has just passed a cluster.sync after writing hist[buf]. With want_total,
// T is p and the target becomes max(0.5, p * total).
template <typename RankToBucket>
__device__ void select_bucket(cg::cluster_group& cl, Shared& sh, int C, int rank, int buf, unsigned long long above,
                              double T, bool want_total, RankToBucket rank_to_bucket) {
  const int P = (kBuckets + C - 1) / C;
  const int b_lo = rank * P, nb = max(0, min(kBuckets, b_lo + P) - b_lo);  // this CTA's buckets [b_lo, b_lo + nb)
  for (int t = static_cast<int>(threadIdx.x); t < C * nb; t += kThreads)
    sh.red[t] = cl.map_shared_rank(&sh, t / nb)->hist[buf][b_lo + t % nb];
  if (threadIdx.x == 0) sh.sel = -1;
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < nb) {
    unsigned long long s = 0;
#pragma unroll
    for (int r = 0; r < kMaxCluster; ++r)
      if (r < C) s += sh.red[r * nb + threadIdx.x];
    sh.own[threadIdx.x] = s;
  }
  __syncthreads();
  for (int t = static_cast<int>(threadIdx.x); t < C * nb; t += kThreads)
    cl.map_shared_rank(&sh, t / nb)->tot[b_lo + t % nb] = sh.own[t % nb];
  cl.sync();
  const int j = rank_to_bucket(static_cast<int>(threadIdx.x));
  const unsigned long long h = sh.tot[j];
  unsigned long long total;
  const unsigned long long ex = block_excl_scan(h, &total, sh);
  if (want_total && threadIdx.x == 0) sh.z = total;
  const double Tz = want_total ? fmax(0.5, T * static_cast<double>(total)) : T;
  const unsigned long long a0 = above + ex;
  if (h > 0 && static_cast<double>(a0) < Tz && static_cast<double>(a0 + h) >= Tz) {
    sh.sel = j;
    sh.sel_above = a0;
    sh.sel_mass = h;
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kThreads, 1) topp_kernel(const __grid_constant__ Args a) {
  cg::cluster_group cl = cg::this_cluster();
  const int C = static_cast<int>(cl.num_blocks());
  const int rank = static_cast<int>(cl.block_rank());
  const int row = blockIdx.x / C;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Shared& sh = *reinterpret_cast<Shared*>(smem_raw);
  float* xs = reinterpret_cast<float*>(smem_raw + ((sizeof(Shared) + 15) & ~static_cast<size_t>(15)));
  const float p = a.p[row];
  const int lo = rank * a.slice;
  const int n = max(0, min(a.slice, a.V - lo));
  const int n_iter = (n + kThreads - 1) / kThreads;  // warp-uniform loop trip count
  float* g = a.logits + static_cast<long long>(row) * a.ld + lo;
  const bool fused = a.src != nullptr;  // write every output element (the copy) instead of masking in place
  const bool vec = (reinterpret_cast<uintptr_t>(g) & 15u) == 0;
  // fused mode, nothing to mask (p >= 1): the copy only
  auto copy_out = [&]() {
    for (int i = threadIdx.x; i < n; i += kThreads) g[i] = xs[i];
  };
  uint32_t* wh = &sh.whist[warp % kCopies][0][0];
  TMARK(0);

  // 1. slice -> shared memory, row max; clear this warp's histogram
  float m = -__int_as_float(0x7f800000);
  if (fused) {
    if (a.src_bf16) {
      const unsigned short* sp =
          reinterpret_cast<const unsigned short*>(a.src) + static_cast<long long>(row) * a.src_ld + lo;
      const bool v8 = (reinterpret_cast<uintptr_t>(sp) & 15u) == 0;
      int done = 0;
      if (v8) {
        for (int i = 8 * threadIdx.x; i + 7 < n; i += 8 * kThreads) {
          const uint4 u = *reinterpret_cast<const uint4*>(sp + i);
          const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float x0 = __uint_as_float(w[q] << 16), x1 = __uint_as_float(w[q] & 0xFFFF0000u);
            xs[i + 2 * q] = x0, xs[i + 2 * q + 1] = x1;
            m = fmaxf(m, fmaxf(x0, x1));
          }
        }
        done = n & ~7;
      }
      for (int i = done + threadIdx.x; i < n; i += kThreads) {
        const float x = __uint_as_float(static_cast<uint32_t>(sp[i]) << 16);
        xs[i] = x, m = fmaxf(m, x);
      }
    } else {
      const float* sp = reinterpret_cast<const float*>(a.src) + static_cast<long long>(row) * a.src_ld + lo;
      for (int i = threadIdx.x; i < n; i += kThreads) xs[i] = sp[i], m = fmaxf(m, sp[i]);
    }
    if (!(p < 1.0f)) {  // uniform over the cluster
      __syncthreads();
      copy_out();
      return;
    }
  } else {
    if (!(p < 1.0f)) return;  // uniform over the cluster: nothing to do for this row
    if (vec) {
      for (int i = 4 * threadIdx.x; i + 3 < n; i += 4 * kThreads) {
        const float4 v = *reinterpret_cast<const float4*>(g + i);
        *reinterpret_cast<float4*>(xs + i) = v;
        m = fmaxf(m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
      }
      for (int i = (n & ~3) + threadIdx.x; i < n; i += kThreads) xs[i] = g[i], m = fmaxf(m, g[i]);
    } else {
      for (int i = threadIdx.x; i < n; i += kThreads) xs[i] = g[i], m = fmaxf(m, g[i]);
    }
  }
  for (int b = threadIdx.x; b < kCopies * 3 * kBuckets; b += kThreads) (&sh.whist[0][0][0])[b] = 0;
  m = warp_reduce(m, [](float x, float y) { return fmaxf(x, y); });
  if (lane == 0) sh.redf[warp] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    float v = sh.redf[0];
    for (int w = 1; w < kWarps; ++w) v = fmaxf(v, sh.redf[w]);
    sh.cta_max = v;
  }
  TMARK(1);
  cl.sync();
  TMARK(2);
  float M = -__int_as_float(0x7f800000);
#pragma unroll
  for (int r = 0; r < kMaxCluster; ++r)
    if (r < C) M = fmaxf(M, cl.map_shared_rank(&sh, r)->cta_max);
  if (!finite_or_inf_neg(M) || M == __int_as_float(0x7f800000)) {  // no finite maximum: leave the row
    cl.sync();
    if (fused) copy_out();
    return;
  }

  // 2. level 0: linear buckets of d = M - x
  for (int it = 0; it < n_iter; ++it) {
    const int i = it * kThreads + threadIdx.x;
    int b = -1;
    unsigned long long f = 0;
    if (i < n) {
      const float x = xs[i];
      if (finite_or_inf_neg(x)) {
        const float d = M - x;
        f = fmass(d);
        b = f ? lin_bucket(d) : -1;
      }
    }
    warp_hist_add(wh, b, f);
  }
  TMARK(3);
  merge_warp_hists(sh, 0);
  cl.sync();
  TMARK(4);
  select_bucket(cl, sh, C, rank, 0, 0ull, static_cast<double>(p), true, [](int r) { return r; });
  TMARK(5);
  const int b0 = sh.sel;
  const double T = fmax(0.5, static_cast<double>(p) * static_cast<double>(sh.z));
  unsigned long long above = sh.sel_above, region_mass = sh.sel_mass;
  // members of the boundary region: f > 0 and level-0 bucket b0
  auto member0 = [&](float x, float& d) {
    if (!finite_or_inf_neg(x)) return false;
    d = M - x;
    return lin_bucket(d) == b0 && fmass(d) > 0;
  };

  // 3. key range of the region, then 9-bit key levels (descending key = ascending rank)
  uint32_t kmin = 0xFFFFFFFFu, kmax = 0u;
  for (int i = threadIdx.x; i < n; i += kThreads) {
    float d;
    if (member0(xs[i], d)) {
      const uint32_t k = okey(xs[i]);
      kmin = min(kmin, k);
      kmax = max(kmax, k);
    }
  }
  kmin = warp_reduce(kmin, [](uint32_t x, uint32_t y) { return min(x, y); });
  kmax = warp_reduce(kmax, [](uint32_t x, uint32_t y) { return max(x, y); });
  if (lane == 0) sh.redk[0][warp] = kmin, sh.redk[1][warp] = kmax;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t a0 = sh.redk[0][0], a1 = sh.redk[1][0];
    for (int w = 1; w < kWarps; ++w) a0 = min(a0, sh.redk[0][w]), a1 = max(a1, sh.redk[1][w]);
    sh.cta_kmin = a0, sh.cta_kmax = a1;
  }
  cl.sync();
  kmin = 0xFFFFFFFFu, kmax = 0u;
#pragma unroll
  for (int r = 0; r < kMaxCluster; ++r) {
    if (r < C) {
      const Shared* o = cl.map_shared_rank(&sh, r);
      kmin = min(kmin, o->cta_kmin);
      kmax = max(kmax, o->cta_kmax);
    }
  }
  TMARK(6);
  const uint32_t range = kmax - kmin;
  const int nbits = range == 0 ? 0 : 32 - __clz(range);
  const int levels = (nbits + kBits - 1) / kBits;
  uint32_t prefix = 0;  // selected high bits of (key - kmin)
  int buf = 0;
  for (int l = 0; l < levels; ++l) {
    const int shift = kBits * (levels - 1 - l);
    buf ^= 1;
    for (int it = 0; it < n_iter; ++it) {
      const int i = it * kThreads + threadIdx.x;
      int b = -1;
      unsigned long long f = 0;
      float d;
      if (i < n && member0(xs[i], d)) {
        const uint32_t rel = okey(xs[i]) - kmin;
        if ((rel >> shift) >> kBits == prefix) {
          b = static_cast<int>((rel >> shift) & (kBuckets - 1));
          f = fmass(d);
        }
      }
      warp_hist_add(wh, b, f);
    }
    // hist[buf] was last read by the peers at level l - 2 (or not at all): they are all past the previous cluster.sync
    merge_warp_hists(sh, buf);
    cl.sync();
    select_bucket(cl, sh, C, rank, buf, above, T, false, [](int r) { return kBuckets - 1 - r; });
    above = sh.sel_above;
    region_mass = sh.sel_mass;
    prefix = (prefix << kBits) | static_cast<uint32_t>(sh.sel);
    TMARK(7 + l);
  }
  TMARK(12);
  const uint32_t kstar = kmin + prefix;
  const float xstar = okey_inv(kstar);
  const unsigned long long fstar = fmass(M - xstar);
  const unsigned long long cnt = region_mass / fstar;  // every duplicate of x* has the same mass
  const double need = ceil((T - static_cast<double>(above)) / static_cast<double>(fstar));
  const unsigned long long keep_dups =
      static_cast<unsigned long long>(fmin(static_cast<double>(cnt), fmax(1.0, need)));

  // 4./5. mask pass; the tie path ranks the duplicates of x* by index
  const float mv = a.mask_value;
  if (keep_dups >= cnt) {
    cl.sync();  // peers may still read this CTA's histograms
    if (vec) {
      for (int i = 4 * threadIdx.x; i + 3 < n; i += 4 * kThreads) {
        float4 v = *reinterpret_cast<const float4*>(xs + i);
        const bool k0 = okey(v.x) >= kstar, k1 = okey(v.y) >= kstar, k2 = okey(v.z) >= kstar, k3 = okey(v.w) >= kstar;
        if (fused || !(k0 && k1 && k2 && k3)) {
          v.x = k0 ? v.x : mv, v.y = k1 ? v.y : mv, v.z = k2 ? v.z : mv, v.w = k3 ? v.w : mv;
          *reinterpret_cast<float4*>(g + i) = v;
        }
      }
      for (int i = (n & ~3) + threadIdx.x; i < n; i += kThreads)
        if (fused || okey(xs[i]) < kstar) g[i] = okey(xs[i]) < kstar ? mv : xs[i];
    } else {
      for (int i = threadIdx.x; i < n; i += kThreads)
        if (fused || okey(xs[i]) < kstar) g[i] = okey(xs[i]) < kstar ? mv : xs[i];
    }
    TMARK(13);
    return;
  }
  // pass 1: duplicates per (iteration, warp) in index order (index = it * kThreads + warp * 32 + lane)
  for (int it = 0; it < n_iter; ++it) {
    const int i = it * kThreads + threadIdx.x;
    const unsigned bal = __ballot_sync(0xFFFFFFFFu, i < n && okey(xs[i]) == kstar);
    if (lane == 0) sh.dupc[it * kWarps + warp] = __popc(bal);
  }
  __syncthreads();
  // exclusive prefix over the (iteration, warp) counts: thread t owns entries 2t, 2t + 1 (n_iter * kWarps <= 1024)
  const int e0 = 2 * threadIdx.x, ne = n_iter * kWarps;
  const unsigned long long c0 = e0 < ne ? sh.dupc[e0] : 0, c1 = e0 + 1 < ne ? sh.dupc[e0 + 1] : 0;
  unsigned long long tot;
  const unsigned long long ex = block_excl_scan(c0 + c1, &tot, sh);
  if (e0 < ne) sh.dupc[e0] = static_cast<int>(ex);
  if (e0 + 1 < ne) sh.dupc[e0 + 1] = static_cast<int>(ex + c0);
  if (threadIdx.x == 0) sh.cta_dups = tot;
  cl.sync();  // every CTA's count is published (and no peer reads our histograms any more)
  unsigned long long before = 0;  // duplicates in lower-ranked CTAs
#pragma unroll
  for (int r = 0; r < kMaxCluster; ++r)
    if (r < rank) before += cl.map_shared_rank(&sh, r)->cta_dups;
  cl.sync();  // every CTA has read its peers' counts
  // pass 2: mask
  for (int it = 0; it < n_iter; ++it) {
    const int i = it * kThreads + threadIdx.x;
    const bool in = i < n;
    const uint32_t k = in ? okey(xs[i]) : 0u;
    const bool dup = in && k == kstar;
    const unsigned bal = __ballot_sync(0xFFFFFFFFu, dup);
    const unsigned long long my_rank =
        before + static_cast<unsigned long long>(sh.dupc[it * kWarps + warp]) + __popc(bal & ((1u << lane) - 1u));
    const bool keep = k > kstar || (dup && my_rank < keep_dups);
    if (in && (fused || !keep)) g[i] = keep ? xs[i] : mv;
  }
  TMARK(14);
}

}  // namespace

extern "C" int w6tp_abi() { return 3; }  // v5 kernel, same Args as v3 / v4
extern "C" int w6tp_args_size() { return static_cast<int>(sizeof(Args)); }
extern "C" int w6tp_max_cluster() { return kMaxCluster; }
extern "C" int w6tp_trace(long long* out) {  // copies the 32 marks (TP_TRACE builds; -1 otherwise)
#ifdef TP_TRACE
  return static_cast<int>(cudaMemcpyFromSymbol(out, g_trace, sizeof(g_trace)));
#else
  (void)out;
  return -1;
#endif
}

// How many clusters of C CTAs can be resident at once (the launcher picks the largest C with rows <= this).
extern "C" int w6tp_max_clusters(int C, int V) {
  if (C < 4 || C > kMaxCluster) return 0;
  const int slice = ((V + C - 1) / C + 3) & ~3;
  const size_t smem = ((sizeof(Shared) + 15) & ~static_cast<size_t>(15)) + static_cast<size_t>(slice) * sizeof(float);
  if (cudaFuncSetAttribute(topp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)) != cudaSuccess)
    return 0;
  if (C > 8 && cudaFuncSetAttribute(topp_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess)
    return 0;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(C * 256);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = C;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  int n = 0;
  if (cudaOccupancyMaxActiveClusters(&n, topp_kernel, &cfg) != cudaSuccess) return 0;
  return n;
}

// C CTAs per row (cluster size 4..16; > 8 needs the non-portable attribute), on `stream`.
extern "C" int w6tp_launch(const Args* a0, int C, void* stream) {
  if (C < 4 || C > kMaxCluster || a0->B < 1) return -1;  // C >= 4: a slice must fit in shared memory
  Args a = *a0;
  a.slice = ((a.V + C - 1) / C + 3) & ~3;
  if ((a.slice + kThreads - 1) / kThreads > kMaxIters) return -2;
  const size_t smem = ((sizeof(Shared) + 15) & ~static_cast<size_t>(15)) + static_cast<size_t>(a.slice) * sizeof(float);
  cudaError_t e = cudaFuncSetAttribute(topp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  if (e != cudaSuccess) return static_cast<int>(e);
  if (C > 8) {
    e = cudaFuncSetAttribute(topp_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (e != cudaSuccess) return static_cast<int>(e);
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(a.B * C);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = static_cast<cudaStream_t>(stream);
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = C;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  e = cudaLaunchKernelEx(&cfg, topp_kernel, a);
  return static_cast<int>(e);
}
