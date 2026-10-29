// W6 tail item 3: DSpark's sequential Markov sampling loop as ONE persistent kernel.
//
// Production (DSparkSpeculator._sample_sequential with W5's tail_markov) runs, per draft step s = 0..S-1:
//   bias = bf16(cuBLAS(embed_s @ markov_w2^T))                      [R, V]   (66 MB of w2 read per step)
//   logits = bf16(f32(base[:, s]) + f32(bias)); q_cache[req, s] = logits
//   token_s = argmax(logits / temp + Gumbel(seed, sample_pos - 1 + 2^30, token))   (plain argmax at temp 0)
//   embed_{s+1} = markov_w1[token_s]
// i.e. 2 launches per step. Here one cooperative grid (one CTA per SM) owns contiguous vocabulary slices: each CTA
// queues its whole w2 slice for L2 at start (bulk prefetch), streams it into tensor-core fragments at step 0 and keeps
// the first res_rows in shared memory for the later steps, which read the rest from L2. Per step, every request of the
// batch at once: the bias GEMV runs on the tensor cores (mma.sync m16n8k16, bf16 x bf16 -> fp32; the K dimension is
// permuted identically in w2 and the embeddings so every lane loads contiguous 16 B pieces), then each row's logits
// and q-cache entries exactly as production rounds them,
// production's Gumbel noise bit for bit (murmur3 + the same fp32 operation sequence Triton emits), its slice's best
// (packed 64-bit key: order-preserving value, lowest token on ties, NaN on top), one atomicMax per request, then a
// grid barrier; every CTA decodes the winners and loads the next embeddings itself.
// The only numerical change vs production is the bias GEMV's fp32 accumulation order (cuBLAS's is not reproduced),
// which can move a bias by one bf16 ulp. The kernel writes exactly the logits it samples from into the q cache, so the
// rejection sampler's acceptance test stays exact: a draft-only change (acceptance only).
//   nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -Xcompiler -fPIC -shared w6mk.cu -o libw6mk.so
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace w6mk {
struct Args {
  const __nv_bfloat16* base;  // [R, S, V] base logits
  long long base_s0, base_s1;
  const int* input_ids;        // drafter input ids
  const long long* anchor_idx; // [R]
  const int* idx_mapping;      // [R * S] req state per (req, step), -1 = inert
  const long long* sample_pos; // [R * S]
  const float* temp;           // [max_reqs]
  const long long* seeds;      // [max_reqs]
  __nv_bfloat16* cache;        // q cache [max_reqs, K, V]
  long long cache_s0, cache_s1;
  long long* draft_tokens;     // [max_reqs, K]
  long long dt_s0;
  const __nv_bfloat16* w1;     // markov_w1 [V, kRank]
  const __nv_bfloat16* w2;     // markov_w2 [V, kRank]
  unsigned long long* slots;   // [R * S] packed winners, zero between launches
  unsigned int* counters;      // [S + 1] step barriers + exit ticket, zero between launches
  long long* trace;            // optional globaltimer marks (tests): [0..63] CTA 0, [64 + cta] step-0 pass end
  int R, S, V, rows_per_cta, res_rows;
  int dbg;                     // tests: bit 0 no streamed chunks, bit 1 no resident rows, bit 2 no producer copies
  const int* n_active;         // optional device int: run only steps 0..n-1 (clamped to 1..S); null = all S
};
}  // namespace w6mk
using w6mk::Args;

namespace {

constexpr int kRank = 256;       // dspark_markov_rank: one row = 512 B = 32 lanes x 16 B
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kTile = 16;        // rows per MMA tile (m16n8k16: 16 w2 rows x 8 requests, K = 256 in 16 k-blocks)
constexpr int kResStride = 36;   // resident rows padded to 576 B: a fragment load (8 rows x 64 B) is bank-conflict free
constexpr int kMaxReqs = 8;      // requests per launch handled by one kernel instance
constexpr int kMaxSteps = 8;
#ifndef FM_STEP
#define FM_STEP -1  // tests: -DFM_STEP=s records per-warp marks of step s (trace[256 + 12 warp ..]) and prologue marks
#endif

// ---------------------------------------------------------------- production's draft Gumbel draw, bit for bit
__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint32_t rotl32(uint32_t v, int s) { return (v << s) | (v >> (32 - s)); }
__device__ __forceinline__ uint32_t mm3_mix(uint32_t h, uint32_t key) {
  key *= 0xCC9E2D51u;
  key = rotl32(key, 15);
  key *= 0x1B873593u;
  h ^= key;
  h = rotl32(h, 13);
  return h * 5u + 0xE6546B64u;
}
__device__ __forceinline__ uint32_t mm3_fmix(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  return h ^ (h >> 16);
}
// gumbel.murmur3_hash32(seed, pos, offset, domain=0)
__device__ __forceinline__ uint32_t mm3_hash(long long seed, long long pos, uint32_t offset) {
  uint32_t h = 0;
  h = mm3_mix(h, static_cast<uint32_t>(seed & 0xFFFFFFFFll));
  h = mm3_mix(h, static_cast<uint32_t>((seed >> 32) & 0xFFFFFFFFll));
  h = mm3_mix(h, static_cast<uint32_t>(pos & 0xFFFFFFFFll));
  h = mm3_mix(h, offset);
  return mm3_fmix(h ^ 16u);
}
__device__ __forceinline__ float fdiv_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// gumbel.murmur3_uniform32: hi16 * 2^-16 + (lo16 + 0.5) * 2^-32, as Triton emits it (add, mul, fma)
__device__ __forceinline__ float mm3_uniform(long long seed, long long pos, uint32_t token) {
  const uint32_t r = mm3_hash(seed, pos, token);
  const float hi = static_cast<float>(static_cast<int>(r >> 16));
  const float lo = static_cast<float>(static_cast<int>(r & 0xFFFFu));
  return __fmaf_rn(hi, 1.52587890625e-05f, __fmul_rn(__fadd_rn(lo, 0.5f), 2.3283064365386963e-10f));
}
// gumbel._log1p_neg_stable(u) = log(1 - u)
__device__ __forceinline__ float log1p_neg_stable(float v) {
  float p = 0.125f;
  p = __fmaf_rn(v, p, 0.142857149f);  // 1/7
  p = __fmaf_rn(v, p, 0.166666672f);  // 1/6
  p = __fmaf_rn(v, p, 0.200000003f);  // 1/5
  p = __fmaf_rn(v, p, 0.25f);
  p = __fmaf_rn(v, p, 0.333333343f);  // 1/3
  p = __fmaf_rn(v, p, 0.5f);
  p = __fmaf_rn(v, p, 1.0f);
  const float series = __fmul_rn(-v, p);
  const float direct = logf(fmaxf(__fsub_rn(1.0f, v), 5.960464477539063e-08f));
  return v < 0.25f ? series : direct;
}
// value gumbel_noised_argmax maximizes for one token (IS_DRAFTING, APPLY_TEMPERATURE, fp32)
__device__ __forceinline__ float noised(float logit, float temp, long long seed, long long pos, uint32_t token) {
  if (temp == 0.0f) return logit;
  const float x = fdiv_full(logit, temp);
  const float u = mm3_uniform(seed, pos + (1ll << 30), token);
  const float g = -logf(-log1p_neg_stable(u));
  return __fadd_rn(x, g);
}
// W5's packed winner key: high 32 = order-preserving value (NaN on top, -0 == +0), low 31 = 2^31 - 1 - token
__device__ __forceinline__ unsigned long long pack(float v, int token) {
  uint32_t b = __float_as_uint(v == 0.0f ? 0.0f : v);
  uint32_t o = (b >> 31) ? ~b : (b | 0x80000000u);
  if (v != v) o = 0xFFFFFFFFu;
  return (static_cast<unsigned long long>(o) << 31) | static_cast<unsigned long long>(2147483647 - token);
}
__device__ __forceinline__ int unpack_token(unsigned long long k) { return 2147483647 - static_cast<int>(k & 2147483647ull); }

// ---------------------------------------------------------------- tensor-core GEMV
// m16n8k16 fragment ownership (lane = 4 g + t): A rows g, g + 8, logical k columns {2t, 2t+1, 2t+8, 2t+9} per k-block;
// B column g (= request), same k columns; C rows g, g + 8, columns 2t, 2t + 1. Logical (k-block kb, column c) maps to
// the physical k = 8 (4 (kb >> 1) + t) + 4 (kb & 1) + {0, 1} for c in {2t, 2t+1} and + {2, 3} for c in {2t+8, 2t+9}:
// at piece i lane t reads the 16 B chunk 4 i + t of a row, so the 4 lanes of a row read 64 contiguous bytes.
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// One 16-row tile: lo = row g's 128 B piece (8 x 16 B), hi = row g + 8's; e = this lane's embedding piece for
// request g (zeros past the batch). Returns C fragments (dots for rows g, g + 8 x requests 2t, 2t + 1).
__device__ __forceinline__ void tile_mma(float (&c)[4], const uint4 (&lo)[8], const uint4 (&hi)[8],
                                         const uint4 (&e)[8]) {
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // two independent accumulator chains, summed at the end
  c[0] = c[1] = c[2] = c[3] = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {  // k-blocks 2i, 2i + 1 live in the i-th 16 B piece
    mma16816(c, lo[i].x, hi[i].x, lo[i].y, hi[i].y, e[i].x, e[i].y);
    mma16816(d, lo[i].z, hi[i].z, lo[i].w, hi[i].w, e[i].z, e[i].w);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) c[j] = __fadd_rn(c[j], d[j]);
}
// Resident tile straight from shared memory (rows padded to kResStride uint4): 2 x 16 B per piece, no register tile.
__device__ __forceinline__ void tile_mma_smem(float (&c)[4], const uint4* res, int r0, int g, int t,
                                              const uint4 (&e)[8]) {
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  c[0] = c[1] = c[2] = c[3] = 0.0f;
  const uint4* rl = res + (r0 + g) * 36 + t;
  const uint4* rh = res + (r0 + g + 8) * 36 + t;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint4 lo = rl[4 * i], hi = rh[4 * i];
    mma16816(c, lo.x, hi.x, lo.y, hi.y, e[i].x, e[i].y);
    mma16816(d, lo.z, hi.z, lo.w, hi.w, e[i].z, e[i].w);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) c[j] = __fadd_rn(c[j], d[j]);
}
__device__ __forceinline__ void load_tile(uint4 (&lo)[8], uint4 (&hi)[8], const uint4* w2, int r0, int nrows, int g,
                                          int t) {
  const bool vlo = r0 + g < nrows, vhi = r0 + g + 8 < nrows;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    lo[i] = vlo ? __ldcg(w2 + static_cast<size_t>(r0 + g) * 32 + 4 * i + t) : make_uint4(0, 0, 0, 0);
    hi[i] = vhi ? __ldcg(w2 + static_cast<size_t>(r0 + g + 8) * 32 + 4 * i + t) : make_uint4(0, 0, 0, 0);
  }
}
// Keep a register tile resident for the later steps (same padded layout tile_mma_smem reads).
__device__ __forceinline__ void store_tile(uint4* res, const uint4 (&lo)[8], const uint4 (&hi)[8], int r0, int g,
                                           int t) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    res[(r0 + g) * kResStride + 4 * i + t] = lo[i];
    res[(r0 + g + 8) * kResStride + 4 * i + t] = hi[i];
  }
}

// Row tail for one (row, request): bias, logits, q-cache entry, noised value -> packed key.
__device__ __forceinline__ unsigned long long row_key(float dot, __nv_bfloat16 b, int token, __nv_bfloat16* cache_row,
                                                   float temp, long long seed, long long pos) {
  const __nv_bfloat16 bias = __float2bfloat16_rn(dot);
  const __nv_bfloat16 logit = __float2bfloat16_rn(__fadd_rn(__bfloat162float(b), __bfloat162float(bias)));
  if (cache_row != nullptr) cache_row[token] = logit;
  return pack(noised(__bfloat162float(logit), temp, seed, pos, token), token);
}

__device__ __forceinline__ unsigned long long warp_max64(unsigned long long k) {
#pragma unroll
  for (int s = 16; s >= 1; s >>= 1) {
    const unsigned long long o = __shfl_xor_sync(0xFFFFFFFFu, k, s);
    k = o > k ? o : k;
  }
  return k;
}
__device__ __forceinline__ uint32_t sptr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__global__ void __launch_bounds__(kThreads, 1) markov_kernel(const __grid_constant__ Args a) {
  extern __shared__ __align__(128) uint4 res[];  // [res_rows][kResStride]: resident w2 rows (64 B of padding each)
  __shared__ uint4 e_s[kMaxReqs][32];  // this step's markov_w1 rows of every request
  __shared__ unsigned long long cta_best[kWarps][kMaxReqs];
  __shared__ int tokens[kMaxReqs];
  __shared__ int p_st[kMaxReqs * kMaxSteps];
  __shared__ float p_temp[kMaxReqs * kMaxSteps];
  __shared__ long long p_seed[kMaxReqs * kMaxSteps], p_pos[kMaxReqs * kMaxSteps];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int G = gridDim.x;
  const int row_lo = blockIdx.x * a.rows_per_cta;
  const int nrows = max(0, min(a.V - row_lo, a.rows_per_cta));
  const int n_tiles = (nrows + kTile - 1) / kTile;
  const uint4* w2 = reinterpret_cast<const uint4*>(a.w2) + static_cast<size_t>(row_lo) * 32;
  int mark = 0;
  auto MARK = [&]() {
    if (a.trace != nullptr && blockIdx.x == 0 && threadIdx.x == 0) a.trace[mark] = gtimer();
    ++mark;
  };
  MARK();
  auto PM = [&](int i) {
    if (FM_STEP >= 0 && a.trace != nullptr && blockIdx.x == 0 && threadIdx.x == 0) a.trace[40 + i] = gtimer();
  };
  // warp 1 queues this CTA's whole w2 slice for L2 (HBM streams it while step 0 consumes it in the same order);
  // warp 0 reads the per-request parameters and anchor tokens.
  if (warp == 1 && lane < 16) {
    const int chunk_rows = (nrows + 15) / 16;
    const int r_lo = lane * chunk_rows, r_hi = min(nrows, r_lo + chunk_rows);
    if (r_lo < r_hi)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(w2 + static_cast<size_t>(r_lo) * 32),
                   "r"(static_cast<uint32_t>(r_hi - r_lo) * 512u)
                   : "memory");
  }
  if (warp == 0) {
    for (int i = lane; i < a.R * a.S; i += 32) {
      const int st = a.idx_mapping[i];
      p_st[i] = st;
      p_temp[i] = st >= 0 ? a.temp[st] : 0.0f;
      p_seed[i] = st >= 0 ? a.seeds[st] : 0;
      p_pos[i] = a.sample_pos[i] - 1;
    }
    if (lane < a.R) tokens[lane] = a.input_ids[a.anchor_idx[lane]];
  }
  PM(0);
  __syncthreads();
  PM(1);
  int fm = 0;
  auto FM = [&](int s_) {
    if (a.trace != nullptr && blockIdx.x == 0 && lane == 0 && s_ == FM_STEP && fm < 12) a.trace[256 + warp * 12 + fm] = gtimer();
    if (s_ == FM_STEP) ++fm;
  };

  // Early exit (ABI 4): the next verify uses only the first n drafts, so steps n..S-1 are skipped; their draft
  // tokens and q-cache rows are left as they were (the caller guarantees nothing reads them).
  const int s_run = a.n_active != nullptr ? max(1, min(a.S, __ldcg(a.n_active))) : a.S;
  for (int s = 0; s < s_run; ++s) {
    FM(s);
    // this step's embeddings: markov_w1 rows of the requests' previous tokens (the anchor at step 0)
    for (int i = threadIdx.x; i < a.R * 32; i += kThreads)
      e_s[i >> 5][i & 31] = __ldcg(reinterpret_cast<const uint4*>(a.w1 + static_cast<size_t>(tokens[i >> 5]) * kRank) +
                                   (i & 31));
    __syncthreads();
    FM(s);  // embeddings in shared memory
    // tiles of this warp: slot j = tile warp + 8 j. Slots 0..kResSlots-1 are resident (shared memory, read per
    // piece), the rest are streamed from L2 (HBM at step 0) two tiles ahead in registers. At step 0 the streamed
    // tiles go first (the resident copy is still landing); afterwards the resident ones run while the first two
    // streamed tiles are in flight.
    constexpr int kSlots = 8, kResSlots = 3;  // res_rows == kResSlots * kWarps * kTile (384)
    auto tile_valid = [&](int j) { return warp + kWarps * j < n_tiles; };
    auto tile_r0 = [&](int j) { return (warp + kWarps * j) * kTile; };
    constexpr int kPairs = 4;
    const int r16 = lane & 15, half = lane >> 4;
    __nv_bfloat16 bvs[kPairs][kMaxReqs];
#pragma unroll
    for (int p = 0; p < kPairs; ++p) {
      const int j = 2 * p + half;
      const int row = tile_valid(j) ? tile_r0(j) + r16 : nrows;
#pragma unroll
      for (int m = 0; m < kMaxReqs; ++m)
        bvs[p][m] = (m < a.R && row < nrows) ? a.base[m * a.base_s0 + s * a.base_s1 + row_lo + row]
                                             : __float2bfloat16_rn(0.0f);
    }
    unsigned long long best[kMaxReqs];
#pragma unroll
    for (int m = 0; m < kMaxReqs; ++m) best[m] = 0;
    uint4 e[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) e[i] = g < a.R ? e_s[g][4 * i + t] : make_uint4(0, 0, 0, 0);
    float cs[kSlots][4];
#pragma unroll
    for (int j = 0; j < kSlots; ++j) cs[j][0] = cs[j][1] = cs[j][2] = cs[j][3] = 0.0f;
    // Streams tiles first..kSlots-1 through registers two tiles ahead; at step 0 (first == 0) the resident slots are
    // among them and are kept in shared memory for the later steps.
    uint4 loX[8], hiX[8], loY[8], hiY[8];
    auto streamed = [&](auto first_c) {
      constexpr int first = decltype(first_c)::value;
      if (tile_valid(first)) load_tile(loX, hiX, w2, tile_r0(first), nrows, g, t);
      if (tile_valid(first + 1)) load_tile(loY, hiY, w2, tile_r0(first + 1), nrows, g, t);
      if (first > 0) {  // the resident tiles run while the first two streamed tiles are in flight
#pragma unroll
        for (int j = 0; j < kResSlots; ++j)
          if (tile_valid(j)) tile_mma_smem(cs[j], res, tile_r0(j), g, t, e);
      }
#pragma unroll
      for (int j = first; j < kSlots; ++j) {
        if (!tile_valid(j)) break;
        if (((j - first) & 1) == 0) {
          tile_mma(cs[j], loX, hiX, e);
          if (j < kResSlots) store_tile(res, loX, hiX, tile_r0(j), g, t);
          if (j + 2 < kSlots && tile_valid(j + 2)) load_tile(loX, hiX, w2, tile_r0(j + 2), nrows, g, t);
        } else {
          tile_mma(cs[j], loY, hiY, e);
          if (j < kResSlots) store_tile(res, loY, hiY, tile_r0(j), g, t);
          if (j + 2 < kSlots && tile_valid(j + 2)) load_tile(loY, hiY, w2, tile_r0(j + 2), nrows, g, t);
        }
      }
    };
    if (s == 0)
      streamed(std::integral_constant<int, 0>{});
    else
      streamed(std::integral_constant<int, kResSlots>{});
    FM(s);  // all MMAs done
    // phase 2: row tails, 4 independent rows per lane (row r16 of the tiles of its half), every request
    const int src = (r16 & 7) * 4;
#pragma unroll
    for (int m = 0; m < kMaxReqs; ++m) {
      if (m < a.R) {
        const int st = p_st[m * a.S + s];
        __nv_bfloat16* cache_row = st >= 0 ? a.cache + st * a.cache_s0 + s * a.cache_s1 : nullptr;
        const float temp = p_temp[m * a.S + s];
        const long long seed = p_seed[m * a.S + s], pos = p_pos[m * a.S + s];
        const int sl = src + (m >> 1), j = m & 1;
#pragma unroll
        for (int p = 0; p < kPairs; ++p) {
          const float a0 = __shfl_sync(0xFFFFFFFFu, cs[2 * p][j], sl), a2 = __shfl_sync(0xFFFFFFFFu, cs[2 * p][2 + j], sl);
          const float b0 = __shfl_sync(0xFFFFFFFFu, cs[2 * p + 1][j], sl);
          const float b2 = __shfl_sync(0xFFFFFFFFu, cs[2 * p + 1][2 + j], sl);
          const float dot = half == 0 ? (r16 < 8 ? a0 : a2) : (r16 < 8 ? b0 : b2);
          const int row = tile_valid(2 * p + half) ? tile_r0(2 * p + half) + r16 : nrows;
          if (row < nrows) {
            const unsigned long long key = row_key(dot, bvs[p][m], row_lo + row, cache_row, temp, seed, pos);
            best[m] = key > best[m] ? key : best[m];
          }
        }
      }
    }
    FM(s);  // tails done
    // per request: warp max -> CTA max -> one atomicMax
#pragma unroll
    for (int m = 0; m < kMaxReqs; ++m) {
      if (m < a.R) {
        unsigned long long kk = best[m];
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
          const unsigned long long other = __shfl_xor_sync(0xFFFFFFFFu, kk, o);
          kk = other > kk ? other : kk;
        }
        if (lane == 0) cta_best[warp][m] = kk;
      }
    }
    __syncthreads();
    MARK();
    if (s == 0 && a.trace != nullptr && threadIdx.x == 0) a.trace[64 + blockIdx.x] = gtimer();  // step-0 pass end
    if (threadIdx.x < a.R) {
      unsigned long long k = cta_best[0][threadIdx.x];
#pragma unroll
      for (int w = 1; w < kWarps; ++w) k = cta_best[w][threadIdx.x] > k ? cta_best[w][threadIdx.x] : k;
      atomicMax(a.slots + threadIdx.x * a.S + s, k);
    }
    __syncthreads();
    // grid barrier for step s, then every CTA decodes the winners itself
    if (threadIdx.x == 0) {
      __threadfence();
      atomicAdd(a.counters + s, 1u);
      while (*reinterpret_cast<volatile unsigned int*>(a.counters + s) < static_cast<unsigned int>(G)) {
      }
      __threadfence();
    }
    __syncthreads();
    MARK();
    if (threadIdx.x < a.R) {
      const int tok = unpack_token(__ldcg(a.slots + threadIdx.x * a.S + s));
      tokens[threadIdx.x] = tok;
      if (blockIdx.x == 0) a.draft_tokens[threadIdx.x * a.dt_s0 + s] = tok;
    }
    __syncthreads();
  }
  // exit ticket: every CTA has passed every barrier; the last one resets the slots and counters
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(a.counters + a.S, 1u) == static_cast<unsigned int>(G - 1)) {
      for (int i = 0; i < a.R * a.S; ++i) a.slots[i] = 0;
      for (int i = 0; i <= a.S; ++i) a.counters[i] = 0;
      __threadfence();
    }
  }
}

}  // namespace

extern "C" int w6mk_abi() { return 4; }
extern "C" int w6mk_args_size() { return static_cast<int>(sizeof(Args)); }
extern "C" int w6mk_max_reqs() { return kMaxReqs; }
extern "C" int w6mk_tile() { return kTile; }
extern "C" int w6mk_res_row_bytes() { return kResStride * 16; }

// CTAs that fit on one SM with `smem` dynamic bytes (the launch needs 1 per SM, all resident).
extern "C" int w6mk_ctas_per_sm(int smem) {
  int n = 0;
  cudaFuncSetAttribute(markov_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  const cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, markov_kernel, kThreads, smem);
  return e == cudaSuccess ? n : -static_cast<int>(e);
}

// Always a cooperative launch: the per-step grid barriers need every CTA co-resident (grid <= SMs x ctas_per_sm), and
// the waits have no timeout (a preempted context would count its descheduled time against one).
extern "C" int w6mk_launch(const Args* a, int grid, int smem, void* stream) {
  if (a->R < 1 || a->R > kMaxReqs || a->S < 1 || a->S > kMaxSteps || a->res_rows % kTile != 0) return -1;
  if ((a->rows_per_cta + kTile - 1) / kTile > 8 * kWarps) return -3;  // tile slots per warp (4 pairs)
  if (a->res_rows != 3 * kWarps * kTile) return -4;                   // resident slots 0..2 of every warp
  cudaError_t e = cudaFuncSetAttribute(markov_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return static_cast<int>(e);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = static_cast<size_t>(smem);
  cfg.stream = static_cast<cudaStream_t>(stream);
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeCooperative;
  at[0].val.cooperative = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  e = cudaLaunchKernelEx(&cfg, markov_kernel, *a);
  return static_cast<int>(e);
}
