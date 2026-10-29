// Candidate selection for long rows (layer 20 at long contexts), exact, as three PDL-chained stages:
//   CandKeysStage   task (row, kKeyChunk blocks): block scores -> order-preserving uint32 keys in global scratch,
//                   a 4096-bin histogram of the top 12 key bits and its 64-bin (top 6 bits) coarse sum
//   CandSplitStage  task (row, chunk): the k-th largest key's 12-bit bin d from the coarse then the fine histograms
//                   (every chunk CTA derives the same d); the chunk's blocks above bin d -> ordered chunk-local ids,
//                   its blocks in bin d -> ordered (key low 20 bits << 12 | chunk-local id) entries
//   CandFinalStage  task row: copy the above-bin ids (fewer than k) to the output; the bin's entries (smem list) ->
//                   3 radix passes -> exact k-th key T and how many keys == T to take (ascending block order);
//                   append the bin's selected ids; -1 fill
// Same set as CandStage / torch.topk (tie rule, -inf excluded); the above-bin ids come first, each part ascending
// (the consumers' expansion sorts). Every pass over a row's 2 MB of logits / 256 KB of keys is spread over
// rows x chunks CTAs; the per-row stage touches only histogram sums, < k ids and the threshold bin's keys.
#pragma once
#include "w3/stage_cand.cuh"

namespace w3 {
namespace cand_long {
constexpr int kChunk = 4096;     // blocks per split chunk: a 12-bit chunk-local id packs with a key's low 20 bits
constexpr int kKeyChunk = 2048;  // blocks per keys-stage task (one 64 KB sweep of 16 float4 per thread)
constexpr int kBins = 4096;   // top 12 key bits
constexpr int kMaxChunks = 256;

__device__ __forceinline__ uint32_t lanemask_lt() { return (1u << lane_id()) - 1u; }
__device__ __forceinline__ int warp_sum(int v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ int nvalid_blocks(int end, int width, int bs) {
  return end > 0 ? min((end + bs - 1) / bs, (width + bs - 1) / bs) : 0;
}
// Exclusive scan over the 256 compute threads in thread order; ws = 9 ints of smem. Returns the prefix, sets *total.
__device__ __forceinline__ int block_excl_scan(int v, int* ws, int* total) {
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane_id() >= o) x += y;
  }
  if (lane_id() == 31) ws[warp_id()] = x;
  compute_sync();
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int w = 0; w < 8; ++w) {
      const int s = ws[w];
      ws[w] = acc;
      acc += s;
    }
    ws[8] = acc;
  }
  compute_sync();
  const int res = ws[warp_id()] + x - v;
  *total = ws[8];
  compute_sync();
  return res;
}
// counter[key] += 1 for every lane with pred, one atomic per distinct key in the warp (ties collapse)
__device__ __forceinline__ void warp_count(uint32_t* counter, bool pred, uint32_t key) {
  const uint32_t act = __ballot_sync(0xffffffffu, pred);
  if (pred) {
    const uint32_t grp = __match_any_sync(act, key);
    if (lane_id() == __ffs(grp) - 1) atomicAdd(&counter[key], static_cast<uint32_t>(__popc(grp)));
  }
}
}  // namespace cand_long

struct CandKeysStage {
  struct Params {
    const float* logits;
    long long ld;
    int width;
    const int* row_end;   // valid columns of row r: row_end[r / row_repeat]
    int row_repeat;
    int block_size;   // 8
    uint32_t* gkeys;  // [rows, gkeys_stride]
    int gkeys_stride;
    uint16_t* hist;   // [rows, nkc, kBins] counts (<= kKeyChunk)
    uint32_t* coarse; // [rows, nkc, 64]
    int nkc;          // keys-stage tasks per row
    int rows;
    int k;            // rows with <= k valid blocks (one task, k <= kKeyChunk) are selected here entirely
    int* out;
    int out_stride;
  };
  static constexpr int kTraceId = 4;
  static constexpr bool kHasWeights = false;
  static constexpr const char* kName = "cand_keys";
  __host__ __device__ static int tasks(const Params& p) { return p.rows * p.nkc; }
  __host__ __device__ static uint32_t smem_bytes(int, int) { return cand_long::kBins * 4 + 64; }
  __device__ static void produce(const Params&, int, Ring&) {}

  template <int N, class Sync>
  __device__ static void consume(const Params& p, int task, int iter, bool, Sync& sync, Ring&, uint8_t* smem, uint32_t) {
    using namespace cand_long;
    uint32_t* h = reinterpret_cast<uint32_t*>(smem);
    const int r = task / p.nkc, ch = task % p.nkc;
    const int t = threadIdx.x;
    if (iter == 0) sync.wait();
    const int end = p.row_end[r / p.row_repeat];
    const int nvalid = nvalid_blocks(end, p.width, p.block_size);
    const int b_lo = ch * kKeyChunk, b_hi = min(b_lo + kKeyChunk, nvalid);
    const bool take_all = nvalid <= p.k;  // every block with a score above -inf is a candidate
    if (b_lo >= b_hi && !(take_all && ch == 0)) return;  // past the valid prefix: nothing reads this task's output
    uint16_t* hrow = p.hist + (static_cast<long long>(r) * p.nkc + ch) * kBins;
    if (!take_all) {
      for (int i = t; i < kBins; i += kComputeThreads) h[i] = 0;
      compute_sync();
    }
    {
      const float* row = p.logits + static_cast<long long>(r) * p.ld;
      uint32_t* gk = p.gkeys + static_cast<long long>(r) * p.gkeys_stride;
      const int half = t & 1;
      constexpr int U = kKeyChunk / 128;
      const float ninf = __int_as_float(0xff800000);
      for (int b0 = b_lo; b0 < b_hi; b0 += 128 * U) {
        float4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const int b = b0 + u * 128 + (t >> 1);
          const int c = b * 8 + half * 4;
          v[u] = make_float4(ninf, ninf, ninf, ninf);
          if (b < b_hi) {
            if (c + 4 <= end && c + 4 <= p.width) {
              v[u] = *reinterpret_cast<const float4*>(row + c);
            } else {
              for (int i = 0; i < 4; ++i)
                if (c + i < end && c + i < p.width) (&v[u].x)[i] = row[c + i];
            }
          }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const int b = b0 + u * 128 + (t >> 1);
          float m = CandStage::nanmax(CandStage::nanmax(v[u].x, v[u].y), CandStage::nanmax(v[u].z, v[u].w));
          m = CandStage::nanmax(m, __shfl_xor_sync(0xffffffffu, m, 1));
          const bool own = half == 0 && b < b_hi;
          const uint32_t k = (b == CandStage::pinned_block(end, 8)) ? CandStage::fkey(__int_as_float(0x7f800000))
                                                                     : CandStage::fkey(m);
          if (own && take_all) {
            h[b] = k;  // b_lo == 0
          } else if (own) {
            gk[b] = k;
            atomicAdd(&h[k >> 20], 1u);
          }
        }
      }
    }
    compute_sync();
    if (take_all) {
      // the whole row: ordered compaction of the keys above -inf, thread t owns blocks [8t, 8t + 8); -1 tail
      const uint32_t kninf = CandStage::fkey(__int_as_float(0xff800000));
      const int n = b_hi;
      int cnt = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) cnt += 8 * t + j < n && h[8 * t + j] != kninf;
      int total;
      int pos = block_excl_scan(cnt, reinterpret_cast<int*>(h + kKeyChunk), &total);
      int* orow = p.out + static_cast<long long>(r) * p.out_stride;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (8 * t + j < n && h[8 * t + j] != kninf) orow[pos++] = 8 * t + j;
      for (int i = total + t; i < p.k; i += kComputeThreads) orow[i] = -1;
      compute_sync();
      return;
    }
    for (int i = t; i < kBins; i += kComputeThreads) hrow[i] = static_cast<uint16_t>(h[i]);
    {
      // coarse bin j = sum of fine bins [64j, 64j + 64): 4 threads per coarse bin, 16 bins each
      uint32_t v = 0;
#pragma unroll
      for (int i = 0; i < 16; ++i) v += h[t * 16 + i];
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      if ((t & 3) == 0) p.coarse[(static_cast<long long>(r) * p.nkc + ch) * 64 + (t >> 2)] = v;
    }
    compute_sync();
  }
};

struct CandSplitStage {
  struct Params {
    uint32_t* gkeys;         // keys in; each chunk's bin entries out, (key & 0xFFFFF) << 12 | chunk-local id
    int gkeys_stride;
    const uint16_t* hist;    // [rows, nkc, kBins]   (keys-stage tasks)
    const uint32_t* coarse;  // [rows, nkc, 64]
    int nkc;
    int nchunks;
    const int* row_end;   // valid columns of row r: row_end[r / row_repeat]
    int row_repeat;
    int width;
    int block_size;
    int k;
    uint16_t* above_ids;     // [rows, nchunks, k]      chunk-local ids of the blocks above bin d
    uint16_t* hist2;         // [rows, nchunks, 1024]   counts of key bits 19..10 over the chunk's bin-d blocks
    int* chunk_meta;         // [rows, nchunks, 2]      (above count, bin count)
    int* row_meta;           // [rows, 4]               (d, -, -, -)
    int rows;
  };
  static constexpr int kTraceId = 2;
  static constexpr bool kHasWeights = false;
  static constexpr const char* kName = "cand_split";
  static constexpr int kPerLane = cand_long::kChunk / kComputeThreads;  // 16 keys per lane, a warp owns 512 blocks
  __host__ __device__ static int tasks(const Params& p) { return p.rows * p.nchunks; }
  __host__ __device__ static uint32_t smem_bytes(int, int) { return (64 + 256) * 4 + 8 * 1024 * 4; }
  __device__ static void produce(const Params&, int, Ring&) {}

  // The k-th largest key's 12-bit bin, from the descending prefix of the coarse and then the fine bin sums.
  // 64 bins x nch keys-stage tasks per level: thread t sums bin t % 64 over every 4th task, 8 loads in flight.
  __device__ static uint32_t threshold_bin(const Params& p, int r, int nch, int* sc) {
    using namespace cand_long;
    const int t = threadIdx.x, j = t & 63, g = t >> 6;
    int* part = sc + 64;  // [4][64]
    const long long row0 = static_cast<long long>(r) * p.nkc;  // nch = the row's valid keys-stage tasks
    auto level_sum = [&](auto base, int stride) -> int {  // bin (63 - j) of the level, summed over tasks
      int v = 0;
      for (int c0 = g; c0 < nch; c0 += 32) {
        uint32_t x[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const int c = c0 + 4 * u;
          x[u] = c < nch ? __ldcg(base + static_cast<long long>(c) * stride) : 0u;
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) v += static_cast<int>(x[u]);
      }
      part[g * 64 + j] = v;
      compute_sync();
      const int s = t < 64 ? part[t] + part[64 + t] + part[128 + t] + part[192 + t] : 0;
      compute_sync();
      return s;
    };
    int v = level_sum(p.coarse + row0 * 64 + 63 - j, 64);
    int tot;
    int excl = block_excl_scan(v, sc + 2, &tot);
    if (v > 0 && excl < p.k && excl + v >= p.k) {
      sc[0] = 63 - t;
      sc[1] = p.k - excl;
    }
    compute_sync();
    const int c6 = sc[0], k6 = sc[1];
    compute_sync();
    v = level_sum(p.hist + row0 * kBins + c6 * 64 + 63 - j, kBins);
    excl = block_excl_scan(v, sc + 2, &tot);
    if (v > 0 && excl < k6 && excl + v >= k6) sc[0] = c6 * 64 + 63 - t;
    compute_sync();
    const uint32_t d = static_cast<uint32_t>(sc[0]);
    compute_sync();
    return d;
  }

  template <int N, class Sync>
  __device__ static void consume(const Params& p, int task, int iter, bool, Sync& sync, Ring&, uint8_t* smem, uint32_t) {
    using namespace cand_long;
    int* sc = reinterpret_cast<int*>(smem);  // [0..1] pick, [2..10] scan, [16..31] warp counts, [64..319] sums
    uint32_t* wh = reinterpret_cast<uint32_t*>(sc + 320);  // [8 warps][1024] bin-d digit counts
    const int r = task / p.nchunks, c = task % p.nchunks;
    if (iter == 0) sync.wait();
    const int nvalid = nvalid_blocks(p.row_end[r / p.row_repeat], p.width, p.block_size);
    const int nch = (nvalid + kChunk - 1) / kChunk;
    if (c >= nch || nvalid <= p.k) return;  // rows with <= k valid blocks were selected by CandKeysStage
    trace(kTraceId, 2);
    const uint32_t ninf = CandStage::fkey(__int_as_float(0xff800000));
    const int w = warp_id(), lane = lane_id();
    const uint32_t lt = lanemask_lt();
    const uint32_t* gk = p.gkeys + static_cast<long long>(r) * p.gkeys_stride;
    const int lbase = w * 32 * kPerLane;  // chunk-local first block of this warp
    const int base = c * kChunk + lbase;
    uint32_t kk[kPerLane];  // issued before the threshold search, which only reads histograms; consumed (ballots below)
                            // before the first compute_sync, so the chunk's key slots can then take its bin entries
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int b = base + i * 32 + lane;
      kk[i] = b < nvalid ? __ldcg(gk + b) : 0u;  // key 0 (a negative NaN) is never produced: below every bin
    }
    for (int i = threadIdx.x; i < 8 * 1024; i += kComputeThreads) wh[i] = 0;  // visible after threshold_bin's syncs
    const uint32_t d = threshold_bin(p, r, (nvalid + kKeyChunk - 1) / kKeyChunk, sc);
    trace(kTraceId, 3);
    int na = 0, nb = 0;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      na += __popc(__ballot_sync(0xffffffffu, (kk[i] >> 20) > d && kk[i] != 0u));
      nb += __popc(__ballot_sync(0xffffffffu, (kk[i] >> 20) == d && kk[i] != 0u));
    }
    if (lane == 0) {
      sc[16 + w] = na;
      sc[24 + w] = nb;
    }
    compute_sync();
    trace(kTraceId, 4);
    int ab = 0, bb = 0, at = 0, bt = 0;
    for (int v = 0; v < 8; ++v) {
      if (v < w) {
        ab += sc[16 + v];
        bb += sc[24 + v];
      }
      at += sc[16 + v];
      bt += sc[24 + v];
    }
    const long long cell = static_cast<long long>(r) * p.nchunks + c;
    uint16_t* aid = p.above_ids + cell * p.k;
    uint32_t* bent = p.gkeys + static_cast<long long>(r) * p.gkeys_stride + c * kChunk;  // over this chunk's keys
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const bool ia = (kk[i] >> 20) > d && kk[i] != 0u, ib = (kk[i] >> 20) == d && kk[i] != 0u;
      const uint32_t am = __ballot_sync(0xffffffffu, ia), bm = __ballot_sync(0xffffffffu, ib);
      const uint32_t local = static_cast<uint32_t>(lbase + i * 32 + lane);
      if (ia) aid[ab + __popc(am & lt)] = static_cast<uint16_t>(local);
      if (ib) {
        bent[bb + __popc(bm & lt)] = (kk[i] & 0xFFFFFu) << 12 | local;
        atomicAdd(&wh[w * 1024 + ((kk[i] >> 10) & 1023u)], 1u);
      }
      ab += __popc(am);
      bb += __popc(bm);
    }
    compute_sync();
    {
      const int j = 4 * threadIdx.x;  // digits [j, j + 4)
      uint32_t v[4] = {0, 0, 0, 0};
#pragma unroll
      for (int ww = 0; ww < 8; ++ww)
#pragma unroll
        for (int u = 0; u < 4; ++u) v[u] += wh[ww * 1024 + j + u];
      *reinterpret_cast<uint2*>(p.hist2 + cell * 1024 + j) = make_uint2(v[0] | v[1] << 16, v[2] | v[3] << 16);
    }
    if (threadIdx.x == 0) {
      p.chunk_meta[cell * 2] = at;
      p.chunk_meta[cell * 2 + 1] = bt;
      if (c == 0) p.row_meta[r * 4] = static_cast<int>(d);
    }
    trace(kTraceId, 5);
    compute_sync();
  }
};

struct CandFinalStage {
  static constexpr int kResCap = 12288;  // bin entries kept resident in smem (incl. 16 B padding per chunk)
  static constexpr int kSlots = kResCap / cand_long::kChunk;  // else streamed through this many chunk slots
  static constexpr int kAboveCap = 2048 + 8 * cand_long::kMaxChunks;
  struct Params {
    const uint16_t* above_ids;
    const uint16_t* hist2;  // [rows, nchunks, 1024] per-chunk counts of bin-d key bits 19..10
    const uint32_t* gkeys;  // bin entries, chunk c at row offset c * kChunk (written by CandSplitStage)
    int gkeys_stride;
    const int* chunk_meta;
    const int* row_meta;
    int nchunks;
    const int* row_end;   // valid columns of row r: row_end[r / row_repeat]
    int row_repeat;
    int width;
    int block_size;
    int k;                // <= 2048, multiple of 8
    int* out;
    int out_stride;
    int rows;
  };
  static constexpr int kTraceId = 3;
  static constexpr bool kHasWeights = false;
  static constexpr const char* kName = "cand_final";
  __host__ __device__ static int tasks(const Params& p) { return p.rows; }
  __host__ __device__ static uint32_t smem_bytes(int, int) {
    return kResCap * 4 + kAboveCap * 2 + 8 * 1024 * 4 + 4 * 256 * 4 + 64 * 4 + 4 * 8;
  }
  __device__ static void produce(const Params&, int, Ring&) {}

  // chunk holding flat entry i, given the chunks' exclusive bases (ascending; empty chunks repeat a base)
  __device__ static int chunk_of(const int* base, int nch, int i) {
    int lo = 0, hi = nch - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (base[mid] <= i) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  template <int N, class Sync>
  __device__ static void consume(const Params& p, int r, int iter, bool, Sync& sync, Ring&, uint8_t* smem, uint32_t) {
    using namespace cand_long;
    uint32_t* bres = reinterpret_cast<uint32_t*>(smem);              // per-chunk bin entries, 16 B aligned segments
    uint16_t* abuf = reinterpret_cast<uint16_t*>(bres + kResCap);    // per-chunk above ids, 16 B aligned segments
    uint32_t* hist = reinterpret_cast<uint32_t*>(abuf + kAboveCap);  // [8 warps][1024] digit counts
    int* abase = reinterpret_cast<int*>(hist + 8 * 1024);
    int* bbase = abase + 256;
    int* aoff = bbase + 256;
    int* boff = aoff + 256;
    int* sc = boff + 256;  // [0..1] pick, [2..10] scan
    uint64_t* bar = reinterpret_cast<uint64_t*>(sc + 64);  // [0] first copies, [1..kSlots] streaming slots
    const int t = threadIdx.x;
    if (t == 0) {
      for (int i = 0; i <= kSlots; ++i) mbar_init(bar + i, 1);
      fence_mbar_init();
    }
    if (iter == 0) sync.wait();
    trace(kTraceId, 2);
    const int nvalid = nvalid_blocks(p.row_end[r / p.row_repeat], p.width, p.block_size);
    const int nch = (nvalid + kChunk - 1) / kChunk;
    if (nvalid <= p.k) return;  // selected entirely by CandKeysStage
    const uint32_t ninf = CandStage::fkey(__int_as_float(0xff800000));
    const long long row0 = static_cast<long long>(r) * p.nchunks;
    const uint32_t* gent = p.gkeys + static_cast<long long>(r) * p.gkeys_stride;
    const int na = t < nch ? p.chunk_meta[(row0 + t) * 2] : 0;
    const int nb = t < nch ? p.chunk_meta[(row0 + t) * 2 + 1] : 0;
    int A, B, AP, BP;
    abase[t] = block_excl_scan(na, sc + 2, &A);
    bbase[t] = block_excl_scan(nb, sc + 2, &B);
    aoff[t] = block_excl_scan((na + 7) & ~7, sc + 2, &AP);
    boff[t] = block_excl_scan((nb + 3) & ~3, sc + 2, &BP);
    const bool need_bin = B > 0;  // always: bin d holds the k-th key
    const bool resident = need_bin && BP <= kResCap;
    // one round trip: every chunk's above ids (and bin entries when they fit) by bulk copies on one mbarrier
    const uint32_t bytes = AP * 2 + (resident ? BP * 4 : 0);
    if (t == 0 && bytes > 0) {
      fence_proxy_async_global();
      fence_proxy_async_smem();
      mbar_arrive_expect_tx(bar, bytes);
      const uint64_t pol = policy_evict_first();
      for (int c = 0; c < nch; ++c) {
        const int a = p.chunk_meta[(row0 + c) * 2], b = p.chunk_meta[(row0 + c) * 2 + 1];
        if (a > 0)
          bulk_load(abuf + aoff[c], p.above_ids + (row0 + c) * p.k, ((a + 7) & ~7) * 2, bar, pol);
        if (resident && b > 0)
          bulk_load(bres + boff[c], gent + c * kChunk, ((b + 3) & ~3) * 4, bar, pol);
      }
    }
    if (bytes > 0) mbar_wait(bar, 0);
    trace(kTraceId, 6);
    int* orow = p.out + static_cast<long long>(r) * p.out_stride;
    for (int i = t; i < A; i += kComputeThreads) {
      const int c = chunk_of(abase, nch, i);
      orow[i] = c * kChunk + abuf[aoff[c] + i - abase[c]];
    }
    trace(kTraceId, 3);
    int total = A;
    if (need_bin) {
      auto nb_of = [&](int c) -> int { return (c + 1 < nch ? bbase[c + 1] : B) - bbase[c]; };
      uint32_t ph = 0;  // phase bit per streaming slot
      // f(get, n): a set of n bin entries in ascending block order; get(i, id) -> entry i and its block id. Resident:
      // one set (all B entries); streamed: one set per chunk, each through one of kSlots bres slots (bulk copies).
      auto for_each_set = [&](auto&& f) {  // f returns true to stop early
        if (resident) {
          f([&](int i, int& c, int* id) -> uint32_t {  // c: running chunk hint, ascending i per thread
            while (c + 1 < nch && bbase[c + 1] <= i) ++c;
            const uint32_t e = bres[boff[c] + i - bbase[c]];
            *id = c * kChunk + static_cast<int>(e & 0xFFFu);
            return e;
          }, B);
          return;
        }
        int held[kSlots];
        int issued = 0, cn = 0;
        auto issue = [&]() {
          while (cn < nch && nb_of(cn) == 0) ++cn;
          if (cn >= nch) return;
          const int s = issued % kSlots;
          held[s] = cn;
          if (t == 0) {
            const uint32_t by = ((nb_of(cn) + 3) & ~3) * 4;
            fence_proxy_async_smem();  // the slot's generic-proxy reads (ordered by compute_sync) before the async write
            mbar_arrive_expect_tx(bar + 1 + s, by);
            bulk_load(bres + s * kChunk, gent + static_cast<long long>(cn) * kChunk, by, bar + 1 + s, policy_evict_first());
          }
          ++issued;
          ++cn;
        };
        for (int s = 0; s < kSlots; ++s) issue();
        bool stop = false;
        for (int used = 0; used < issued; ++used) {
          const int s = used % kSlots;
          mbar_wait(bar + 1 + s, (ph >> s) & 1u);  // also drains copies still in flight after a stop
          ph ^= 1u << s;
          if (stop) continue;
          const uint32_t* buf = bres + s * kChunk;
          const int cbase = held[s] * kChunk;
          stop = f([&](int i, int&, int* id) -> uint32_t {
            const uint32_t e = buf[i];
            *id = cbase + static_cast<int>(e & 0xFFFu);
            return e;
          }, nb_of(held[s]));
          compute_sync();  // slot s consumed by every thread before it is refilled
          if (!stop) issue();
        }
      };
      // thread t's 16 consecutive entries [i0, i0 + 16) of a 4096-entry tile
      auto load16 = [&](auto&& get, int i0, int n, uint32_t (&e)[16], int (&id)[16]) {
        int c = i0 < n ? chunk_of(bbase, nch, i0) : 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          e[j] = 0u;
          id[j] = 0;
          if (i0 + j < n) e[j] = get(i0 + j, c, &id[j]);
        }
      };
      trace(kTraceId, 7);
      // exact T inside bin d: key bits 19..10 from the chunks' digit counts, bits 9..0 from one pass over the entries
      const uint32_t d = static_cast<uint32_t>(p.row_meta[r * 4]);
      int krem = p.k - A;
      const int lo4 = 4 * (255 - t);  // thread t owns digits [lo4, lo4 + 4): thread order = descending digits
      auto pick = [&](const int (&hv)[4]) -> int {  // digit holding the krem-th entry; updates krem
        int s_t = hv[0] + hv[1] + hv[2] + hv[3], tot;
        const int excl = block_excl_scan(s_t, sc + 2, &tot);
        if (s_t > 0 && excl < krem && excl + s_t >= krem) {
          int acc = excl;
          bool found = false;
#pragma unroll
          for (int j = 3; j >= 0; --j) {
            if (!found && hv[j] > 0 && acc + hv[j] >= krem) {
              sc[0] = lo4 + j;
              sc[1] = krem - acc;
              found = true;
            }
            acc += hv[j];
          }
        }
        compute_sync();
        const int dg = sc[0];
        krem = sc[1];
        compute_sync();
        return dg;
      };
      uint32_t low, mask = 1023u << 10;
      {
        int hv[4] = {0, 0, 0, 0};
        for (int c0 = 0; c0 < nch; c0 += 8) {
          uint2 x[8];
#pragma unroll
          for (int u = 0; u < 8; ++u)
            x[u] = c0 + u < nch ? __ldcg(reinterpret_cast<const uint2*>(p.hist2 + (row0 + c0 + u) * 1024 + lo4))
                                : make_uint2(0, 0);
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            hv[0] += x[u].x & 0xFFFF;
            hv[1] += x[u].x >> 16;
            hv[2] += x[u].y & 0xFFFF;
            hv[3] += x[u].y >> 16;
          }
        }
        low = static_cast<uint32_t>(pick(hv)) << 10;
      }
      trace(kTraceId, 8);
      uint32_t* whist = hist + warp_id() * 1024;
      {
        for (int i = t; i < 8 * 1024; i += kComputeThreads) hist[i] = 0;
        compute_sync();
        for_each_set([&](auto&& get, int n) {
          for (int i0 = 16 * t; i0 - 16 * t < n; i0 += 16 * kComputeThreads) {
            uint32_t e[16];
            int id[16];
            load16(get, i0, n, e, id);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
              const uint32_t kl = e[j] >> 12;
              if (i0 + j < n && (kl & mask) == low) atomicAdd(&whist[kl & 1023u], 1u);
            }
          }
          return false;
        });
        compute_sync();
        int hv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int v = 0;
#pragma unroll
          for (int w = 0; w < 8; ++w) v += static_cast<int>(hist[w * 1024 + lo4 + j]);
          hv[j] = v;
        }
        low |= static_cast<uint32_t>(pick(hv));
      }
      const int gt_total = p.k - A - krem;  // bin entries > T; krem = entries == T to take
      if ((d << 20 | low) == ninf) krem = 0;  // fewer than k scores above -inf: -inf blocks are never taken
      trace(kTraceId, 4);
      // ordered selection: every entry > T, then the first krem == T (ascending block order); stops once both are met
      int gt_run = 0, eq_run = 0;
      for_each_set([&](auto&& get, int n) -> bool {
        for (int base = 0; base < n; base += 16 * kComputeThreads) {
          const int i0 = base + 16 * t;
          uint32_t e[16];
          int id[16];
          load16(get, i0, n, e, id);
          int g = 0, q = 0;
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            const bool ok = i0 + j < n;
            g += ok && (e[j] >> 12) > low;
            q += ok && (e[j] >> 12) == low;
          }
          int tot;
          const int pre = block_excl_scan(g | (q << 16), sc + 2, &tot);
          int gb = gt_run + (pre & 0xFFFF), eb = eq_run + (pre >> 16);
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            if (i0 + j >= n) continue;
            const uint32_t kl = e[j] >> 12;
            if (kl > low) {
              orow[A + gb + min(eb, krem)] = id[j];
              ++gb;
            } else if (kl == low) {
              if (eb < krem) orow[A + gb + eb] = id[j];
              ++eb;
            }
          }
          gt_run += tot & 0xFFFF;
          eq_run += tot >> 16;
          if (gt_run >= gt_total && eq_run >= krem) return true;  // uniform: block-scan totals
        }
        return false;
      });
      total = A + gt_run + min(eq_run, krem);
    }
    for (int i = total + t; i < p.k; i += kComputeThreads) orow[i] = -1;
    trace(kTraceId, 5);
    compute_sync();
  }
};

}  // namespace w3
