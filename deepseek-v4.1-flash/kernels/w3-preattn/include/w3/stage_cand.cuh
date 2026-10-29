// Stage cand: the candidate source's (layer 20) block selection, exact. Replaces select_candidate_blocks
// (_block_scores_kernel + torch.topk over the whole max_model_len row + _store_candidates_kernel, ~70 us of
// radix-select/sort launches at decode) with one launch that scans only each row's valid prefix.
//
// Semantics reproduced: block score = NaN-propagating max of the block's valid fp32 logits (cols < end), the row's
// newest block ((end - 1) / block_size, if inside the width) pinned to +inf, blocks past the valid prefix -inf; keep the top-k blocks with torch.topk's tie rule
// (all scores above the k-th, then equal-to-k-th scores in ascending block order); -inf scores become -1.
// The output row is the same SET as production's (production stores it in descending-score order; every consumer
// expands and sorts it, see candidate_blocks_to_sparse_indices), written in ascending block order, -1 padded.
// Optionally also emits the consumers' DeepGEMM sparse block ids + valid-column counts (what the first candidate
// consumer's _expand_candidates_kernel computes), so consumers can skip that launch.
// Task = row; 256 threads; exact radix select (4 x 8-bit digits) over order-preserving uint32 keys.
#pragma once
#include "w3/fw.cuh"

namespace w3 {

struct CandStage {
  struct Params {
    const float* logits;  // [rows, ld]
    long long ld;
    int width;            // logits columns (max_model_len // ratio)
    const int* row_end;   // valid columns of row r: row_end[r / row_repeat] (decode rows of a request share it)
    int row_repeat;
    int block_size;       // 8
    int k;                // 2048
    int* out;             // [rows, out_stride] candidate block ids, ascending, -1 padded
    int out_stride;
    // optional expansion for ratio-1 candidate consumers (sparse_block_kv == block_size)
    int* sparse_idx;      // [rows, si_stride] or null
    int si_stride;
    int* sparse_end;      // [rows]
    const int* cons_ke;   // [rows] consumer K range end (compressed context length)
    int rows;
    int cache_keys;       // smem key cache capacity (blocks)
    uint32_t* gkeys;      // [rows, gkeys_stride] global key scratch for rows past the smem cache (or null: recompute)
    int gkeys_stride;
  };
  static constexpr int kTraceId = 1;
  static constexpr bool kHasWeights = false;
  static constexpr const char* kName = "cand";
  __host__ __device__ static int tasks(const Params& p) { return p.rows; }
  __host__ __device__ static uint32_t smem_bytes(int cache_keys, int) { return cache_keys * 4 + 256 * 4 + 512 * 4 + 64; }
  __device__ static void produce(const Params&, int, Ring&) {}

  __device__ static uint32_t fkey(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  }
  __device__ static float nanmax(float a, float b) { return (a != a || b != b) ? __int_as_float(0x7fffffff) : fmaxf(a, b); }

  // the newest block (holding column end - 1) is pinned to +inf, as in _block_scores_kernel; none when end > width
  __device__ static int pinned_block(int end, int block_size) { return end > 0 ? (end - 1) / block_size : -1; }
  __device__ static uint32_t block_key(const Params& p, const float* row, int end, int, int b) {
    if (b == pinned_block(end, p.block_size)) return fkey(__int_as_float(0x7f800000));
    float m = __int_as_float(0xff800000);
    const int c0 = b * p.block_size;
    if (p.block_size == 8 && c0 + 8 <= end && c0 + 8 <= p.width) {
      const float4 a = *reinterpret_cast<const float4*>(row + c0);
      const float4 c = *reinterpret_cast<const float4*>(row + c0 + 4);
      m = nanmax(nanmax(nanmax(a.x, a.y), nanmax(a.z, a.w)), nanmax(nanmax(c.x, c.y), nanmax(c.z, c.w)));
      return fkey(nanmax(__int_as_float(0xff800000), m));
    }
    for (int i = 0; i < p.block_size; ++i) {
      const int c = c0 + i;
      if (c < end && c < p.width) m = nanmax(m, row[c]);
    }
    return fkey(m);
  }

  template <int N, class Sync>
  __device__ static void consume(const Params& p, int r, int iter, bool, Sync& sync, Ring&, uint8_t* smem, uint32_t) {
    uint32_t* keys = reinterpret_cast<uint32_t*>(smem);
    uint32_t* hist = keys + p.cache_keys;
    int* scan = reinterpret_cast<int*>(hist + 256);
    const int t = threadIdx.x;
    const int end = p.row_end[r / p.row_repeat];
    if (iter == 0) sync.wait();
    const int nvalid = end > 0 ? min((end + p.block_size - 1) / p.block_size, (p.width + p.block_size - 1) / p.block_size) : 0;
    const float* row = p.logits + static_cast<long long>(r) * p.ld;
    const bool cached = nvalid <= p.cache_keys;
    const bool global = !cached && p.gkeys != nullptr && nvalid <= p.gkeys_stride;
    uint32_t* gk = global ? p.gkeys + static_cast<long long>(r) * p.gkeys_stride : nullptr;
    auto key_of = [&](int b) -> uint32_t {
      return cached ? keys[b] : (global ? __ldcg(gk + b) : block_key(p, row, end, nvalid, b));
    };
    uint32_t* kdst = cached ? keys : gk;
    trace(kTraceId, 2);
    if (cached || global) {
      if (p.block_size == 8) {
        // lane pair (2j, 2j+1) reads block b's two float4s; 128 blocks per 256-thread sweep, 8 sweeps in flight
        const int half = t & 1;
        constexpr int U = 8;
        for (int b0 = 0; b0 < nvalid; b0 += 128 * U) {
          float4 v[U];
#pragma unroll
          for (int u = 0; u < U; ++u) {
            const int b = b0 + u * 128 + (t >> 1);
            const int c = b * 8 + half * 4;
            if (b < nvalid && c + 4 <= end && c + 4 <= p.width) {
              v[u] = *reinterpret_cast<const float4*>(row + c);
            } else {
              const float ninf = __int_as_float(0xff800000);
              v[u] = make_float4(ninf, ninf, ninf, ninf);
              for (int i = 0; i < 4; ++i)
                if (b < nvalid && c + i < end && c + i < p.width) (&v[u].x)[i] = row[c + i];
            }
          }
#pragma unroll
          for (int u = 0; u < U; ++u) {
            const int b = b0 + u * 128 + (t >> 1);
            float m = nanmax(nanmax(v[u].x, v[u].y), nanmax(v[u].z, v[u].w));
            m = nanmax(m, __shfl_xor_sync(0xffffffffu, m, 1));
            if (half == 0 && b < nvalid)
              kdst[b] = (b == pinned_block(end, 8)) ? fkey(__int_as_float(0x7f800000)) : fkey(m);
          }
        }
      } else {
        for (int b = t; b < nvalid; b += kComputeThreads) kdst[b] = block_key(p, row, end, nvalid, b);
      }
    }
    if (global) __threadfence_block();
    compute_sync();
    trace(kTraceId, 3);
    const uint32_t ninf = fkey(__int_as_float(0xff800000));
    // ---- exact radix select of the k-th largest key
    uint32_t prefix = 0, mask = 0;
    int krem = p.k;
    const bool take_all = nvalid <= p.k;
    // digit search over nbins (<= 256) descending bins held in hist[]: sets prefix/mask/krem
    auto pick_digit = [&](int shift, int bits) {
      const int d = 255 - t;
      const int h = d < (1 << bits) ? static_cast<int>(hist[d]) : 0;
      int x = h;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane_id() >= o) x += y;
      }
      if (lane_id() == 31) scan[2 + warp_id()] = x;
      compute_sync();
      int base = 0;
      for (int w = 0; w < warp_id(); ++w) base += scan[2 + w];
      const int incl = base + x, excl = incl - h;
      if (h > 0 && excl < krem && incl >= krem) {
        scan[0] = d;
        scan[1] = krem - excl;
      }
      compute_sync();
      prefix |= static_cast<uint32_t>(scan[0]) << shift;
      mask |= ((1u << bits) - 1u) << shift;
      krem = scan[1];
      compute_sync();
    };
    // one histogram pass over a source of n keys (list in smem, or the row's keys)
    auto hist_pass = [&](const uint32_t* list, int n, int shift, int bits) {
      for (int i = t; i < 256; i += kComputeThreads) hist[i] = 0;
      compute_sync();
      for (int i = t; i < n; i += kComputeThreads) {
        const uint32_t kk = list ? list[i] : key_of(i);
        if ((kk & mask) == prefix) atomicAdd(&hist[(kk >> shift) & ((1u << bits) - 1u)], 1u);
      }
      compute_sync();
      pick_digit(shift, bits);
    };
    if (!take_all)
      for (int pass = 0; pass < 4; ++pass) hist_pass(nullptr, nvalid, 24 - 8 * pass, 8);
    trace(kTraceId, 4);
    // ---- ordered compaction: thread t owns blocks [t*C, (t+1)*C)
    const int C = (nvalid + kComputeThreads - 1) / kComputeThreads;
    const int b0 = min(t * C, nvalid), b1 = min(b0 + C, nvalid);
    int n_eq = 0, n_gt = 0;
    if (!take_all)
      for (int b = b0; b < b1; ++b) {
        const uint32_t kk = key_of(b);
        n_gt += kk > prefix;
        n_eq += kk == prefix;
      }
    // exclusive scans of n_eq (tie rank) and of the selected count (output slot)
    auto excl_scan = [&](int v) -> int {
      int x = v;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane_id() >= o) x += y;
      }
      if (lane_id() == 31) scan[2 + warp_id()] = x;
      compute_sync();
      if (t == 0) {
        int acc = 0;
        for (int w = 0; w < 8; ++w) {
          const int s = scan[2 + w];
          scan[2 + w] = acc;
          acc += s;
        }
        scan[10] = acc;
      }
      compute_sync();
      const int res = scan[2 + warp_id()] + x - v;
      compute_sync();
      return res;
    };
    const int eq_before = excl_scan(n_eq);
    const int take_eq = take_all ? 0 : max(0, min(n_eq, krem - eq_before));
    int n_sel = 0;
    if (take_all) {
      for (int b = b0; b < b1; ++b) n_sel += key_of(b) != ninf;
    } else {
      n_sel = n_gt + take_eq;
    }
    int slot = excl_scan(n_sel);
    const int total = scan[10];
    int eq_seen = 0;
    int* orow = p.out + static_cast<long long>(r) * p.out_stride;
    for (int b = b0; b < b1; ++b) {
      const uint32_t kk = key_of(b);
      bool sel;
      if (take_all) {
        sel = kk != ninf;
      } else {
        sel = kk > prefix || (kk == prefix && eq_seen < take_eq);
        if (kk == prefix) ++eq_seen;
        sel = sel && kk != ninf;
      }
      if (sel) orow[slot++] = b;
    }
    compute_sync();
    trace(kTraceId, 5);
    for (int i = total + t; i < p.k; i += kComputeThreads) orow[i] = -1;
    // ---- optional consumer expansion (ratio 1: sparse block = candidate block)
    if (p.sparse_idx) {
      compute_sync();
      __threadfence_block();
      const int ke = p.cons_ke[r];
      const int max_block = (max(ke, 0) + p.block_size - 1) / p.block_size;
      // candidates are ascending: the valid ones (< max_block) form a prefix
      int nv = 0;
      for (int i = t; i < total; i += kComputeThreads) nv += orow[i] < max_block;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) nv += __shfl_xor_sync(0xffffffffu, nv, o);
      if (lane_id() == 0) scan[2 + warp_id()] = nv;
      compute_sync();
      if (t == 0) {
        int s = 0;
        for (int w = 0; w < 8; ++w) s += scan[2 + w];
        scan[11] = s;
      }
      compute_sync();
      const int n_valid = scan[11];
      const int last = n_valid > 0 ? orow[n_valid - 1] : -1;
      int* si = p.sparse_idx + static_cast<long long>(r) * p.si_stride;
      for (int i = t; i < p.k; i += kComputeThreads) si[i] = i < n_valid ? orow[i] : (n_valid > 0 ? last : 0);
      if (t == 0) {
        const int last_start = last * p.block_size;
        const int fill = min(max(ke - last_start, 1), p.block_size);
        p.sparse_end[r] = n_valid > 0 ? (n_valid - 1) * p.block_size + fill : 0;
      }
    }
    compute_sync();
  }
};

}  // namespace w3
