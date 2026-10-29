// Peer tier v3, GB300 side: device-callable send / wait / add stages.
//
// These are written as warp- or CTA-level __device__ stages so a fused router/top-k kernel (W4) or a persistent
// layer kernel (W8) can call them directly; peer3_gb300.cu wraps them in standalone CUDA-graph-safe kernels.
//
// Call sequence per MoE layer call (T tokens, 1 <= T <= kMaxTokens):
//   tag = peer3::current_tag(state, layer, T)           same value for every token of the call
//   per token t:  peer3::send_token_warp(...)            any time after the token's top-k exists
//   ... hot experts (MegaMoE) ...
//   one thread:   ok = peer3::wait_done(seg, tag, ...)   ALWAYS once per call (also when no token is cold)
//   per cold t:   peer3::add_row(...)                    y[t] = bf16(float(y[t]) + float(out[t]))
//   once:         peer3::advance(state)                  after every reader of `state` for this call is done
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "peer3_proto.cuh"

namespace peer3 {

// GB300-side device state (device memory, zero-initialized, then state[0] = 1):
//   state[0] counter of the current call (1..15)        state[1] finish arrival counter
struct GbStats {               // device memory (GB_STATS_WORDS int64)
  unsigned long long calls, waited, wait_ns, timeouts;
  unsigned long long hist[16];  // exposed-wait histogram, bucket i = [2^(i-1), 2^i) us, bucket 0 = < 1 us
  unsigned long long last_wait_ns, last_tag;  // the latest call's wait, forwarded by the next send (post_prev_wait)
};

__device__ __forceinline__ uint32_t current_tag(const uint32_t* state, int layer, int T) {
  return make_tag(state[0], uint32_t(layer), uint32_t(T));
}

// Route split for one token: cold-local ids (0xFF = not cold) packed in 48 bits, hot rows written for MegaMoE.
template <typename IdT>
__device__ __forceinline__ uint64_t split_routes(const IdT* ids, const int* row_map, IdT* hot_out, int lane) {
  uint32_t cold = 0xFF;
  if (lane < kTopk) {
    const long long e = static_cast<long long>(ids[lane]);
    IdT hot = IdT(-1);
    if (e >= 0) {
      const int r = row_map[e];
      if (r >= 0) hot = IdT(r);
      else cold = uint32_t(-r - 1);
    }
    if (hot_out != nullptr) hot_out[lane] = hot;
  }
  uint64_t packed = 0;
#pragma unroll
  for (int k = 0; k < kTopk; ++k) packed |= uint64_t(__shfl_sync(0xffffffffu, cold, k) & 0xFF) << (8 * k);
  return packed;
}

// Optional, statistics only: the token's 6 global expert ids (int16, -1 = masked) and the call's tag into its record,
// by lanes 0..6 (any 7 threads). Must precede the token's word0 release, as send_token_* orders it: then the RTX
// counts the call's distinct hot experts for the sidecar's statistics. Calls without it are counted as unknown.
template <typename IdT>
__device__ __forceinline__ void post_route_ids(char* seg, uint32_t tag, int t, const IdT* ids_row, int lane) {
  char* rec = seg + kOffRec + uint64_t(t) * kRecBytes;
  if (lane < kTopk) reinterpret_cast<int16_t*>(rec + kRecGids)[lane] = int16_t(ids_row[lane]);
  if (lane == kTopk) *reinterpret_cast<uint16_t*>(rec + kRecGidsTag) = uint16_t(tag);
}

// Optional, statistics only: one thread of the CTA/warp that publishes token 0, before that token's release.
// Forwards the GB300's wait of the previous call (from wait_done's GbStats) through record 0 to the RTX, which puts it
// into the sidecar's per-call statistics. `tag` is the call being published.
__device__ __forceinline__ void post_prev_wait(char* seg, const GbStats* stats, uint32_t tag) {
  if (stats == nullptr) return;
  const uint64_t ns = stats->last_wait_ns, last = stats->last_tag;
  const uint64_t w = ns > 0x7FFFFFFFull ? 0x7FFFFFFFull : ns;
  *reinterpret_cast<uint64_t*>(seg + kOffRec + kRecPrevWait) = w | ((last & 0xFFFFull) << 32) | (uint64_t(tag) << 48);
}

__device__ __forceinline__ bool any_cold(uint64_t ids48) {
#pragma unroll
  for (int k = 0; k < kTopk; ++k)
    if (word0_id(ids48, k) != 0xFF) return true;
  return false;
}

// Quantize 32-value block b of the token's bf16 row to MXFP8 into the segment (the route_send2 recipe).
__device__ __forceinline__ void quant_block(char* seg, int t, const __nv_bfloat16* x_row, int b) {
  uint8_t* xq = reinterpret_cast<uint8_t*>(seg + kOffXq) + uint64_t(t) * kHidden;
  uint8_t* xs = reinterpret_cast<uint8_t*>(seg + kOffXs) + uint64_t(t) * kScalesPerRow;
  {
    {
      const uint4* src = reinterpret_cast<const uint4*>(x_row + b * 32);
      float v[32];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 u = src[i];
        const uint32_t ws[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          v[i * 8 + j * 2 + 0] = __uint_as_float(ws[j] << 16);
          v[i * 8 + j * 2 + 1] = __uint_as_float(ws[j] & 0xFFFF0000u);
        }
      }
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
      const uint32_t sb = mxfp8_scale_bits(amax);
      const float inv = __uint_as_float((254u - sb) << 23);
      uint32_t q[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v[i * 4 + 0] * inv, v[i * 4 + 1] * inv),
                                                                 __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v[i * 4 + 2] * inv, v[i * 4 + 3] * inv),
                                                                 __NV_SATFINITE, __NV_E4M3);
        q[i] = uint32_t(lo) | (uint32_t(hi) << 16);
      }
      uint4* dst = reinterpret_cast<uint4*>(xq + b * 32);
      dst[0] = make_uint4(q[0], q[1], q[2], q[3]);
      dst[1] = make_uint4(q[4], q[5], q[6], q[7]);
      xs[b] = uint8_t(sb);
    }
  }
}

// One warp per token: quantize the row (only if the token has a cold pick), write the route weights, then lane 0
// publishes word0 with a system-scope release. `seg` is the device view of the host segment. For W8's route stage.
__device__ __forceinline__ void send_token_warp(char* seg, uint32_t tag, int t, uint64_t ids48, const float* w,
                                                const __nv_bfloat16* x_row, int lane) {
  char* rec = seg + kOffRec + uint64_t(t) * kRecBytes;
  if (any_cold(ids48)) {
    if (lane < kTopk) reinterpret_cast<float*>(rec + 8)[lane] = w[lane];
    for (int b = lane; b < kScalesPerRow; b += 32) quant_block(seg, t, x_row, b);
  }
  __syncwarp();
  // The release is cumulative over the warp's writes ordered by __syncwarp (a release store is
  // fence.acq_rel.sys + store: one system fence per token, no extra one).
  if (lane == 0) st_release_sys_u64(rec, make_word0(tag, ids48));
}

// One CTA (>= kScalesPerRow threads) per token: one 32-value block per thread, then thread 0 publishes.
__device__ __forceinline__ void send_token_cta(char* seg, uint32_t tag, int t, uint64_t ids48, const float* w,
                                               const __nv_bfloat16* x_row) {
  char* rec = seg + kOffRec + uint64_t(t) * kRecBytes;
  if (any_cold(ids48)) {
    if (threadIdx.x < kTopk) reinterpret_cast<float*>(rec + 8)[threadIdx.x] = w[threadIdx.x];
    if (threadIdx.x < kScalesPerRow) quant_block(seg, t, x_row, threadIdx.x);
  }
  __syncthreads();
  if (threadIdx.x == 0) st_release_sys_u64(rec, make_word0(tag, ids48));  // cumulative over the CTA's writes
}

// One thread: wait (acquire, sys) until the RTX acknowledged `tag`. Returns false on timeout.
__device__ __forceinline__ bool wait_done(const char* seg, uint32_t tag, long long max_spins, GbStats* stats) {
  const uint64_t t0 = globaltimer();
  long long spins = 0;
  uint64_t done = ld_acquire_sys_u64(seg + kOffDone);
  while (done != tag && spins < max_spins) {
    done = ld_acquire_sys_u64(seg + kOffDone);
    ++spins;
  }
  const bool ok = done == tag;
  if (stats != nullptr) {
    const uint64_t ns = globaltimer() - t0;
    atomicAdd(&stats->calls, 1ull);
    atomicAdd(&stats->wait_ns, ns);
    if (!ok) atomicAdd(&stats->timeouts, 1ull);
    int b = 0;
    for (uint64_t us = ns / 1000; us > 0 && b < 15; us >>= 1) ++b;
    atomicAdd(&stats->hist[b], 1ull);
    stats->last_wait_ns = ns;
    stats->last_tag = tag;
  }
  return ok;
}

__device__ __forceinline__ uint4 add_bf16x8(uint4 yv, const uint4 o) {
  uint32_t* yw = reinterpret_cast<uint32_t*>(&yv);
  const uint32_t ow[4] = {o.x, o.y, o.z, o.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float y0 = __uint_as_float(yw[j] << 16) + __uint_as_float(ow[j] << 16);
    const float y1 = __uint_as_float(yw[j] & 0xFFFF0000u) + __uint_as_float(ow[j] & 0xFFFF0000u);
    const __nv_bfloat162 r = __floats2bfloat162_rn(y0, y1);
    yw[j] = *reinterpret_cast<const uint32_t*>(&r);
  }
  return yv;
}

// CTA-level: y[t] = bf16(float(y[t]) + float(out[t])) over the hidden dim (after a successful wait_done). Each
// thread issues all its host reads before using any, so the row costs one host round trip per kAddBatch chunks.
constexpr int kAddBatch = 4;
__device__ __forceinline__ void add_row(__nv_bfloat16* y_row, const char* seg, int t) {
  const char* out = seg + kOffOut + uint64_t(t) * kHidden * 2;
  uint4* y4 = reinterpret_cast<uint4*>(y_row);
  for (int c0 = threadIdx.x; c0 < kHidden / 8; c0 += kAddBatch * blockDim.x) {
    uint4 o[kAddBatch];
#pragma unroll
    for (int i = 0; i < kAddBatch; ++i) {
      const int c = c0 + i * blockDim.x;
      if (c < kHidden / 8) o[i] = ld_volatile_v4(out + c * 16);
    }
#pragma unroll
    for (int i = 0; i < kAddBatch; ++i) {
      const int c = c0 + i * blockDim.x;
      if (c < kHidden / 8) y4[c] = add_bf16x8(y4[c], o[i]);
    }
  }
}

__device__ __forceinline__ void advance(uint32_t* state) { state[0] = next_counter(state[0]); }

}  // namespace peer3
