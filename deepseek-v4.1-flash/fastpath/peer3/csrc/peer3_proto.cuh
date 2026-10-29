// Peer tier v3: GB300 <-> RTX PRO 6000 cold-expert protocol through one pinned, host-mapped segment.
//
// Per MoE layer call with T <= kMaxTokens tokens, every token publishes its own record (one release store), so the
// send can be issued per token straight from a router/top-k stage (W4) or a persistent layer kernel (W8):
//   rec[t].w[6]                 fp32 route weights (top-k order)
//   xq[t][5120], xs[t][160]     the token's MXFP8 activation row (only written when the token has a cold pick)
//   rec[t].word0 (release, sys) = tag(16) | ids(6 x 8): cold-local expert id per top-k slot, 0xFF = not cold
// tag = counter(4) | layer(6) | T - 1 (6); the counter cycles 1..15 once per layer call (so tag != 0 and two
// consecutive calls never share a tag), the layer selects the RTX's weight table entry.
// The RTX persistent kernel acknowledges EVERY call (also calls without cold picks) by writing done = tag
// (release, sys) after its output rows out[t][5120] (bf16, the weighted sum of the token's cold experts in top-k
// order) are written. The GB300 waits for done == tag before it reads rows and before the next call reuses records.
#pragma once
#include <cstdint>

namespace peer3 {

constexpr int kHidden = 5120;
constexpr int kInter = 2304;
constexpr int kTopk = 6;
constexpr int kMaxTokens = 64;
constexpr int kScalesPerRow = kHidden / 32;  // 160

// Segment layout (bytes). Every region starts on its own 4 KiB page; control words own whole 128 B lines.
constexpr uint64_t kOffMagic = 0;          // u64, GB300 writes kMagic at init
constexpr uint64_t kOffVersion = 64;       // u64, sidecar writes kProtoVersion at init; the GB300 refuses others
constexpr uint64_t kOffDone = 128;         // u64, RTX: tag of the last completed call
constexpr uint64_t kOffRtxState = 256;     // u64: the sidecar sets 1 at launch, the kernel 0 when it exits
constexpr uint64_t kOffCtrl = 384;         // u64, sidecar host -> RTX kernel: 1 = exit the persistent kernel
constexpr uint64_t kOffRtxStats = 512;     // u64[8], RTX: calls, calls with work, pairs, experts, ...
// u64, GB300 (W4 expert_pred, at A1 of a layer) -> RTX: the layer's predicted cold experts, one 64-bit store: bits
// 0..31 cold rows (u8, best first, 0xFF = none), 32..37 MoE layer, 38..40 count k (<= 4), 48..63 a sequence number
// (a new hint changes the word). The RTX L2-prefetches them while it waits for the layer's call. Hint only.
constexpr uint64_t kOffHint = 1024;
// d1 (W4 COPY_AHEAD): u32 host ready word. The GB300 router arms 0 before a call's records, the RTX sets 1 after the
// call's done tag (release), and the GB300 copy chain waits on it (a stream memory op) to pull the out rows into
// device memory during MegaMoE. The sidecar advertises it with kCapReady at kOffCap at init.
constexpr uint64_t kOffReady = 1152;
constexpr uint64_t kOffCap = 1160;
constexpr uint64_t kCapReady = 0x52454459;
constexpr uint64_t kOffRec = 4096;         // kMaxTokens records of kRecBytes
constexpr uint64_t kRecBytes = 128;        // word0 at +0, w[6] at +8, global ids int16[6] at +32 (optional)
constexpr uint64_t kOffXq = kOffRec + kMaxTokens * kRecBytes;            // 12288
constexpr uint64_t kOffXs = kOffXq + uint64_t(kMaxTokens) * kHidden;    // MXFP8 scales
constexpr uint64_t kOffOut = ((kOffXs + kMaxTokens * kScalesPerRow + 4095) / 4096) * 4096;
// Call statistics ring: after acknowledging a call, the RTX writes one 32-byte record (fields, system fence, then
// seq); the sidecar's host thread folds the records into per-layer histograms. Record: u64 seq, u32 tag, u16 rows
// (tokens with a cold pick), u16 pairs (cold routes), u16 distinct cold experts, u16 distinct hot experts
// (0xFFFF = unknown), u32 RTX busy ns (call release to last CTA done; 0 without work), u32 GB300 exposed wait ns of
// the previous call + u32 that call's tag (0 when not forwarded).
constexpr uint64_t kOffStatRing = ((kOffOut + uint64_t(kMaxTokens) * kHidden * 2 + 4095) / 4096) * 4096;
constexpr uint32_t kStatRing = 32768;
constexpr uint64_t kStatRecBytes = 32;
constexpr uint64_t kSegBytes = kOffStatRing + uint64_t(kStatRing) * kStatRecBytes;
// Optional, statistics only: the token's 6 global expert ids (int16, -1 = masked) at rec + 32 and a copy of the
// call's tag (u16) at rec + 44, written before word0. The RTX counts distinct hot experts only when the tag matches.
constexpr uint64_t kRecGids = 32;
constexpr uint64_t kRecGidsTag = 44;
// Optional, statistics only, record 0 of a call: the GB300's wait for its PREVIOUS call, forwarded by the publisher
// of token 0 (peer3::post_prev_wait): u64 at +48 = wait ns (31 bits) |
// previous call's tag << 32 | this call's tag << 48. The RTX takes it only when the carrier tag is this call's (a
// publisher that does not forward leaves a stale value there) and copies it into the call's stats record.
constexpr uint64_t kRecPrevWait = 48;
constexpr int kNumExperts = 384;              // global routed experts per layer
constexpr uint64_t kMagic = 0x3352454550ull;  // "PEER3"
constexpr uint64_t kProtoVersion = 2;         // v3.1: stats ring, version word (v3.0 had none); the hint is additive

constexpr int kMaxLayers = 64;
__host__ __device__ constexpr uint32_t make_tag(uint32_t counter, uint32_t layer, uint32_t T) {
  return (counter << 12) | (layer << 6) | (T - 1);
}
__host__ __device__ constexpr uint32_t tag_tokens(uint32_t tag) { return (tag & 63u) + 1; }
__host__ __device__ constexpr uint32_t tag_layer(uint32_t tag) { return (tag >> 6) & 63u; }
__host__ __device__ constexpr uint32_t tag_counter(uint32_t tag) { return tag >> 12; }
__host__ __device__ constexpr uint32_t next_counter(uint32_t c) { return c >= 15 ? 1 : c + 1; }
__host__ __device__ constexpr uint64_t make_word0(uint32_t tag, uint64_t ids48) { return (uint64_t(tag) << 48) | ids48; }
__host__ __device__ constexpr uint32_t word0_tag(uint64_t w) { return uint32_t(w >> 48); }
__host__ __device__ constexpr uint32_t word0_id(uint64_t w, int k) { return uint32_t((w >> (8 * k)) & 0xFF); }

// ---------------------------------------------------------------------------------------------------------------
// PTX helpers (sm_70+)
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint64_t ld_acquire_sys_u64(const void* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys_u64(void* p, uint64_t v) {
  asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_relaxed_sys_u64(void* p, uint64_t v) {
  asm volatile("st.relaxed.sys.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_gpu_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_gpu_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t atom_add_acq_rel_gpu(uint32_t* p, uint32_t v) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ uint4 ld_volatile_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// ---------------------------------------------------------------------------------------------------------------
// MXFP8 row quantizer, bit-identical to peer_fused2._route_send2 (vLLM _mxfp8_e4m3_quantize recipe): per 32 values
// e8m0 = clamp(ceil(log2(amax / 448)) + 127, 0, 254) from the fp32 bits of div_rn(amax, 448), all-zero blocks at 0,
// scale capped at 253 like the reference, values times the exact 2^(127 - e) and rounded to e4m3 (RNE, satfinite).
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t mxfp8_scale_bits(float amax) {
  const float q = __fdiv_rn(amax, 448.0f);
  const uint32_t bits = __float_as_uint(q);
  const uint32_t ex = (bits >> 23) & 0xFF, mant = bits & 0x7FFFFF;
  uint32_t sb = ex == 0 ? (mant > 0x400000 ? 1u : 0u) : ex + (mant != 0 ? 1u : 0u);
  return sb < 253 ? sb : 253;
}

}  // namespace peer3
