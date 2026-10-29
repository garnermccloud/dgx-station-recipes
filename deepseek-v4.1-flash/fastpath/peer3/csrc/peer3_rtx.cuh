// Peer tier v3, RTX side: one persistent cooperative kernel that serves every small (T <= 64) cold-expert call.
//
// Control: CTA 0 polls slot 0's word0 (acquire, sys); on a new tag its first T threads acquire their slots and
// thread 0 builds the plan in shared memory (distinct cold experts in first-appearance order, (token, slot) pairs
// grouped by expert), publishes it and bumps the epoch; a call without cold picks is acknowledged by CTA 0 alone.
// After the release CTA 0 reads the route weights (needed only by the FC1 epilogues) and flags them.
// Every CTA then, in this order:
//   1. issues the 16 KB weight tiles of its first FC1 item (cp.async.bulk into shared memory, L2 evict-first);
//   2. stages its share of the cold tokens' MXFP8 rows from the host segment into device memory as bf16 (the
//      values of the b12x sidecar's _dequant), one 16-byte chunk per thread, and counts itself in;
//   3. FC1 items (expert, 256-channel tile pair, K slice): fp32 partials; the last slice of a tile pair sums the
//      slices in order and applies MegaMoE's SwiGLU epilogue (bf16 gate/up, clamp 10, __expf, rcp.approx, times the
//      route weight) and MXFP8 (UE8M0 per 32) quantization. After its last FC1 tile the CTA already issues the
//      tiles of its first FC2 item;
//   4. FC2 items (expert, 256-column tile, K slice) wait for their expert's FC1 tiles; the last item of a column
//      tile sums the K slices in order, rounds every (token, slot) output to bf16 and adds the token's cold slots in
//      top-k order in fp32 (MegaMoE's combine), writing bf16 rows to the host segment.
// The 20 column-tile finalizers fence at system scope and count; the last releases done = tag. Work items are
// balanced over the grid for any number of distinct experts, every sum has a fixed order and nothing atomic touches
// data, so the output is deterministic.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "peer3_proto.cuh"

namespace peer3 {

constexpr int kMaxPairs = kMaxTokens * kTopk;  // 384 cold (token, slot) pairs per call
constexpr int kMaxExperts = 129;
constexpr int kExpWords = (kMaxExperts + 31) / 32;  // 5-word bitmap of cold experts
#ifndef PEER3_CALL_COPIES
#define PEER3_CALL_COPIES 16
#endif
constexpr int kCallCopies = PEER3_CALL_COPIES;  // copies of the call block; CTA b reads copy b % kCallCopies
constexpr int kCallWords = 80;                  // header + kMaxTokens slot words, padded to whole 128 B lines
constexpr int kHintWord = kCallWords - 1;       // the call block's word 79: CTA 0's copy of the newest hint word
constexpr int kThreads = 256;
constexpr int kUpTiles = kInter / 256;         // 9 tile pairs (up tile i, gate tile i + 9) of w13
constexpr int kOutTiles = kHidden / 256;       // 20 column tiles of w2
constexpr int kKt1 = kHidden / 128;            // 40 K tiles for FC1
constexpr int kKt2 = kInter / 128;             // 18 K tiles for FC2
constexpr int kMT = 8;                         // tokens per chunk (the MMA's N)
constexpr int kTileBytes = 16384;
constexpr int kSfBytes = 1024;
constexpr int kStages = 2;
constexpr int kPart1Rows = 1344;               // default capacity: ks1 * pairs (fp32 [.., 512] FC1 partials)
constexpr int kPart2Rows = 640;                // default capacity: ks2 * pairs (fp32 [.., 256] per column tile)
constexpr float kSwigluLimit = 10.0f;
constexpr int kPrefetchMax = 8;                // most popular cold experts per layer kept for the L2 prefetch
constexpr uint32_t kPopHalveEvery = 1024;      // a layer's popularity counters halve every this many of its calls

// One layer's cold experts on the RTX, in b12x's prepared ("rp") layout (b12x-spec/spec.py).
struct LayerWeights {
  const uint8_t* w13;
  const uint8_t* s13;
  const uint8_t* w2;
  const uint8_t* s2;
  int n_cold;
  int fmt;  // must be 1 (b12x prepared)
};

// A word that many CTAs poll or count on gets its own 128 B line (pollers on a shared line slow every access to it).
struct alignas(128) Hot {
  uint32_t v;
  uint32_t pad_[31];
};

// Per-call synchronization carries the RTX call sequence number (seq, +1 per released call, kept across launches in
// the call block) instead of counters reset between calls: a flag is ready when it equals the current seq, so no
// reset ever races a release. Arrival counters wrap by themselves (atom.inc), and the last arriver publishes.
struct alignas(128) TpFlags {
  uint32_t tp[32];  // FC1 tile pair tp of this plan expert finalized (inter rows written) in call seq
};
constexpr int kStageInFlight = 8;  // host loads in flight per staging thread
constexpr int kRowChunks = kHidden / 16 + kScalesPerRow / 16;  // 320 value chunks + 10 scale chunks per row
__host__ __device__ constexpr int num_stagers(int n_rows) {
  return (n_rows * kRowChunks + kThreads * kStageInFlight - 1) / (kThreads * kStageInFlight);
}
constexpr int kMaxStagers = num_stagers(kMaxTokens);
static_assert(kMaxStagers <= 32, "one polling thread per stager, all in warp 0");

constexpr int kNumEvSlots = 20;
struct RtxWork {  // device memory, zeroed once before the first launch
  uint32_t last_tag, stat_seq, pad_[30];       // CTA 0 only
  Hot end_count, end_seq;                      // CTAs done with the call (wraps at G); the last sets end_seq = seq
  Hot done_count, fin_seq;                     // column tiles finalized (wraps at 20); the last sets fin_seq = seq
  Hot w_ready;                                 // route weights of call seq read (CTA G-1)
  Hot xready[kMaxStagers];                     // stager i's rows of call seq staged
  // The call block, kCallCopies copies written by CTA 0 with relaxed stores and no fence: word 0 = seq << 32 | tag
  // (tag 0 = exit), word 1 + t = token t's word0 with seq & 0xFFFF in place of the tag (every slot is rewritten each
  // call, so a slot word not yet updated is always from call seq - 1 and never validates).
  alignas(128) unsigned long long call[kCallCopies][kCallWords];
  float pair_w[kMaxPairs];                     // route weight per plan pair (read from the host by CTA G-1)
  uint32_t fc1_count[kMaxExperts * kUpTiles];  // K slices done per (expert, tile pair); wraps at ks1
  TpFlags fc1_tp[kMaxExperts];
  Hot fc2_count[kOutTiles];                    // items done per column tile; wraps at n_exp * ks2
  unsigned long long call0;                    // globaltimer when CTA 0 released the call
  unsigned long long cta_ev[256][kNumEvSlots];  // PEER3_EVENTS: per-CTA event times of the current call, see kEv*
  unsigned long long ev_sum[kNumEvSlots];       // sums over calls with work of (latest CTA's event - call0)
  unsigned long long ev_min_sum[kNumEvSlots];   // ... and of (earliest CTA's event - call0)
  unsigned long long stats[8];
  // Next-layer L2 prefetch (prefetch_k > 0): per-layer popularity of the cold experts (calls touching each) and the
  // current top list per layer.
  uint32_t pop_calls[kMaxLayers];
  uint32_t top_n[kMaxLayers];
  uint16_t top[kMaxLayers][kPrefetchMax];
  uint32_t pop[kMaxLayers][kMaxExperts];
};
// Event slots: 0 x staged, 1 FC1 item math done, 2 FC1 epilogue done, 3 FC2 waited for FC1, 4 FC2 item math done,
// 5 done released, 6 first FC1 tile landed, 7 last CTA awake, 8 finalize start, 9 finalizer fenced, 10 plan out.
// 11 first tiles issued, 12 FC1 item's x wait done, 13 x slice in shared memory, 14 every CTA's plan built,
// 15 FC1 item's x wait entered, 16 CTA 0 published the call, 17 stager's host loads landed, 18 FC2 tiles issued.
constexpr int kEvX = 0, kEvFc1 = 1, kEvFc1Epi = 2, kEvFc2Wait = 3, kEvFc2 = 4, kEvDone = 5, kEvFirstTile = 6,
              kEvWake = 7, kEvFinStart = 8, kEvFinFence = 9, kEvPlan = 10, kEvIssue = 11, kEvXWait = 12,
              kEvXSlice = 13, kEvPlanAll = 14, kEvXWaitIn = 15, kEvPub = 16, kEvXLoads = 17, kEvIssue2 = 18;
constexpr int kNumEv = 19;

struct RtxParams {
  char* seg;                        // device view of the host segment
  const LayerWeights* layers;       // [kMaxLayers]
  RtxWork* work;
  uint8_t* xstage;                  // staged MXFP8 rows: e4m3 [kMaxTokens][kHidden], then UE8M0 [kMaxTokens][160]
  float* part1;                     // [kPart1Rows * kUpTiles][512]
  float* part2;                     // [kPart2Rows * kOutTiles][256]
  uint8_t* inter;                   // [kMaxPairs][kInter] e4m3
  uint8_t* inter_sf;                // [kMaxPairs][kInter / 32] ue8m0
  unsigned long long idle_limit_ns; // 0 = serve until the control word asks to exit
  unsigned long long wait_limit_ns; // every in-call wait traps after this (protocol bug guard)
  int target_items;                 // FC1/FC2 items to aim for (default: the grid size)
  int part1_rows, part2_rows;       // capacities of part1 / part2 in (K slice x pair) rows
  const uint4* flush;               // benchmarks only: swept after each call so weights are not L2-resident
  unsigned long long flush_bytes;
  int hint_k;                       // > 0: L2-prefetch up to this many hinted cold experts (kOffHint) before a call
  int hint_min_layer;               // ... and skip the popularity prefetch (prefetch_k) for layers from here on
  unsigned long long hint_stagger_ns;  // ... one hinted expert after the other, this far apart
  int prefetch_k;                   // after each call, prefetch the next layer's k most popular cold experts to L2
  int bench_discard;                // benchmarks only: drop the next layer's weights from L2 before the prefetch
};

}  // namespace peer3
