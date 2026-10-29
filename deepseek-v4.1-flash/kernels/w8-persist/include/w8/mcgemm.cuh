// W8: multi-chain tcgen05 MXFP8 GEMM engine for decode widths (<= 32 tokens).
//
// Measured on the GB300 (tests/tc_chains.cu): one tcgen05 accumulation chain with the full per-K-block sequence
// (mbarrier wait + fence + 2 scale copies + 4 MMAs + commit) consumes ~51 GB/s of weights; chains issued from
// different warps run concurrently (2: ~103 GB/s per SM, 4: ~185). So a CTA runs C chains at once, one issuer warp per
// chain (warps 0..C-1, lane 0), each with its own TMEM accumulator and scale columns:
//   McGemm<C, NPAD, KS>: T = C / KS weight tiles of 128 rows, each split into KS equal K parts inside the CTA.
//   Chain c = tile c % T, part c / T. The epilogue sums a tile's parts in fixed order ((part 0 + part 1) + ...), so
//   a task's result is deterministic; per chain the numerics are tcgen05 block_scale with a sequential K order.
//   KS = 1: every chain covers the task's whole K range (bitwise equal to any production kernel that accumulates the
//   same K range sequentially).
//
// SMEM ring slot = 2 chains' 128-K block (CPS = 2 chains per slot; 1 when C = 1): [CPS x 16 KB weight tiles]
// [CPS x 512 B scale atoms]; a ring step (one K block of every part) uses C / CPS slots, slot j holding chains
// CPS j .. CPS j + CPS - 1. Every issuer commits to its slot's empty barrier (ring.empty_count = CPS).
// Activations: the whole task K range in SMEM as SW128 [kb][NPAD rows][128 B] tiles + one 512 B UTCCP scale atom per
// K block (token t: lane t, word 0), loaded once per task by the caller; part k of the range starts at kb = k * nkb/KS.
// Warps 4..7 run the epilogue (TMEM lanes 32 (w % 4) ..), tile by tile.
#pragma once
#include "runtime.cuh"

namespace w8 {

constexpr int kMcTileBytes = 128 * 128;
constexpr int kMcSfBytes = 512;
__host__ __device__ constexpr int mc_slot_bytes(int CPS) { return ((CPS * (kMcTileBytes + kMcSfBytes)) + 1023) / 1024 * 1024; }
// TMEM columns of chain c: accumulator [32c, 32c + 32), SFA [128 + 8c, +4), SFB [128 + 8c + 4, +4)
__host__ __device__ constexpr uint32_t mc_tmem_d(int c) { return 32u * c; }
__host__ __device__ constexpr uint32_t mc_tmem_sfa(int c) { return 128u + 8u * c; }
__host__ __device__ constexpr uint32_t mc_tmem_sfb(int c) { return 128u + 8u * c + 4u; }
constexpr uint32_t kMcTmemCols = 256;

template <int C>
struct McBars {
  uint64_t acc_full[C];  // count 1 (tcgen05.commit of chain c)
  __device__ void init() {
    for (int c = 0; c < C; ++c) mbar_init(&acc_full[c], 1);
  }
};

constexpr int kMcMaxChains = 4;
struct McCounters {
  uint32_t ring_it = 0;                     // ring slots consumed
  uint32_t wsf_n = 0;                       // McChains tasks consumed (weight-scale buffer phases)
  uint32_t acc_n[kMcMaxChains] = {0, 0, 0, 0};  // accumulator hand-offs of each chain (tasks may use fewer chains)
};

template <int C, int NPAD, int KS = 1>
struct McGemm {
  static_assert(C == 1 || C == 2 || C == 4, "1, 2 or 4 chains");
  static_assert(C <= kMcMaxChains, "counters");
  static_assert(KS >= 1 && C % KS == 0, "K parts divide the chains");
  static_assert(NPAD == 16 || NPAD == 32, "M=128 UMMA needs N % 16 == 0");
  static constexpr int T = C / KS;               // weight tiles per task
  static constexpr int CPS = C >= 2 ? 2 : 1;     // chains per ring slot
  static constexpr int SPS = C / CPS;            // ring slots per step
  static constexpr int SLOT_BYTES = mc_slot_bytes(CPS);
  static constexpr int ACT_KB_BYTES = NPAD * 128;

  // Producer (one lane of the producer warp): K blocks kb0 .. kb0 + nkb (nkb % KS == 0) of the task's T = 2 tiles,
  // rows row0 .. row0 + 256 (consecutive), in the order the issuers consume them. Every ring slot holds both tiles of
  // one K block and costs two TMA-unit operations (each costs ~55 ns of issue time whatever its size, W4's
  // micro_bulk): one 2D box of 256 rows x 128 B (wmap's box) and one copy of both tiles' 512 B scale atoms,
  // sf(dst, kb, bar) (1 KB: tile 0's atom, then tile 1's).
  //   l2_tail >= 0: once the slot l2_tail of this task has been issued (the ring is then full), L2-prefetch the
  //   task's remaining tiles, so they stream while the ring drains.
  //   pace(it): called before acquiring ring slot it (the producer may hold back here).
  template <class RingT, class SF, class Pace = void (*)(uint32_t)>
  __device__ static void produce(const RingT& ring, uint32_t& it, const CUtensorMap* wmap, int row0, int kb0, int nkb,
                                 uint64_t policy, Diag* diag, SF&& sf, int l2_tail = -1, Pace&& pace = [](uint32_t) {}) {
    static_assert(CPS == 2 && T == 2, "a ring slot = the two tiles of one K block");
    const int nkbp = nkb / KS;
    for (int i = 0; i < nkbp; ++i)
      for (int j = 0; j < SPS; ++j, ++it) {
        if (i * SPS + j == l2_tail + 1)
          for (int i2 = i; i2 < nkbp; ++i2)
            for (int j2 = (i2 == i ? j : 0); j2 < SPS; ++j2) tma_prefetch_2d(wmap, (kb0 + j2 * nkbp + i2) * 128, row0);
        pace(it);
        uint8_t* slot = ring.acquire(it, diag);
        const int kb = kb0 + j * nkbp + i;  // slot j holds part j (chains 2j, 2j + 1 = tiles 0, 1)
        tma_load_2d(slot, wmap, kb * 128, row0, ring.full_bar(it), policy);
        sf(slot + CPS * kMcTileBytes, kb, ring.full_bar(it));
        ring.submit(it, CPS * (kMcTileBytes + kMcSfBytes));
      }
  }

  // Consumer (all compute threads): the task's C chains over nkb K blocks; act / act_sf as described above.
  //   epi(t, row, acc[NPAD]) runs on warps 4..7 for every tile (row = TMEM lane 0..127 = weight row in the tile),
  //   acc = the fixed-order sum of the tile's K parts.
  //   stamps (optional, debug): chain 0 records when each step's slot was full, stamps[i] (i < 16)
  template <class RingT, int CB, class Epi>
  __device__ static void run(const RingT& ring, McBars<CB>* bars, McCounters& cn, const uint8_t* act,
                             const uint8_t* act_sf, int nkb, uint32_t tmem, Diag* diag, Epi&& epi,
                             unsigned long long* stamps = nullptr) {
    static_assert(RingT::SLOT_BYTES >= SLOT_BYTES, "ring slot");
    static_assert(CB >= C, "a completion barrier per chain");
    const uint32_t w = warp_id(), lane = lane_id();
    const int nkbp = nkb / KS;
    if (w < static_cast<uint32_t>(C)) {
      // ---------------- issuer of chain w
      umma::fence_after_sync();
      if (lane == 0) {
        const int c = static_cast<int>(w), q = c % CPS, part = c / T;
        const uint32_t d = tmem + mc_tmem_d(c), sfa = tmem + mc_tmem_sfa(c), sfb = tmem + mc_tmem_sfb(c);
        for (int i = 0; i < nkbp; ++i) {
          const uint32_t it = cn.ring_it + i * SPS + c / CPS;
          uint8_t* slot = ring.wait_full(it, diag);
          if (stamps != nullptr && c == 0 && i < 16) stamps[i] = globaltimer();
          umma::fence_after_sync();
          const int ka = part * nkbp + i;  // the activation K block of this step
          umma::cp_sf_32x128b_x4(sfa, umma::sdesc_sf(slot + CPS * kMcTileBytes + q * kMcSfBytes));
          umma::cp_sf_32x128b_x4(sfb, umma::sdesc_sf(act_sf + ka * 512));
          const uint64_t ad = umma::sdesc_k128_sw128(slot + q * kMcTileBytes);
          const uint64_t bd = umma::sdesc_k128_sw128(act + ka * ACT_KB_BYTES);
#pragma unroll
          for (uint32_t kk = 0; kk < 4; ++kk)
            umma::mma_mxf8(d, ad + 2 * kk, bd + 2 * kk, umma::idesc_mxf8(128, NPAD, kk, kk), sfa, sfb,
                           (i > 0 || kk > 0) ? 1u : 0u);
          umma::commit(ring.empty_bar(it));
        }
        umma::commit(&bars->acc_full[c]);
      }
      __syncwarp();
    } else if (w >= 4 && w < 8) {
      // ---------------- epilogue, tile by tile; a tile's K parts summed in part order
      const uint32_t qd = w & 3;
      for (int t = 0; t < T; ++t) {
        float acc[NPAD];
#pragma unroll 1
        for (int k = 0; k < KS; ++k) {
          const int c = t + k * T;
          mbar_wait(&bars->acc_full[c], cn.acc_n[c] & 1, diag, 0xC3);
          umma::fence_after_sync();
          uint32_t r[NPAD];
          umma::ld_32x32b<NPAD>(tmem + ((qd * 32) << 16) + mc_tmem_d(c), r);
          umma::ld_wait();
#pragma unroll
          for (int i = 0; i < NPAD; ++i) acc[i] = k == 0 ? __uint_as_float(r[i]) : acc[i] + __uint_as_float(r[i]);
        }
        if (stamps != nullptr && w == 4 && lane == 0 && t == 0) stamps[6] = globaltimer();  // acc of tile 0 in registers
        epi(t, static_cast<int>(qd * 32 + lane), acc);
      }
      if (stamps != nullptr && w == 4 && lane == 0) stamps[7] = globaltimer();  // epilogue stores issued
      umma::fence_before_sync();
    }
    compute_sync();
    cn.ring_it += nkbp * SPS;
#pragma unroll
    for (int c = 0; c < C; ++c) cn.acc_n[c] += 1;
  }
};

// McChains<C, NPAD, AS>: C (1 or 2) independent accumulation chains per CTA, each with its own weight tile and K range
// (production split-K partitions: equal or not, e.g. wq_a+wkv at M = 2: 12 / 12 / 12 / 4 K blocks), its own
// accumulator, and its own epilogue call (no cross-chain sum: the caller combines chains in its own order).
// Ring slot = ONE chain's two consecutive K blocks of weights (2 x 16 KB = 32 KB, one 3D TMA box), so an issuer waits /
// commits once per 2 K blocks (ring.empty_count = 1). Slots are issued step by step: step i holds slot (chain c, K
// blocks kb0_c + 2 i, + 1) for every chain with i < nkb_c / 2, in chain order (producer and issuers count the same
// sequence). The weight scales of a task (flashinfer 128x4 atoms, UTCCP order: chain c's nkb_c atoms are contiguous,
// sfa(dst, c, bytes, bar) copies them) go into the stage area once per task (wsf, before the slots) and into TMEM up
// front, as do the activation scales: per K block the issuer then runs only the 4 MMAs.
// Weight map: 3D {128 B, rows, K blocks}, box {128, 128, 2}, SW128. Activations: the CTA's K range in SMEM as SW128
// [kb][NPAD][128 B] tiles + one scale atom per K block at act_sf + kb * AS, token t's 4 sub-block bytes at t * 16
// (AS = NPAD * 16 packs the atoms: the MMA reads SFB rows < NPAD only; measured: an SFB TMEM address that is not a
// multiple of 4 columns faults, so 4 K blocks cannot share one copy). Chain c's first K block is act index act_kb0_c.
// TMEM: accumulator of chain c at column 32 c; from column 32 C, per chain: weight scales [4 nkb_c], activation scales
// double-buffered [8] (4 * sum(nkb) + 8 C + 32 C <= 256).
// run() order per issuer: wait for the weight scales and copy them all into TMEM (they do not depend on the
// activations: in a persistent grid this runs while an earlier stage still does), then wait for the activations
// (act_bar); per K block one activation-scale copy + 4 MMAs.
// Barriers (McWsf): wsf_full (producer expect-tx + the copies), wsf_free (count 2: 2 / C tcgen05.commits per issuer
// once its scale copies are done, so GEMMs with 1 and 2 chains can share it: the next task may overwrite wsf).
struct McChain {
  int row0;     // first weight row (a 128-row tile)
  int kb0;      // first K block of the weight
  int nkb;      // K blocks (even)
  int act_kb0;  // index of its first K block in the CTA's activation image
};

struct McWsf {  // in SMEM; each side (producer lane, compute threads) counts its tasks in its own register
  uint64_t full, free_;
  __device__ void init() {
    mbar_init(&full, 1);
    mbar_init(&free_, 2);
  }
};

template <int C, int NPAD, int AS = 512>
struct McChains {
  static_assert(C == 1 || C == 2, "1 or 2 chains");
  static_assert(NPAD == 8 || NPAD == 16 || NPAD == 32, "N = 8 / 16 / 32");
  static_assert(AS % 16 == 0 && AS >= NPAD * 16 && AS <= 512, "atom stride");
  static constexpr int SLOT_BYTES = 2 * kMcTileBytes;
  static constexpr int ACT_KB_BYTES = NPAD * 128;
  __host__ __device__ static constexpr int act_bytes(int nkb) { return nkb * ACT_KB_BYTES; }
  __host__ __device__ static constexpr int act_sf_bytes(int nkb) { return nkb * AS + (512 - AS); }
  // byte of token t, sub-block sub of K block kb in the activation scale atoms
  __host__ __device__ static constexpr int act_sf_off(int t, int kb, int sub) { return kb * AS + t * 16 + sub; }
  __host__ __device__ static constexpr int wsf_bytes(int nkb_total) { return nkb_total * kMcSfBytes; }
  // ring position of (chain c, step i) relative to the task's first slot
  __device__ static int pos(const McChain (&ch)[C], int c, int i) {
    if (C == 1) return i;
    const int n0 = ch[0].nkb / 2, n1 = ch[C - 1].nkb / 2, lo = n0 < n1 ? n0 : n1;
    if (i < lo) return 2 * i + c;
    return 2 * lo + (i - lo);  // only the longer chain has steps left
  }
  __device__ static int slots(const McChain (&ch)[C]) {
    int n = 0;
    for (int c = 0; c < C; ++c) n += ch[c].nkb / 2;
    return n;
  }
  __device__ static int wsf_off(const McChain (&ch)[C], int c) {  // byte offset of chain c's atoms in wsf
    return c == 0 ? 0 : ch[0].nkb * kMcSfBytes;
  }

  // Producer (one lane): the task's weight scales into wsf (once the previous task's issuers released it), then the
  // weight slots.
  // pace(k): called before the task's k-th weight request (ring slot or L2-tail prefetch); may hold the producer back
  // (rate-limits the weight stream so latency-bound stages keep short memory queues).
  template <class RingT, class SFA, class Pace = void (*)(int)>
  __device__ static void produce(const RingT& ring, uint32_t& it, McWsf& ws, uint32_t& n_tasks, uint8_t* wsf,
                                 const CUtensorMap* wmap, const McChain (&ch)[C], uint64_t policy, Diag* diag, SFA&& sfa,
                                 int l2_tail = -1, Pace&& pace = [](int) {}) {
    if (n_tasks > 0) mbar_wait(&ws.free_, (n_tasks - 1) & 1, diag, 0xC5);
    int bytes = 0;
    for (int c = 0; c < C; ++c) bytes += ch[c].nkb * kMcSfBytes;
    mbar_arrive_expect_tx(&ws.full, static_cast<uint32_t>(bytes));
    for (int c = 0; c < C; ++c) sfa(wsf + wsf_off(ch, c), c, ch[c].nkb * kMcSfBytes, &ws.full);
    ++n_tasks;
    int nmax = 0;
    for (int c = 0; c < C; ++c) nmax = ch[c].nkb / 2 > nmax ? ch[c].nkb / 2 : nmax;
    int k = 0;
    for (int i = 0; i < nmax; ++i)
      for (int c = 0; c < C; ++c) {
        if (i >= ch[c].nkb / 2) continue;
        if (k == l2_tail + 1) {  // the ring is full: L2-prefetch the task's remaining slots
          int kp = k;
          for (int i2 = i; i2 < nmax; ++i2)
            for (int c2 = (i2 == i ? c : 0); c2 < C; ++c2)
              if (i2 < ch[c2].nkb / 2) {
                pace(kp++);
                tma_prefetch_3d(wmap, 0, ch[c2].row0, ch[c2].kb0 + 2 * i2);
              }
        }
        if (k <= l2_tail) pace(k);
        uint8_t* slot = ring.acquire(it, diag);
        tma_load_3d_hint(slot, wmap, 0, ch[c].row0, ch[c].kb0 + 2 * i, ring.full_bar(it), policy);
        ring.submit(it, SLOT_BYTES);
        ++it;
        ++k;
      }
  }

  // Issuer of chain CC (the whole warp CC calls it). Descriptors: the 64-bit SW128 K-major SMEM descriptor is a
  // constant high word plus (address >> 4) in the low 14 bits, so the loop only adds 32-bit offsets.
  template <int CC, class RingT, int CB>
  __device__ static void issue(const RingT& ring, McBars<CB>* bars, const McCounters& cn, McWsf& ws, const uint8_t* wsf,
                               uint64_t* act_bar, uint32_t act_phase, const uint8_t* act, const uint8_t* act_sf,
                               const McChain (&ch)[C], uint32_t tmem, Diag* diag, unsigned long long* stamps) {
    constexpr uint32_t kHi = (1024u >> 4) | (1u << 14) | (2u << 29);  // SBO 1024 B, version 1, SWIZZLE_128B
    umma::fence_after_sync();
    const int nkb = ch[CC].nkb, nst = nkb / 2;
    int col = 32 * C;
    if (CC == 1) col += 4 * ch[0].nkb + 8;
    if (col + 4 * nkb + 8 > static_cast<int>(kMcTmemCols)) __trap();  // scale columns must fit
    const uint32_t d = tmem + mc_tmem_d(CC), sfa0 = tmem + static_cast<uint32_t>(col),
                   sfb0 = sfa0 + 4u * static_cast<uint32_t>(nkb);
    mbar_wait_warp(&ws.full, cn.wsf_n & 1, diag, 0xC6);
    umma::fence_after_sync();
    const uint8_t* wsf_c = wsf + wsf_off(ch, CC);
    for (int j = 0; j < nkb; ++j) umma::cp_sf_32x128b_x4_w(sfa0 + 4u * j, umma::sdesc_sf(wsf_c + j * kMcSfBytes));
#pragma unroll
    for (int k = 0; k < 2 / C; ++k) umma::commit_w(&ws.free_);  // wsf may be refilled once these copies are done
    if (stamps != nullptr && CC == 0 && lane_id() == 0) stamps[14] = globaltimer();
    mbar_wait_warp(act_bar, act_phase, diag, 0xC7);
    if (stamps != nullptr && CC == 0 && lane_id() == 0) stamps[15] = globaltimer();
    umma::fence_after_sync();
    const uint32_t ring_lo = smem_u32(ring.base) >> 4;
    const uint32_t act_lo = (smem_u32(act) >> 4) + static_cast<uint32_t>(ch[CC].act_kb0) * (ACT_KB_BYTES >> 4);
    const uint8_t* sf_c = act_sf + ch[CC].act_kb0 * AS;
    // ring position of step i: pos(ch, CC, i) (equal chains: 2 i + CC; one chain: i)
    uint32_t it = cn.ring_it + static_cast<uint32_t>(pos(ch, CC, 0));
    for (int i = 0; i < nst; ++i) {
      const uint32_t slot_i = it % RingT::NSLOT;
      mbar_wait_warp(&ring.full[slot_i], (it / RingT::NSLOT) & 1, diag, 0xE1);
      umma::fence_after_sync();
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int kw = 2 * i + h;
        const uint32_t sfb = sfb0 + 4u * h;
        umma::cp_sf_32x128b_x4_w(sfb, umma::sdesc_sf(sf_c + kw * AS));
        const uint32_t a_lo = ring_lo + slot_i * (RingT::SLOT_BYTES >> 4) + h * (kMcTileBytes >> 4);
        const uint32_t b_lo = act_lo + static_cast<uint32_t>(kw) * (ACT_KB_BYTES >> 4);
#pragma unroll
        for (uint32_t kk = 0; kk < 4; ++kk)
          umma::mma_mxf8_w2(d, a_lo + 2 * kk, b_lo + 2 * kk, kHi, umma::idesc_mxf8(128, NPAD, kk, kk), sfa0 + 4u * kw,
                            sfb, (i > 0 || h > 0 || kk > 0) ? 1u : 0u);
      }
      umma::commit_w(&ring.empty[slot_i]);
      const int next = pos(ch, CC, i + 1) - pos(ch, CC, i);
      it += static_cast<uint32_t>(next);
    }
    umma::commit_w(&bars->acc_full[CC]);
  }

  // Both chains (C == 2, equal lengths) from ONE issuer warp, interleaved step by step in ring order (pos 2 i + c):
  // the tensor pipe overlaps the two dependent chains from a single instruction stream (tests/mma_lat_probe.cu: ~53 clk
  // per MMA amortized vs ~127 for one chain), with no second issuer warp. Same operands and per-chain MMA order as
  // issue<0> + issue<1>, so the accumulators are identical.
  template <class RingT, int CB>
  __device__ static void issue_pair(const RingT& ring, McBars<CB>* bars, const McCounters& cn, McWsf& ws,
                                    const uint8_t* wsf, uint64_t* act_bar, uint32_t act_phase, const uint8_t* act,
                                    const uint8_t* act_sf, const McChain (&ch)[C], uint32_t tmem, Diag* diag) {
    static_assert(C == 2, "a pair of chains");
    constexpr uint32_t kHi = (1024u >> 4) | (1u << 14) | (2u << 29);  // SBO 1024 B, version 1, SWIZZLE_128B
    umma::fence_after_sync();
    const int nkb = ch[0].nkb, nst = nkb / 2;
    if (ch[1].nkb != nkb) __trap();  // ring order 2 i + c needs equal chains
    uint32_t sfa0[2], sfb0[2], act_lo[2];
    const uint8_t* sf_c[2];
    mbar_wait_warp(&ws.full, cn.wsf_n & 1, diag, 0xC6);
    umma::fence_after_sync();
    int col = 32 * C;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      if (col + 4 * nkb + 8 > static_cast<int>(kMcTmemCols)) __trap();  // scale columns must fit
      sfa0[c] = tmem + static_cast<uint32_t>(col);
      sfb0[c] = sfa0[c] + 4u * static_cast<uint32_t>(nkb);
      const uint8_t* wsf_c = wsf + wsf_off(ch, c);
      for (int j = 0; j < nkb; ++j) umma::cp_sf_32x128b_x4_w(sfa0[c] + 4u * j, umma::sdesc_sf(wsf_c + j * kMcSfBytes));
      col += 4 * nkb + 8;
    }
    umma::commit_w(&ws.free_);  // wsf_free counts 2
    umma::commit_w(&ws.free_);
    mbar_wait_warp(act_bar, act_phase, diag, 0xC7);
    umma::fence_after_sync();
    const uint32_t ring_lo = smem_u32(ring.base) >> 4;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      act_lo[c] = (smem_u32(act) >> 4) + static_cast<uint32_t>(ch[c].act_kb0) * (ACT_KB_BYTES >> 4);
      sf_c[c] = act_sf + ch[c].act_kb0 * AS;
    }
    uint32_t it = cn.ring_it;
    for (int i = 0; i < nst; ++i) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const uint32_t slot_i = it % RingT::NSLOT;
        mbar_wait_warp(&ring.full[slot_i], (it / RingT::NSLOT) & 1, diag, 0xE1);
        umma::fence_after_sync();
        const uint32_t d = tmem + mc_tmem_d(c);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int kw = 2 * i + h;
          const uint32_t sfb = sfb0[c] + 4u * h;
          umma::cp_sf_32x128b_x4_w(sfb, umma::sdesc_sf(sf_c[c] + kw * AS));
          const uint32_t a_lo = ring_lo + slot_i * (RingT::SLOT_BYTES >> 4) + h * (kMcTileBytes >> 4);
          const uint32_t b_lo = act_lo[c] + static_cast<uint32_t>(kw) * (ACT_KB_BYTES >> 4);
#pragma unroll
          for (uint32_t kk = 0; kk < 4; ++kk)
            umma::mma_mxf8_w2(d, a_lo + 2 * kk, b_lo + 2 * kk, kHi, umma::idesc_mxf8(128, NPAD, kk, kk), sfa0[c] + 4u * kw,
                              sfb, (i > 0 || h > 0 || kk > 0) ? 1u : 0u);
        }
        umma::commit_w(&ring.empty[slot_i]);
        if (i == nst - 1) umma::commit_w(&bars->acc_full[c]);
        ++it;
      }
    }
  }

  // epi(c, row, acc[NPAD]) on warps 4..7 for every chain (row = TMEM lane = weight row in the tile).
  // act_bar / act_phase: the activations' arrival (mbarrier; the issuers wait for it after the weight scales).
  // PAIR (C == 2): both chains from warp 0 (issue_pair) instead of one issuer warp per chain.
  template <bool PAIR = false, class RingT, int CB, class Epi>
  __device__ static void run(const RingT& ring, McBars<CB>* bars, McCounters& cn, McWsf& ws, const uint8_t* wsf,
                             uint64_t* act_bar, uint32_t act_phase, const uint8_t* act, const uint8_t* act_sf,
                             const McChain (&ch)[C], uint32_t tmem, Diag* diag, Epi&& epi,
                             unsigned long long* stamps = nullptr) {
    static_assert(RingT::SLOT_BYTES >= SLOT_BYTES, "ring slot");
    static_assert(CB >= C, "a completion barrier per chain");
    const uint32_t w = warp_id(), lane = lane_id();
    if (w < static_cast<uint32_t>(C)) {
      // the whole warp issues (warp-uniform operands, one lane elected inside each tcgen05 op: see umma.cuh); the chain
      // index is a template argument so every operand is compile-time or uniform arithmetic (no R2UR per MMA)
      if constexpr (PAIR) {
        if (w == 0) issue_pair(ring, bars, cn, ws, wsf, act_bar, act_phase, act, act_sf, ch, tmem, diag);
      } else {
        if (w == 0) issue<0>(ring, bars, cn, ws, wsf, act_bar, act_phase, act, act_sf, ch, tmem, diag, stamps);
        if (C == 2 && w == 1) issue<C - 1>(ring, bars, cn, ws, wsf, act_bar, act_phase, act, act_sf, ch, tmem, diag, stamps);
      }
      __syncwarp();
    } else if (w >= 4 && w < 8) {
      const uint32_t qd = w & 3;
      for (int c = 0; c < C; ++c) {
        mbar_wait(&bars->acc_full[c], cn.acc_n[c] & 1, diag, 0xC3);
        umma::fence_after_sync();
        uint32_t r[NPAD];
        umma::ld_32x32b<NPAD>(tmem + ((qd * 32) << 16) + mc_tmem_d(c), r);
        umma::ld_wait();
        float acc[NPAD];
#pragma unroll
        for (int i = 0; i < NPAD; ++i) acc[i] = __uint_as_float(r[i]);
        epi(c, static_cast<int>(qd * 32 + lane), acc);
      }
      umma::fence_before_sync();
    }
    compute_sync();
    ++cn.wsf_n;
    cn.ring_it += static_cast<uint32_t>(slots(ch));
#pragma unroll
    for (int c = 0; c < C; ++c) cn.acc_n[c] += 1;
  }
};

}  // namespace w8
