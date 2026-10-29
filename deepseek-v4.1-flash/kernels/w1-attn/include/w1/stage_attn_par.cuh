// W1 bitwise-parallel mode: FlashMLA's exact decode arithmetic (see stage_attn_seq.cuh), spread over the GPU.
//
// FlashMLA's result depends on processing a token's 64-key blocks in order: the lazy running max m_b decides how P is
// rounded to bf16, and O / l are accumulated block after block. Only m_b's dependence is cheap (64 block maxima), so:
//   Score task (token t, block b), all 64 heads: dequant K_b, S = Q K_b^T (FlashMLA's rails and K order), publish the
//     block's raw per-(key half, head) maxima; read the maxima of blocks <= b, replay FlashMLA's lazy-rescale
//     trajectory to get m_b exactly; publish P_b = bf16(exp2(fma(S, mult, -m_b))) as the smem image of the PV B
//     operand, and the per-(head, key half) sums (FlashMLA's even/odd chains).
//   PV task (token t, 32-head group g, 128-dim tile mt): dequant V for its dims of every block up front, replay the
//     trajectory itself (f_b, rescale_b, final m), then in block order: rescale O by f_b if rescale_b,
//     O += V_b^T P_b^T (K=16 steps in key order), l = fmaf(l, f_b, sum_b); epilogue (sink, inverse RoPE, MXFP8 cast)
//     for its (32 heads x 128 dims).
// Every value FlashMLA rounds is computed by the same operations in the same order, so the output is bit-identical.
//
// Latency, not math, is the cost at decode:
//   * hand-over: an L2 round trip is ~0.5 us, so every published value travels as a tagged word, 64-bit
//     (payload | launch tag << 32), single-copy atomic, written relaxed and polled relaxed in batches: a consumer that
//     sees the tag sees the payload. No fences, no flags, one round trip per hand-over. Tags are unique per launch
//     (epoch), so the scratch is never cleared.
//   * instruction fetch: GB300 runs cold straight-line code at ~3 cycles / instruction (measured, bench/icache_probe)
//     and a launch executes each instruction about once, so the stages are compact loops, not unrolled bodies.
// Waits point backwards in the task order (score tasks block-major, then PV tasks), so a resident persistent grid
// always progresses; every poll is bounded (trap after 0.5 s).
#pragma once
#include "attn_params.cuh"
#include "ptx.cuh"
#include "stage_attn_seq.cuh"
#include "../w8/w8_shim.cuh"

namespace w1 {

constexpr int ParMaxBlocks = 10;

// ------------------------------------------------------------------------------------------------ tagged words
__device__ __forceinline__ uint64_t tw_make(uint32_t v, uint32_t tag) { return ((uint64_t)tag << 32) | v; }
__device__ __forceinline__ void tw_st(uint64_t* p, uint32_t v, uint32_t tag) {
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(tw_make(v, tag)) : "memory");
}
__device__ __forceinline__ void tw_st2(uint64_t* p, uint32_t v0, uint32_t v1, uint32_t tag) {
    asm volatile("st.relaxed.gpu.global.v2.b64 [%0], {%1, %2};" ::"l"(p), "l"(tw_make(v0, tag)), "l"(tw_make(v1, tag))
                 : "memory");
}
__device__ __forceinline__ void tw_ld2(const uint64_t* p, uint64_t& a, uint64_t& b) {
    asm volatile("ld.relaxed.gpu.global.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
__device__ __forceinline__ uint64_t tw_ld(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ bool tw_ok(uint64_t w, uint32_t tag) { return (uint32_t)(w >> 32) == tag; }

__device__ __noinline__ void tw_timeout(const uint64_t* p, uint32_t tag) {
#ifdef W1_DEBUG_TIMEOUT
    printf("w1 tagged-word timeout cta %d tid %d addr %p want %u\n", blockIdx.x, threadIdx.x, p, tag);
#endif
    __trap();
}

// Batched poll of K word pairs (16 B, 2 tagged words each): entry k with active(k) reads src(k) and, once both words
// carry the tag, hands the two payloads to sink(k, lo, hi). Every pass issues all pending loads before checking any.
template <int K, class Active, class Src, class Sink>
__device__ __forceinline__ void poll_pairs(uint32_t tag, Active active, Src src, Sink sink) {
    static_assert(K <= 32, "pending mask");
    uint32_t pending = 0;
#pragma unroll
    for (int k = 0; k < K; ++k)
        if (active(k)) pending |= 1u << k;
    uint64_t t0 = 0;
    while (pending) {
        uint64_t w0[K], w1[K];
#pragma unroll
        for (int k = 0; k < K; ++k)
            if (pending & (1u << k)) tw_ld2(src(k), w0[k], w1[k]);
#pragma unroll
        for (int k = 0; k < K; ++k)
            if ((pending & (1u << k)) && tw_ok(w0[k], tag) && tw_ok(w1[k], tag)) {
                sink(k, (uint32_t)w0[k], (uint32_t)w1[k]);
                pending &= ~(1u << k);
            }
        if (pending) {
            if (t0 == 0)
                t0 = w8::globaltimer();
            else if (w8::globaltimer() - t0 > 500000000ull)
                tw_timeout(src(__ffs(pending) - 1), tag);
        }
    }
}

struct ParScratch {
    uint64_t* cmax;  // [S][NB][2 key halves][64 heads] raw block maxima
    uint64_t* psum;  // [S][NB][2 key halves][64 heads] FlashMLA per-thread block sums
    uint64_t* pimg;  // [S][NB][2 groups][1024] words of the P^T smem image (32 heads x 64 keys bf16, SW128 K-major)
    uint64_t* hint;  // [S][NB] unused (kept for the scratch layout)
    int nb_max;      // blocks per token in the task grid (ceil(total slots / 64))
};

// The launch tag: warp 8 reads the epoch after griddepcontrol.wait and hands it over through smem + named barrier 3
// (288 threads); compute threads collect it once, at their first hand-over.
struct ParTag {
    static constexpr int kBarrier = 3;
    const volatile uint32_t* slot;
    uint32_t tag = 0;
    bool have = false;
    __device__ __forceinline__ uint32_t get() {
        if (!have) {
            asm volatile("bar.sync %0, %1;" ::"n"(kBarrier), "n"(w8::kBlockThreads) : "memory");
            tag = *slot;
            have = true;
        }
        return tag;
    }
};

// mbarriers live outside the stages' smem layouts and are initialised once per CTA (re-initialising a live mbarrier
// is undefined); every thread tracks the completed-phase counts identically.
// mbarrier parity waits are only unambiguous with at most one phase outstanding: every wait below directly follows
// its own commit / copy, with nothing else committed to the barrier in between.
struct ParBars {
    uint64_t* score;  // QK commit
    uint64_t* q;      // Q bulk copy (score task)
    uint64_t* pvc;    // PV commits
    uint32_t n_score = 0, n_q = 0, n_pv3 = 0;
    __device__ void init() {
        if (threadIdx.x == 0) {
            mbar_init(score, 1);
            mbar_init(q, 1);
            mbar_init(pvc, 1);
            fence_mbar_init();
        }
    }
};

// FlashMLA's lazy-rescale trajectory over blocks 0..last for the warp's 32 heads (one 32-head decision group; lane =
// head) from the raw block maxima cm[bb * 128 + kh * 64] in smem. Calls step(bb, m_bb, f_bb, rescale_bb).
template <class Step>
__device__ __forceinline__ void replay_trajectory(const AttnParams& p, const float* cm, int last, Step step) {
    float mi = kMaxInit;
#pragma unroll 1
    for (int bb = 0; bb <= last; ++bb) {
        const float r0 = cm[bb * 128], r1 = cm[bb * 128 + 64];
        const float c = fmaxf(r0 * p.sm_scale_log2, r1 * p.sm_scale_log2);
        const bool sc = __any_sync(0xffffffffu, c - mi > 6.0f);
        float nm, f;
        if (sc) {
            nm = fmaxf(c, mi);
            f = ex2_approx(mi - nm);
        } else {
            nm = mi;
            f = 1.f;
        }
        mi = nm;
        step(bb, mi, f, sc);
    }
}

// 16 dims of one row: fp8 (16 B, ue8m0 scale) or fp4 (8 B in .x/.y, e4m3 scale) -> two bf16 chunks (exact)
__device__ __forceinline__ void dequant16(const uint4& d, bool fp4, uint32_t sc, uint4& o0, uint4& o1) {
    if (fp4) {
        uint32_t s0, s1;
        e4m3x4_to_bf16x2x2(sc, s0, s1);
        const uint32_t s = __byte_perm(s0, 0, 0x1010);
        uint32_t h[4];
        e2m1x8_to_bf16x2x4(d.x, h);
        o0 = make_uint4(hmul2_bf16(h[0], s), hmul2_bf16(h[1], s), hmul2_bf16(h[2], s), hmul2_bf16(h[3], s));
        e2m1x8_to_bf16x2x4(d.y, h);
        o1 = make_uint4(hmul2_bf16(h[0], s), hmul2_bf16(h[1], s), hmul2_bf16(h[2], s), hmul2_bf16(h[3], s));
    } else {
        const uint32_t s = ue8m0_to_bf16x2(sc);
        uint32_t a, bb;
        e4m3x4_to_bf16x2x2(d.x, a, bb);
        o0.x = hmul2_bf16(a, s);
        o0.y = hmul2_bf16(bb, s);
        e4m3x4_to_bf16x2x2(d.y, a, bb);
        o0.z = hmul2_bf16(a, s);
        o0.w = hmul2_bf16(bb, s);
        e4m3x4_to_bf16x2x2(d.z, a, bb);
        o1.x = hmul2_bf16(a, s);
        o1.y = hmul2_bf16(bb, s);
        e4m3x4_to_bf16x2x2(d.w, a, bb);
        o1.z = hmul2_bf16(a, s);
        o1.w = hmul2_bf16(bb, s);
    }
}

// ------------------------------------------------------------------------------------------------ score task
// Work is laid out for instruction-level parallelism: 8 warps (2 per scheduler) cannot hide latency, so every phase
// is a fully unrolled batch of independent loads / math, all 8 warps busy.
struct ParScore {
    static constexpr int kXS = 65;  // xch row stride (floats): rows and columns both bank-conflict free
    static constexpr uint32_t kKvOff = 0;                    // [8 atoms][64 keys][128 B]
    static constexpr uint32_t kQOff = 65536;                 // [4 pairs][2 rails x 64 heads][128 B]
    static constexpr uint32_t kQPair = 128 * 128;
    static constexpr uint32_t kXchOff = kQOff + 4 * kQPair;  // [64 keys][kXS] fp32 (rail-1 S, then S, then d)
    static constexpr uint32_t kPOff = (kXchOff + 64 * kXS * 4 + 1023) / 1024 * 1024;  // [2 groups][32][64] bf16
    static constexpr uint32_t kRawStride = 528;              // raw row pitch (16 B pad: conflict-free dequant reads)
    static constexpr uint32_t kRawOff = kPOff + 8192;        // [64 rows][kRawStride] raw K rows (fp8 512 / fp4 256 B)
    static constexpr uint32_t kTabOff = kRawOff + 64 * kRawStride;  // [64] row data pointers
    static constexpr uint32_t kSclOff = kTabOff + 64 * 8;    // [64][32] scale bytes
    static constexpr uint32_t kFmtOff = kSclOff + 64 * 32;   // [64]
    static constexpr uint32_t kTrajOff = kFmtOff + 64;       // [64] m_b
    static constexpr uint32_t kCsOff = kTrajOff + 64 * 4;    // [64] cos | sin of the token's position
    static constexpr uint32_t kCmOff = kCsOff + 64 * 4;      // [block][2 key halves][64 heads] raw maxima
    static constexpr uint32_t kSmemBytes = kCmOff + ParMaxBlocks * 128 * 4;
    static constexpr uint32_t kTmemS = 0;  // 128 lanes (rail x key) x 128 cols (rail x head)
    static constexpr uint32_t kTmemCols = 128;

    // q (fused layout, element (dc*1024 + h*16 + e)) -> smem Q tile by cp.async; 16 units of 16 B per thread
    __device__ static void issue_q(const AttnParams& p, uint8_t* smem, int t, int tid) {
        const uint16_t* qt = p.q + (int64_t)t * p.q_tok_stride;
        uint8_t* qs = smem + kQOff;
#pragma unroll 4
        for (int k = 0; k < 16; ++k) {
            const int idx = tid + 256 * k;  // 16 B unit = (dc, h, half)
            const int dc = idx >> 7, h = (idx >> 1) & 63, half = idx & 1;
            const int d0 = dc * 16 + half * 8;
            const uint32_t atom = d0 >> 6, rail = atom & 1, pair = atom >> 1;
            const uint32_t row = rail * 64 + h;
            cp_async16(qs + pair * kQPair + row * 128 + ((((d0 & 63) >> 3) ^ (row & 7)) << 4), qt + (int64_t)idx * 8);
        }
    }
    // FlashMLA's q RoPE on dims 448..511 = atom 7 = pair 3, rail 1 rows (64 + h): 512 units of 16 B, 2 per thread
    __device__ static void rope_q(uint8_t* smem, int tid) {
        const float* cs = reinterpret_cast<const float*>(smem + kCsOff);
        uint8_t* base = smem + kQOff + 3 * kQPair + 64 * 128;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int u = tid + 256 * k;
            const int r = 64 + (u >> 3), cphys = u & 7, c = cphys ^ (r & 7);
            uint4* ptr = reinterpret_cast<uint4*>(base + u * 16);
            uint4 v = *ptr;
            uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = 4 * c + e;
                const float cc = cs[i], s = cs[32 + i];
                const float x0 = __uint_as_float(w[e] << 16), x1 = __uint_as_float(w[e] & 0xFFFF0000u);
                w[e] = pack_bf16x2(fmaf(x0, cc, (-x1) * s), fmaf(x0, s, x1 * cc));
            }
            *ptr = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }

    template <class Sync>
    __device__ static void consume(const AttnParams& p, const ParScratch& ps, int task, Sync& sync, ParTag& tg,
                                   uint8_t* smem, uint32_t tmem_base, ParBars& bars) {
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const int NB = ps.nb_max;
        const int b = task / p.s_q, t = task - b * p.s_q;  // block-major: earlier blocks first
        sync.wait_extern();
        const int nb = AttnSeq::num_blocks(p, t);
        if (b >= nb) return;  // FlashMLA processes no such block; nobody waits for it
        const int64_t tb = (int64_t)t * NB + b;

        // ---- loads. Every global access is lane-coalesced (the L1 takes ~one 128 B line request per cycle).
        //      The dependent slot-index loads go out first, Q (cp.async) while they fly, then the rows.
        uint8_t* fmt = smem + kFmtOff;
        uint64_t* tab = reinterpret_cast<uint64_t*>(smem + kTabOff);
        int sidx = -1, slen = 0, spos = 0;
        if (tid >= 64 && tid < 128) spos = load_position(p, t);
        const int j = b * 64 + tid;
        const bool swa = j < p.swa_width;
        const int jj = swa ? j : j - p.swa_width;
        if (tid < 64) {
            if (swa || (p.comp != nullptr && jj < p.comp_width)) {
                sidx = __ldg((swa ? p.swa_idx + (int64_t)t * p.swa_idx_stride : p.comp_idx + (int64_t)t * p.comp_idx_stride) +
                             jj);
                slen = __ldg((swa ? p.swa_lens : p.comp_lens) + t);
            }
        }
        float cs_v = 0.f;
        if (tid >= 64 && tid < 128) cs_v = __ldg(p.cos_sin + (int64_t)spos * kRopeDim + tid - 64);
        if (tid == 0) {
            // the token's q (64 KB, contiguous) by the TMA engine into the K tile region (free until dequant): the
            // L1 request queue stays free for the rows
            mbar_arrive_expect_tx(bars.q, 65536);
            bulk_g2s(smem + kKvOff, p.q + (int64_t)t * p.q_tok_stride, 65536, bars.q);
        }
        if (tid < 64) {
            uint8_t f = AttnSeq::kInvalid;
            const uint8_t* data = nullptr;
            uint8_t* sdst = smem + kSclOff + tid * 32;
            if (jj < slen && sidx >= 0) {
                const int page_tokens = swa ? p.swa_page : p.comp_page;
                const int blk = sidx / page_tokens, off = sidx - blk * page_tokens;
                const uint8_t* page =
                    (swa ? p.swa : p.comp) + (int64_t)blk * (swa ? p.swa_block_stride : p.comp_block_stride);
                const int row = swa ? 512 : 256, srow = swa ? 16 : 32;
                f = swa ? AttnSeq::kFp8 : AttnSeq::kFp4;
                data = page + off * row;
                const uint8_t* scale = page + (int64_t)page_tokens * row + off * srow;
                cp_async16(sdst, scale);
                if (!swa) cp_async16(sdst + 16, scale + 16);
            } else {
                *reinterpret_cast<uint4*>(sdst) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(sdst + 16) = make_uint4(0, 0, 0, 0);
            }
            fmt[tid] = f;
            tab[tid] = reinterpret_cast<uint64_t>(data);
        }
#ifdef W1_DBG_LOADS
        cta_stamp(p, 2);
#endif
        w8::compute_sync();  // table
        // raw rows: warp w copies rows w + 8 i, one row per instruction (fp8: 32 lanes x 16 B; fp4: 16 lanes)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = warp + 8 * i;
            const uint8_t f = fmt[r];
            if (f != AttnSeq::kInvalid && (f == AttnSeq::kFp8 || lane < 16))
                cp_async16(smem + kRawOff + r * kRawStride + 16 * lane,
                           reinterpret_cast<const uint8_t*>(tab[r]) + 16 * lane);
        }
        if (tid >= 64 && tid < 128) reinterpret_cast<float*>(smem + kCsOff)[tid - 64] = cs_v;
#ifdef W1_DBG_LOADS
        cta_stamp(p, 3);
#endif
        cp_async_wait_all();
        mbar_wait(bars.q, bars.n_q & 1);
        ++bars.n_q;
        w8::compute_sync();  // rows, scales, Q, cos/sin visible
#ifdef W1_DBG_LOADS
        cta_stamp(p, 4);
#endif
        {
            // q staging (global layout: 16 B unit (dc, h, half)) -> swizzled Q tile with FlashMLA's q RoPE; the
            // staging (K tile region) is overwritten only by the dequant, after the barrier below
            const float* cs = reinterpret_cast<const float*>(smem + kCsOff);
            uint8_t* qs = smem + kQOff;
#pragma unroll 4
            for (int k = 0; k < 16; ++k) {
                const int idx = tid + 256 * k;
                const int dc = idx >> 7, h = (idx >> 1) & 63, half = idx & 1;
                const int d0 = dc * 16 + half * 8;
                uint4 v = reinterpret_cast<const uint4*>(smem + kKvOff)[idx];
                if (d0 >= kHeadDim - kRopeDim) {
                    uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int i = (d0 - (kHeadDim - kRopeDim)) / 2 + e;
                        const float c = cs[i], sn = cs[32 + i];
                        const float x0 = __uint_as_float(w[e] << 16), x1 = __uint_as_float(w[e] & 0xFFFF0000u);
                        w[e] = pack_bf16x2(fmaf(x0, c, (-x1) * sn), fmaf(x0, sn, x1 * c));
                    }
                    v = make_uint4(w[0], w[1], w[2], w[3]);
                }
                const uint32_t atom = d0 >> 6, rail = atom & 1, pair = atom >> 1;
                const uint32_t row = rail * 64 + h;
                *reinterpret_cast<uint4*>(qs + pair * kQPair + row * 128 + ((((d0 & 63) >> 3) ^ (row & 7)) << 4)) = v;
            }
            w8::compute_sync();  // every staging read done before the K tile is written
        }
        {
            // dequant, bank-conflict free and fully unrolled: warp w owns atom w (dims 64 w ..), pass i rows
            // 8 i + lane / 4, lane dims 64 w + 16 (lane & 3) .. +16
            const uint8_t* scl = smem + kSclOff;
            uint8_t* kv = smem + kKvOff;
            const int d0 = 64 * warp + 16 * (lane & 3);
            uint4 d[8];
            uint32_t sc[8];
            bool fp4[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = 8 * i + (lane >> 2);
                const uint8_t f = fmt[r];
                fp4[i] = f == AttnSeq::kFp4;
                const uint8_t* raw = smem + kRawOff + r * kRawStride;
                d[i] = make_uint4(0, 0, 0, 0);
                if (f == AttnSeq::kFp8) {
                    d[i] = *reinterpret_cast<const uint4*>(raw + d0);
                } else if (fp4[i]) {
                    const uint2 v = *reinterpret_cast<const uint2*>(raw + d0 / 2);
                    d[i].x = v.x;
                    d[i].y = v.y;
                }
                sc[i] = scl[r * 32 + (fp4[i] ? (d0 >> 4) : (d0 >> 5))];
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = 8 * i + (lane >> 2);
                uint4 o0, o1;
                dequant16(d[i], fp4[i], sc[i], o0, o1);
                *reinterpret_cast<uint4*>(kv + AttnSeq::kv_off(r, d0)) = o0;
                *reinterpret_cast<uint4*>(kv + AttnSeq::kv_off(r, d0 + 8)) = o1;
            }
        }
        fence_proxy_async_smem();
        w8::compute_sync();
#ifdef W1_DBG_LOADS
        cta_stamp(p, 5);
        return;
#endif
        cta_stamp(p, 2);

        // ---- S^T = K Q^T, both decision groups at once (N = 2 rails x 64 heads)
        uint64_t* bar = bars.score;
        if (tid == 0) {
            tc_fence_after();
            const uint32_t idesc = umma_idesc_bf16(128, 128, false, false);
            const uint32_t ka = smem_u32(smem + kKvOff), qa = smem_u32(smem + kQOff);
#pragma unroll
            for (int s = 0; s < 16; ++s)
                tc_mma_ss(tmem_base + kTmemS, umma_desc_sw128(ka + (s >> 2) * 16384 + (s & 3) * 32, 16, 1024),
                          umma_desc_sw128(qa + (s >> 2) * kQPair + (s & 3) * 32, 16, 1024), idesc, s > 0);
            tc_commit(bar);
        }
        mbar_wait(bar, bars.n_score & 1);
        ++bars.n_score;
        tc_fence_after();
        cta_stamp(p, 3);

        const int q4 = warp & 3, hh = warp >> 2;  // hh: heads 32*hh .. (= decision group)
        const bool rail0 = q4 < 2;
        const int key = 32 * (q4 & 1) + lane;
        float* xch = reinterpret_cast<float*>(smem + kXchOff);
        float* xrow = xch + key * kXS + 32 * hh;
        {
            // this thread's 32 S values: rail 0 or 1 of key `key`, heads 32 hh ..
            const uint32_t ts = tmem_base + kTmemS + ((uint32_t)(32 * q4) << 16) + (rail0 ? 0 : 64) + 32 * hh;
            uint32_t a[16], c[16];
            tc_ld_32x32b<16>(ts, a);
            tc_ld_32x32b<16>(ts + 16, c);
            tc_wait_ld();
            if (!rail0) {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    xrow[i] = __uint_as_float(a[i]);
                    xrow[16 + i] = __uint_as_float(c[i]);
                }
            }
            tg.get();  // uniform point: first hand-over of this CTA
            w8::compute_sync();
            if (rail0) {
                // S = rail0 + rail1 (FlashMLA's order), -inf for invalid keys, over the rail-1 values in xch
                const bool kvalid = fmt[key] != AttnSeq::kInvalid;
                float r1[32];
#pragma unroll
                for (int i = 0; i < 32; ++i) r1[i] = xrow[i];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    xrow[i] = kvalid ? __uint_as_float(a[i]) + r1[i] : -INFINITY;
                    xrow[16 + i] = kvalid ? __uint_as_float(c[i]) + r1[16 + i] : -INFINITY;
                }
            }
        }
        w8::compute_sync();
        const uint32_t tag = tg.get();
        // raw block maxima per (key half, head) = column max over xch: 2 threads per column (16 keys each, a tree),
        // combined through one shuffle
        {
            const int col = tid >> 1, part = tid & 1;  // col = kh * 64 + h
            const int kh = col >> 6, h = col & 63;
            const float* cp = xch + (32 * kh + 16 * part) * kXS + h;
            float v[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) v[k] = cp[k * kXS];
#pragma unroll
            for (int w = 8; w >= 1; w >>= 1)
#pragma unroll
                for (int k = 0; k < w; ++k) v[k] = fmaxf(v[k], v[k + w]);
            const float mx = fmaxf(v[0], __shfl_xor_sync(0xffffffffu, v[0], 1));
            if (part == 0) tw_st(ps.cmax + (tb * 2 + kh) * 64 + h, __float_as_uint(mx), tag);
        }
        cta_stamp(p, 4);

        // ---- maxima of blocks 0..b (the token's words [0, (b + 1) * 128), as pairs), then FlashMLA's lazy-rescale
        //      trajectory (warps 0 / 1 = decision groups 0 / 1)
        float* traj = reinterpret_cast<float*>(smem + kTrajOff);
        float* cm = reinterpret_cast<float*>(smem + kCmOff);
        {
            const uint64_t* src = ps.cmax + (int64_t)t * NB * 128;
            const int npair = (b + 1) * 64;
            poll_pairs<3>(
                tag, [&](int k) { return tid + 256 * k < npair; }, [&](int k) { return src + 2 * (tid + 256 * k); },
                [&](int k, uint32_t lo, uint32_t hi) {
                    *reinterpret_cast<float2*>(cm + 2 * (tid + 256 * k)) =
                        make_float2(__uint_as_float(lo), __uint_as_float(hi));
                });
        }
        w8::compute_sync();
        if (warp < 2) {
            const int h = 32 * warp + lane;
            float m_b = 0.f;
            replay_trajectory(p, cm + h, b, [&](int, float m, float, bool) { m_b = m; });
            traj[h] = m_b;
        }
        w8::compute_sync();
        cta_stamp(p, 5);

        // ---- P = bf16(exp2(fma(s, mult, -m_b))) -> P^T image per group; d -> xch for the sums. All 8 warps:
        //      warp (q4, hh) takes key 32 (q4 & 1) + lane, heads 32 hh + 16 (q4 >> 1) .. +16
        {
            const int ib = 16 * (q4 >> 1);
            uint8_t* pt = smem + kPOff + hh * 4096;
            const float* mb = traj + 32 * hh;
            float x[16], d[16];
#pragma unroll
            for (int e = 0; e < 16; ++e) x[e] = xrow[ib + e];
#pragma unroll
            for (int e = 0; e < 16; ++e) d[e] = ex2_approx(fmaf(x[e], p.sm_scale_log2, -mb[ib + e]));
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const int i = ib + e;  // head within the group
                *reinterpret_cast<__nv_bfloat16*>(pt + i * 128 + (((key >> 3) ^ (i & 7)) << 4) + (key & 7) * 2) =
                    __float2bfloat16_rn(d[e]);
                xrow[i] = d[e];
            }
        }
        w8::compute_sync();
        // FlashMLA's per-thread sum: (even keys chain) + (odd keys chain) over a key half, per head
        if (tid < 128) {
            const int h = tid & 63, half = tid >> 6;
            const float* xc = xch + (32 * half) * kXS + h;
            float v[32];
#pragma unroll
            for (int i = 0; i < 32; ++i) v[i] = xc[i * kXS];
            float sx = 0.f, sy = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                sx = sx + v[2 * i];
                sy = sy + v[2 * i + 1];
            }
            tw_st(ps.psum + (tb * 2 + half) * 64 + h, __float_as_uint(sx + sy), tag);
        }
        {
            // the 2 x 4 KB image as 1024 tagged word pairs, lane-coalesced: thread tid writes pairs tid + 256 k
            uint64_t* dst = ps.pimg + tb * 2048;
            const uint2* src = reinterpret_cast<const uint2*>(smem + kPOff);
            uint2 v[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) v[k] = src[tid + 256 * k];
#pragma unroll
            for (int k = 0; k < 4; ++k) tw_st2(dst + 2 * (tid + 256 * k), v[k].x, v[k].y, tag);
        }
        cta_stamp(p, 6);
        tc_fence_before();
    }
};

// ------------------------------------------------------------------------------------------------ PV task
struct ParPV {
    static constexpr int kMaxBlocks = ParMaxBlocks;
    static constexpr uint32_t kVOff = 0;  // [block][2 atoms][64 keys][128 B] = 16 KB each; raw rows of block bb are
                                          // staged in region bb + 1 (consumed before it is overwritten)
    static constexpr uint32_t kPOff = kMaxBlocks * 16384;        // [block][4096] P^T images (also raw stage of bb 9)
    static constexpr uint32_t kSclOff = kPOff + kMaxBlocks * 4096;  // [640][16] scale bytes
    static constexpr uint32_t kFmtOff = kSclOff + 640 * 16;      // [640]
    static constexpr uint32_t kFOff = kFmtOff + 640;             // [block][32] f_b
    static constexpr uint32_t kPsOff = kFOff + kMaxBlocks * 32 * 4;   // [block][2 halves][64] block sums (all heads)
    static constexpr uint32_t kRfOff = kPsOff + kMaxBlocks * 128 * 4;  // [block] rescale flag
    static constexpr uint32_t kMOff = kRfOff + 16 * 4;           // [32] final m
    static constexpr uint32_t kLOff = kMOff + 32 * 4;            // [2][32] l
    static constexpr uint32_t kOscOff = kLOff + 64 * 4;          // [32]
    static constexpr uint32_t kCsOff = kOscOff + 32 * 4;         // [64] cos | sin
    static constexpr uint32_t kSinkOff = kCsOff + 64 * 4;        // [32]
    static constexpr uint32_t kCmOff = kSinkOff + 32 * 4;        // [block][2 key halves][64] raw maxima (all heads)
    static constexpr uint32_t kSmemBytes = kCmOff + kMaxBlocks * 128 * 4;
    static constexpr uint32_t kStageOff = kSclOff;               // epilogue staging (4 KB) reuses the dead scales
    static constexpr uint32_t kTabOff = kPOff + kMaxBlocks * 4096 - 640 * 8;  // [640] row pointers, in the P area's
                                                                              // tail (raw stage of block 9 is its head)
    static_assert(kPOff + 16384 / 2 <= kTabOff, "row table overlaps block 9's raw stage");
    static constexpr uint32_t kTmemO = 0;  // O^T: 128 lanes (dims) x 32 cols (heads)
    static constexpr uint32_t kTmemCols = 32;

    // V tile (key k, local dim d' < 128) offset inside one block's 16 KB tile
    __device__ static __forceinline__ uint32_t v_off(uint32_t k, uint32_t d) {
        return (d >> 6) * 8192u + k * 128u + ((((d & 63u) >> 3) ^ (k & 7u)) << 4) + ((d & 7u) << 1);
    }

    // slot j: format, row pointer (our 128 dims: 128 B fp8 / 64 B fp4) and its 16 B scale chunk (cp.async)
    __device__ static __forceinline__ void slot_table(const AttnParams& p, uint8_t* smem, int j, int idx, int len,
                                                      int mt) {
        const bool swa = j < p.swa_width;
        const int jj = swa ? j : j - p.swa_width;
        uint8_t f = AttnSeq::kInvalid;
        const uint8_t* data = nullptr;
        uint8_t* sdst = smem + kSclOff + j * 16;
        if (jj < len && idx >= 0) {
            const int page_tokens = swa ? p.swa_page : p.comp_page;
            const int blk = idx / page_tokens, off = idx - blk * page_tokens;
            const uint8_t* page =
                (swa ? p.swa : p.comp) + (int64_t)blk * (swa ? p.swa_block_stride : p.comp_block_stride);
            const int row = swa ? 512 : 256, srow = swa ? 16 : 32;
            f = swa ? AttnSeq::kFp8 : AttnSeq::kFp4;
            data = page + off * row + (swa ? 128 * mt : 64 * mt);
            const uint8_t* scale = page + (int64_t)page_tokens * row + off * srow;
            cp_async16(sdst, scale + (swa ? 0 : ((8 * mt) & ~15)));  // fp8: all 16; fp4: the 16 holding ours
        } else {
            *reinterpret_cast<uint4*>(sdst) = make_uint4(0, 0, 0, 0);
        }
        reinterpret_cast<uint64_t*>(smem + kTabOff)[j] = reinterpret_cast<uint64_t>(data);
        smem[kFmtOff + j] = f;
    }

    template <class Sync>
    __device__ static void consume(const AttnParams& p, const ParScratch& ps, int task, Sync& sync, ParTag& tg,
                                   uint8_t* smem, uint32_t tmem_base, ParBars& bars) {
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const int NB = ps.nb_max;
        const int t = task / 8, g = (task >> 2) & 1, mt = task & 3;
        const int h0 = 32 * g;
        sync.wait_extern();
        const int nb = AttnSeq::num_blocks(p, t);
        const int nslots = nb * 64;

        // ---- every load at once: slot j = tid + 256 k (index loads first, then its scale + raw row), cos/sin, sink
        int spos = 0;
        if (tid < 64) spos = load_position(p, t);
        float cs_v = 0.f;  // cos/sin (tid < 64) or sink (64..95), stored once the rows are in flight
        {
            int idx[3], len[3];
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                idx[k] = -1;
                len[k] = 0;
                const int j = tid + 256 * k;
                if (j < nslots) {
                    const bool swa = j < p.swa_width;
                    const int jj = swa ? j : j - p.swa_width;
                    if (swa || (p.comp != nullptr && jj < p.comp_width)) {
                        idx[k] = __ldg((swa ? p.swa_idx + (int64_t)t * p.swa_idx_stride
                                            : p.comp_idx + (int64_t)t * p.comp_idx_stride) +
                                       jj);
                        len[k] = __ldg((swa ? p.swa_lens : p.comp_lens) + t);
                    }
                }
            }
#ifdef W1_DBG_LOADS
            if (idx[0] == 12345678) smem[0] = 1;
            cta_stamp(p, 2);
#endif
            if (tid < 64)
                cs_v = __ldg(p.cos_sin + (int64_t)spos * kRopeDim + tid);
            else if (tid < 96)
                cs_v = __ldg(p.sink + h0 + tid - 64);
#pragma unroll 1
            for (int k = 0; k < 3; ++k)
                if (tid + 256 * k < nslots) slot_table(p, smem, tid + 256 * k, idx[k], len[k], mt);
        }
        w8::compute_sync();  // table
        // raw rows, lane-coalesced: 8 lanes per row (fp8 128 B; fp4 64 B on lanes & 7 < 4), 4 rows per instruction;
        // block bb's rows are staged in region bb + 1
        {
            const int c = lane & 7;
#pragma unroll 1
            for (int bb = 0; bb < nb; ++bb)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = 8 * warp + 4 * h + (lane >> 3), j = bb * 64 + r;
                    const uint8_t f = smem[kFmtOff + j];
                    if (f == AttnSeq::kFp8 || (f == AttnSeq::kFp4 && c < 4))
                        cp_async16(smem + kVOff + (bb + 1) * 16384 + r * 128 + 16 * c,
                                   reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint64_t*>(smem + kTabOff)[j]) +
                                       16 * c);
                }
        }
        if (tid < 64)
            reinterpret_cast<float*>(smem + kCsOff)[tid] = cs_v;
        else if (tid < 96)
            reinterpret_cast<float*>(smem + kSinkOff)[tid - 64] = cs_v;
#ifdef W1_DBG_LOADS
        cta_stamp(p, 3);
#endif
        cp_async_wait_all();
        w8::compute_sync();
#ifdef W1_DBG_LOADS
        cta_stamp(p, 4);
        return;
#endif
#ifndef W1_DBG_EPI
        cta_stamp(p, 2);
#endif

        // ---- dequant V (our 128 dims), block by block: rows (warp * 4 + lane / 8) + 32 kk, dims [16 (lane & 7), +16)
#pragma unroll 1
        for (int bb = 0; bb < nb; ++bb) {
            const uint8_t* raw = smem + kVOff + (bb + 1) * 16384;
            uint8_t* tile = smem + kVOff + bb * 16384;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int r = (warp * 4 + (lane >> 3)) + 32 * kk;
                const int j = bb * 64 + r;
                const uint8_t f = smem[kFmtOff + j];
                const bool fp4 = f == AttnSeq::kFp4;
                const int dl = 16 * (lane & 7);  // local dim
                uint4 d = make_uint4(0, 0, 0, 0);
                if (f == AttnSeq::kFp8) {
                    d = *reinterpret_cast<const uint4*>(raw + r * 128 + dl);
                } else if (fp4) {
                    const uint2 v = *reinterpret_cast<const uint2*>(raw + r * 128 + dl / 2);
                    d.x = v.x;
                    d.y = v.y;
                }
                // scale byte: fp8 -> global scale index (128 mt + dl) / 32 within the 16 B row
                //             fp4 -> (128 mt + dl) / 16 - ((8 mt) & ~15) within the copied 16 B
                const int si = fp4 ? (8 * mt + (dl >> 4)) - ((8 * mt) & ~15) : (4 * mt + (dl >> 5));
                uint4 o0, o1;
                dequant16(d, fp4, smem[kSclOff + j * 16 + si], o0, o1);
                *reinterpret_cast<uint4*>(tile + v_off(r, dl)) = o0;
                *reinterpret_cast<uint4*>(tile + v_off(r, dl + 8)) = o1;
            }
            // block bb's raw rows (region bb + 1) are overwritten by block bb + 1's tile rows; a row's raw bytes and its
            // tile row belong to the same 8 lanes of one warp, so a warp barrier orders them
            __syncwarp();
        }
        fence_proxy_async_smem();
#ifndef W1_DBG_EPI
        cta_stamp(p, 3);
#endif

        // ---- hand-over, one batched poll: our group's P images, raw maxima and block sums (all heads)
        const uint32_t tag = tg.get();
        float* F = reinterpret_cast<float*>(smem + kFOff);
        float* PS = reinterpret_cast<float*>(smem + kPsOff);
        int* RF = reinterpret_cast<int*>(smem + kRfOff);
        float* M = reinterpret_cast<float*>(smem + kMOff);
        float* CM = reinterpret_cast<float*>(smem + kCmOff);
        const uint64_t* cm_t = ps.cmax + (int64_t)t * NB * 128;
        const uint64_t* ps_t = ps.psum + (int64_t)t * NB * 128;
        const uint64_t* pimg_t = ps.pimg + (int64_t)t * NB * 2048 + g * 1024;
        // our group's maxima / sums: per (block, key half) 32 words = 16 pairs; pair q -> (q / 32, q / 16 % 2, q % 16)
        const int nq = nb * 32;
        auto qoff = [&](int k) {
            const int q = tid + 256 * k;
            return ((q >> 5) * 2 + ((q >> 4) & 1)) * 64 + h0 + 2 * (q & 15);
        };
        // (A) the raw maxima are published right after each score task's QK, about a microsecond before P: poll them
        //     first and replay FlashMLA's trajectory while P is still on its way
        poll_pairs<2>(
            tag, [&](int k) { return tid + 256 * k < nq; }, [&](int k) { return cm_t + qoff(k); },
            [&](int k, uint32_t lo, uint32_t hi) {
                *reinterpret_cast<float2*>(CM + qoff(k)) = make_float2(__uint_as_float(lo), __uint_as_float(hi));
            });
        w8::compute_sync();
        if (warp == 0) {
            float m_last = 0.f;
            replay_trajectory(p, CM + h0 + lane, nb - 1, [&](int bb, float m, float f, bool sc) {
                F[bb * 32 + lane] = f;
                if (lane == 0) RF[bb] = sc ? 1 : 0;
                m_last = m;
            });
            M[lane] = m_last;
        }
        // (B) P images and block sums, one batched poll (re-polls whatever is not tagged yet; no hint round trip)
        {
            // entries: [0, 2 kMaxBlocks) P pairs (block e / 2, pair tid + 256 (e % 2)), then 2 sum pairs
            constexpr int kP = 2 * kMaxBlocks;
            poll_pairs<kP + 2>(
                tag, [&](int e) { return e < kP ? (e >> 1) < nb : tid + 256 * (e - kP) < nq; },
                [&](int e) -> const uint64_t* {
                    if (e < kP) return pimg_t + (int64_t)(e >> 1) * 2048 + 2 * (tid + 256 * (e & 1));
                    return ps_t + qoff(e - kP);
                },
                [&](int e, uint32_t lo, uint32_t hi) {
                    if (e < kP) {
                        *reinterpret_cast<uint2*>(smem + kPOff + (e >> 1) * 4096 + 8 * (tid + 256 * (e & 1))) =
                            make_uint2(lo, hi);
                    } else {
                        *reinterpret_cast<float2*>(PS + qoff(e - kP)) =
                            make_float2(__uint_as_float(lo), __uint_as_float(hi));
                    }
                });
        }
        fence_proxy_async_smem();
        w8::compute_sync();
#ifndef W1_DBG_EPI
        cta_stamp(p, 4);
#endif

        // ---- block loop in FlashMLA order, TMEM accumulator O^T [128 dims x 32 heads]. Thread 0 only issues MMAs
        //      (back to back unless a block rescales O); threads 32..95 run the l chains meanwhile.
        const uint32_t idesc = umma_idesc_bf16(128, 32, true, false);
        const int q4 = warp & 3;
        const uint32_t lane_base = (uint32_t)(32 * q4) << 16;
        uint64_t* bar = bars.pvc;  // PV commits (v3 scheme: each wait follows its commit, nothing else outstanding)
        uint32_t rmask = 0;  // blocks > 0 that rescale O (uniform)
#pragma unroll 1
        for (int bb = 1; bb < nb; ++bb) rmask |= (RF[bb] != 0 ? 1u : 0u) << bb;
        auto issue_pv = [&](int bb) {
            tc_fence_after();
            const uint32_t va = smem_u32(smem + kVOff + bb * 16384), pa = smem_u32(smem + kPOff + bb * 4096);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
                tc_mma_ss(tmem_base + kTmemO, umma_desc_sw128(va + ks * 2048, 8192, 1024),
                          umma_desc_sw128(pa + ks * 32, 16, 1024), idesc, (bb > 0 || ks > 0));
        };
        float* lbuf = reinterpret_cast<float*>(smem + kLOff);
        if (tid >= 32 && tid < 96) {  // l per (head, key half): fmaf(l, f_b, sum_b), FlashMLA's order
            const int h = tid & 31, half = (tid - 32) >> 5;
            float li = 0.f;
#pragma unroll 1
            for (int bb = 0; bb < nb; ++bb) li = fmaf(li, F[bb * 32 + h], PS[bb * 128 + half * 64 + h0 + h]);
            lbuf[half * 32 + h] = li;
        }
        if (rmask == 0) {
            if (tid == 0) {
#pragma unroll 1
                for (int bb = 0; bb < nb; ++bb) issue_pv(bb);
            }
        } else {
#pragma unroll 1
            for (int bb = 0; bb < nb; ++bb) {
                if (rmask & (1u << bb)) {
                    // O *= f_bb once PV(bb-1) has landed
                    if (tid == 0) tc_commit(bar);
                    mbar_wait(bar, bars.n_pv3 & 1);
                    ++bars.n_pv3;
                    tc_fence_after();
                    if (warp < 4) {
#pragma unroll 1
                        for (int c = 0; c < 4; ++c) {
                            uint32_t v[8];
                            const uint32_t ta = tmem_base + kTmemO + lane_base + 8 * c;
                            tc_ld_32x32b<8>(ta, v);
                            tc_wait_ld();
#pragma unroll
                            for (int i = 0; i < 8; ++i)
                                v[i] = __float_as_uint(__uint_as_float(v[i]) * F[bb * 32 + 8 * c + i]);
                            tc_st_32x32b<8>(ta, v);
                        }
                        tc_wait_st();
                    }
                    tc_fence_before();
                    w8::compute_sync();
                }
                if (tid == 0) issue_pv(bb);
            }
        }
        if (tid == 0) tc_commit(bar);
        mbar_wait(bar, bars.n_pv3 & 1);
        ++bars.n_pv3;
        tc_fence_after();
#ifndef W1_DBG_EPI
        cta_stamp(p, 5);
#endif

        // ---- epilogue (FlashMLA store_o for our 32 heads x 128 dims): warp w -> dims 32 (w & 3) + lane of the tile,
        //      heads 16 (w >> 2) ..
        w8::compute_sync();  // l
#ifdef W1_DBG_EPI
        cta_stamp(p, 2);
#endif
        if (tid < 32) {
            const float l = lbuf[tid] + lbuf[32 + tid];
            const float m = M[tid];
            const float sink_l2 = reinterpret_cast<const float*>(smem + kSinkOff)[tid] * kL2E;
            float osc;
            asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(osc) : "f"(1.0f), "f"(l + ex2_approx(sink_l2 - m)));
            reinterpret_cast<float*>(smem + kOscOff)[tid] = l == 0.f ? 0.f : osc;
        }
        w8::compute_sync();
#ifdef W1_DBG_EPI
        cta_stamp(p, 3);
#endif
        {
            // warp w: dims 32 (w & 3) + lane of the tile, heads 16 (w >> 2) ..; shuffle-free (smem exchange per warp)
            const float* cs = reinterpret_cast<const float*>(smem + kCsOff);
            const float* oscv = reinterpret_cast<const float*>(smem + kOscOff);
            uint8_t* stage = smem + kStageOff;
            float* E = reinterpret_cast<float*>(smem + kPOff) + warp * (16 * 33);  // [16 heads][33]; P area is dead
            float* MU = reinterpret_cast<float*>(smem + kPOff) + 8 * 16 * 33 + warp * 16;
            const int dim = 128 * mt + 32 * q4 + lane;
            const int j = dim >> 5;
            const bool rope = dim >= kHeadDim - kRopeDim;  // warp-uniform
            const int hb = 16 * (warp >> 2);
            uint32_t v[16];
            tc_ld_32x32b<16>(tmem_base + kTmemO + lane_base + hb, v);
            tc_wait_ld();
#ifdef W1_DBG_EPI
            cta_stamp(p, 4);
#endif
            float y[16];
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                y[e] = __uint_as_float(v[e]);
                E[e * 33 + lane] = y[e];
            }
            __syncwarp();
            if (rope) {
                // FlashMLA's inverse RoPE on the unnormalized O: pairs (2i, 2i + 1) = lanes (l & ~1, l | 1)
                const int i = (dim - (kHeadDim - kRopeDim)) >> 1;
                const float cc = cs[i], sn = -cs[32 + i];
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const float other = E[e * 33 + (lane ^ 1)];
                    const float x0 = (lane & 1) ? other : y[e], x1 = (lane & 1) ? y[e] : other;
                    y[e] = (lane & 1) ? fmaf(x0, sn, x1 * cc) : fmaf(x0, cc, (-x1) * sn);
                }
                __syncwarp();
#pragma unroll
                for (int e = 0; e < 16; ++e) E[e * 33 + lane] = y[e];
                __syncwarp();
            }
            if (lane < 16) {
                // head hb + lane: amax over the warp's 32 dims, FlashMLA's ue8m0 scale and fp8 multiplier
                const int h = hb + lane;
                const float* row = E + lane * 33;
                float amax_raw = fabsf(row[0]);
#pragma unroll 8
                for (int k = 1; k < 32; ++k) amax_raw = fmaxf(amax_raw, fabsf(row[k]));
                const float osc = oscv[h];
                const float amax = fmaxf(1e-4f, amax_raw * osc);
                float sf;
                asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(sf) : "f"(amax), "f"(448.0f));
                const uint32_t sfb = __float_as_uint(sf);
                const int exp_sf = (int)((sfb - 1u) >> 23) + (1 - 127);
                MU[lane] = osc * __uint_as_float((uint32_t)(127 - exp_sf) << 23);
                const int G = (h0 + h) >> 3, sfi = (h & 7) + j * 8;
                uint8_t* sf_t = reinterpret_cast<uint8_t*>(p.out_sf + t * p.sf_stride_tok);
                sf_t[(G * p.sf_stride_grp + (sfi >> 2) * p.sf_stride_hd) * 4 + (sfi & 3)] = (uint8_t)(exp_sf + 127);
            }
            __syncwarp();
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const int h = hb + e;
                const uint8_t q8 = (uint8_t)(f32x2_to_e4m3x2(y[e] * MU[e], 0.f) & 0xFF);
                stage[(h >> 3) * 1024 + ((j & 3) * 8 + (h & 7)) * 32 + lane] = q8;
            }
        }
#ifdef W1_DBG_EPI
        cta_stamp(p, 5);
#endif
        w8::compute_sync();
        {
            // 4 wv groups x (4 chunks x 8 heads x 32) = 4 x 1 KB, each contiguous in the output
            const int grp = tid >> 6, off = (tid & 63) * 16;
            uint8_t* dst = p.out_fp8 + (int64_t)t * (kHeads * kHeadDim) + ((h0 >> 3) + grp) * 4096 + mt * 1024 + off;
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(smem + kStageOff + grp * 1024 + off);
        }
        cta_stamp(p, 6);
        tc_fence_before();
    }
};

}  // namespace w1
