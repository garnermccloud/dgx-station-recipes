// W1 bitwise "split-dims sequential" mode (SS): FlashMLA's exact decode arithmetic with no cross-CTA hand-over.
//
// Task = (token t, 32-head decision group g, dim tiles [T * mtb, T * mtb + T) of 128 dims). Each task runs FlashMLA's
// per-group sequence over the token's 64-key blocks in order (see stage_attn_seq.cuh for the arithmetic): QK for its
// 32 heads over all 512 dims (replicated across the 4 / T dim tasks of the group), the lazy-rescale max, P, the l
// chains, and PV + epilogue for its own 128 T dims only. Nothing is exchanged between CTAs, so the cost is one launch,
// two L2 round trips (slot indices, then rows) and the per-block pipeline:
//   K block b (raw rows staged in place in tile b & 1, dequantized after a CTA barrier) -> QK_b into S ring b & 1
//   (issued as soon as the tile is ready, so QK_{b+1} overlaps softmax_b) -> softmax_b -> PV_b.
// Blocks b + 2.. reuse tile b & 1 once PV_b has read it; their raw rows come through one staging slot, prefetched one
// block ahead.
//
// Instruction-level parallelism: 8 warps cannot hide latency, so each phase is a fully unrolled batch of independent
// work on all warps; shuffles (which carry divergence guards in this inlined code) are avoided in hot phases.
#pragma once
#include "attn_params.cuh"
#include "ptx.cuh"
#include "stage_attn_par.cuh"
#include "stage_attn_seq.cuh"
#include "../w8/w8_shim.cuh"

namespace w1 {

template <int T>  // 128-dim tiles per task: 1 (8 tasks / token), 2 (4) or 4 (2)
struct AttnSS {
    static_assert(T == 1 || T == 2 || T == 4, "T");
    static constexpr int NH = 32;
    static constexpr int kMaxBlocks = 10;
    static constexpr int kTasksPerToken = 2 * (4 / T);
    static constexpr int kXS = 33;            // xch row stride (floats): rows and columns conflict free
    static constexpr int kRawStride = 528;    // raw staging row: data (512 fp8 / 256 fp4) then its scales

    // ---- SMEM
    static constexpr uint32_t kTileOff = 0;                    // [2][8 atoms][64 keys][128 B] bf16 (raw staged inside)
    static constexpr uint32_t kQOff = 2 * 65536;               // [4 pairs][2 rails x 32 heads][128 B]
    static constexpr uint32_t kQPair = 2 * NH * 128;
    static constexpr uint32_t kXchOff = kQOff + 4 * kQPair;    // [64 keys][kXS] fp32: rail-1 S, then S, then d
    static constexpr uint32_t kPOff = (kXchOff + 64 * kXS * 4 + 1023) / 1024 * 1024;  // [2][32 heads][64 keys] bf16
    static constexpr uint32_t kTabOff = kPOff + 2 * 4096;      // [640] row data pointers
    static constexpr uint32_t kSTabOff = kTabOff + 640 * 8;    // [640] row scale pointers
    static constexpr uint32_t kFmtOff = kSTabOff + 640 * 8;    // [640]
    static constexpr uint32_t kCsOff = kFmtOff + 640;          // [64] cos | sin
    static constexpr uint32_t kSinkOff = kCsOff + 64 * 4;      // [32]
    static constexpr uint32_t kRedOff = kSinkOff + 32 * 4;     // [2 key halves][32] block max, [32] m, [32] f
    static constexpr uint32_t kLOff = kRedOff + 4 * NH * 4;    // [2][32] l
    static constexpr uint32_t kMOff = kLOff + 2 * NH * 4;      // [32] final m
    static constexpr uint32_t kOscOff = kMOff + NH * 4;        // [32]
    static constexpr uint32_t kStage2Off = (kOscOff + NH * 4 + 127) / 128 * 128;  // raw rows of block 2 (3-block
                                                                                  // windows: no exposed round trip)
    static constexpr uint32_t kSmemBytes = kStage2Off + 64 * kRawStride;
    // epilogue (tile 0 is dead by then): E exchange [8 warps][16][33] fp32, multipliers [8][16], fp8 stage
    static constexpr uint32_t kEOff = 0;
    static constexpr uint32_t kMuOff = 8 * 16 * 33 * 4;
    static constexpr uint32_t kStageOff = 32768;               // [T][4 wv groups][1 KB]

    static constexpr uint32_t kTmemS = 0;      // S ring: 2 x 64 cols (128 lanes = rail x key, cols = rail x head)
    static constexpr uint32_t kTmemO = 128;    // O^T: T x 32 cols (128 lanes = dims)
    static constexpr uint32_t kTmemCols = 256;

    // tcgen05.mma issue stalls the issuing thread ~50 cycles per instruction (bench/mma_issue.cu), so the MMAs are
    // issued by warp 8 (lane 0), which waits for "tile ready" / "P ready" mbarriers (one arrival per compute warp)
    // and commits to the "QK done" / "PV done" barriers the compute warps wait on. Every barrier has at most one
    // phase outstanding: tile b + 2 reuses buffer b & 1 only after PV_b, P_{b+2} needs S_{b+2}, issued after PV_b.
    // "PV_b done" uses per-block barriers that complete once per task (blocks >= nb at the task's end): a parity
    // wait is only unambiguous with at most one phase outstanding.
    struct Bars {
        uint64_t* qk;      // [2] QK done, per S buffer
        uint64_t* pvc;     // [kMaxBlocks] PV_b done
        uint64_t* tready;  // [2] tile ready, per buffer (8 arrivals)
        uint64_t* pready;  // [2] P (and O rescale) ready, per buffer (8 arrivals)
        uint32_t n_qk0 = 0, n_qk1 = 0, n_task = 0;            // compute side
        uint32_t n_tr0 = 0, n_tr1 = 0, n_pr0 = 0, n_pr1 = 0, n_itask = 0;  // issuer side
    };
    // compute warps: this warp's writes (smem: fence.proxy.async done by the writer; TMEM: tcgen05.wait::st) are
    // complete -> one arrival
    __device__ static __forceinline__ void signal(uint64_t* bar, int lane) {
        tc_fence_before();
        __syncwarp();
        if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
    }

    // ---- K rows of block bb (data + scales) -> raw staging in tile buf, coalesced: one row per warp instruction
    __device__ static void issue_rows(uint8_t* smem, int bb, uint8_t* stage, int warp, int lane) {
        const uint8_t* fmt = smem + kFmtOff + bb * 64;
        const uint64_t* tab = reinterpret_cast<const uint64_t*>(smem + kTabOff) + bb * 64;
        const uint64_t* stab = reinterpret_cast<const uint64_t*>(smem + kSTabOff) + bb * 64;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = warp + 8 * i;
            const uint8_t f = fmt[r];
            const uint8_t* data = reinterpret_cast<const uint8_t*>(tab[r]);
            const uint8_t* scale = reinterpret_cast<const uint8_t*>(stab[r]);
            uint8_t* dst = stage + r * kRawStride;
            if (f == AttnSeq::kFp8) {
                cp_async16(dst + 16 * lane, data + 16 * lane);  // 512 B
                if (lane == 0) cp_async16(dst + 512, scale);   // 16 ue8m0
            } else if (f == AttnSeq::kFp4) {
                if (lane < 16)
                    cp_async16(dst + 16 * lane, data + 16 * lane);  // 256 B
                else if (lane < 18)
                    cp_async16(dst + 256 + 16 * (lane - 16), scale + 16 * (lane - 16));  // 32 e4m3
            }
        }
    }

    // ---- dequant tile buf in place: all reads (registers), CTA barrier, all writes; warp w = atom w
    __device__ static void dequant(uint8_t* smem, int bb, int buf, const uint8_t* stage, int warp, int lane) {
        const uint8_t* fmt = smem + kFmtOff + bb * 64;
        uint8_t* tile = smem + kTileOff + buf * 65536;
        const int d0 = 64 * warp + 16 * (lane & 3);
        uint4 d[8];
        uint32_t sc[8];
        bool fp4[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = 8 * i + (lane >> 2);
            const uint8_t f = fmt[r];
            fp4[i] = f == AttnSeq::kFp4;
            const uint8_t* raw = stage + r * kRawStride;
            d[i] = make_uint4(0, 0, 0, 0);
            sc[i] = 0;
            if (f == AttnSeq::kFp8) {
                d[i] = *reinterpret_cast<const uint4*>(raw + d0);
                sc[i] = raw[512 + (d0 >> 5)];
            } else if (fp4[i]) {
                const uint2 v = *reinterpret_cast<const uint2*>(raw + d0 / 2);
                d[i].x = v.x;
                d[i].y = v.y;
                sc[i] = raw[256 + (d0 >> 4)];
            }
        }
        w8::compute_sync();  // every raw byte read before any tile byte is written
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = 8 * i + (lane >> 2);
            uint4 o0, o1;
            dequant16(d[i], fp4[i], sc[i], o0, o1);
            *reinterpret_cast<uint4*>(tile + AttnSeq::kv_off(r, d0)) = o0;
            *reinterpret_cast<uint4*>(tile + AttnSeq::kv_off(r, d0 + 8)) = o1;
        }
        fence_proxy_async_smem();
    }

    __device__ static void issue_qk(uint8_t* smem, int buf, uint32_t tmem_base, uint64_t* bar) {
        tc_fence_after();
        const uint32_t idesc = umma_idesc_bf16(128, 2 * NH, false, false);
        const uint32_t ka = smem_u32(smem + kTileOff + buf * 65536), qa = smem_u32(smem + kQOff);
#pragma unroll
        for (int s = 0; s < 16; ++s)
            tc_mma_ss(tmem_base + kTmemS + 64 * buf, umma_desc_sw128(ka + (s >> 2) * 16384 + (s & 3) * 32, 16, 1024),
                      umma_desc_sw128(qa + (s >> 2) * kQPair + (s & 3) * 32, 16, 1024), idesc, s > 0);
        tc_commit(bar);
    }

    __device__ static void issue_pv(uint8_t* smem, int buf, int b, int mtb, uint32_t tmem_base, uint64_t* bar) {
        tc_fence_after();
        const uint32_t idesc = umma_idesc_bf16(128, NH, true, false);
        const uint32_t va = smem_u32(smem + kTileOff + buf * 65536), pa = smem_u32(smem + kPOff + buf * 4096);
#pragma unroll
        for (int u = 0; u < T; ++u) {
            const int m = T * mtb + u;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
                tc_mma_ss(tmem_base + kTmemO + 32 * u, umma_desc_sw128(va + 2 * m * 8192 + ks * 2048, 8192, 1024),
                          umma_desc_sw128(pa + ks * 32, 16, 1024), idesc, (b > 0 || ks > 0));
        }
        tc_commit(bar);
    }

    // warp 8, lane 0: the task's MMAs in the compute warps' event order
    template <class Sync>
    __device__ static void issue_loop(const AttnParams& p, int task, Sync& sync, uint8_t* smem, uint32_t tmem_base,
                                      Bars& bars) {
        const int t = task / kTasksPerToken, rem = task - t * kTasksPerToken;
        const int mtb = rem - (rem / (4 / T)) * (4 / T);
        sync.wait_extern();
        const int nb = AttnSeq::num_blocks(p, t);
        auto wait_tile = [&](int buf) {
            if (buf == 0) {
                mbar_wait(bars.tready, bars.n_tr0 & 1);
                ++bars.n_tr0;
            } else {
                mbar_wait(bars.tready + 1, bars.n_tr1 & 1);
                ++bars.n_tr1;
            }
        };
        wait_tile(0);
        issue_qk(smem, 0, tmem_base, bars.qk);
        if (nb > 1) {
            wait_tile(1);
            issue_qk(smem, 1, tmem_base, bars.qk + 1);
        }
#pragma unroll 1
        for (int b = 0; b < nb; ++b) {
            const int buf = b & 1;
            if (buf == 0) {
                mbar_wait(bars.pready, bars.n_pr0 & 1);
                ++bars.n_pr0;
            } else {
                mbar_wait(bars.pready + 1, bars.n_pr1 & 1);
                ++bars.n_pr1;
            }
            issue_pv(smem, buf, b, mtb, tmem_base, bars.pvc + b);
            if (b + 2 < nb) {
                wait_tile(buf);
                issue_qk(smem, buf, tmem_base, bars.qk + buf);
            }
        }
#pragma unroll 1
        for (int b = nb; b < kMaxBlocks; ++b) tc_commit(bars.pvc + b);
        ++bars.n_itask;
    }

    template <class Sync>
    __device__ static void consume(const AttnParams& p, int task, Sync& sync, uint8_t* smem, uint32_t tmem_base,
                                   Bars& bars) {
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const int t = task / kTasksPerToken, rem = task - t * kTasksPerToken;
        const int g = rem / (4 / T), mtb = rem - g * (4 / T);
        const int h0 = NH * g;
        sync.wait_extern();
        const int nb = AttnSeq::num_blocks(p, t);
        const int nslots = nb * 64;
        uint8_t* fmt = smem + kFmtOff;
        uint64_t* tab = reinterpret_cast<uint64_t*>(smem + kTabOff);

        // ---- loads: two dependent chains run side by side, position -> cos/sin and slot index -> rows; Q
        //      (cp.async) is issued while the first round trip flies
        int spos = 0;
        if (tid < 64) spos = load_position(p, t);
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
        float cs_v = 0.f, sink_v = 0.f;
        if (tid < 64)
            cs_v = __ldg(p.cos_sin + (int64_t)spos * kRopeDim + tid);
        else if (tid < 96)
            sink_v = __ldg(p.sink + h0 + tid - 64);
        {
            const uint16_t* qt = p.q + (int64_t)t * p.q_tok_stride;
            uint8_t* qs = smem + kQOff;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int u = tid + 256 * k;  // 16 B unit (dc, head, half) of our 32 heads
                const int dc = u >> 6, hh = (u >> 1) & 31, half = u & 1;
                const int d0 = dc * 16 + half * 8;
                const uint32_t atom = d0 >> 6, rail = atom & 1, pair = atom >> 1;
                const uint32_t row = rail * NH + hh;
                cp_async16(qs + pair * kQPair + row * 128 + ((((d0 & 63) >> 3) ^ (row & 7)) << 4),
                           qt + (int64_t)(dc * kHeads + h0 + hh) * 16 + half * 8);
            }
        }
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int j = tid + 256 * k;
            if (j >= nslots) continue;
            const bool swa = j < p.swa_width;
            const int jj = swa ? j : j - p.swa_width;
            uint8_t f = AttnSeq::kInvalid;
            const uint8_t *data = nullptr, *scale = nullptr;
            if (jj < len[k] && idx[k] >= 0) {
                const int page_tokens = swa ? p.swa_page : p.comp_page;
                const int blk = idx[k] / page_tokens, off = idx[k] - blk * page_tokens;
                const uint8_t* page =
                    (swa ? p.swa : p.comp) + (int64_t)blk * (swa ? p.swa_block_stride : p.comp_block_stride);
                const int row = swa ? 512 : 256, srow = swa ? 16 : 32;
                f = swa ? AttnSeq::kFp8 : AttnSeq::kFp4;
                data = page + off * row;
                scale = page + (int64_t)page_tokens * row + off * srow;
            }
            fmt[j] = f;
            tab[j] = reinterpret_cast<uint64_t>(data);
            reinterpret_cast<uint64_t*>(smem + kSTabOff)[j] = reinterpret_cast<uint64_t>(scale);
        }
#ifdef W1_DBG_LOADS
        cta_stamp(p, 2);  // thread 0's slot indices have returned
#endif
        w8::compute_sync();  // tables
#ifdef W1_DBG_LOADS
        cta_stamp(p, 3);
#endif
        // rows + scales of blocks 0 and 1 (one commit group each)
        issue_rows(smem, 0, smem + kTileOff, warp, lane);
        asm volatile("cp.async.commit_group;" ::: "memory");
        if (nb > 1) issue_rows(smem, 1, smem + kTileOff + 65536, warp, lane);
        asm volatile("cp.async.commit_group;" ::: "memory");
        if (nb > 2) issue_rows(smem, 2, smem + kStage2Off, warp, lane);
        asm volatile("cp.async.commit_group;" ::: "memory");
        if (tid < 64)
            reinterpret_cast<float*>(smem + kCsOff)[tid] = cs_v;
        else if (tid < 96)
            reinterpret_cast<float*>(smem + kSinkOff)[tid - 64] = sink_v;
#ifdef W1_DBG_LOADS
        cta_stamp(p, 4);
#else
        cta_stamp(p, 2);
#endif

        // ---- block 0: Q RoPE + dequant, QK_0; block 1: dequant, QK_1
        asm volatile("cp.async.wait_group 2;" ::: "memory");
        w8::compute_sync();
#ifdef W1_DBG_LOADS
        cta_stamp(p, 5);
#endif
        {
            // FlashMLA's q RoPE on dims 448..511 = atom 7 = pair 3, rail 1 rows (32 + h): 256 units of 16 B
            const float* cs = reinterpret_cast<const float*>(smem + kCsOff);
            const int r = NH + (tid >> 3), cphys = tid & 7, c = cphys ^ (r & 7);
            uint4* ptr = reinterpret_cast<uint4*>(smem + kQOff + 3 * kQPair + r * 128 + cphys * 16);
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
        dequant(smem, 0, 0, smem + kTileOff, warp, lane);
        fence_proxy_async_smem();
        signal(bars.tready, lane);
        if (nb > 1) {
            asm volatile("cp.async.wait_group 1;" ::: "memory");
            w8::compute_sync();
            dequant(smem, 1, 1, smem + kTileOff + 65536, warp, lane);
            signal(bars.tready + 1, lane);
        }
#ifdef W1_DBG_LOADS
        cta_stamp(p, 6);
        return;
#endif
        cta_stamp(p, 3);

        // ---- the block loop, FlashMLA's order
        const int q4 = warp & 3, hh = warp >> 2;
        const bool rail0 = q4 < 2;
        const int key = 32 * (q4 & 1) + lane;
        const uint32_t lane_base = (uint32_t)(32 * q4) << 16;
        const float mult = p.sm_scale_log2;
        float* xch = reinterpret_cast<float*>(smem + kXchOff);
        float* red = reinterpret_cast<float*>(smem + kRedOff);
        float mi = kMaxInit, li = 0.f, real_mi = -INFINITY;  // state threads tid < 64: head tid & 31, key half tid >> 5
#pragma unroll 1
        for (int b = 0; b < nb; ++b) {
            const int buf = b & 1;
            // S_b landed
            if (buf == 0) {
                mbar_wait(bars.qk, bars.n_qk0 & 1);
                ++bars.n_qk0;
            } else {
                mbar_wait(bars.qk + 1, bars.n_qk1 & 1);
                ++bars.n_qk1;
            }
            tc_fence_after();
            // ---- S = rail0 + rail1 into xch[key][head] (-inf for invalid keys)
            float* xrow = xch + key * kXS + 16 * hh;
            {
                uint32_t v[16];
                tc_ld_32x32b<16>(tmem_base + kTmemS + 64 * buf + lane_base + (rail0 ? 0 : NH) + 16 * hh, v);
                tc_wait_ld();
                if (!rail0) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) xrow[i] = __uint_as_float(v[i]);
                }
                w8::compute_sync();
                if (rail0) {
                    const bool kvalid = fmt[b * 64 + key] != AttnSeq::kInvalid;
                    float r1[16];
#pragma unroll
                    for (int i = 0; i < 16; ++i) r1[i] = xrow[i];
#pragma unroll
                    for (int i = 0; i < 16; ++i) xrow[i] = kvalid ? __uint_as_float(v[i]) + r1[i] : -INFINITY;
                }
            }
            w8::compute_sync();
            // ---- FlashMLA's lazy rescale in the 64 state threads (head h, key half kh): each takes both halves' raw
            //      maxima straight from xch (max is order-free), so no extra barrier
            float scale_old = 1.f, m_new = 0.f;
            bool scale_o;
            {
                const int h = tid & 31, kh = (tid >> 5) & 1;
                float cmax = -INFINITY;
                if (tid < 64) {
                    float v[32], w[32];
#pragma unroll
                    for (int k = 0; k < 32; ++k) {
                        v[k] = xch[(32 * kh + k) * kXS + h];
                        w[k] = xch[(32 * (kh ^ 1) + k) * kXS + h];
                    }
#pragma unroll
                    for (int s2 = 16; s2 >= 1; s2 >>= 1)
#pragma unroll
                        for (int k = 0; k < s2; ++k) {
                            v[k] = fmaxf(v[k], v[k + s2]);
                            w[k] = fmaxf(w[k], w[k + s2]);
                        }
                    // FlashMLA: own half max * mult, then max with the other half's (both pre-multiplied)
                    cmax = fmaxf(v[0] * mult, w[0] * mult);
                    real_mi = fmaxf(real_mi, cmax);
                }
                const bool want = tid < 64 && (cmax - mi > 6.0f);
                scale_o = bar_red_or(2, 256, want);
                if (tid < 64) {
                    if (scale_o) {
                        m_new = fmaxf(cmax, mi);
                        scale_old = ex2_approx(mi - m_new);
                    } else {
                        m_new = mi;
                    }
                    mi = m_new;
                    if (tid < 32) {
                        red[2 * NH + h] = m_new;
                        red[3 * NH + h] = scale_old;
                    }
                }
            }
            w8::compute_sync();
            // ---- P = bf16(exp2(fma(s, mult, -m))) -> P^T image (buffer b & 1). Warp w: key 32 (w & 1) + lane, heads
            //      8 (w >> 1) .. +8. The l sums are taken by the state threads from S and m directly (same d, bit for
            //      bit), so P does not round-trip through xch.
            {
                const int hb8 = 8 * (warp >> 1);
                const int pkey = 32 * (warp & 1) + lane;
                const float* prow = xch + pkey * kXS + hb8;
                uint8_t* pt = smem + kPOff + buf * 4096;
                float x[8], d[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) x[e] = prow[e];
#pragma unroll
                for (int e = 0; e < 8; ++e) d[e] = ex2_approx(fmaf(x[e], mult, -red[2 * NH + hb8 + e]));
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int h = hb8 + e;
                    *reinterpret_cast<__nv_bfloat16*>(pt + h * 128 + (((pkey >> 3) ^ (h & 7)) << 4) + (pkey & 7) * 2) =
                        __float2bfloat16_rn(d[e]);
                }
                fence_proxy_async_smem();
            }
            // l: FlashMLA's per-thread sum = (even keys chain) + (odd keys chain) over the thread's 32 keys
            if (tid < 64) {
                const int h = tid & 31, kh = tid >> 5;
                const float* xc = xch + (32 * kh) * kXS + h;
                float d[32];
#pragma unroll
                for (int i = 0; i < 32; ++i) d[i] = ex2_approx(fmaf(xc[i * kXS], mult, -m_new));
                float sx = 0.f, sy = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    sx = sx + d[2 * i];
                    sy = sy + d[2 * i + 1];
                }
                li = fmaf(li, scale_old, sx + sy);
            }
            // ---- rescale O (all 128 TMEM lanes = dims, this thread's 16 heads) once PV_{b-1} has landed
            const uint32_t ph = bars.n_task & 1;
            if (scale_o && b > 0) {
                mbar_wait(bars.pvc + b - 1, ph);
                tc_fence_after();
                if (warp < 8) {
                    const float* f = red + 3 * NH + 16 * hh;
#pragma unroll
                    for (int u = 0; u < T; ++u) {
                        uint32_t v[16];
                        const uint32_t ta = tmem_base + kTmemO + 32 * u + lane_base + 16 * hh;
                        tc_ld_32x32b<16>(ta, v);
                        tc_wait_ld();
#pragma unroll
                        for (int i = 0; i < 16; ++i) v[i] = __float_as_uint(__uint_as_float(v[i]) * f[i]);
                        tc_st_32x32b<16>(ta, v);
                    }
                    tc_wait_st();
                }
            }
            // ---- O^T += V^T P^T (issued by warp 8 once every warp's P and O rescale are done)
            signal(bars.pready + buf, lane);
            // ---- block b + 2 reuses tile / S / P buffer b & 1 once PV_b is done. Its raw rows were prefetched into
            //      the stage slot one block earlier; once they are read, block b + 3's rows go into the slot.
            if (b + 2 < nb) {
                mbar_wait(bars.pvc + b, ph);
                tc_fence_after();
                cp_async_wait_all();
                w8::compute_sync();
                dequant(smem, b + 2, buf, smem + kStage2Off, warp, lane);  // CTA barrier between reads and writes
                if (b + 3 < nb) issue_rows(smem, b + 3, smem + kStage2Off, warp, lane);
                asm volatile("cp.async.commit_group;" ::: "memory");
                signal(bars.tready + buf, lane);
            }
            w8::compute_sync();  // xch / red are rewritten by the next block
        }
        mbar_wait(bars.pvc + nb - 1, bars.n_task & 1);
        ++bars.n_task;
        tc_fence_after();
        cta_stamp(p, 4);

        // ---- epilogue: FlashMLA's store_o for our 32 heads x 128 T dims
        float* lbuf = reinterpret_cast<float*>(smem + kLOff);
        if (tid < 64) {
            if (real_mi == -INFINITY) {
                li = 0.f;
                mi = -INFINITY;
            }
            lbuf[(tid >> 5) * NH + (tid & 31)] = li;
            if (tid < 32) reinterpret_cast<float*>(smem + kMOff)[tid] = mi;
        }
        w8::compute_sync();
        if (tid < NH) {
            const float l = lbuf[tid] + lbuf[NH + tid];
            const float m = reinterpret_cast<float*>(smem + kMOff)[tid];
            const float sink_l2 = reinterpret_cast<const float*>(smem + kSinkOff)[tid] * kL2E;
            float osc;
            asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(osc) : "f"(1.0f), "f"(l + ex2_approx(sink_l2 - m)));
            reinterpret_cast<float*>(smem + kOscOff)[tid] = l == 0.f ? 0.f : osc;
        }
        w8::compute_sync();
        {
            const float* cs = reinterpret_cast<const float*>(smem + kCsOff);
            const float* oscv = reinterpret_cast<const float*>(smem + kOscOff);
            float* E = reinterpret_cast<float*>(smem + kEOff) + warp * (16 * 33);
            float* MU = reinterpret_cast<float*>(smem + kMuOff) + warp * 16;
            const int hb = 16 * hh;
#pragma unroll 1
            for (int u = 0; u < T; ++u) {
                const int m = T * mtb + u;
                uint8_t* stage = smem + kStageOff + u * 4096;
                const int dim = 128 * m + 32 * q4 + lane;
                const int j = dim >> 5;
                const bool rope = dim >= kHeadDim - kRopeDim;  // warp-uniform
                uint32_t v[16];
                tc_ld_32x32b<16>(tmem_base + kTmemO + 32 * u + lane_base + hb, v);
                tc_wait_ld();
                float y[16];
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    y[e] = __uint_as_float(v[e]);
                    E[e * 33 + lane] = y[e];
                }
                __syncwarp();
                if (rope) {
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
                    const int h = hb + lane;
                    const float* row = E + lane * 33;
                    float a[32];
#pragma unroll
                    for (int k = 0; k < 32; ++k) a[k] = fabsf(row[k]);
#pragma unroll
                    for (int w = 16; w >= 1; w >>= 1)
#pragma unroll
                        for (int k = 0; k < w; ++k) a[k] = fmaxf(a[k], a[k + w]);
                    const float osc = oscv[h];
                    const float amax = fmaxf(1e-4f, a[0] * osc);
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
                __syncwarp();
            }
        }
        w8::compute_sync();
        {
            // T x 4 wv groups x 1 KB, each contiguous in the output
#pragma unroll
            for (int u = 0; u < T; ++u) {
                const int m = T * mtb + u;
                const int grp = tid >> 6, off = (tid & 63) * 16;
                uint8_t* dst = p.out_fp8 + (int64_t)t * (kHeads * kHeadDim) + ((h0 >> 3) + grp) * 4096 + m * 1024 + off;
                *reinterpret_cast<uint4*>(dst) =
                    *reinterpret_cast<const uint4*>(smem + kStageOff + u * 4096 + grp * 1024 + off);
            }
        }
        cta_stamp(p, 5);
        tc_fence_before();
    }
};

}  // namespace w1
