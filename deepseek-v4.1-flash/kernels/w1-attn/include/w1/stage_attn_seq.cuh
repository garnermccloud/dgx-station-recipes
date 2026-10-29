// W1 bitwise mode: FlashMLA mega-attention decode arithmetic, reproduced operation for operation, parallel over
// (token, 32-head decision group) instead of one CTA per token.
//
// What makes FlashMLA's output a function of more than the math (and what this stage reproduces exactly):
//   * QK: two "rails" = the even and the odd 64-dim atoms of the 512 dims, each a 16-step K=16 tensor-core chain in
//     ascending dim order; S = rail0 + rail1 (fp32).
//   * 64-key blocks in unified slot order [window | compressed], invalid slots -inf; block max (raw) * multiplier;
//     lazy rescale: rescale only if ANY of the 32 heads of the group exceeds its reference by > 6 (FlashMLA's
//     __any_sync over a 32-head warp); P = bf16(exp2(fma(s, mult, -m))); l per (head, key half) =
//     fmaf(l, f, (sum of even keys) + (sum of odd keys)), each a sequential chain; l = l_half0 + l_half1.
//   * O accumulated in TMEM across blocks: K=16 tensor-core steps in key order, rescale = fp32 multiply.
//   * Epilogue: conjugate RoPE on the unnormalized O, amax * (1 / (l + exp2(sink*log2e - m))) via div.approx,
//     ue8m0 by FlashMLA's bit trick, cvt.rn.satfinite e4m3.
// Task = (token, group of 32 heads); every key block runs in order inside the task. No partials, no combine.
#pragma once
#include <type_traits>
#include "attn_params.cuh"
#include "ptx.cuh"
#include "../w8/w8_shim.cuh"

namespace w1 {

// transposed warp reduction of 16 per-lane values (one per head): after 4 halving steps + 1 butterfly each lane holds
// the result for head (lane >> 1) & 15.
template <bool kMax>
__device__ __forceinline__ float warp_reduce16(const float (&v)[16], int lane) {
    auto op = [](float a, float b) { return kMax ? fmaxf(a, b) : a + b; };
    float k8[8], k4[4], k2[2];
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4, b1 = lane & 2;
#pragma unroll
    for (int j = 0; j < 8; ++j) k8[j] = op(b4 ? v[8 + j] : v[j], __shfl_xor_sync(0xffffffffu, b4 ? v[j] : v[8 + j], 16));
#pragma unroll
    for (int j = 0; j < 4; ++j) k4[j] = op(b3 ? k8[4 + j] : k8[j], __shfl_xor_sync(0xffffffffu, b3 ? k8[j] : k8[4 + j], 8));
#pragma unroll
    for (int j = 0; j < 2; ++j) k2[j] = op(b2 ? k4[2 + j] : k4[j], __shfl_xor_sync(0xffffffffu, b2 ? k4[j] : k4[2 + j], 4));
    float k1 = op(b1 ? k2[1] : k2[0], __shfl_xor_sync(0xffffffffu, b1 ? k2[0] : k2[1], 2));
    k1 = op(k1, __shfl_xor_sync(0xffffffffu, k1, 1));
    return k1;  // head = 8*b4 + 4*b3 + 2*b2 + b1 = (lane >> 1) & 15
}

struct AttnSeq {
    using Params = AttnParams;
    static constexpr const char* kName = "attn_seq";
    static constexpr int NH = 32;      // heads per task = FlashMLA's lazy-rescale decision group
    static constexpr int kHalf = 16;   // heads per thread (warps 0-3: heads 0-15, warps 4-7: heads 16-31)
    static constexpr int kXS = 36;     // xch row stride (floats)

    // ---- SMEM (bytes; tiles 1024-aligned)
    static constexpr uint32_t kKvOff = 0;                 // [8 atoms][64 keys][128 B] bf16 (FlashMLA's own layout)
    static constexpr uint32_t kQOff = 65536;              // [4 atom pairs][2 rails x 32 heads][128 B]
    static constexpr uint32_t kQPairStride = 2 * NH * 128;
    static constexpr uint32_t kPOff = kQOff + 4 * kQPairStride;   // P^T: [32 heads][64 keys] bf16, one atom column
    static constexpr uint32_t kXchOff = kPOff + NH * 128;          // [64 keys][kXS] fp32: rail-1 S, then d (fp32 P)
    static constexpr uint32_t kTabOff = kXchOff + 64 * kXS * 4;    // [64] data pointers
    static constexpr uint32_t kSclOff = kTabOff + 64 * 8;          // [64][32] scale bytes
    static constexpr uint32_t kFmtOff = kSclOff + 64 * 32;         // [64]
    static constexpr uint32_t kRedOff = kFmtOff + 64;              // [2 key halves][32] block max, [32] m, [32] f
    static constexpr uint32_t kLOff = kRedOff + 4 * NH * 4;        // [2 key halves][32] l
    static constexpr uint32_t kMOff = kLOff + 2 * NH * 4;          // [32] final m
    static constexpr uint32_t kOscOff = kMOff + NH * 4;            // [32] out scale
    static constexpr uint32_t kBarOff = kOscOff + NH * 4;
    static constexpr uint32_t kSmemBytes = kBarOff + 8;
    static constexpr uint32_t kOutStageOff = 0;  // epilogue: fp8 of the 4 wv groups (16 KB), reuses the KV tile

    static constexpr uint32_t kTmemS = 0;    // S^T: 128 lanes (rail x key) x 64 cols (rail x head)
    static constexpr uint32_t kTmemO = 64;   // O^T: 4 tiles x 32 cols
    static constexpr uint32_t kTmemCols = 256;

    enum : uint8_t { kInvalid = 0, kFp8 = 1, kFp4 = 2 };

    __host__ __device__ static constexpr uint32_t smem_bytes(int) { return kSmemBytes; }
    __host__ __device__ static constexpr uint32_t tmem_cols(int) { return kTmemCols; }
    __host__ __device__ static int num_tasks(const Params& p, int S) { return S * (kHeads / NH); }

    __device__ static __forceinline__ uint32_t kv_off(uint32_t k, uint32_t d) {  // FlashMLA kv_slot layout
        return (d >> 6) * 8192u + k * 128u + ((((d & 63u) >> 3) ^ (k & 7u)) << 4) + ((d & 7u) << 1);
    }

    // slot j of token t in FlashMLA decode's unified space: [0, swa_width) window, then compressed
    __device__ static void resolve_slot(const Params& p, int t, int j, const uint8_t*& data, const uint8_t*& scale,
                                        uint8_t& fmt) {
        data = nullptr;
        scale = nullptr;
        fmt = kInvalid;
        const bool swa = j < p.swa_width;
        const int jj = swa ? j : j - p.swa_width;
        if (!swa && (p.comp == nullptr || jj >= p.comp_width)) return;
        const int* idx_p = swa ? p.swa_idx + (int64_t)t * p.swa_idx_stride + jj
                               : p.comp_idx + (int64_t)t * p.comp_idx_stride + jj;
        const int idx = __ldg(idx_p);
        const int len = __ldg((swa ? p.swa_lens : p.comp_lens) + t);
        if (jj >= len || idx < 0) return;
        const int page_tokens = swa ? p.swa_page : p.comp_page;
        const int blk = idx / page_tokens, off = idx - blk * page_tokens;
        const uint8_t* page = (swa ? p.swa : p.comp) + (int64_t)blk * (swa ? p.swa_block_stride : p.comp_block_stride);
        const int row = swa ? 512 : 256, srow = swa ? 16 : 32;
        data = page + off * row;
        scale = page + (int64_t)page_tokens * row + off * srow;
        fmt = swa ? kFp8 : kFp4;
    }

    // FlashMLA decode: num_kv_blocks = ceil(topk + extra_len, 64) with the extra cache, ceil(topk_len, 64) without
    __device__ static int num_blocks(const Params& p, int t) {
        int nb;
        if (p.comp != nullptr)
            nb = (p.swa_width + __ldg(p.comp_lens + t) + 63) / 64;
        else
            nb = (__ldg(p.swa_lens + t) + 63) / 64;
        return nb < 1 ? 1 : nb;
    }

    // ---- raw rows of one block: pass i = row (warp + 8 i); lane owns dims [16 lane, +16)
    struct Raw {
        uint4 d[8];
    };
    __device__ static void build_table(const Params& p, uint8_t* smem, int t, int b, int tid) {
        if (tid < 64) {
            const uint8_t *data, *scale;
            uint8_t f;
            resolve_slot(p, t, b * 64 + tid, data, scale, f);
            reinterpret_cast<uint64_t*>(smem + kTabOff)[tid] = reinterpret_cast<uint64_t>(data);
            smem[kFmtOff + tid] = f;
            uint8_t* sdst = smem + kSclOff + tid * 32;
            if (f == kInvalid) {
                *reinterpret_cast<uint4*>(sdst) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(sdst + 16) = make_uint4(0, 0, 0, 0);
            } else {
                cp_async16(sdst, scale);
                if (f == kFp4) cp_async16(sdst + 16, scale + 16);
            }
        }
    }
    __device__ static void load_raw(const uint8_t* smem, int warp, int lane, Raw& raw) {
        const uint64_t* tab = reinterpret_cast<const uint64_t*>(smem + kTabOff);
        const uint8_t* fmt = smem + kFmtOff;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = warp + 8 * i;
            const uint8_t f = fmt[r];
            raw.d[i] = make_uint4(0, 0, 0, 0);
            if (f == kFp8) {
                raw.d[i] = __ldg(reinterpret_cast<const uint4*>(tab[r]) + lane);
            } else if (f == kFp4) {
                const uint2 v = __ldg(reinterpret_cast<const uint2*>(tab[r]) + lane);
                raw.d[i].x = v.x;
                raw.d[i].y = v.y;
            }
        }
    }
    __device__ static void convert_store(uint8_t* smem, int warp, int lane, const Raw& raw) {
        const uint8_t* fmt = smem + kFmtOff;
        const uint8_t* scl = smem + kSclOff;
        uint8_t* kv = smem + kKvOff;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = warp + 8 * i;
            const bool fp4 = fmt[r] == kFp4;
            const uint32_t sc = scl[r * 32 + (fp4 ? lane : (lane >> 1))];
            uint4 o0, o1;
            if (fp4) {
                uint32_t s0, s1;
                e4m3x4_to_bf16x2x2(sc, s0, s1);
                const uint32_t s = __byte_perm(s0, 0, 0x1010);
                uint32_t h[4];
                e2m1x8_to_bf16x2x4(raw.d[i].x, h);
                o0 = make_uint4(hmul2_bf16(h[0], s), hmul2_bf16(h[1], s), hmul2_bf16(h[2], s), hmul2_bf16(h[3], s));
                e2m1x8_to_bf16x2x4(raw.d[i].y, h);
                o1 = make_uint4(hmul2_bf16(h[0], s), hmul2_bf16(h[1], s), hmul2_bf16(h[2], s), hmul2_bf16(h[3], s));
            } else {  // fp8 or invalid (zeros, scale 0 -> 0)
                const uint32_t s = ue8m0_to_bf16x2(sc);
                uint32_t a, bb;
                e4m3x4_to_bf16x2x2(raw.d[i].x, a, bb);
                o0.x = hmul2_bf16(a, s);
                o0.y = hmul2_bf16(bb, s);
                e4m3x4_to_bf16x2x2(raw.d[i].y, a, bb);
                o0.z = hmul2_bf16(a, s);
                o0.w = hmul2_bf16(bb, s);
                e4m3x4_to_bf16x2x2(raw.d[i].z, a, bb);
                o1.x = hmul2_bf16(a, s);
                o1.y = hmul2_bf16(bb, s);
                e4m3x4_to_bf16x2x2(raw.d[i].w, a, bb);
                o1.z = hmul2_bf16(a, s);
                o1.w = hmul2_bf16(bb, s);
            }
            *reinterpret_cast<uint4*>(kv + kv_off(r, 16 * lane)) = o0;
            *reinterpret_cast<uint4*>(kv + kv_off(r, 16 * lane + 8)) = o1;
        }
    }

    // Q of heads h0..h0+31 -> [pair][rail * 32 + h][128 B] with FlashMLA's RoPE (fp32 FMA, bf16 rn)
    __device__ static void load_q(const Params& p, uint8_t* smem, int t, int h0, int tid) {
        const uint16_t* qt = p.q + (int64_t)t * p.q_tok_stride;
        const int pos = load_position(p, t);
        const float* cs = p.cos_sin + (int64_t)pos * kRopeDim;
        uint8_t* qs = smem + kQOff;
        constexpr int kN = NH * 64 / 256;  // 8 x 16 B per thread
        uint4 v[kN];
#pragma unroll
        for (int k = 0; k < kN; ++k) {
            const int idx = tid + 256 * k;
            const int dc = idx / (NH * 2), hh = (idx % (NH * 2)) >> 1, half = idx & 1;
            v[k] = __ldg(reinterpret_cast<const uint4*>(qt + dc * (kHeads * 16) + (h0 + hh) * 16 + half * 8));
        }
#pragma unroll
        for (int k = 0; k < kN; ++k) {
            const int idx = tid + 256 * k;
            const int dc = idx / (NH * 2), hh = (idx % (NH * 2)) >> 1, half = idx & 1;
            const int d0 = dc * 16 + half * 8;
            if (d0 >= kHeadDim - kRopeDim) {
                uint32_t w[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = (d0 - (kHeadDim - kRopeDim)) / 2 + e;
                    const float c = __ldg(cs + i), s = __ldg(cs + 32 + i);
                    const float x0 = __uint_as_float(w[e] << 16), x1 = __uint_as_float(w[e] & 0xFFFF0000u);
                    w[e] = pack_bf16x2(fmaf(x0, c, (-x1) * s), fmaf(x0, s, x1 * c));
                }
                v[k] = make_uint4(w[0], w[1], w[2], w[3]);
            }
            const uint32_t atom = d0 >> 6, rail = atom & 1, pair = atom >> 1;
            const uint32_t row = rail * NH + hh;
            *reinterpret_cast<uint4*>(qs + pair * kQPairStride + row * 128 + ((((d0 & 63) >> 3) ^ (row & 7)) << 4)) = v[k];
        }
    }

    template <class Sync>
    __device__ static void consume(const Params& p, int S, int task, Sync& sync, w8::Ring&, uint8_t* smem,
                                   uint32_t tmem_base) {
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const int t = task / (kHeads / NH);
        const int h0 = (task - t * (kHeads / NH)) * NH;
        uint64_t* bar = reinterpret_cast<uint64_t*>(smem + kBarOff);
        uint32_t phase = 0;
        if (tid == 0) {
            mbar_init(bar, 1);
            fence_mbar_init();
        }
        sync.wait_extern();
        const int nb = num_blocks(p, t);
        const int q4 = warp & 3, hh = warp >> 2;
        const int hb = hh * kHalf;          // this thread's 16 heads (within the group)
        const bool rail0 = q4 < 2;
        const int key = 32 * (q4 & 1) + lane;
        const uint32_t lane_base = (uint32_t)(32 * q4) << 16;
        const float mult = p.sm_scale_log2;
        float* xch = reinterpret_cast<float*>(smem + kXchOff);
        float* red = reinterpret_cast<float*>(smem + kRedOff);
        float* lbuf = reinterpret_cast<float*>(smem + kLOff);
        // FlashMLA's per-(head, key half) state: thread (h_l = tid & 31, kh = (tid >> 5) & 1) of warps 0-1 owns head
        // h_l (+0) and warps 2-3 head h_l + ... -> use 64 "state threads": tid < 64: head = tid & 31, kh = tid >> 5
        float mi = kMaxInit, li = 0.f, real_mi = -INFINITY;  // valid in threads tid < 64
        const uint32_t idesc_qk = umma_idesc_bf16(128, 2 * NH, false, false);
        const uint32_t idesc_pv = umma_idesc_bf16(128, NH, true, false);
        const uint32_t kv_addr = smem_u32(smem + kKvOff), q_addr = smem_u32(smem + kQOff),
                       p_addr = smem_u32(smem + kPOff);

        build_table(p, smem, t, 0, tid);
        load_q(p, smem, t, h0, tid);
        if (tid < 64) cp_async_wait_all();
        w8::compute_sync();
        Raw raw;
        load_raw(smem, warp, lane, raw);

        for (int b = 0; b < nb; ++b) {
            convert_store(smem, warp, lane, raw);
            fence_proxy_async_smem();
            w8::compute_sync();
            // ---- QK: rails = even / odd atoms, K steps in ascending dim order
            if (tid == 0) {
                tc_fence_after();
#pragma unroll
                for (int s = 0; s < 16; ++s) {
                    const uint32_t koff = (s >> 2) * 16384 + (s & 3) * 32;
                    const uint32_t qoff = (s >> 2) * kQPairStride + (s & 3) * 32;
                    tc_mma_ss(tmem_base + kTmemS, umma_desc_sw128(kv_addr + koff, 16, 1024),
                              umma_desc_sw128(q_addr + qoff, 16, 1024), idesc_qk, s > 0);
                }
                tc_commit(bar);
            }
            // next block's table + raw rows (the K tile is still being read by the tensor core)
            const int fmt_valid_key = smem[kFmtOff + key];  // read this block's validity before the table is rebuilt
            mbar_wait(bar, phase);
            phase ^= 1;
            tc_fence_after();
            w8::compute_sync();  // everyone has read this block's fmt
            if (b + 1 < nb) build_table(p, smem, t, b + 1, tid);

            // ---- S = rail0 + rail1
            uint32_t sr[kHalf];
            tc_ld_32x32b<kHalf>(tmem_base + kTmemS + lane_base + (rail0 ? 0 : NH) + hb, sr);
            tc_wait_ld();
            float* xrow = xch + key * kXS + hb;
            if (!rail0) {
#pragma unroll
                for (int i = 0; i < kHalf; i += 4)
                    *reinterpret_cast<uint4*>(xrow + i) = make_uint4(sr[i], sr[i + 1], sr[i + 2], sr[i + 3]);
            }
            w8::compute_sync();
            float x[kHalf];
            if (rail0) {
                const bool kvalid = fmt_valid_key != kInvalid;
#pragma unroll
                for (int i = 0; i < kHalf; i += 4) {
                    const float4 o = *reinterpret_cast<const float4*>(xrow + i);
                    x[i] = kvalid ? __uint_as_float(sr[i]) + o.x : -INFINITY;
                    x[i + 1] = kvalid ? __uint_as_float(sr[i + 1]) + o.y : -INFINITY;
                    x[i + 2] = kvalid ? __uint_as_float(sr[i + 2]) + o.z : -INFINITY;
                    x[i + 3] = kvalid ? __uint_as_float(sr[i + 3]) + o.w : -INFINITY;
                }
                // per-head max over this warp's 32 keys (raw scores; max is order-free)
                const float mx = warp_reduce16<true>(x, lane);
                if ((lane & 1) == 0) red[(q4 & 1) * NH + hb + ((lane >> 1) & 15)] = mx;
            }
            w8::compute_sync();
            // ---- FlashMLA's lazy rescale, in the 64 state threads (head h, key half kh)
            float new_max = 0.f, scale_old = 1.f;
            bool scale_o = false;
            {
                const int h = tid & 31;
                float cmax = -INFINITY;
                if (tid < 64) {
                    // FlashMLA: own half max * mult, then max with the other half's (both pre-multiplied)
                    const float own = red[(tid >> 5) * NH + h] * mult, other = red[((tid >> 5) ^ 1) * NH + h] * mult;
                    cmax = fmaxf(own, other);
                    real_mi = fmaxf(real_mi, cmax);
                }
                const bool want = tid < 64 && (cmax - mi > 6.0f);
                scale_o = bar_red_or(2, 256, want);
                if (tid < 64) {
                    if (scale_o) {
                        new_max = fmaxf(cmax, mi);
                        scale_old = ex2_approx(mi - new_max);
                    } else {
                        new_max = mi;
                    }
                    mi = new_max;
                    red[2 * NH + h + (tid >> 5) * 0] = new_max;  // broadcast m (one writer per head is enough)
                    if (tid < 32) red[3 * NH + h] = scale_old;
                }
            }
            w8::compute_sync();
            // ---- P = bf16(exp2(fma(s, mult, -m))) and d (fp32) for the exact sums
            if (rail0) {
                uint8_t* pt = smem + kPOff;
#pragma unroll
                for (int i = 0; i < kHalf; ++i) {
                    const int h = hb + i;
                    const float d = ex2_approx(fmaf(x[i], mult, -red[2 * NH + h]));
                    *reinterpret_cast<__nv_bfloat16*>(pt + h * 128 + (((key >> 3) ^ (h & 7)) << 4) + (key & 7) * 2) =
                        __float2bfloat16_rn(d);
                    xch[key * kXS + h] = d;  // rail-1 partials are consumed; reuse as d[key][head]
                }
                fence_proxy_async_smem();
            }
            // rescale O (all 128 TMEM lanes = dims, this thread's 16 heads)
            if (scale_o && b > 0) {
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    uint32_t v[kHalf];
                    const uint32_t ta = tmem_base + kTmemO + lane_base + mt * NH + hb;
                    tc_ld_32x32b<kHalf>(ta, v);
                    tc_wait_ld();
#pragma unroll
                    for (int i = 0; i < kHalf; ++i)
                        v[i] = __float_as_uint(__uint_as_float(v[i]) * red[3 * NH + hb + i]);
                    tc_st_32x32b<kHalf>(ta, v);
                }
                tc_wait_st();
            }
            tc_fence_before();
            w8::compute_sync();
            // ---- O^T += V^T P^T (K steps of 16 keys in order)
            if (tid == 0) {
                tc_fence_after();
#pragma unroll
                for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                    for (int ks = 0; ks < 4; ++ks)
                        tc_mma_ss(tmem_base + kTmemO + mt * NH,
                                  umma_desc_sw128(kv_addr + 2 * mt * 8192 + ks * 2048, 8192, 1024),
                                  umma_desc_sw128(p_addr + ks * 32, 16, 1024), idesc_pv, (b > 0 || ks > 0));
                tc_commit(bar);
            }
            // l: FlashMLA's per-thread sum = (even keys chain) + (odd keys chain) over the thread's 32 keys
            if (tid < 64) {
                const int h = tid & 31, kh = tid >> 5;
                float sx = 0.f, sy = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    sx = sx + xch[(32 * kh + 2 * i) * kXS + h];
                    sy = sy + xch[(32 * kh + 2 * i + 1) * kXS + h];
                }
                li = fmaf(li, scale_old, sx + sy);
            }
            if (b + 1 < nb) {
                if (tid < 64) cp_async_wait_all();
            }
            mbar_wait(bar, phase);
            phase ^= 1;
            tc_fence_after();
            w8::compute_sync();  // tile, P, xch free; next table landed
            if (b + 1 < nb) load_raw(smem, warp, lane, raw);
        }

        // ---- epilogue: FlashMLA's store_o
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
            const float sink_l2 = __ldg(p.sink + h0 + tid) * kL2E;
            float osc;
            asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(osc) : "f"(1.0f), "f"(l + ex2_approx(sink_l2 - m)));
            reinterpret_cast<float*>(smem + kOscOff)[tid] = l == 0.f ? 0.f : osc;
        }
        w8::compute_sync();
        const int pos = load_position(p, t);
        const float* cs = p.cos_sin + (int64_t)pos * kRopeDim;
        uint8_t* stage = smem + kOutStageOff;  // [4 wv groups][16 chunks][8 heads][32] fp8 = this task's 16 KB
        uint8_t* sf_t = reinterpret_cast<uint8_t*>(p.out_sf + t * p.sf_stride_tok);
        const float* oscv = reinterpret_cast<const float*>(smem + kOscOff);
#pragma unroll 1
        for (int mt = 0; mt < 4; ++mt) {
            uint32_t v[kHalf];
            tc_ld_32x32b<kHalf>(tmem_base + kTmemO + lane_base + mt * NH + hb, v);
            tc_wait_ld();
            const int dim = 128 * mt + 32 * q4 + lane;
            const int j = dim >> 5;  // 32-dim group
            const bool rope = dim >= kHeadDim - kRopeDim;
            float cc = 0.f, sn = 0.f;
            if (rope) {
                const int i = (dim - (kHeadDim - kRopeDim)) >> 1;
                cc = __ldg(cs + i);
                sn = -__ldg(cs + 32 + i);  // conjugate
            }
            float y[kHalf];
#pragma unroll
            for (int i = 0; i < kHalf; ++i) {
                y[i] = __uint_as_float(v[i]);
                if (rope) {
                    const float other = __shfl_xor_sync(0xffffffffu, y[i], 1);
                    const float x0 = (lane & 1) ? other : y[i], x1 = (lane & 1) ? y[i] : other;
                    y[i] = (lane & 1) ? fmaf(x0, sn, x1 * cc) : fmaf(x0, cc, (-x1) * sn);
                }
            }
            float ay[kHalf];
#pragma unroll
            for (int i = 0; i < kHalf; ++i) ay[i] = fabsf(y[i]);
            const float amax_h = warp_reduce16<true>(ay, lane);  // head hb + ((lane >> 1) & 15)
#pragma unroll
            for (int i = 0; i < kHalf; ++i) {
                const float amax_raw = __shfl_sync(0xffffffffu, amax_h, 2 * i);
                const int h = hb + i;
                const float osc = oscv[h];
                const float amax = fmaxf(1e-4f, amax_raw * osc);
                float sf;
                asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(sf) : "f"(amax), "f"(448.0f));
                const uint32_t sfb = __float_as_uint(sf);
                const int exp_sf = (int)((sfb - 1u) >> 23) + (1 - 127);
                const float sf_inv = __uint_as_float((uint32_t)(127 - exp_sf) << 23);
                const float mlt = osc * sf_inv;
                const uint8_t q8 = (uint8_t)(f32x2_to_e4m3x2(y[i] * mlt, 0.f) & 0xFF);
                const int grp = h >> 3, hin = h & 7;  // wv group within this task's 4
                stage[grp * 4096 + (j * 8 + hin) * 32 + lane] = q8;
                if (lane == 0) {
                    const int H = h0 + h, G = H >> 3, sfi = hin + j * 8;
                    sf_t[(G * p.sf_stride_grp + (sfi >> 2) * p.sf_stride_hd) * 4 + (sfi & 3)] = (uint8_t)(exp_sf + 127);
                }
            }
        }
        w8::compute_sync();
        uint8_t* out_t = p.out_fp8 + (int64_t)t * (kHeads * kHeadDim) + (h0 >> 3) * 4096;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int off = (tid + 256 * k) * 16;
            *reinterpret_cast<uint4*>(out_t + off) = *reinterpret_cast<const uint4*>(stage + off);
        }
        tc_fence_before();
    }
};

}  // namespace w1
