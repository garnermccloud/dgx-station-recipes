// W1 stage 1 of 2: split-KV partial attention for DeepSeek-V4.1 decode (W8 stage contract v1).
//
// Task = (token t, key chunk c, head group g of NH=16 heads). A chunk is `nsub` (1..2) sub-blocks of 64 slots of the
// token's unified slot list [window slots (swa_width) | compressed top-k slots (comp_width)] -- FlashMLA's slot order
// and validity rule (position < length && index >= 0). Per task:
//   1. Q loads issued first; slot table from the indices; raw MXFP8 / NVFP4 rows of every sub-block -> registers (all
//      loads in flight, nothing waits on them early); Q GPT-J RoPE on dims 448.. (fp32 FMA -> bf16, as FlashMLA).
//   2. per sub-block: exact bf16 dequant into a "rail-folded" SW128 tile (rows 0-63 = keys x dims 0-255, rows 64-127 =
//      the same keys x dims 256-511) so S^T = K Q^T is 16 tcgen05.mma of M=128, N=2*NH, K=16: S = rail0 + rail1.
//      Masked softmax in the log2 domain with FlashMLA's lazy-rescale rule (rescale O only when a block max exceeds
//      the running reference by > 6); P^T bf16 -> smem; O^T += V^T P^T (M=128 dims x 4 tiles, N=NH, K=64).
//   3. O^T partial (fp32, [quad][dim][4 heads] so every thread stores 16 B) + (m, l) per head -> scratch.
// 256 compute threads, <= 96 KB SMEM, 128 TMEM columns, no __syncthreads, no griddepcontrol.
#pragma once
#include <type_traits>
#include "attn_params.cuh"
#include "ptx.cuh"
#include "../w8/w8_shim.cuh"

namespace w1 {

// Max / sum of 8 per-lane values (one per head) over the 32 lanes of a warp with 9 shuffles: halve the value set at
// each of the first three steps. Every lane of the group (lane & ~3) ends with the result for head (lane >> 2) & 7.
template <bool kMax>
__device__ __forceinline__ float warp_reduce8(const float (&v)[8], int lane) {
    auto op = [](float a, float b) { return kMax ? fmaxf(a, b) : a + b; };
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
    float k4[4], k2[2];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float r = __shfl_xor_sync(0xffffffffu, b4 ? v[j] : v[4 + j], 16);
        k4[j] = op(b4 ? v[4 + j] : v[j], r);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float r = __shfl_xor_sync(0xffffffffu, b3 ? k4[j] : k4[2 + j], 8);
        k2[j] = op(b3 ? k4[2 + j] : k4[j], r);
    }
    float k1 = op(b2 ? k2[1] : k2[0], __shfl_xor_sync(0xffffffffu, b2 ? k2[0] : k2[1], 4));
    k1 = op(k1, __shfl_xor_sync(0xffffffffu, k1, 2));
    k1 = op(k1, __shfl_xor_sync(0xffffffffu, k1, 1));
    return k1;
}

template <int NH, int NSUB_MAX>
struct AttnSplit {
    static_assert(NH == 16, "this build supports 16 heads per split task");
    static_assert(NSUB_MAX >= 1 && NSUB_MAX <= 2, "1 or 2 sub-blocks per task");
    using Params = AttnParams;
    static constexpr const char* kName = "attn_split";
    static constexpr int kHalf = NH / 2;  // heads per thread in the softmax / epilogue
    static constexpr int kXchStride = NH + 4;  // floats; 80 B rows keep the STS/LDS.128 conflict-free

    // ---- SMEM plan (bytes from the stage base; tiles 1024-aligned)
    static constexpr uint32_t kKvOff = 0;          // rail-folded K/V tile: [a' 4][rail 2][64 keys][128 B]
    static constexpr uint32_t kQOff = 65536;       // rail-folded Q tile:   [a' 4][2*NH rows][128 B]
    static constexpr uint32_t kQAStride = 2 * NH * 128;
    static constexpr uint32_t kPOff = kQOff + 4 * kQAStride;   // P^T tile: [NH rows][64 keys] bf16 (one atom col)
    static constexpr uint32_t kXchOff = kPOff + NH * 128;      // rail-1 partial scores [64 keys][kXchStride] fp32
    static constexpr uint32_t kTabOff = kXchOff + 64 * kXchStride * 4;  // per slot: data ptr, scale ptr
    static constexpr uint32_t kSclOff = kTabOff + 64 * NSUB_MAX * 16;   // per slot: 32 B of scales (fp8 uses 16)
    static constexpr uint32_t kFmtOff = kSclOff + 64 * NSUB_MAX * 32;
    static constexpr uint32_t kRedMaxOff = (kFmtOff + 64 * NSUB_MAX + 15) / 16 * 16;  // [2 key halves][NH]
    static constexpr uint32_t kRedSumOff = kRedMaxOff + 2 * NH * 4;
    static constexpr uint32_t kBarOff = kRedSumOff + 2 * NH * 4;
    static constexpr uint32_t kSmemBytes = kBarOff + 8;
    static_assert(kSmemBytes <= 96 * 1024, "stage SMEM budget");

    // ---- TMEM plan (columns from tmem_base)
    static constexpr uint32_t kTmemS = 0;    // S^T rail-folded: 128 lanes x 2*NH columns
    static constexpr uint32_t kTmemO = 64;   // O^T: 4 tiles x NH columns
    static constexpr uint32_t kTmemCols = 128;

    enum : uint8_t { kInvalid = 0, kFp8 = 1, kFp4 = 2 };
    enum : int { kSbFp8 = 0, kSbFp4 = 1, kSbMixed = 2 };  // format of a 64-slot sub-block

    __host__ __device__ static constexpr uint32_t smem_bytes(int) { return kSmemBytes; }
    __host__ __device__ static constexpr uint32_t tmem_cols(int) { return kTmemCols; }
    __host__ __device__ static int num_tasks(const Params& p, int S) { return S * p.n_chunks * p.head_groups; }

    // byte offset of (key k, dim d) in the rail-folded K/V tile
    __device__ static __forceinline__ uint32_t kv_off(uint32_t k, uint32_t d) {
        return ((d & 255u) >> 6) * 16384u + (d >> 8) * 8192u + k * 128u + ((((d & 63u) >> 3) ^ (k & 7u)) << 4) +
               ((d & 7u) << 1);
    }

    // ------------------------------------------------------------------------------------------------ slots
    // One dependent round trip: the index and the length are loaded together (FlashMLA's rule applied after).
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

    __host__ __device__ static int sub_kind(const Params& p, int slot_begin) {
        if (slot_begin + 64 <= p.swa_width) return kSbFp8;
        if (slot_begin >= p.swa_width) return kSbFp4;
        return kSbMixed;
    }

    // ------------------------------------------------------------------------------------------------ raw rows
    // A thread's share of one sub-block: pass i = row (warp + 8 i); lane l owns dims [16 l, 16 l + 16) of the row:
    // fp8 rows 16 B of data + ue8m0 byte l/2, fp4 rows 8 B of data + e4m3 byte l. Invalid rows load nothing (zeros).
    template <int KIND>
    struct Raw {
        typename std::conditional<KIND == kSbFp4, uint2, uint4>::type d[8];
    };

    template <int KIND>
    __device__ static void load_raw(const uint8_t* smem, int sb, int warp, int lane, Raw<KIND>& raw) {
        const uint64_t* tab = reinterpret_cast<const uint64_t*>(smem + kTabOff) + 64 * sb;
        const uint8_t* fmt = smem + kFmtOff + 64 * sb;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = warp + 8 * i;
            const uint8_t f = fmt[r];
            if constexpr (KIND == kSbFp4) {
                raw.d[i] = make_uint2(0, 0);
                if (f == kFp4) raw.d[i] = __ldg(reinterpret_cast<const uint2*>(tab[r]) + lane);
            } else {
                raw.d[i] = make_uint4(0, 0, 0, 0);
                if (f == kFp8) {
                    raw.d[i] = __ldg(reinterpret_cast<const uint4*>(tab[r]) + lane);
                } else if (KIND == kSbMixed && f == kFp4) {
                    const uint2 v = __ldg(reinterpret_cast<const uint2*>(tab[r]) + lane);
                    raw.d[i].x = v.x;
                    raw.d[i].y = v.y;
                }
            }
        }
    }

    // ---- exact conversions (16 dims of one row -> two 16 B bf16 chunks)
    __device__ static __forceinline__ void cvt_fp8_16(const uint4& d, uint32_t sc, uint4& o0, uint4& o1) {
        const uint32_t s = ue8m0_to_bf16x2(sc);
        uint32_t a, b;
        e4m3x4_to_bf16x2x2(d.x, a, b);
        o0.x = hmul2_bf16(a, s);
        o0.y = hmul2_bf16(b, s);
        e4m3x4_to_bf16x2x2(d.y, a, b);
        o0.z = hmul2_bf16(a, s);
        o0.w = hmul2_bf16(b, s);
        e4m3x4_to_bf16x2x2(d.z, a, b);
        o1.x = hmul2_bf16(a, s);
        o1.y = hmul2_bf16(b, s);
        e4m3x4_to_bf16x2x2(d.w, a, b);
        o1.z = hmul2_bf16(a, s);
        o1.w = hmul2_bf16(b, s);
    }
    // 16 e2m1 (8 B) with one e4m3 scale -> two 16 B bf16 chunks (exact: <= 2 + 4 significant bits)
    __device__ static __forceinline__ void cvt_fp4_16(uint32_t w0, uint32_t w1, uint32_t sc, uint4& o0, uint4& o1) {
        uint32_t s0, s1;
        e4m3x4_to_bf16x2x2(sc, s0, s1);
        const uint32_t s = __byte_perm(s0, 0, 0x1010);
        uint32_t h[4];
        e2m1x8_to_bf16x2x4(w0, h);
        o0.x = hmul2_bf16(h[0], s);
        o0.y = hmul2_bf16(h[1], s);
        o0.z = hmul2_bf16(h[2], s);
        o0.w = hmul2_bf16(h[3], s);
        e2m1x8_to_bf16x2x4(w1, h);
        o1.x = hmul2_bf16(h[0], s);
        o1.y = hmul2_bf16(h[1], s);
        o1.z = hmul2_bf16(h[2], s);
        o1.w = hmul2_bf16(h[3], s);
    }

    template <int KIND>
    __device__ static void convert_store(uint8_t* smem, int sb, int warp, int lane, const Raw<KIND>& raw) {
        const uint8_t* fmt = smem + kFmtOff + 64 * sb;
        const uint8_t* scl = smem + kSclOff + 64 * 32 * sb;
        uint8_t* kv = smem + kKvOff;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = warp + 8 * i;
            const bool row_fp4 = KIND == kSbFp4 || (KIND == kSbMixed && fmt[r] == kFp4);
            const uint32_t sc = scl[r * 32 + (row_fp4 ? lane : (lane >> 1))];
            uint4 o0, o1;
            // invalid rows were loaded as zeros with zero scales: they dequantize to 0
            if constexpr (KIND == kSbFp4) {
                cvt_fp4_16(raw.d[i].x, raw.d[i].y, sc, o0, o1);
            } else if constexpr (KIND == kSbFp8) {
                cvt_fp8_16(raw.d[i], sc, o0, o1);
            } else {
                if (row_fp4)
                    cvt_fp4_16(raw.d[i].x, raw.d[i].y, sc, o0, o1);
                else
                    cvt_fp8_16(raw.d[i], sc, o0, o1);
            }
            *reinterpret_cast<uint4*>(kv + kv_off(r, 16 * lane)) = o0;
            *reinterpret_cast<uint4*>(kv + kv_off(r, 16 * lane + 8)) = o1;
        }
    }

    // ---- Q of heads h0..h0+NH-1 (16 B per thread x 4), RoPE'd, bf16 -> rail-folded SW128 tile
    static constexpr int kQLoads = NH * 64 / 256;
    __device__ static __forceinline__ void q_coords(int tid, int k, int& dc, int& hh, int& half) {
        const int idx = tid + 256 * k;
        dc = idx / (NH * 2);
        hh = (idx % (NH * 2)) >> 1;
        half = idx & 1;
    }
    __device__ static void issue_q(const Params& p, int t, int h0, int tid, uint4 (&v)[kQLoads]) {
        const uint16_t* qt = p.q + (int64_t)t * p.q_tok_stride;
#pragma unroll
        for (int k = 0; k < kQLoads; ++k) {
            int dc, hh, half;
            q_coords(tid, k, dc, hh, half);
            v[k] = __ldg(reinterpret_cast<const uint4*>(qt + dc * (kHeads * 16) + (h0 + hh) * 16 + half * 8));
        }
    }
    __device__ static void store_q(const Params& p, uint8_t* smem, int t, int tid, uint4 (&v)[kQLoads]) {
        const int pos = load_position(p, t);
        const float* cs = p.cos_sin + (int64_t)pos * kRopeDim;
        uint8_t* qs = smem + kQOff;
#pragma unroll
        for (int k = 0; k < kQLoads; ++k) {
            int dc, hh, half;
            q_coords(tid, k, dc, hh, half);
            const int d0 = dc * 16 + half * 8;
            if (d0 >= kHeadDim - kRopeDim) {
                uint32_t w[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = (d0 - (kHeadDim - kRopeDim)) / 2 + e;
                    const float c = __ldg(cs + i), s = __ldg(cs + 32 + i);
                    const float x0 = __uint_as_float(w[e] << 16), x1 = __uint_as_float(w[e] & 0xFFFF0000u);
                    const float y0 = fmaf(x0, c, (-x1) * s);
                    const float y1 = fmaf(x0, s, x1 * c);
                    w[e] = pack_bf16x2(y0, y1);
                }
                v[k] = make_uint4(w[0], w[1], w[2], w[3]);
            }
            const uint32_t row = hh + NH * (d0 >> 8);
            const uint32_t dd = d0 & 255;
            *reinterpret_cast<uint4*>(qs + (dd >> 6) * kQAStride + row * 128 + ((((dd & 63) >> 3) ^ (row & 7)) << 4)) =
                v[k];
        }
    }

    // ------------------------------------------------------------------------------------------------ consume
    template <class Sync>
    __device__ static void consume(const Params& p, int S, int task, Sync& sync, w8::Ring&, uint8_t* smem,
                                   uint32_t tmem_base) {
        const int G = p.head_groups;
        const int t = task / (p.n_chunks * G);
        const int c = (task - t * (p.n_chunks * G)) / G;
        const int slot0 = c * p.nsub * kBlockKeys;
        const int k0 = sub_kind(p, slot0), k1 = p.nsub > 1 ? sub_kind(p, slot0 + kBlockKeys) : k0;
        if (k0 == kSbFp8 && k1 == kSbFp8)
            body<kSbFp8, kSbFp8>(p, task, sync, smem, tmem_base);
        else if (k0 == kSbFp4 && k1 == kSbFp4)
            body<kSbFp4, kSbFp4>(p, task, sync, smem, tmem_base);
        else
            body<kSbMixed, kSbMixed>(p, task, sync, smem, tmem_base);
    }

    template <int K0, int K1, class Sync>
    __device__ static __noinline__ void body(const Params& p, int task, Sync& sync, uint8_t* smem,
                                             uint32_t tmem_base) {
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const int G = p.head_groups;
        const int t = task / (p.n_chunks * G);
        const int rem = task - t * (p.n_chunks * G);
        const int c = rem / G, g = rem - c * G;
        const int h0 = g * NH;
        const int nsub = p.nsub;
        const int slot0 = c * nsub * kBlockKeys;
        uint64_t* bar = reinterpret_cast<uint64_t*>(smem + kBarOff);
        uint32_t phase = 0;
        int ts = 0;
        dbg_stamp(p, task, ts);

        if (tid == 0) {
            mbar_init(bar, 1);
            fence_mbar_init();
        }
        sync.wait_extern();  // q, indices, caches all come from earlier launches
        uint4 qv[kQLoads];
        issue_q(p, t, h0, tid, qv);

        // ---- 1. slot table, every raw row in flight, then Q
        const bool slot_thread = tid < nsub * kBlockKeys;
        if (slot_thread) {
            const uint8_t *data, *scale;
            uint8_t f;
            resolve_slot(p, t, slot0 + tid, data, scale, f);
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
        w8::compute_sync();
        dbg_stamp(p, task, ts);
        Raw<K0> raw0;
        Raw<K1> raw1;
        load_raw(smem, 0, warp, lane, raw0);
        if (NSUB_MAX > 1 && nsub > 1) load_raw(smem, 1, warp, lane, raw1);
        store_q(p, smem, t, tid, qv);
        if (slot_thread) cp_async_wait_all();
        w8::compute_sync();  // scales landed

        const int q4 = warp & 3, hh = warp >> 2;
        const int hb = hh * kHalf;
        const bool rail0 = q4 < 2;
        const int key = 32 * (q4 & 1) + lane;  // the key this thread's TMEM lane holds (rail 0 or rail 1)
        float m_ref[kHalf], l_run[kHalf];  // running reference / sum of this thread's heads hb..hb+7
        bool rescale = false;
#pragma unroll
        for (int h = 0; h < kHalf; ++h) m_ref[h] = kMaxInit;
#pragma unroll
        for (int i = 0; i < kHalf; ++i) l_run[i] = 0.f;
        const float cl2 = p.sm_scale_log2;
        const uint32_t idesc_qk = umma_idesc_bf16(128, 2 * NH, false, false);
        const uint32_t idesc_pv = umma_idesc_bf16(128, NH, true, false);
        const uint32_t kv_addr = smem_u32(smem + kKvOff), q_addr = smem_u32(smem + kQOff),
                       p_addr = smem_u32(smem + kPOff);
        float* red_max = reinterpret_cast<float*>(smem + kRedMaxOff);
        float* red_sum = reinterpret_cast<float*>(smem + kRedSumOff);
        float* xch = reinterpret_cast<float*>(smem + kXchOff);
        const uint32_t lane_base = (uint32_t)(32 * q4) << 16;

        convert_store<K0>(smem, 0, warp, lane, raw0);
        fence_proxy_async_smem();
        w8::compute_sync();
        dbg_stamp(p, task, ts);

#pragma unroll
        for (int sb = 0; sb < NSUB_MAX; ++sb) {
            if (sb >= nsub) break;
            // ---- S^T = K Q^T (rail-folded)
            if (tid == 0) {
                tc_fence_after();
#pragma unroll
                for (int s = 0; s < 16; ++s) {
                    const uint32_t koff = (s >> 2) * 16384 + (s & 3) * 32;
                    const uint32_t qoff = (s >> 2) * kQAStride + (s & 3) * 32;
                    tc_mma_ss(tmem_base + kTmemS, umma_desc_sw128(kv_addr + koff, 16, 1024),
                              umma_desc_sw128(q_addr + qoff, 16, 1024), idesc_qk, s > 0);
                }
                tc_commit(bar);
            }
            mbar_wait(bar, phase);
            phase ^= 1;
            tc_fence_after();
            dbg_stamp(p, task, ts);

            // ---- scores: rail-1 threads hand their half to rail-0 threads
            uint32_t sraw[kHalf];
            tc_ld_32x32b<kHalf>(tmem_base + kTmemS + lane_base + (rail0 ? 0 : NH) + hb, sraw);
            tc_wait_ld();
            float* xrow = xch + key * kXchStride + hb;
            if (!rail0) {
                *reinterpret_cast<uint4*>(xrow) = make_uint4(sraw[0], sraw[1], sraw[2], sraw[3]);
                *reinterpret_cast<uint4*>(xrow + 4) = make_uint4(sraw[4], sraw[5], sraw[6], sraw[7]);
            }
            w8::compute_sync();
            float x[kHalf];
            if (rail0) {
                const float4 a = *reinterpret_cast<const float4*>(xrow);
                const float4 b = *reinterpret_cast<const float4*>(xrow + 4);
                const float o[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                const bool kvalid = smem[kFmtOff + sb * kBlockKeys + key] != kInvalid;
#pragma unroll
                for (int i = 0; i < kHalf; ++i) x[i] = kvalid ? __uint_as_float(sraw[i]) + o[i] : -INFINITY;
                const float mx = warp_reduce8<true>(x, lane);
                if ((lane & 3) == 0) red_max[(q4 & 1) * NH + hb + (lane >> 2)] = mx;
            }
            w8::compute_sync();
            dbg_stamp(p, task, ts);
            // FlashMLA: block max (raw) * multiplier; rescale iff any head's max exceeds its reference by > 6
            float factor[kHalf], m_new_own[kHalf];
            {
                const float4 a0 = *reinterpret_cast<const float4*>(red_max + hb);
                const float4 a1 = *reinterpret_cast<const float4*>(red_max + hb + 4);
                const float4 b0 = *reinterpret_cast<const float4*>(red_max + NH + hb);
                const float4 b1 = *reinterpret_cast<const float4*>(red_max + NH + hb + 4);
                const float mb[8] = {fmaxf(a0.x, b0.x) * cl2, fmaxf(a0.y, b0.y) * cl2, fmaxf(a0.z, b0.z) * cl2,
                                     fmaxf(a0.w, b0.w) * cl2, fmaxf(a1.x, b1.x) * cl2, fmaxf(a1.y, b1.y) * cl2,
                                     fmaxf(a1.z, b1.z) * cl2, fmaxf(a1.w, b1.w) * cl2};
                bool own = false;
#pragma unroll
                for (int i = 0; i < kHalf; ++i) own |= (mb[i] - m_ref[i] > 6.0f);
                const bool need = bar_red_or(2, 256, own);
#pragma unroll
                for (int i = 0; i < kHalf; ++i) {
                    m_new_own[i] = need ? fmaxf(mb[i], m_ref[i]) : m_ref[i];
                    factor[i] = need ? ex2_approx(m_ref[i] - m_new_own[i]) : 1.0f;
                    m_ref[i] = m_new_own[i];
                }
                rescale = need && sb > 0;
            }
            float d[kHalf];
            if (rail0) {
                uint8_t* pt = smem + kPOff;
#pragma unroll
                for (int i = 0; i < kHalf; ++i) {
                    d[i] = ex2_approx(fmaf(x[i], cl2, -m_new_own[i]));
                    const int h = hb + i;
                    *reinterpret_cast<__nv_bfloat16*>(pt + h * 128 + (((key >> 3) ^ (h & 7)) << 4) + (key & 7) * 2) =
                        __float2bfloat16_rn(d[i]);
                }
                fence_proxy_async_smem();
            }
            if (rescale) {
                // every warp rescales its 32 TMEM lanes (dims) of each O^T tile, heads hb..hb+kHalf-1
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    uint32_t v[kHalf];
                    const uint32_t ta = tmem_base + kTmemO + lane_base + mt * NH + hb;
                    tc_ld_32x32b<kHalf>(ta, v);
                    tc_wait_ld();
#pragma unroll
                    for (int i = 0; i < kHalf; ++i) v[i] = __float_as_uint(__uint_as_float(v[i]) * factor[i]);
                    tc_st_32x32b<kHalf>(ta, v);
                }
                tc_wait_st();
            }
            tc_fence_before();
            w8::compute_sync();
            dbg_stamp(p, task, ts);

            // ---- O^T += V^T P^T
            if (tid == 0) {
                tc_fence_after();
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    const uint32_t abase = kv_addr + (2 * (mt & 1)) * 16384 + (mt >> 1) * 8192;
#pragma unroll
                    for (int ks = 0; ks < 4; ++ks)
                        tc_mma_ss(tmem_base + kTmemO + mt * NH, umma_desc_sw128(abase + ks * 2048, 16384, 1024),
                                  umma_desc_sw128(p_addr + ks * 32, 16, 1024), idesc_pv, (sb > 0 || ks > 0));
                }
                tc_commit(bar);
            }
            // while the tensor core runs PV: the block's P sums (only needed for l)
            if (rail0) {
                const float sm = warp_reduce8<false>(d, lane);
                if ((lane & 3) == 0) red_sum[(q4 & 1) * NH + hb + (lane >> 2)] = sm;
            }
            w8::compute_sync();
            {
                const float4 a = *reinterpret_cast<const float4*>(red_sum + hb);
                const float4 b = *reinterpret_cast<const float4*>(red_sum + hb + 4);
                const float4 a2 = *reinterpret_cast<const float4*>(red_sum + NH + hb);
                const float4 b2 = *reinterpret_cast<const float4*>(red_sum + NH + hb + 4);
                const float lb[8] = {a.x + a2.x, a.y + a2.y, a.z + a2.z, a.w + a2.w,
                                     b.x + b2.x, b.y + b2.y, b.z + b2.z, b.w + b2.w};
#pragma unroll
                for (int i = 0; i < kHalf; ++i) l_run[i] = fmaf(l_run[i], factor[i], lb[i]);
            }
            mbar_wait(bar, phase);
            phase ^= 1;
            tc_fence_after();
            dbg_stamp(p, task, ts);
            if (NSUB_MAX > 1 && sb == 0 && nsub > 1) {
                convert_store<K1>(smem, 1, warp, lane, raw1);
                fence_proxy_async_smem();
            }
            w8::compute_sync();  // tile / red buffers free for the next sub-block
        }

        // ---- 3. partial O^T [t][c][quad][dim][4] + (m, l) -> scratch
        const int64_t pbase = ((int64_t)t * p.n_chunks + c);
        float4* po = reinterpret_cast<float4*>(p.part_o) + (pbase * 16 + (h0 + hb) / 4) * kHeadDim;
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            uint32_t v[kHalf];
            tc_ld_32x32b<kHalf>(tmem_base + kTmemO + lane_base + mt * NH + hb, v);
            tc_wait_ld();
            const int dim = 128 * mt + 32 * q4 + lane;
            __stcg(po + dim, make_float4(__uint_as_float(v[0]), __uint_as_float(v[1]), __uint_as_float(v[2]),
                                         __uint_as_float(v[3])));
            __stcg(po + kHeadDim + dim, make_float4(__uint_as_float(v[4]), __uint_as_float(v[5]),
                                                    __uint_as_float(v[6]), __uint_as_float(v[7])));
        }
        if (q4 == 0 && lane == 0) {
            float2* pm = reinterpret_cast<float2*>(p.part_ml) + pbase * kHeads + h0 + hb;
#pragma unroll
            for (int i = 0; i < kHalf; ++i) __stcg(pm + i, make_float2(m_ref[i], l_run[i]));
        }
        dbg_stamp(p, task, ts);
        tc_fence_before();
    }
};

}  // namespace w1
