// W1 stage 2 of 2: fixed-order split-KV combine + attention sink + inverse GPT-J RoPE + MXFP8 cast, writing the
// exact (data, packed-ue8m0 scale) buffers FlashMLA's mega-attention kernel writes for the grouped wo_a GEMM.
//
// Task = (token t, quad of 4 heads). Inputs: the token's n_chunks partials of those heads ([dim][4] fp32, one 8 KB bulk
// copy each) and their (m, l). Per head: M = max_c m_c, w_c = exp2(m_c - M), L = sum_c w_c l_c (c ascending),
// out_scale = L == 0 ? 0 : 1 / (L + exp2(sink * log2e - M)) (div.approx, as FlashMLA's fast-math __fdividef),
// O = sum_c w_c O_c (c ascending). Per (head, 32-dim group): conjugate RoPE on dims 448.. of the unnormalized O,
// amax * out_scale, clamp 1e-4, sf = amax / 448 (div.approx), e = ceil(log2 sf) (FlashMLA's bit trick),
// e4m3 = cvt.rn.satfinite(O * (out_scale * 2^-e)), scale byte e + 127 -- FlashMLA's epilogue arithmetic.
#pragma once
#include "attn_params.cuh"
#include "ptx.cuh"
#include "../w8/w8_shim.cuh"

namespace w1 {

template <int CMAX>
struct AttnCombine {
    using Params = AttnParams;
    static constexpr const char* kName = "attn_combine";
    static constexpr int CH = 4;  // heads per task (one partial quad)

    static constexpr uint32_t kOOff = 0;                                    // [c][512 dims][4 heads] fp32
    static constexpr uint32_t kMlOff = kOOff + CMAX * kHeadDim * CH * 4;    // [c][h] float2
    static constexpr uint32_t kWOff = kMlOff + CMAX * CH * 8;               // [c][h] weights
    static constexpr uint32_t kScOff = kWOff + CMAX * CH * 4;               // [h] out_scale
    static constexpr uint32_t kBarOff = (kScOff + CH * 4 + 7) / 8 * 8;
    static constexpr uint32_t kSmemBytes = kBarOff + 8;
    static_assert(kSmemBytes <= 96 * 1024, "stage SMEM budget");

    __host__ __device__ static constexpr uint32_t smem_bytes(int) { return kSmemBytes; }
    __host__ __device__ static constexpr uint32_t tmem_cols(int) { return 0; }
    __host__ __device__ static int num_tasks(const Params&, int S) { return S * (kHeads / CH); }

    __device__ static float div_approx(float a, float b) {
        float r;
        asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
        return r;
    }

    // Split tasks this combine task reads: (t, c, g) for c = 0..n_chunks-1, g = quad / (heads per split task / 4)
    __host__ __device__ static w8::Dep dep(const Params& p, int task, int split_heads, int& stride) {
        const int t = task / (kHeads / CH), quad = task - t * (kHeads / CH);
        const int g = quad / (split_heads / CH);
        stride = p.head_groups;
        const int lo = t * p.n_chunks * p.head_groups + g;
        return w8::Dep{p.flags, lo, lo + p.n_chunks * p.head_groups};
    }

    template <class Sync>
    __device__ static void consume(const Params& p, int S, int task, Sync& sync, w8::Ring&, uint8_t* smem, uint32_t) {
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const int t = task / (kHeads / CH);
        const int hq = task - t * (kHeads / CH);
        const int h0 = hq * CH;
        const int C = p.n_chunks;
        uint64_t* bar = reinterpret_cast<uint64_t*>(smem + kBarOff);
        const float* osm = reinterpret_cast<const float*>(smem + kOOff);
        float2* ml = reinterpret_cast<float2*>(smem + kMlOff);
        float* wsm = reinterpret_cast<float*>(smem + kWOff);
        float* scm = reinterpret_cast<float*>(smem + kScOff);
        if (tid == 0) {
            mbar_init(bar, 1);
            fence_mbar_init();
        }
        int stride;
        sync.wait(dep(p, task, 16, stride), stride);  // the partials of every chunk of these heads
        w8::compute_sync();
        if (p.dbg_ts != nullptr && threadIdx.x == 0) {
            unsigned long long tt;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tt));
            p.dbg_ts[64 + blockIdx.x * 8 + 4] = tt;
        }
        if (tid == 0) {
            fence_proxy_async_global();  // generic-proxy partial stores of other CTAs -> TMA reads
            mbar_arrive_expect_tx(bar, (uint32_t)(C * kHeadDim * CH * 4));
            for (int c = 0; c < C; ++c)
                bulk_g2s(smem + kOOff + c * kHeadDim * CH * 4,
                         p.part_o + ((((int64_t)t * C + c) * 16 + hq) * kHeadDim) * CH, kHeadDim * CH * 4, bar);
        }
        if (tid < C * CH) {
            const int c = tid / CH, h = tid - c * CH;
            ml[tid] = __ldcg(reinterpret_cast<const float2*>(p.part_ml) + ((int64_t)t * C + c) * kHeads + h0 + h);
        }
        w8::compute_sync();
        if (tid < CH) {
            float M = kMaxInit;
            for (int c = 0; c < C; ++c) M = fmaxf(M, ml[c * CH + tid].x);
            float L = 0.f;
            for (int c = 0; c < C; ++c) {
                const float w = ex2_approx(ml[c * CH + tid].x - M);
                wsm[c * CH + tid] = w;
                L = fmaf(w, ml[c * CH + tid].y, L);
            }
            const float sink_l2 = __ldg(p.sink + h0 + tid) * kL2E;
            scm[tid] = L == 0.f ? 0.f : div_approx(1.0f, L + ex2_approx(sink_l2 - M));
        }
        w8::compute_sync();
        mbar_wait(bar, 0);

        const int pos = load_position(p, t);
        const float* cs = p.cos_sin + (int64_t)pos * kRopeDim;
        uint8_t* out_t = p.out_fp8 + (int64_t)t * (kHeads * kHeadDim);
        uint8_t* sf_t = reinterpret_cast<uint8_t*>(p.out_sf + t * p.sf_stride_tok);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            const int j = warp + 8 * jj;  // 32-dim group
            const int dim = 32 * j + lane;
            float o[CH] = {0.f, 0.f, 0.f, 0.f};
            for (int c = 0; c < C; ++c) {
                const float4 v = *reinterpret_cast<const float4*>(osm + (c * kHeadDim + dim) * CH);
                const float4 w = *reinterpret_cast<const float4*>(wsm + c * CH);
                o[0] = fmaf(w.x, v.x, o[0]);
                o[1] = fmaf(w.y, v.y, o[1]);
                o[2] = fmaf(w.z, v.z, o[2]);
                o[3] = fmaf(w.w, v.w, o[3]);
            }
            const bool rope = j >= (kHeadDim - kRopeDim) / 32;
            float cc = 0.f, sn = 0.f;
            if (rope) {
                const int i = (dim - (kHeadDim - kRopeDim)) >> 1;
                cc = __ldg(cs + i);
                sn = -__ldg(cs + 32 + i);
            }
#pragma unroll
            for (int h = 0; h < CH; ++h) {
                float y = o[h];
                if (rope) {
                    const float other = __shfl_xor_sync(0xffffffffu, y, 1);
                    const float x0 = (lane & 1) ? other : y, x1 = (lane & 1) ? y : other;
                    y = (lane & 1) ? fmaf(x0, sn, x1 * cc) : fmaf(x0, cc, (-x1) * sn);
                }
                float amax = fabsf(y);
#pragma unroll
                for (int s = 16; s >= 1; s >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, s));
                const float osc = scm[h];
                amax = fmaxf(1e-4f, amax * osc);
                const float sf = div_approx(amax, 448.0f);
                const uint32_t sfb = __float_as_uint(sf);
                const int exp_sf = (int)((sfb - 1u) >> 23) + (1 - 127);
                const float sf_inv = __uint_as_float((uint32_t)(127 - exp_sf) << 23);
                const float mult = osc * sf_inv;
                const uint8_t q8 = (uint8_t)(f32x2_to_e4m3x2(y * mult, 0.f) & 0xFF);
                const int H = h0 + h, grp = H >> 3, hin = H & 7;
                out_t[grp * (kWvGroup * kHeadDim) + (j * kWvGroup + hin) * 32 + lane] = q8;
                if (lane == 0) {
                    const int sfi = hin + j * kWvGroup;
                    sf_t[(grp * p.sf_stride_grp + (sfi >> 2) * p.sf_stride_hd) * 4 + (sfi & 3)] =
                        (uint8_t)(exp_sf + 127);
                }
            }
        }
    }
};

}  // namespace w1
