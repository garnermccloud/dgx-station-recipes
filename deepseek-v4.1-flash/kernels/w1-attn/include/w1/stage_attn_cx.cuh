// W1 bitwise cluster mode (CX): FlashMLA's exact decode arithmetic on a 2-CTA cluster per token, as a decoupled
// warp-specialized pipeline.
//
// FlashMLA splits the 512 dims into two "rails" (rail r = the 64-dim atoms of parity r, 4 atoms) and computes
// S = rail0 + rail1. CTA r of the cluster owns rail r:
//   * K / V: CTA r loads and dequantizes only its 4 atoms of every key row.
//   * QK: rail r for all 64 heads (FlashMLA's 16 K=16 steps), 128 keys (a pair of 64-key blocks) per MMA.
//   * softmax: CTA r runs FlashMLA's lazy-rescale softmax for decision group r (heads 32r .. 32r + 31). It pushes
//     its rail of the partner's heads to the partner over DSMEM and adds the partner's rail of its own heads (one
//     fp32 add, commutative), then pushes its half of each P block and its rescale factors.
//   * PV + epilogue: CTA r accumulates O for its 256 dims x all 64 heads in FlashMLA's order and writes its MXFP8
//     output (every 32-dim scale group lies inside one atom).
// Every rounded value comes from the same operations in the same order as FlashMLA: the output is bit-identical.
//
// Roles (14 warps); every hand-over is an mbarrier with at most one phase outstanding:
//   warps 0-7: TMEM quarter q = w & 3 (S lanes 32q.. = keys 32 (q & 1).. of the pair's block q >> 1) and head
//     half c = w >> 2. c == r: SUM warps (add the partner's rail, block half-maxima, then P and FlashMLA's block
//     sums of their (head, key half) once the chain warp published m). c != r: PUSH warps (push this CTA's rail of
//     the partner's heads as soon as QK lands; per block in order: wait both P halves, rescale O where a group
//     rescaled, release PV). Afterwards warps 0-7 run the epilogue.
//   warp 8: MMA issuer (lane 0): QK of a pair once its tile is ready, PV_b once released, whichever comes first.
//   warp 9: the softmax chain (lane = head of the group): block maxima -> m, rescale factor, flag; at the end the l
//     replay, the m / l exchange with the partner and the output scales.
//   warps 10-13: loaders: from pair 2 on, dequantize the staged rows of pair q into the tile freed by PV of pair
//     q - 2, then stage the rows of pair q + 1.
// The first loads (slot tables, Q, rows of pairs 0-2, dequant of pairs 0 and 1) run on warps 0-7 and 10-13.
#pragma once
#include "attn_params.cuh"
#include "ptx.cuh"
#include "stage_attn_par.cuh"
#include "stage_attn_seq.cuh"
#include "../w8/w8_shim.cuh"

namespace w1 {

// W1_DBG_CX: clock64 event stamps at dbg_ts[64 + 152 * 16 + cta * 128 + k]
__device__ __forceinline__ void cx_stamp(const AttnParams& p, int k) {
#ifdef W1_DBG_CX
    if (p.dbg_ts != nullptr) p.dbg_ts[64 + 152 * 16 + blockIdx.x * 128 + k] = clock64();
#endif
}

struct AttnCX {
    static constexpr int NH = 32;
    static constexpr int kMaxBlocks = 10, kMaxPairs = 5;
    static constexpr int kWarps = 15, kThreads = kWarps * 32;
    static constexpr int kMmaWarp = 8, kChainWarp = 9, kLoadWarp0 = 10, kPvWarp = 14;
    // named barriers (0 = __syncthreads)
    static constexpr int kBarSum = 1;        // + block parity: the 2 SUM warps of that parity (64 threads)
    static constexpr int kBarLoad = 3;       // loader warps (128)
    static constexpr int kBarInit = 4;       // warps 0-7 and the loaders (384)
    static constexpr int kInitThreads = 384;
    static constexpr int kBarEpi = 5;        // warps 0-7 (256)

    // ---- SMEM
    static constexpr uint32_t kTileOff = 0;                    // [2][4 atoms][128 keys][128 B] bf16
    static constexpr uint32_t kQOff = 2 * 65536;               // [4 atoms][64 heads][128 B] bf16 (my rail)
    static constexpr int kPSlots = 4;                          // P ring: P_b in slot b % 4 (PV_{b-4} has read it)
    static constexpr uint32_t kPOff = kQOff + 4 * 8192;        // [4][64 heads][64 keys] bf16 P^T
    static constexpr uint32_t kXOff = kPOff + kPSlots * 8192;  // [2][64 keys][32 heads] fp32 (16 B chunks ^ key & 7)
    static constexpr uint32_t kIdxOff = kXOff + 2 * 8192;      // [640] int32 global slot
    static constexpr uint32_t kFmtOff = kIdxOff + 640 * 4;     // [640]
    static constexpr uint32_t kCsOff = kFmtOff + 640;          // [64] cos | sin
    static constexpr uint32_t kSinkOff = kCsOff + 256;         // [64]
    static constexpr uint32_t kHmaxOff = kSinkOff + 256;       // [10][2 key halves][32] raw block half-maxima
    static constexpr uint32_t kMOff = kHmaxOff + 2560;         // [10][32] m after block b
    static constexpr int kAuxW = 36;                           // per block: f[32], flag (int), pad: one 144 B bulk copy
    static constexpr uint32_t kFOff = kMOff + 1280;            // [10][kAuxW] rescale factors and flag of block b
    static constexpr uint32_t kSumsOff = kFOff + 10 * kAuxW * 4;  // [10][2][32] FlashMLA's per-thread block sums
    static constexpr uint32_t kAuxOff = kSumsOff + 2560;       // [4][36]: the partner's f[32], flag (per P slot)
    static constexpr uint32_t kMfinOff = kAuxOff + kPSlots * 144;  // [64] final m
    static constexpr uint32_t kLfinOff = kMfinOff + 256;       // [64] final l
    static constexpr uint32_t kOscOff = kLfinOff + 256;        // [64]
    static constexpr uint32_t kSigOff = kOscOff + 256;         // [4] landing words of remote signals
    static constexpr uint32_t kSmemBytes = kSigOff + 16;
    static constexpr uint32_t kQStageOff = 65536;              // Q as loaded (4 x 8 KB), in tile 1 before pair 1
    static_assert(kAuxOff % 16 == 0 && kXOff % 1024 == 0 && kPOff % 1024 == 0, "alignment");
    static_assert(kSmemBytes + 1024 + 1024 <= 232448, "smem");
    static constexpr uint32_t kEOff = 0;  // epilogue exchange [8 warps][32][33] fp32 (tile 0 is dead by then)
    static constexpr uint32_t kOutStageOff = 8 * 32 * 33 * 4;  // epilogue fp8 bytes [8 warps][32 heads][32 dims]

    static constexpr uint32_t kTmemS = 0;      // S per pair: 2 x 64 cols (128 lanes = keys of the pair, cols = heads)
    static constexpr uint32_t kTmemO = 128;    // O^T: 2 tiles x 64 cols (128 lanes = dims of 2 atoms, cols = heads)
    static constexpr uint32_t kTmemK = 256;    // K per pair: 2 x 128 cols (lane = key, col c = my dims 2c, 2c + 1)
    static constexpr uint32_t kTmemCols = 512;

    struct Bars {
        uint64_t* kready;  // [5] the pair's K is in its TMEM slot
        uint64_t* vready;  // [5] the pair's V is in its smem tile
        uint64_t* sfree;   // [5] the pair's S has been read (8 warps: SUM + PUSH of both blocks)
        uint64_t* qkd;     // [5] QK of the pair done (commit)
        uint64_t* pready;  // [10] PV_b released (4 PUSH warps)
        uint64_t* pvc;     // [10] PV_b done (commit)
        uint64_t* peer;    // [10] partner's PV_b done (its multicast commit)
        uint64_t* sx;      // [2] partner's rail landed in X[parity] (own expect + 8 KB)
        uint64_t* px;      // [4] partner's P half + aux landed in P slot b % 4 (own expect + 4228 B)
        uint64_t* xfree;   // [2] partner consumed its X[parity] (own expect + 4 B signal)
        uint64_t* cmax;    // [10] block half-maxima written (2 SUM warps)
        uint64_t* mready;  // [10] m / f / flag of block b published (chain warp)
        uint64_t* pmine;   // [10] my P half and block sums written
        uint64_t* fin;     // partner's final m / l landed (own expect + 256 B)
        uint64_t* osc;     // output scales ready (chain warp)
        uint64_t* qld;     // Q landed (own expect + 32 KB)
        static constexpr int kCount = 5 + 5 + 5 + 5 + 10 + 10 + 10 + 2 + 4 + 2 + 10 + 10 + 10 + 1 + 1 + 1;
        __device__ void carve(uint64_t* b) {
            kready = b;
            vready = b + 5;
            sfree = b + 10;
            qkd = b + 15;
            pready = b + 20;
            pvc = b + 30;
            peer = b + 40;
            sx = b + 50;
            px = b + 52;
            xfree = b + 56;
            cmax = b + 58;
            mready = b + 68;
            pmine = b + 78;
            fin = b + 88;
            osc = b + 89;
            qld = b + 90;
        }
        __device__ void init() {
            for (int i = 0; i < kCount; ++i) mbar_init(kready + i, 1);
            for (int i = 0; i < 10; ++i) {
                mbar_init(pready + i, 4);
                mbar_init(cmax + i, 2);
            }
            for (int i = 0; i < 5; ++i) mbar_init(sfree + i, 8);
            fence_mbar_init();
        }
    };

    __device__ static __forceinline__ void expect(uint64_t* bar, uint32_t bytes) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
                     : "memory");
    }
    __device__ static __forceinline__ void named_bar(int id, int n) {
        asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
    }
    // warp-uniform non-blocking mbarrier tests (lane 0 tests, the warp agrees)
    __device__ static __forceinline__ bool ready(uint64_t* bar, uint32_t phase, int lane) {
        uint32_t ok = 0;
        if (lane == 0) ok = mbar_try_wait(bar, phase);
        return __shfl_sync(0xffffffffu, ok, 0) != 0;
    }
    __device__ static __forceinline__ bool ready_cluster(uint64_t* bar, uint32_t phase, int lane) {
        uint32_t ok = 0;
        if (lane == 0)
            asm volatile(
                "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                " selp.u32 %0, 1, 0, p;\n}"
                : "=r"(ok)
                : "r"(smem_u32(bar)), "r"(phase)
                : "memory");
        return __shfl_sync(0xffffffffu, ok, 0) != 0;
    }
    // X: byte offset of 16 B chunk `chunk` (heads 4 chunk ..) of key row `key`
    __device__ static __forceinline__ uint32_t xswz(int key, int chunk) {
        return key * 128 + (((chunk ^ key) & 7) << 4);
    }

    // ---- K / V rows: thread = key row (TMEM lane), so it can write its row of K into the pair's TMEM slot. A row
    //      is loaded once into registers (my atoms only) and dequantized twice: K into TMEM as soon as the slot is
    //      free (QK of pair q - 2 done), V into the smem tile once PV of pair q - 2 has read it.
    template <int A, int W>  // atoms held; W = 16 words an atom (fp8 or NVFP4 rows) or 8 (NVFP4 rows only)
    struct Row {
        uint32_t w[A][W];   // fp8: 64 B an atom; NVFP4: 32 B (w[k][0..7])
        uint32_t sc[A];     // fp8: 2 ue8m0 (low 16 bits); NVFP4: 4 e4m3
        uint8_t f;
    };
    __device__ static __forceinline__ void ld32(const uint8_t* src, uint32_t* w) {
        asm volatile("ld.global.nc.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                     : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]), "=r"(w[4]), "=r"(w[5]), "=r"(w[6]), "=r"(w[7])
                     : "l"(src));
    }
    // my atoms k0 .. k0 + A - 1 of key row `row` of pair q
    template <int A, int W>
    __device__ static __forceinline__ void load_row(const AttnParams& p, const uint8_t* smem, int q, int row, int r,
                                                    int k0, Row<A, W>& x) {
        const int j = 128 * q + row;
        x.f = smem[kFmtOff + j];
        const int gidx = reinterpret_cast<const int32_t*>(smem + kIdxOff)[j];
#pragma unroll
        for (int k = 0; k < A; ++k) {
#pragma unroll
            for (int e = 0; e < W; ++e) x.w[k][e] = 0;
            x.sc[k] = 0;
        }
        if (W == 16 && x.f == AttnSeq::kFp8) {
            const int sh = __ffs(p.swa_page) - 1, off = gidx & (p.swa_page - 1);
            const uint8_t* page = p.swa + (int64_t)(gidx >> sh) * p.swa_block_stride;
#pragma unroll
            for (int k = 0; k < A; ++k) {
                const int a = r + 2 * (k0 + k);
                if constexpr (W == 16) {
                    ld32(page + off * 512 + 64 * a, x.w[k]);
                    ld32(page + off * 512 + 64 * a + 32, x.w[k] + 8);
                }
                x.sc[k] = __ldg(reinterpret_cast<const uint16_t*>(page + p.swa_page * 512 + off * 16 + 2 * a));
            }
        } else if (x.f == AttnSeq::kFp4) {
            const int sh = __ffs(p.comp_page) - 1, off = gidx & (p.comp_page - 1);
            const uint8_t* page = p.comp + (int64_t)(gidx >> sh) * p.comp_block_stride;
#pragma unroll
            for (int k = 0; k < A; ++k) {
                const int a = r + 2 * (k0 + k);
                ld32(page + off * 256 + 32 * a, x.w[k]);
                x.sc[k] = __ldg(reinterpret_cast<const uint32_t*>(page + p.comp_page * 256 + off * 32 + 4 * a));
            }
        }
    }
    // the 16-dim chunk c (0..3) of held atom k: 8 bf16x2 words (zeros for an invalid row)
    template <int A, int W>
    __device__ static __forceinline__ void chunk16(const Row<A, W>& x, int k, int c, uint4& o0, uint4& o1) {
        if (W == 16 && x.f == AttnSeq::kFp8) {
            dequant16(make_uint4(x.w[k][(4 * c) % W], x.w[k][(4 * c + 1) % W], x.w[k][(4 * c + 2) % W],
                                 x.w[k][(4 * c + 3) % W]),
                      false, (x.sc[k] >> (8 * (c >> 1))) & 0xFF, o0, o1);
        } else if (x.f == AttnSeq::kFp4) {
            dequant16(make_uint4(x.w[k][2 * c], x.w[k][2 * c + 1], 0, 0), true, (x.sc[k] >> (8 * c)) & 0xFF, o0, o1);
        } else {
            o0 = make_uint4(0, 0, 0, 0);
            o1 = o0;
        }
    }
    // K of my atoms k0.. into TMEM slot `kslot` (lanes of this warp's quarter; thread = key row)
    template <int A, int W>
    __device__ static __forceinline__ void write_k(uint32_t kslot, int quarter, int k0, const Row<A, W>& x) {
#pragma unroll
        for (int k = 0; k < A; ++k)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                uint32_t v[16];
#pragma unroll
                for (int c2 = 0; c2 < 2; ++c2) {
                    uint4 o0, o1;
                    chunk16(x, k, 2 * hf + c2, o0, o1);
                    v[8 * c2] = o0.x;
                    v[8 * c2 + 1] = o0.y;
                    v[8 * c2 + 2] = o0.z;
                    v[8 * c2 + 3] = o0.w;
                    v[8 * c2 + 4] = o1.x;
                    v[8 * c2 + 5] = o1.y;
                    v[8 * c2 + 6] = o1.z;
                    v[8 * c2 + 7] = o1.w;
                }
                tc_st_32x32b<16>(kslot + 32 * (k0 + k) + 16 * hf + ((uint32_t)(32 * quarter) << 16), v);
            }
    }
    // V of a pair = its K, copied from the TMEM slot into the smem tile (thread = key row of this warp's quarter)
    __device__ static __forceinline__ void copy_v(uint32_t kslot, int quarter, uint8_t* tile, int row) {
#pragma unroll 1
        for (int k = 0; k < 4; ++k) {
            uint32_t v0[16], v1[16];
            const uint32_t ta = kslot + 32 * k + ((uint32_t)(32 * quarter) << 16);
            tc_ld_32x32b<16>(ta, v0);
            tc_ld_32x32b<16>(ta + 16, v1);
            tc_wait_ld();
            uint8_t* trow = tile + k * 16384 + row * 128;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                *reinterpret_cast<uint4*>(trow + (((i ^ row) & 7) << 4)) =
                    make_uint4(v0[4 * i], v0[4 * i + 1], v0[4 * i + 2], v0[4 * i + 3]);
                *reinterpret_cast<uint4*>(trow + ((((4 + i) ^ row) & 7) << 4)) =
                    make_uint4(v1[4 * i], v1[4 * i + 1], v1[4 * i + 2], v1[4 * i + 3]);
            }
        }
    }
    // V of my atoms k0.. into the smem tile (row-major 128 B per atom row, 16 B chunks ^ row & 7)
    template <int A, int W>
    __device__ static __forceinline__ void write_v(uint8_t* tile, int row, int k0, const Row<A, W>& x) {
#pragma unroll
        for (int k = 0; k < A; ++k) {
            uint8_t* trow = tile + (k0 + k) * 16384 + row * 128;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                uint4 o0, o1;
                chunk16(x, k, c, o0, o1);
                *reinterpret_cast<uint4*>(trow + ((((2 * c) ^ row) & 7) << 4)) = o0;
                *reinterpret_cast<uint4*>(trow + ((((2 * c + 1) ^ row) & 7) << 4)) = o1;
            }
        }
    }

    // ---- MMAs
    __device__ static __forceinline__ void mma_ts(uint32_t d, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc,
                                                  uint32_t acc) {
        asm volatile(
            "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}" ::"r"(
                d),
            "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(acc)
            : "memory");
    }
    // S of pair pp = K (TMEM slot, A operand) x Q^T (smem): FlashMLA's 16 K=16 steps of the rail, in order
    __device__ static void issue_qk(uint8_t* smem, int pp, uint32_t tmem_base, uint64_t* bar) {
        tc_fence_after();
        const uint32_t idesc = umma_idesc_bf16(128, 64, false, false);
        const uint32_t qa = smem_u32(smem + kQOff), ka = tmem_base + kTmemK + 128 * (pp & 1);
#pragma unroll
        for (int s = 0; s < 16; ++s)
            mma_ts(tmem_base + kTmemS + 64 * (pp & 1), ka + 8 * s,
                   umma_desc_sw128(qa + (s >> 2) * 8192 + (s & 3) * 32, 16, 1024), idesc, s > 0);
        tc_commit(bar);
    }
    __device__ static void issue_pv(uint8_t* smem, int b, uint32_t tmem_base) {
        tc_fence_after();
        const uint32_t idesc = umma_idesc_bf16(128, 64, true, false);
        const uint32_t va = smem_u32(smem + kTileOff + ((b >> 1) & 1) * 65536) + (b & 1) * 64 * 128;
        const uint32_t pa = smem_u32(smem + kPOff + (b % kPSlots) * 8192);
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
                tc_mma_ss(tmem_base + kTmemO + 64 * u,
                          umma_desc_sw128(va + 2 * u * 16384 + ks * 2048, 16384, 1024),
                          umma_desc_sw128(pa + ks * 32, 16, 1024), idesc, (b > 0 || ks > 0));
    }
    // warp 8 lane 0: QK of pair q once its K is in TMEM and pair q - 2's S buffer has been read
    __device__ static void qk_loop(const AttnParams* g_dbg_p, int nb, uint8_t* smem, uint32_t tmem_base, Bars& bars) {
        const int npairs = (nb + 1) >> 1;
#pragma unroll 1
        for (int q = 0; q < npairs; ++q) {
            if (q >= 2) mbar_wait(bars.sfree + q - 2, 0);
            mbar_wait(bars.kready + q, 0);
            cx_stamp(*g_dbg_p, 102 + q);
            issue_qk(smem, q, tmem_base, bars.qkd + q);
            cx_stamp(*g_dbg_p, 107 + q);
        }
    }
    // warp 14 lane 0: PV_b once its V tile is written and both P halves are in (FlashMLA's block order)
    __device__ static void pv_loop(const AttnParams& p, int nb, uint8_t* smem, uint32_t tmem_base, Bars& bars, int r) {
        const uint16_t partner = (uint16_t)(1u << (r ^ 1));
#pragma unroll 1
        for (int b = 0; b < nb; ++b) {
            if ((b & 1) == 0) mbar_wait(bars.vready + (b >> 1), 0);
            mbar_wait(bars.pready + b, 0);
            cx_stamp(p, 67 + b);
            issue_pv(smem, b, tmem_base);
            if (b == 4 || b == 5) cx_stamp(p, 124 + (b & 1));
            tc_commit(bars.pvc + b);
            tc_commit_mc(bars.peer + b, partner);
        }
        // the partner waits on every "PV done" barrier of ours
        for (int b = nb; b < kMaxBlocks; ++b) tc_commit_mc(bars.peer + b, partner);
    }

    // ---- warp 9: FlashMLA's lazy rescale for group r, then l, the m / l exchange and the output scales
    __device__ static void chain_loop(const AttnParams& p, int nb, uint8_t* smem, Bars& bars, int r, int lane) {
        const uint32_t pr = (uint32_t)(r ^ 1);
        const float mult = p.sm_scale_log2;
        const float* hmax = reinterpret_cast<const float*>(smem + kHmaxOff);
        float* mbuf = reinterpret_cast<float*>(smem + kMOff);
        float* fbuf = reinterpret_cast<float*>(smem + kFOff);
        const float* sums = reinterpret_cast<const float*>(smem + kSumsOff);
        if (lane == 0) expect(bars.fin, 2 * NH * 4);
        float mi = kMaxInit, real_mi = -INFINITY;
#pragma unroll 1
        for (int b = 0; b < nb; ++b) {
            mbar_wait(bars.cmax + b, 0);
            // FlashMLA: own half max * mult, then max with the other half's (both pre-multiplied)
            const float cmax = fmaxf(hmax[(2 * b) * NH + lane] * mult, hmax[(2 * b + 1) * NH + lane] * mult);
            real_mi = fmaxf(real_mi, cmax);
            const bool sc = __any_sync(0xffffffffu, cmax - mi > 6.0f);
            float m_new = mi, f = 1.f;
            if (sc) {
                m_new = fmaxf(cmax, mi);
                f = ex2_approx(mi - m_new);
            }
            mi = m_new;
            mbuf[b * NH + lane] = m_new;
            fbuf[b * kAuxW + lane] = f;
            if (lane == 0) reinterpret_cast<int*>(fbuf)[b * kAuxW + NH] = sc ? 1 : 0;
            __syncwarp();
            if (lane == 0) mbar_arrive_local(bars.mready + b);
        }
        // l: FlashMLA's per-thread chains l = fmaf(l, f_b, sum_b), one per key half, in block order
        float l0 = 0.f, l1 = 0.f;
#pragma unroll 1
        for (int b = 0; b < nb; ++b) {
            mbar_wait(bars.pmine + b, 0);
            const float f = fbuf[b * kAuxW + lane];
            l0 = fmaf(l0, f, sums[(2 * b) * NH + lane]);
            l1 = fmaf(l1, f, sums[(2 * b + 1) * NH + lane]);
        }
        if (real_mi == -INFINITY) {
            l0 = 0.f;
            l1 = 0.f;
            mi = -INFINITY;
        }
        float* mfin = reinterpret_cast<float*>(smem + kMfinOff);
        float* lfin = reinterpret_cast<float*>(smem + kLfinOff);
        const int hg = NH * r + lane;
        const float l = l0 + l1;
        mfin[hg] = mi;
        lfin[hg] = l;
        const uint32_t rb = mapa(bars.fin, pr);
        st_async_b32(mapa(mfin + hg, pr), __float_as_uint(mi), rb);
        st_async_b32(mapa(lfin + hg, pr), __float_as_uint(l), rb);
        mbar_wait_cluster(bars.fin, 0);
        const float* sink = reinterpret_cast<const float*>(smem + kSinkOff);
        float* oscv = reinterpret_cast<float*>(smem + kOscOff);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int hh = lane + 32 * k;
            const float lh = lfin[hh];
            float osc;
            asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(osc) : "f"(1.0f), "f"(lh + ex2_approx(sink[hh] * kL2E - mfin[hh])));
            oscv[hh] = lh == 0.f ? 0.f : osc;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive_local(bars.osc);
        // every multicast "PV done" of the partner has landed before this CTA exits
#pragma unroll 1
        for (int b = 0; b < kMaxBlocks; ++b) mbar_wait_cluster(bars.peer + b, 0);
    }

    // ---- SUM warps of block parity h: key half kq = warp & 1 (thread = key while summing, = head afterwards).
    struct SumCtx {
        const AttnParams* p;
        uint8_t* smem;
        uint32_t tmem_base;
        Bars* bars;
        int r, h, kq, lane, nb;
    };
    // S = my rail + the partner's rail into X[h]; this thread's (head lane, key half kq) values -> x; half max out
    __device__ static void s_stage(const SumCtx& c, int b, float (&x)[32]) {
        const AttnParams& p = *c.p;
        uint8_t* smem = c.smem;
        Bars& bars = *c.bars;
        const uint32_t pr = (uint32_t)(c.r ^ 1);
        const int tid64 = 32 * c.kq + c.lane, key = 32 * c.kq + c.lane, pp = b >> 1, lane = c.lane;
        uint8_t* X = smem + kXOff + c.h * 8192;
        if (tid64 == 0) expect(bars.sx + c.h, 64 * NH * 4);
        mbar_wait(bars.qkd + pp, 0);
        tc_fence_after();
        if (tid64 == 0) cx_stamp(p, 4 + 6 * b);
        uint32_t v[32];
        {
            const uint32_t ts =
                c.tmem_base + kTmemS + 64 * (pp & 1) + 32 * c.r + ((uint32_t)(32 * (2 * c.h + c.kq)) << 16);
            uint32_t a0[16], a1[16];
            tc_ld_32x32b<16>(ts, a0);
            tc_ld_32x32b<16>(ts + 16, a1);
            tc_wait_ld();
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive_local(bars.sfree + pp);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                v[i] = a0[i];
                v[16 + i] = a1[i];
            }
        }
        mbar_wait_cluster(bars.sx + c.h, (b >> 1) & 1);
        if (tid64 == 0) cx_stamp(p, 5 + 6 * b);
        {
            const bool kvalid = smem[kFmtOff + 64 * b + key] != AttnSeq::kInvalid;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float4* ptr = reinterpret_cast<float4*>(X + xswz(key, i));
                const float4 o = *ptr;
                float4 y;
                y.x = kvalid ? __uint_as_float(v[4 * i]) + o.x : -INFINITY;
                y.y = kvalid ? __uint_as_float(v[4 * i + 1]) + o.y : -INFINITY;
                y.z = kvalid ? __uint_as_float(v[4 * i + 2]) + o.z : -INFINITY;
                y.w = kvalid ? __uint_as_float(v[4 * i + 3]) + o.w : -INFINITY;
                *ptr = y;
            }
        }
        named_bar(kBarSum + c.h, 64);
#pragma unroll
        for (int i = 0; i < 32; ++i)
            x[i] = *reinterpret_cast<const float*>(X + xswz(32 * c.kq + i, lane >> 2) + 4 * (lane & 3));
        {
            float t[32];
#pragma unroll
            for (int i = 0; i < 32; ++i) t[i] = x[i];
#pragma unroll
            for (int s2 = 16; s2 >= 1; s2 >>= 1)
#pragma unroll
                for (int i = 0; i < s2; ++i) t[i] = fmaxf(t[i], t[i + s2]);
            reinterpret_cast<float*>(smem + kHmaxOff)[(2 * b + c.kq) * NH + lane] = t[0];
        }
        __syncwarp();
        if (lane == 0) mbar_arrive_local(bars.cmax + b);
        named_bar(kBarSum + c.h, 64);  // every X read done: the partner may push block b + 2
        if (tid64 == 0 && b + 2 < c.nb)
            st_async_b32(mapa(smem + kSigOff + 4 * c.h, pr), 0u, mapa(bars.xfree + c.h, pr));
        if (tid64 == 0) cx_stamp(p, 6 + 6 * b);
    }
    // P = bf16(exp2(fma(s, mult, -m))) and FlashMLA's block sum (even keys chain) + (odd keys chain); my P rows and
    // aux (f, flag) -> the partner's P slot
    __device__ static void p_stage(const SumCtx& c, int b, const float (&x)[32]) {
        const AttnParams& p = *c.p;
        uint8_t* smem = c.smem;
        Bars& bars = *c.bars;
        const uint32_t pr = (uint32_t)(c.r ^ 1);
        const int tid64 = 32 * c.kq + c.lane, lane = c.lane;
        mbar_wait(bars.mready + b, 0);
        if (tid64 == 0) cx_stamp(p, 7 + 6 * b);
        const float m = reinterpret_cast<const float*>(smem + kMOff)[b * NH + lane];
        float d[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) d[i] = ex2_approx(fmaf(x[i], p.sm_scale_log2, -m));
        {
            float sx = 0.f, sy = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                sx = sx + d[2 * i];
                sy = sy + d[2 * i + 1];
            }
            reinterpret_cast<float*>(smem + kSumsOff)[(2 * b + c.kq) * NH + lane] = sx + sy;
        }
        uint8_t* P = smem + kPOff + (b % kPSlots) * 8192;
        if (b >= kPSlots) mbar_wait(bars.pvc + b - kPSlots, 0);  // my PV_{b-4} has read P slot b % 4
        {
            const int row = NH * c.r + lane;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t w[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) w[e] = pack_bf16x2(d[8 * j + 2 * e], d[8 * j + 2 * e + 1]);
                *reinterpret_cast<uint4*>(P + row * 128 + ((((4 * c.kq + j) ^ row) & 7) << 4)) =
                    make_uint4(w[0], w[1], w[2], w[3]);
            }
        }
        fence_proxy_async_smem();
        named_bar(kBarSum + c.h, 64);
        if (tid64 == 0) {
            mbar_arrive_local(bars.pmine + b);
            // once the partner's PV_{b-4} has read its slot b % 4
            if (b >= kPSlots) mbar_wait_cluster(bars.peer + b - kPSlots, 0);
            const uint32_t rb = mapa(bars.px + b % kPSlots, pr);
            bulk_s2r(mapa(P + NH * c.r * 128, pr), P + NH * c.r * 128, NH * 128, rb);
            bulk_s2r(mapa(smem + kAuxOff + (b % kPSlots) * 144, pr), smem + kFOff + b * kAuxW * 4, kAuxW * 4, rb);
            cx_stamp(p, 8 + 6 * b);
        }
    }
    __device__ static void sum_loop(const AttnParams& p, int nb, uint8_t* smem, uint32_t tmem_base, Bars& bars, int r,
                                    int h, int kq, int lane) {
        const SumCtx c{&p, smem, tmem_base, &bars, r, h, kq, lane, nb};
#pragma unroll 1
        for (int b = h; b < nb; b += 2) {
            float x[32];
            s_stage(c, b, x);
            p_stage(c, b, x);
        }
    }

    // ---- PUSH warps: quarter q (block parity q >> 1, keys 32 (q & 1) ..). Push the partner's heads of every block
    //      of parity q >> 1; for every block in order, once both P halves are in, rescale O and release PV_b
    __device__ static void push_loop(const AttnParams& p, int nb, uint8_t* smem, uint32_t tmem_base, Bars& bars, int r,
                                     int q, int lane) {
        const uint32_t pr = (uint32_t)(r ^ 1);
        const int h = q >> 1, kq = q & 1, key = 32 * kq + lane;
        const float* fbuf = reinterpret_cast<const float*>(smem + kFOff);
        const uint32_t xdst = mapa(smem + kXOff + h * 8192, pr), xbar = mapa(bars.sx + h, pr);
        int nb_push = h, nb_prep = 0, px_armed = -1;
        while (nb_push < nb || nb_prep < nb) {
            if (nb_push < nb) {
                const int b = nb_push, pp = b >> 1;
                bool ok = ready(bars.qkd + pp, 0, lane);
                if (ok && b >= 2) ok = ready_cluster(bars.xfree + h, ((b - 2) >> 1) & 1, lane);
                if (ok) {
                    tc_fence_after();
                    const uint32_t ts =
                        tmem_base + kTmemS + 64 * (pp & 1) + 32 * (r ^ 1) + ((uint32_t)(32 * q) << 16);
                    uint32_t a0[16], a1[16];
                    tc_ld_32x32b<16>(ts, a0);
                    tc_ld_32x32b<16>(ts + 16, a1);
                    tc_wait_ld();
                    tc_fence_before();
                    __syncwarp();
                    if (lane == 0) mbar_arrive_local(bars.sfree + pp);
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        st_async_v4(xdst + xswz(key, i), a0[4 * i], a0[4 * i + 1], a0[4 * i + 2], a0[4 * i + 3], xbar);
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        st_async_v4(xdst + xswz(key, 4 + i), a1[4 * i], a1[4 * i + 1], a1[4 * i + 2], a1[4 * i + 3],
                                    xbar);
                    if (kq == 0 && lane == 0 && b + 2 < nb) expect(bars.xfree + h, 4);
                    nb_push += 2;
                    continue;
                }
            }
            if (nb_prep < nb) {
                const int b = nb_prep, s = b % kPSlots;
                if (q == 0 && lane == 0 && px_armed < b) {
                    expect(bars.px + s, NH * 128 + kAuxW * 4);
                    px_armed = b;
                }
                const bool ok =
                    ready(bars.pmine + b, 0, lane) && ready_cluster(bars.px + s, (b / kPSlots) & 1, lane);
                if (ok) {
                    const float* aux = reinterpret_cast<const float*>(smem + kAuxOff + s * 144);
                    const bool my_sc = reinterpret_cast<const int*>(fbuf)[b * kAuxW + NH] != 0;
                    const bool pr_sc = reinterpret_cast<const int*>(aux)[NH] != 0;
                    if (b > 0 && (my_sc || pr_sc)) {
                        // O *= f for each group that rescaled, once PV_{b-1} has landed: this warp's 32 TMEM lanes
                        mbar_wait(bars.pvc + b - 1, 0);
                        tc_fence_after();
#pragma unroll 1
                        for (int g = 0; g < 2; ++g) {
                            if (!(g == r ? my_sc : pr_sc)) continue;
                            const float* f = g == r ? fbuf + b * kAuxW : aux;
#pragma unroll 1
                            for (int u = 0; u < 2; ++u) {
                                uint32_t o0[16], o1[16];
                                const uint32_t ta =
                                    tmem_base + kTmemO + 64 * u + 32 * g + ((uint32_t)(32 * q) << 16);
                                tc_ld_32x32b<16>(ta, o0);
                                tc_ld_32x32b<16>(ta + 16, o1);
                                tc_wait_ld();
#pragma unroll
                                for (int i = 0; i < 16; ++i) {
                                    o0[i] = __float_as_uint(__uint_as_float(o0[i]) * f[i]);
                                    o1[i] = __float_as_uint(__uint_as_float(o1[i]) * f[16 + i]);
                                }
                                tc_st_32x32b<16>(ta, o0);
                                tc_st_32x32b<16>(ta + 16, o1);
                            }
                        }
                        tc_wait_st();
                    }
                    tc_fence_before();
                    __syncwarp();
                    if (lane == 0) mbar_arrive_local(bars.pready + b);
                    if (q == 0 && lane == 0) cx_stamp(p, 9 + 6 * b);
                    ++nb_prep;
                    continue;
                }
            }

        }
    }

    // ---- loader warps (thread = key row of quarter w & 3). K of pair q goes into TMEM slot q & 1 once QK of pair
    //      q - 2 is done and V of pair q - 2 has been copied out of it; V of pair q is copied from the slot into its
    //      smem tile once PV of pair q - 2 has read the tile. A pair's raw rows sit in registers (b0 / b1 by parity)
    //      from their load until their K is written, so the next-but-one pair loads right after.
    __device__ static __forceinline__ int loader_quarter(int lt) { return (lt >> 5) ^ 2; }  // warps 10..13
    __device__ static void write_pair_k(const AttnParams& p, int q, int npairs, uint8_t* smem, uint32_t tmem_base,
                                        Bars& bars, int r, int lt, Row<4, 8>& x) {
        const int quarter = loader_quarter(lt), row = 32 * quarter + (lt & 31);
        mbar_wait(bars.qkd + q - 2, 0);
        tc_fence_after();
        if (lt == 0) cx_stamp(p, 77 + 5 * q);
        write_k(tmem_base + kTmemK + 128 * (q & 1), quarter, 0, x);
        tc_wait_st();
        tc_fence_before();
        named_bar(kBarLoad, 128);
        if (lt == 0) {
            mbar_arrive_local(bars.kready + q);
            cx_stamp(p, 112 + 2 * q);
        }
        if (q + 2 < npairs) load_row(p, smem, q + 2, row, r, 0, x);
        if (lt == 0) cx_stamp(p, 79 + 5 * q);
    }
    __device__ static void copy_pair_v(const AttnParams& p, int q, uint8_t* smem, uint32_t tmem_base, Bars& bars,
                                       int lt) {
        const int quarter = loader_quarter(lt), row = 32 * quarter + (lt & 31);
        mbar_wait(bars.pvc + 2 * q - 3, 0);
        tc_fence_after();
        if (lt == 0) cx_stamp(p, 78 + 5 * q);
        copy_v(tmem_base + kTmemK + 128 * (q & 1), quarter, smem + kTileOff + (q & 1) * 65536, row);
        fence_proxy_async_smem();
        tc_fence_before();
        named_bar(kBarLoad, 128);
        if (lt == 0) {
            mbar_arrive_local(bars.vready + q);
            cx_stamp(p, 113 + 2 * q);
        }
    }
    __device__ static void load_loop(const AttnParams& p, int nb, uint8_t* smem, uint32_t tmem_base, Bars& bars,
                                     int r, int lt, Row<4, 8>& b0, Row<4, 8>& b1) {
        const int npairs = (nb + 1) >> 1;
        if (npairs > 2) write_pair_k(p, 2, npairs, smem, tmem_base, bars, r, lt, b0);
        if (npairs > 3) write_pair_k(p, 3, npairs, smem, tmem_base, bars, r, lt, b1);
#pragma unroll 1
        for (int q = 2; q < npairs; ++q) {
            copy_pair_v(p, q, smem, tmem_base, bars, lt);
            if (q + 2 < npairs) {
                if (q & 1)
                    write_pair_k(p, q + 2, npairs, smem, tmem_base, bars, r, lt, b1);
                else
                    write_pair_k(p, q + 2, npairs, smem, tmem_base, bars, r, lt, b0);
            }
        }
    }

    // ---- first loads: warps 0-7 and the loaders (thread t of kInitThreads)
    __device__ static void init_phase(const AttnParams& p, int tok, int nb, uint8_t* smem, uint32_t tmem_base,
                                      Bars& bars, int r, int t, bool loader, Row<4, 8>& b0, Row<4, 8>& b1) {
        const int npairs = (nb + 1) >> 1, nslots = nb * 64;
        int32_t* idxs = reinterpret_cast<int32_t*>(smem + kIdxOff);
        uint8_t* fmt = smem + kFmtOff;
        int spos = 0;
        if (t < 64) spos = load_position(p, tok);
        int idx[2], len[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            idx[k] = -1;
            len[k] = 0;
            const int j = t + kInitThreads * k;
            if (j < nslots) {
                const bool swa = j < p.swa_width;
                const int jj = swa ? j : j - p.swa_width;
                if (swa || (p.comp != nullptr && jj < p.comp_width)) {
                    idx[k] = __ldg((swa ? p.swa_idx + (int64_t)tok * p.swa_idx_stride
                                        : p.comp_idx + (int64_t)tok * p.comp_idx_stride) +
                                   jj);
                    len[k] = __ldg((swa ? p.swa_lens : p.comp_lens) + tok);
                }
            }
        }
        float cs_v = 0.f, sink_v = 0.f;
        if (t < 64) {
            cs_v = __ldg(p.cos_sin + (int64_t)spos * kRopeDim + t);
            sink_v = __ldg(p.sink + t);
        }
        if (t == 0) {
            // my rail's Q: atom 2k + r is 4 x 2 KB ([cd][64 heads][16]) contiguous -> 4 bulk copies, reorganized below
            const uint16_t* qt = p.q + (int64_t)tok * p.q_tok_stride;
            expect(bars.qld, 32768);
#pragma unroll
            for (int k = 0; k < 4; ++k) bulk_g2s(smem + kQStageOff + 8192 * k, qt + (r + 2 * k) * 4096, 8192, bars.qld);
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int j = t + kInitThreads * k;
            if (j >= 128 * npairs) continue;
            uint8_t f = AttnSeq::kInvalid;
            int gidx = -1;
            if (j < nslots) {
                const bool swa = j < p.swa_width;
                const int jj = swa ? j : j - p.swa_width;
                if (jj < len[k] && idx[k] >= 0) {
                    f = swa ? AttnSeq::kFp8 : AttnSeq::kFp4;
                    gidx = idx[k];
                }
            }
            fmt[j] = f;
            idxs[j] = gidx;
        }
        named_bar(kBarInit, kInitThreads);  // tables
        if (t == 0) cx_stamp(p, 1);
        // rows into registers (thread = key row): pair 0 on warps 0-7 (warp w: atoms 2 (w >> 2) ..), pairs 1 and 2
        // on the loaders (pair 2 held until its slots are free)
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const int quarter = warp & 3, row = 32 * quarter + lane;
        Row<2, 16> x0;
        if (!loader) {
            load_row(p, smem, 0, row, r, 2 * (warp >> 2), x0);
        } else {
            if (npairs > 1) load_row(p, smem, 1, row, r, 0, b1);
            if (npairs > 2) load_row(p, smem, 2, row, r, 0, b0);
        }
        if (t == 0) cx_stamp(p, 122);
        if (t < 64) {
            reinterpret_cast<float*>(smem + kCsOff)[t] = cs_v;
            reinterpret_cast<float*>(smem + kSinkOff)[t] = sink_v;
        }
        // Q: [k][cd][head][16] as loaded -> [k][head][128 B] swizzled (16 B unit (cd, head, half))
        mbar_wait(bars.qld, 0);
        for (int u = t; u < 2048; u += kInitThreads) {
            const int k = u >> 9, cd = (u >> 7) & 3, hh = (u >> 1) & 63, half = u & 1;
            *reinterpret_cast<uint4*>(smem + kQOff + k * 8192 + hh * 128 + ((((cd * 2 + half)) ^ (hh & 7)) << 4)) =
                *reinterpret_cast<const uint4*>(smem + kQStageOff + k * 8192 + cd * 2048 + hh * 32 + half * 16);
        }
        named_bar(kBarInit, kInitThreads);  // Q, cos / sin, sink
        if (t == 0) cx_stamp(p, 2);
        if (r == 1) {
            // FlashMLA's q RoPE on dims 448..511 = atom 7 = my atom 3: 64 rows x 8 units of 16 B
            const float* cs = reinterpret_cast<const float*>(smem + kCsOff);
            for (int u = t; u < 512; u += kInitThreads) {
                const int hh = u >> 3, cphys = u & 7, c = cphys ^ (hh & 7);
                uint4* ptr = reinterpret_cast<uint4*>(smem + kQOff + 3 * 8192 + hh * 128 + cphys * 16);
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
        named_bar(kBarInit, kInitThreads);  // RoPE'd Q
        if (t == 0) cx_stamp(p, 123);
        // K of pair 0 (warps 0-7) and pair 1 (loaders) first: QK can start; then their V (pair 1's tile held Q's
        // staging copy, read above)
        if (!loader) {
            write_k(tmem_base + kTmemK, quarter, 2 * (warp >> 2), x0);
            tc_wait_st();
            tc_fence_before();
            named_bar(kBarEpi, 256);
            if (t == 0) mbar_arrive_local(bars.kready);
            write_v(smem + kTileOff, row, 2 * (warp >> 2), x0);
            fence_proxy_async_smem();
            named_bar(kBarEpi, 256);
            if (t == 0) {
                mbar_arrive_local(bars.vready);
                cx_stamp(p, 3);
            }
        } else if (npairs > 1) {
            write_k(tmem_base + kTmemK + 128, quarter, 0, b1);
            tc_wait_st();
            tc_fence_before();
            named_bar(kBarLoad, 128);
            if (t == 256) mbar_arrive_local(bars.kready + 1);
            write_v(smem + kTileOff + 65536, row, 0, b1);
            fence_proxy_async_smem();
            named_bar(kBarLoad, 128);
            if (t == 256) mbar_arrive_local(bars.vready + 1);
        }
        if (loader && npairs > 3) load_row(p, smem, 3, row, r, 0, b1);
    }

    // ---- epilogue: FlashMLA's store_o for my 256 dims x 64 heads. Warp w: TMEM lanes 32 (w & 3) of tile u
    //      (= dims 32 (w & 1) .. of atom 2u + ((w & 3) >> 1)), heads 32 (w >> 2) ..
    __device__ static void epilogue(const AttnParams& p, uint8_t* smem, uint32_t tmem_base, int t, int r, int warp,
                                    int lane) {
        const int q4 = warp & 3, hb = 32 * (warp >> 2);
        const float* cs = reinterpret_cast<const float*>(smem + kCsOff);
        const float* oscv = reinterpret_cast<const float*>(smem + kOscOff);
        float* E = reinterpret_cast<float*>(smem + kEOff) + warp * (32 * 33);
        uint8_t* out_t = p.out_fp8 + (int64_t)t * (kHeads * kHeadDim);
        uint8_t* sf_t = reinterpret_cast<uint8_t*>(p.out_sf + t * p.sf_stride_tok);
#pragma unroll 1
        for (int u = 0; u < 2; ++u) {
            const int a = r + 4 * u + 2 * (q4 >> 1);  // global atom
            const int dim = 64 * a + 32 * (q4 & 1) + lane;
            const int j = dim >> 5;
            const bool rope = dim >= kHeadDim - kRopeDim;  // warp-uniform
            float y[32];
            {
                uint32_t v0[16], v1[16];
                const uint32_t ta = tmem_base + kTmemO + 64 * u + hb + ((uint32_t)(32 * q4) << 16);
                tc_ld_32x32b<16>(ta, v0);
                tc_ld_32x32b<16>(ta + 16, v1);
                tc_wait_ld();
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    y[e] = __uint_as_float(v0[e]);
                    y[16 + e] = __uint_as_float(v1[e]);
                }
            }
            if (threadIdx.x == 0) cx_stamp(p, 126 + u);
#pragma unroll
            for (int e = 0; e < 32; ++e) E[e * 33 + lane] = y[e];
            __syncwarp();
            if (rope) {
                const int i = (dim - (kHeadDim - kRopeDim)) >> 1;
                const float cc = cs[i], sn = -cs[32 + i];
#pragma unroll
                for (int e = 0; e < 32; ++e) {
                    const float other = E[e * 33 + (lane ^ 1)];
                    const float x0 = (lane & 1) ? other : y[e], x1 = (lane & 1) ? y[e] : other;
                    y[e] = (lane & 1) ? fmaf(x0, sn, x1 * cc) : fmaf(x0, cc, (-x1) * sn);
                }
                __syncwarp();
#pragma unroll
                for (int e = 0; e < 32; ++e) E[e * 33 + lane] = y[e];
                __syncwarp();
            }
            // lane = head hb + lane: amax over the warp's 32 dims (one MXFP8 group), scale, multiplier
            float mu;
            {
                const int hh = hb + lane;
                const float* row = E + lane * 33;
                float am[32];
#pragma unroll
                for (int k = 0; k < 32; ++k) am[k] = fabsf(row[k]);
#pragma unroll
                for (int w = 16; w >= 1; w >>= 1)
#pragma unroll
                    for (int k = 0; k < w; ++k) am[k] = fmaxf(am[k], am[k + w]);
                const float osc = oscv[hh];
                const float amax = fmaxf(1e-4f, am[0] * osc);
                float sf;
                asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(sf) : "f"(amax), "f"(448.0f));
                const uint32_t sfb = __float_as_uint(sf);
                const int exp_sf = (int)((sfb - 1u) >> 23) + (1 - 127);
                mu = osc * __uint_as_float((uint32_t)(127 - exp_sf) << 23);
                const int G = hh >> 3, sfi = (hh & 7) + j * 8;
                sf_t[(G * p.sf_stride_grp + (sfi >> 2) * p.sf_stride_hd) * 4 + (sfi & 3)] = (uint8_t)(exp_sf + 127);
            }
            __syncwarp();
            E[lane * 33 + 32] = mu;  // row lane: head hb + lane's multiplier in the pad column
            __syncwarp();
            // fp8 bytes -> [head][dim] staging, then lane = head stores its 32 B (contiguous 256 B per 8-head group)
            uint8_t* st = smem + kOutStageOff + warp * 1024;
#pragma unroll
            for (int e = 0; e < 32; ++e)
                st[e * 32 + lane] = (uint8_t)(f32x2_to_e4m3x2(y[e] * E[e * 33 + 32], 0.f) & 0xFF);
            __syncwarp();
            {
                const int hh = hb + lane;
                const uint4* src = reinterpret_cast<const uint4*>(st + lane * 32);
                uint4* dst = reinterpret_cast<uint4*>(out_t + (hh >> 3) * 4096 + ((j * 8) + (hh & 7)) * 32);
                dst[0] = src[0];
                dst[1] = src[1];
            }
            __syncwarp();
        }
    }

    // ---- one token per cluster
    __device__ static void run(const AttnParams& p, uint8_t* smem, uint32_t tmem_base, Bars& bars, int r, int tok) {
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        w8::StandaloneSync sync;
        sync.wait_extern();
        const int nb = AttnSeq::num_blocks(p, tok);
        if (warp == kMmaWarp) {
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            if (lane == 0) qk_loop(&p, nb, smem, tmem_base, bars);
            __syncwarp();
            return;
        }
        if (warp == kPvWarp) {
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            if (lane == 0) pv_loop(p, nb, smem, tmem_base, bars, r);
            __syncwarp();
            return;
        }
        if (warp == kChainWarp) {
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            chain_loop(p, nb, smem, bars, r, lane);
            return;
        }
        const bool loader = warp >= kLoadWarp0;
        const int t = loader ? tid - 64 : tid;  // 0 .. 383
        if (t == 0) cx_stamp(p, 0);
        Row<4, 8> b0, b1;
        init_phase(p, tok, nb, smem, tmem_base, bars, r, t, loader, b0, b1);
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (loader) {
            load_loop(p, nb, smem, tmem_base, bars, r, t - 256, b0, b1);
            return;
        }
        const int q = warp & 3, c = warp >> 2;
        if (c == r)
            sum_loop(p, nb, smem, tmem_base, bars, r, q >> 1, q & 1, lane);
        else
            push_loop(p, nb, smem, tmem_base, bars, r, q, lane);
        mbar_wait(bars.pvc + nb - 1, 0);
        tc_fence_after();
        mbar_wait(bars.osc, 0);
        if (tid == 0) cx_stamp(p, 64);
        epilogue(p, smem, tmem_base, tok, r, warp, lane);
        if (tid == 0) cx_stamp(p, 66);
    }
};

}  // namespace w1
