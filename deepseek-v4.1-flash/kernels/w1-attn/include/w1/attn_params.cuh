// Parameters shared by W1's two attention stages (split -> combine) for one DeepSeek-V4.1 layer at decode.
#pragma once
#include <cstdint>

namespace w1 {

constexpr int kHeads = 64;
constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kBlockKeys = 64;     // keys per MMA sub-block
constexpr int kWvGroup = 8;        // heads per wo_a group
constexpr float kL2E = 1.4426950408889634f;
constexpr float kMaxInit = -1e30f;  // FlashMLA's MAX_INIT_VAL

struct AttnParams {
    // ---- query, rope, sink
    const uint16_t* q;        // bf16 [s_q, 64 * 512], FlashMLA chunk-interleaved layout, token stride below
    int64_t q_tok_stride;     // elements
    const void* positions;    // [s_q] int32 or int64
    int positions_i64;
    const float* cos_sin;     // [P, 64] fp32: cos(32) | sin(32)
    const float* sink;        // [64] fp32 (-inf for padding heads)
    float sm_scale_log2;      // softmax_scale * log2(e), as FlashMLA computes it on the host
    int s_q;

    // ---- sliding-window cache: V4.1 MXFP8 records, [nb, page, 528] with data rows then scale rows per page
    const uint8_t* swa;
    int64_t swa_block_stride;  // bytes
    int swa_page;              // tokens per page
    const int32_t* swa_idx;    // [s_q, swa_width] global slots (block * page + off), < 0 invalid
    int64_t swa_idx_stride;
    int swa_width;
    const int32_t* swa_lens;   // [s_q]

    // ---- compressed cache (cr > 0): V4.1 NVFP4 records, [nb, page, 288]; null when the layer is window-only
    const uint8_t* comp;
    int64_t comp_block_stride;
    int comp_page;
    const int32_t* comp_idx;   // [s_q, comp_width] global slots, < 0 invalid (or local indices if comp_block_table)
    int64_t comp_idx_stride;
    int comp_width;
    const int32_t* comp_lens;  // [s_q]; null with comp_block_table (lengths are counted in-kernel)
    // Fused compute_global_topk_indices_and_lens: when non-null, comp_idx holds the indexer's LOCAL indices and the
    // global slot is block_table[token_to_req[t], local / comp_page] * comp_page + local % comp_page.
    const int32_t* comp_block_table;
    int64_t comp_block_table_stride;
    const int32_t* token_to_req;
    const uint8_t* is_valid_token;  // bool

    // ---- work decomposition
    int n_chunks;      // key chunks per token (each chunk = nsub * 64 unified slots: window first, then compressed)
    int nsub;          // 64-key sub-blocks per chunk
    int head_groups;   // 64 / heads-per-split-task
    int comb_heads;    // heads per combine task

    // ---- split -> combine scratch (fp32, fixed layout)
    float* part_o;     // [s_q, n_chunks, 16 head quads, 512 dims, 4 heads]
    float* part_ml;    // [s_q, n_chunks, 64, 2] (m in log2 units, l)
    uint32_t* flags;   // [split tasks] epoch-tagged "partial published" flags (never cleared)
    uint32_t* epoch;   // [2]: launch epoch, CTA arrive counter

    // ---- optional W7 L2 prefetch of the following GEMMs' weights (issued before griddepcontrol.wait)
    const int64_t* prefetch_table;  // [n, 2] (address, bytes)
    int prefetch_n;

    // ---- outputs, exactly what alloc_mega_attn_output hands the kernel
    uint8_t* out_fp8;          // [s_q, 8, 4096] e4m3, token stride 32768 B
    int32_t* out_sf;           // packed ue8m0, int32 view [s_q, 8, 32]
    int64_t sf_stride_tok, sf_stride_grp, sf_stride_hd;  // int32 elements

    // ---- bitwise-parallel mode scratch (see stage_attn_par.cuh): tagged 64-bit words, never cleared
    uint64_t* par_cmax;  // [s_q][NB][2][64]
    uint64_t* par_psum;  // [s_q][NB][2][64]
    uint64_t* par_pimg;  // [s_q][NB][2][1024]
    uint64_t* par_hint;  // [s_q][NB]
    int par_nb;

    unsigned long long* dbg_ts;  // optional: phase timestamps of task 0 (debug builds)
};

__device__ __forceinline__ void dbg_stamp(const AttnParams& p, int task, int& k) {
    if (p.dbg_ts != nullptr && task == 0 && threadIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        p.dbg_ts[k] = t;
    }
    ++k;
}

// per-CTA phase timestamp k (< 8) of the CTA's last task (debug: dbg_ts != nullptr): globaltimer ns at
// [64 + cta * 8 + k], SM clock at [64 + 152 * 8 + cta * 8 + k]
__device__ __forceinline__ void cta_stamp(const AttnParams& p, int k) {
    if (p.dbg_ts != nullptr && threadIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        p.dbg_ts[64 + blockIdx.x * 8 + k] = t;
#ifndef W1_DBG_SS
        p.dbg_ts[64 + 152 * 8 + blockIdx.x * 8 + k] = clock64();
#endif
    }
}

__device__ __forceinline__ int load_position(const AttnParams& p, int t) {
    return p.positions_i64 ? (int)reinterpret_cast<const int64_t*>(p.positions)[t]
                           : reinterpret_cast<const int32_t*>(p.positions)[t];
}

}  // namespace w1
