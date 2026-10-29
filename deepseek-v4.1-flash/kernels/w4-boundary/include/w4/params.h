// Host/device parameter blocks of the W4 kernels (plain structs; includable from host C++).
#pragma once

#include <cstdint>
#include <cuda_bf16.h>

namespace w4 {

struct SeamParams {
  // inputs
  const __nv_bfloat16* x;         // [T, H] sublayer output (attention wo_b output, or MegaMoE y)
  const __nv_bfloat16* residual;  // [T, 4, H]
  const float* post;              // [T, 4] post-mix of the sublayer that produced x
  const float* comb;              // [T, 4, 4] comb[t][j][i]
  const float* pre;               // [T, 4] pre-mix this collapse uses (carried from the previous boundary)
  const __nv_bfloat16* norm_w;    // [H]
  // optional cold-expert rows (peer output) added into x first: x[t] = bf16(x[t] + pout[pos[t]]) if has[t]
  const int8_t* cold_has;
  const int32_t* cold_pos;
  const __nv_bfloat16* cold_out;
  const int32_t* cold_rows;       // [1] number of cold rows (0 -> no add)
  // outputs
  __nv_bfloat16* residual_out;    // [T, 4, H]
  __nv_bfloat16* aux_out;         // [T, H] mean of the 4 streams, or null
  __nv_bfloat16* layer_input;     // [T, H] bf16, or null
  uint8_t* fi_q;                  // [T, H] fp8 (FlashInfer MXFP8), or null
  uint8_t* fi_sf_swz;             // swizzled 128x4 ue8m0 scales [ceil(T/128)*128 * 160], or null
  uint8_t* fi_sf_lin;             // linear [T, 160] ue8m0 scales (same values), or null
  uint8_t* prep_x;                // [T, H] fp8 (MegaMoE staging recipe), or null
  int32_t* prep_sf;               // [T, H/128] packed ue8m0, or null
  int32_t* prep_shared_sf;        // shared-expert scale view (MN-major permuted), or null
  int shared_block_m;
  int T;
  // optional fused peer-v3 finish of the MoE call that produced x (W6 peer3 W4_INTEGRATION.md section 6): wait for
  // the RTX acknowledgement of tag (p3_state[0], p3_layer, T), add out rows of tokens with cold_has[t] into x
  // (bf16(x + out), written back to p3_y), then advance the call counter once. p3_seg == null: none.
  const uint8_t* p3_seg;
  uint32_t* p3_state;             // [2] peer3 GB300 state ([1] = arrival counter, zero at rest)
  unsigned long long* p3_stats;   // peer3 GbStats (CTA 0 records the wait), or null
  int p3_stats_words;             // 20 (v3.0) or >= 22 (v3.1: last_wait_ns / last_tag)
  int p3_layer;
  long long p3_spins;
  __nv_bfloat16* p3_y;            // = x (the MoE output), receives the finished rows
  // optional device mirror of the segment (COPY_AHEAD): the done tag at +128, a ready word at +136 (u32, set by the
  // copy-engine chain after it copied the tag and the out rows, cleared by the call's last CTA) and the out rows at
  // kOffOut. Null: the rows and the tag come from the host segment only.
  uint8_t* p3_mirror;
  // 1: the padding rows [T, 128) of fi_sf_swz are zero already (a persistent per-T buffer zeroed once); skip them
  int sf_pad_ready;
  // optional MegaMoE combine done here (W6 fork v11, skip_combine = 1: MegaMoE writes no y). x is then ignored: the
  // sublayer output of token t is bf16_rn(acc), acc = 0.f + slot_k[t] (add.rn.f32.bf16, per element) over the present
  // slots in k order: slot k < 6 iff cmb_topk[t][k] >= 0, slot 6 (the shared expert) always, as v5's combine. Slot k
  // of token t: cmb_slots + k * cmb_slot_stride + t * cmb_tok_stride (bf16 [H]). The finished y (after the fused
  // finish's cold add) is written to p3_y for every token. Null: x holds y.
  const uint8_t* cmb_slots;
  unsigned long long cmb_slot_stride;
  unsigned long long cmb_tok_stride;
  const long long* cmb_topk;  // [T, 6] the MegaMoE input topk_idx (symm.topk_idx)
  unsigned long long* dbg;  // optional per-phase %globaltimer stamps (CTA 0, thread 0)
};

struct MixParams {
  const __nv_bfloat16* rn;  // [T, 4*H] residual_new
  const float* fn_img;      // [16 splits][122880 B] hc_fn, tf32 RNE, laid out as UMMA's smem image (mix_fn_image)
  const float* scale;       // [3]
  const float* base;        // [24]
  float* mul;               // [16, T, 24] split partials (scratch)
  float* sqr;               // [16, T]
  float* post_out;          // [T, 4]
  float* comb_out;          // [T, 16]
  float* pre_out;           // [T, 4]
  uint32_t* counters;       // [ceil(T/16)] arrival counters, zero at rest
  int T;
  // 1: launched on the main stream as the PDL dependent of the router (which triggers it after its own dependency
  // wait, so this kernel's inputs are complete): no dependency wait at the start; griddepcontrol.wait at the end, so
  // this kernel completes only after the router has (MegaMoE follows in stream order). 0: the side-stream launch.
  // 2: as 1, and the dependent (MegaMoE) is released at this kernel's first instruction: its prologue (cluster sync,
  // barrier init, TMEM alloc) overlaps us, and its own dependency wait still waits for this grid's completion.
  int pdl_mode;
  // optional: bytes the CTAs pull into L2 at their start (split s: [s * 122880, (s + 1) * 122880)), e.g. the attention
  // side's mix prefetching the same layer's FFN-side hc_fn image, which the FFN-side mix reads after the attention
  const uint8_t* l2_prefetch;
  long long l2_prefetch_bytes;
  unsigned long long* dbg;
};

struct RouterParams {
  // inputs
  const __nv_bfloat16* li;      // [T, 5120] FFN layer input (bf16)
  const __nv_bfloat16* gate_w;  // [384, 5120]
  const float* bias;            // [384] e_score_correction_bias
  const float* bias_vl;         // [384] image-span bias, or null (text-only routing)
  const long long* input_ids;   // [T] (needed with bias_vl)
  const int32_t* row_map;       // [384] hot row, or -(cold row + 1)
  const bool* is_padding;       // [T] or null (MegaMoE staging masks padded rows)
  float routed_scale;           // 1.5
  int T;
  int layer;
  // scratch
  float* logits;                // [T, 384] (written only for debugging / tests; may be null)
  float* cand_v;                // [T][24][6] per-cluster top-6 candidates: biased score, raw weight, expert id
  float* cand_w;
  int32_t* cand_i;
  uint32_t* sync;               // [4] zero at rest: leader arrivals, send arrivals, route-ready flag
  // outputs (any may be null except mega_*)
  long long* topk_ids;          // [T, 6] expert ids (dsv4_topk's topk_ids, int64)
  float* topk_w;                // [T, 6] (dsv4_topk's topk_weights)
  long long* hot_ids;           // [T, 6] row_map hot row or -1 (route_send2's hot ids)
  long long* mega_idx;          // [T, 6] MegaMoE staged top-k index (prepare_megamoe_inputs' topk_idx_out)
  float* mega_w;                // [T, 6] MegaMoE staged top-k weights
  int32_t* cold_ids;            // [T, 6] local cold row or -1
  int8_t* has;                  // [T] token has a cold route
  int32_t* pos;                 // [T] packed cold-row position (cumsum(has) - 1)
  int32_t* rows;                // [1] number of cold rows
  // MegaMoE activation staging of li (prepare_megamoe_inputs' x / x_sf / shared_x_sf), or prep_x == null.
  // Scale tensors are int32 with element (t, g) at t * stride_m + g * stride_k (g = 128-column group).
  uint8_t* prep_x;              // [T, 5120] fp8, row stride 5120
  int32_t* prep_sf;
  int prep_sf_sm, prep_sf_sk;
  int32_t* prep_shared_sf;      // or null
  int prep_ssf_sm, prep_ssf_sk;
  int shared_block_m;
  // optional cold-row send (peer tier v2 layout, as peer_fused2._route_send2); peer_words == null: no send.
  // cold_q / cold_sf: FlashInfer-recipe MXFP8 rows of li ([T, 5120] fp8, [T, 160] ue8m0). With quant_cold = 1 the
  // router writes them itself (scratch); otherwise they are inputs (e.g. the FFN seam's fi_q / fi_sf_lin).
  uint8_t* cold_q;
  uint8_t* cold_sf;
  int quant_cold;
  uint8_t* peer_x;
  uint8_t* peer_xs;
  int32_t* peer_ids;
  float* peer_w;
  long long* peer_words;
  long long* peer_header;
  long long* peer_seq;
  int peer_pad;
  // optional cold-row send in peer tier v3 (W6, peer3_proto.cuh): p3_seg == null: none. Exclusive with v2. Every
  // token publishes its record (word0 = tag | 6 cold-local ids, 0xFF = not cold; release, sys) after its MXFP8 row
  // and route weights (written only for tokens with a cold pick); tag = counter (p3_state[0]) | layer | T - 1.
  uint8_t* p3_seg;              // device view of the host-mapped v3 segment
  const uint32_t* p3_state;     // [2] peer3 GB300 state: [0] call counter (advanced by peer3's finish)
  // 16 < T <= 64 (router_mega.cu, the Mega gate replica): image-token mask (MegaGateRoutingMetadata's; null: none),
  // split partial logits [8][T][384] and per-expert-tile split arrival counters [24] (zero at rest)
  const bool* image_mask;
  float* mg_part;
  uint32_t* mg_count;
  // 1: order the publish with the arrival atomics / cluster barrier (gpu / cluster scope release-acquire) and one
  // system-scope release per record, without the per-CTA fence.acq_rel.sys (W6 section 6: those only cost time)
  int lean_fence;
  // 1: trigger the PDL dependent (the FFN-side mix, M3) only after this kernel's own dependency wait, not before it
  int trigger_late;
  // optional: the writer CTA stores mega_ctl_val into *mega_ctl (W6 MegaMoE fork v11's skip_combine word, read by
  // the next MegaMoE launch, which clears it after use; 1 = the post-MoE seam combines). Null: none.
  uint32_t* mega_ctl;
  uint32_t mega_ctl_val;
  unsigned long long* dbg;
};

// DSpark drafter MoE glue (draft_router.cu), 2 <= T <= 64: bit-exact replica of production's bf16_mega_gate
// (128 experts, top-3) -> prepare_megamoe_inputs, writing MegaMoE's symmetric-buffer rows directly.
struct DraftRouterParams {
  const __nv_bfloat16* li;      // [T, 5120] FFN layer input
  const __nv_bfloat16* gate_w;  // [128, 5120]
  const float* bias;            // [128] e_score_correction_bias
  const float* bias_vl;         // [128] image bias (used for tokens with image_mask), or null
  const bool* image_mask;       // [T] MegaGateRoutingMetadata.image_token_mask, or null
  const bool* is_padding;       // [T] or null: padded rows stage topk_idx -1 / weight 0
  float routed_scale;
  int T;
  long long* topk_idx;          // [T, 3] MegaMoE staged top-k index
  float* topk_w;                // [T, 3] MegaMoE staged top-k weights
  uint8_t* prep_x;              // [T, 5120] fp8
  int32_t* prep_sf;             // packed ue8m0, element (t, g) at t * prep_sf_sm + g * prep_sf_sk
  int prep_sf_sm, prep_sf_sk;
  int32_t* prep_shared_sf;      // shared-expert scale view, or null
  int prep_ssf_sm, prep_ssf_sk;
  int shared_block_m;
  float* logits;                // [T, 128] split-summed gate logits (tests), or null
  unsigned long long* dbg;
};

// DSpark drafter head (head.cu): mhc_post_tilelang (residual_new, bf16) then hc_collapse_triton (acc = 0; acc +=
// pre[i] * rn[i], separate fp32 multiply and add; bf16 store) in one launch, for any T.
struct HeadParams {
  const __nv_bfloat16* x;         // [T, 5120] the last layer's MoE output
  const __nv_bfloat16* residual;  // [T, 4, 5120]
  const float* post;              // [T, 4]
  const float* comb;              // [T, 16] comb[t][j][i] at j * 4 + i
  const float* pre;               // [T, 4]
  __nv_bfloat16* out;             // [T, 5120]
  int T;
  unsigned long long* dbg;
};

// T > 16 boundary: bit-exact replica of DeepGEMM sm100_mega_mhc_impl<5120, 40 splits, 152, shifted, bf16 out>
// (production's mhc_shifted_post_pre at T > 16) as mega_seam (residual_new + layer input) and mega_mix (next mixes).
struct MegaParams {
  const __nv_bfloat16* x;         // [T, 5120]
  const __nv_bfloat16* residual;  // [T, 4, 5120]
  const float* post;              // [T, 4]
  const float* comb;              // [T, 16] (input route i, output route o) at i * 4 + o
  const float* pre;               // [T, 4] shifted previous pre mix
  const __nv_bfloat16* norm_w;    // [5120]
  const float* fn;                // [24, 4 * 5120] hc_fn as stored (fp32; rounded to tf32 RNE in the kernel)
  const float* scale;             // [3]
  const float* base;              // [24]
  __nv_bfloat16* residual_out;    // [T, 4, 5120]
  __nv_bfloat16* y_out;           // [T, 5120]
  float* post_out;                // [T, 4]
  float* comb_out;                // [T, 16]
  float* pre_out;                 // [T, 4]
  float* gemm_part;               // scratch [40][T][24]
  float* hc_part;                 // scratch [40][T]
  uint32_t* counters;             // [ceil(T / 16)], zero at rest
  float hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, rms_eps, rms_scale;
  int sinkhorn_iters;
  int T;
  // optional L2 prefetch, issued by mega_seam at its start: the next kernel's weights (the drafter router's gate),
  // split evenly over the seam's CTAs; null: none
  const uint8_t* l2_prefetch;
  long long l2_prefetch_bytes;
  unsigned long long* dbg;
};

}  // namespace w4
