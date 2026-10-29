// Torch bindings for the W3 standalone stage kernels (PDL launches on the current stream).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "w3/standalone.cuh"
#include "w3/stage_wq_b.cuh"
#include "w3/stage_wqkv.cuh"
#include "w3/stage_qkv.cuh"
#include "w3/stage_kt.cuh"
#include "w3/stage_remap.cuh"
#include "w3/stage_cand.cuh"
#include "w3/stage_cand_long.cuh"
#include "w3/stage_compress.cuh"
#include "w3/host_tmap.h"

using namespace w3;

namespace {

int num_sms() {
  static int n = [] {
    int d, v;
    cudaGetDevice(&d);
    cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d);
    return v;
  }();
  return n;
}

template <class A, class B, int N>
// trigger: 1 = dependents may launch at the kernel's start, 0 = after the producer warp's weight loads, -1 = only at
// exit (a dependent, also one across a captured event join, never overlaps this kernel or what it waited for)
void launch_stage(const StagePair<A, B>& pp, int grid, int nslots, uint32_t stage_bytes, bool pdl, int trigger) {
  auto kern = stage_kernel<A, B, N>;
  const size_t smem = standalone_smem(nslots, stage_bytes);
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, pp, nslots, stage_bytes, trigger));
}

int n_mma_for(int T) {
  if (T <= 8) return 8;
  if (T <= 16) return 16;
  TORCH_CHECK(false, "T > 16 not yet supported by the W3 fast path");
}

template <class A, class B>
void dispatch_n(int N, const StagePair<A, B>& pp, int grid, int nslots, uint32_t sb, bool pdl, bool early) {
  switch (N) {
    case 8: launch_stage<A, B, 8>(pp, grid, nslots, sb, pdl, early); break;
    case 16: launch_stage<A, B, 16>(pp, grid, nslots, sb, pdl, early); break;
    default: TORCH_CHECK(false, "N");
  }
}

template <class T>
T* ptr(const c10::optional<torch::Tensor>& t) {
  return t.has_value() ? reinterpret_cast<T*>(t->data_ptr()) : nullptr;
}

}  // namespace

// q[T, n_out] (bf16) = xq[T, K] (e4m3) @ w[n_out, K]^T, MXFP8 F8_128x4 scales, sequential K (wq_b semantics).
void wq_b_gemm(torch::Tensor w_nk, torch::Tensor w_sf, torch::Tensor xq, torch::Tensor x_sf, torch::Tensor out,
               int64_t nslots, bool pdl) {
  const int n_out = w_nk.size(0), K = w_nk.size(1), T = xq.size(0);
  TORCH_CHECK(n_out % 128 == 0 && K % 128 == 0 && xq.size(1) == K && out.size(0) == T && out.size(1) == n_out);
  StagePair<WqBStage, NoStage> pp{};
  auto& p = pp.a;
  p.tm_w = make_tmap_2d_u8(w_nk.data_ptr(), n_out, K, K, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  p.w_sf = w_sf.data_ptr<uint8_t>();
  p.xq = reinterpret_cast<const uint8_t*>(xq.data_ptr());
  p.x_sf = x_sf.data_ptr<uint8_t>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.T = T;
  p.n_out = n_out;
  p.k_groups = K / 128;
  const int N = n_mma_for(T);
  dispatch_n(N, pp, std::min(WqBStage::tasks(p), num_sms()), nslots, WqBStage::smem_bytes(N, p.k_groups), pdl, true);
}

// fp32 partials of x @ [wq_a; wkv]^T split over K as production does + (optionally) the kt stage.
void wqkv_kt(torch::Tensor w_nk, torch::Tensor w_sf, torch::Tensor x, torch::Tensor part, std::vector<int64_t> kb,
             c10::optional<torch::Tensor> topk, c10::optional<torch::Tensor> token_to_req,
             c10::optional<torch::Tensor> block_table, int64_t block_size, c10::optional<torch::Tensor> is_valid,
             c10::optional<torch::Tensor> positions, c10::optional<torch::Tensor> gidx,
             c10::optional<torch::Tensor> glens, c10::optional<torch::Tensor> pos32, int64_t nslots, bool pdl,
             int64_t l2_prefetch) {
  const int n_out = w_nk.size(0), K = w_nk.size(1), T = x.size(0);
  const int nsplit = static_cast<int>(kb.size()) - 1;
  TORCH_CHECK(nsplit >= 1 && nsplit <= 4 && kb.front() == 0 && kb.back() == K / 128);
  StagePair<WqkvStage, KtStage> pp{};
  auto& a = pp.a;
  a.tm_w = make_tmap_2d_u8(w_nk.data_ptr(), n_out, K, K, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  a.w_sf = w_sf.data_ptr<uint8_t>();
  a.w = w_nk.data_ptr<uint8_t>();
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.part = part.data_ptr<float>();
  a.T = T;
  a.n_out = n_out;
  a.k_groups = K / 128;
  a.nsplit = nsplit;
  a.max_chunks = 0;
  a.l2_prefetch = l2_prefetch;
  for (int i = 0; i <= nsplit; ++i) a.kb[i] = static_cast<int>(kb[i]);
  for (int i = 0; i < nsplit; ++i) a.max_chunks = std::max(a.max_chunks, a.kb[i + 1] - a.kb[i]);
  auto& b = pp.b;
  b.T = (topk.has_value() || pos32.has_value()) ? T : 0;
  b.topk = ptr<int>(topk);
  b.topk_stride = topk.has_value() ? topk->stride(0) : 0;
  b.topk_n = 512;
  b.token_to_req = ptr<int>(token_to_req);
  b.block_table = ptr<int>(block_table);
  b.bt_stride = block_table.has_value() ? block_table->stride(0) : 0;
  b.block_size = block_size;
  b.is_valid = ptr<bool>(is_valid);
  b.positions = ptr<long long>(positions);
  b.gidx = ptr<int>(gidx);
  b.glens = ptr<int>(glens);
  b.pos32 = ptr<int>(pos32);
  const int N = n_mma_for(T);
  const int grid = WqkvStage::tasks(a) + KtStage::tasks(b);
  dispatch_n(N, pp, grid, nslots, WqkvStage::smem_bytes(N, a.max_chunks), pdl, true);
}

void qkv(torch::Tensor part, int64_t nsplit, torch::Tensor q_w, torch::Tensor kv_w, double eps, torch::Tensor qx8,
         torch::Tensor qx8_sf, c10::optional<torch::Tensor> qr_kv_out, c10::optional<torch::Tensor> kv_out,
         torch::Tensor swa_cache, torch::Tensor slot_mapping, torch::Tensor positions, torch::Tensor cos_sin,
         int64_t swa_block_size, bool pdl) {
  StagePair<QkvStage, NoStage> pp{};
  auto& p = pp.a;
  p.part = part.data_ptr<float>();
  p.nsplit = nsplit;
  p.qr_kv_in = nullptr;
  p.ld_in = 0;
  p.T = qx8.size(0);
  p.q_w = reinterpret_cast<const __nv_bfloat16*>(q_w.data_ptr());
  p.kv_w = reinterpret_cast<const __nv_bfloat16*>(kv_w.data_ptr());
  p.eps = static_cast<float>(eps);
  p.qx8 = reinterpret_cast<uint8_t*>(qx8.data_ptr());
  p.qx8_sf = qx8_sf.data_ptr<uint8_t>();
  p.qr_kv_out = ptr<__nv_bfloat16>(qr_kv_out);
  p.kv_out = ptr<__nv_bfloat16>(kv_out);
  p.swa_cache = swa_cache.data_ptr<uint8_t>();
  p.swa_block_stride = swa_cache.stride(0);
  p.slot_mapping = reinterpret_cast<const long long*>(slot_mapping.data_ptr());
  p.num_insert = slot_mapping.numel();
  p.zero_pad_sf = 0;
  p.split_roles = 0;
  p.positions = reinterpret_cast<const long long*>(positions.data_ptr());
  p.cos_sin = cos_sin.data_ptr<float>();
  p.swa_block_size = swa_block_size;
  launch_stage<QkvStage, NoStage, 8>(pp, p.T, 0, QkvStage::smem_bytes(8, 0), pdl, true);
}

void set_trace(c10::optional<torch::Tensor> buf) {
#ifdef W3_TRACE
  unsigned long long* p = buf.has_value() ? reinterpret_cast<unsigned long long*>(buf->data_ptr()) : nullptr;
  C10_CUDA_CHECK(cudaMemcpyToSymbol(g_trace, &p, sizeof(p)));
#endif
}

// Production-GEMM variant: qr_kv (bf16, from fused_wqa_wkv) -> q/kv norm + MXFP8 q + SWA insert, plus the kt stage.
void qkv_kt(torch::Tensor qr_kv, torch::Tensor q_w, torch::Tensor kv_w, double eps, torch::Tensor qx8,
            torch::Tensor qx8_sf, c10::optional<torch::Tensor> kv_out, torch::Tensor swa_cache, torch::Tensor slot_mapping,
            torch::Tensor positions, torch::Tensor cos_sin, int64_t swa_block_size, c10::optional<torch::Tensor> topk,
            c10::optional<torch::Tensor> token_to_req, c10::optional<torch::Tensor> block_table, int64_t block_size,
            c10::optional<torch::Tensor> is_valid, c10::optional<torch::Tensor> gidx, c10::optional<torch::Tensor> glens,
            c10::optional<torch::Tensor> pos32, bool pdl) {
  TORCH_CHECK(qr_kv.stride(1) == 1 && qr_kv.size(1) >= 1792);
  const int T = qx8.size(0);
  StagePair<QkvStage, KtStage> pp{};
  auto& p = pp.a;
  p.part = nullptr;
  p.nsplit = 1;
  p.qr_kv_in = reinterpret_cast<const __nv_bfloat16*>(qr_kv.data_ptr());
  p.ld_in = qr_kv.stride(0);
  p.T = T;
  p.q_w = reinterpret_cast<const __nv_bfloat16*>(q_w.data_ptr());
  p.kv_w = reinterpret_cast<const __nv_bfloat16*>(kv_w.data_ptr());
  p.eps = static_cast<float>(eps);
  p.qx8 = reinterpret_cast<uint8_t*>(qx8.data_ptr());
  p.qx8_sf = qx8_sf.data_ptr<uint8_t>();
  p.qr_kv_out = nullptr;
  p.kv_out = ptr<__nv_bfloat16>(kv_out);
  p.swa_cache = swa_cache.data_ptr<uint8_t>();
  p.swa_block_stride = swa_cache.stride(0);
  p.slot_mapping = reinterpret_cast<const long long*>(slot_mapping.data_ptr());
  p.num_insert = slot_mapping.numel();
  p.zero_pad_sf = 1;
  p.split_roles = 1;
  p.positions = reinterpret_cast<const long long*>(positions.data_ptr());
  p.cos_sin = cos_sin.data_ptr<float>();
  p.swa_block_size = swa_block_size;
  auto& b = pp.b;
  b.T = (topk.has_value() || pos32.has_value()) ? T : 0;
  b.topk = ptr<int>(topk);
  b.topk_stride = topk.has_value() ? topk->stride(0) : 0;
  b.topk_n = 512;
  b.token_to_req = ptr<int>(token_to_req);
  b.block_table = ptr<int>(block_table);
  b.bt_stride = block_table.has_value() ? block_table->stride(0) : 0;
  b.block_size = block_size;
  b.is_valid = ptr<bool>(is_valid);
  b.positions = reinterpret_cast<const long long*>(positions.data_ptr());
  b.gidx = ptr<int>(gidx);
  b.glens = ptr<int>(glens);
  b.pos32 = ptr<int>(pos32);
  launch_stage<QkvStage, KtStage, 8>(pp, QkvStage::tasks(p) + b.T, 0, QkvStage::smem_bytes(8, 0), pdl, true);
}

// Sparse top-k columns -> request-local positions (topk buffer) -> attention key table (+ int32 positions).
void remap_global(torch::Tensor cols, torch::Tensor sparse_idx, torch::Tensor row_ks, int64_t width, int64_t sbk,
                  torch::Tensor topk_out, torch::Tensor token_to_req, torch::Tensor block_table, int64_t block_size,
                  torch::Tensor is_valid, torch::Tensor gidx, torch::Tensor glens, torch::Tensor positions,
                  c10::optional<torch::Tensor> pos32, bool pdl, bool trigger_at_exit) {
  StagePair<RemapStage, NoStage> pp{};
  auto& p = pp.a;
  p.T = gidx.size(0);
  p.cols = cols.data_ptr<int>();
  p.col_stride = cols.stride(0);
  p.sparse_idx = sparse_idx.data_ptr<int>();
  p.si_stride = sparse_idx.stride(0);
  p.row_ks = row_ks.data_ptr<int>();
  p.width = width;
  p.sbk = sbk;
  p.topk_n = gidx.size(1);
  p.topk_out = topk_out.data_ptr<int>();
  p.topk_stride = topk_out.stride(0);
  p.token_to_req = token_to_req.data_ptr<int>();
  p.block_table = block_table.data_ptr<int>();
  p.bt_stride = block_table.stride(0);
  p.block_size = block_size;
  p.is_valid = is_valid.data_ptr<bool>();
  p.gidx = gidx.data_ptr<int>();
  p.glens = glens.data_ptr<int>();
  p.positions = reinterpret_cast<const long long*>(positions.data_ptr());
  p.pos32 = ptr<int>(pos32);
  launch_stage<RemapStage, NoStage, 8>(pp, p.T, 0, RemapStage::smem_bytes(8, 0), pdl, trigger_at_exit ? -1 : 1);
}

// Candidate selection (layer 20). scratch = null: one launch (CandStage; keys cached in smem, or gkeys past 49152
// blocks). scratch = int32 [cand_scratch_size(rows, width, block_size, k)]: the three-stage long-row path (needs gkeys).
// Compressor state save + pooling + RMSNorm (_fused_save_compress_norm_kernel), bitwise; same arguments.
void compress(torch::Tensor kv_score, torch::Tensor positions, c10::optional<torch::Tensor> state, torch::Tensor slots,
              c10::optional<torch::Tensor> qsl, c10::optional<torch::Tensor> req_ids, torch::Tensor norm_w, double eps,
              int64_t ratio, torch::Tensor latent, bool pdl) {
  TORCH_CHECK(ratio == 1 || ratio == 2);
  TORCH_CHECK(kv_score.dtype() == torch::kFloat32 && kv_score.stride(1) == 1 && kv_score.size(1) == 512 * ratio &&
              kv_score.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(kv_score.data_ptr()) % 16 == 0);
  TORCH_CHECK(positions.dtype() == torch::kInt64 && positions.is_contiguous());
  TORCH_CHECK(slots.dtype() == torch::kInt64 && slots.is_contiguous());
  TORCH_CHECK(norm_w.dtype() == torch::kBFloat16 && norm_w.is_contiguous() && norm_w.numel() == 512);
  TORCH_CHECK(latent.dtype() == torch::kBFloat16 && latent.is_contiguous() && latent.size(1) == 512);
  StagePair<CompressStage, NoStage> pp{};
  auto& p = pp.a;
  p.raw = kv_score.data_ptr<float>();
  p.raw_stride = kv_score.stride(0);
  p.positions = reinterpret_cast<const long long*>(positions.data_ptr());
  p.slots = reinterpret_cast<const long long*>(slots.data_ptr());
  p.norm_w = reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr());
  p.latent = reinterpret_cast<__nv_bfloat16*>(latent.data_ptr());
  p.num_tokens = slots.numel();
  p.ratio = ratio;
  p.eps = static_cast<float>(eps);
  if (ratio == 2) {
    TORCH_CHECK(state.has_value() && qsl.has_value() && req_ids.has_value());
    TORCH_CHECK(state->dtype() == torch::kFloat32 && state->dim() == 3 && state->size(2) == 1024 && state->stride(2) == 1 &&
                state->stride(1) % 4 == 0 && state->stride(0) % 4 == 0 && state->size(1) <= CompressStage::kMaxRing);
    TORCH_CHECK(qsl->dtype() == torch::kInt32 && qsl->is_contiguous() && req_ids->dtype() == torch::kInt32 &&
                req_ids->is_contiguous());
    p.state = state->data_ptr<float>();
    p.state_stride = state->stride(0);
    p.state_row_stride = state->stride(1);
    p.ring = state->size(1);
    p.qsl = qsl->data_ptr<int>();
    p.req_ids = req_ids->data_ptr<int>();
    p.num_reqs = qsl->numel() - 1;
  }
  const int grid = CompressStage::tasks(p);
  if (p.num_tokens == 0 || grid == 0) return;
  launch_stage<CompressStage, NoStage, 8>(pp, grid, 0, CompressStage::smem_bytes(0, 0), pdl, true);
}

int64_t cand_scratch_size(int64_t rows, int64_t width, int64_t block_size, int64_t k) {
  using namespace cand_long;
  const int64_t nb = (width + block_size - 1) / block_size, nch = (nb + kChunk - 1) / kChunk;
  const int64_t nkc = (nb + kKeyChunk - 1) / kKeyChunk;
  return rows * nkc * (kBins / 2 + 64) + rows * nch * (2 + 512) + rows * 4 + (rows * nch * k + 1) / 2;
}

void cand_select(torch::Tensor logits, torch::Tensor row_end, int64_t block_size, torch::Tensor out,
                 c10::optional<torch::Tensor> sparse_idx, c10::optional<torch::Tensor> sparse_end,
                 c10::optional<torch::Tensor> cons_ke, c10::optional<torch::Tensor> gkeys,
                 c10::optional<torch::Tensor> scratch, int64_t row_repeat, bool pdl) {
  TORCH_CHECK(logits.dtype() == torch::kFloat32 && logits.stride(1) == 1);
  TORCH_CHECK(row_repeat >= 1 && row_end.is_contiguous() && row_end.numel() * row_repeat >= logits.size(0));
  const int rr = static_cast<int>(row_repeat);
  TORCH_CHECK(row_end.dtype() == torch::kInt32 && out.dtype() == torch::kInt32 && out.stride(1) == 1);
  const int rows = logits.size(0), width = logits.size(1), k = out.size(1);
  const int nb = static_cast<int>((width + block_size - 1) / block_size);
  uint32_t* gk = gkeys.has_value() ? reinterpret_cast<uint32_t*>(gkeys->data_ptr()) : nullptr;
  const int gks = gkeys.has_value() ? gkeys->stride(0) : 0;
  if (scratch.has_value()) {
    using namespace cand_long;
    const int nch = (nb + kChunk - 1) / kChunk, nkc = (nb + kKeyChunk - 1) / kKeyChunk;
    TORCH_CHECK(gk && gks >= nb, "long-row path needs gkeys [rows, >= width / block_size]");
    TORCH_CHECK(!sparse_idx.has_value(), "long-row path: no fused consumer expansion");
    TORCH_CHECK(nch <= kMaxChunks && k <= 2048 && k % 8 == 0 && scratch->dtype() == torch::kInt32 && scratch->is_contiguous() &&
                scratch->numel() >= cand_scratch_size(rows, width, block_size, k));
    // layout keeps the bulk-copied above ids 16 B aligned: hist (u16) | coarse | above ids | chunk meta | row meta;
    // the bin entries overwrite each chunk's keys in gkeys
    uint16_t* hist = reinterpret_cast<uint16_t*>(scratch->data_ptr());
    uint32_t* coarse = reinterpret_cast<uint32_t*>(hist + static_cast<int64_t>(rows) * nkc * kBins);
    uint16_t* above_ids = reinterpret_cast<uint16_t*>(coarse + static_cast<int64_t>(rows) * nkc * 64);
    uint16_t* hist2 = above_ids + static_cast<int64_t>(rows) * nch * k;
    int* chunk_meta = reinterpret_cast<int*>(hist2 + static_cast<int64_t>(rows) * nch * 1024);
    int* row_meta = chunk_meta + static_cast<int64_t>(rows) * nch * 2;
    TORCH_CHECK(reinterpret_cast<uintptr_t>(hist) % 16 == 0 && reinterpret_cast<uintptr_t>(gk) % 16 == 0 && gks % 4 == 0);
    TORCH_CHECK(block_size == 8 && logits.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0,
                "long-row path: block_size 8, 16 B aligned logits rows");
    const int bs = static_cast<int>(block_size);
    int* re = row_end.data_ptr<int>();
    StagePair<CandKeysStage, NoStage> s1{};
    s1.a = {logits.data_ptr<float>(), logits.stride(0), width, re, rr, bs, gk, gks, hist, coarse, nkc, rows,
            k, out.data_ptr<int>(), static_cast<int>(out.stride(0))};
    launch_stage<CandKeysStage, NoStage, 8>(s1, CandKeysStage::tasks(s1.a), 0, CandKeysStage::smem_bytes(0, 0), pdl, true);
    StagePair<CandSplitStage, NoStage> s2{};
    s2.a = {gk, gks, hist, coarse, nkc, nch, re, rr, width, bs, k, above_ids, hist2, chunk_meta, row_meta, rows};
    launch_stage<CandSplitStage, NoStage, 8>(s2, CandSplitStage::tasks(s2.a), 0, CandSplitStage::smem_bytes(0, 0), true, true);
    StagePair<CandFinalStage, NoStage> s3{};
    s3.a = {above_ids, hist2, gk, gks, chunk_meta, row_meta, nch, re, rr, width, bs, k, out.data_ptr<int>(),
            static_cast<int>(out.stride(0)), rows};
    launch_stage<CandFinalStage, NoStage, 8>(s3, rows, 0, CandFinalStage::smem_bytes(0, 0), true, true);
    return;
  }
  StagePair<CandStage, NoStage> pp{};
  auto& p = pp.a;
  p.logits = logits.data_ptr<float>();
  p.ld = logits.stride(0);
  p.width = width;
  p.row_end = row_end.data_ptr<int>();
  p.row_repeat = rr;
  p.block_size = block_size;
  p.k = out.size(1);
  p.out = out.data_ptr<int>();
  p.out_stride = out.stride(0);
  p.sparse_idx = ptr<int>(sparse_idx);
  p.si_stride = sparse_idx.has_value() ? sparse_idx->stride(0) : 0;
  p.sparse_end = ptr<int>(sparse_end);
  p.cons_ke = ptr<int>(cons_ke);
  p.rows = rows;
  // key cache: the whole valid prefix when it fits (<= 192 KB of smem), else global scratch / recompute per pass
  p.cache_keys = std::min(nb, 49152);
  p.gkeys = gk;
  p.gkeys_stride = gks;
  launch_stage<CandStage, NoStage, 8>(pp, p.rows, 0, CandStage::smem_bytes(p.cache_keys, 0), pdl, true);
}

// The attention key table alone (memoized per step per index-source group by the caller) + int32 positions.
void kt_table(torch::Tensor topk, torch::Tensor token_to_req, torch::Tensor block_table, int64_t block_size,
              torch::Tensor is_valid, torch::Tensor positions, torch::Tensor gidx, torch::Tensor glens,
              c10::optional<torch::Tensor> pos32, bool pdl) {
  StagePair<KtStage, NoStage> pp{};
  auto& b = pp.a;
  b.T = gidx.size(0);
  b.topk = topk.data_ptr<int>();
  b.topk_stride = topk.stride(0);
  b.topk_n = gidx.size(1);
  b.token_to_req = token_to_req.data_ptr<int>();
  b.block_table = block_table.data_ptr<int>();
  b.bt_stride = block_table.stride(0);
  b.block_size = block_size;
  b.is_valid = is_valid.data_ptr<bool>();
  b.positions = reinterpret_cast<const long long*>(positions.data_ptr());
  b.gidx = gidx.data_ptr<int>();
  b.glens = glens.data_ptr<int>();
  b.pos32 = ptr<int>(pos32);
  launch_stage<KtStage, NoStage, 8>(pp, b.T, 0, KtStage::smem_bytes(8, 0), pdl, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kt_table", &kt_table);
  m.def("cand_select", &cand_select);
  m.def("cand_scratch_size", &cand_scratch_size);
  m.def("compress", &compress);
  m.def("qkv_kt", &qkv_kt);
  m.def("remap_global", &remap_global);
  m.def("set_trace", &set_trace);
  m.def("wq_b_gemm", &wq_b_gemm);
  m.def("wqkv_kt", &wqkv_kt);
  m.def("qkv", &qkv);
}
