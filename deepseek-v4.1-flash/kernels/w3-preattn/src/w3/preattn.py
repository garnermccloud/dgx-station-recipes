"""preattn fastpath v1: the production GEMMs (FlashInfer LL / cute-dsl, untouched) with everything between and
around them replaced by W3 kernels. Same inputs/outputs as w3.baseline.BaselineChain.

Per layer (decode, T <= 16):
  production : mxfp8 quant(x) -> LL fused_wqa_wkv  (|| weights_proj on aux stream for index sources)
  W3 qkv_kt  : one launch (PDL): q/kv RMSNorm (Triton order), MXFP8 q, RoPE + MXFP8 SWA-cache insert, and for
               compressed consumer layers the attention key table (global top-k slots + lens) + int32 positions.
               Replaces _q_kv_norm_quant_kernel, fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel (which ran after
               wq_b), positions.to(int32) and _compute_global_topk_indices_and_lens.
  production : LL wq_b
All outputs are bitwise identical to the production chain (tests/test_preattn.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .baseline import BaselineChain, ChainOut, mm
from .fused import ext
from .state import KERNEL_BLOCK, SWA_BLOCK, DecodeState
from .weights import EPS, HEAD_DIM, N_HEADS, LayerWeights


@dataclass
class PreattnBuffers:
    qx8: torch.Tensor
    qx8_sf: torch.Tensor
    gidx: torch.Tensor | None
    glens: torch.Tensor | None
    pos32: torch.Tensor
    kv: torch.Tensor | None


def make_buffers(T: int, compressed: bool, device: torch.device, debug: bool = False) -> PreattnBuffers:
    return PreattnBuffers(
        qx8=torch.empty(T, 1280, dtype=torch.uint8, device=device),
        qx8_sf=torch.zeros(((T + 127) // 128) * 128 * 40, dtype=torch.uint8, device=device),
        gidx=torch.empty(T, 512, dtype=torch.int32, device=device) if compressed else None,
        glens=torch.empty(T, dtype=torch.int32, device=device) if compressed else None,
        pos32=torch.empty(T, dtype=torch.int32, device=device),
        kv=torch.empty(T, 512, dtype=torch.bfloat16, device=device) if debug else None,
    )


def qkv_kt(qr_kv: torch.Tensor, w: LayerWeights, cos_sin: torch.Tensor, st: DecodeState, buf: PreattnBuffers,
           with_kt: bool, pdl: bool = True) -> None:
    """The fused W3 launch between the two production GEMMs."""
    T = qr_kv.shape[0]
    ratio = w.compress_ratio
    kt = with_kt and ratio > 0
    ext().qkv_kt(qr_kv, w.q_norm, w.kv_norm, EPS, buf.qx8, buf.qx8_sf, buf.kv,
                 st.swa_cache.view(st.swa_cache.shape[0], -1), st.swa_slots, st.positions, cos_sin, SWA_BLOCK,
                 st.topk_local[:T] if kt else None, st.token_to_req if kt else None,
                 st.comp_block_table[: st.reqs] if kt else None, KERNEL_BLOCK // max(ratio, 1),
                 st.is_valid if kt else None, buf.gidx if kt else None, buf.glens if kt else None, buf.pos32, pdl)


def kt_table(w: LayerWeights, st: DecodeState, buf: PreattnBuffers, pdl: bool = True) -> None:
    """Attention key table (global top-k slots + lens) + int32 positions for compressed consumer layers.
    Identical for every layer of an index-source group in a step (#11), so the integration memoizes it."""
    T = buf.pos32.shape[0]
    ext().kt_table(st.topk_local[:T], st.token_to_req, st.comp_block_table[: st.reqs], KERNEL_BLOCK // w.compress_ratio,
                   st.is_valid, st.positions, buf.gidx, buf.glens, buf.pos32, pdl)


class PreattnChain(BaselineChain):
    """Production GEMMs + W3 qkv for the non-index layers (A / B / E types).

    kt_mode: "memo" = the key table comes from the step memo (computed once per index-source group; not in this
    layer's chain), "side" = computed here on a side stream joined at the end, "inline" = inside the qkv launch.
    """

    def __init__(self, w: LayerWeights, cos_sin: torch.Tensor, T: int, debug: bool = False, kt_mode: str = "side"):
        super().__init__(w, cos_sin, index_source=False, sparse_logits=False, first_consumer=False)
        self.buf = make_buffers(T, w.compress_ratio > 0, w.q_norm.device, debug)
        self.kt_mode = kt_mode if w.compress_ratio > 0 else "none"
        self.side = torch.cuda.Stream()
        self.ev = [torch.cuda.Event() for _ in range(2)]

    def run(self, x: torch.Tensor, st: DecodeState) -> ChainOut:
        w, b = self.w, self.buf
        if self.kt_mode == "side":
            self.ev[0].record()
            with torch.cuda.stream(self.side):
                self.ev[0].wait()
                kt_table(w, st, b, pdl=False)
                self.ev[1].record()
        qr_kv, _ = self.input_projections(x)
        qkv_kt(qr_kv, w, self.cos_sin, st, b, with_kt=self.kt_mode == "inline")
        q = mm(w.wq_b, b.qx8.view(torch.float8_e4m3fn), b.qx8_sf).view(-1, N_HEADS, HEAD_DIM)
        if self.kt_mode == "side":
            self.ev[1].wait()
        return ChainOut(q=q, qr_kv=qr_kv, kv=b.kv, positions_i32=b.pos32, global_idx=b.gidx, global_lens=b.glens,
                        index_q=None, index_q_scale=None, index_weights=None, weights_proj_out=None,
                        kernel_metadata=None)


class PreattnIndexChain(BaselineChain):
    """Candidate-consumer index-source layers (24/28/32/36): production GEMMs, production DeepGEMM sparse logits and
    DeepSelect top-k, rescheduled:
      main : quant -> LL wqa_wkv (|| weights_proj) -> W3 qkv_kt (norm/quant/insert, no key table yet) -> LL wq_b
      side : (after qkv_kt) idx wq_b -> fused_indexer_q_rope_quant -> [candidate expansion + DeepGEMM schedule on the
             first consumer] -> paged sparse MQA logits -> DeepSelect -> W3 remap_global (topk buffer + key table)
    The topk_indices_buffer[:T] = -1 fill is dropped (remap_global writes every column of rows [0, T)).
    """

    def __init__(self, w: LayerWeights, cos_sin: torch.Tensor, T: int, first_consumer: bool, debug: bool = False,
                 side_priority: int = 0):
        super().__init__(w, cos_sin, index_source=True, sparse_logits=True, first_consumer=first_consumer)
        self.buf = make_buffers(T, True, w.q_norm.device, debug)
        self.side = torch.cuda.Stream(priority=side_priority)
        self.ev = [torch.cuda.Event() for _ in range(2)]

    def indexer_branch(self, st: DecodeState, idx_w: torch.Tensor):
        from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import candidate_blocks_to_sparse_indices
        from vllm.model_executor.layers.indexer_topk import deep_select_topk
        from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
        from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant
        from vllm.utils.deep_gemm import fp8_fp4_paged_sparse_mqa_logits, get_paged_sparse_mqa_logits_metadata

        from .state import CAND_BLOCK_SIZE, TOPK
        from .weights import IDX_DIM, IDX_HEADS

        w, b = self.w, self.buf
        T = b.qx8.shape[0]
        iq = mm(w.idx_wq_b, b.qx8.view(torch.float8_e4m3fn), b.qx8_sf).view(-1, IDX_HEADS, IDX_DIM)
        (qv, qs), weights = fused_indexer_q_rope_quant(st.positions, iq, self.cos_sin, idx_w, IDX_DIM ** -0.5,
                                                       IDX_HEADS ** -0.5, use_fp4=True, weights_out_dtype=torch.bfloat16)
        kv_view = kv_cache_as_quant_view(st.idx_kcache, IDX_DIM, use_fp4_cache=True)
        if self.first_consumer or self.kernel_metadata is None:
            candidate_blocks_to_sparse_indices(st.candidate_blocks[:T], st.row_ks, st.idx_row_ke, CAND_BLOCK_SIZE, 8,
                                               out=(st.sparse_indices, st.sparse_end))
            self.kernel_metadata = get_paged_sparse_mqa_logits_metadata(
                st.idx_row_ke, st.idx_block_table, st.idx_row_indices, kv_view.shape[1], st.sparse_indices,
                qv.view(torch.int8).dtype, 8)
        logits = fp8_fp4_paged_sparse_mqa_logits((qv[:T].view(torch.int8).unsqueeze(1), qs[:T].unsqueeze(1)), kv_view,
                                                 weights[:T], self.kernel_metadata, st.sparse_indices.shape[1], 8)
        cols = st.col_indices[:T]
        deep_select_topk(logits, TOPK, end=st.sparse_end, output_idx=cols)
        ext().remap_global(cols, st.sparse_indices, st.row_ks, logits.shape[1], 8, st.topk_local,
                           st.token_to_req, st.comp_block_table[: st.reqs], KERNEL_BLOCK // w.compress_ratio,
                           st.is_valid, b.gidx, b.glens, st.positions, None, True)
        return qv, qs, weights

    def run(self, x: torch.Tensor, st: DecodeState) -> ChainOut:
        w, b = self.w, self.buf
        qr_kv, idx_w = self.input_projections(x)
        qkv_kt(qr_kv, w, self.cos_sin, st, b, with_kt=False)
        self.ev[0].record()
        with torch.cuda.stream(self.side):
            self.ev[0].wait()
            qv, qs, weights = self.indexer_branch(st, idx_w)
            self.ev[1].record()
        q = mm(w.wq_b, b.qx8.view(torch.float8_e4m3fn), b.qx8_sf).view(-1, N_HEADS, HEAD_DIM)
        self.ev[1].wait()
        return ChainOut(q=q, qr_kv=qr_kv, kv=b.kv, positions_i32=b.pos32, global_idx=b.gidx, global_lens=b.glens,
                        index_q=qv, index_q_scale=qs, index_weights=weights, weights_proj_out=idx_w,
                        kernel_metadata=self.kernel_metadata)
