"""The production pre-attention chain of one DeepSeek-V4.1 decoder layer, op for op (vLLM af7f9488 + the ds41
prod hook: MEGA_LL_GEMM=1 ll_gemm.py, pinned FlashInfer tactics, mega attention at TP1, nvfp4_ds_mla KV cache,
MXFP4 indexer with sparse logits).

Call sites mirrored (vllm/models/deepseek_v41/attention.py):
  DeepseekV4Attention.forward
    _run_parallel_input_projections  -> execute_in_parallel(fused_wqa_wkv | compressor kv_score, weights_proj)
    _split_qkv_and_norm              -> fused_q_kv_rmsnorm_quant (Triton _q_kv_norm_quant_kernel)
    _prepare_and_attn
      project_query_and_cache_kv     -> wq_b (ll_gemm) + fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
      indexer.forward_q              -> indexer.wq_b (cute-dsl) + fused_indexer_q_rope_quant (MXFP4, bf16 weights)
    _sparse_indexer_and_attn
      SparseMQAIndexer.forward       -> fill -1, sparse_mqa_logits_paged_decode (DeepGEMM + DeepSelect + remap)
      DeepseekV4MegaAttnAttention.forward_mqa (pre-kernel part)
                                     -> positions.to(int32), compute_global_topk_indices_and_lens
Kv-source layers (2, 8, 14, 20: compressor, index-key production, dense indexer) are not covered here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .state import CAND_BLOCK_SIZE, KERNEL_BLOCK, SWA_BLOCK, TOPK, DecodeState
from .weights import EPS, IDX_DIM, IDX_HEADS, N_HEADS, HEAD_DIM, LayerWeights, Mxfp8Weight

LL_MAX_M = 8
LL_SHAPES = {(5120, 1792), (1280, 32768), (8192, 5120)}


@dataclass(frozen=True)
class ChainOut:
    q: torch.Tensor  # [T, 64, 512] bf16, mega-attention interleaved layout (what forward_mqa receives)
    qr_kv: torch.Tensor  # [T, 1792] bf16 (intermediate, for diagnosis)
    kv: torch.Tensor  # [T, 512] bf16 normed kv
    positions_i32: torch.Tensor | None
    global_idx: torch.Tensor | None  # [T, 512] int32
    global_lens: torch.Tensor | None  # [T] int32
    index_q: torch.Tensor | None  # [T, 32, 64] uint8 packed e2m1
    index_q_scale: torch.Tensor | None  # [T, 32] int32 (4 ue8m0)
    index_weights: torch.Tensor | None  # [T, 32] bf16
    weights_proj_out: torch.Tensor | None  # [T, 32] bf16
    kernel_metadata: torch.Tensor | None


def mm(wt: Mxfp8Weight, xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """FlashInferCutedslMxfp8LinearKernel.apply_weights under the prod ll_gemm patch."""
    from vllm.utils import flashinfer as vfi

    m = xq.shape[0]
    backend = "cutedsl_low_latency" if m <= LL_MAX_M and (wt.k, wt.n) in LL_SHAPES else "cute-dsl"
    return vfi.mm_mxfp8(xq, wt.w_kn, xs, wt.s_swz, out_dtype=torch.bfloat16, backend=backend)


class BaselineChain:
    """Holds the per-layer constants (weights, rope table, aux streams) and runs the chain."""

    def __init__(self, w: LayerWeights, cos_sin: torch.Tensor, index_source: bool, sparse_logits: bool,
                 first_consumer: bool):
        self.w = w
        self.cos_sin = cos_sin
        self.ratio = w.compress_ratio
        self.index_source = index_source
        self.sparse_logits = sparse_logits
        self.first_consumer = first_consumer
        self.aux = [torch.cuda.Stream() for _ in range(2)]
        self.events = [torch.cuda.Event() for _ in range(4)]
        self.kernel_metadata: torch.Tensor | None = None

    def input_projections(self, x: torch.Tensor):
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
        from vllm.utils.multi_stream_utils import execute_in_parallel

        w = self.w

        def wqa():
            xq, xs = mxfp8_e4m3_quantize(x.view(-1, 5120), is_sf_swizzled_layout=True)
            return mm(w.wqa_wkv, xq, xs)

        aux_fns = [None, None]
        if w.weights_proj is not None:
            aux_fns[1] = lambda: torch.nn.functional.linear(x, w.weights_proj)
        qr_kv, (_, idx_w) = execute_in_parallel(wqa, aux_fns, self.events[0], self.events[1:3], self.aux,
                                                enable=x.shape[0] <= 1024)
        return qr_kv, idx_w

    def run(self, x: torch.Tensor, st: DecodeState) -> ChainOut:
        from vllm.models.deepseek_v41.common.ops import (compute_global_topk_indices_and_lens,
                                                         fused_indexer_q_rope_quant)
        from vllm.models.deepseek_v41.common.ops.query_quant import fused_q_kv_rmsnorm_quant

        w = self.w
        T = x.shape[0]
        qr_kv, idx_w = self.input_projections(x)
        qr, kv = qr_kv.split([1280, 512], dim=-1)
        qa, kv = fused_q_kv_rmsnorm_quant(qr, kv, w.q_norm, w.kv_norm, EPS)
        q = mm(w.wq_b, qa.data, qa.scale).view(-1, N_HEADS, HEAD_DIM)
        swa2d = st.swa_cache.view(st.swa_cache.shape[0], -1)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, swa2d, st.swa_slots, st.positions, self.cos_sin, 0, EPS, SWA_BLOCK,
            False, True, False, True)
        index_q = index_q_scale = index_weights = None
        if self.index_source:
            iq = mm(w.idx_wq_b, qa.data, qa.scale).view(-1, IDX_HEADS, IDX_DIM)
            (index_q, index_q_scale), index_weights = fused_indexer_q_rope_quant(
                st.positions, iq, self.cos_sin, idx_w, IDX_DIM ** -0.5, IDX_HEADS ** -0.5, use_fp4=True,
                weights_out_dtype=torch.bfloat16 if self.sparse_logits else torch.float32)
            self.sparse_indexer(index_q, index_q_scale, index_weights, st)
        pos32 = gidx = glens = None
        if self.ratio > 0:
            pos32 = st.positions.to(torch.int32)
            gidx, glens = compute_global_topk_indices_and_lens(
                st.topk_local[:T], st.token_to_req, st.comp_block_table[: st.reqs],
                KERNEL_BLOCK // self.ratio, st.is_valid[:T])
        else:
            pos32 = st.positions.to(torch.int32)
        return ChainOut(q=q, qr_kv=qr_kv, kv=kv, positions_i32=pos32, global_idx=gidx, global_lens=glens,
                        index_q=index_q, index_q_scale=index_q_scale, index_weights=index_weights,
                        weights_proj_out=idx_w, kernel_metadata=self.kernel_metadata)

    def sparse_indexer(self, q_values, q_scale, weights, st: DecodeState) -> None:
        """SparseMQAIndexer.forward, decode rows only."""
        from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import sparse_mqa_logits_paged_decode
        from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view

        assert self.sparse_logits
        T = q_values.shape[0]
        buf = st.topk_local
        buf[:T] = -1
        km = None if self.first_consumer else self.kernel_metadata
        self.kernel_metadata = sparse_mqa_logits_paged_decode(
            q_values[:T].view(torch.int8).unsqueeze(1),
            q_scale[:T].unsqueeze(1),
            kv_cache_as_quant_view(st.idx_kcache, IDX_DIM, use_fp4_cache=True),
            weights[:T],
            st.idx_row_ke,
            st.idx_block_table,
            st.idx_row_indices,
            st.candidate_blocks[:T],
            CAND_BLOCK_SIZE,
            8,
            TOPK,
            buf[:T, :TOPK],
            row_ks=st.row_ks,
            sparse_indices=st.sparse_indices,
            end=st.sparse_end,
            col_indices=st.col_indices,
            kernel_metadata=km,
        )
