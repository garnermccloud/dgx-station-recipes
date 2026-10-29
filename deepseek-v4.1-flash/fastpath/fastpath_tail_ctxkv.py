"""tail_ctxkv: the DSpark drafter's context-KV projection for all three draft layers as one quantization + one GEMM.

DSparkDeepseekV4Model.precompute_and_store_context_kv runs, per draft layer, the full fused_wqa_wkv projection of
the same main_x (MXFP8 quantize + LL GEMM with N = 1280 + 512) and keeps only the 512 wkv columns, then kv_norm and
the RoPE/quant/insert op: 3 x ~13.6 us at the head of the drafter graph. This component quantizes main_x once and
multiplies it by the three layers' wkv rows stacked into one [1536, 5120] MXFP8 weight (row slices of the layers'
own processed weights and swizzled scales, copied once in load_draft_model: 7.9 MB + 0.25 MB), then runs each
layer's production kv_norm and insert on its 512-column slice.

Same quantized activations, same weight bytes, same kernel family (FlashInfer mm_mxfp8, cutedsl_low_latency at
M <= 8, cute-dsl above); only the GEMM's N changes, so its split-K schedule may differ and a kv value can move by a
BF16 rounding step. The context KV feeds only the drafter's attention: a DRAFT-ONLY change (acceptance only); the
measured difference against production's per-layer GEMMs is in tests/test_ctxkv.py.
"""
from __future__ import annotations

import sys

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

SPECULATOR_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
DSPARK_MODEL_MODULE = "vllm.models.deepseek_v41.nvidia.dspark"
LL_MAX_M = 8


class StackedWkv:
    """One MXFP8 weight holding the wkv rows of every draft layer's fused_wqa_wkv."""

    def __init__(self, layers):
        import torch

        attn0 = layers[0].attn
        self.q_lora_rank = attn0.q_lora_rank
        self.head_dim = attn0.head_dim
        rows, scales = [], []
        for layer in layers:
            lin = layer.attn.fused_wqa_wkv
            w = lin.weight  # [K, N] column-major view of an [N, K] row-major tensor
            K, N = w.shape
            store = w.t()
            assert store.is_contiguous() and N == self.q_lora_rank + self.head_dim
            assert self.q_lora_rank % 128 == 0 and N % 128 == 0, "row slice must fall on 128-row scale tiles"
            per_row = lin.weight_scale.numel() // N  # bytes of swizzled scales per weight row
            assert lin.weight_scale.numel() == N * per_row
            rows.append(store[self.q_lora_rank:])
            scales.append(lin.weight_scale.reshape(-1)[self.q_lora_rank * per_row:])
        self.store = torch.cat(rows).contiguous()  # [L * head_dim, K]
        self.weight = self.store.t()  # [K, L * head_dim] column-major, as mm_mxfp8 takes operand B
        self.scale = torch.cat(scales).contiguous()
        self.num_layers = len(layers)
        self.nbytes = self.store.numel() * self.store.element_size() + self.scale.numel() * self.scale.element_size()

    def project(self, main_x):
        """[T, L * head_dim] BF16 = main_x @ stacked wkv^T, production's MXFP8 path."""
        import torch

        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
        from vllm.utils import flashinfer as vfi

        K = self.weight.shape[0]
        x = main_x.view(-1, K)
        xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        backend = "cutedsl_low_latency" if x.shape[0] <= LL_MAX_M else "cute-dsl"
        return vfi.mm_mxfp8(xq, self.weight, xs, self.scale, out_dtype=torch.bfloat16, backend=backend)


def unsupported_reason(draft_model) -> str | None:
    import torch

    inner = getattr(draft_model, "model", None)
    layers = getattr(inner, "layers", None)
    if not layers:
        return "no draft layers"
    for layer in layers:
        attn = layer.attn
        lin = getattr(attn, "fused_wqa_wkv", None)
        w = getattr(lin, "weight", None)
        if w is None or w.dtype != torch.float8_e4m3fn or getattr(lin, "weight_scale", None) is None:
            return "fused_wqa_wkv is not an MXFP8 linear"
        if tuple(w.shape) != (attn.hidden_size if hasattr(attn, "hidden_size") else w.shape[0],
                              attn.q_lora_rank + attn.head_dim) or not w.t().is_contiguous():
            return f"fused_wqa_wkv weight {tuple(w.shape)} is not the column-major [K, q_lora + head_dim] layout"
    return None


def precompute_fused(self, main_x, context_positions, context_slot_mappings, insert_fn):
    st: StackedWkv = self._w5_ctxkv
    kv_all = st.project(main_x)
    d = st.head_dim
    for i, layer in enumerate(self.layers):
        slot_mapping = None if context_slot_mappings is None else context_slot_mappings[i]
        kv = layer.attn.kv_norm(kv_all[..., i * d:(i + 1) * d])
        if slot_mapping is None:
            continue
        insert_fn(layer.attn, kv, context_positions, slot_mapping)


@register("tail_ctxkv", DSPARK_MODEL_MODULE)
def patch_model(module) -> None:
    cls = module.DSparkDeepseekV4Model
    if getattr(cls, "_w5_ctxkv_patched", False):
        return
    def precompute_and_store_context_kv(self, main_x, context_positions, context_slot_mappings=None):
        if getattr(self, "_w5_ctxkv", None) is None:
            raise RuntimeError("tail_ctxkv: stacked wkv was not built in load_draft_model")
        return precompute_fused(self, main_x, context_positions, context_slot_mappings, module._insert_context_kv)

    cls.precompute_and_store_context_kv = precompute_and_store_context_kv
    cls._w5_ctxkv_patched = True


@register("tail_ctxkv", SPECULATOR_MODULE)
def patch_speculator(module) -> None:
    """Build the stacked weight once the drafter's weights are loaded and post-processed (before KV profiling)."""
    cls = module.DSparkSpeculator
    if getattr(cls, "_w5_ctxkv_loader", False):
        return
    orig_load = cls.load_draft_model

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = orig_load(self, target_model, target_attn_layer_names)
        reason = unsupported_reason(model)
        if reason:
            raise RuntimeError(f"tail_ctxkv: unsupported draft model: {reason}")
        st = StackedWkv(model.model.layers)
        model.model._w5_ctxkv = st
        sys.stderr.write(f"DS41_FASTPATH tail_ctxkv: one projection for {st.num_layers} draft layers' context KV, "
                         f"extra HBM {st.nbytes:,} B (allocated in load_draft_model)\n")
        return model

    cls.load_draft_model = load_draft_model
    cls._w5_ctxkv_loader = True
