"""ds41 fastpath component "tail_draft_embed" (W6): the DSpark drafter's input glue as one kernel.

DSparkDeepseekV4Model.forward starts, inside the drafter's CUDA graph, with 6 launches:
  - embed_tokens(input_ids): an int32 -> int64 cast and index_select;
  - MegaMoE's gate metadata: image_token_mask = (ids >= base) & (ids < base + 5), 3 small kernels;
  - the hc expansion inputs_embeds.unsqueeze(-2).repeat(1, hc_mult, 1).
In the ab-rebal trace they cost ~12.9 us of the drafter's 607 us.

This component runs the same forward with one Triton kernel in their place. The kernel copies each token's embedding
row straight into the [T, hc_mult, H] stream and writes the mask. Pure row copies and integer compares, so the result
is bitwise identical; test_draft_embed.py compares it with the stock ops.

Composition:
  - The replacement is the stock forward re-stated with the new prologue. Its body after the prologue (the layer loop,
    mhc_post_tilelang, hc_collapse_triton) resolves those names from the module at call time.
  - So W4's boundary wrapper, which swaps the module's mhc_post_tilelang / hc_collapse_triton for its fused head
    during the call, still applies, provided this component is installed BEFORE "boundary" in DS41_FASTPATH (an earlier
    component patches first, so the later wrapper sits outside).
  - The image and the patched forward are fixed by the release; compatibility is asserted once in preflight
    (`vllm/vllm-openai:v0.31.0`). There is no per-component source digest or stock fallback: a missing symbol or a
    wrapper-order error raises at boot.

Shape routing: inputs_embeds given, sequence parallel, an empty batch, non-int32/int64 ids, or an embedding that is
not a plain [V, H] bf16 table with unit column stride take the stock forward for that call.
Enable: copy next to components.py, import it in components.py, put "tail_draft_embed" in DS41_FASTPATH before
"boundary".
"""
from __future__ import annotations

import sys

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

NAME = "tail_draft_embed"
DSPARK_MODULE = "vllm.models.deepseek_v41.nvidia.dspark"
_kernel = None


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH {NAME}: {msg}\n")


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    from vllm.triton_utils import tl, triton

    @triton.jit(do_not_specialize=["w_stride", "base"])
    def _draft_embed_kernel(ids_ptr, w_ptr, w_stride, o_ptr, mask_ptr, base, HAS_MASK: tl.constexpr,
                            HC: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        t = tl.program_id(0)
        cb = tl.program_id(1)
        cols = cb * BLOCK + tl.arange(0, BLOCK)
        ok = cols < D
        tok = tl.load(ids_ptr + t).to(tl.int64)
        v = tl.load(w_ptr + tok * w_stride + cols, mask=ok)
        for h in tl.static_range(HC):
            tl.store(o_ptr + (t * HC + h) * D + cols, v, mask=ok)
        if HAS_MASK:
            if cb == 0:
                tl.store(mask_ptr + t, (tok >= base) & (tok < base + 5))

    _kernel = _draft_embed_kernel
    return _kernel


def draft_embed(input_ids, weight, hc_mult: int, image_base: int | None):
    """(hidden [T, hc_mult, H] = weight[input_ids] repeated, image_token_mask [T] bool or None)."""
    import torch

    T, D = input_ids.shape[0], weight.shape[1]
    out = torch.empty(T, hc_mult, D, dtype=weight.dtype, device=weight.device)
    mask = torch.empty(T, dtype=torch.bool, device=weight.device) if image_base is not None else None
    block = 1024
    _get_kernel()[(T, (D + block - 1) // block)](
        input_ids, weight, weight.stride(0), out, mask if mask is not None else out, image_base or 0,
        HAS_MASK=mask is not None, HC=hc_mult, D=D, BLOCK=block, num_warps=4)
    return out, mask


def eligible(model, input_ids, inputs_embeds) -> bool:
    import torch

    if inputs_embeds is not None or getattr(model, "use_sequence_parallel", False):
        return False
    if input_ids is None or input_ids.dim() != 1 or input_ids.shape[0] == 0 or not input_ids.is_contiguous():
        return False
    if input_ids.dtype not in (torch.int32, torch.int64):
        return False
    w = getattr(getattr(model, "embed_tokens", None), "weight", None)
    if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.stride(1) != 1 or w.device != input_ids.device:
        return False
    emb = model.embed_tokens
    if getattr(emb, "tp_size", 1) != 1:
        return False
    return type(model).embed_input_ids is _STOCK_EMBED["fn"]


_STOCK_EMBED: dict = {"fn": None}
_META: dict = {"cls": None}


def make_forward(module, orig):
    """The audited DSparkDeepseekV4Model.forward with the fused prologue (same body otherwise)."""

    def forward(self, input_ids, positions, inputs_embeds=None):
        if not eligible(self, input_ids, inputs_embeds):
            return orig(self, input_ids, positions, inputs_embeds)
        image_base = module.IMAGE_SENTINEL_BASE_ID if getattr(self.config, "vision_n_layers", 0) > 0 else None
        hidden_states, image_token_mask = draft_embed(input_ids, self.embed_tokens.weight, self.hc_mult,
                                                      image_base if self.use_mega_moe else None)
        mega_gate_metadata = None
        if self.use_mega_moe:  # prepare_mega_gate_routing_metadata(has_hash_routing=False, ...)
            mega_gate_metadata = _META["cls"](None, image_token_mask, None)
        residual = post_mix = res_mix = pre_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix, pre_mix, _ = layer(
                hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual,
                mega_gate_metadata=mega_gate_metadata)
        hidden_states = module.mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        assert pre_mix is not None
        return module.hc_collapse_triton(hidden_states, pre_mix)

    forward.__w6_draft_embed__ = True
    return forward


@register(NAME, DSPARK_MODULE)
def patch(module) -> None:
    cls = module.DSparkDeepseekV4Model
    if getattr(cls.forward, "__w6_draft_embed__", False):
        return
    if getattr(cls.forward, "__w4_boundary__", False):
        raise RuntimeError("tail_draft_embed must be installed before boundary")
    meta_cls = module.prepare_mega_gate_routing_metadata.__globals__.get("MegaGateRoutingMetadata")
    if meta_cls is None:
        raise RuntimeError("MegaGateRoutingMetadata not found next to prepare_mega_gate_routing_metadata")
    _META["cls"] = meta_cls
    _STOCK_EMBED["fn"] = cls.embed_input_ids
    cls.forward = make_forward(module, cls.forward)
    _log("patched DSparkDeepseekV4Model.forward: embedding, gate image mask and hc expansion in one launch")
