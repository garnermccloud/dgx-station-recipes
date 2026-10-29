"""tail_embed: the next step's input embeddings as one gather kernel straight into the CUDA-graph input buffer (W5).

DeepSeek-V4.1 is served as a multimodal model, so every step's eager prep builds inputs_embeds outside the target
graph (EncoderRunner.get_inputs_embeds): `input_ids.long()` (a cast kernel), F.embedding (index_select, ~6 us for
6 rows on the GB300: its small-index kernel is latency bound) and a device copy of the [n, 5120] result into the
persistent EncoderRunner.inputs_embeds buffer -- 3 ops, ~10 us GPU on the critical path. Here, for steps without
multimodal embeddings, one Triton kernel copies weight[input_ids[i]] into inputs_embeds[i] directly.

A pure copy of embedding rows: bitwise identical by construction. The embedding is resolved once
(language_model.model.embed_tokens: an unquantized TP-1 VocabParallelEmbedding whose weight matches the buffer), and
the first real call computes the stock result too and compares it bitwise with the kernel's (one sync, once); a
mismatch or an unexpected layout raises. Shape routing: steps with images, empty batches, or more tokens than the
buffer take the stock path.
"""
from __future__ import annotations

import sys

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

ENCODER_RUNNER_MODULE = "vllm.v1.worker.gpu.mm.encoder_runner"
_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    from vllm.triton_utils import tl, triton

    @triton.jit(do_not_specialize=["w_stride", "o_stride"])
    def _embed_gather_kernel(ids_ptr, w_ptr, w_stride, o_ptr, o_stride, D: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = cols < D
        row = tl.load(ids_ptr + i).to(tl.int64)
        v = tl.load(w_ptr + row * w_stride + cols, mask=ok)
        tl.store(o_ptr + i.to(tl.int64) * o_stride + cols, v, mask=ok)

    _kernel = _embed_gather_kernel
    return _kernel


def gather_rows(weight, ids, out) -> None:
    """out[i] = weight[ids[i]] for i < len(ids)."""
    n = ids.shape[0]
    d = weight.shape[1]
    block = 1024
    _get_kernel()[(n, (d + block - 1) // block)](ids, weight, weight.stride(0), out, out.stride(0), D=d, BLOCK=block,
                                                 num_warps=4)


def resolve_embedding(runner):
    """The embedding weight the stock path gathers from, or None when the layout is not the plain one."""
    import torch

    model = runner.model
    for m in (getattr(model, "language_model", None), model):
        inner = getattr(m, "model", None)
        emb = getattr(inner, "embed_tokens", None)
        if emb is None:
            continue
        w = getattr(emb, "weight", None)
        qm = getattr(emb, "quant_method", None)
        ok = (type(emb).__name__ == "VocabParallelEmbedding" and getattr(emb, "tp_size", 1) == 1
              and type(qm).__name__ == "UnquantizedEmbeddingMethod" and isinstance(w, torch.Tensor)
              and w.dim() == 2 and w.stride(1) == 1 and w.dtype == runner.inputs_embeds.dtype
              and w.shape[1] == runner.inputs_embeds.shape[1] and w.device == runner.inputs_embeds.device
              and runner.inputs_embeds.stride(1) == 1)
        return w if ok else None
    return None


def make_get_inputs_embeds(orig):
    def get_inputs_embeds(self, input_ids, mm_embeds, is_mm_embed):
        import torch

        n = input_ids.shape[0]
        if mm_embeds or n == 0 or n > self.inputs_embeds.shape[0] or input_ids.dim() != 1:
            return orig(self, input_ids, mm_embeds, is_mm_embed)
        state = self.__dict__.get("_w5_embed")
        if state is None:
            state = resolve_embedding(self)
            if state is None:
                raise RuntimeError("tail_embed: embedding layout is not the plain [V, H] table")
            self._w5_embed = state
            self._w5_embed_checked = False
        if not self._w5_embed_checked:
            ref = orig(self, input_ids, mm_embeds, is_mm_embed)[:n].clone()
            gather_rows(state, input_ids, self.inputs_embeds)
            if not torch.equal(ref, self.inputs_embeds[:n]):
                raise RuntimeError("tail_embed: self-check mismatch")
            self._w5_embed_checked = True
            sys.stderr.write("DS41_FASTPATH tail_embed: self-check bitwise equal, gather kernel active\n")
            return self.inputs_embeds
        gather_rows(state, input_ids, self.inputs_embeds)
        return self.inputs_embeds

    return get_inputs_embeds


@register("tail_embed", ENCODER_RUNNER_MODULE)
def patch(module) -> None:
    cls = module.EncoderRunner
    if getattr(cls, "_w5_embed_patched", False):
        return
    cls.get_inputs_embeds = make_get_inputs_embeds(cls.get_inputs_embeds)
    cls._w5_embed_patched = True
    sys.stderr.write("DS41_FASTPATH tail_embed: text-only steps gather input embeddings in one kernel\n")
