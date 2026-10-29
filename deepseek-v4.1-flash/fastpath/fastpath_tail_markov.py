"""tail_markov: DSpark's sequential Markov sampling loop with one fused kernel per draft position.

Production's eager loop (DSparkSpeculator._sample_sequential, captured in the drafter's FULL graph) runs, per draft
position: an embedding lookup of the previous token, the Markov bias GEMM (cuBLAS), a BF16 add of the base logits,
``sample_pos - 1``, the Gumbel block kernel, a torch argmax over the block winners, a gather and a copy into
``draft_tokens``: 8-9 launches, ~22 us at batch 1. This component keeps the bias GEMM exactly as production calls it
(same module, same shapes, so cuBLAS returns the same bits) and replaces everything else with one Triton kernel per
position, after the idea in NVIDIA vLLM #61869 (``spec_decode/dspark/fused_sample.py``), adapted to this nightly's
``gumbel_sample``: no ``spec_step_seed``, the draw is keyed by ``(seed, sample_pos - 1 + salt, token)``.

Each program handles one (request, 1024-token vocab block), exactly gumbel_sample's tiling:
  logits = bf16(fp32(base) + fp32(bias))             # torch's BF16 add: fp32 sum, one RNE rounding
  draft_logits[req_state, step, block] = logits      # the cache the rejection sampler reads as q
  block winner = gumbel_noised_argmax(fp32(logits))  # the production jit function itself
The last program of a request to arrive (arrival counter) reduces the block winners with torch.argmax semantics
(first NaN, else first maximum), stores the draft token, and copies the token's Markov embedding row into the next
position's GEMM input. Output tokens and the q cache are bitwise identical to the eager loop (tests/test_markov.py).
"""
from __future__ import annotations

import sys

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

# torch / Triton / vLLM are imported lazily: components.py is imported by sitecustomize at interpreter start, before
# vLLM, in every Python process of the container.

SPECULATOR_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"


def unsupported_reason(spec) -> str | None:
    """Why the fused loop cannot stand in for production's _sample_sequential on this speculator (None = it can)."""
    import torch

    if spec.draft_logits is None:
        return "greedy drafts (no draft_logits cache)"
    if getattr(spec, "_d2t_scatter_index", None) is not None:
        return "reduced draft vocabulary"
    if getattr(spec, "_draft_topk", None) is not None:
        return "dspark_draft_topk"
    if spec.use_confidence_head or spec.enable_adaptive_verification:
        return "adaptive verification"
    if spec.draft_watermarker is not None:
        return "draft watermarking"
    if getattr(spec, "sample_from_anchor", True) is not True:
        return "1+N fill-in layout"
    if not isinstance(getattr(spec.model.model.markov_head, "markov_w1", None), torch.nn.Embedding):
        return "markov_w1 is not an nn.Embedding"
    return None


def build_sampler(spec):
    from w5_markov import FusedMarkovSampler

    w1 = spec.model.model.markov_head.markov_w1.weight
    return FusedMarkovSampler(spec.max_num_reqs, spec.vocab_size, w1, spec.use_fp64_gumbel)


def fused_sample_sequential(spec, num_reqs: int, head_hidden) -> None:
    """Drop-in for DSparkSpeculator._sample_sequential (full vocabulary, probabilistic drafts)."""
    sampler = spec._w5_markov
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base_logits = spec.model.compute_draft_logits(sample_hidden)
    base_logits = base_logits.view(num_reqs, n_spec, base_logits.shape[-1])
    idx_map = spec.sample_idx_mapping[:num_sample]
    sample_pos = spec.sample_pos[:num_sample]
    embed = sampler.embed_anchor(spec.input_buffers.input_ids, spec._anchor_idx, num_reqs)
    for i in range(n_spec):
        bias = spec.model.markov_bias(embed)
        sampler.sample_step(base_logits, bias, i, idx_map, sample_pos, spec.temperature, spec.seeds,
                            spec.draft_logits, spec.draft_tokens)


def build_gemv(spec):
    from w5_markov_gemv import FusedMarkovGemv

    head = spec.model.model.markov_head
    return FusedMarkovGemv(spec.max_num_reqs, head.markov_w2.weight, head.markov_w1.weight)


def gemv_sample_sequential(spec, num_reqs: int, head_hidden) -> None:
    """tail_markov_gemv: the Markov-bias GEMV runs inside the per-position sampling kernel."""
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base_logits = spec.model.compute_draft_logits(sample_hidden)
    base_logits = base_logits.view(num_reqs, n_spec, base_logits.shape[-1])
    spec._w5_markov.run(base_logits, spec.input_buffers.input_ids, spec.num_query_per_req,
                        spec.sample_idx_mapping[:num_sample], spec.sample_pos[:num_sample], spec.temperature,
                        spec.seeds, spec.draft_logits, spec.draft_tokens)


def _make_patch(name: str, builder, loop):
    def patch(module) -> None:
        """Swap DSparkSpeculator._sample_sequential for the fused loop; an unsupported speculator raises.

        Batch > 1 is design routing: fastpath_tail_markov_fused routes those batches to this W5 loop (not to stock).
        """
        cls = module.DSparkSpeculator
        if getattr(cls, "_w5_markov_patched", None) is not None:
            raise RuntimeError(f"{name}: {cls._w5_markov_patched} is already installed; install one tail_markov* component")
        orig_load = cls.load_draft_model

        def load_draft_model(self, target_model, target_attn_layer_names):
            model = orig_load(self, target_model, target_attn_layer_names)
            self.model = model  # unsupported_reason/build read self.model; the caller assigns it too
            reason = unsupported_reason(self)
            if reason:
                raise RuntimeError(f"{name}: unsupported speculator: {reason}")
            self._w5_markov = builder(self)
            sys.stderr.write(f"DS41_FASTPATH {name}: fused Markov loop\n")
            return model

        def _sample_sequential(self, num_reqs, head_hidden):
            if getattr(self, "_w5_markov", None) is None:
                raise RuntimeError(f"{name}: fused Markov loop was not built in load_draft_model")
            return loop(self, num_reqs, head_hidden)

        cls.load_draft_model = load_draft_model
        cls._sample_sequential = _sample_sequential
        cls._w5_markov_patched = name

    patch.__name__ = f"patch_{name}"
    return register(name, SPECULATOR_MODULE)(patch)


patch = _make_patch("tail_markov", build_sampler, fused_sample_sequential)
patch_gemv = _make_patch("tail_markov_gemv", build_gemv, gemv_sample_sequential)
