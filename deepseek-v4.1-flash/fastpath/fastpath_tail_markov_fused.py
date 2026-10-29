"""ds41 fastpath component "tail_markov_fused" (W6): DSpark's whole sequential Markov loop as ONE kernel launch.

Supersedes W5's tail_markov (arm one or the other). Production with tail_markov runs, inside the drafter's CUDA graph,
an anchor-embedding kernel plus, per draft position, the Markov-bias GEMV (cuBLAS, the 66 MB markov_w2 read from L2
every step) and W5's fused sampling kernel: 11 launches, ~77 us at batch 1. This component replaces
DSparkSpeculator._sample_sequential after the base logits with w6_markov.FusedMarkovLoop: one cooperative launch, one
CTA per SM, each owning a contiguous vocabulary slice of markov_w2 that it keeps in shared memory and registers after
the first position streams it from HBM (libw6mk.so, w6mk.cu).

Exactness: the draw is production's bit for bit given the logits (murmur3 Gumbel noise, temperature division,
lowest-token tie-break; tests/test_markov_fused.py re-runs production's sampling kernel on our cached logits and gets
our tokens). The only numerical change is the bias GEMV's fp32 summation order, which can move a bias by one bf16 ulp;
the kernel writes exactly the logits it samples from into the q cache the rejection sampler reads, so the output
distribution stays exactly the target's: a DRAFT-ONLY change (acceptance only; KL vs production q in the test).

Batch routing: the one-launch kernel takes one request; batches above that route to W5's fused loop (still one
kernel per draft position, but not the stock loop). Setup failure or a speculator outside the shipped recipe raises.
The image and checkpoint are fixed by preflight.

Enable: copy fastpath_tail_markov_fused.py, w6_markov.py and libw6mk.so next to components.py (fastpath_tail_markov.py
and w5_markov.py are the batch-routing path), add
`import fastpath_tail_markov_fused  # noqa: F401,E402  W6 one-launch Markov loop` to components.py and put
"tail_markov_fused" in DS41_FASTPATH.
"""
from __future__ import annotations

import sys

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

NAME = "tail_markov_fused"


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH {NAME}: {msg}\n")


def build(spec):
    """The fused loop for this speculator. Raises when the shipped recipe is not the one this kernel supports."""
    import fastpath_tail_markov as w5
    import torch
    import w6_markov

    reason = w5.unsupported_reason(spec)
    if reason is None and spec.use_fp64_gumbel:
        reason = "fp64 Gumbel"
    head = spec.model.model.markov_head
    w1 = head.markov_w1.weight
    w2 = getattr(head.markov_w2, "weight", None)
    if reason is None and (w2 is None or w2.dtype != torch.bfloat16 or w1.dtype != torch.bfloat16
                           or w2.shape != w1.shape or w2.shape[1] != w6_markov.RANK or w2.shape[0] != spec.vocab_size):
        reason = "markov_w1/w2 are not [vocab, 256] bf16"
    if reason is None and spec.draft_logits.dtype != torch.bfloat16:
        reason = "q cache is not bf16"
    if reason is not None:
        raise RuntimeError(f"{NAME}: unsupported speculator: {reason}")
    loop = w6_markov.FusedMarkovLoop(w1.contiguous(), w2.contiguous(), spec.max_num_reqs,
                                     spec.num_speculative_steps)
    return loop


def fused_loop(spec, num_reqs: int, head_hidden) -> bool:
    """Run the one-launch loop; False when this batch routes to W5's loop (above the kernel's batch limit)."""
    loop = spec._w6_markov
    if not loop.supports(num_reqs):
        return False
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base_logits = spec.model.compute_draft_logits(sample_hidden)
    base_logits = base_logits.view(num_reqs, n_spec, base_logits.shape[-1])
    loop.run(base_logits, spec.input_buffers.input_ids, spec._anchor_idx[:num_reqs],
             spec.sample_idx_mapping[:num_sample], spec.sample_pos[:num_sample], spec.temperature, spec.seeds,
             spec.draft_logits, spec.draft_tokens)
    return True


@register(NAME, "vllm.v1.worker.gpu.spec_decode.dspark.speculator")
def patch(module) -> None:
    import fastpath_tail_markov as w5

    cls = module.DSparkSpeculator
    if getattr(cls, "_w5_markov_patched", None) is not None:
        raise RuntimeError(f"{NAME}: {cls._w5_markov_patched} is already installed; install one tail_markov* component")
    orig_load = cls.load_draft_model

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = orig_load(self, target_model, target_attn_layer_names)
        self.model = model
        reason5 = w5.unsupported_reason(self)
        if reason5 is not None:
            raise RuntimeError(f"{NAME}: unsupported speculator: {reason5}")
        self._w5_markov = w5.build_sampler(self)  # batch routing above the one-launch kernel's limit
        self._w6_markov = build(self)
        loop = self._w6_markov
        _log(f"one-launch Markov loop: {loop.grid} CTAs x {loop.rows_per_cta} rows ({loop.res_rows} resident in "
             f"shared memory, the rest streamed), batches <= {loop.max_batch}; W5's loop above")
        return model

    def _sample_sequential(self, num_reqs, head_hidden):
        if fused_loop(self, num_reqs, head_hidden):
            return None
        return w5.fused_sample_sequential(self, num_reqs, head_hidden)

    cls.load_draft_model = load_draft_model
    cls._sample_sequential = _sample_sequential
    cls._w5_markov_patched = NAME
    _log("patched vllm.v1.worker.gpu.spec_decode.dspark.speculator.DSparkSpeculator")
