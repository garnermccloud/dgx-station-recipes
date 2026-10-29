"""ds41 fastpath component "topp_fast" (W6): exact top-p masking in one launch for the verify / sampling path.

Production samples thinking-mode requests at top_p >= 0.95 (ds41_sampling), so every verify step's
apply_sampling_params runs vLLM's apply_top_k_top_p over [requests x (K + 1), 129280] fp32 logits: the Triton split
pipeline, 7 launches, ~60-70 us at C1 and ~400-500 us at C8 on the GB300. This component replaces
SamplingStates.apply_top_k_top_p, for calls with top-p active and top-k off, by w6_topp.apply_top_p (libw6tp.so,
w6tp.cu): one thread-block cluster per row, the slice in shared memory, an exact radix-style boundary search on
46-bit fixed-point masses summed as integers.

Numerics (a deterministic summation-order change, see w6tp.cu): the kept set is the sort-based definition's
(descending logit, ties by lower index, keep while the mass before a token is < p, at least one token), with masses
expf(x - max) in exact fixed-point sums instead of fp32 partial sums of tl.exp. vLLM's pipeline approximates the same
definition (geometric pivot ladder, 5 rounds, fp32 sums) and keeps a superset when it does not converge;
tests/test_topp.py counts, per row, which of the two kept sets matches an fp64 reference. Kept logits are untouched,
masked ones become -inf, exactly as the stock op writes them.

Calls with top-k active, non-fp32 or non-unit-stride logits keep the stock op.

Fused copy: Sampler.apply_sampling_params (the verify path's first step) copies the bf16 target logits to a new fp32
tensor, then runs the logits processors, the thinking budget, temperature and min_p before top-p. When none of them
does anything for the batch (the same numpy-state checks their own early returns make: no logit bias / allowed or
stop tokens, no penalties, no bad words, no thinking budget, every temperature 0 or 1, every min_p 0) and only top-p
is active, the component makes that copy inside the top-p kernel instead (w6_topp.apply_top_p(logits, p, out=...)):
one launch for copy + top-p, same output bits as copy + this component's top-p. Anything else keeps the stock method.

Enable: copy fastpath_topp.py, w6_topp.py and libw6tp.so next to components.py, add
`import fastpath_topp  # noqa: F401,E402  W6 exact one-launch top-p` to components.py and put "topp_fast" in
DS41_FASTPATH. A kernel launch failure raises; there is no per-call stock fallback.
"""
from __future__ import annotations

import sys

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

NAME = "topp_fast"


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH {NAME}: {msg}\n")


def fast_ok(logits, top_k, top_p) -> bool:
    """True when the one-launch kernel handles this call (top-p only, fp32 [B, V] with unit stride on CUDA)."""
    import torch

    return (top_k is None and top_p is not None and logits.dtype == torch.float32 and logits.ndim == 2
            and logits.stride(1) == 1 and logits.is_cuda and top_p.dtype == torch.float32)


@register(NAME, "vllm.v1.worker.gpu.sample.states")
def patch(module) -> None:
    import w6_topp

    w6_topp.lib()  # load (and ABI-check) the library at patch time, not on the first request
    cls = module.SamplingStates

    def apply_top_k_top_p(self, logits, expanded_idx_mapping, idx_mapping_np):
        top_k, top_p = self.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
        if top_k is None and top_p is None:
            return logits
        if fast_ok(logits, top_k, top_p):
            return w6_topp.apply_top_p(logits, top_p.contiguous())
        return module.apply_top_k_top_p(logits, top_k, top_p)

    cls.apply_top_k_top_p = apply_top_k_top_p
    _log(f"patched {module.__name__}.SamplingStates.apply_top_k_top_p (top-p only calls -> w6_topp)")


def fused_copy_ok(sampler, logits, idx_mapping_np) -> bool:
    """True when apply_sampling_params would only copy logits to fp32 and apply top-k / top-p (every processor,
    the thinking budget, temperature and min_p skip this batch by their own checks)."""
    import numpy as np
    import torch
    from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
    from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
    from vllm.v1.worker.gpu.sample.penalties import PenaltiesState

    if not np.any(sampler.needs_logits_processing[idx_mapping_np]):
        return False  # the stock method returns the logits untouched (no copy)
    if logits.dtype not in (torch.bfloat16, torch.float32) or logits.ndim != 2 or logits.stride(1) != 1:
        return False
    procs = sampler.logits_processors
    if len(procs) != 3 or not (isinstance(procs[0], LogitBiasState) and isinstance(procs[1], PenaltiesState)
                               and isinstance(procs[2], BadWordsState)):
        return False  # custom processors: stock
    if np.any(procs[0].use_logit_bias[idx_mapping_np]) or np.any(procs[1].use_penalty[idx_mapping_np]):
        return False
    if int(procs[2].num_bad_words.np[idx_mapping_np].max()) != 0:
        return False
    tb = sampler.thinking_budget_state
    if tb.enabled and np.any(tb.use_thinking_budget[idx_mapping_np]):
        return False
    ss = sampler.sampling_states
    t = ss.temperature.np[idx_mapping_np]
    if not np.all((t == 0.0) | (t == 1.0)) or np.any(ss.min_p.np[idx_mapping_np] != 0.0):
        return False
    return True


@register(NAME, "vllm.v1.worker.gpu.sample.sampler")
def patch_sampler(module) -> None:
    import torch
    import w6_topp

    cls = module.Sampler
    orig = cls.apply_sampling_params

    def apply_sampling_params(self, logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, pos, input_ids,
                              expanded_local_pos, seq_lens_upper_bound_np, skip_top_k_top_p: bool = False):
        if not skip_top_k_top_p and fused_copy_ok(self, logits, idx_mapping_np):
            top_k, top_p = self.sampling_states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
            if top_k is None and top_p is not None:
                out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
                return w6_topp.apply_top_p(logits, top_p.contiguous(), out=out)
        return orig(self, logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, pos, input_ids,
                    expanded_local_pos, seq_lens_upper_bound_np, skip_top_k_top_p)

    cls.apply_sampling_params = apply_sampling_params
    _log(f"patched {module.__name__}.Sampler.apply_sampling_params (fp32 copy fused into top-p when nothing else "
         f"processes the batch)")
