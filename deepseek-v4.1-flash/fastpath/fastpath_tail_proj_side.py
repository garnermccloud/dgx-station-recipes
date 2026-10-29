"""ds41 fastpath component "tail_proj_side" (W6): DSpark's target-hidden projection on a side stream, under the
rejection sampler.

DFlashSpeculator.propose starts with main_x = combine_hidden_states(cat(aux_hidden_states)) (DSpark: cat of three
target layers, MXFP8 quantize, FlashInfer split-K GEMM 15360 -> 5120, main_norm; ~25-31 us) before anything else
can run. Its inputs are ready when the target model finishes, so this component launches exactly that call on a
side stream as soon as the rejection sampler starts (the LM head is done; the verify region that follows is
latency-bound, so the projection mostly hides under it: -15..-18 us per step in tail/proj/probe_sideproj.py), and
propose then waits for it instead of computing it.

Same call, same inputs, same kernels, same workspace lane (the drafter's): the result is bit-identical; only the
stream changes. Side-stream rules: the fork waits on an event recorded on the main stream at the rejection
sampler's entry; propose joins (main waits on the side stream's event) before its first kernel, which is the copy of
main_x into the drafter's persistent hidden-state buffer; sample_tokens joins on exit if propose did not (no fork
outlives its step); nothing forks while a CUDA graph is captured, in dummy runs, without aux hidden states, with
context parallelism or on a non-last pipeline rank. The projected tensor is record_stream'ed to the main stream and
the aux inputs to the side stream, so the caching allocator cannot reuse either early.

Enable: copy fastpath_tail_proj_side.py next to components.py, add
`import fastpath_tail_proj_side  # noqa: F401,E402  W6 DSpark projection on a side stream` to components.py and put
"tail_proj_side" in DS41_FASTPATH. A missing DFlash speculator or a fork error raises; shape/state routing
(no aux hidden states, capture, context parallel, non-last rank) keeps the stock order for that step.
"""
from __future__ import annotations

import sys

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

NAME = "tail_proj_side"
RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"
REJECTION_MODULE = "vllm.v1.worker.gpu.spec_decode.rejection_sampler"
DFLASH_MODULE = "vllm.v1.worker.gpu.spec_decode.dflash.speculator"

_PENDING = None  # the current step's fork (the engine core runs one step at a time on one thread)
_SIDE: dict = {}  # device index -> side stream
STATS = {"forks": 0, "joined_in_propose": 0, "joined_at_exit": 0}


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH {NAME}: {msg}\n")


def _side_stream(device):
    import torch

    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SIDE:
        _SIDE[idx] = torch.cuda.Stream(device=idx)
    return _SIDE[idx]


class Fork:
    """One step's projection: launched at the rejection sampler's entry, joined by propose (or at step exit)."""

    def __init__(self, spec, aux, lane: int):
        self.spec, self.aux, self.lane = spec, aux, lane
        self.launched = self.joined = False
        self.hs = self.done = None

    def launch(self) -> None:
        import torch
        from vllm.v1.worker.workspace import use_workspace_lane

        main = torch.cuda.current_stream()
        side = _side_stream(self.aux[0].device)
        ready = torch.cuda.Event()
        ready.record(main)
        side.wait_event(ready)
        with torch.cuda.stream(side), use_workspace_lane(self.lane):
            hs = self.spec.model.combine_hidden_states(torch.cat(self.aux, dim=-1))
            self.done = torch.cuda.Event()
            self.done.record(side)
        for t in self.aux:
            t.record_stream(side)
        hs.record_stream(main)
        self.hs = hs
        self.launched = True
        STATS["forks"] += 1

    def join(self) -> None:
        import torch

        torch.cuda.current_stream().wait_event(self.done)
        self.joined = True


def eligible(runner, state) -> bool:
    """A step whose propose would compute combine_hidden_states(cat(aux)) on this rank."""
    import torch
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    if state is None or not getattr(runner, "is_last_pp_rank", True) or getattr(runner, "pcp_manager", None):
        return False
    spec = getattr(runner, "speculator", None)
    if not isinstance(spec, DFlashSpeculator) or not hasattr(spec.model, "combine_hidden_states"):
        return False
    aux = getattr(state, "aux_hidden_states", None)
    if not aux or not all(isinstance(t, torch.Tensor) and t.is_cuda for t in aux):
        return False
    return not torch.cuda.is_current_stream_capturing()


@register(NAME, RUNNER_MODULE)
def patch_runner(module) -> None:
    cls = module.GPUModelRunner
    orig = cls.sample_tokens

    def sample_tokens(self, grammar_output):
        global _PENDING
        state = self.execute_model_state
        fork = (Fork(self.speculator, state.aux_hidden_states, int(getattr(self, "_draft_workspace_lane", 0)))
                if eligible(self, state) else None)
        _PENDING = fork
        try:
            return orig(self, grammar_output)
        finally:
            _PENDING = None
            if fork is not None and fork.launched and not fork.joined:
                fork.join()
                STATS["joined_at_exit"] += 1

    cls.sample_tokens = sample_tokens
    _log(f"patched {module.__name__}.GPUModelRunner.sample_tokens")


@register(NAME, REJECTION_MODULE)
def patch_rejection(module) -> None:
    cls = module.RejectionSampler
    orig = cls.__call__

    def __call__(self, *args, **kwargs):
        fork = _PENDING
        if fork is not None and not fork.launched:
            fork.launch()
        return orig(self, *args, **kwargs)

    cls.__call__ = __call__
    _log(f"patched {module.__name__}.RejectionSampler.__call__ (fork point)")


@register(NAME, DFLASH_MODULE)
def patch_dflash(module) -> None:
    cls = module.DFlashSpeculator
    orig = cls.propose

    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states, *args,
                **kwargs):
        fork = _PENDING
        if (fork is not None and fork.launched and not fork.joined and fork.spec is self
                and aux_hidden_states is fork.aux and not kwargs.get("dummy_run", False)):
            fork.join()
            STATS["joined_in_propose"] += 1
            # propose's own branch for aux_hidden_states=None copies last_hidden_states into self.hidden_states:
            # handing it the projected main_x reproduces the stock sequence minus the projection itself
            return orig(self, input_batch, attn_metadata, slot_mappings, fork.hs, None, *args, **kwargs)
        return orig(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states, *args,
                    **kwargs)

    cls.propose = propose
    _log(f"patched {module.__name__}.DFlashSpeculator.propose (join point)")
