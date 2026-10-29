"""ds41 fastpath component "engram_side" (W8): the engram layers' kv GEMM runs on the engram prefetch stream, beside
the previous layer's dense part, instead of on the critical path.

Engram.forward computes kv = wkv(embed(hash_ids).flatten(-2)): an MXFP8 quantize and a 6144 -> 25600 GEMM over 157 MB
of weights (31 us at C1, layers 1 and 14), from rows the engram prefetch stream looked up at the model's start. Only
the post kernel after it reads the hidden state. So for each target engram layer E, at the start of layer E-1 this
component forks the engram prefetch stream from the main stream (it already holds the lookups, so the rows are ready
in stream order) and enqueues the same quantize + GEMM there, with FlashInfer's persistent kernel limited to
W8_ENGRAM_SMS SMs (32: best of 16-152 in tests/probe_engram_overlap.py) so layer E-1's own kernels keep the
rest. The main stream waits for it as soon as layer E-1's attention module returns (wqkv_a .. wo_b: the one-layer
window the probe measured), before that layer's FFN boundary and MegaMoE, so the side GEMM never overlaps MegaMoE; and
layer E's Engram.forward uses that kv instead of calling wkv. Nothing outside these hooks is touched: no deep_gemm
attribute (v1 joined in a deep_gemm.fp8_fp4_mega_moe wrapper; ab-w8pqeng-1021, which ran v1, lost W4's MoE path and
W6's fold).

Bitwise: same rows, same quantize, same GEMM tactic (the pinned autotune entry for the width); only the persistent
grid differs, and FlashInfer's static persistent scheduler computes each 128-wide N tile in one CTA over the whole K
whatever the grid (no split-K), so kv is bit-identical (tests/test_engram_side.py, in a captured graph).

Scope: decode forwards captured as FULL CUDA graphs or run eagerly (never inside piecewise graphs; vLLM captures FULL
graphs with the forward context's mode NONE, so the rule is the decoder layer's own: not in_piecewise_cudagraph()),
T <= W8_ENGRAM_MAX_T (32), engram rows offloaded with a prefetch stream, TP / DP 1, no sequence parallelism. Anything else routes
the stock Engram.forward unchanged. The limited-SM kernels are compiled after the model loads and checked bitwise
against the stock GEMM for every width in 1..W8_ENGRAM_MAX_T, so nothing compiles inside a
capture; a width that is not warm (or not bitwise) is not forked.

Enable: copy this file next to components.py, add `import fastpath_engram_side  # noqa: F401,E402  W8 engram kv on
the side stream` to components.py and put "engram_side" in DS41_FASTPATH. A warm-up mismatch or failure raises.
Engagement: the hook's `ACTIVE engram_side` at load, and per captured width "engram kv on the side stream for
layers [1, 14] (T=..)".
"""
from __future__ import annotations

import contextlib
import sys

try:
    from fastpath_hook import bounded, register
except ImportError:  # imported directly by tests
    from contextlib import nullcontext

    def bounded(what, seconds):
        return nullcontext()

    def register(component, target):
        return lambda fn: fn

NAME = "engram_side"
SMS = 32
MAX_T = 32
WARM = tuple(
    range(1, MAX_T + 1))
MODEL_MODULE = "vllm.models.deepseek_v41.nvidia.model"
ENGRAM_MODULE = "vllm.models.deepseek_v41.nvidia.engram"
RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"

_ENGRAMS: dict[int, object] = {}   # target layer id -> its Engram module
_FORK_AT: dict[int, int] = {}      # layer id whose start forks -> engram layer id
_PENDING: dict[int, tuple] = {}    # id(engram) -> (kv, done event, T)
_JOIN: list = []                   # done events the main stream joins before the next MegaMoE
_WARM_T: set[int] = set()          # widths whose limited-SM kernels are compiled
_KERNEL_CACHE: dict = {}           # FlashInfer cute-dsl kernels compiled with the SM limit
_LOGGED: set = set()


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH {NAME}: {msg}\n")


@contextlib.contextmanager
def limited_sms(sms: int):
    """FlashInfer's cute-dsl MXFP8 dense GEMM compiled / looked up with at most `sms` SMs of persistent clusters."""
    import flashinfer.cute_dsl.utils as cu
    import flashinfer.gemm.gemm_base as gb

    old_fn, old_cache = cu.get_max_active_clusters, gb._CUTE_DSL_MM_MXFP8_KERNEL_CACHE
    cu.get_max_active_clusters = lambda size: max(1, min(old_fn(size), sms // max(1, size)))
    gb._CUTE_DSL_MM_MXFP8_KERNEL_CACHE = _KERNEL_CACHE
    try:
        yield
    finally:
        cu.get_max_active_clusters = old_fn
        gb._CUTE_DSL_MM_MXFP8_KERNEL_CACHE = old_cache


def side_kv(engram, T: int):
    """kv = wkv(rows[:T, :n_hash_cols].flatten(-2)) on the current stream with the SM-limited kernel, after this
    layer's lookup (its prefetch-done event, as PR #61868's EngramPrefetch.start waits for it)."""
    import torch

    done = getattr(engram, "_prefetch_done", None)
    if done is not None:
        torch.cuda.current_stream().wait_event(done)
    x = engram.staged_rows[:T, : engram.embed_tokens.n_hash_cols].flatten(-2)
    with limited_sms(SMS):
        return engram.wkv(x)


def _cute_dsl_wkv(engram) -> bool:
    """The wkv GEMM is FlashInfer's CuTe DSL MXFP8 kernel (as vLLM PR #61868's EngramPrefetch requires: the CUTLASS
    MXFP8 backend shares one process-wide workspace with GEMMs on other streams, the CuTe DSL one does not)."""
    kern = getattr(getattr(engram.wkv, "quant_method", None), "kernel", None)
    if kern is None:  # tests hand a bare module carrying the kernel
        kern = getattr(engram.wkv, "kern", None)
    return type(kern).__name__ == "FlashInferCutedslMxfp8LinearKernel"


def _why_not(engram, T: int) -> str | None:
    """None when this engram call may fork; else the first reason it may not (logged once per reason)."""
    import torch

    if T < 1 or T > MAX_T:
        return f"T={T} outside 1..{MAX_T}"
    if T not in _WARM_T:
        return f"T={T} not warm"
    if getattr(engram, "_prefetch_stream", None) is None:
        return "no prefetch stream (rows not offloaded)"
    if getattr(engram, "use_sequence_parallel", False):
        return "sequence parallel"
    emb = engram.embed_tokens
    if getattr(emb, "tp_size", 1) != 1 or getattr(emb, "dp_size", 1) != 1 or getattr(engram, "_batch", None) is not None:
        return "TP / DP > 1 or batched lookup"
    if not _cute_dsl_wkv(engram):
        kern = getattr(getattr(engram.wkv, "quant_method", None), "kernel", None)
        return f"wkv kernel {type(kern).__name__} is not FlashInfer CuTe DSL"
    try:
        # vLLM captures (and replays) FULL graphs with the forward context's mode NONE inside torch.cuda.graph
        # (cudagraph_utils.CudaGraphManager.capture), so a FULL graph is "capturing and not piecewise", the rule the
        # decoder layer's own mHC overlap uses. Eager forwards (not capturing) fork too; piecewise graphs never do.
        from vllm.forward_context import in_piecewise_cudagraph

        if in_piecewise_cudagraph():
            return "piecewise graph"
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"engram_side: forward context check failed: {e!r}") from e
    if not torch.cuda.is_available():
        return "no CUDA"
    return None


def _eligible(engram, T: int) -> bool:
    why = _why_not(engram, T)
    if why is not None and why not in _LOGGED and len(_LOGGED) < 64:
        _LOGGED.add(why)
        _log(f"not forked: {why}")
    return why is None


def fork(engram, T: int) -> bool:
    """At the start of the layer before `engram`'s: its kv on the engram prefetch stream. True when forked."""
    import torch

    stale = _PENDING.pop(id(engram), None)  # never let a kv outlive the forward that forked it
    if stale is not None and stale[1] in _JOIN:
        _JOIN.remove(stale[1])
    if not _eligible(engram, T):
        return False
    side = engram._prefetch_stream
    main = torch.cuda.current_stream()
    start = torch.cuda.Event()
    start.record(main)
    side.wait_event(start)  # never before this layer starts (in particular never beside the previous MegaMoE)
    with torch.cuda.stream(side):
        kv = side_kv(engram, T)
        done = torch.cuda.Event()
        done.record(side)
    _PENDING[id(engram)] = (kv, done, T)
    _JOIN.append(done)
    return True


def join() -> None:
    """The main stream waits for every forked engram kv (layer E-1's attention has returned: no side GEMM runs beside
    its FFN boundary or MegaMoE)."""
    import torch

    while _JOIN:
        torch.cuda.current_stream().wait_event(_JOIN.pop())


def take(engram, T: int):
    """The forked kv for this engram call (joined to the current stream), or None."""
    import torch

    p = _PENDING.pop(id(engram), None)
    if p is None:
        return None
    kv, done, t = p
    main = torch.cuda.current_stream()
    main.wait_event(done)
    if done in _JOIN:
        _JOIN.remove(done)
    if t != T:
        return None
    kv.record_stream(main)
    return kv


def warm(engrams) -> None:
    """Compile the SM-limited kernels for the widths in WARM (eager, outside any capture) and check them bitwise."""
    import torch

    if not engrams:
        return
    with bounded("engram_side warm (SM-limited GEMMs + cuda synchronize)", 300):
        for t in WARM:
            if t > MAX_T:
                continue
            ok = True
            for e in engrams:
                dev = e.staged_rows.device
                x = torch.randn(t, e.embed_tokens.n_hash_cols * e.staged_rows.shape[-1], device=dev,
                                dtype=torch.bfloat16)
                ref = e.wkv(x)
                with limited_sms(SMS):
                    got = e.wkv(x)
                torch.cuda.synchronize()
                ok &= torch.equal(ref.view(torch.int16), got.view(torch.int16))
            if not ok:
                raise RuntimeError(f"engram_side: SM-limited kv differs from the stock GEMM at T={t}")
            _WARM_T.add(t)
    _log(f"warm: SM-limited ({SMS} SMs) engram kv kernels bitwise-checked for T in {sorted(_WARM_T)}")


@register(NAME, ENGRAM_MODULE)
def _patch_engram(module) -> None:
    cls = module.Engram
    if getattr(cls, "_w8_engram_side", False):
        return
    orig = cls.forward

    def forward(self, hidden_states, hash_ids, token_mask=None):
        kv = take(self, hash_ids.shape[0]) if _PENDING else None
        if kv is None:
            return orig(self, hidden_states, hash_ids, token_mask)
        self.wkv.forward = lambda _x, _kv=kv: _kv  # the GEMM already ran on the side stream
        try:
            return orig(self, hidden_states, hash_ids, token_mask)
        finally:
            del self.wkv.forward

    cls.forward = forward
    cls._w8_engram_side = True


@register(NAME, MODEL_MODULE)
def _patch_model(module) -> None:
    import torch

    cls = module.DeepseekV4DecoderLayer
    if getattr(cls, "_w8_engram_side", False):
        return
    orig_init, orig_fwd = cls.__init__, cls.forward

    def __init__(self, vllm_config, prefix, *args, **kwargs):
        orig_init(self, vllm_config, prefix, *args, **kwargs)
        lid = module.extract_layer_index(prefix)
        self._w8_lid = lid
        n_target = vllm_config.model_config.hf_config.num_hidden_layers
        if self.engram is not None and 1 <= lid < n_target:
            _ENGRAMS[lid] = self.engram
            _FORK_AT[lid - 1] = lid
        if 0 <= lid < n_target and isinstance(getattr(self, "attn", None), torch.nn.Module):
            _join_after(self.attn)

    def forward(self, x, positions, input_ids, *args, **kwargs):
        tgt = _FORK_AT.get(getattr(self, "_w8_lid", -1))
        if tgt is not None:
            hashes = kwargs.get("engram_hashes", args[4] if len(args) > 4 else None)
            T = positions.shape[0]
            if hashes is None:
                _PENDING.pop(id(_ENGRAMS[tgt]), None)
            elif fork(_ENGRAMS[tgt], T) and (tgt, T) not in _LOGGED:
                _LOGGED.add((tgt, T))
                if len(_LOGGED) <= 16:
                    _log(f"engram kv on the side stream for layer {tgt} (T={T}, {SMS} SMs)")
        return orig_fwd(self, x, positions, input_ids, *args, **kwargs)

    cls.__init__ = __init__
    cls.forward = forward
    cls._w8_engram_side = True
    _log(f"engram kv forked at the layer before each engram layer, joined when its attention returns, "
         f"{SMS} SMs, T <= {MAX_T}")


def _join_after(attn) -> None:
    """Layer attention module: when it returns, the main stream joins any forked engram kv (instance-level, so the
    class methods other components patch stay as they are)."""
    def forward(*args, **kwargs):
        out = type(attn).forward(attn, *args, **kwargs)
        if _JOIN:
            join()
        return out

    attn.forward = forward


@register(NAME, RUNNER_MODULE)
def _patch_runner(module) -> None:
    cls = module.GPUModelRunner
    if getattr(cls, "_w8_engram_side", False):
        return
    orig = cls.load_model

    def load_model(self, *args, **kwargs):
        with bounded("GPUModelRunner.load_model (weights + post-load finalize)", 1200):
            out = orig(self, *args, **kwargs)
            if _ENGRAMS:
                warm(list(_ENGRAMS.values()))
        return out

    cls.load_model = load_model
    cls._w8_engram_side = True
