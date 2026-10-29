"""ds41 fastpath component "lowb_aux" (W6): PR #61868 item 8, the attention side projections on the CuTe DSL ll_bf16
low-batch kernels at decode row counts.

At small T, production runs three bf16 side GEMMs through cuBLAS (split-K + reduce):
  - compressor kv_score = torch.mm(x, fused_wkv_wgate.weight.T, out_dtype=fp32)  (layers 2-39, [1024|512, 5120])
  - indexer weights_proj(x)                                                       (8 index layers, [32, 5120], bf16)
  - indexer wk(latent) -> MXFP4 K store                                           (8 index layers, [128, 512], bf16)
This component runs them on the image's LLBf16Gemm (FP32 x @ W.T; deterministic: split-K reduces through DSMEM in a
fixed order) with the PR's tuned dispatch, up to per-op row limits, measured on the GB300 (lowb/bench.py). Results
differ from cuBLAS only in the K summation order (not bitwise; kv_score is closer to fp64). weights_proj casts the
FP32 result to bf16; wk feeds an FP32-input copy of the image's MXFP4 indexer K store kernel whose only change is
the load (`.to(bf16).to(fp32)`), so it writes what production's store writes for bf16(ll output).

Kernels for every row count up to the limits are compiled and run once eagerly on the first eager call (the profile
run, before graph capture); a call whose kernel was not warmed, or that runs while a graph is captured before the
warmup, takes the stock path. Larger T, FP8 indexer caches and non-bf16 weights also take the stock path.

Enable: copy this file next to components.py, add `import fastpath_lowb_aux  # noqa: F401,E402  W6 low-batch side
projections` to components.py and put "lowb_aux" in DS41_FASTPATH. The row limits are fixed by the shipped recipe;
a missing CuTe DSL kernel raises at boot. Larger T and non-bf16/FP8 inputs route to the stock projections.
"""
from __future__ import annotations

import sys

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

NAME = "lowb_aux"
LIMIT_KV2 = 16   # kv_score, compress ratio 2: [1024, 5120]
LIMIT_KV1 = 24   # kv_score, compress ratio 1: [512, 5120]
LIMIT_WP = 12    # indexer weights_proj [32, 5120]
LIMIT_WK = 16    # indexer wk [128, 512]
ATTN_MODULE = "vllm.models.deepseek_v41.attention"

# (K, N) -> row limit
SHAPE_LIMITS = {(5120, 1024): LIMIT_KV2, (5120, 512): LIMIT_KV1, (5120, 32): LIMIT_WP, (512, 128): LIMIT_WK}

_STATE = {"gemm": None, "warmed": set(), "warm_done": False, "store": None}
STATS = {"ll_kv": 0, "ll_wp": 0, "ll_wk": 0, "stock": 0}


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH {NAME}: {msg}\n")


def _gemm():
    """The image's LLBf16Gemm with PR #61868's tuned dispatch (aux_projections.py) for these shapes."""
    if _STATE["gemm"] is None:
        from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import LLBf16Gemm

        class AuxGemm(LLBf16Gemm):
            TUNED = {(5120, 1024): {m: ("splitk", (4, 4)) for m in (5, 6, 7, 8)},
                     (5120, 512): {5: ("dotprod", 128), 6: ("dotprod", 64), 8: ("splitk", (4, 6))}}

            def dispatch(self, *, M, K, N):
                t = self.TUNED.get((K, N), {}).get(M)
                if t is not None:
                    backend, cfg = t
                    if backend == "dotprod":
                        return self.CompileKey(backend="dotprod", M=M, K=K, bs=cfg)
                    return self.CompileKey(backend="splitk", split_k=cfg[0], num_stages=cfg[1])
                if N < 128:
                    return self.CompileKey(backend="dotprod", M=M, K=K, bs=128)
                return super().dispatch(M=M, K=K, N=N)

        _STATE["gemm"] = AuxGemm()
    return _STATE["gemm"]


def _capturing() -> bool:
    import torch

    return torch.cuda.is_current_stream_capturing()


def _warm(device) -> None:
    """Compile and run once every (shape, M) the limits allow, eagerly (before any graph capture)."""
    import torch

    if _STATE["warm_done"] or _capturing():
        return
    _STATE["warm_done"] = True
    g = _gemm()
    n = 0
    for (K, N), limit in SHAPE_LIMITS.items():
        if limit <= 0:
            continue
        w = torch.zeros(N, K, dtype=torch.bfloat16, device=device)
        for M in range(1, limit + 1):
            x = torch.zeros(M, K, dtype=torch.bfloat16, device=device)
            g(x, w)
            _STATE["warmed"].add((K, N, M))
            n += 1
    torch.cuda.synchronize(device)
    _log(f"warmed {n} ll_bf16 (shape, rows) kernels; limits kv2 {LIMIT_KV2} kv1 {LIMIT_KV1} wp {LIMIT_WP} wk {LIMIT_WK}")


def _use(x, w) -> bool:
    """Whether x @ w.T takes ll_bf16 here."""
    import torch

    M, K = x.shape
    N = w.shape[0]
    return ((K, N, M) in _STATE["warmed"] and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
            and x.is_contiguous() and w.is_contiguous() and w.dim() == 2)


def _store_kernel():
    """An FP32-input copy of the image's MXFP4 indexer K store (only the load differs: FP32 -> bf16 -> FP32)."""
    if _STATE["store"] is None:
        from vllm.models.deepseek_v41.common.ops import _fp32x2_to_fp4x2
        from vllm.triton_utils import tl, triton

        @triton.jit
        def _k_store_fp32(k_pre_ptr, k_pre_stride, positions_ptr, rms_norm_weight_ptr, rms_norm_eps,
                          cos_sin_cache_ptr, cos_sin_stride, k_cache_ptr, kv_slot_mapping_ptr, kv_cache_block_size,
                          HEAD_SIZE: tl.constexpr, ROPE_HEAD_DIM: tl.constexpr, COMPRESS_RATIO: tl.constexpr,
                          TOKEN_STRIDE: tl.constexpr, SCALE_DIM: tl.constexpr, KV_BLOCK_STRIDE: tl.constexpr):
            token_idx = tl.program_id(0)
            kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
            if kv_slot_idx < 0:
                return
            position = tl.load(positions_ptr + token_idx)
            if (position + 1) % COMPRESS_RATIO != 0:
                return
            block = tl.arange(0, HEAD_SIZE)
            # the only change: the FP32 ll_bf16 output, rounded to the bf16 a cuBLAS bf16 GEMM output would hold
            k = tl.load(k_pre_ptr + token_idx * k_pre_stride + block).to(tl.bfloat16).to(tl.float32)
            rms_w = tl.load(rms_norm_weight_ptr + block).to(tl.float32)
            variance = tl.sum(k * k, axis=0) / HEAD_SIZE
            k = (k * tl.rsqrt(variance + rms_norm_eps) * rms_w).to(tl.bfloat16)
            k = k.to(tl.float32)
            NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
            NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
            NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
            HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
            even, odd = tl.split(tl.reshape(k, (NUM_PAIRS, 2)))
            pair_idx = tl.arange(0, NUM_PAIRS)
            rope_pair_local = pair_idx - NOPE_PAIRS
            is_rope_pair = rope_pair_local >= 0
            cs_idx = tl.maximum(rope_pair_local, 0)
            compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
            cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
            cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
            sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)
            new_even = even * cos_v - odd * sin_v
            new_odd = odd * cos_v + even * sin_v
            new_even = new_even.to(tl.bfloat16).to(tl.float32)
            new_odd = new_odd.to(tl.bfloat16).to(tl.float32)
            kv_block_idx = kv_slot_idx // kv_cache_block_size
            kv_pos_in_block = kv_slot_idx % kv_cache_block_size
            cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
            val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
            scale_ptr = cache_block_ptr + kv_cache_block_size * TOKEN_STRIDE + kv_pos_in_block * SCALE_DIM
            N_QUANT_BLOCKS: tl.constexpr = HEAD_SIZE // 32
            HALF_BLOCK: tl.constexpr = 16
            even_2d = tl.reshape(new_even, (N_QUANT_BLOCKS, HALF_BLOCK))
            odd_2d = tl.reshape(new_odd, (N_QUANT_BLOCKS, HALF_BLOCK))
            amax = tl.maximum(tl.max(tl.abs(even_2d), axis=1), tl.max(tl.abs(odd_2d), axis=1))
            amax = tl.maximum(amax, 6.0 * (2**-126))
            log2_ratio = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
            log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
            inv_scale = tl.exp2(-log2_ratio)
            ue8m0 = (log2_ratio + 127.0).to(tl.uint8)
            inv_scale_col = tl.reshape(inv_scale, (N_QUANT_BLOCKS, 1))
            packed = _fp32x2_to_fp4x2(even_2d * inv_scale_col, odd_2d * inv_scale_col)
            packed_flat = tl.reshape(packed, (TOKEN_STRIDE,))
            tl.store(val_ptr + tl.arange(0, TOKEN_STRIDE), packed_flat)
            tl.store(scale_ptr + tl.arange(0, SCALE_DIM), ue8m0)

        _STATE["store"] = _k_store_fp32
    return _STATE["store"]


def store_fp32(k_pre, positions, cos_sin_cache, rms_norm_weight, rms_norm_eps, k_cache, kv_slot_mapping,
               compress_ratio) -> None:
    """indexer_k_norm_rope_store(..., use_fp4_cache=True) for an FP32 k_pre (CUDA, non-shuffled layout)."""
    from vllm.models.deepseek_v41.common.ops import MXFP4_BLOCK_SIZE

    num_tokens = kv_slot_mapping.numel()
    assert k_pre.ndim == 2 and k_pre.shape[1] == 128 and k_pre.stride(1) == 1
    assert num_tokens <= k_pre.shape[0] and num_tokens <= positions.numel() and compress_ratio in (1, 2)
    if num_tokens == 0:
        return
    head_dim = k_pre.shape[1]
    _store_kernel()[(num_tokens,)](
        k_pre, k_pre.stride(0), positions, rms_norm_weight, rms_norm_eps, cos_sin_cache, cos_sin_cache.stride(0),
        k_cache, kv_slot_mapping, k_cache.shape[1], HEAD_SIZE=head_dim, ROPE_HEAD_DIM=64,
        COMPRESS_RATIO=compress_ratio, TOKEN_STRIDE=head_dim // 2, SCALE_DIM=head_dim // MXFP4_BLOCK_SIZE,
        KV_BLOCK_STRIDE=k_cache.stride(0), num_warps=1, launch_pdl=False)


@register(NAME, ATTN_MODULE)
def _patch(mod) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
    if not ll_bf16.is_available():
        raise RuntimeError("lowb_aux: CuTe DSL LLBf16Gemm is not available in this image")
    import torch

    Attn, Indexer = mod.DeepseekV4Attention, mod.DeepseekV4Indexer
    orig_run = Attn._run_parallel_input_projections
    orig_produce_k = Indexer._produce_k

    def _run_parallel_input_projections(self, hidden_states):
        _warm(hidden_states.device)
        compressor, indexer = self.compressor, self.indexer
        ll_kv = compressor is not None and _use(hidden_states, compressor.fused_wkv_wgate.weight)
        ll_wp = indexer is not None and _use(hidden_states, indexer.weights_proj.weight)
        if not (ll_kv or ll_wp):
            STATS["stock"] += 1
            return orig_run(self, hidden_states)
        # The image's body (deepseek_v41/attention.py), with the ll_bf16 closures where they apply
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            aux_streams = aux_streams[:2]
        aux_fns = [None, None]
        if compressor is not None:
            w_kv = compressor.fused_wkv_wgate.weight
            if ll_kv:
                STATS["ll_kv"] += 1
                aux_fns[0] = lambda: _gemm()(hidden_states, w_kv)
            else:
                aux_fns[0] = lambda: torch.mm(hidden_states, w_kv.T, out_dtype=torch.float32)
        if indexer is not None:
            if ll_wp:
                STATS["ll_wp"] += 1
                w_wp = indexer.weights_proj.weight
                aux_fns[1] = lambda: _gemm()(hidden_states, w_wp).to(w_wp.dtype)
            else:
                def indexer_weights_proj():
                    weights, _ = indexer.weights_proj(hidden_states)
                    return weights
                aux_fns[1] = indexer_weights_proj
        qr_kv, (kv_score, indexer_weights) = mod.execute_in_parallel(
            lambda: self._fused_wqa_wkv_gemm(hidden_states),
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:3],
            aux_streams,
            enable=hidden_states.shape[0] <= mod.envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        )
        return qr_kv, kv_score, indexer_weights

    def _produce_k(self, latent, positions, rotary_emb):
        attn_metadata = mod.get_forward_context().attn_metadata
        if (latent is None or not isinstance(attn_metadata, dict) or not self.owns_k or not self.use_fp4_kv
                or latent.dtype != torch.bfloat16 or not _use(latent, self.wk.weight)
                or __import__("vllm.platforms", fromlist=["current_platform"]).current_platform.is_rocm()):
            return orig_produce_k(self, latent, positions, rotary_emb)
        STATS["ll_wk"] += 1
        indexer_metadata = attn_metadata[self.k_cache.prefix]
        store_fp32(_gemm()(latent, self.wk.weight), positions, rotary_emb.cos_sin_cache, self.k_norm.weight,
                   self.k_norm.variance_epsilon, self.k_cache.kv_cache, indexer_metadata.slot_mapping,
                   self.compress_ratio)

    Attn._run_parallel_input_projections = _run_parallel_input_projections
    Indexer._produce_k = _produce_k
    _log(f"patched {ATTN_MODULE}.DeepseekV4Attention._run_parallel_input_projections and "
         f"DeepseekV4Indexer._produce_k (ll_bf16 up to kv2 {LIMIT_KV2} / kv1 {LIMIT_KV1} / wp {LIMIT_WP} / wk "
         f"{LIMIT_WK} rows)")
