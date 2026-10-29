"""MXFP8 dense linears at M <= 8 on FlashInfer's split-K low-latency CuTe kernel (SM100/SM103).

vLLM's FlashInferCutedslMxfp8LinearKernel always asks mm_mxfp8 for backend="cute-dsl", whose persistent kernel
launches one CTA per 128-wide N tile: 40 CTAs for wo_b (N=5120) and 14-28 for wq_a+wkv (N=1792) on a 152-SM GB300,
so decode-sized GEMMs stream weights at 2-4.5 TB/s. "cutedsl_low_latency" (M <= 8) splits K and takes the same
operands (column-major [K, N] weight, 128x4-swizzled ue8m0 scales). Measured L2-cold at M=6 under CUDA graphs
(scratch gemm_bench.py, 2026-09-28): wq_a+wkv 4.64 -> 3.39 us, wq_b 9.28 -> 7.53 us, wo_b 11.63 -> 9.43 us.

The low-latency tactics are tuned here at weight-load time for M = 1..8, because vLLM's own autotune pass runs the
forward at max_num_batched_tokens and never reaches this backend; an untuned shape would take the default tactic
inside CUDA-graph capture.
"""
from __future__ import annotations

import os
import sys

import torch

LL_MAX_M = 8
# (K, N) pairs measured faster on cutedsl_low_latency at M <= 8. DSpark's fc (K=15360) was a tie; 25600x6144 unmeasured.
LL_SHAPES = {(5120, 1792), (1280, 32768), (8192, 5120)}
TARGET = "vllm.model_executor.kernels.linear.mxfp8.flashinfer"
_tuned: set = set()
_LOG = lambda m: sys.stderr.write(f"LL_GEMM {m}\n")


def _tune(weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
    K, N = weight.shape
    if (K, N) in _tuned:
        return
    _tuned.add((K, N))
    if not os.environ.get("MEGA_FI_TUNE_FILE"):
        # Live tuning ends in an unbounded torch.cuda.synchronize() on the weight-loader thread and
        # would also override the pinned file (live entries outrank loaded ones).  Refuse it.
        raise RuntimeError(f"LL_GEMM K={K} N={N}: MEGA_FI_TUNE_FILE is required; refusing unbounded live tune")
    _LOG(f"cutedsl_low_latency K={K} N={N}: tactics from the pinned FlashInfer autotune file, no live tune")


def patch(module) -> None:
    cls = module.FlashInferCutedslMxfp8LinearKernel
    if getattr(cls, "_ll_gemm", False):
        return
    from vllm.model_executor.layers.fusion.quant_activation import as_quantized_activation
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils import flashinfer as vllm_flashinfer

    orig_apply = cls.apply_weights
    orig_process = cls.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        orig_process(self, layer)
        if tuple(layer.weight.shape) in LL_SHAPES:
            _tune(layer.weight, layer.weight_scale)

    def apply_weights(self, layer, x, bias=None):
        weight = layer.weight  # [K, N], column-major
        K, N = weight.shape
        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            m = qa.data.numel() // K
        else:
            m = x.numel() // K
        # Routing: larger M, unmeasured shapes and biased GEMMs stay on vLLM's kernel.
        if m > LL_MAX_M or (K, N) not in LL_SHAPES or bias is not None:
            return orig_apply(self, layer, x, bias)
        if qa is not None:
            input_mxfp8, input_scale = qa.data, qa.scale
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            out_dtype, input_shape = x.dtype, x.shape
            input_mxfp8, input_scale = mxfp8_e4m3_quantize(x.view(-1, K), is_sf_swizzled_layout=True)
        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8.view(-1, K), weight, input_scale, layer.weight_scale,
            out_dtype=out_dtype, backend="cutedsl_low_latency",
        )
        return output.view(*input_shape[:-1], N)

    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply_weights = apply_weights
    cls._ll_gemm = True
    _LOG(f"ACTIVE {TARGET}.FlashInferCutedslMxfp8LinearKernel: M <= {LL_MAX_M} -> cutedsl_low_latency")
