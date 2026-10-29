"""Python entry points for W1's decode attention kernel (build + launch)."""

import math
import os
from dataclasses import dataclass

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_mod = None


def _load_prebuilt(path: str):
    """Import an already built extension .so (read-only deployments: no JIT, no lock file)."""
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.ExtensionFileLoader("w1_attn_ext", path)
    spec = importlib.util.spec_from_loader("w1_attn_ext", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _prebuilt_path() -> str | None:
    """The prebuilt w1_attn_ext.so inside this package; the release ships it."""
    cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w1_attn_ext.so")
    return cand if os.path.exists(cand) else None


def _ext():
    """The prebuilt extension. A missing .so raises: runtime never JIT-builds."""
    global _mod
    if _mod is not None:
        return _mod
    pre = _prebuilt_path()
    if pre is None:
        raise RuntimeError("w1_attn_ext.so not found next to w1attn/ops.py")
    _mod = _load_prebuilt(pre)
    return _mod


def build():
    """Build w1_attn_ext.so from csrc/ + include/ (kernels/w1-attn/build.sh only; never called at runtime)."""
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.3a"
    # nvcc 13.4 (pip nvidia-cuda-nvcc in the image) has the one-instruction e4m3/e2m1 -> bf16x2 conversions;
    # the image's /usr/local/cuda (headers, cudart) is 13.0.
    nvcc134 = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin/nvcc"
    nvcc = nvcc134 if os.path.exists(nvcc134) else ""
    if nvcc:
        os.environ["PYTORCH_NVCC"] = nvcc
    build_dir = os.path.join(_ROOT, "build", "w1_attn")
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="w1_attn_ext",
        sources=[os.path.join(_ROOT, "csrc", "w1_attn.cu")],
        extra_include_paths=[os.path.join(_ROOT, "include")],
        extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK", "-Xptxas=-v"],
        build_directory=build_dir,
        verbose=False,
    )


SPLIT_HEADS = 16
NUM_SMS = 152
MAX_CHUNKS = 10


BITWISE = None  # sentinel plan: FlashMLA-bitwise sequential mode


@dataclass(frozen=True)
class Plan:
    n_chunks: int  # key chunks per token
    nsub: int  # 64-key sub-blocks per chunk


def make_plan(s_q: int, total_slots: int, num_sms: int = NUM_SMS) -> Plan:
    """One 64-key sub-block per split task if every task fits one wave, else two."""
    groups = 64 // SPLIT_HEADS
    nsub = 1
    if s_q * math.ceil(total_slots / 64) * groups > num_sms or math.ceil(total_slots / 64) > MAX_CHUNKS:
        nsub = 2
    return Plan(math.ceil(total_slots / (64 * nsub)), nsub)


def sm_scale_log2(sm_scale: float) -> float:
    """``sm_scale * LOG_2_E`` in fp32, as FlashMLA's host code computes it."""
    return float(np.float32(sm_scale) * np.float32(1.4426950408889634))


@dataclass
class Scratch:
    """Device scratch of one call site (one per max s_q; reused by every layer, launches are stream-ordered)."""

    part_o: torch.Tensor
    part_ml: torch.Tensor
    flags: torch.Tensor
    epoch: torch.Tensor


def alloc_scratch(max_s_q: int, device, max_chunks: int = MAX_CHUNKS) -> Scratch:
    return Scratch(
        part_o=torch.empty(max_s_q * max_chunks * 64 * 512, dtype=torch.float32, device=device),
        part_ml=torch.empty(max_s_q * max_chunks * 64 * 2, dtype=torch.float32, device=device),
        flags=torch.zeros(max_s_q * max_chunks * 4, dtype=torch.int32, device=device),
        epoch=torch.zeros(2, dtype=torch.int32, device=device),
    )


def attn_decode(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    sink: torch.Tensor,
    sm_scale: float,
    swa_cache: torch.Tensor,
    swa_idx: torch.Tensor,
    swa_lens: torch.Tensor,
    comp_cache: torch.Tensor | None,
    comp_idx: torch.Tensor | None,
    comp_lens: torch.Tensor | None,
    out_data: torch.Tensor,
    out_scale: torch.Tensor,
    scratch: Scratch,
    plan: Plan | None = None,
    pdl: bool = True,
    prefetch_table: torch.Tensor | None = None,
    max_ctas: int = 0,
    dbg_ts: torch.Tensor | None = None,
) -> None:
    """Drop-in for ``fused_norm_rope_attn_rope_cast_decode`` at TP1 / 64 heads (no q norm, GPT-J RoPE, MXFP8 out).

    Shapes as at the vLLM call site: q [s_q, 64, 512] bf16 in the fused layout; swa_cache [nb, page, 528] uint8
    (page stride via stride(0)); swa_idx [s_q, W] int32 global slots; swa_lens [s_q]; comp_cache [nb, page, 288]
    uint8; comp_idx [s_q, K] int32 global slots; comp_lens [s_q]; out_data [s_q, 8, 4096] fp8; out_scale int32
    [s_q, 8, 32] MN-major view; positions int32 or int64. prefetch_table: optional int64 [n, 2] (address, bytes)
    of weights to pull into L2 (W7).
    """
    s_q = q.shape[0]
    total = swa_idx.shape[-1] + (comp_idx.shape[-1] if comp_cache is not None else 0)
    if plan == "ss":  # bitwise split-dims sequential: dim tiles per task by width (tasks <= SMs)
        plan = Plan(-2, 1 if 8 * s_q <= NUM_SMS else (2 if 4 * s_q <= NUM_SMS else 4))
    elif plan == "cx":
        plan = Plan(-3, 0)  # bitwise 2-CTA cluster per token
    elif plan == "bitwise":
        plan = Plan(-1, 0)  # bitwise-parallel
    elif plan == "bitwise_seq":
        plan = Plan(0, 0)
    elif plan is None:
        plan = make_plan(s_q, total)
    _ext().w1_attn_decode(
        q, positions, cos_sin, sink, sm_scale_log2(sm_scale), swa_cache, swa_idx, swa_lens, comp_cache, comp_idx,
        comp_lens, out_data, out_scale, scratch.part_o, scratch.part_ml, scratch.flags, scratch.epoch,
        prefetch_table, plan.n_chunks, plan.nsub, pdl, max_ctas, dbg_ts,
    )
