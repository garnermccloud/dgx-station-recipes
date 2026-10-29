"""Real DeepSeek-V4.1-Flash attention weights for one layer, processed the way vLLM af7f9488 processes them.

vLLM path being mirrored (TP1, deepseek_v4_fp8 quant config, weight_block_size [32, 32], ue8m0 scales):
  * ModelOptLinearMethod(kMxfp8Static): the [N/32, K/32] block scale is ``repeat_interleave(32, dim=0)``-ed to a
    per-row [N, K/32] MXFP8 scale (KMxfp8Static.get_scale_weight_loader).
  * fused_wqa_wkv = MergedColumnParallelLinear([wq_a 1280, wkv 512]) -> rows concatenated.
  * DeepseekV4MegaAttnAttention.finalize_loaded_weights: permute_wq_b_ (rows + per-row scales) runs on the raw
    [N, K] layout before the linear kernel's process_weights_after_loading.
  * FlashInferCutedslMxfp8LinearKernel.process_weights_after_loading: scales -> F8_128x4 swizzle, weight -> [K, N]
    column-major view (``.t()`` of the contiguous [N, K]).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

MODEL_DIR = Path("/model")
HIDDEN = 5120
Q_RANK = 1280
HEAD_DIM = 512
N_HEADS = 64
IDX_HEADS = 32
IDX_DIM = 128
COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20
KV_SOURCES = (2, 8, 14, 20)
INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
CANDIDATE_SOURCE = 20
EPS = 1e-20


@dataclass(frozen=True)
class Mxfp8Weight:
    """One MXFP8 linear: raw [N, K] e4m3 + per-row [N, K/32] ue8m0 (uint8), and the FlashInfer-processed views."""

    w_nk: torch.Tensor  # [N, K] float8_e4m3fn, row-major (after any load-time row permutation)
    s_nk: torch.Tensor  # [N, K/32] uint8 ue8m0
    w_kn: torch.Tensor  # [K, N] column-major view, what mm_mxfp8 takes as B
    s_swz: torch.Tensor  # F8_128x4 swizzled scales, 1D uint8

    @property
    def n(self) -> int:
        return self.w_nk.shape[0]

    @property
    def k(self) -> int:
        return self.w_nk.shape[1]


@dataclass(frozen=True)
class LayerWeights:
    layer: int
    compress_ratio: int
    wqa_wkv: Mxfp8Weight  # [1792, 5120]
    wq_b: Mxfp8Weight  # [32768, 1280], rows permuted for mega attention
    q_norm: torch.Tensor  # [1280] bf16
    kv_norm: torch.Tensor  # [512] bf16
    idx_wq_b: Mxfp8Weight | None  # [4096, 1280] (index-source layers)
    weights_proj: torch.Tensor | None  # [32, 5120] bf16 (index-source layers)
    compressor_wkv_wgate: torch.Tensor | None  # [512 * (1 + (ratio==2)), 5120] bf16 (kv-source layers)
    compressor_norm: torch.Tensor | None
    idx_wk: torch.Tensor | None  # [128, 512] bf16 (kv-source layers)
    idx_k_norm: torch.Tensor | None


@lru_cache(maxsize=1)
def _weight_map() -> dict[str, str]:
    return json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]


def load_tensor(name: str) -> torch.Tensor:
    """Read one checkpoint tensor to CPU."""
    with safe_open(str(MODEL_DIR / _weight_map()[name]), framework="pt") as f:
        return f.get_tensor(name)


def has_tensor(name: str) -> bool:
    return name in _weight_map()


def swizzle_mxfp8_scale(sf: torch.Tensor, m: int, k: int) -> torch.Tensor:
    """vllm.model_executor.layers.quantization.utils.mxfp8_utils.swizzle_mxfp8_scale (row-major -> F8_128x4)."""
    num_m_tiles = (m + 127) // 128
    num_k_tiles = (k + 127) // 128
    padded = torch.zeros((num_m_tiles * 128, num_k_tiles * 4), dtype=sf.dtype, device=sf.device)
    padded[:m, : k // 32] = sf
    return padded.view(num_m_tiles, 4, 32, num_k_tiles, 4).transpose(1, 3).contiguous().view(-1)


def q_fused_permutation(num_heads: int, head_dim: int = HEAD_DIM) -> torch.Tensor:
    """vllm.models.deepseek_v41.common.ops.fused_layout.q_fused_permutation."""
    h = torch.arange(num_heads).view(num_heads, 1)
    d = torch.arange(head_dim).view(1, head_dim)
    fused_index = (d // 16) * (num_heads * 16) + h * 16 + d % 16
    perm = torch.empty(num_heads * head_dim, dtype=torch.long)
    perm[fused_index.reshape(-1)] = torch.arange(num_heads * head_dim)
    return perm


def _mxfp8(prefixes: list[str], device: torch.device, row_perm: torch.Tensor | None = None) -> Mxfp8Weight:
    ws, ss = [], []
    for p in prefixes:
        w = load_tensor(f"{p}.weight")
        s = load_tensor(f"{p}.scale").view(torch.uint8)
        block_rows = w.shape[0] // s.shape[0]
        ws.append(w)
        ss.append(s.repeat_interleave(block_rows, dim=0))
    w_nk = torch.cat(ws, 0).to(device)
    s_nk = torch.cat(ss, 0).to(device).contiguous()
    if row_perm is not None:
        row_perm = row_perm.to(device)
        w_nk = w_nk.view(torch.uint8)[row_perm].view(torch.float8_e4m3fn)
        s_nk = s_nk[row_perm].contiguous()
    w_nk = w_nk.contiguous()
    n, k = w_nk.shape
    return Mxfp8Weight(w_nk=w_nk, s_nk=s_nk, w_kn=w_nk.t(), s_swz=swizzle_mxfp8_scale(s_nk, n, k))


def load_layer(layer: int, device: torch.device) -> LayerWeights:
    """Load and process one decoder layer's pre-attention weights."""
    a = f"layers.{layer}.attn"
    ratio = COMPRESS_RATIOS[layer]
    is_index = layer in INDEX_SOURCES
    is_kv = layer in KV_SOURCES
    bf = lambda n: load_tensor(n).to(device=device, dtype=torch.bfloat16).contiguous()  # noqa: E731
    comp = None
    if is_kv:
        parts = [load_tensor(f"{a}.compressor.wkv.weight")]
        if has_tensor(f"{a}.compressor.wgate.weight"):
            parts.append(load_tensor(f"{a}.compressor.wgate.weight"))
        comp = torch.cat(parts, 0).to(device=device, dtype=torch.bfloat16).contiguous()
    return LayerWeights(
        layer=layer,
        compress_ratio=ratio,
        wqa_wkv=_mxfp8([f"{a}.wq_a", f"{a}.wkv"], device),
        wq_b=_mxfp8([f"{a}.wq_b"], device, row_perm=q_fused_permutation(N_HEADS)),
        q_norm=bf(f"{a}.q_norm.weight"),
        kv_norm=bf(f"{a}.kv_norm.weight"),
        idx_wq_b=_mxfp8([f"{a}.indexer.wq_b"], device) if is_index else None,
        weights_proj=bf(f"{a}.indexer.weights_proj.weight") if is_index else None,
        compressor_wkv_wgate=comp,
        compressor_norm=bf(f"{a}.compressor.norm.weight") if is_kv else None,
        idx_wk=bf(f"{a}.indexer.wk.weight") if is_kv else None,
        idx_k_norm=bf(f"{a}.indexer.k_norm.weight") if is_kv else None,
    )


def attn_norm_weight(layer: int, device: torch.device) -> torch.Tensor:
    return load_tensor(f"layers.{layer}.attn_norm.weight").to(device=device, dtype=torch.bfloat16)
