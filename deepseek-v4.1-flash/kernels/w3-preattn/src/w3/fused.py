"""The W3 fused pre-attention chain (standalone PDL kernels), same inputs/outputs as w3.baseline.BaselineChain.

Launch sequence per layer (all on the current stream, programmatic stream serialization):
  K1 wqkv+kt : 14 x nsplit MXFP8 GEMM tasks (x quantized in-task) -> fp32 split partials; + T key-table tasks
  K2 qkv     : T tasks: split reduce (production order) -> q_norm/kv_norm -> MXFP8 q -> RoPE/MXFP8 SWA insert
  K3 wq_b    : 256 x 128-row tasks over 152 CTAs -> bf16 q in the mega-attention layout
K3 is launched while K1 runs; its producer warps stream wq_b into their rings before griddepcontrol.wait.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .build import load
from .state import KERNEL_BLOCK, SWA_BLOCK, DecodeState
from .weights import EPS, LayerWeights

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        _EXT = load("w3_ext", ("csrc/w3_ext.cu",), ())
    return _EXT


def wqkv_splits(T: int) -> list[int]:
    """Production split-K boundaries (128-element chunks of K=5120) for fused_wqa_wkv at T tokens.

    FlashInfer LL GEMM (T <= 8) tactics from the pinned tune file, bucketed to the next power of two:
    M1 (256,4,1,2), M2 (512,2,1,4), M4 (128,8,1,4), M8 (128,9,1,2) = (cta_k, stages, sfb, split_k).
    T 9..16 runs the cute-dsl persistent kernel (tactic split 4); matched bitwise at T=12 in tests.
    """
    if T == 1 or 5 <= T <= 8:
        return [0, 20, 40]
    if T == 2:
        return [0, 12, 24, 36, 40]
    if T <= 4 or T <= 16:
        return [0, 10, 20, 30, 40]
    raise ValueError("T > 16 not supported yet")


@dataclass
class FusedOut:
    q: torch.Tensor
    qx8: torch.Tensor
    qx8_sf: torch.Tensor
    qr_kv: torch.Tensor | None
    kv: torch.Tensor | None
    global_idx: torch.Tensor | None
    global_lens: torch.Tensor | None
    positions_i32: torch.Tensor


class FusedChain:
    def __init__(self, w: LayerWeights, cos_sin: torch.Tensor, T: int, debug: bool = False,
                 nslots_k1: int = 11, nslots_k3: int = 11):
        dev = w.q_norm.device
        self.w, self.cos_sin, self.T = w, cos_sin, T
        self.kb = wqkv_splits(T)
        self.nsplit = len(self.kb) - 1
        self.ratio = w.compress_ratio
        self.nslots_k1, self.nslots_k3 = nslots_k1, nslots_k3
        self.l2_prefetch = 0
        self.stages = (1, 2, 3)
        self.part = torch.empty(self.nsplit, T, 1792, dtype=torch.float32, device=dev)
        self.qx8 = torch.empty(T, 1280, dtype=torch.uint8, device=dev)
        self.qx8_sf = torch.zeros(((T + 127) // 128) * 128 * 40, dtype=torch.uint8, device=dev)
        self.q = torch.empty(T, 32768, dtype=torch.bfloat16, device=dev)
        self.gidx = torch.empty(T, 512, dtype=torch.int32, device=dev) if self.ratio > 0 else None
        self.glens = torch.empty(T, dtype=torch.int32, device=dev) if self.ratio > 0 else None
        self.pos32 = torch.empty(T, dtype=torch.int32, device=dev)
        self.qr_kv = torch.empty(T, 1792, dtype=torch.bfloat16, device=dev) if debug else None
        self.kv = torch.empty(T, 512, dtype=torch.bfloat16, device=dev) if debug else None

    def run(self, x: torch.Tensor, st: DecodeState, pdl: bool = True) -> FusedOut:
        e, w, T = ext(), self.w, self.T
        compressed = self.ratio > 0
        if 1 in self.stages:
          e.wqkv_kt(w.wqa_wkv.w_nk.view(torch.uint8), w.wqa_wkv.s_swz, x, self.part, self.kb,
                  st.topk_local[:T] if compressed else None, st.token_to_req if compressed else None,
                  st.comp_block_table[: st.reqs] if compressed else None,
                  KERNEL_BLOCK // max(self.ratio, 1), st.is_valid if compressed else None, st.positions,
                  self.gidx, self.glens, self.pos32, self.nslots_k1, pdl, self.l2_prefetch)
        if 2 in self.stages:
          e.qkv(self.part, self.nsplit, w.q_norm, w.kv_norm, EPS, self.qx8, self.qx8_sf, self.qr_kv, self.kv,
              st.swa_cache.view(st.swa_cache.shape[0], -1), st.swa_slots, st.positions, self.cos_sin, SWA_BLOCK, pdl)
        if 3 in self.stages:
          e.wq_b_gemm(w.wq_b.w_nk.view(torch.uint8), w.wq_b.s_swz, self.qx8, self.qx8_sf, self.q, self.nslots_k3, pdl)
        return FusedOut(q=self.q.view(T, 64, 512), qx8=self.qx8, qx8_sf=self.qx8_sf, qr_kv=self.qr_kv, kv=self.kv,
                        global_idx=self.gidx, global_lens=self.glens, positions_i32=self.pos32)
