"""Synthetic but shape-exact decode-step state for one DeepSeek-V4.1 layer (caches + attention metadata).

Mirrors what vLLM's builders hand the layer for a pure uniform-decode (DSpark verify) step:
  * SWA cache: DeepseekV4SWACache, block_size 32, V4.1 MXFP8 record (528 B/token), page 32*528 = 16896 B.
  * Compressed attention: topk_indices_buffer [T, 512] request-local compressed positions from the index
    source; flashmla block table [reqs, pages] of the kv source's cache, kernel block size 128 tokens, so
    ``block_size // ratio`` compressed slots per page.
  * Indexer (sparse logits, ratio-1 consumers 24/28/32/36): MXFP4 K cache pages of 128 positions x 68 B
    (64 B e2m1 + 4 ue8m0, data region then scale region), per-row (flattened) block table, row_ke = compressed
    context length, candidate blocks [T, 2048] of 8 positions each from the candidate source.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

SWA_BLOCK = 32
SWA_RECORD = 528
KERNEL_BLOCK = 128
TOPK = 512
CAND_BLOCKS = 2048
CAND_BLOCK_SIZE = 8
IDX_BYTES = 68


@dataclass(frozen=True)
class DecodeState:
    tokens: int
    reqs: int
    positions: torch.Tensor  # [T] int64
    token_to_req: torch.Tensor  # [T] int32
    is_valid: torch.Tensor  # [T] bool
    swa_cache: torch.Tensor  # [blocks, 16896] uint8
    swa_slots: torch.Tensor  # [T] int64
    topk_local: torch.Tensor  # [T, 512] int32 (topk_indices_buffer content from the index source)
    comp_block_table: torch.Tensor  # [reqs, pages] int32
    # indexer (sparse logits) state
    idx_kcache: torch.Tensor  # [pages, 128, 68] uint8 (page stride 8704)
    idx_row_ke: torch.Tensor  # [T] int32
    idx_block_table: torch.Tensor  # [T, pages_per_req] int32
    idx_row_indices: torch.Tensor  # [T] int32
    candidate_blocks: torch.Tensor  # [T, 2048] int32
    row_ks: torch.Tensor  # [T] int32 zeros
    sparse_indices: torch.Tensor  # [T, 2048] int32
    sparse_end: torch.Tensor  # [T] int32
    col_indices: torch.Tensor  # [T, 512] int32


def _unique_rand(n: int, k: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randperm(n, generator=gen)[:k]


def make_state(tokens: int, tokens_per_req: int, ctx: int, ratio: int, device: torch.device,
               seed: int = 0, spread: int = 977) -> DecodeState:
    """``tokens`` query rows, ``tokens_per_req`` per request, request i with ``ctx + spread * i`` tokens of context."""
    assert tokens % tokens_per_req == 0
    reqs = tokens // tokens_per_req
    gen = torch.Generator().manual_seed(seed)
    seq_lens = [ctx + spread * i for i in range(reqs)]
    pos = torch.cat([torch.arange(s - tokens_per_req, s) for s in seq_lens])
    tok2req = torch.repeat_interleave(torch.arange(reqs), tokens_per_req)

    # SWA cache: enough blocks for each request's window + new tokens, slots unique.
    swa_blocks_per_req = 8
    swa_cache = torch.randint(0, 256, (reqs * swa_blocks_per_req, SWA_BLOCK * SWA_RECORD), generator=gen,
                              dtype=torch.uint8)
    slots = []
    for t in range(tokens):
        r, p = int(tok2req[t]), int(pos[t])
        blk = r * swa_blocks_per_req + (p // SWA_BLOCK) % swa_blocks_per_req
        slots.append(blk * SWA_BLOCK + p % SWA_BLOCK)

    eff_ratio = max(ratio, 1)
    comp_page = KERNEL_BLOCK // eff_ratio
    comp_pages = max(-(-((s + 1) // eff_ratio) // comp_page) for s in seq_lens)
    comp_bt = torch.stack([_unique_rand(reqs * comp_pages * 2, comp_pages, gen) for _ in range(reqs)]).int()
    topk_local = torch.stack([_unique_rand(max((int(p) + 1) // eff_ratio, TOPK), TOPK, gen) for p in pos]).int()

    # Indexer K cache (ratio-1 layers only use it; harmless otherwise).
    idx_pages_per_req = max(-(-s // KERNEL_BLOCK) for s in seq_lens)
    n_pages = reqs * idx_pages_per_req
    kc = torch.empty((n_pages, KERNEL_BLOCK * IDX_BYTES), dtype=torch.uint8)
    kc[:, : KERNEL_BLOCK * 64] = torch.randint(0, 256, (n_pages, KERNEL_BLOCK * 64), generator=gen,
                                               dtype=torch.uint8)
    kc[:, KERNEL_BLOCK * 64:] = torch.randint(118, 128, (n_pages, KERNEL_BLOCK * 4), generator=gen,
                                              dtype=torch.uint8)
    page_perm = torch.randperm(n_pages, generator=gen).int().view(reqs, idx_pages_per_req)
    idx_bt = page_perm[tok2req].contiguous()
    row_ke = (pos + 1).int()  # ratio-1 compressed context length per row
    cands = []
    base = None
    for t in range(tokens):
        nblk = (int(pos[t]) + CAND_BLOCK_SIZE) // CAND_BLOCK_SIZE
        if t % tokens_per_req == 0:
            base = _unique_rand(nblk, CAND_BLOCKS, gen)
        c = base.clone()
        c[:64] = _unique_rand(nblk, 64, gen)  # rows of one request share most candidate blocks
        cands.append(c)
    cand = torch.stack(cands).int()

    d = dict(device=device)
    return DecodeState(
        tokens=tokens,
        reqs=reqs,
        positions=pos.to(torch.int64).to(**d),
        token_to_req=tok2req.int().to(**d),
        is_valid=torch.ones(tokens, dtype=torch.bool, **d),
        swa_cache=swa_cache.to(**d),
        swa_slots=torch.tensor(slots, dtype=torch.int64, **d),
        topk_local=topk_local.to(**d),
        comp_block_table=comp_bt.to(**d),
        idx_kcache=kc.to(**d).as_strided((n_pages, KERNEL_BLOCK, IDX_BYTES), (KERNEL_BLOCK * IDX_BYTES, IDX_BYTES, 1)),
        idx_row_ke=row_ke.to(**d),
        idx_block_table=idx_bt.to(**d),
        idx_row_indices=tok2req.int().to(**d),
        candidate_blocks=cand.to(**d),
        row_ks=torch.zeros(tokens, dtype=torch.int32, **d),
        sparse_indices=torch.zeros((tokens, CAND_BLOCKS), dtype=torch.int32, **d),
        sparse_end=torch.zeros(tokens, dtype=torch.int32, **d),
        col_indices=torch.zeros((tokens, TOPK), dtype=torch.int32, **d),
    )


def cos_sin_slice(ratio: int, max_pos: int, device: torch.device) -> torch.Tensor:
    """The layer's rotary cos||sin cache rows [0, max_pos) (fp32 [max_pos, 64]), built by vLLM's own code."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
    from vllm.transformers_utils.config import get_config

    cfg = get_config("/model", trust_remote_code=True)
    with set_current_vllm_config(VllmConfig()):
        rope = build_deepseek_v4_rope(cfg, head_dim=512, rope_head_dim=64,
                                      max_position_embeddings=cfg.max_position_embeddings,
                                      compress_ratio=ratio)
    return rope.cos_sin_cache[:max_pos].to(device=device, dtype=torch.float32).contiguous()
