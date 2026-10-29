"""Fused DSpark Markov sampling kernels shared by the W5 tail components (tail_markov, tail_fr*).

One Triton kernel per draft position replaces production's add + Gumbel + argmax + gather + copy + next-token
embedding glue; see fastpath_tail_markov.py for the exactness argument. `_markov_sample_keys_kernel` is the
reduced-vocabulary form: lanes are rows of a reduced draft head, `keys` maps them to target token ids, which key the
Gumbel noise and select the q-cache columns, exactly like production's d2t scatter into a -inf full-vocab row.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax

# gumbel_sample's block size, so block winners and tie-breaking agree with production.
BLOCK_SIZE = 1024
NUM_WARPS = 8


@triton.jit
def _markov_embed_anchor_kernel(
    input_ids_ptr,
    anchor_idx_ptr,
    weight_ptr,
    weight_stride,
    embed_ptr,
    embed_stride,
    rank,
    BLOCK_R: tl.constexpr,
):
    """embed[req] = markov_w1[input_ids[anchor_idx[req]]] (production: an index kernel + an embedding kernel)."""
    req = tl.program_id(0).to(tl.int64)
    anchor = tl.load(anchor_idx_ptr + req)
    token = tl.load(input_ids_ptr + anchor).to(tl.int64)
    r = tl.arange(0, BLOCK_R)
    row = tl.load(weight_ptr + token * weight_stride + r, mask=r < rank)
    tl.store(embed_ptr + req * embed_stride + r, row, mask=r < rank)


@triton.jit
def markov_sample_block(
    logits,
    keys,
    mask,
    req,
    step,
    num_steps,
    num_blocks,
    block_idx,
    idx_mapping_ptr,
    sample_pos_ptr,
    temp_ptr,
    seeds_ptr,
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    cache_cols,
    draft_tokens_ptr,
    draft_tokens_stride,
    embed_weight_ptr,
    embed_weight_stride,
    embed_ptr,
    embed_stride,
    rank,
    partial_max_ptr,
    partial_idx_ptr,
    counter_ptr,
    BLOCK_P: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_FP64: tl.constexpr,
    HAS_NEXT: tl.constexpr,
    KEYS_ARE_BLOCK: tl.constexpr,
):
    """Shared stage: cache store + Gumbel block winner + last-arrival reduction for one (request, vocab block).

    ``logits`` are already rounded to the cache dtype; ``keys`` are the target token ids of the block (the Gumbel
    noise key and the cache column); ``cache_cols`` are the cache columns to write (== keys for a full vocab).
    """
    row = req * num_steps + step
    req_state_idx = tl.load(idx_mapping_ptr + row).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    tl.store(
        logits_cache_ptr + req_state_idx * logits_cache_stride_0 + step * logits_cache_stride_1 + cache_cols,
        logits,
        mask=mask & is_valid_req,
    )
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(tl.float32)
    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    # sample_pos is the predicted token's position; the draw is keyed by the position before it.
    pos = tl.load(sample_pos_ptr + row) - 1
    value, idx = gumbel_noised_argmax(
        logits.to(tl.float32), keys, mask, seed, pos, temp,
        IS_DRAFTING=True, USE_FP64=USE_FP64, APPLY_TEMPERATURE=True,
    )
    if KEYS_ARE_BLOCK:
        winner_key = block_idx * logits.shape[0] + idx
    else:
        winner_key = tl.sum(tl.where(tl.arange(0, logits.shape[0]) == idx, keys, 0))
    if not USE_FP64:
        # One 64-bit atomic max per block replaces the per-block partial arrays: high 32 bits = the fp32 winner value
        # mapped to an order-preserving uint32 (any NaN maps to the top, as torch.argmax treats NaN as the maximum),
        # low 31 bits = (2^31 - 1 - token), so equal values resolve to the lowest token -- torch.argmax's first
        # maximum over blocks, each block's winner already being its lowest-index maximum.
        # -0.0 and +0.0 compare equal in torch.argmax; give them one key so the lowest token wins the tie.
        vbits = tl.where(value == 0.0, 0.0, value).to(tl.uint32, bitcast=True)
        ordered = tl.where((vbits >> 31) != 0, ~vbits, vbits | 0x80000000)
        ordered = tl.where(value != value, 0xFFFFFFFF, ordered)
        packed = (ordered.to(tl.int64) << 31) | (2147483647 - winner_key.to(tl.int64))
        tl.atomic_max(partial_max_ptr + req, packed, sem="acq_rel", scope="gpu")
    else:
        tl.store(partial_max_ptr + req * BLOCK_P + block_idx, value)
        tl.store(partial_idx_ptr + req * BLOCK_P + block_idx, winner_key)

    # The last block of this request to arrive finishes the reduction. The barrier orders every thread's partial
    # stores before the release.
    tl.debug_barrier()
    arrival = tl.atomic_add(counter_ptr + req, 1, sem="acq_rel", scope="gpu")
    if arrival == num_blocks - 1:
        if not USE_FP64:
            # Read the final packed winner and reset the slot for the next launch in one atomic.
            best = tl.atomic_xchg(partial_max_ptr + req, 0, sem="acq_rel", scope="gpu")
            token = 2147483647 - (best & 2147483647)
        else:
            p = tl.arange(0, BLOCK_P)
            p_mask = p < num_blocks
            values = tl.load(partial_max_ptr + req * BLOCK_P + p, mask=p_mask, other=float("-inf"), volatile=True)
            # torch.argmax semantics over the block winners: the first NaN if any, else the first maximum.
            is_nan = values != values
            first_nan = tl.min(tl.where(is_nan, p, BLOCK_P))
            best = tl.max(tl.where(is_nan, float("-inf"), values))
            first_max = tl.min(tl.where(p_mask & (values == best), p, BLOCK_P))
            winner = tl.where(first_nan < BLOCK_P, first_nan, first_max)
            token = tl.load(partial_idx_ptr + req * BLOCK_P + winner, volatile=True).to(tl.int64)
        tl.store(draft_tokens_ptr + req * draft_tokens_stride + step, token)
        if HAS_NEXT:
            r = tl.arange(0, BLOCK_R)
            embed_row = tl.load(embed_weight_ptr + token * embed_weight_stride + r, mask=r < rank)
            tl.store(embed_ptr + req * embed_stride + r, embed_row, mask=r < rank)
        tl.atomic_xchg(counter_ptr + req, 0, sem="acq_rel", scope="gpu")


@triton.jit(do_not_specialize=["step"])
def _markov_sample_kernel(
    base_logits_ptr,  # [num_reqs, num_steps, V]
    base_stride_0,
    base_stride_1,
    bias_ptr,  # [num_reqs, V]
    bias_stride,
    step,
    num_steps,
    idx_mapping_ptr,  # [num_reqs * num_steps], (req, step) order
    sample_pos_ptr,
    temp_ptr,
    seeds_ptr,
    logits_cache_ptr,  # [max_num_reqs, num_steps, V]
    logits_cache_stride_0,
    logits_cache_stride_1,
    draft_tokens_ptr,  # [max_num_reqs, num_steps] int64
    draft_tokens_stride,
    embed_weight_ptr,  # [V, rank]
    embed_weight_stride,
    embed_ptr,  # [num_reqs, rank]
    embed_stride,
    rank,
    partial_max_ptr,
    partial_idx_ptr,
    counter_ptr,
    vocab_size,
    num_blocks,
    BLOCK: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_FP64: tl.constexpr,
    HAS_NEXT: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK + tl.arange(0, BLOCK)
    mask = block < vocab_size
    base = tl.load(base_logits_ptr + req * base_stride_0 + step * base_stride_1 + block, mask=mask,
                   other=float("-inf"))
    bias = tl.load(bias_ptr + req * bias_stride + block, mask=mask, other=0.0)
    # The eager add's rounding: fp32 sum, rounded once to the cache dtype.
    logits = (base.to(tl.float32) + bias.to(tl.float32)).to(logits_cache_ptr.dtype.element_ty)
    markov_sample_block(
        logits, block, mask, req, step, num_steps, num_blocks, block_idx,
        idx_mapping_ptr, sample_pos_ptr, temp_ptr, seeds_ptr,
        logits_cache_ptr, logits_cache_stride_0, logits_cache_stride_1, block,
        draft_tokens_ptr, draft_tokens_stride,
        embed_weight_ptr, embed_weight_stride, embed_ptr, embed_stride, rank,
        partial_max_ptr, partial_idx_ptr, counter_ptr,
        BLOCK_P=BLOCK_P, BLOCK_R=BLOCK_R, USE_FP64=USE_FP64, HAS_NEXT=HAS_NEXT, KEYS_ARE_BLOCK=True,
    )


@triton.jit(do_not_specialize=["step"])
def _markov_sample_keys_kernel(
    base_logits_ptr,  # [num_reqs, num_steps, Kv] (reduced vocabulary, rows in ascending target-id order)
    base_stride_0,
    base_stride_1,
    bias_ptr,  # [num_reqs, Kv]
    bias_stride,
    keys_ptr,  # [Kv] int32 target token id of each reduced row, ascending
    step,
    num_steps,
    idx_mapping_ptr,
    sample_pos_ptr,
    temp_ptr,
    seeds_ptr,
    logits_cache_ptr,  # [max_num_reqs, num_steps, V] full-vocab q cache, -inf outside the reduced set
    logits_cache_stride_0,
    logits_cache_stride_1,
    draft_tokens_ptr,
    draft_tokens_stride,
    embed_weight_ptr,
    embed_weight_stride,
    embed_ptr,
    embed_stride,
    rank,
    partial_max_ptr,
    partial_idx_ptr,
    counter_ptr,
    vocab_size,
    num_blocks,
    BLOCK: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_FP64: tl.constexpr,
    HAS_NEXT: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK + tl.arange(0, BLOCK)
    mask = block < vocab_size
    keys = tl.load(keys_ptr + block, mask=mask, other=0).to(tl.int64)
    base = tl.load(base_logits_ptr + req * base_stride_0 + step * base_stride_1 + block, mask=mask,
                   other=float("-inf"))
    bias = tl.load(bias_ptr + req * bias_stride + block, mask=mask, other=0.0)
    logits = (base.to(tl.float32) + bias.to(tl.float32)).to(logits_cache_ptr.dtype.element_ty)
    markov_sample_block(
        logits, keys, mask, req, step, num_steps, num_blocks, block_idx,
        idx_mapping_ptr, sample_pos_ptr, temp_ptr, seeds_ptr,
        logits_cache_ptr, logits_cache_stride_0, logits_cache_stride_1, keys,
        draft_tokens_ptr, draft_tokens_stride,
        embed_weight_ptr, embed_weight_stride, embed_ptr, embed_stride, rank,
        partial_max_ptr, partial_idx_ptr, counter_ptr,
        BLOCK_P=BLOCK_P, BLOCK_R=BLOCK_R, USE_FP64=USE_FP64, HAS_NEXT=HAS_NEXT, KEYS_ARE_BLOCK=False,
    )


class FusedMarkovSampler:
    """Persistent buffers and launchers of the fused Markov loop (fixed addresses for CUDA-graph replay)."""

    def __init__(self, max_num_reqs: int, vocab_size: int, embed_weight: torch.Tensor, use_fp64: bool,
                 keys: torch.Tensor | None = None):
        """`vocab_size` = sampled columns; `keys` (int32, ascending) maps them to target ids for a reduced draft
        vocabulary (None = the full target vocabulary)."""
        assert embed_weight.dim() == 2 and embed_weight.stride(1) == 1
        assert keys is None or (keys.dtype == torch.int32 and keys.numel() == vocab_size)
        device = embed_weight.device
        self.embed_weight = embed_weight
        self.keys = keys
        self.vocab_size = vocab_size
        self.use_fp64 = use_fp64
        self.num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
        self.block_p = triton.next_power_of_2(self.num_blocks)
        self.rank = embed_weight.shape[1]
        self.block_r = triton.next_power_of_2(self.rank)
        # Input of the Markov bias GEMM; rewritten by every sampling step.
        self.markov_embed = torch.zeros(max_num_reqs, self.rank, dtype=embed_weight.dtype, device=device)
        # fp32 draws: one packed int64 winner per request, zero between launches (the last block resets it).
        # fp64 draws: per-block partial winners, reduced by the last block.
        self.partial_max = (torch.empty(max_num_reqs, self.block_p, dtype=torch.float64, device=device) if use_fp64
                            else torch.zeros(max_num_reqs, dtype=torch.int64, device=device))
        self.partial_idx = torch.empty(max_num_reqs, self.block_p, dtype=torch.int32, device=device)
        # Zero between launches: the last block of each request resets its counter.
        self.counter = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    def embed_anchor(self, input_ids: torch.Tensor, anchor_idx: torch.Tensor, num_reqs: int) -> torch.Tensor:
        embed = self.markov_embed[:num_reqs]
        _markov_embed_anchor_kernel[(num_reqs,)](
            input_ids, anchor_idx, self.embed_weight, self.embed_weight.stride(0), embed, embed.stride(0),
            self.rank, BLOCK_R=self.block_r,
        )
        return embed

    def sample_step(self, base_logits, bias, step, idx_mapping, sample_pos, temperature, seeds, logits_cache,
                    draft_tokens) -> None:
        num_reqs, num_steps, vocab_size = base_logits.shape
        assert vocab_size == self.vocab_size and base_logits.stride(2) == 1 and bias.stride(1) == 1
        assert logits_cache.size(-1) >= vocab_size and logits_cache.stride(2) == 1
        assert torch.promote_types(base_logits.dtype, bias.dtype) == logits_cache.dtype
        embed = self.markov_embed[:num_reqs]
        kernel = _markov_sample_kernel if self.keys is None else _markov_sample_keys_kernel
        keys_arg = () if self.keys is None else (self.keys,)
        kernel[(num_reqs, self.num_blocks)](
            base_logits, base_logits.stride(0), base_logits.stride(1), bias, bias.stride(0), *keys_arg, step,
            num_steps, idx_mapping, sample_pos, temperature, seeds,
            logits_cache, logits_cache.stride(0), logits_cache.stride(1),
            draft_tokens, draft_tokens.stride(0),
            self.embed_weight, self.embed_weight.stride(0), embed, embed.stride(0), self.rank,
            self.partial_max, self.partial_idx, self.counter, vocab_size, self.num_blocks,
            BLOCK=BLOCK_SIZE, BLOCK_P=self.block_p, BLOCK_R=self.block_r, USE_FP64=self.use_fp64,
            HAS_NEXT=step + 1 < num_steps, num_warps=NUM_WARPS,
        )
