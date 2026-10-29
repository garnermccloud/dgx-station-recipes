"""Fused Markov-bias GEMV + sampling, one kernel per draft position (W5 tail_markov_gemv / tail_fr*_gemv).

tail_markov keeps production's cuBLAS bias GEMM for bitwise output, so each draft position still costs a GEMM launch
(~8 us: the 66 MB Markov w2 streamed from L2 by 1010 tiles of 128x8) plus the fused sampling kernel (~5 us). Here
both run in one kernel per position:

  every program owns BLOCK_ROWS rows of w2 (and of the base logits) for ALL requests of the batch;
  bias tile [BLOCK_ROWS, R_PAD] = w2_tile [BLOCK_ROWS, 256] @ embed^T [256, R_PAD]   (tensor cores, fp32 accumulate)
  logits = bf16(fp32(base) + fp32(bf16(bias)))   -- production's two roundings: the GEMM output, then the add
  q cache store, Gumbel noise (production's murmur3 draw, keyed by (seed, sample_pos - 1 + salt, token)), per-request
  block winner, one packed 64-bit atomic max per request; the last program to arrive writes the draft tokens and the
  next position's Markov embeddings (w1 rows of the winners).

The draw given the logits is production's: same noise function per (seed, position, token), same fp32 math, and
the winner is the lowest token among maxima, i.e. what gumbel_sample's two-level argmax returns for any tiling.
The only numerical change is the bias GEMV's accumulation order (fp32, then the same BF16 rounding), which can move a
bias by one BF16 ulp. That changes q, not the exactness of the output distribution: the kernel writes exactly the
logits it samples from into the q cache the rejection sampler reads. DRAFT-ONLY change (acceptance only).

Launched with programmatic dependent launch: everything that does not depend on the previous position (base logits,
w2 rows) is loaded before griddepcontrol.wait.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import _DRAFT_NOISE_SALT, _log1p_neg_stable, murmur3_uniform32

BLOCK_ROWS = 128
NUM_WARPS = 4
USE_PDL = True
BLOCK_K = 64
NUM_STAGES = 3


@triton.jit
def _embed_rows_kernel(token_ptr, token_stride, weight_ptr, weight_stride, embed_ptr, embed_stride, rank,
                       BLOCK_R: tl.constexpr):
    """embed[req] = markov_w1[token[req * token_stride]] (the anchor embedding of position 0)."""
    req = tl.program_id(0).to(tl.int64)
    token = tl.load(token_ptr + req * token_stride).to(tl.int64)
    r = tl.arange(0, BLOCK_R)
    tl.store(embed_ptr + req * embed_stride + r, tl.load(weight_ptr + token * weight_stride + r, mask=r < rank),
             mask=r < rank)


@triton.jit
def _sample_column(lf, keys, row_mask, pid, r, num_steps, step, idx_mapping_ptr, sample_pos_ptr, temp_ptr, seeds_ptr,
                   cache_ptr, cache_stride_0, cache_stride_1, best_ptr, BLOCK_ROWS: tl.constexpr, KEYS: tl.constexpr):
    """Request r's share of one draft position on this program's rows: q-cache store + Gumbel winner + packed max."""
    srow = r * num_steps + step
    req_state = tl.load(idx_mapping_ptr + srow).to(tl.int64)
    valid = req_state >= 0
    tl.store(cache_ptr + req_state * cache_stride_0 + step * cache_stride_1 + keys,
             lf.to(cache_ptr.dtype.element_ty), mask=row_mask & valid)
    temp = tl.load(temp_ptr + req_state, mask=valid, other=0.0).to(tl.float32)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    pos = tl.load(sample_pos_ptr + srow) - 1
    # gumbel_noised_argmax(IS_DRAFTING=True, USE_FP64=False, APPLY_TEMPERATURE=True), production's arithmetic.
    v = lf
    if temp != 0.0:
        v = v / temp
        u = murmur3_uniform32(seed, pos + _DRAFT_NOISE_SALT, keys)
        v = tl.where(row_mask, v + (-tl.log(-_log1p_neg_stable(u))), float("-inf"))
    value, idx = tl.max(v, axis=0, return_indices=True)
    if KEYS:
        win_key = tl.sum(tl.where(tl.arange(0, BLOCK_ROWS) == idx, keys, 0))
    else:
        win_key = pid * BLOCK_ROWS + idx.to(tl.int64)
    vz = tl.where(value == 0.0, 0.0, value)
    vbits = vz.to(tl.uint32, bitcast=True)
    ordered = tl.where((vbits >> 31) != 0, ~vbits, vbits | 0x80000000)
    ordered = tl.where(value != value, 0xFFFFFFFF, ordered)
    packed = (ordered.to(tl.int64) << 31) | (2147483647 - win_key)
    tl.atomic_max(best_ptr + r, packed, sem="acq_rel", scope="gpu")


@triton.jit(do_not_specialize=["step", "num_reqs"])
def _markov_gemv_sample_kernel(
    base_ptr, base_stride_0, base_stride_1,  # [R, num_steps, Kv]
    w2_ptr, w2_stride,  # [Kv, RANK]
    keys_ptr,  # [Kv] int32 target ids (reduced vocab) -- unused when KEYS is False
    embed_ptr, embed_stride,  # [R, RANK] this position's Markov embeddings
    next_embed_ptr,  # [R, RANK] next position's (written by the last program)
    w1_ptr, w1_stride,  # [V, RANK]
    anchor_ptr, anchor_stride,  # step 0 only (FUSE_ANCHOR): token ids whose w1 rows are this step's embeddings
    step, num_steps, num_reqs,
    idx_mapping_ptr, sample_pos_ptr, temp_ptr, seeds_ptr,
    cache_ptr, cache_stride_0, cache_stride_1,
    draft_tokens_ptr, draft_tokens_stride,
    best_ptr,  # [max_reqs] int64 packed winners, zero between launches
    counter_ptr,  # [1] int32 arrivals, zero between launches
    Kv,
    RANK: tl.constexpr, BLOCK_ROWS: tl.constexpr, R_PAD: tl.constexpr, KEYS: tl.constexpr,
    HAS_NEXT: tl.constexpr, FUSE_ANCHOR: tl.constexpr, USE_PDL: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < Kv
    if KEYS:
        keys = tl.load(keys_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    else:
        keys = rows.to(tl.int64)
    reqs = tl.arange(0, R_PAD)
    req_mask = reqs < num_reqs
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    if FUSE_ANCHOR:
        tok = tl.load(anchor_ptr + reqs.to(tl.int64) * anchor_stride, mask=req_mask, other=0).to(tl.int64)
        e_base = w1_ptr + tok[None, :] * w1_stride
    else:
        e_base = embed_ptr + reqs[None, :].to(tl.int64) * embed_stride
    # bias tile [BLOCK_ROWS, R_PAD] = w2 rows @ embeddings^T on tensor cores, fp32 accumulation over RANK in
    # BLOCK_K steps (pipelined loads), then the GEMM's BF16 rounding.
    acc = tl.zeros((BLOCK_ROWS, R_PAD), dtype=tl.float32)
    for k0 in tl.range(0, RANK, BLOCK_K, num_stages=NUM_STAGES):
        kk = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w2_ptr + rows[:, None].to(tl.int64) * w2_stride + kk[None, :], mask=row_mask[:, None], other=0.0)
        et = tl.load(e_base + kk[:, None], mask=req_mask[None, :], other=0.0)  # [BLOCK_K, R_PAD]
        acc = tl.dot(w, et, acc)
    bias2 = acc.to(tl.bfloat16).to(tl.float32)
    for r in range(num_reqs):
        bias = tl.sum(tl.where(reqs[None, :] == r, bias2, 0.0), axis=1)
        base = tl.load(base_ptr + r * base_stride_0 + step * base_stride_1 + rows, mask=row_mask, other=float("-inf"))
        lf = (base.to(tl.float32) + bias).to(cache_ptr.dtype.element_ty).to(tl.float32)
        _sample_column(lf, keys, row_mask, pid, r, num_steps, step, idx_mapping_ptr, sample_pos_ptr, temp_ptr,
                       seeds_ptr, cache_ptr, cache_stride_0, cache_stride_1, best_ptr, BLOCK_ROWS, KEYS)
    tl.debug_barrier()
    arrival = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    if arrival == tl.num_programs(0) - 1:
        k = tl.arange(0, RANK)
        for r in range(num_reqs):
            best = tl.atomic_xchg(best_ptr + r, 0, sem="acq_rel", scope="gpu")
            token = 2147483647 - (best & 2147483647)
            tl.store(draft_tokens_ptr + r * draft_tokens_stride + step, token)
            if HAS_NEXT:
                tl.store(next_embed_ptr + r * embed_stride + k, tl.load(w1_ptr + token * w1_stride + k))
        tl.atomic_xchg(counter_ptr, 0, sem="acq_rel", scope="gpu")


class FusedMarkovGemv:
    """Buffers + launchers: the whole draft-position loop is `num_steps` kernels (anchor embedding fused into the
    first), each a GEMV + sample over `Kv` rows (`keys` maps reduced rows to target ids; None = full vocab)."""

    def __init__(self, max_num_reqs: int, w2: torch.Tensor, w1: torch.Tensor, keys: torch.Tensor | None = None):
        assert w2.dim() == 2 and w2.stride(1) == 1 and w1.stride(1) == 1 and w2.shape[1] == w1.shape[1]
        assert w2.dtype == torch.bfloat16 and w1.dtype == torch.bfloat16
        self.w2, self.w1, self.keys = w2, w1, keys
        self.rank = w2.shape[1]
        self.kv = w2.shape[0]
        dev = w2.device
        self.embed = torch.zeros(2, max_num_reqs, self.rank, dtype=torch.bfloat16, device=dev)  # ping-pong
        self.best = torch.zeros(max_num_reqs, dtype=torch.int64, device=dev)
        self.counter = torch.zeros(1, dtype=torch.int32, device=dev)
        self.grid = (triton.cdiv(self.kv, BLOCK_ROWS),)

    def run(self, base_logits, anchor_tokens, anchor_stride, idx_mapping, sample_pos, temperature, seeds,
            logits_cache, draft_tokens) -> None:
        """All positions: base_logits [R, num_steps, Kv]; anchor_tokens[r * anchor_stride] = position-0 token."""
        num_reqs, num_steps, kv = base_logits.shape
        assert kv == self.kv and base_logits.stride(2) == 1 and logits_cache.stride(2) == 1
        r_pad = max(16, triton.next_power_of_2(num_reqs))
        keys = self.keys if self.keys is not None else self.counter  # dummy pointer when unused
        for step in range(num_steps):
            cur, nxt = self.embed[step % 2], self.embed[(step + 1) % 2]
            _markov_gemv_sample_kernel[self.grid](
                base_logits, base_logits.stride(0), base_logits.stride(1),
                self.w2, self.w2.stride(0), keys,
                cur, cur.stride(0), nxt, self.w1, self.w1.stride(0),
                anchor_tokens, anchor_stride,
                step, num_steps, num_reqs,
                idx_mapping, sample_pos, temperature, seeds,
                logits_cache, logits_cache.stride(0), logits_cache.stride(1),
                draft_tokens, draft_tokens.stride(0),
                self.best, self.counter, kv,
                RANK=self.rank, BLOCK_ROWS=BLOCK_ROWS, R_PAD=r_pad, KEYS=self.keys is not None,
                HAS_NEXT=step + 1 < num_steps, FUSE_ANCHOR=step == 0, USE_PDL=USE_PDL and step > 0,
                BLOCK_K=BLOCK_K, NUM_STAGES=NUM_STAGES, num_warps=NUM_WARPS, launch_pdl=USE_PDL and step > 0,
            )


# ---- separate bias GEMV (no atomics) for tail_markov's fused sampling kernel ------------------------------------
@triton.jit
def _bias_gemv_kernel(e_ptr, e_stride, w_ptr, w_stride, out_ptr, out_stride, M, V,
                      RANK: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr, M_PAD: tl.constexpr,
                      STAGES: tl.constexpr):
    """bias[M, V] = bf16(e[M, RANK] @ w[V, RANK]^T), fp32 accumulation on tensor cores (swap-AB: V is the MMA M side)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_V + tl.arange(0, BLOCK_V)
    ms = tl.arange(0, M_PAD)
    acc = tl.zeros((BLOCK_V, M_PAD), dtype=tl.float32)
    for k0 in tl.range(0, RANK, BLOCK_K, num_stages=STAGES):
        kk = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + rows[:, None].to(tl.int64) * w_stride + kk[None, :], mask=(rows < V)[:, None], other=0.0)
        et = tl.load(e_ptr + ms[None, :] * e_stride + kk[:, None], mask=(ms < M)[None, :], other=0.0)
        acc = tl.dot(w, et, acc)
    tl.store(out_ptr + ms[None, :] * out_stride + rows[:, None], acc.to(tl.bfloat16),
             mask=(rows < V)[:, None] & (ms < M)[None, :])


def bias_gemv(e: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    m, r = e.shape
    v = w.shape[0]
    if out is None:
        out = torch.empty(m, v, dtype=torch.bfloat16, device=e.device)
    bv = 128
    _bias_gemv_kernel[(triton.cdiv(v, bv),)](
        e, e.stride(0), w, w.stride(0), out, out.stride(0), m, v, RANK=r, BLOCK_V=bv,
        BLOCK_K=128, M_PAD=max(16, triton.next_power_of_2(m)), STAGES=2, num_warps=4)
    return out
