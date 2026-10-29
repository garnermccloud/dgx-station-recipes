"""GPU suffix (copy) drafting for one decoding request: match, draft chain, one-hot draft distribution.

W6 v2 (post-draft fusion): draft() runs match and chain as ONE launch (_match_chain_kernel: every program matches its
block, takes a ticket, the last one runs the chain) and stores the copy-mode flag and the counters straight into
host-mapped memory (lflag_host / stats_host are UvaBuffer host tensors), so the two DtoH copies are gone. Outputs
are bit-identical to the three-launch version (tests/test_post_fused.py runs both side by side).

All three kernels read the request's token history straight from the model runner's `all_token_ids` (UVA, host
memory mapped into the GPU) and its device-side `total_len`, so nothing waits on the CPU and nothing is copied.

  match     For every candidate position j of the history, the length of the longest common suffix of
            tok[:j] and two query sequences, capped at LCAP:
              A = tok[:T]                (the history: drafts the whole chain by copying)
              B = tok[:T] + d[0:ND]      (the history extended by DSpark's ND drafts: continues after them)
            and keeps, per query, the longest match, ties to the most recent j (atomic max of l << 32 | j).
  chain     One program. If l_A >= gpu_th the chain is the copy of tok[j_A:] (periodic past T, which is the
            self-overlapping case of a repeat); otherwise DSpark's ND drafts, then the copy that follows match B
            (or a filler token when there is none). Writes the chain into the drafter's draft-token row, the
            one-hot flags and tokens, and l_A for the scheduler.
  onehot    Writes q = one-hot(token) (0 at the token, -inf elsewhere, pre-temperature logits) into the
            rejection sampler's draft-logits cache for every one-hot chain position. reset_onehot restores
            -inf at the token columns of DSpark-owned rows (< ND) before the drafter writes them again, so a
            drafter that writes only part of the row (reduced vocabularies) still sees -inf elsewhere.

A one-hot q is the exact distribution of a deterministic draft (the copy is a function of the context and of
DSpark's sampled drafts, never of the verification randomness), so block or token-wise rejection with it is
lossless: q(x) = 1 gives acceptance min(1, p(x)), and the residual max(p - q, 0) is p without x.
"""
from __future__ import annotations

import types

# torch / Triton are imported lazily (this module is importable without them; the kernels are built on first use).
LCAP = 32          # longest match length measured (thresholds above this are never met)
MATCH_BLOCK = 256
_KERNELS: types.SimpleNamespace | None = None


def kernels() -> types.SimpleNamespace:
    """Build the four Triton kernels once (match, chain, onehot, reset)."""
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _build()
    return _KERNELS


def _build() -> types.SimpleNamespace:
    from vllm.triton_utils import tl, triton


    @triton.jit
    def _match_kernel(tok_ptr, tok_stride, hist_idx_ptr, total_len_ptr, idx_mapping_ptr, drafts_ptr, best_ptr,
                      LCAP: tl.constexpr, ND: tl.constexpr, BLOCK: tl.constexpr):
        # Each program takes BLOCK candidate end positions j and loads their [BLOCK, LCAP] windows tok[j-1-k] as one 2D
        # (coalesced, mutually independent) load; the match length is the first mismatching k, a min-reduction.
        pid = tl.program_id(0)
        slot = tl.load(idx_mapping_ptr).to(tl.int64)
        T = tl.load(total_len_ptr + slot).to(tl.int64)
        if pid * BLOCK > T:
            return
        row = tok_ptr + tl.load(hist_idx_ptr).to(tl.int64) * tok_stride
        k = tl.arange(0, LCAP).to(tl.int64)
        # query tails, k = 0 the most recent token: A = tok[:T], B = tok[:T] + d[:ND]
        ta_pos = T - 1 - k
        tail_a = tl.load(row + tl.maximum(ta_pos, 0))
        tail_a = tl.where(ta_pos >= 0, tail_a, -2)
        tb_pos = T - 1 - (k - ND)
        tail_bh = tl.load(row + tl.maximum(tb_pos, 0))
        tail_bd = tl.load(drafts_ptr + tl.where(k < ND, ND - 1 - k, 0)).to(tl.int32)
        tail_b = tl.where(k < ND, tail_bd, tl.where(tb_pos >= 0, tail_bh, -2))
        j = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        valid = (j >= 1) & (j <= T - 1)
        # filter: only positions whose last token matches a query's last token load their window
        w0 = tl.load(row + j - 1, mask=valid, other=-1)
        t_a0 = tl.load(row + tl.maximum(T - 1, 0))
        t_b0 = tl.load(drafts_ptr + (ND - 1)).to(tl.int32)
        cand = valid & ((w0 == t_a0) | (w0 == t_b0))
        if tl.max(cand.to(tl.int32), axis=0) == 0:
            return
        pos = j[:, None] - 1 - k[None, :]
        win = tl.load(row + pos, mask=cand[:, None] & (pos >= 0), other=-1)
        kk = k[None, :] + tl.zeros([BLOCK, LCAP], dtype=tl.int64)
        len_a = tl.min(tl.where(win == tail_a[None, :], LCAP, kk), axis=1)
        len_b = tl.min(tl.where(win == tail_b[None, :], LCAP, kk), axis=1)
        key_a = tl.where(cand & (len_a > 0), (len_a << 32) | j, 0)
        key_b = tl.where(cand & (len_b > 0), (len_b << 32) | j, 0)
        tl.atomic_max(best_ptr + 0, tl.max(key_a, axis=0))
        tl.atomic_max(best_ptr + 1, tl.max(key_b, axis=0))


    @triton.jit
    def _chain_kernel(tok_ptr, tok_stride, hist_idx_ptr, total_len_ptr, idx_mapping_ptr, drafts_ptr, best_ptr,
                      onehot_ptr, inj_tok_ptr, inj_slot_ptr, prev_tok_ptr, prev_slot_ptr, lflag_ptr, stats_ptr,
                      gpu_th, filler, K: tl.constexpr, ND: tl.constexpr, KP: tl.constexpr):
        slot = tl.load(idx_mapping_ptr).to(tl.int64)
        # rows >= ND hold -inf except one 0 at the last injected token: remember where it is before overwriting
        pp = tl.arange(0, KP)
        tl.store(prev_tok_ptr + pp, tl.load(inj_tok_ptr + pp, mask=pp < K, other=0), mask=pp < K)
        tl.store(prev_slot_ptr, tl.load(inj_slot_ptr))
        tl.debug_barrier()
        T = tl.load(total_len_ptr + slot).to(tl.int64)
        row = tok_ptr + tl.load(hist_idx_ptr).to(tl.int64) * tok_stride
        key_a = tl.load(best_ptr + 0)
        key_b = tl.load(best_ptr + 1)
        tl.debug_barrier()
        tl.store(best_ptr + 0, 0)
        tl.store(best_ptr + 1, 0)
        l_a = key_a >> 32
        j_a = key_a & 0xFFFFFFFF
        l_b = key_b >> 32
        j_b = key_b & 0xFFFFFFFF
        use_copy = (l_a >= gpu_th) & (l_a > 0)

        p = tl.arange(0, KP).to(tl.int64)
        pm = p < K
        # copy A: tok[j_a + p], periodic with period T - j_a past the end of the history
        per_a = tl.maximum(T - j_a, 1)
        tok_a = tl.load(row + j_a + p % per_a, mask=pm & use_copy, other=0).to(tl.int64)
        # copy B after the ND drafts: virtual sequence tok[:T] + d[:ND], from j_b, periodic with period T + ND - j_b
        q = p - ND
        per_b = tl.maximum(T + ND - j_b, 1)
        vb = j_b + tl.where(q >= 0, q, 0) % per_b
        in_hist = vb < T
        has_b = (q >= 0) & (l_b > 0)
        tok_bh = tl.load(row + vb, mask=pm & has_b & in_hist, other=0).to(tl.int64)
        tok_bd = tl.load(drafts_ptr + tl.where(in_hist, 0, vb - T), mask=pm & has_b & (~in_hist), other=0)
        tok_b = tl.where(in_hist, tok_bh, tok_bd)
        dsp = tl.load(drafts_ptr + p, mask=pm & (p < ND), other=0)
        chain = tl.where(use_copy, tok_a, tl.where(p < ND, dsp, tl.where(l_b > 0, tok_b, filler)))
        onehot = use_copy | (p >= ND)
        tl.store(drafts_ptr + p, chain, mask=pm)
        tl.store(onehot_ptr + p, onehot.to(tl.int8), mask=pm)
        tl.store(inj_tok_ptr + p, chain, mask=pm)
        tl.store(inj_slot_ptr, slot)
        tl.store(lflag_ptr + slot, l_a.to(tl.int32))
        # stats: [batch-1 proposals, copy chains, extension matches, sum l_a]
        tl.atomic_add(stats_ptr + 0, 1)
        tl.atomic_add(stats_ptr + 1, use_copy.to(tl.int64))
        tl.atomic_add(stats_ptr + 2, ((~use_copy) & (l_b > 0)).to(tl.int64))
        tl.atomic_add(stats_ptr + 3, l_a)


    @triton.jit
    def _match_chain_kernel(tok_ptr, tok_stride, hist_idx_ptr, total_len_ptr, idx_mapping_ptr, drafts_ptr, best_ptr,
                            ticket_ptr, onehot_ptr, inj_tok_ptr, inj_slot_ptr, prev_tok_ptr, prev_slot_ptr, lflag_ptr,
                            stats_ptr, lflag_host_ptr, stats_host_ptr, gpu_th, filler, num_slots,
                            LCAP: tl.constexpr, ND: tl.constexpr, BLOCK: tl.constexpr, K: tl.constexpr,
                            KP: tl.constexpr, SP: tl.constexpr):
        # _match_kernel's program, then a ticket; the last program to take one runs _chain_kernel's body (one launch
        # instead of two) and also stores the flag and the counters into host-mapped memory (no DtoH copies).
        pid = tl.program_id(0)
        slot = tl.load(idx_mapping_ptr).to(tl.int64)
        T = tl.load(total_len_ptr + slot).to(tl.int64)
        row = tok_ptr + tl.load(hist_idx_ptr).to(tl.int64) * tok_stride
        if pid * BLOCK <= T:
            k = tl.arange(0, LCAP).to(tl.int64)
            ta_pos = T - 1 - k
            tail_a = tl.load(row + tl.maximum(ta_pos, 0))
            tail_a = tl.where(ta_pos >= 0, tail_a, -2)
            tb_pos = T - 1 - (k - ND)
            tail_bh = tl.load(row + tl.maximum(tb_pos, 0))
            tail_bd = tl.load(drafts_ptr + tl.where(k < ND, ND - 1 - k, 0)).to(tl.int32)
            tail_b = tl.where(k < ND, tail_bd, tl.where(tb_pos >= 0, tail_bh, -2))
            j = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            valid = (j >= 1) & (j <= T - 1)
            w0 = tl.load(row + j - 1, mask=valid, other=-1)
            t_a0 = tl.load(row + tl.maximum(T - 1, 0))
            t_b0 = tl.load(drafts_ptr + (ND - 1)).to(tl.int32)
            cand = valid & ((w0 == t_a0) | (w0 == t_b0))
            if tl.max(cand.to(tl.int32), axis=0) != 0:
                pos = j[:, None] - 1 - k[None, :]
                win = tl.load(row + pos, mask=cand[:, None] & (pos >= 0), other=-1)
                kk = k[None, :] + tl.zeros([BLOCK, LCAP], dtype=tl.int64)
                len_a = tl.min(tl.where(win == tail_a[None, :], LCAP, kk), axis=1)
                len_b = tl.min(tl.where(win == tail_b[None, :], LCAP, kk), axis=1)
                key_a = tl.where(cand & (len_a > 0), (len_a << 32) | j, 0)
                key_b = tl.where(cand & (len_b > 0), (len_b << 32) | j, 0)
                tl.atomic_max(best_ptr + 0, tl.max(key_a, axis=0))
                tl.atomic_max(best_ptr + 1, tl.max(key_b, axis=0))
        ticket = tl.atomic_add(ticket_ptr, 1)  # acq_rel: every program's maxima are visible to the last one
        if ticket == tl.num_programs(0) - 1:
            pp = tl.arange(0, KP)
            tl.store(prev_tok_ptr + pp, tl.load(inj_tok_ptr + pp, mask=pp < K, other=0), mask=pp < K)
            tl.store(prev_slot_ptr, tl.load(inj_slot_ptr))
            tl.debug_barrier()
            key_a = tl.atomic_xchg(best_ptr + 0, 0)
            key_b = tl.atomic_xchg(best_ptr + 1, 0)
            tl.store(ticket_ptr, 0)
            l_a = key_a >> 32
            j_a = key_a & 0xFFFFFFFF
            l_b = key_b >> 32
            j_b = key_b & 0xFFFFFFFF
            use_copy = (l_a >= gpu_th) & (l_a > 0)
            p = tl.arange(0, KP).to(tl.int64)
            pm = p < K
            per_a = tl.maximum(T - j_a, 1)
            tok_a = tl.load(row + j_a + p % per_a, mask=pm & use_copy, other=0).to(tl.int64)
            q = p - ND
            per_b = tl.maximum(T + ND - j_b, 1)
            vb = j_b + tl.where(q >= 0, q, 0) % per_b
            in_hist = vb < T
            has_b = (q >= 0) & (l_b > 0)
            tok_bh = tl.load(row + vb, mask=pm & has_b & in_hist, other=0).to(tl.int64)
            tok_bd = tl.load(drafts_ptr + tl.where(in_hist, 0, vb - T), mask=pm & has_b & (~in_hist), other=0)
            tok_b = tl.where(in_hist, tok_bh, tok_bd)
            dsp = tl.load(drafts_ptr + p, mask=pm & (p < ND), other=0)
            chain = tl.where(use_copy, tok_a, tl.where(p < ND, dsp, tl.where(l_b > 0, tok_b, filler)))
            onehot = use_copy | (p >= ND)
            tl.store(drafts_ptr + p, chain, mask=pm)
            tl.store(onehot_ptr + p, onehot.to(tl.int8), mask=pm)
            tl.store(inj_tok_ptr + p, chain, mask=pm)
            tl.store(inj_slot_ptr, slot)
            tl.store(lflag_ptr + slot, l_a.to(tl.int32))
            # the whole flag array, as the three-launch version's lflag_host.copy_(lflag) delivered it
            r = tl.arange(0, SP)
            rm = r < num_slots
            flags = tl.where(r == slot, l_a.to(tl.int32), tl.load(lflag_ptr + r, mask=rm, other=0))
            tl.store(lflag_host_ptr + r, flags, mask=rm)
            s0 = tl.atomic_add(stats_ptr + 0, 1)
            s1 = tl.atomic_add(stats_ptr + 1, use_copy.to(tl.int64))
            s2 = tl.atomic_add(stats_ptr + 2, ((~use_copy) & (l_b > 0)).to(tl.int64))
            s3 = tl.atomic_add(stats_ptr + 3, l_a)
            tl.store(stats_host_ptr + 0, s0 + 1)
            tl.store(stats_host_ptr + 1, s1 + use_copy.to(tl.int64))
            tl.store(stats_host_ptr + 2, s2 + ((~use_copy) & (l_b > 0)).to(tl.int64))
            tl.store(stats_host_ptr + 3, s3 + l_a)


    @triton.jit
    def _onehot_kernel(logits_ptr, stride0, stride1, onehot_ptr, inj_tok_ptr, inj_slot_ptr, prev_tok_ptr,
                       prev_slot_ptr, vocab, ND: tl.constexpr, VB: tl.constexpr):
        # rows < ND (DSpark's) become one-hot only for a pure copy chain: full-row rewrite, restored by _reset_kernel.
        # rows >= ND are only ever written here: move the single 0 from last step's (slot, token) to this step's.
        p = tl.program_id(0)
        vblk = tl.program_id(1)
        slot = tl.load(inj_slot_ptr).to(tl.int64)
        tok = tl.load(inj_tok_ptr + p)
        if p < ND:
            if tl.load(onehot_ptr + p) != 0:
                cols = vblk * VB + tl.arange(0, VB).to(tl.int64)
                vals = tl.where(cols == tok, 0.0, float("-inf"))
                tl.store(logits_ptr + slot * stride0 + p * stride1 + cols, vals.to(logits_ptr.dtype.element_ty),
                         mask=cols < vocab)
        elif vblk == 0:
            prev_slot = tl.load(prev_slot_ptr).to(tl.int64)
            prev_tok = tl.load(prev_tok_ptr + p)
            tl.store(logits_ptr + prev_slot * stride0 + p * stride1 + prev_tok,
                     tl.full([], float("-inf"), tl.float32).to(logits_ptr.dtype.element_ty))
            tl.debug_barrier()
            tl.store(logits_ptr + slot * stride0 + p * stride1 + tok, tl.zeros([], tl.float32).to(logits_ptr.dtype.element_ty))


    @triton.jit
    def _reset_kernel(logits_ptr, stride0, stride1, onehot_ptr, inj_tok_ptr, inj_slot_ptr, ND: tl.constexpr,
                      NDP: tl.constexpr):
        p = tl.arange(0, NDP)
        m = p < ND
        flag = tl.load(onehot_ptr + p, mask=m, other=0)
        slot = tl.load(inj_slot_ptr).to(tl.int64)
        tok = tl.load(inj_tok_ptr + p, mask=m, other=0)
        ptrs = logits_ptr + slot * stride0 + p.to(tl.int64) * stride1 + tok
        tl.store(ptrs, tl.full([NDP], float("-inf"), tl.float32).to(logits_ptr.dtype.element_ty),
                 mask=m & (flag != 0))
        tl.store(onehot_ptr + p, tl.zeros([NDP], tl.int8), mask=m)

    return types.SimpleNamespace(match=_match_kernel, chain=_chain_kernel, match_chain=_match_chain_kernel,
                                 onehot=_onehot_kernel, reset=_reset_kernel, triton=triton)


class SuffixDrafter:
    """Persistent device buffers and launches for batch-1 suffix drafting.

    The matcher scans a device mirror of the batch-1 request's history instead of the runner's UVA copy (a 512K-token
    UVA scan costs ~110 us; HBM ~10x less). The mirror holds tok[:ub + 1] of one request: a full copy when the request
    (or its slot) changes or after any multi-request step, else only the window that can have changed since the last
    batch-1 step (the async upper bound ub overshoots the true length by at most the in-flight drafts).
    """

    def __init__(self, max_num_reqs: int, k_total: int, k_dspark: int, filler: int, max_model_len: int, device):
        import torch

        self.kern = kernels()
        self.K, self.ND, self.filler = k_total, k_dspark, filler
        self.best = torch.zeros(2, dtype=torch.int64, device=device)
        self.onehot = torch.zeros(k_total, dtype=torch.int8, device=device)
        self.inj_tok = torch.zeros(k_total, dtype=torch.int64, device=device)
        self.inj_slot = torch.zeros(1, dtype=torch.int64, device=device)
        self.prev_tok = torch.zeros(k_total, dtype=torch.int64, device=device)
        self.prev_slot = torch.zeros(1, dtype=torch.int64, device=device)
        from vllm.v1.worker.gpu.buffer_utils import UvaBuffer

        self.lflag = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.stats = torch.zeros(4, dtype=torch.int64, device=device)
        # host-mapped copies the fused kernel stores into directly (pinned host tensors for the scheduler / logs,
        # their device views for the kernel): no DtoH copies
        self._lflag_buf = UvaBuffer(max_num_reqs, torch.int32)
        self._stats_buf = UvaBuffer(4, torch.int64)
        self.lflag_host, self.stats_host = self._lflag_buf.cpu, self._stats_buf.cpu
        self.ticket = torch.zeros(1, dtype=torch.int32, device=device)
        self.pending_reset = False
        self.mirror = torch.zeros(1, max_model_len + 64, dtype=torch.int32, device=device)
        self.mirror_slot0 = torch.zeros(1, dtype=torch.int32, device=device)  # the mirror is "slot 0" of itself
        self.mirror_key: tuple | None = None   # (req_id, slot) mirrored
        self.mirror_hi = 0                     # tok[:mirror_hi] copied
        self.window = 4 * (k_total + 1) + 16
        self.full_copies = 0

    def invalidate(self) -> None:
        """A step ran without the mirror (multi-request batch): the next batch-1 step copies the whole history."""
        self.mirror_key = None

    def sync_mirror(self, all_token_ids, req_id: str, slot: int, ub: int) -> None:
        """Make mirror[:ub + 1] equal all_token_ids[slot, :ub + 1] for every position the matcher may read."""
        hi = min(ub + 1, all_token_ids.shape[1], self.mirror.shape[1])
        if self.mirror_key != (req_id, slot):
            lo = 0
            self.full_copies += 1
        else:
            lo = max(0, min(self.mirror_hi, hi) - self.window)
        if hi > lo:
            self.mirror[0, lo:hi].copy_(all_token_ids[slot, lo:hi], non_blocking=True)
        self.mirror_key, self.mirror_hi = (req_id, slot), hi

    def sync_mirror_async(self, all_token_ids, req_id: str, slot: int, ub: int):
        """sync_mirror's copy on a side stream that starts where the main stream is now (before the drafter graph,
        which it then runs under); returns the event the main stream must wait on before draft()."""
        import torch

        main = torch.cuda.current_stream()
        if getattr(self, "_side", None) is None:
            self._side = torch.cuda.Stream(device=main.device)
        ready = torch.cuda.Event()
        ready.record(main)
        self._side.wait_event(ready)
        with torch.cuda.stream(self._side):
            self.sync_mirror(all_token_ids, req_id, slot, ub)
            done = torch.cuda.Event()
            done.record(self._side)
        return done

    def reset_onehot(self, draft_logits) -> None:
        """Before the drafter writes its rows: put -inf back at last step's one-hot token columns (rows < ND)."""
        if not self.pending_reset:
            return
        k = self.kern
        k.reset[(1,)](draft_logits, draft_logits.stride(0), draft_logits.stride(1), self.onehot,
                            self.inj_tok, self.inj_slot, ND=self.ND, NDP=k.triton.next_power_of_2(self.ND))
        self.pending_reset = False

    def draft(self, history, hist_idx, total_len, idx_mapping, drafts_row, draft_logits, seq_len_upper: int,
              gpu_th: int) -> None:
        """Rewrite drafts_row (the drafter's [K] int64 row for batch index 0) and the one-hot q rows.

        history[hist_idx[0]] is the token row to scan (the mirror, or the runner's all_token_ids for tests);
        total_len[idx_mapping[0]] its true length; idx_mapping[0] the request's slot for the q cache and flags.
        """
        k = self.kern
        grid = (k.triton.cdiv(max(seq_len_upper, 2), MATCH_BLOCK),)
        k.match_chain[grid](history, history.stride(0), hist_idx, total_len, idx_mapping, drafts_row, self.best,
                            self.ticket, self.onehot, self.inj_tok, self.inj_slot, self.prev_tok, self.prev_slot,
                            self.lflag, self.stats, self._lflag_buf.uva(), self._stats_buf.uva(), gpu_th, self.filler,
                            self.lflag.shape[0], LCAP=LCAP, ND=self.ND, BLOCK=MATCH_BLOCK, K=self.K,
                            KP=k.triton.next_power_of_2(self.K), SP=k.triton.next_power_of_2(self.lflag.shape[0]))
        if draft_logits is not None:
            vocab = draft_logits.shape[-1]
            VB = 8192
            k.onehot[(self.K, k.triton.cdiv(vocab, VB))](
                draft_logits, draft_logits.stride(0), draft_logits.stride(1), self.onehot, self.inj_tok,
                self.inj_slot, self.prev_tok, self.prev_slot, vocab, ND=self.ND, VB=VB)
            self.pending_reset = True
