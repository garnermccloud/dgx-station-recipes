"""tail_dflash_prep: one launch of the draft input preparation for every draft KV-cache group.

Production's DFlashSpeculator.propose calls prepare_dflash_inputs once per draft KV-cache group (3 for the DSpark
drafter: one per draft layer's sliding-window cache), each launch rewriting every shared output (query input ids,
positions, query_start_loc, seq_lens, context positions, sampling indices, temperature, seeds) plus its own group's
query and context slot mappings. Each launch costs ~8.5 us at batch 1, almost all of it the CUDA-graph padding
loops, which step through max_num_tokens (8192) in BLOCK_SIZE = 16 chunks in a single program.

This component runs one launch with a third grid axis over the draft groups (after NVIDIA vLLM #61869's multi-group
_prepare_dflash_inputs_kernel, re-derived from this nightly's kernel): group g writes its own slot mappings, group 0
also writes the shared outputs, and the padding loops step 1024 entries at a time. Every output buffer ends up
bitwise identical to the production sequence (tests/test_dflash_prep.py); only the redundant rewrites are gone.

Wiring: DFlashSpeculator.propose is wrapped only to publish the running speculator; the module-level
prepare_dflash_inputs it calls per group is replaced by a function that does the whole multi-group launch on the
first group's call and returns immediately for the others. propose itself is untouched.
"""
from __future__ import annotations

import sys
import threading

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

DFLASH_MODULE = "vllm.v1.worker.gpu.spec_decode.dflash.speculator"
_current = threading.local()
_kernel = None


def _build_kernel():
    from vllm.triton_utils import tl, triton
    from vllm.v1.worker.gpu.block_table import _load_ptr
    from vllm.v1.worker.gpu.cp_utils import cp_local_slot

    @triton.jit
    def _prepare_dflash_inputs_multigroup_kernel(
        out_input_ids_ptr, out_query_positions_ptr, out_query_start_loc_ptr, out_seq_lens_ptr,
        out_query_slot_mappings_ptr, query_slot_mappings_stride,  # [num_kv_cache_groups, max_num_tokens]
        out_context_positions_ptr,
        out_context_slot_mappings_ptr, context_slot_mappings_stride,  # [num_draft_groups, max_num_tokens]
        out_sample_indices_ptr, out_sample_pos_ptr, out_sample_idx_mapping_ptr, out_temperature_ptr, out_seeds_ptr,
        target_positions_ptr, target_query_start_loc_ptr, idx_mapping_ptr, last_sampled_ptr, next_prefill_tokens_ptr,
        num_sampled_ptr, num_rejected_ptr, temperature_ptr, seeds_ptr,
        draft_group_ids_ptr,  # [num_draft_groups] KV-cache group id of each draft group
        block_table_ptrs, block_table_strides, kernel_block_sizes_ptr,  # [num_kv_cache_groups]
        parallel_drafting_token_id, num_query_per_req, num_speculative_steps, max_num_reqs, max_num_tokens,
        max_model_len, cp_rank,
        SAMPLE_FROM_ANCHOR: tl.constexpr, PAD_SLOT_ID: tl.constexpr, CP_SIZE: tl.constexpr,
        CP_INTERLEAVE: tl.constexpr, BLOCK_SIZE: tl.constexpr, PAD_BLOCK_SIZE: tl.constexpr,
    ):
        req_idx = tl.program_id(0)
        block_idx = tl.program_id(1)
        draft_group = tl.program_id(2)
        is_first_group = draft_group == 0
        num_reqs = tl.num_programs(0)
        req_state_idx = tl.load(idx_mapping_ptr + req_idx)

        group_id = tl.load(draft_group_ids_ptr + draft_group)
        block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
        block_table_stride = tl.load(block_table_strides + group_id)
        block_size = tl.load(kernel_block_sizes_ptr + group_id)
        out_query_slot_mapping_ptr = out_query_slot_mappings_ptr + group_id * query_slot_mappings_stride
        out_context_slot_mapping_ptr = out_context_slot_mappings_ptr + draft_group * context_slot_mappings_stride

        ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
        ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
        num_ctx = ctx_end - ctx_start
        num_rejected = tl.load(num_rejected_ptr + req_idx)
        valid_ctx_end = ctx_end - num_rejected
        num_valid_ctx = valid_ctx_end - ctx_start
        num_sampled = tl.load(num_sampled_ptr + req_idx)
        if num_sampled > 0:
            bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
        else:
            bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)
        last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
        query_base = req_idx * num_query_per_req

        j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        is_ctx = j < num_ctx
        is_valid_ctx = j < num_valid_ctx
        is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)
        query_off = j - num_valid_ctx

        # --- Context positions / slots ---
        ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
        ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
        ctx_block_num = ctx_pos // (block_size * CP_SIZE)
        ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
        ctx_block_id = tl.load(block_table_ptr + req_idx * block_table_stride + ctx_block_num,
                               mask=is_valid_ctx, other=0).to(tl.int64)
        ctx_resident = is_valid_ctx & (ctx_block_id != 0)
        local_ctx_slot = cp_local_slot(ctx_pos, ctx_block_id, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE,
                                       PAD_SLOT_ID)
        ctx_slot = tl.where(ctx_resident, local_ctx_slot, PAD_SLOT_ID)
        tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

        # --- Query slots ---
        query_pos = last_valid_pos + 1 + query_off
        query_idx = query_base + query_off
        q_block_num = query_pos // (block_size * CP_SIZE)
        q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
        q_block_id = tl.load(block_table_ptr + req_idx * block_table_stride + q_block_num,
                             mask=is_query, other=0).to(tl.int64)
        q_resident = is_query & (q_block_id != 0)
        local_q_slot = cp_local_slot(query_pos, q_block_id, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE,
                                     PAD_SLOT_ID)
        q_slot = tl.where(q_resident, local_q_slot, PAD_SLOT_ID)
        tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

        if is_first_group:
            tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
            is_bonus = is_query & (query_off == 0)
            input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)
            tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
            clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
            tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
            sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
            is_sample = is_query & (query_off >= sample_off)
            sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
            sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
            tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
            tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
            tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

        if block_idx == 0:
            if is_first_group:
                tl.store(out_query_start_loc_ptr + req_idx, query_base)
                tl.store(out_seq_lens_ptr + req_idx, tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len))
                tl.store(out_temperature_ptr + req_state_idx, tl.load(temperature_ptr + req_state_idx))
                tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))
            if req_idx == num_reqs - 1:
                q_pad_start = num_reqs * num_query_per_req
                if is_first_group:
                    last_query_end = num_reqs * num_query_per_req
                    for i in range(num_reqs, max_num_reqs + 1, PAD_BLOCK_SIZE):
                        block = i + tl.arange(0, PAD_BLOCK_SIZE)
                        tl.store(out_query_start_loc_ptr + block, last_query_end, mask=block < max_num_reqs + 1)
                    for i in range(num_reqs, max_num_reqs, PAD_BLOCK_SIZE):
                        block = i + tl.arange(0, PAD_BLOCK_SIZE)
                        tl.store(out_seq_lens_ptr + block, 0, mask=block < max_num_reqs)
                    pad_start = num_reqs * num_speculative_steps
                    pad_end = max_num_reqs * num_speculative_steps
                    for i in range(pad_start, pad_end, PAD_BLOCK_SIZE):
                        block = i + tl.arange(0, PAD_BLOCK_SIZE)
                        mask = block < pad_end
                        tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                        tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                        tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
                for i in range(q_pad_start, max_num_tokens, PAD_BLOCK_SIZE):
                    block = i + tl.arange(0, PAD_BLOCK_SIZE)
                    tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=block < max_num_tokens)

    return _prepare_dflash_inputs_multigroup_kernel


class _GroupLayout:
    """Device arrays the multi-group launch reads, built once per speculator (fixed buffer addresses)."""

    def __init__(self, spec):
        import torch

        bt = spec.block_tables
        dev = spec.device
        self.gids = list(spec.draft_kv_cache_group_ids)
        self.draft_group_ids = torch.tensor(self.gids, dtype=torch.int32, device=dev)
        for g in self.gids:
            assert bt.input_block_tables[g].stride(0) == bt.block_tables[g].gpu.stride(0)
        self.first_ctx_ptr = spec._context_slot_mappings[0].data_ptr()


def prepare_dflash_inputs_all_groups(spec, input_buffers, context_positions, sample_indices, sample_pos,
                                     sample_idx_mapping, temperature, seeds, input_batch, num_sampled, num_rejected,
                                     last_sampled, next_prefill_tokens, input_temperature, input_seeds,
                                     parallel_drafting_token_id, num_query_per_req, num_speculative_steps,
                                     max_num_reqs, max_num_tokens, max_model_len, sample_from_anchor) -> None:
    """The whole multi-group launch (what production's per-group loop computes)."""
    global _kernel
    from vllm.triton_utils import triton
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    if _kernel is None:
        _kernel = _build_kernel()
    layout = spec._w5_dflash_layout
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    block = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, block)
    qsm = spec.block_tables.slot_mappings
    csm = spec._context_slot_mappings
    assert qsm.stride(1) == 1 and csm.stride(1) == 1
    _kernel[(num_reqs, num_blocks, len(layout.gids))](
        input_buffers.input_ids, input_buffers.positions, input_buffers.query_start_loc, input_buffers.seq_lens,
        qsm, qsm.stride(0), context_positions, csm, csm.stride(0),
        sample_indices, sample_pos, sample_idx_mapping, temperature, seeds,
        input_batch.positions, input_batch.query_start_loc, input_batch.idx_mapping, last_sampled,
        next_prefill_tokens, num_sampled, num_rejected, input_temperature, input_seeds,
        layout.draft_group_ids, spec.block_tables.input_block_table_ptrs, spec.block_tables.block_table_strides,
        spec.block_tables.kernel_block_sizes_tensor,
        parallel_drafting_token_id, num_query_per_req, num_speculative_steps, max_num_reqs, max_num_tokens,
        max_model_len, spec.block_tables.cp_rank,
        SAMPLE_FROM_ANCHOR=sample_from_anchor, PAD_SLOT_ID=PAD_SLOT_ID, CP_SIZE=spec.dcp_size,
        CP_INTERLEAVE=spec.block_tables.cp_interleave, BLOCK_SIZE=block, PAD_BLOCK_SIZE=1024,
    )


def unsupported_reason(spec) -> str | None:
    bt = spec.block_tables
    for attr in ("input_block_table_ptrs", "block_table_strides", "kernel_block_sizes_tensor"):
        if not hasattr(bt, attr):
            return f"BlockTables has no {attr}"
    return None


@register("tail_dflash_prep", DFLASH_MODULE)
def patch(module) -> None:
    cls = module.DFlashSpeculator
    if getattr(cls, "_w5_dflash_prep", False):
        return
    orig_propose = cls.propose
    orig_prepare = module.prepare_dflash_inputs

    def propose(self, *args, **kwargs):
        # Built on the first propose after set_attn (memory-profiling proposes run before the block tables exist).
        ready = hasattr(self, "_context_slot_mappings") and hasattr(self, "block_tables")
        if getattr(self, "_w5_dflash_layout", None) is None and ready:
            reason = unsupported_reason(self)
            self._w5_dflash_layout = False if reason else _GroupLayout(self)
            sys.stderr.write("DS41_FASTPATH tail_dflash_prep: "
                             + (f"one launch for draft KV groups {self._w5_dflash_layout.gids}\n" if not reason
                                else f"stock per-group launches: {reason}\n"))
        _current.spec = self if getattr(self, "_w5_dflash_layout", None) else None
        try:
            return orig_propose(self, *args, **kwargs)
        finally:
            _current.spec = None

    def prepare_dflash_inputs(input_buffers, query_slot_mapping, context_positions, context_slot_mapping,
                              sample_indices, sample_pos, sample_idx_mapping, temperature, seeds, input_batch,
                              num_sampled, num_rejected, last_sampled, next_prefill_tokens, input_temperature,
                              input_seeds, block_table, block_size, cp_rank, cp_size, cp_interleave,
                              parallel_drafting_token_id, num_query_per_req, num_speculative_steps, max_num_reqs,
                              max_num_tokens, max_model_len, sample_from_anchor=False):
        spec = getattr(_current, "spec", None)
        if spec is None:
            return orig_prepare(input_buffers, query_slot_mapping, context_positions, context_slot_mapping,
                                sample_indices, sample_pos, sample_idx_mapping, temperature, seeds, input_batch,
                                num_sampled, num_rejected, last_sampled, next_prefill_tokens, input_temperature,
                                input_seeds, block_table, block_size, cp_rank, cp_size, cp_interleave,
                                parallel_drafting_token_id, num_query_per_req, num_speculative_steps,
                                max_num_reqs, max_num_tokens, max_model_len, sample_from_anchor)
        if context_slot_mapping.data_ptr() != spec._w5_dflash_layout.first_ctx_ptr:
            return None  # groups 1..G-1: already written by group 0's multi-group launch
        prepare_dflash_inputs_all_groups(
            spec, input_buffers, context_positions, sample_indices, sample_pos, sample_idx_mapping, temperature,
            seeds, input_batch, num_sampled, num_rejected, last_sampled, next_prefill_tokens, input_temperature,
            input_seeds, parallel_drafting_token_id, num_query_per_req, num_speculative_steps, max_num_reqs,
            max_num_tokens, max_model_len, sample_from_anchor)
        return None

    cls.propose = propose
    module.prepare_dflash_inputs = prepare_dflash_inputs
    cls._w5_dflash_prep = True
