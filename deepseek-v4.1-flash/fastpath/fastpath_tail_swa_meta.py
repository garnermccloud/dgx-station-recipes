"""tail_swa_meta: one kernel per sliding-window metadata build for pure-decode batches.

Every decode step builds DeepseekSparseSWAMetadataBuilder metadata for 10 KV-cache groups (7 target groups before the
target forward, 3 draft groups before the drafter), and each decode-only build launches three tiny kernels on the
critical path (~6 us): `torch.ge(slot_mapping, 0, out=is_valid_token)`, the `decode_swa_lens[n:] = 0` fill, and the
causal (target) or DSpark non-causal (drafter) SWA index kernel. Here one kernel does all three:

  every program p < len(slot_mapping) writes is_valid_token[p] = slot_mapping[p] >= 0;
  program p < num_decode_tokens builds token p's window indices and length from that same validity, exactly as
  _compute_swa_indices_and_lens_kernel / ComputeDSparkNoncausalSWAIndicesKernel do (same integer arithmetic);
  the last program zero-fills decode_swa_lens[num_decode_tokens:].

All outputs are integer index data, bitwise identical to the three production ops (tests/test_swa_meta.py). Builds
with prefill tokens, image visibility, or anything else take the stock build() unchanged.

Multi-group launch: inside one build_attn_metadata call (the target's 8 SWA groups, the
drafter's 3) the decode builds only record their arguments; when build_attn_metadata returns, ONE kernel
(grid = tokens x groups) builds every group, reading each group's pointers and constants from a small cached device
table (rebuilt only when a group's buffers or block table move). Deferral happens only inside the wrapped
build_attn_metadata (model_states.default for the target, spec_decode.speculator for the drafter), whose finally
clause always flushes before the call returns, so nothing can read a group's metadata before it is written. The other
builders between the SWA builds (sparse MLA, indexer, compressor) neither read SWA metadata nor write any SWA input
(slot mappings, SWA block tables, seq_lens, query_start_loc, token_to_req, replay starts).

Nested DSpark block (W6, n=5): in the drafter's non-causal block, query rows 0..n-1
see the context window plus only the block's first n tokens; rows n.. see the whole block (stock). With n = 5 (the
1017 checkpoint's trained block) a 7-token block's first 5 rows attend to exactly what a 5-token block's rows attend
to, so drafts 1-5 of a K7 drafter are those of the K5 drafter (the rest of the drafter is per row); blocks of <= n
tokens are unchanged. A non-causal build that falls back to the stock build cannot nest: logged once.
"""
from __future__ import annotations

import sys

try:
    from fastpath_hook import register
except ImportError:  # imported directly by tests
    def register(component, target):
        return lambda fn: fn

SWA_MODULE = "vllm.v1.attention.backends.mla.sparse_swa"
_kernel = None
NEST = 5
_nest_fallback_logged = False


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    from vllm.triton_utils import tl, triton

    @triton.jit(do_not_specialize=["swa_indices_stride", "block_table_stride", "num_slots", "num_decode_tokens",
                                   "lens_len"])
    def _swa_decode_meta_kernel(
        slot_mapping_ptr, num_slots, is_valid_token_ptr,
        swa_indices_ptr, swa_indices_stride, swa_lens_ptr, lens_len,
        window_size, index_width,
        query_start_loc_ptr, seq_lens_ptr, token_to_req_indices_ptr,
        block_table_ptr, block_table_stride, block_size, replay_start_ptr,
        num_decode_tokens,
        NONCAUSAL: tl.constexpr, TRITON_BLOCK_SIZE: tl.constexpr, FILL_BLOCK: tl.constexpr, NEST: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid < num_slots:
            is_valid = tl.load(slot_mapping_ptr + pid) >= 0
            tl.store(is_valid_token_ptr + pid, is_valid)
        else:
            is_valid = False
        if pid == tl.num_programs(0) - 1:
            for i in range(num_decode_tokens, lens_len, FILL_BLOCK):
                offs = i + tl.arange(0, FILL_BLOCK)
                tl.store(swa_lens_ptr + offs, 0, mask=offs < lens_len)
        if pid >= num_decode_tokens:
            return
        if not is_valid:
            tl.store(swa_lens_ptr + pid, 0)
            for i in range(0, index_width, TRITON_BLOCK_SIZE):
                offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
                tl.store(swa_indices_ptr + pid * swa_indices_stride + offset, -1, mask=offset < index_width)
            return
        req_idx = tl.load(token_to_req_indices_ptr + pid)
        query_start = tl.load(query_start_loc_ptr + req_idx)
        query_end = tl.load(query_start_loc_ptr + req_idx + 1)
        query_len = query_end - query_start
        seq_len = tl.load(seq_lens_ptr + req_idx)
        prefix_len = seq_len - query_len
        if NONCAUSAL:
            start_pos = tl.maximum(prefix_len - window_size, 0)
            end_pos = seq_len
            if NEST > 0:  # rows < NEST of the block see only its first NEST tokens
                end_pos = tl.where(pid - query_start < NEST, tl.minimum(seq_len, prefix_len + NEST), seq_len)
        else:
            pos = prefix_len + pid - query_start
            start_pos = tl.maximum(pos - (window_size - 1), 0)
            start_pos = tl.maximum(start_pos, tl.load(replay_start_ptr + req_idx))
            end_pos = pos + 1
        swa_len = end_pos - start_pos
        tl.store(swa_lens_ptr + pid, swa_len)
        for i in range(0, index_width, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            pos_offset = start_pos + offset
            block_indices = pos_offset // block_size
            block_numbers = tl.load(block_table_ptr + req_idx * block_table_stride + block_indices,
                                    mask=pos_offset < end_pos)
            block_offsets = pos_offset % block_size
            slot_ids = block_numbers * block_size + block_offsets
            slot_ids = tl.where(offset < swa_len, slot_ids, -1)
            tl.store(swa_indices_ptr + pid * swa_indices_stride + offset, slot_ids, mask=offset < index_width)

    _kernel = _swa_decode_meta_kernel
    return _kernel


def fused_decode_meta(slot_mapping, is_valid_token, swa_indices, swa_lens, window_size, index_width,
                      query_start_loc, seq_lens, token_to_req_indices, block_table, block_size, replay_start,
                      num_decode_tokens, noncausal: bool) -> None:
    """is_valid_token = slot_mapping >= 0; swa_lens[n:] = 0; decode window indices/lens for tokens [0, n)."""
    num_slots = slot_mapping.shape[0]
    grid = (max(num_slots, num_decode_tokens, 1),)
    _get_kernel()[grid](
        slot_mapping, num_slots, is_valid_token,
        swa_indices, swa_indices.stride(0), swa_lens, swa_lens.shape[0],
        window_size, index_width,
        query_start_loc, seq_lens, token_to_req_indices,
        block_table, block_table.stride(0), block_size, replay_start,
        num_decode_tokens,
        NONCAUSAL=noncausal, TRITON_BLOCK_SIZE=1024, FILL_BLOCK=1024, NEST=NEST if noncausal else 0,
    )


_NF = 16  # int64 fields per group in the device table
_pending: list | None = None  # decode builds recorded inside a wrapped build_attn_metadata call
_tables: dict = {}  # field tuple of a group set -> its device table
_multi_kernel = None


def _get_multi_kernel():
    global _multi_kernel
    if _multi_kernel is not None:
        return _multi_kernel
    from vllm.triton_utils import tl, triton

    @triton.jit(do_not_specialize=["num_slots", "num_decode_tokens"])
    def _swa_decode_meta_multi_kernel(table_ptr, num_slots, num_decode_tokens, NF: tl.constexpr,
                                      TRITON_BLOCK_SIZE: tl.constexpr, FILL_BLOCK: tl.constexpr, NEST: tl.constexpr):
        """_swa_decode_meta_kernel for group program_id(1), its pointers/constants read from row g of the table."""
        pid = tl.program_id(0)
        t = table_ptr + tl.program_id(1) * NF
        slot_mapping_ptr = tl.load(t + 0).to(tl.pointer_type(tl.int64))
        is_valid_token_ptr = tl.load(t + 1).to(tl.pointer_type(tl.int8))
        swa_indices_ptr = tl.load(t + 2).to(tl.pointer_type(tl.int32))
        swa_indices_stride = tl.load(t + 3)
        swa_lens_ptr = tl.load(t + 4).to(tl.pointer_type(tl.int32))
        lens_len = tl.load(t + 5).to(tl.int32)
        window_size = tl.load(t + 6).to(tl.int32)
        index_width = tl.load(t + 7).to(tl.int32)
        query_start_loc_ptr = tl.load(t + 8).to(tl.pointer_type(tl.int32))
        seq_lens_ptr = tl.load(t + 9).to(tl.pointer_type(tl.int32))
        token_to_req_indices_ptr = tl.load(t + 10).to(tl.pointer_type(tl.int32))
        block_table_ptr = tl.load(t + 11).to(tl.pointer_type(tl.int32))
        block_table_stride = tl.load(t + 12)
        block_size = tl.load(t + 13).to(tl.int32)
        replay_start_ptr = tl.load(t + 14).to(tl.pointer_type(tl.int32))
        noncausal = tl.load(t + 15) != 0
        if pid < num_slots:
            is_valid = tl.load(slot_mapping_ptr + pid) >= 0
            tl.store(is_valid_token_ptr + pid, is_valid.to(tl.int8))
        else:
            is_valid = False
        if pid == tl.num_programs(0) - 1:
            for i in range(num_decode_tokens, lens_len, FILL_BLOCK):
                offs = i + tl.arange(0, FILL_BLOCK)
                tl.store(swa_lens_ptr + offs, 0, mask=offs < lens_len)
        if pid >= num_decode_tokens:
            return
        if not is_valid:
            tl.store(swa_lens_ptr + pid, 0)
            for i in range(0, index_width, TRITON_BLOCK_SIZE):
                offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
                tl.store(swa_indices_ptr + pid * swa_indices_stride + offset, -1, mask=offset < index_width)
            return
        req_idx = tl.load(token_to_req_indices_ptr + pid)
        query_start = tl.load(query_start_loc_ptr + req_idx)
        query_end = tl.load(query_start_loc_ptr + req_idx + 1)
        query_len = query_end - query_start
        seq_len = tl.load(seq_lens_ptr + req_idx)
        prefix_len = seq_len - query_len
        if noncausal:
            start_pos = tl.maximum(prefix_len - window_size, 0)
            end_pos = seq_len
            if NEST > 0:  # rows < NEST of the block see only its first NEST tokens
                end_pos = tl.where(pid - query_start < NEST, tl.minimum(seq_len, prefix_len + NEST), seq_len)
        else:
            pos = prefix_len + pid - query_start
            start_pos = tl.maximum(pos - (window_size - 1), 0)
            start_pos = tl.maximum(start_pos, tl.load(replay_start_ptr + req_idx))
            end_pos = pos + 1
        swa_len = end_pos - start_pos
        tl.store(swa_lens_ptr + pid, swa_len)
        for i in range(0, index_width, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            pos_offset = start_pos + offset
            block_indices = pos_offset // block_size
            block_numbers = tl.load(block_table_ptr + req_idx * block_table_stride + block_indices,
                                    mask=pos_offset < end_pos)
            block_offsets = pos_offset % block_size
            slot_ids = block_numbers * block_size + block_offsets
            slot_ids = tl.where(offset < swa_len, slot_ids, -1)
            tl.store(swa_indices_ptr + pid * swa_indices_stride + offset, slot_ids, mask=offset < index_width)

    _multi_kernel = _swa_decode_meta_multi_kernel
    return _multi_kernel


def _fields(slot_mapping, is_valid_token, swa_indices, swa_lens, window_size, index_width, query_start_loc,
            seq_lens, token_to_req_indices, block_table, block_size, replay_start, num_decode_tokens, noncausal):
    """One group's table row (pointers + constants); dtypes asserted to match the kernel's pointer casts."""
    import torch

    for t, dt in ((slot_mapping, torch.int64), (is_valid_token, torch.bool), (swa_indices, torch.int32),
                  (swa_lens, torch.int32), (query_start_loc, torch.int32), (seq_lens, torch.int32),
                  (token_to_req_indices, torch.int32), (block_table, torch.int32), (replay_start, torch.int32)):
        assert t.dtype == dt, (t.dtype, dt)
    assert block_table.stride(1) == 1 and swa_indices.stride(-1) == 1
    return (slot_mapping.data_ptr(), is_valid_token.data_ptr(), swa_indices.data_ptr(), swa_indices.stride(0),
            swa_lens.data_ptr(), swa_lens.shape[0], int(window_size), int(index_width), query_start_loc.data_ptr(),
            seq_lens.data_ptr(), token_to_req_indices.data_ptr(), block_table.data_ptr(), block_table.stride(0),
            int(block_size), replay_start.data_ptr(), int(bool(noncausal)))


def _launch_multi(entries) -> None:
    import torch

    key = tuple(_fields(*a) for a in entries)
    table = _tables.get(key)
    if table is None:
        if len(_tables) > 64:
            _tables.clear()
        dev = entries[0][0].device
        table = torch.tensor([v for row in key for v in row], dtype=torch.int64, device=dev)
        _tables[key] = table
    num_slots = entries[0][0].shape[0]
    num_decode_tokens = entries[0][12]
    grid = (max(num_slots, num_decode_tokens, 1), len(entries))
    _get_multi_kernel()[grid](table, num_slots, num_decode_tokens, NF=_NF, TRITON_BLOCK_SIZE=1024,
                              FILL_BLOCK=1024, NEST=NEST)


def flush() -> None:
    """Build every recorded group: one multi-group launch per run of equal (num_slots, num_decode_tokens)."""
    global _pending
    entries, _pending = _pending, None
    if not entries:
        return
    batch = []
    for a in entries + [None]:
        if batch and (a is None or (a[0].shape[0], a[12]) != (batch[0][0].shape[0], batch[0][12])):
            if len(batch) == 1:
                fused_decode_meta(*batch[0])
            else:
                _launch_multi(batch)
            batch = []
        if a is not None:
            batch.append(a)


def wrap_build_attn_metadata(orig):
    """build_attn_metadata that defers its SWA decode builds and launches them together before returning."""
    import functools

    @functools.wraps(orig)
    def build_attn_metadata(*args, **kwargs):
        global _pending
        if _pending is not None:  # nested (never seen): stay immediate
            return orig(*args, **kwargs)
        _pending = []
        try:
            return orig(*args, **kwargs)
        finally:
            flush()

    build_attn_metadata._w5_swa_multi = True
    return build_attn_metadata


def _log_nest_fallback(num_prefill_tokens: int, num_decode_tokens: int) -> None:
    global _nest_fallback_logged
    if not _nest_fallback_logged:
        _nest_fallback_logged = True
        sys.stderr.write(f"DS41_FASTPATH tail_swa_meta: a non-causal (DSpark) build took the stock path (prefill "
                         f"tokens {num_prefill_tokens}, decode tokens {num_decode_tokens}): not nested\n")


def make_build(module, orig_build):
    """Decode-only fast path of DeepseekSparseSWAMetadataBuilder.build (stock build otherwise)."""
    import torch

    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    Metadata = module.DeepseekSparseSWAMetadata
    layer_keys = (module._LAYER_TYPE_SWAONLY, module._LAYER_TYPE_C4A, module._LAYER_TYPE_C128A,
                  module._LAYER_TYPE_C1A, module._LAYER_TYPE_C2A)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, replay_start=None):
        cam = common_attn_metadata
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            cam, decode_threshold=self.decode_threshold)
        if num_prefill_tokens > 0 or num_decode_tokens == 0:
            if NEST and not cam.causal:
                _log_nest_fallback(num_prefill_tokens, num_decode_tokens)
            return orig_build(self, common_prefix_len, cam, fast_build, replay_start)
        seq_lens = cam.seq_lens
        query_start_loc = cam.query_start_loc
        block_table = cam.block_table_tensor
        slot_mapping = cam.slot_mapping
        token_to_req_indices = cam.token_to_req_indices(self.token_to_req_indices)
        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        if replay_start is None:
            replay_start = self.no_replay_start
        non_causal = not cam.causal
        decode_swa_width = self.noncausal_index_width if non_causal else self.window_size
        decode_swa_indices = self.decode_swa_indices
        if non_causal:
            assert self.is_dspark, ("Non-causal DeepseekV4 SWA is only supported for the DSpark speculation mode, "
                                    "but causal=False was set without DSpark.")
            if self.decode_swa_indices_noncausal is None:
                self.decode_swa_indices_noncausal = torch.zeros(self._max_tokens, 1, self.noncausal_index_width,
                                                                dtype=torch.int32, device=self.device)
            decode_swa_indices = self.decode_swa_indices_noncausal
        meta_args = (slot_mapping, is_valid_token, decode_swa_indices, self.decode_swa_lens, self.window_size,
                     decode_swa_width if non_causal else decode_swa_indices.shape[-1], query_start_loc,
                     seq_lens, token_to_req_indices, block_table, self.block_size, replay_start,
                     num_decode_tokens, non_causal)
        if _pending is not None:
            _pending.append(meta_args)
        else:
            fused_decode_meta(*meta_args)
        deepseek_v4_fields = self._build_deepseek_v4_metadata(num_decodes, num_prefills, seq_lens,
                                                              cam.seq_lens_cpu_upper_bound, query_start_loc,
                                                              cam.query_start_loc_cpu, replay_start)
        tile_sched = self.build_tile_scheduler(num_decode_tokens)
        return Metadata(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_valid_token=is_valid_token,
            token_to_req_indices=token_to_req_indices,
            decode_swa_indices=decode_swa_indices[:num_decode_tokens],
            decode_swa_lens=self.decode_swa_lens[:num_decode_tokens],
            decode_swa_width=decode_swa_width,
            prefill_swa_indices=None,
            prefill_swa_lens=None,
            prefill_left_visible=None,
            prefill_right_visible=None,
            replay_start=replay_start,
            block_size=self.block_size,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            max_decode_query_len=min(cam.max_query_len, self.decode_threshold),
            tile_sched_swaonly=tile_sched[layer_keys[0]],
            tile_sched_c4a=tile_sched[layer_keys[1]],
            tile_sched_c128a=tile_sched[layer_keys[2]],
            tile_sched_c1a=tile_sched[layer_keys[3]],
            tile_sched_c2a=tile_sched[layer_keys[4]],
            **deepseek_v4_fields,
        )

    return build


@register("tail_swa_meta", SWA_MODULE)
def patch(module) -> None:
    cls = module.DeepseekSparseSWAMetadataBuilder
    if getattr(cls, "_w5_swa_meta", False):
        return
    cls.build = make_build(module, cls.build)
    cls._w5_swa_meta = True
    sys.stderr.write("DS41_FASTPATH tail_swa_meta: decode-only SWA metadata builds use one fused kernel"
                     + (f"; DSpark block nested at {NEST} rows" if NEST else "") + "\n")


def _wrap_module_build_attn_metadata(module, who: str) -> None:
    fn = getattr(module, "build_attn_metadata", None)
    if fn is None or getattr(fn, "_w5_swa_multi", False):
        return
    module.build_attn_metadata = wrap_build_attn_metadata(fn)
    sys.stderr.write(f"DS41_FASTPATH tail_swa_meta: {who} SWA decode builds launch as one multi-group kernel\n")


@register("tail_swa_meta", "vllm.v1.worker.gpu.model_states.default")
def patch_target_builds(module) -> None:
    _wrap_module_build_attn_metadata(module, "target")


@register("tail_swa_meta", "vllm.v1.worker.gpu.spec_decode.speculator")
def patch_draft_builds(module) -> None:
    _wrap_module_build_attn_metadata(module, "drafter")
