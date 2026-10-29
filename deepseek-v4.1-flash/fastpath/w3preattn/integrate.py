"""W3 preattn fastpath for vLLM af7f9488 (DeepSeek-V4.1, FlashMLA mega attention, TP1, nvfp4_ds_mla).

Every patch is bitwise identical to the unpatched model (tests/test_hook.py, tests/test_preattn.py, tests/test_cand.py)
and falls through to the original method whenever its conditions do not hold. Features (patch_* below):

  qkv    DeepseekV4Attention.forward + ._fused_qnorm_rope_kv_insert
         _split_qkv_and_norm (Triton _q_kv_norm_quant_kernel) + the KV half of _fused_qnorm_rope_kv_insert (which ran
         after wq_b) -> one W3 PDL launch between fused_wqa_wkv and wq_b: q/kv RMSNorm, MXFP8 q, RoPE + MXFP8 SWA insert.
  memo   DeepseekV4MegaAttnAttention._decode_compressed_kv_and_topk + .forward_mqa
         positions.to(int32) and compute_global_topk_indices_and_lens once per step per (index source, kv source,
         compress ratio) instead of once per layer (vLLM #61868 item 11).
  cand   sparse_attn_indexer._select_candidate_blocks (candidate source, layer 20)
         -> W3 exact candidate selection over each row's valid prefix (same set as torch.topk).
  index  DeepseekV4Attention._prepare_and_attn (candidate-consumer index layers 24/28/32/36, pure decode)
         -> the indexer branch on aux stream 2 concurrent with wq_b, fill dropped, remap + key table fused (#14).
  cstreams  DeepseekV4Attention._prepare_and_attn (compressor + indexer layers 2/8/14/20, pure decode)
         -> indexer queries, compressor + K store, main-cache insert and wq_b on four streams (#14).
  compress  compressor.fused_save_compress_norm (Triton _fused_save_compress_norm_kernel), ratio-2 layers
         -> W3 compress stage: same math op for op, ring tail rows saved in parallel, PDL.

The CUDA extension is the prebuilt w3_ext*.so next to this file; a missing extension raises. Build it with
`kernels/w3-preattn/build.sh`.
"""
from __future__ import annotations

import glob
import os
import sys
import weakref

import torch

_ORIG: dict = {}
_EXT = None
_SEEN: set = set()


def _ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    here = os.path.dirname(os.path.abspath(__file__))
    prebuilt = next(iter(sorted(glob.glob(os.path.join(here, "w3_ext*.so")))), None)
    if prebuilt is None:
        raise RuntimeError("w3_ext*.so not found next to w3preattn/integrate.py")
    import importlib.util

    spec = importlib.util.spec_from_file_location("w3_ext", prebuilt)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _EXT = mod
    return _EXT


def _log_once(what: str, msg: str) -> None:
    if what not in _SEEN:
        _SEEN.add(what)
        sys.stderr.write(f"W3 preattn: {msg}\n")


# ------------------------------------------------------------------ step memo (#11)
class _StepMemo:
    """Per-forward cache, valid while both the forward context and its attn_metadata are the same objects (vLLM
    creates both per forward; the decoder-replay sub-batch gets its own). References are weak where possible, so a
    new object can never alias a dead one's id."""

    def __init__(self):
        self._refs: tuple = ()
        self.data: dict = {}

    def get(self, attn_metadata) -> dict:
        from vllm.forward_context import get_forward_context

        objs = (get_forward_context(), attn_metadata)
        if len(self._refs) != 2 or any(r() is not o for r, o in zip(self._refs, objs)):
            self._refs = tuple(_ref(o) for o in objs)
            self.data = {}
        return self.data


def _ref(o):
    try:
        return weakref.ref(o)
    except TypeError:
        return lambda o=o: o


_MEMO = _StepMemo()


# ------------------------------------------------------------------ eligibility
def _eligible(self, qr_kv: torch.Tensor) -> bool:
    return (
        getattr(self, "accepts_unnormed_unroped_query", False)
        and self.n_local_heads == self.padded_heads
        and self.kv_mxfp8
        and self.q_lora_rank == 1280
        and self.head_dim == 512
        and qr_kv.dtype == torch.bfloat16
        and qr_kv.stride(-1) == 1
        and self.swa_cache_layer.kv_cache.dtype == torch.uint8
        and self._can_fuse_query_quant
        and self._prepare_and_attn_fn == self._prepare_and_attn  # model runner V2 (no wide eager region)
    )


def _w3_qkv(self, qr_kv: torch.Tensor, positions: torch.Tensor, attn_metadata):
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

    swa_md = attn_metadata[self.swa_cache_layer.prefix]
    T = qr_kv.shape[0]
    dev = qr_kv.device
    qx8 = torch.empty((T, 1280), dtype=torch.float8_e4m3fn, device=dev)
    sf = torch.empty(((T + 127) // 128) * 128 * 40, dtype=torch.uint8, device=dev)
    kv = torch.empty((T, 512), dtype=torch.bfloat16, device=dev)
    cache = self.swa_cache_layer.kv_cache
    _ext().qkv_kt(qr_kv, self.q_norm.weight.data, self.kv_norm.weight.data, self.eps, qx8, sf, kv,
                  cache.view(cache.shape[0], -1), swa_md.slot_mapping, positions, self.rotary_emb.cos_sin_cache,
                  swa_md.block_size, None, None, None, 1, None, None, None, None, True)
    qa = QuantizedActivation(qx8, sf, qr_kv.dtype, torch.Size((T, 1280)), kMxfp8Dynamic)
    return qa, kv


def _forward(self, positions, hidden_states, llama_4_scaling=None):
    from vllm.forward_context import get_forward_context

    attn_out = self._alloc_attn_out(hidden_states.shape[0], hidden_states)
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict) and self.indexer is not None and self.compressor is None:
        _fork_consumer_schedule(self, attn_metadata)
    qr_kv, kv_score, indexer_weights = self._run_parallel_input_projections(hidden_states)
    if isinstance(attn_metadata, dict) and _eligible(self, qr_kv):
        qr, kv = _w3_qkv(self, qr_kv, positions, attn_metadata)
        qr_scale = None
        _log_once("qkv", f"qkv fused path active ({type(self).__name__}, {qr_kv.shape[0]} tokens)")
        self._w3_kv_inserted = True
    else:
        qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
        self._w3_kv_inserted = False
    try:
        self._prepare_and_attn_fn(hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, attn_out)
    finally:
        self._w3_kv_inserted = False
    return self._o_proj(attn_out, positions)


def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
    if getattr(self, "_w3_kv_inserted", False):
        return q  # mega attention reads q as produced by wq_b; the W3 launch inserted kv
    return _ORIG["insert"](self, q, kv, positions, attn_metadata)


# ------------------------------------------------------------------ memoized key table + int32 positions
def _decode_compressed_kv_and_topk(self, flashmla_metadata, swa_metadata):
    if self.compress_ratio == 0:
        return _ORIG["dkt"](self, flashmla_metadata, swa_metadata)
    from vllm.forward_context import get_forward_context

    memo = _MEMO.get(get_forward_context().attn_metadata)
    key = ("kt", self.index_source_layer_id, self.kv_source_layer_id, self.compress_ratio,
           swa_metadata.num_decode_tokens)
    hit = memo.get(key)
    if hit is None:
        hit = _ORIG["dkt"](self, flashmla_metadata, swa_metadata)
        memo[key] = hit
    return hit


def _memo_pos32(positions: torch.Tensor, attn_metadata) -> torch.Tensor:
    memo = _MEMO.get(attn_metadata)
    key = ("pos32", positions.data_ptr(), positions.shape[0])
    p32 = memo.get(key)
    if p32 is None:
        p32 = positions.to(torch.int32)
        memo[key] = p32
    return p32


def _forward_mqa(self, q, kv, positions, output):
    """DeepseekV4MegaAttnAttention.forward_mqa (af7f9488) with the int32 positions memoized per step."""
    from vllm.forward_context import get_forward_context
    from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import _token_slice

    if not self._fused_layouts_ready:
        raise RuntimeError(f"{self.prefix}: wq_b / wo_a were never permuted for the mega attention kernel.")
    attn_metadata = get_forward_context().attn_metadata
    if attn_metadata is None:
        self._reserve_prefill_workspace(q)
        output.data.zero_()
        output.scale.zero_()
        return
    assert isinstance(attn_metadata, dict)
    assert q.shape[1] == self.padded_heads
    flashmla_metadata = (attn_metadata.get(self.compressed_cache_prefix)
                         if self.compressed_cache_prefix is not None else None)
    swa_metadata = attn_metadata[self.swa_cache_layer.prefix]
    positions_int32 = _memo_pos32(positions, attn_metadata)
    num_decode_tokens = swa_metadata.num_decode_tokens
    if swa_metadata.num_prefills > 0:
        self._forward_prefill_mega(q[num_decode_tokens:], positions_int32[num_decode_tokens:], flashmla_metadata,
                                   swa_metadata, output, num_decode_tokens)
    if swa_metadata.num_decodes > 0:
        self._forward_decode_mega(q[:num_decode_tokens], positions_int32[:num_decode_tokens], flashmla_metadata,
                                  swa_metadata, _token_slice(output, 0, num_decode_tokens))


# ------------------------------------------------------------------ multi-stream guard
_ALL_SIDE_STREAMS = frozenset({"index", "cstreams", "schedule"})




_SIDE_STREAMS = _ALL_SIDE_STREAMS


def _multistream_ok(feature: str) -> bool:
    """Whether this forward may fork `feature`'s work onto side streams and join it later.

    Only inside one layer's pre-attention region: every W3 side stream forks after the layer's start (so after the
    previous layer's MegaMoE) and joins before its sparse-indexer + attention region, so no side-stream kernel ever
    runs beside attention, MegaMoE or any other kernel that needs the whole GPU.
    Yes in FULL CUDA graphs (one capture holds fork and join) and in eager forwards; never in PIECEWISE graphs:
    breakable capture runs the attention / indexer region as eager segments between captured ones, so an event
    recorded in one segment cannot be waited on in another (CUDA: "dependency created on uncaptured work in another
    stream"), and torch.compile piecewise graphs split at the same ops."""
    if feature not in _SIDE_STREAMS:
        return False
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return False
    mode = get_forward_context().cudagraph_runtime_mode
    if mode == CUDAGraphMode.FULL:
        return True
    if mode == CUDAGraphMode.PIECEWISE:
        return False
    return not BreakableCUDAGraphCapture.is_active()


# ------------------------------------------------------------------ candidate-consumer index layers (24/28/32/36)
def _is_candidate_consumer(self) -> bool:
    from vllm.model_executor.layers.sparse_mqa_indexer import SparseMQAIndexer

    return (self.indexer is not None and self.compressor is None
            and isinstance(self.indexer.indexer_op, SparseMQAIndexer) and self.aux_stream_list is not None
            and self.compress_ratio > 0)


def _index_layer_eligible(self) -> bool:
    return _multistream_ok("index") and _is_candidate_consumer(self)


def _sparse_mqa_decode(self, q_quant, weights, attn_metadata) -> bool:
    """SparseMQAIndexer.forward for pure-decode steps with the fill dropped and remap + key table fused (one W3
    launch). Fills the step memo with this layer's key table. Returns False when the step is not pure decode."""
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import candidate_blocks_to_sparse_indices
    from vllm.model_executor.layers.indexer_topk import deep_select_topk
    from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
    from vllm.utils.deep_gemm import fp8_fp4_paged_sparse_mqa_logits, get_paged_sparse_mqa_logits_metadata

    op = self.indexer.indexer_op
    md = attn_metadata[op.k_cache.prefix]
    swa = attn_metadata[self.swa_cache_layer.prefix]
    fl = attn_metadata.get(self.compressed_cache_prefix)
    if md.num_prefills > 0 or md.num_decodes == 0 or swa.num_prefills > 0 or fl is None:
        return False
    rows_md = md.sparse_decode
    num_rows = rows_md.row_ke.shape[0]
    T = swa.num_decode_tokens
    q_values, q_scale = q_quant
    if num_rows != T or q_values.shape[0] != num_rows:  # padded rows: keep production's fill of the whole buffer
        return False
    kv_cache = op.k_cache.kv_cache
    kv_view = kv_cache_as_quant_view(kv_cache, op.head_dim, use_fp4_cache=True)
    sbk = md.sparse_block_kv
    if rows_md.kernel_metadata is None:
        candidate_blocks_to_sparse_indices(op.candidate_blocks[:num_rows], rows_md.row_ks, rows_md.row_ke,
                                           op.candidate_block_size, sbk, out=(rows_md.sparse_indices, rows_md.end))
        rows_md.kernel_metadata = get_paged_sparse_mqa_logits_metadata(
            rows_md.row_ke, rows_md.block_table, rows_md.row_indices, kv_view.shape[1], rows_md.sparse_indices,
            q_values.view(torch.int8).dtype, sbk)
    logits = fp8_fp4_paged_sparse_mqa_logits(
        (q_values[:num_rows].view(torch.int8).unsqueeze(1), q_scale[:num_rows].unsqueeze(1)), kv_view,
        weights[:num_rows], rows_md.kernel_metadata, rows_md.sparse_indices.shape[1], sbk)
    cols = rows_md.col_indices[:num_rows]
    deep_select_topk(logits, op.topk_tokens, end=rows_md.end, output_idx=cols)
    gidx = torch.empty((T, op.topk_tokens), dtype=torch.int32, device=logits.device)
    glens = torch.empty((T,), dtype=torch.int32, device=logits.device)
    # The remap ends the indexer branch's side stream. In a captured graph the event join before attention becomes a
    # programmatic edge for PDL-launched attention kernels (tests/probe_join.py), so the remap only lets dependents
    # launch at its exit: by then it has waited for DeepSelect, which waited for the sparse logits.
    _ext().remap_global(cols, rows_md.sparse_indices, rows_md.row_ks, logits.shape[1], sbk, op.topk_indices_buffer,
                        swa.token_to_req_indices, fl.block_table[: swa.num_decodes], fl.block_size // self.compress_ratio,
                        swa.is_valid_token[:T], gidx, glens, _positions_of(attn_metadata), None, True, True)
    memo = _MEMO.get(attn_metadata)
    # same value _decode_compressed_kv_and_topk returns: (compressed cache, global slots, lens)
    memo[("kt", self.index_source_layer_id, self.kv_source_layer_id, self.compress_ratio, T)] = (
        self._compressed_kv_cache().unsqueeze(-2), gidx, glens)
    return True


def _positions_of(attn_metadata):
    return _MEMO.get(attn_metadata)["positions"]


def _prepare_and_attn(self, hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, attn_out):
    """DeepseekV4Attention._prepare_and_attn with the installed multi-stream schedules (#14) for pure-decode steps:
    "index" for the candidate-consumer index layers, "compress" for the compressor + indexer layers; everything else
    (and every step with prefill rows) runs the stock method."""
    from vllm.forward_context import get_forward_context

    args = (hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, attn_out)
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict) and attn_metadata[self.swa_cache_layer.prefix].num_prefills == 0:
        if "index" in _ARMED and _index_layer_eligible(self):
            return _prepare_and_attn_index(self, attn_metadata, *args)
        if "cstreams" in _ARMED and _compress_layer_eligible(self, attn_metadata):
            return _prepare_and_attn_compress(self, attn_metadata, *args)
    if isinstance(attn_metadata, dict) and self.indexer is not None and self.compressor is None:
        ready = _MEMO.get(attn_metadata).get("schedule_ready")
        if ready is not None:
            ready.wait()  # a candidate consumer on the stock path reads the forked schedule
    return _ORIG["prepare"](self, *args)


def _prepare_and_attn_index(self, attn_metadata, hidden_states, qr, kv, qr_scale, kv_score, indexer_weights,
                            positions, attn_out):
    """Candidate-consumer index layers. The indexer query path (indexer wq_b -> q RoPE/quant) runs on aux stream 2
    beside wq_b; the rest of the branch (sparse logits -> DeepSelect -> W3 remap + key table) follows once wq_b is
    done, because DeepGEMM's sparse-logits kernel asserts it owns the SM's tensor memory (tmem column 0) and must
    never share an SM with a tensor-memory kernel of another stream. Joined before the attention region."""
    _MEMO.get(attn_metadata)["positions"] = positions
    _log_once("index", f"candidate-consumer indexer branch on a side stream ({type(self).__name__})")
    indexer = self.indexer
    side = self.aux_stream_list[2]
    start, wq_b_done, branch_done = _events(self, 5)[:3]
    ready = _MEMO.get(attn_metadata).get("schedule_ready")
    start.record()
    with torch.cuda.stream(side):
        start.wait()
        index_q, index_q_scale, index_w = indexer(qr, None, indexer_weights, positions, self.indexer_rotary_emb,
                                                  qr_scale)
    q = self._wq_b_proj(qr, qr_scale).view(-1, self.n_local_heads, self.head_dim)
    q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
    wq_b_done.record()
    with torch.cuda.stream(side):
        wq_b_done.wait()
        if ready is not None:
            ready.wait()  # the expansion + schedule forked at this layer's start
        q_quant = (index_q, index_q_scale) if index_q_scale is not None else index_q
        handled = index_q_scale is not None and _sparse_mqa_decode(self, q_quant, index_w, attn_metadata)
        if index_q is not None and not handled:
            indexer.indexer_op(hidden_states, q_quant, None, index_w)
        branch_done.record()
    branch_done.wait()
    self.forward_mqa(q, kv, positions, attn_out)


def _compress_layer_eligible(self, attn_metadata) -> bool:
    indexer = self.indexer
    if (not _multistream_ok("cstreams") or self.compressor is None or indexer is None
            or self.aux_stream_list is None or len(self.aux_stream_list) < 3):
        return False
    # indexer.forward's short-context branch (fill + no queries) only runs outside graph capture: keep it stock
    md = attn_metadata[indexer.k_cache.prefix]
    return not (md.max_seq_len // indexer.compress_ratio <= indexer.topk_tokens
                and not torch.cuda.is_current_stream_capturing())


def _events(self, n: int) -> list:
    ev = getattr(self, "_w3_events", None)
    if ev is None:
        ev = [torch.cuda.Event() for _ in range(n)]
        self._w3_events = ev
    return ev


def _prepare_and_attn_compress(self, attn_metadata, hidden_states, qr, kv, qr_scale, kv_score, indexer_weights,
                               positions, attn_out):
    """Compressor + indexer layers (kv / index sources 2, 8, 14 and the candidate source 20). Stock runs
    [wq_b + insert || compressor], then [indexer: wk + K store, then indexer wq_b + q RoPE/quant || main-cache insert];
    the indexer queries only need qr, so here every independent branch starts as soon as its inputs exist:
      main : wq_b -> q / SWA insert
      aux1 : indexer wq_b -> q RoPE/quant                      (needs qr)
      aux0 : compressor (state save + pool + norm) -> wk -> K store   (needs kv_score)
      aux2 : compressed main-cache RoPE/quant insert            (needs the compressor latent)
    joined before the eager sparse-indexer + attention region. Same kernels, same inputs: bitwise."""
    indexer, compressor = self.indexer, self.compressor
    aux0, aux1, aux2 = self.aux_stream_list[:3]
    start, latent_ready, k_done, q_done, ins_done = _events(self, 5)
    start.record()
    with torch.cuda.stream(aux1):
        start.wait()
        index_q, index_q_scale, index_w = indexer.forward_q(qr, qr_scale, indexer_weights, positions,
                                                            self.indexer_rotary_emb)
        q_done.record()
    with torch.cuda.stream(aux0):
        start.wait()
        latent = compressor(kv_score, positions)
        latent_ready.record()
        if indexer.owns_k:
            indexer._produce_k(latent, positions, self.indexer_rotary_emb)
        k_done.record()
    with torch.cuda.stream(aux2):
        latent_ready.wait()
        compressor.insert_cache(latent, positions, self.rotary_emb)
        ins_done.record()
    q = self._wq_b_proj(qr, qr_scale).view(-1, self.n_local_heads, self.head_dim)
    q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
    k_done.wait()
    q_done.wait()
    ins_done.wait()
    _log_once("compress", f"compressor-layer streams active ({type(self).__name__})")
    if index_q is not None and _noclear_ok(self, attn_metadata, hidden_states.shape[0]):
        _log_once("noclear", "compressor-layer indexer: top-k buffer pre-fill skipped (the decode top-k writes every row)")
        fused = _topkg_prepare(self, attn_metadata, hidden_states.shape[0])
        if fused is None:
            _indexer_noclear(self.indexer.indexer_op, hidden_states, index_q, index_q_scale, index_w)
        else:
            import vllm.model_executor.layers.sparse_attn_indexer as sai

            orig = sai.get_indexer_topk
            sai.get_indexer_topk = lambda backend, _f=fused: _f
            try:
                _indexer_noclear(self.indexer.indexer_op, hidden_states, index_q, index_q_scale, index_w)
            finally:
                sai.get_indexer_topk = orig
            fused.publish()
        self.forward_mqa(q, kv, positions, attn_out)
        return
    self._sparse_indexer_and_attn(hidden_states, index_q, index_q_scale, index_w, q, kv, positions, attn_out)


_TG = {}


def _topkg_lib():
    """libw8topkg.so (W8: vLLM's cooperative_topk body fused with the key-table translation); None if unavailable."""
    if "lib" not in _TG:
        import ctypes

        P, L = ctypes.c_void_p, ctypes.c_longlong

        class Args(ctypes.Structure):
            _fields_ = [("logits", P), ("stride", L), ("lengths", P), ("topk", P), ("workspace", P), ("gidx", P),
                        ("glens", P), ("token_to_req", P), ("block_table", P), ("bt_stride", L), ("block_size", L),
                        ("is_valid", P), ("rows", L)]

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libw8topkg.so")
        lib = None
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            lib.w8tg_abi.restype = L
            lib.w8tg_abi.argtypes = [ctypes.c_int]
            lib.w8tg_launch.restype = ctypes.c_int
            lib.w8tg_launch.argtypes = [ctypes.POINTER(Args), P]
            if lib.w8tg_abi(0) != ctypes.sizeof(Args) or lib.w8tg_abi(1) != Args.rows.offset:
                raise RuntimeError("libw8topkg.so ABI mismatch")
        _TG["lib"], _TG["Args"] = lib, Args
    return _TG["lib"]


class _FusedTopkGlobal:
    """Stands in for SparseIndexerTopk on one compressor-layer indexer call: the cooperative top-k into
    topk_indices_buffer plus the key table, in one launch; publish() puts the key table into the step memo."""

    def __init__(self, layer, attn_metadata, flashmla_md, swa_md):
        self.layer, self.attn_metadata, self.fmd, self.smd = layer, attn_metadata, flashmla_md, swa_md
        self.out = None

    def __call__(self, logits, seq_lens, next_n, topk_indices, topk_tokens, max_seq_len):
        import ctypes

        from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE
        from vllm.v1.worker.workspace import current_workspace_manager

        rows = logits.shape[0]
        n = self.smd.num_decode_tokens
        assert rows == n and topk_tokens == 512 and topk_indices.shape[0] == rows and topk_indices.stride(1) == 1
        assert seq_lens.is_contiguous() and seq_lens.numel() == rows and seq_lens.dtype == torch.int32
        (ws,) = current_workspace_manager().get_simultaneous(((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8))
        bt = self.fmd.block_table[: self.smd.num_decodes]
        gidx = torch.empty((rows, 512), dtype=torch.int32, device=logits.device)
        glens = torch.empty((rows,), dtype=torch.int32, device=logits.device)
        a = _TG["Args"](logits.data_ptr(), logits.stride(0), seq_lens.data_ptr(), topk_indices.data_ptr(),
                        ws.data_ptr(), gidx.data_ptr(), glens.data_ptr(), self.smd.token_to_req_indices.data_ptr(),
                        bt.data_ptr(), bt.stride(0), self.fmd.block_size // self.layer.compress_ratio,
                        self.smd.is_valid_token.data_ptr(), rows)
        r = _topkg_lib().w8tg_launch(ctypes.byref(a), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if r != 0:
            raise RuntimeError(f"w8tg_launch failed: {r}")
        self.out = (gidx, glens)

    def publish(self) -> None:
        if self.out is None:
            return
        key = ("kt", self.layer.index_source_layer_id, self.layer.kv_source_layer_id, self.layer.compress_ratio,
               self.smd.num_decode_tokens)
        _MEMO.get(self.attn_metadata)[key] = (self.layer._compressed_kv_cache().unsqueeze(-2), *self.out)


def _topkg_prepare(self, attn_metadata, num_tokens: int):
    """The fused top-k + key table for this compressor-layer call, or None (stock top-k): the library present, the
    memo installed (it is what hands the key table to attention), the stock key-table inputs in their expected forms."""
    if "topkg" in _DISABLED or "dkt" not in _ORIG or _topkg_lib() is None:
        return None
    fmd = attn_metadata.get(getattr(self, "compressed_cache_prefix", None) or "")
    smd = attn_metadata.get(self.swa_cache_layer.prefix)
    if fmd is None or smd is None or getattr(smd, "is_valid_token", None) is None or self.compress_ratio == 0:
        return None
    if smd.num_decode_tokens != num_tokens or smd.num_prefills != 0:
        return None
    bt, ttr, iv = fmd.block_table, smd.token_to_req_indices, smd.is_valid_token
    if bt.dtype != torch.int32 or bt.stride(1) != 1 or ttr.dtype != torch.int32 or iv.dtype != torch.bool:
        return None
    if fmd.block_size % self.compress_ratio != 0:
        return None
    _log_once("topkg", "compressor-layer indexer: top-k + key table in one launch (libw8topkg)")
    return _FusedTopkGlobal(self, attn_metadata, fmd, smd)


def _noclear_ok(self, attn_metadata, num_tokens: int) -> bool:
    """The decode top-k alone writes topk_indices_buffer[:num_tokens, :topk] (so the stock -1 pre-fill is dead): a
    pure-decode, unpadded, uniform batch, the full buffer width, the "auto" backend within cooperative_topk's row
    limit (cooperative, else persistent: both write every entry, -1 for short rows), no PCP / DCP."""
    if "noclear" in _DISABLED:
        return False
    op = getattr(self.indexer, "indexer_op", None)
    if op is None or type(op).__name__ != "SparseAttnIndexer":
        return False
    md = attn_metadata.get(op.k_cache.prefix) if isinstance(attn_metadata, dict) else None
    dec = getattr(md, "decode", None)
    if md is None or dec is None or getattr(md, "num_prefills", 1) != 0 or getattr(md, "num_prefill_tokens", 1) != 0:
        return False
    if getattr(dec, "requires_padding", True) or dec.decode_lens.shape[0] == 0:
        return False
    if num_tokens % dec.decode_lens.shape[0] != 0 or getattr(md, "num_decode_tokens", -1) != num_tokens:
        return False
    if op.topk_tokens != op.topk_indices_buffer.shape[1] or op.topk_backend != "auto" or num_tokens > 64:
        return False
    return op.dcp_world_size == 1 and not op.use_pcp


def _indexer_noclear(op, hidden_states, q_values, q_scale, weights):
    """SparseAttnIndexer.forward_cuda with skip_topk_buffer_clear=True (every other argument as the stock op)."""
    from vllm.model_executor.layers.sparse_attn_indexer import _encode_layer_name

    return torch.ops.vllm.sparse_attn_indexer(
        hidden_states, _encode_layer_name(op.k_cache.prefix), op.k_cache.kv_cache, q_values, q_scale, None, weights,
        op.quant_block_size, op.scale_fmt, op.topk_tokens, op.head_dim, op.max_model_len, op.max_total_seq_len,
        op.topk_indices_buffer, op.skip_k_cache_insert, op.use_pcp, _encode_layer_name(op.dense_mha_metadata_layer_name),
        op.use_fp4_cache, op.dcp_rank, op.dcp_world_size, op.cp_kv_cache_interleave_size, True,
        candidate_blocks=op.candidate_blocks, candidate_block_size=op.candidate_block_size,
        candidate_write=op.candidate_write, topk_backend=op.topk_backend)


# ------------------------------------------------------------------ compressor state save + pooling + norm
def _fused_save_compress_norm(kv_score, positions, state_cache, slot_mapping, query_start_loc, token_to_req_indices,
                              rms_norm_weight, rms_norm_eps, compress_ratio, latent_out):
    """fused_compress_quant_cache.fused_save_compress_norm on the W3 compress stage (bitwise)."""
    args = (kv_score, positions, state_cache, slot_mapping, query_start_loc, token_to_req_indices, rms_norm_weight,
            rms_norm_eps, compress_ratio, latent_out)
    # ratio 2 only: there the ring tail store makes the Triton kernel latency-bound (W3 6.0-6.8 -> 3.9-4.9 us with a
    # cold L2); for ratio 1 the Triton kernel is already ~0.5 us faster than the stage
    ok = (compress_ratio == 2 and kv_score.dtype == torch.float32 and kv_score.stride(1) == 1
          and kv_score.shape[1] == 512 * compress_ratio and kv_score.stride(0) % 4 == 0 and kv_score.data_ptr() % 16 == 0
          and positions.dtype == torch.int64 and positions.is_contiguous() and slot_mapping.dtype == torch.int64
          and slot_mapping.is_contiguous() and rms_norm_weight.dtype == torch.bfloat16
          and rms_norm_weight.is_contiguous() and rms_norm_weight.numel() == 512 and latent_out.dtype == torch.bfloat16
          and latent_out.is_contiguous() and slot_mapping.numel() <= min(kv_score.shape[0], positions.numel()))
    if ok and compress_ratio == 2:
        ok = (state_cache is not None and state_cache.dtype == torch.float32 and state_cache.dim() == 3
              and state_cache.shape[1] <= 8 and state_cache.shape[2] == 1024 and state_cache.stride(2) == 1
              and state_cache.stride(1) % 4 == 0 and state_cache.stride(0) % 4 == 0
              and query_start_loc is not None and query_start_loc.dtype == torch.int32
              and query_start_loc.is_contiguous() and token_to_req_indices is not None
              and token_to_req_indices.dtype == torch.int32 and token_to_req_indices.is_contiguous())
    if not ok:
        return _ORIG["fscn"](*args)
    if slot_mapping.numel() == 0:
        return None
    _ext().compress(kv_score, positions, state_cache if compress_ratio == 2 else None, slot_mapping,
                    query_start_loc if compress_ratio == 2 else None, token_to_req_indices if compress_ratio == 2 else None,
                    rms_norm_weight, rms_norm_eps, compress_ratio, latent_out, True)
    _log_once(f"compress{compress_ratio}", f"compressor save/pool/norm on the W3 stage (ratio {compress_ratio})")
    return None


# ------------------------------------------------------------------ candidate selection (layer 20)
def _select_candidate_blocks(logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat=1):
    """sparse_attn_indexer._select_candidate_blocks for decode rows (row_ks is None) on W3's exact three-stage select:
    the same candidate set as torch.topk with its tie rule (ids ascending per part instead of by score; the
    consumers' expansion sorts)."""
    rows, width = logits.shape
    if (row_ks is not None or rows == 0 or width == 0 or block_size != 8 or logits.dtype != torch.float32
            or logits.stride(1) != 1 or logits.stride(0) % 4 or logits.data_ptr() % 16 or out.dtype != torch.int32
            or out.stride(1) != 1 or out.shape[1] != topk_blocks or topk_blocks > 2048 or topk_blocks % 8
            or row_ke.dtype != torch.int32 or row_ke.numel() * row_repeat < rows or rows > _CAND_MAX_ROWS):
        return _ORIG["select"](logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat)
    gk, scratch = _cand_scratch(rows, width, block_size, topk_blocks, logits.device)
    _ext().cand_select(logits, row_ke.contiguous(), block_size, out, None, None, None, gk, scratch, row_repeat, True)
    _log_once("cand", f"candidate select on the W3 path ({rows} rows, width {width})")


_SIDE: dict = {}


def _fork_consumer_schedule(self, attn_metadata) -> None:
    """At the start of the first candidate consumer (layer 24) of a decode step, before its input projections: the
    sparse block expansion + DeepGEMM schedule on a side stream. Production builds them inside the layer's indexer
    branch, on its critical path (~22 us in the trace); here they run beside the input projections, wq_b and the
    indexer query path, and the indexer branch waits for them right before its sparse logits. Same production calls,
    same inputs (the shared candidate buffer, the consumer rows' metadata). Layers 28/32/36 reuse the schedule as in
    production. The fork never leaves the layer's pre-attention region (see _multistream_ok): the join is the indexer
    branch's wait before its sparse logits, or, with the index side stream off, the stock path's wait in
    _prepare_and_attn."""
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import candidate_blocks_to_sparse_indices
    from vllm.utils.deep_gemm import get_paged_sparse_mqa_logits_metadata

    if not ("index" in _ARMED and _multistream_ok("schedule") and _is_candidate_consumer(self)):
        return
    op = self.indexer.indexer_op
    md = attn_metadata.get(op.k_cache.prefix)
    swa = attn_metadata.get(self.swa_cache_layer.prefix)
    rows_md = getattr(md, "sparse_decode", None)
    if (rows_md is None or swa is None or md.num_prefills > 0 or md.num_decodes == 0 or swa.num_prefills > 0
            or rows_md.kernel_metadata is not None or op.candidate_blocks is None):
        return
    dev = op.candidate_blocks.device
    side = _SIDE.get(("stream", dev))
    if side is None:
        if torch.cuda.is_current_stream_capturing():
            return  # never create a stream inside a capture; the eager warmup forwards create it
        side = _SIDE[("stream", dev)] = torch.cuda.Stream(device=dev)
    n = rows_md.row_ke.shape[0]
    kv_view_pages = op.k_cache.kv_cache.shape[1]
    start, done = torch.cuda.Event(), torch.cuda.Event()
    start.record()
    with torch.cuda.stream(side):
        start.wait()
        candidate_blocks_to_sparse_indices(op.candidate_blocks[:n], rows_md.row_ks, rows_md.row_ke,
                                           op.candidate_block_size, md.sparse_block_kv,
                                           out=(rows_md.sparse_indices, rows_md.end))
        rows_md.kernel_metadata = get_paged_sparse_mqa_logits_metadata(
            rows_md.row_ke, rows_md.block_table, rows_md.row_indices, kv_view_pages, rows_md.sparse_indices,
            torch.int8, md.sparse_block_kv)
        done.record()
    _MEMO.get(attn_metadata)["schedule_ready"] = done
    _log_once("schedule", "candidate consumers' expansion + schedule forked at the first consumer's layer start")


_CAND_MAX_ROWS = 128  # scratch ~0.6 MB per row at width 512K
_CAND_SCRATCH: dict = {}


def _cand_scratch(rows: int, width: int, block_size: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Keys [rows, width / block_size] + stage scratch, grown to the largest row count seen (CUDA graphs capture the
    largest batch first) and shared by every smaller one."""
    key = (width, block_size, k, device)
    have = _CAND_SCRATCH.get(key)
    if have is None or have[0] < rows:
        nb = (width + block_size - 1) // block_size
        gk = torch.empty((rows, (nb + 3) // 4 * 4), dtype=torch.int32, device=device)
        scratch = torch.empty(_ext().cand_scratch_size(rows, width, block_size, k), dtype=torch.int32, device=device)
        have = (rows, gk, scratch)
        _CAND_SCRATCH[key] = have
    return have[1][:rows], have[2]


def _patch(owner, name: str, new, key: str) -> None:
    if key in _ORIG:
        return
    _ORIG[key] = getattr(owner, name)
    setattr(owner, name, new)


def patch_qkv(attention_module) -> None:
    """vllm.models.deepseek_v41.attention"""
    cls = attention_module.DeepseekV4Attention
    _patch(cls, "forward", _forward, "forward")
    _patch(cls, "_fused_qnorm_rope_kv_insert", _fused_qnorm_rope_kv_insert, "insert")


def patch_memo(mega_module) -> None:
    """vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn"""
    cls = mega_module.DeepseekV4MegaAttnAttention
    _patch(cls, "_decode_compressed_kv_and_topk", _decode_compressed_kv_and_topk, "dkt")
    _patch(cls, "forward_mqa", _forward_mqa, "forward_mqa")


def patch_cand(indexer_module) -> None:
    """vllm.model_executor.layers.sparse_attn_indexer"""
    _patch(indexer_module, "_select_candidate_blocks", _select_candidate_blocks, "select")


_ARMED: set = set()
_DISABLED: set = set()


def patch_index(attention_module) -> None:
    """vllm.models.deepseek_v41.attention (fills the step memo; arm together with memo)"""
    _ARMED.add("index")
    _patch(attention_module.DeepseekV4Attention, "_prepare_and_attn", _prepare_and_attn, "prepare")


def patch_cstreams(attention_module) -> None:
    """vllm.models.deepseek_v41.attention"""
    _ARMED.add("cstreams")
    _patch(attention_module.DeepseekV4Attention, "_prepare_and_attn", _prepare_and_attn, "prepare")


def patch_compress(compressor_module) -> None:
    """vllm.models.deepseek_v41.compressor"""
    _patch(compressor_module, "fused_save_compress_norm", _fused_save_compress_norm, "fscn")


FEATURES = {
    "qkv": ("vllm.models.deepseek_v41.attention", patch_qkv),
    "memo": ("vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn", patch_memo),
    "cand": ("vllm.model_executor.layers.sparse_attn_indexer", patch_cand),
    "index": ("vllm.models.deepseek_v41.attention", patch_index),
    "cstreams": ("vllm.models.deepseek_v41.attention", patch_cstreams),
    "compress": ("vllm.models.deepseek_v41.compressor", patch_compress),
}


def install(features: str = "qkv,memo,cand,index,cstreams,compress") -> None:
    """Import the target modules and patch them now (tests; the fastpath bundle patches on import instead)."""
    import importlib

    for f in (x.strip() for x in features.split(",") if x.strip()):
        target, fn = FEATURES[f]
        fn(importlib.import_module(target))
    sys.stderr.write(f"W3 preattn hook installed: {features}\n")
