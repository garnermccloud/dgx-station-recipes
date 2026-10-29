"""vLLM integration: DeepseekV4MegaAttnAttention's decode attention on W1's bitwise kernels.

Replaced call: the decode branch of ``DeepseekV4MegaAttnAttention.forward_mqa`` in
``vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py`` -> ``_forward_decode_mega(q, positions_int32,
flashmla_metadata, swa_metadata, out)`` -> ``torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode``. W1 takes
the same tensors and writes the same (fp8 data, packed ue8m0 scale) buffers, bit-identical to the stock op.

Installed per call when all hold (else the previously installed ``_forward_decode_mega`` runs untouched):
  * layer class in {sparse, swa, draft}: sparse = compressed layers (compress_ratio 1 / 2), swa = window-only target
    layers (0-1), draft = window-only drafter layers (layer_id >= 40);
  * 1 <= num_decode_tokens <= the class's width limit. Sparse layers: up to 16 on the bitwise-parallel kernel, then
    up to 76 on the 2-CTA cluster kernel (stage_attn_cx.cuh; needs a window of <= 128 slots and power-of-two pages).
    Window-only layers: up to 24 (swa and draft) on the split-dims kernel;
  * TP1 shapes: 64 padded heads, bf16 q, MXFP8 (528 B) window cache, NVFP4 (288 B) compressed cache, <= 10 blocks of
    64 keys per token (window + top-k); a call that fails this after the top-k runs the stock op inline.
An installed layer prepares W1 in its constructor (before any CUDA-graph capture). ``stats()`` counts calls per path;
``capture_summary()`` (logged after GPUModelRunner.capture_model) gives the W1 / stock split of captured calls.
"""

import sys

import torch

from . import ops

LAYERS = {"sparse", "swa", "draft"}
MAX_SQ = 16  # sparse layers: bitwise-parallel kernel (stage_attn_par.cuh)
MAX_SQ_CX = 76  # sparse layers above MAX_SQ: cluster kernel (stage_attn_cx.cuh)
MAX_SQ_SWA = 24  # window-only layers: split-dims kernel (stage_attn_ss.cuh)
NUM_LAYERS = 40
MAX_BLOCKS = 10
_scratch: dict = {}
_stats = {"w1": 0, "stock": 0, "stock_shape": 0}
_capture: dict = {}  # (width, path) -> calls recorded into CUDA graphs
_logged: set = set()


def stats() -> dict:
    return dict(_stats)


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH attn: {msg}\n")


def prepare(device: torch.device) -> ops.Scratch:
    """Everything W1 needs before CUDA-graph capture, once per device: the extension, the kernel's smem opt-in, and
    the zeroed scratch (one per device, shared by every layer: launches are stream-ordered and tags unique per
    launch). Called from the layer constructor, so the first decode forward under capture already runs W1."""
    s = _scratch.get(device)
    if s is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("W1 attn: scratch must be prepared before CUDA-graph capture")
        ext = ops._ext()
        ext.w1_attn_prepare()
        words = MAX_SQ * MAX_BLOCKS * (128 + 128 + 2048 + 1)  # tagged 64-bit words (stage_attn_par.cuh)
        with torch.cuda.device(device):
            s = ops.Scratch(
                part_o=torch.zeros(2 * words, dtype=torch.float32, device=device),
                part_ml=torch.zeros(1, dtype=torch.float32, device=device),
                flags=torch.zeros(1, dtype=torch.int32, device=device),
                epoch=torch.zeros(2, dtype=torch.int32, device=device),
            )
        _scratch[device] = s
        _log(f"prepared on {device}: scratch {2 * words * 4 / 2**20:.1f} MiB; layers={sorted(LAYERS)} "
             f"max_sq sparse={MAX_SQ} cluster={MAX_SQ_CX} swa={MAX_SQ_SWA}")
    return s


def _capture_note(width: int, path: str, prefix: str, cls_: str) -> None:
    """Count calls recorded into CUDA graphs per (width, path); log the first W1 layer of each captured width."""
    key = (width, path)
    first = key not in _capture
    _capture[key] = _capture.get(key, 0) + 1
    if first and path == "w1":
        _log(f"capture width {width}: {prefix} ({cls_}) on the W1 kernel")


def capture_summary() -> str:
    widths = sorted({w for w, _ in _capture})
    parts = [f"{w}:{_capture.get((w, 'w1'), 0)}/{_capture.get((w, 'stock'), 0)}" for w in widths]
    return (f"captured decode attention calls, W1/stock per width: {' '.join(parts) or 'none'} "
            f"(total W1 {sum(v for (w, p), v in _capture.items() if p == 'w1')}, "
            f"stock {sum(v for (w, p), v in _capture.items() if p != 'w1')})")


def _u8(t: torch.Tensor | None) -> torch.Tensor | None:
    """Byte view of a 1-byte cache (vLLM may type the V4.1 records as float8_e4m3fn)."""
    if t is None or t.dtype == torch.uint8 or t.element_size() != 1:
        return t
    return t.view(torch.uint8)


def _layer_class(self) -> str:
    if (getattr(self, "compress_ratio", 0) or 0) != 0:
        return "sparse"
    return "draft" if getattr(self, "layer_id", 0) >= NUM_LAYERS else "swa"


def _shapes_ok(self, q: torch.Tensor, comp: torch.Tensor | None, swa_idx: torch.Tensor, extra_idx) -> bool:
    if q.shape[1] != 64 or q.shape[2] != 512 or q.dtype != torch.bfloat16 or q.stride(-1) != 1:
        return False
    swa = _u8(self.swa_cache_layer.kv_cache)
    if swa.dtype != torch.uint8 or swa.dim() != 3 or swa.shape[-1] != 528 or swa.stride(1) != 528:
        return False
    if comp is not None and (comp.dtype != torch.uint8 or comp.dim() != 3 or comp.shape[-1] != 288
                             or comp.stride(1) != 288):
        return False
    total = swa_idx.shape[-1] + (0 if comp is None else extra_idx.shape[-1])
    return (total + 63) // 64 <= MAX_BLOCKS


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _cx_ok(self, comp: torch.Tensor, swa_idx: torch.Tensor) -> bool:
    """The cluster kernel's extra conditions: the window fits the first key pair, power-of-two pages."""
    return swa_idx.shape[-1] <= 128 and _pow2(self.swa_cache_layer.kv_cache.shape[1]) and _pow2(comp.shape[1])


def _width_limit(cls_: str) -> int:
    return max(MAX_SQ, MAX_SQ_CX) if cls_ == "sparse" else MAX_SQ_SWA


def _mode(n: int, comp: torch.Tensor | None) -> str:
    if comp is None:
        return "ss"
    return "bitwise" if n <= MAX_SQ else "cx"


def _stock(mod, self, q, positions_int32, swa_metadata, extra_cache, extra_idx, extra_len, out) -> None:
    """The stock decode call, verbatim from _forward_decode_mega (top-k already computed)."""
    torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode(
        q,
        self.swa_cache_layer.kv_cache.unsqueeze(-2),
        swa_metadata.decode_swa_indices.view(q.shape[0], -1),
        self.scale,
        mod._HEAD_DIM_V,
        self.attn_sink,
        swa_metadata.decode_swa_lens,
        extra_cache,
        extra_idx,
        extra_len,
        False,  # enable_q_norm
        0.0,  # rms_norm_eps
        positions_int32,
        *mod._ROPE_ARGS,
        self.rotary_emb.cos_sin_cache,
        self.n_wv_group,
        *mod._SF_ARGS,
        out.data,
        out.scale,
    )


def _eligible(self) -> bool:
    return _layer_class(self) in LAYERS


def make_forward_decode_mega(mod, orig):
    def _forward_decode_mega(self, q, positions_int32, flashmla_metadata, swa_metadata, out) -> None:
        cls_ = _layer_class(self)
        n = q.shape[0]
        capturing = torch.cuda.is_current_stream_capturing()
        scratch = None
        if cls_ in LAYERS and 1 <= n <= _width_limit(cls_):
            scratch = _scratch.get(q.device)
            if scratch is None and not capturing:  # constructor hook missed this device: prepare eagerly
                scratch = prepare(q.device)
        if scratch is None:
            _stats["stock"] += 1
            if capturing:
                _capture_note(n, "stock", getattr(self, "prefix", "?"), cls_)
            return orig(self, q, positions_int32, flashmla_metadata, swa_metadata, out)
        extra_cache, extra_idx, extra_len = self._decode_compressed_kv_and_topk(flashmla_metadata, swa_metadata)
        assert swa_metadata.decode_swa_indices is not None
        swa_idx = swa_metadata.decode_swa_indices.view(n, -1)
        comp = None if extra_cache is None else _u8(extra_cache.squeeze(-2))
        mode = _mode(n, comp)
        if not _shapes_ok(self, q, comp, swa_idx, extra_idx) or (mode == "cx" and not _cx_ok(self, comp, swa_idx)):
            _stats["stock_shape"] += 1
            if capturing:
                _capture_note(n, "stock_shape", getattr(self, "prefix", "?"), cls_)
            _stock(mod, self, q, positions_int32, swa_metadata, extra_cache, extra_idx, extra_len, out)
            return
        _stats["w1"] += 1
        if capturing:
            _capture_note(n, "w1", getattr(self, "prefix", "?"), cls_)
        elif cls_ not in _logged:
            _logged.add(cls_)
            _log(f"{getattr(self, 'prefix', '?')} ({cls_}) on the W1 kernel ({n} tokens, eager)")
        ops.attn_decode(q, positions_int32, self.rotary_emb.cos_sin_cache, self.attn_sink, self.scale,
                        _u8(self.swa_cache_layer.kv_cache), swa_idx, swa_metadata.decode_swa_lens, comp, extra_idx,
                        extra_len, out.data, out.scale, scratch, mode)

    return _forward_decode_mega


def make_init(orig_init):
    def __init__(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        if _eligible(self):
            prepare(torch.device("cuda", torch.cuda.current_device()))

    return __init__


def install(mod) -> None:
    """Patch ``mod.DeepseekV4MegaAttnAttention`` (``mod`` = the imported flash_mla_mega_attn module). Idempotent.

    * ``__init__``: an installed layer prepares W1 (extension, smem opt-in, scratch) at model build, before any capture.
    * ``_forward_decode_mega``: wraps whatever is installed (stock or another component's), which stays the fallback.
    """
    cls = mod.DeepseekV4MegaAttnAttention
    if getattr(cls, "_w1_installed", False):
        return
    cls.__init__ = make_init(cls.__init__)
    cls._forward_decode_mega = make_forward_decode_mega(mod, cls._forward_decode_mega)
    cls._w1_installed = True
    _log(f"patched {cls.__name__}.__init__ / _forward_decode_mega (W1_LAYERS={sorted(LAYERS)}, W1_MAX_SQ={MAX_SQ}, "
         f"W1_MAX_SQ_CX={MAX_SQ_CX}, W1_MAX_SQ_SWA={MAX_SQ_SWA})")


def install_capture_summary(runner_mod) -> None:
    """Wrap ``GPUModelRunner.capture_model`` of ``runner_mod`` to log the W1 / stock split of the captured graphs."""
    cls = getattr(runner_mod, "GPUModelRunner", None)
    if cls is None or getattr(cls, "_w1_capture_summary", False) or not hasattr(cls, "capture_model"):
        return
    orig = cls.capture_model

    def capture_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        _log(capture_summary())
        return out

    cls.capture_model = capture_model
    cls._w1_capture_summary = True
