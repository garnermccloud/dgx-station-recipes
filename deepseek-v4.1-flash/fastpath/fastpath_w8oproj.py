"""ds41 fastpath component "w8oproj" (W8): the attention output projection's wo_a + MXFP8 quantize as one W8 kernel.

Replaces, for decode calls with T in 1..8 (eager and graph capture; T 9..16 is built but measured ~1 us slower than
DeepGEMM in the PDL chain, so it stays on the stock path):

  stock  DeepGEMM fp8_einsum("bhr,hdr->bhd") -> z bf16 -> [wo_b linear: mxfp8_e4m3_quantize(z) -> mm_mxfp8]
  here   W8 wo_a kernel (libw8oproj.so) -> (zq, zsf) MXFP8 -> [wo_b linear: mm_mxfp8 on the QuantizedActivation]

so the quantize launch is gone and wo_a's weights stream in the kernel's prologue (before griddepcontrol.wait, while
the attention kernel runs). Window layers (split 1) are bitwise: zq, the swizzled scales and wo_b's output equal the
stock path's at every width (w8-persist tests/test_woa.py, tests/test_oproj_real.py on the image's real linear /
attention classes). Sparse and drafter layers use a deterministic 2-part split-K, whose summation order differs from
the stock kernel's. z is an optional test-only output of the library; serving never passes or reads it.

Patches:
  kernels.linear.mxfp8.deep_gemm.DeepGemmMxfp8BmmLinearKernel.process_weights_after_loading
      after the stock pass (the DeepGEMM scale layout exists only from here on: the model's own finalize, which permutes
      wo_a in place, runs earlier, inside load_weights): wo_a-shaped layers get their scales re-laid out as tile-major
      UTCCP atoms, 1 MB per layer, kept beside the stock scales (duplicated: the stock path stays the fallback); the
      kernels load (eager) on the first one
  model.DeepseekV4Model.finalize_mega_attn_weights
      a second chance after a later finalize pass (builds missing or stale atoms); logs what it saw
  flash_mla_mega_attn.DeepseekV4MegaAttnAttention._o_proj
      the W8 kernel when eligible, else the installed _o_proj
Weight-stream prologue per layer kind (_prologue): sparse layers ring + L2 tail before griddepcontrol.wait, window-only
and drafter layers ring only (measured against the real W1 attention kernels: tests/attn_woa_bench.py).
Split-K constants: sparse and drafter layers use 2 K parts (2 CTAs per tile, 1 chain, ticket prefetch); window layers
stay split 1 (bitwise). No env knob selects this.
Spin waits inside the kernel are bounded at 10 s (no short trap).
Loud: the hooks log entry / result / exceptions (with traceback); the first _o_proj call logs the relayout summary; every
fallback reason is logged once per (reason, T), every engaged T once. Atoms are checked against the scale tensor's
version counter (an in-place change after the relayout makes them stale: rebuilt outside capture, else fallback).
Do not combine with "w8post" (both patch _o_proj); if both are listed, list w8oproj first.

"""
from __future__ import annotations

import ctypes
import os
import sys

from fastpath_hook import bounded, register


class _LazyTorch:
    def __getattr__(self, name: str):
        import torch as _torch

        return getattr(_torch, name)


torch = _LazyTorch()

_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(_DIR, "libw8oproj.so")
# Split-K is a code constant, not an env knob. Sparse and drafter layers gain from the deterministic 2-part split
# (2 CTAs per tile, 1 chain, ticket L2 prefetch); window layers measured +0.21 us/layer and stay split 1 (bitwise).
SPLIT_PARTS = 2
SPLIT_CHAINS = 0
SPLIT_CLUSTER = 2
SPLIT_TICKET = True
SPLIT_KINDS = ("sparse", "drafter")
WIDTHS = tuple(range(1, 9))  # engaged widths (the library builds 1..16)
GROUPS, OLORA, WOA_K = 8, 1024, 4096
TARGET_ATTN = "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn"
TARGET_MODEL = "vllm.models.deepseek_v41.nvidia.model"
TARGET_KERNEL = "vllm.model_executor.kernels.linear.mxfp8.deep_gemm"

P, L = ctypes.c_void_p, ctypes.c_longlong


class Args(ctypes.Structure):
    _fields_ = [("x", P), ("x_s0", L), ("x_s1", L), ("xs", P), ("xs_sg", L), ("xs_skb", L), ("w", P), ("wsf", P),
                ("zq", P), ("zsf", P), ("tl", P), ("ticket", P), ("diag", P), ("S", L), ("split", L), ("pdl", L),
                ("prologue", L), ("z", P), ("z32", P), ("chains", L)]


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH w8oproj: {msg}\n")
    sys.stderr.flush()


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(LIB_PATH)
        _lib.w8o_abi.restype = L
        _lib.w8o_abi.argtypes = [ctypes.c_int]
        _lib.w8o_prepare.restype = ctypes.c_int
        _lib.w8o_launch.restype = ctypes.c_int
        _lib.w8o_launch.argtypes = [ctypes.POINTER(Args), P]
        want = {0: ctypes.sizeof(Args), 1: Args.S.offset, 2: Args.diag.offset, 5: Args.prologue.offset,
                6: Args.z32.offset, 7: Args.chains.offset}
        bad = {k: (v, _lib.w8o_abi(k)) for k, v in want.items() if _lib.w8o_abi(k) != v}
        if _lib.w8o_abi(4) < max(WIDTHS):
            bad[4] = (max(WIDTHS), _lib.w8o_abi(4))
        if bad:
            raise RuntimeError(f"libw8oproj.so ABI mismatch (python, C): {bad}")
    return _lib


# ------------------------------------------------------------------------------------------------ boot-time relayout
_ATOMS: dict = {}  # wo_a.weight_scale data_ptr -> (scale tensor, its _version at the relayout, atoms uint8 [64*32*512])
_PREPARED = False
_STATS = {"kernel_hook_calls": 0, "finalize_calls": 0}


def woa_tile_atoms(ws):
    """DeepGEMM MN-major packed wo_a scales [8, 1024, 32] (int32, row-contiguous) -> tile-major UTCCP atoms
    [64 tiles][32 K blocks][32 lanes][4 words] (word i of lane l = row 32 i + l of the tile)."""
    return ws.permute(0, 2, 1).reshape(GROUPS, 32, 8, 4, 32).permute(0, 2, 1, 4, 3).contiguous().view(torch.uint8)


def _scale_layout_ok(ws) -> str | None:
    """None if ws is DeepGEMM's packed MN-major wo_a scale, else why not."""
    if ws is None:
        return "no weight_scale"
    if ws.dtype != torch.int32:
        return f"weight_scale dtype {ws.dtype} (not yet DeepGEMM's packed int32)"
    if tuple(ws.shape) != (GROUPS, OLORA, WOA_K // 128):
        return f"weight_scale shape {tuple(ws.shape)}"
    if ws.stride(1) != 1:
        return f"weight_scale strides {tuple(ws.stride())} (not MN-major)"
    return None


def _relayout_layer(layer) -> str | None:
    """Build (or rebuild if stale) the atoms of one wo_a-shaped linear; None on success, else the reason."""
    w = getattr(layer, "weight", None)
    if w is None or tuple(w.shape) != (GROUPS, OLORA, WOA_K) or w.dtype != torch.float8_e4m3fn:
        return f"weight {None if w is None else (tuple(w.shape), w.dtype)} is not wo_a's [8, 1024, 4096] e4m3"
    ws = getattr(layer, "weight_scale", None)
    why = _scale_layout_ok(ws)
    if why is not None:
        return why
    ent = _ATOMS.get(ws.data_ptr())
    if ent is None or ent[1] != ws._version:
        _ATOMS[ws.data_ptr()] = (ws, ws._version, woa_tile_atoms(ws))  # keep the source alive with its atoms
    return None


def _prepare_once() -> None:
    global _PREPARED
    if _PREPARED:
        return
    with bounded("w8oproj w8o_prepare (load every width's kernel)", 300):
        r = lib().w8o_prepare()  # load every width's kernel now: a lazy module load would stall under capture
    if r != 0:
        raise RuntimeError(f"w8o_prepare failed: {r}")
    _PREPARED = True
    if not (lib().w8o_abi(3) >> SPLIT_PARTS) & 1:
        raise RuntimeError(f"split {SPLIT_PARTS} not in this library (split mask {lib().w8o_abi(3):#x})")
    _log(f"kernels loaded for T in {WIDTHS[0]}..{WIDTHS[-1]}; split-K constants: "
         f"sparse/drafter {SPLIT_PARTS} K parts ({SPLIT_CLUSTER} CTA(s) per tile x "
         f"{SPLIT_PARTS // SPLIT_CLUSTER} chain(s)), ticket L2 prefetch "
         f"{'on' if SPLIT_TICKET and SPLIT_CLUSTER > 1 else 'off'}; window split 1 (bitwise)")


def _boot_failed(where: str, e: BaseException) -> None:
    raise RuntimeError(f"w8oproj: {where} failed: {e!r}") from e


# ------------------------------------------------------------------------------------------------ per-call state
class _Bufs:
    """Per (device, T): MXFP8 output buffers (scales' padding rows stay zero: written once)."""

    def __init__(self, device, T: int) -> None:
        self.zq = torch.empty(T, GROUPS * OLORA, dtype=torch.float8_e4m3fn, device=device)
        self.zsf = torch.zeros(((T + 127) // 128) * 128 * (GROUPS * OLORA // 32), dtype=torch.uint8, device=device)


_BUFS: dict = {}
_TICKETS: dict = {}  # device index -> u32 [2] start counter of the split-K kernels (zero at boot, never reset)
_DIAG = None
_FUSE_OK: dict = {}  # id(module) -> can_fuse_query_quant([wo_b])
_SEEN: set = set()  # (reason, T, target?) already logged
_SUMMARY_DONE = False


def _ticket(device):
    t = _TICKETS.get(device.index)
    if t is None:
        t = _TICKETS[device.index] = torch.zeros(2, dtype=torch.int32, device=device)
    return t


def _bufs(device, T):
    k = (device.index, T)
    b = _BUFS.get(k)
    if b is None:
        b = _BUFS[k] = _Bufs(device, T)
    return b


def _reason(self, attn_out) -> str | None:
    """None if the W8 kernel can serve this call, else why not."""
    if getattr(self, "gemm_rs", None) is not None:
        return "gemm_rs bound"
    data, scale = getattr(attn_out, "data", None), getattr(attn_out, "scale", None)
    if data is None or scale is None or data.dim() != 3:
        return "attn_out is not a 3-D QuantizedActivation"
    if self.n_local_groups != GROUPS or getattr(self, "o_lora_rank", OLORA) != OLORA:
        return f"groups {self.n_local_groups} / o_lora_rank {getattr(self, 'o_lora_rank', None)}"
    if data.dtype != torch.float8_e4m3fn or data.shape[1] < GROUPS or data.shape[2] != WOA_K or data.stride(2) != 1:
        return f"attn_out data {data.dtype} {tuple(data.shape)} strides {tuple(data.stride())}"
    if scale.dtype != torch.int32 or scale.dim() != 3 or scale.stride(0) != 1 or scale.shape[2] != WOA_K // 128:
        return f"attn_out scale {scale.dtype} {tuple(scale.shape)} strides {tuple(scale.stride())}"
    w = self.wo_a.weight
    if tuple(w.shape) != (GROUPS, OLORA, WOA_K) or not w.is_contiguous() or w.dtype != torch.float8_e4m3fn:
        return f"wo_a weight {w.dtype} {tuple(w.shape)}"
    ws = self.wo_a.weight_scale
    ent = _ATOMS.get(ws.data_ptr())
    if ent is None or ent[1] != ws._version:
        stale = ent is not None
        if torch.cuda.is_current_stream_capturing():
            return "wo_a atoms stale (scale changed in place after the relayout)" if stale else "no wo_a atoms"
        why = _relayout_layer(self.wo_a)  # outside capture: build now
        if why is not None:
            return f"no wo_a atoms: {why}"
    ok = _FUSE_OK.get(id(self))
    if ok is None:
        from vllm.models.deepseek_v41.common.ops.query_quant import can_fuse_query_quant

        ok = _FUSE_OK[id(self)] = bool(can_fuse_query_quant([self.wo_b]))
    if not ok:
        return f"wo_b {type(getattr(getattr(self.wo_b, 'quant_method', None), 'kernel', None)).__name__} does not take " \
               "a FlashInfer MXFP8 QuantizedActivation"
    return None


def _launch(self, attn_out):
    global _DIAG
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

    data, scale = attn_out.data[:, :GROUPS], attn_out.scale[:, :GROUPS]
    T = data.shape[0]
    dev = data.device
    b = _bufs(dev, T)
    if _DIAG is None:
        _DIAG = torch.zeros(16, dtype=torch.int32).pin_memory()
    a = Args()
    a.x, a.x_s0, a.x_s1 = ctypes.c_void_p(data.data_ptr()), data.stride(0), data.stride(1)
    a.xs, a.xs_sg, a.xs_skb = ctypes.c_void_p(scale.data_ptr()), scale.stride(1), scale.stride(2)
    a.w = ctypes.c_void_p(self.wo_a.weight.data_ptr())
    a.wsf = ctypes.c_void_p(_ATOMS[self.wo_a.weight_scale.data_ptr()][2].data_ptr())
    a.zq, a.zsf = ctypes.c_void_p(b.zq.data_ptr()), ctypes.c_void_p(b.zsf.data_ptr())
    a.tl = None
    kind = _kind(self)
    split = SPLIT_PARTS if kind in SPLIT_KINDS else 1
    a.chains = SPLIT_CHAINS if split > 1 else 0
    a.ticket = ctypes.c_void_p(_ticket(dev).data_ptr()) if split > 1 and SPLIT_CLUSTER > 1 and SPLIT_TICKET else None
    if ("split", kind) not in _SEEN:
        _SEEN.add(("split", kind))
        _log(f"split policy {kind}: split={split} chains={int(a.chains)} ticket={bool(a.ticket)}")
    a.diag = ctypes.c_void_p(_DIAG.data_ptr())
    a.S, a.split, a.pdl = T, split, 1
    a.prologue = _prologue(self)
    r = lib().w8o_launch(ctypes.byref(a), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if r != 0:
        raise RuntimeError(f"w8o_launch failed: {r} (T={T})")
    qa = QuantizedActivation(b.zq, b.zsf, torch.bfloat16, torch.Size((T, GROUPS * OLORA)), kMxfp8Dynamic)
    return self._wo_b_proj(qa)


def _kind(self) -> str:
    if (getattr(self, "compress_ratio", 0) or 0) != 0:
        return "sparse"
    return "window" if _is_target(self) else "drafter"


def _prologue(self) -> int:
    """The weight-stream prologue for this layer's attention neighbour (tests/attn_woa_bench.py, real W1 kernels):
    sparse layers (attn_par leaves 44 SMs free) stream the ring plus an L2 prefetch of the tile before
    griddepcontrol.wait (0); window-only and drafter layers (attn_ss leaves ~100 SMs free, so a full prologue slows
    attention by ~2 us) stream the ring only (1)."""
    return 0 if (getattr(self, "compress_ratio", 0) or 0) != 0 else 1


def _summary_once() -> None:
    global _SUMMARY_DONE
    if _SUMMARY_DONE:
        return
    _SUMMARY_DONE = True
    mb = sum(t.numel() for _, _, t in _ATOMS.values()) / 2**20
    _log(f"w8: relayout +{mb / 1024:.3f} GB duplicated, 0 GB in place (wo_a scale atoms, {len(_ATOMS)} layers; kernel "
         f"hook calls {_STATS['kernel_hook_calls']}, finalize calls {_STATS['finalize_calls']}); "
         f"kernels {'loaded' if _PREPARED else 'NOT loaded'}")


def _is_target(self) -> bool:
    """A target-model attention layer (prefix under language_model.; no prefix counts as target: fail closed)."""
    prefix = getattr(self, "prefix", None)
    return not isinstance(prefix, str) or prefix.startswith("language_model.")


def _make_o_proj(orig):
    def _o_proj(self, attn_out, positions):
        _summary_once()
        T = attn_out.data.shape[0] if getattr(attn_out, "data", None) is not None else -1
        if T not in WIDTHS:
            why = f"T={T} outside {WIDTHS[0]}..{WIDTHS[-1]}"
        else:
            try:
                why = _reason(self, attn_out)
                if why is None and not _PREPARED:
                    if torch.cuda.is_current_stream_capturing():
                        why = "kernels not loaded before capture"
                    else:
                        _prepare_once()
            except Exception as e:
                raise RuntimeError(f"w8oproj eligibility check failed: {e!r}") from e
        if why is not None:
            key = (why, T, _is_target(self))
            if key not in _SEEN:
                _SEEN.add(key)
                _log(f"routing to the stock _o_proj: {why} [{getattr(self, 'prefix', getattr(self, 'layer_id', '?'))}, "
                     f"T={T}{'' if _is_target(self) else ', drafter'}]")
            if T in WIDTHS and _is_target(self):
                raise RuntimeError(f"w8oproj did not engage: {why} [{getattr(self, 'prefix', '?')}, T={T}]")
            return orig(self, attn_out, positions)
        key = ("engaged", T, _is_target(self))
        if key not in _SEEN:
            _SEEN.add(key)
            _log(f"engaged at T={T} [{getattr(self, 'prefix', getattr(self, 'layer_id', '?'))}]")
        kind = _kind(self)
        if ("prologue", kind) not in _SEEN:
            _SEEN.add(("prologue", kind))
            pr_ = _prologue(self)
            _log(f"prologue for {kind} layers: p{pr_} ({'ring + L2 tail' if pr_ == 0 else 'ring only'} before "
                 f"griddepcontrol.wait) [first: {getattr(self, 'prefix', '?')}]")
        return _launch(self, attn_out)

    _o_proj.__w8o__ = True
    return _o_proj


@register("w8oproj", TARGET_ATTN)
def _patch_attn(module) -> None:
    lib()  # fail at boot, not at capture, if the library or its ABI is wrong
    cls = module.DeepseekV4MegaAttnAttention
    if not getattr(cls._o_proj, "__w8o__", False):
        cls._o_proj = _make_o_proj(cls._o_proj)
    _log(f"patched {TARGET_ATTN}.DeepseekV4MegaAttnAttention._o_proj (T in {WIDTHS[0]}..{WIDTHS[-1]})")


@register("w8oproj", TARGET_KERNEL)
def _patch_kernel(module) -> None:
    cls = module.DeepGemmMxfp8BmmLinearKernel
    if getattr(cls.process_weights_after_loading, "__w8o__", False):
        return
    orig = cls.process_weights_after_loading

    def process_weights_after_loading(self, layer, _orig=orig):
        _orig(self, layer)
        _STATS["kernel_hook_calls"] += 1
        first = _STATS["kernel_hook_calls"] == 1
        try:
            why = _relayout_layer(layer)
            if first or why is not None:
                _log(f"kernel hook ({getattr(layer, 'prefix', '?')}): "
                     + ("atoms built" if why is None else f"skipped: {why}") + f"; {len(_ATOMS)} layers so far")
            if why is None:
                _prepare_once()
        except Exception as e:
            _boot_failed(f"kernel hook ({getattr(layer, 'prefix', '?')})", e)

    process_weights_after_loading.__w8o__ = True
    cls.process_weights_after_loading = process_weights_after_loading
    _log(f"patched {TARGET_KERNEL}.DeepGemmMxfp8BmmLinearKernel.process_weights_after_loading (boot relayout)")


@register("w8oproj", TARGET_MODEL)
def _patch_model(module) -> None:
    cls = module.DeepseekV4Model
    if getattr(cls.finalize_mega_attn_weights, "__w8o__", False):
        return
    orig = cls.finalize_mega_attn_weights

    def finalize_mega_attn_weights(self, _orig=orig):
        with bounded("DeepseekV4Model.finalize_mega_attn_weights (wo_a relayout)", 300):
            _orig(self)  # wo_a / wq_b permuted into the mega-attention layouts (in place)
            _STATS["finalize_calls"] += 1
            try:
                n, built, why_first = 0, 0, None
                for m in self.modules():
                    wo_a = getattr(m, "wo_a", None)
                    if wo_a is None or not hasattr(wo_a, "weight_scale"):
                        continue
                    n += 1
                    why = _relayout_layer(wo_a)
                    built += why is None
                    why_first = why_first or why
                _log(f"finalize hook (call {_STATS['finalize_calls']}): {n} wo_a layers, {built} with atoms"
                     + (f"; not yet: {why_first}" if why_first else ""))
                if built:
                    _prepare_once()
            except Exception as e:
                _boot_failed("finalize hook", e)

    finalize_mega_attn_weights.__w8o__ = True
    cls.finalize_mega_attn_weights = finalize_mega_attn_weights
    _log(f"patched {TARGET_MODEL}.DeepseekV4Model.finalize_mega_attn_weights (relayout second chance)")
