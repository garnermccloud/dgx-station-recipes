"""ds41 fastpath component "boundary" (W4): one-launch mHC boundaries and a one-launch MoE router, bit-exact.

Replaces, inside FULL CUDA-graph captures of decode steps (the production mHC overlap path, 1 <= T <= 16):

  mHC boundary (both sublayers, DeepseekV4DecoderLayer -> ops.mhc.mhc_shifted_post_pre with a side stream):
      stock   main: mhc_post_tilelang -> mhc_pre_big_fuse_with_norm(input)
              side: deep_gemm sm100_tf32_hc_prenorm_gemm -> mhc_pre_big_fuse_with_norm(stats)
      here    main: w4 seam (post + collapse + RMSNorm, one launch, a 16-CTA cluster)
              side: w4 mix (tf32 split GEMM + split sums of squares + stats, one launch)
  MoE glue (DeepseekV4MoE.forward on the MegaMoE + peer-tier-v2 path, 1 <= T <= 16):
      stock   GateLinear (ll_bf16_gemm: LLBf16Dotprod T <= 4, LLBf16SplitK above) -> dsv4_topk -> _route_send2 -> _prepare_megamoe_inputs -> MegaMoE -> _finish2
      here    w4 router (gate GEMM + top-6 + hot/cold split + cold-row MXFP8 + peer v2 publish + MegaMoE staging of x,
              x_sf, shared_x_sf, topk_idx, topk_weights; one launch) -> [join the mHC side stream] -> MegaMoE -> _finish2

  16 < T <= 64 boundaries (DeepGEMM Mega mHC): w4 mega (mega_seam on the main stream, mega_mix beside it on our own
      side stream, joined at the next boundary or before MegaMoE).
  DSpark drafter layers (registered when the speculator loads the draft model):
      boundaries   stock   Mega mHC at every T (the drafter's layers get no mHC stream)
                   here    w4 mega at every T <= 64 (the FFN-side seam prefetches the layer's gate into L2)
      MoE glue     stock   bf16_mega_gate (128 experts, top-3; above T = 1) -> prepare_megamoe_inputs -> MegaMoE
                   here    w4 drafter router (gate + top-3 + MegaMoE staging, one launch) -> [join the mix] -> MegaMoE
      first layer's attention side (mhc_pre_delayed_tilelang, no carried pre-mix)
                   stock   DeepGEMM tf32 prenorm GEMM -> TileLang fused pre (collapse of stream 0 + RMSNorm, stats)
                   here    w4 seam (stream 0 collapse + RMSNorm) on the main stream, w4 mix beside it on our side stream
      head (DSparkDeepseekV4Model.forward's tail; only the audited forward)
                   stock   mhc_post_tilelang -> hc_collapse_triton
                   here    w4 head (one launch)

Every output is bit-identical to the stock kernels at these widths (w4-boundary tests: test_boundary.py,
test_router.py, test_stages.py, test_mega.py, test_draft_router.py, test_draft_moe.py). Outside them everything runs
stock: eager/piecewise steps at T <= 16, T > 16 routing of the backbone (DeepGEMM mega_gate), hash-routed layers, the
first layer (embedding broadcast), Engram layers and the final collapse.

The FFN-side mix runs on the mHC side stream and is joined before MegaMoE (MegaMoE runs on all 152 SMs with grid
syncs; a side kernel still holding SMs would stall it). hc_fn is consumed as a tf32-RNE mma.sync fragment image, built
for every decoder layer at weight-load time (DeepseekV4Model.finalize_mhc_broadcast_weights), 1.97 MB per sublayer.

Env:
The shipped recipe has the peer-v3 publish, fused finish, lean fence, PDL mix, post fuse and Engram pre-mix on;
there are no env switches for them.
"""
from __future__ import annotations

import ctypes
import os
import sys

from fastpath_hook import bounded, register


class _LazyTorch:
    """torch, imported on first use: components.py is imported by sitecustomize at interpreter start."""

    def __getattr__(self, name: str):
        import torch as _torch

        return getattr(_torch, name)


torch = _LazyTorch()

_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(_DIR, "libw4boundary.so")
SEAM_ON = True
ROUTER_ON = True
# W4_BOUNDARY_PEER3=1: with W6's `peer3` component installed, the router publishes the cold rows to peer tier v3 and the
# layer finishes with peer3's finish (instead of v2 + _finish2). With peer3 installed and this off, the router keeps v2
# and covers only 5 <= T <= 16, so the other T <= 64 stay on peer3's own v3 path.
PEER3_ON = True
# W4_BOUNDARY_FUSE_FINISH=1 (with the v3 publish): the MoE call's peer-v3 finish (wait for the RTX, add the cold rows,
# advance the call counter) runs inside the next layer's attention-side seam instead of its own launch. Only for MoE
# layers whose next consumer is that seam (not the last layer, not before an Engram layer); anything else that would
# see the MoE output first flushes the finish as its own launch.
FUSE_FINISH = True
# W4_BOUNDARY_LEAN_FENCE=1: the router publishes without per-CTA system-scope fences (release/acquire arrivals and one
# st.release.sys per record carry the ordering; W6's protocol section 6). Default off until W6 signs off.
LEAN_FENCE = True
# The attention-side seam MXFP8 output is disabled in the shipped build (kept as a compile-time branch).
# W4_BOUNDARY_MXFP8_OUT was: the attention-side seam also writes the layer input as MXFP8 (FlashInfer
# mxfp8_quantize(is_sf_swizzled_layout=True), bit-identical) and `prequantized(x)` hands it out; FlashInfer's MXFP8
# linear kernels' input quantize (vllm.model_executor.kernels.linear.mxfp8.flashinfer) then skips its launch for x.
MXFP8_OUT = False
TARGET_FI_LINEAR = "vllm.model_executor.kernels.linear.mxfp8.flashinfer"
TARGET_BREAKABLE = "vllm.compilation.breakable_cudagraph"
CLUSTER = 16
# W4_BOUNDARY_MIX_PDL=1 (M3; default 0 since the in-server trace of ab-mixv3-1021: with the deferred mix the graph
# loses seam #2's early launch, +0.83 us per layer) defers the FFN-side mix (T <= 16) to the MoE call, which
# launches it on the main stream as the router's PDL dependent: the router triggers it after its own dependency wait (the seam is complete, so the mix needs no wait of its own and reads rn
# at L2), and the mix waits for the router at its end, so its completion implies the router's: MegaMoE then follows
# in stream order, and the next layer's attention seam after MegaMoE. A consumer other than our router (a stock MoE
# call, the next boundary) gets the side-stream launch and join first.
MIX_PDL = True
# The PDL mix releases MegaMoE at its first instruction, so
# MegaMoE's launch and prologue overlap the mix (MegaMoE does only set-up before its own dependency wait, which
# returns after the mix grid, and so the router grid, has completed).
MIX_EARLY = True
# The FFN-side mix (T <= 16, our router's MoE call) stays on vLLM's mHC side stream and is not joined before
# MegaMoE; vLLM joins the mHC stream after the MoE call, so only the next layer's boundary waits for it
# (bench/mix_offpath.py: MegaMoE ends ~2.4 us earlier than with the M3 in-stream mix; the mix's 16 SMs cost MegaMoE
# ~1 us). MegaMoE runs at kNumSMs = 144 (deep_gemm.set_num_sms around the call); its output is bitwise independent
# of kNumSMs (tests/test_mega_nsm.py). This implies MIX_PDL=0.
MIX_OFFPATH = True
# COMBINE (needs W6's MegaMoE fork v11, fastpath_megamoe.c11_*): for a fused-finish MoE call (T <= 16, our router)
# the router sets v11's skip_combine word, MegaMoE stops after its L2 epilogues (no combine, no y, no grid sync #2),
# and the post-MoE seam sums the 7 combine slots itself (v5's order, bitwise) before the cold add. A finish that
# falls back to peer3's kernel first runs W6's standalone combine (combine_into).
COMBINE = True
# Combine in the seam only for calls with T <= 8; above that the combine stays in MegaMoE. The T cap is counted and
# logged, not silent.
COMBINE_MAX_T = 8
MEGA_SMS = 144
if MIX_OFFPATH:
    MIX_PDL = False
# W4_BOUNDARY_L2_PREFETCH=0: the attention-side mix does not pull the FFN-side hc_fn image into L2.
L2_PREFETCH = True
# W4_BOUNDARY_POST_FUSE=1: the MoE layers before an Engram layer and the last layer also defer their peer-v3 finish
# (FUSE_FINISH) into the post-only boundary that follows (model.mhc_post_tilelang): finish + post in one seam launch.
POST_FUSE = True
# W4_BOUNDARY_ENGRAM_PRE=1: an Engram layer's attention entry (model.mhc_pre_delayed_overlap with the carried pre-mix)
# runs on the seam (post = 0, comb = I) + the mix on the mHC stream instead of TileLang's input epilogue + DeepGEMM.
ENGRAM_PRE = True
# W4_BOUNDARY_COPY_AHEAD=1 (needs a sidecar whose RTX kernel writes the host ready word, advertised at segment
# offset 1160): the RTX's out rows are copied into a device mirror during MegaMoE; see _copy_ahead.
COPY_AHEAD = False
GSIZE = 1               # tokens per seam cluster (C1 A/B 2026-10-24: 129.0 vs 126.4/125.6 steps/s)
SEAM_MAX_T = 16         # production's MHC_OVERLAP_MAX_TOKENS: the stream path we replicate
# 16 < T <= 64: production runs DeepGEMM's Mega mHC; w4 mega replicates it (W4_BOUNDARY_MEGA=0 turns it off)
MEGA_ON = True
MEGA_MAX_T = 64
MEGA_MIX_FIRST = True
MEGA_SPLITS = 40
# W4_BOUNDARY_DRAFT=0 leaves the DSpark drafter's layers stock (default 1: w4 mega at every T <= 64, and the drafter
# router for its 128-expert top-3 gate at 2 <= T <= 64)
DRAFT_ON = True
SPECULATOR_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
DSPARK_MODULE = "vllm.models.deepseek_v41.nvidia.dspark"
ROUTER_MIN_T, ROUTER_MAX_T = 1, 16   # ll_bf16_gemm widths: LLBf16Dotprod (T <= 4), LLBf16SplitK (5..16)
HIDDEN, HC, MIX, EXPERTS, TOPK = 5120, 4, 24, 384, 6
DRAFT_EXPERTS, DRAFT_TOPK = 128, 3   # dspark_n_routed_experts, dspark_num_experts_per_tok
DRAFT_MIN_T = 2                      # at T = 1 production's 128-expert gate takes GateLinear (stays stock)
IMAGE_SENTINEL_LO = 129264
TARGET_MODEL = "vllm.models.deepseek_v41.nvidia.model"
TARGET_V4 = "vllm.models.deepseek_v4.nvidia.model"


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH boundary: {msg}\n")


# ------------------------------------------------------------------------------------------------ C ABI
P = ctypes.c_void_p
I = ctypes.c_int


class SeamParams(ctypes.Structure):
    _fields_ = [(n, P) for n in ("x", "residual", "post", "comb", "pre", "norm_w", "cold_has", "cold_pos", "cold_out",
                                 "cold_rows", "residual_out", "aux_out", "layer_input", "fi_q", "fi_sf_swz",
                                 "fi_sf_lin", "prep_x", "prep_sf", "prep_shared_sf")] + \
               [("shared_block_m", I), ("T", I), ("p3_seg", P), ("p3_state", P), ("p3_stats", P),
                ("p3_stats_words", I), ("p3_layer", I), ("p3_spins", ctypes.c_longlong), ("p3_y", P),
                ("p3_mirror", P), ("sf_pad_ready", I), ("cmb_slots", P), ("cmb_slot_stride", ctypes.c_ulonglong),
                ("cmb_tok_stride", ctypes.c_ulonglong), ("cmb_topk", P), ("dbg", P)]


class MixParams(ctypes.Structure):
    _fields_ = [(n, P) for n in ("rn", "fn_img", "scale", "base", "mul", "sqr", "post_out", "comb_out", "pre_out",
                                 "counters")] + [("T", I), ("pdl_mode", I), ("l2_prefetch", P),
                                                 ("l2_prefetch_bytes", ctypes.c_longlong), ("dbg", P)]


class RouterParams(ctypes.Structure):
    _fields_ = [(n, P) for n in ("li", "gate_w", "bias", "bias_vl", "input_ids", "row_map", "is_padding")] + \
               [("routed_scale", ctypes.c_float), ("T", I), ("layer", I)] + \
               [(n, P) for n in ("logits", "cand_v", "cand_w", "cand_i", "sync", "topk_ids", "topk_w", "hot_ids",
                                 "mega_idx", "mega_w", "cold_ids", "has", "pos", "rows", "prep_x", "prep_sf")] + \
               [("prep_sf_sm", I), ("prep_sf_sk", I), ("prep_shared_sf", P), ("prep_ssf_sm", I), ("prep_ssf_sk", I),
                ("shared_block_m", I), ("cold_q", P), ("cold_sf", P), ("quant_cold", I)] + \
               [(n, P) for n in ("peer_x", "peer_xs", "peer_ids", "peer_w", "peer_words", "peer_header",
                                 "peer_seq")] + [("peer_pad", I), ("p3_seg", P), ("p3_state", P), ("image_mask", P), ("mg_part", P),
                                                   ("mg_count", P), ("lean_fence", I), ("trigger_late", I),
                                                   ("mega_ctl", P), ("mega_ctl_val", ctypes.c_uint32), ("dbg", P)]


class DraftRouterParams(ctypes.Structure):
    _fields_ = [(n, P) for n in ("li", "gate_w", "bias", "bias_vl", "image_mask", "is_padding")] + \
               [("routed_scale", ctypes.c_float), ("T", I), ("topk_idx", P), ("topk_w", P), ("prep_x", P),
                ("prep_sf", P), ("prep_sf_sm", I), ("prep_sf_sk", I), ("prep_shared_sf", P), ("prep_ssf_sm", I),
                ("prep_ssf_sk", I), ("shared_block_m", I), ("logits", P), ("dbg", P)]


class HeadParams(ctypes.Structure):
    _fields_ = [(n, P) for n in ("x", "residual", "post", "comb", "pre", "out")] + [("T", I), ("dbg", P)]


class MegaParams(ctypes.Structure):
    _fields_ = [(n, P) for n in ("x", "residual", "post", "comb", "pre", "norm_w", "fn", "scale", "base",
                                 "residual_out", "y_out", "post_out", "comb_out", "pre_out", "gemm_part", "hc_part",
                                 "counters")] + \
               [(n, ctypes.c_float) for n in ("hc_norm_eps", "hc_pre_eps", "hc_post_scale", "sinkhorn_eps", "rms_eps",
                                              "rms_scale")] + [("sinkhorn_iters", I), ("T", I), ("l2_prefetch", P),
                                                               ("l2_prefetch_bytes", ctypes.c_longlong), ("dbg", P)]


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(LIB_PATH)
        for name, st in (("w4_seam_launch", SeamParams), ("w4_mix_launch", MixParams),
                         ("w4_router_launch", RouterParams), ("w4_router_mega_launch", RouterParams),
                         ("w4_mega_seam_launch", MegaParams),
                         ("w4_mega_mix_launch", MegaParams), ("w4_draft_router_launch", DraftRouterParams),
                         ("w4_head_launch", HeadParams)):
            fn = getattr(_lib, name)
            fn.restype = I
            fn.argtypes = [ctypes.POINTER(st)] + ([I, I, P] if name == "w4_seam_launch" else [I, P])
        _lib.w4_abi.restype = I
        _lib.w4_abi.argtypes = [I]
        want = {0: ctypes.sizeof(SeamParams), 1: SeamParams.dbg.offset, 2: ctypes.sizeof(MixParams),
                3: MixParams.dbg.offset, 4: ctypes.sizeof(RouterParams), 5: RouterParams.dbg.offset,
                6: RouterParams.peer_pad.offset, 7: RouterParams.quant_cold.offset, 8: RouterParams.p3_seg.offset,
                12: ctypes.sizeof(MegaParams), 13: MegaParams.dbg.offset, 15: SeamParams.p3_seg.offset,
                16: ctypes.sizeof(DraftRouterParams), 17: DraftRouterParams.dbg.offset,
                18: ctypes.sizeof(HeadParams), 19: HeadParams.dbg.offset, 20: MixParams.pdl_mode.offset,
                21: RouterParams.trigger_late.offset}
        bad = {k: (v, _lib.w4_abi(k)) for k, v in want.items() if _lib.w4_abi(k) != v}
        if bad:
            raise RuntimeError(f"libw4boundary.so ABI mismatch (python, C): {bad}")
    return _lib


def _p(t):
    return None if t is None else ctypes.c_void_p(t.data_ptr())


def _stream(s=None) -> ctypes.c_void_p:
    return ctypes.c_void_p((s or torch.cuda.current_stream()).cuda_stream)


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} launch failed: cudaError {rc}")


# ------------------------------------------------------------------------------------------------ hc_fn images
def tf32_rne(fn: torch.Tensor) -> torch.Tensor:
    """hc_fn rounded to tf32 with round-to-nearest-even (the operand DeepGEMM's tf32 prenorm GEMM multiplies)."""
    u = fn.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    r = (u + 0xFFF + ((u >> 13) & 1)) & 0xFFFFE000
    r = torch.where(r >= 2**31, r - 2**32, r)
    return r.to(torch.int32).view(torch.float32).reshape(fn.shape).contiguous()


def mix_fn_image(fn: torch.Tensor) -> torch.Tensor:
    """hc_fn [24, 4H] -> the mix kernel's mma.sync B-fragment image, [split 16][slot 4][n-atom 3][step 40][lane 32][2]
    = tf32_rne(fn)[na*8 + lane//4][split*1280 + (slot*40 + step)*8 + lane%4 + 4*j]."""
    fr = tf32_rne(fn)
    dev = fn.device
    s, sl, na, st, ln, j = torch.meshgrid(*(torch.arange(v, device=dev) for v in (16, 4, 3, 40, 32, 2)),
                                          indexing="ij")
    n = na * 8 + ln // 4
    k = s * 1280 + (sl * 40 + st) * 8 + ln % 4 + 4 * j
    return fr[n, k].reshape(-1).contiguous()


_IMAGES: dict[int, torch.Tensor] = {}   # hc_fn data_ptr -> image (kept alive with the fn reference)
_ATTN_FNS: set[int] = set()             # data_ptrs of the hc_attn_fn tensors (attention-side boundaries)
_FFN_IMAGE_OF: dict[int, torch.Tensor] = {}  # hc_attn_fn data_ptr -> the same layer's FFN-side image (L2 prefetch)
_FN_REFS: list[torch.Tensor] = []
# The DSpark drafter's decoder layers (registered when the speculator loads the draft model): their boundaries get no
# mHC side stream, so production runs Mega mHC there at every width; their MoE is the stock 128-expert top-3 path.
_DRAFT_FNS: set[int] = set()            # data_ptrs of the drafter layers' hc_attn_fn / hc_ffn_fn
_DRAFT_MOES: set[int] = set()           # id() of the drafter layers' DeepseekV4MoE modules
_DRAFT_GATES: dict[int, torch.Tensor] = {}  # hc_ffn_fn data_ptr -> that layer's gate weight (prefetched by the seam)
# The drafter's first layer's attention side (mhc_pre_delayed_tilelang with no carried pre-mix): its hc_attn_fn
# data_ptr -> the mix's fragment image; and per device the constant seam inputs that make the seam a plain collapse of
# stream 0 + RMSNorm: x = 0, post = 0, comb = identity, pre = one-hot(0), for T <= MEGA_MAX_T
_DRAFT_ENTRY: dict[int, torch.Tensor] = {}
_ENTRY_CONST: dict[int, tuple] = {}
_DRAFT_REFS: list = []


def build_images(layers) -> int:
    nbytes = 0
    for layer in layers:
        for name in ("hc_attn_fn", "hc_ffn_fn"):
            fn = getattr(layer, name, None)
            if fn is None or fn.dim() != 2 or tuple(fn.shape) != (MIX, HC * HIDDEN) or fn.dtype != torch.float32:
                continue
            if fn.data_ptr() in _IMAGES:
                continue
            img = mix_fn_image(fn.data)
            _IMAGES[fn.data_ptr()] = img
            if name == "hc_attn_fn":
                _ATTN_FNS.add(fn.data_ptr())
            _FN_REFS.append(fn)
            nbytes += img.numel() * 4
        fa, ff = getattr(layer, "hc_attn_fn", None), getattr(layer, "hc_ffn_fn", None)
        if fa is not None and ff is not None and fa.data_ptr() in _IMAGES and ff.data_ptr() in _IMAGES:
            _FFN_IMAGE_OF[fa.data_ptr()] = _IMAGES[ff.data_ptr()]
    return nbytes


# ------------------------------------------------------------------------------------------------ scratch
class _Scratch:
    def __init__(self, device: torch.device) -> None:
        self.mul = torch.empty(16 * MEGA_MAX_T * MIX, dtype=torch.float32, device=device)
        self.sqr = torch.empty(16 * MEGA_MAX_T, dtype=torch.float32, device=device)
        self.mix_counters = torch.zeros(MEGA_MAX_T // 16, dtype=torch.int32, device=device)
        self.cand_v = torch.empty(MEGA_MAX_T * 144, dtype=torch.float32, device=device)
        self.cand_w = torch.empty(MEGA_MAX_T * 144, dtype=torch.float32, device=device)
        self.cand_i = torch.empty(MEGA_MAX_T * 144, dtype=torch.int32, device=device)
        self.mg_part = torch.empty(8 * MEGA_MAX_T * EXPERTS + 2 * MEGA_MAX_T * TOPK, dtype=torch.float32, device=device)
        self.mg_count = torch.zeros(4, dtype=torch.int32, device=device)
        self.router_sync = torch.zeros(4, dtype=torch.int32, device=device)
        # the T <= 16 router's candidate mailboxes (two, alternating by launch; score / weight / id words), seeded with
        # the sentinel 0xFFFFFFFF; the kernel re-arms each one after use. Separate from cand_* (the Mega gate's).
        self.box = torch.full((3, 2 * 16 * 144), -1, dtype=torch.int32, device=device)
        self.mega_gemm = torch.empty(MEGA_SPLITS * MEGA_MAX_T * MIX, dtype=torch.float32, device=device)
        self.mega_hc = torch.empty(MEGA_SPLITS * MEGA_MAX_T, dtype=torch.float32, device=device)
        self.mega_counters = torch.zeros(MEGA_MAX_T // 16, dtype=torch.int32, device=device)
        self.mega_side = torch.cuda.Stream(device)
        # MXFP8 layer input (MXFP8_OUT) per T <= 16: values, and swizzled 128x4 scales with the padding rows zeroed
        # once here (the seam writes rows < T only). Reused every layer: the consumer (the next GEMM) reads them
        # before the next attention seam writes them again.
        self.mx = [(torch.empty(t, HIDDEN, dtype=torch.float8_e4m3fn, device=device),
                    torch.zeros(128 * (HIDDEN // 32), dtype=torch.uint8, device=device))
                   for t in range(1, SEAM_MAX_T + 1)]


_SCRATCH: dict[int, _Scratch] = {}


def _scratch(device: torch.device) -> _Scratch:
    s = _SCRATCH.get(device.index)
    if s is None:
        s = _SCRATCH[device.index] = _Scratch(device)
    return s


# The side stream that holds a pending mix: (stream, ours, capturing). Joined before the next MegaMoE / at the stock
# MoE entry; ours (the T > 16 mega mix on our own side stream, which the decoder layer never waits for) also at the
# next boundary. An entry recorded under a different capture state than the join's is dropped: the capture's start
# synchronized the eager work, and a dependency on it could not be captured.
_PENDING: list[tuple] = []


def _set_pending(stream, ours: bool) -> None:
    _PENDING.clear()
    _PENDING.append((stream, ours, torch.cuda.is_current_stream_capturing()))


def join_pending(ours_only: bool = False) -> None:
    if not _PENDING or (ours_only and not _PENDING[-1][1]):
        return
    stream, _, capturing = _PENDING.pop()
    _PENDING.clear()
    if capturing == torch.cuda.is_current_stream_capturing():
        torch.cuda.current_stream().wait_stream(stream)


# A deferred peer-v3 finish (FUSE_FINISH): the MoE output y, its has[T], layer, and the peer3 handle.
_PENDING_FINISH: list[dict] = []
_FUSE_LAYERS: set[int] = set()   # MoE layer indices whose output feeds the next layer's mhc_shifted_post_pre


# COPY_AHEAD (option d): the router arms the host ready word (0) before its v3 records, and the RTX sets it (1) after
# the call's done tag. After the router, a side stream waits (a stream memory op) for it, copies the done tag and then
# the T out rows into a device mirror with the copy engine during MegaMoE, and sets the mirror's ready word. The
# fused-finish seam polls the mirror (L2), and falls back to the host segment; there is no graph edge into the seam.
# A mirror counts only with this call's tag, and the tag is copied before the rows, so any early pass of the wait
# (a release below) can cost a fallback, never a wrong row. A dead RTX: the seam's timeout sets the word itself, so
# the chain, and the join at the forward's end, complete. The chains queue on one stream.
P3_OFF_DONE, P3_OFF_READY, P3_OFF_CAP, MIRROR_READY = 128, 1152, 1160, 136
P3_CAP_READY = 0x52454459  # "READ": the sidecar sets it when its RTX kernel writes the host ready word
P3_OFF_OUT = ((4096 + 64 * 128 + 64 * 5120 + 64 * 160 + 4095) // 4096) * 4096
_COPY: dict = {}


def _cuda_drv():
    if not _CUDA_DRV:
        _CUDA_DRV.append(ctypes.CDLL("libcuda.so.1"))
    return _CUDA_DRV[0]


def _cu_check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"w4: {what} returned CUresult {rc}")


def _copy_state(p3, device):
    """The mirror and the chain stream, once COPY_AHEAD is on and the sidecar advertises the ready word; else None."""
    if not COPY_AHEAD:
        return None
    st = _COPY.get(device.index)
    if st is None:
        if torch.cuda.is_current_stream_capturing():  # set up in an eager call (the profile run), never in a capture
            return None
        cap = ctypes.c_uint32()
        cu = _cuda_drv()
        _cu_check(cu.cuMemcpyDtoH_v2(ctypes.byref(cap), ctypes.c_uint64(p3.seg.dptr + P3_OFF_CAP), ctypes.c_size_t(4)),
                  "cuMemcpyDtoH (peer3 capability)")
        if cap.value != P3_CAP_READY:
            _log(f"COPY_AHEAD off: the sidecar does not advertise the host ready word (cap {cap.value:#x})")
            st = _COPY[device.index] = {"on": False}
            return None
        mirror = torch.zeros(P3_OFF_OUT + 64 * HIDDEN * 2, dtype=torch.uint8, device=device)
        st = _COPY[device.index] = {"on": True, "mirror": mirror, "stream": torch.cuda.Stream(device),
                                    "pending": False}
        _log("COPY_AHEAD on: the RTX's rows reach a device mirror during MegaMoE (copy engine)")
    return st if st["on"] else None


def _copy_ahead(st, p3, T: int) -> None:
    """Fork the chain after the router's publish (current stream). Graph-capturable: memop and memcpy nodes."""
    cu = _cuda_drv()
    cs = st["stream"]
    cs.wait_stream(torch.cuda.current_stream())
    h = ctypes.c_void_p(cs.cuda_stream)
    seg, mir = p3.seg.dptr, st["mirror"].data_ptr()
    _cu_check(cu.cuStreamWaitValue32_v2(h, ctypes.c_uint64(seg + P3_OFF_READY), ctypes.c_uint32(1),
                                        ctypes.c_uint(0)), "cuStreamWaitValue32")  # GEQ
    _cu_check(cu.cuMemcpyAsync(ctypes.c_uint64(mir + P3_OFF_DONE), ctypes.c_uint64(seg + P3_OFF_DONE),
                               ctypes.c_size_t(8), h), "cuMemcpyAsync (tag)")  # the tag first: see p3_wait_ack
    _cu_check(cu.cuMemcpyAsync(ctypes.c_uint64(mir + P3_OFF_OUT), ctypes.c_uint64(seg + P3_OFF_OUT),
                               ctypes.c_size_t(T * HIDDEN * 2), h), "cuMemcpyAsync (rows)")
    _cu_check(cu.cuStreamWriteValue32_v2(h, ctypes.c_uint64(mir + MIRROR_READY), ctypes.c_uint32(1),
                                         ctypes.c_uint(0)), "cuStreamWriteValue32 (mirror ready)")
    st["pending"] = True
    st["ready_word"] = seg + P3_OFF_READY


def join_copy() -> None:
    """Join the copy-ahead chains into the current stream (they must be joined before a capture ends)."""
    for st in _COPY.values():
        if st.get("on") and st["pending"]:
            torch.cuda.current_stream().wait_stream(st["stream"])
            st["pending"] = False


def release_join_copy() -> None:
    """Before a capture segment ends inside a forward (vLLM's breakable PIECEWISE capture breaks at eager ops, e.g.
    Engram's prefetch finish): release the pending chains' waits on the host ready word, then join them into the
    capturing stream. Their consuming seam may run in a later segment, so without the release a dead RTX would hold
    this segment forever; an early pass only costs that seam a fallback to the host rows (see _copy_ahead)."""
    cur = torch.cuda.current_stream()
    for st in _COPY.values():
        if st.get("on") and st["pending"]:
            _cu_check(_cuda_drv().cuStreamWriteValue32_v2(ctypes.c_void_p(cur.cuda_stream),
                                                          ctypes.c_uint64(st["ready_word"]), ctypes.c_uint32(1),
                                                          ctypes.c_uint(0)), "cuStreamWriteValue32 (release)")
            cur.wait_stream(st["stream"])
            st["pending"] = False


def _finish_now(f) -> None:
    """The finish as its own launch (peer3's kernel; as a PDL dependent where peer3 v3.1 offers it)."""
    m = sys.modules["fastpath_peer3"]
    p3 = f["p3"]
    cm = f.get("combine")
    if cm:  # MegaMoE skipped its combine for this call: W6's standalone combine writes y first (no fold)
        sys.modules["fastpath_megamoe"].combine_into(f["y"], cm["symm"], f["y"].shape[0], layout=cm["layout"])
    extra = (m.PDL,) if hasattr(m, "PDL") else ()
    p3.ext.finish(f["y"], f["has"], p3.seg.dptr, p3.state, p3.stats, m.SPINS, f["layer"], *extra)
    if f.get("mirror"):  # this call forked a copy chain the stock finish does not consume: release its wait (a dead
        cu = _cuda_drv()  # RTX would otherwise hold the chain, and the join at the forward's end, forever)
        _cu_check(cu.cuStreamWriteValue32_v2(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                                             ctypes.c_uint64(p3.seg.dptr + P3_OFF_READY), ctypes.c_uint32(1),
                                             ctypes.c_uint(0)), "cuStreamWriteValue32 (release)")


# The latest attention-side layer input and its MXFP8 form (MXFP8_OUT); the entry holds the layer input, so its
# address cannot be reused while the entry lives. Cleared at the next boundary and at every MoE call.
_PREQ: list[dict] = []


def prequantized(x):
    """x (the attention layer input the last attention-side seam wrote, any view of its [T, 5120] storage) as a
    vLLM QuantizedActivation (kMxfp8Dynamic; data = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)[0], scale =
    its 128x4-swizzled ue8m0 scales), or None."""
    if not _PREQ or not isinstance(x, torch.Tensor):
        return None
    e = _PREQ[-1]
    li = e["x"]
    if x.data_ptr() != li.data_ptr() or x.numel() != li.numel() or x.dtype != li.dtype:
        return None
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

    return QuantizedActivation(e["q"], e["sf"], x.dtype, x.shape, kMxfp8Dynamic)


def _make_fi_quantize(orig):
    def mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False, alignment=0):
        if _PREQ and is_sf_swizzled_layout and alignment == 0 and isinstance(x, torch.Tensor) and x.dim() == 2:
            e = _PREQ[-1]
            li = e["x"]
            if x.data_ptr() == li.data_ptr() and tuple(x.shape) == tuple(li.shape):
                return e["q"], e["sf"]
        return orig(x, is_sf_swizzled_layout, alignment)

    mxfp8_e4m3_quantize.__w4_boundary__ = True
    return mxfp8_e4m3_quantize


def _mega_moe_call(dg, *args, **kwargs) -> None:
    """dg.fp8_fp4_mega_moe; under MIX_OFFPATH with MEGA_SMS > 0 at kNumSMs = MEGA_SMS (the global SM count restored
    after the call)."""
    if not MIX_OFFPATH or MEGA_SMS <= 0:
        dg.fp8_fp4_mega_moe(*args, **kwargs)
        return
    prev = dg.get_num_sms()
    dg.set_num_sms(min(MEGA_SMS, prev))
    try:
        dg.fp8_fp4_mega_moe(*args, **kwargs)
    finally:
        dg.set_num_sms(prev)


def _join_for_moe(moe) -> None:
    """Before a stock MoE call: join the pending mix, except our own T > 16 mix when the next consumer of its
    coefficients is the next layer's boundary (which joins it): the MoE call (gate, send, MegaMoE) needs none of them.
    A deferred FFN-side mix is launched the side-stream way first."""
    _flush_deferred_mix()
    if _PENDING and _PENDING[-1][1]:
        st = getattr(getattr(moe, "experts", None), "_mp", None)
        if st is not None and int(st.idx) in _FUSE_LAYERS:
            return
    join_pending()


def flush_finish() -> None:
    while _PENDING_FINISH:
        _finish_now(_PENDING_FINISH.pop())


# ------------------------------------------------------------------------------------------------ boundary
def _combine_fields(cm) -> dict:
    """SeamParams' combine fields for a deferred finish whose MegaMoE call skipped its combine (COMBINE), else none."""
    if not cm:
        return {}
    base, slot_stride, tok_stride = cm["layout"]
    return dict(cmb_slots=ctypes.c_void_p(base), cmb_slot_stride=slot_stride, cmb_tok_stride=tok_stride,
                cmb_topk=_p(cm["topk"]))


_C11: dict = {}  # symm buffer data_ptr -> (base, slot stride, token stride), read once in an eager call
_COMBINE_OFF = 0
_COMBINE_T_SKIP = 0


def _combine_off(why: str) -> None:
    """Count and log a combine path that cannot run; never silent."""
    global _COMBINE_OFF
    _COMBINE_OFF += 1
    if _COMBINE_OFF == 1 or _COMBINE_OFF % 1024 == 0:
        _log(f"COMBINE off ({why}); count {_COMBINE_OFF}")


def _combine_t_skip(T: int) -> None:
    """Count and log the deliberate T cap; never silent."""
    global _COMBINE_T_SKIP
    _COMBINE_T_SKIP += 1
    if _COMBINE_T_SKIP == 1 or _COMBINE_T_SKIP % 1024 == 0:
        _log(f"COMBINE off: T={T} > COMBINE_MAX_T={COMBINE_MAX_T}; count {_COMBINE_T_SKIP}")


def _c11_for(symm):
    """v11's combine layout for this symm buffer once known (None: no v11, or no eager call has exported it yet)."""
    if not COMBINE:
        return None
    mm = sys.modules.get("fastpath_megamoe")
    if mm is None:
        _combine_off("fastpath_megamoe not loaded")
        return None
    if not hasattr(mm, "c11_layout"):
        _combine_off("no c11_layout")
        return None
    if not getattr(mm, "ACTIVE", True):
        _combine_off("v11 inactive")
        return None
    key = symm.buffer.data_ptr()
    lay = _C11.get(key)
    if lay is None and not torch.cuda.is_current_stream_capturing():
        lay = mm.c11_layout(symm)
        if lay is not None:
            _C11[key] = lay
            _log(f"COMBINE on: the post-MoE seam sums MegaMoE's combine slots (v11 layout {lay[1]} / {lay[2]} B)")
    return lay


def seam(x, residual, post, comb, pre, norm_w, *, capture_aux=False, mxfp8=False, cold=None, out=None, p3fin=None,
         dbg=None, post_only=False):
    """[cold add] + mHC post + collapse + RMSNorm in one launch on the current stream.

    Returns (residual_new, layer_input, aux | empty, (fp8 values, FlashInfer 128x4 swizzled ue8m0 scales) | None).
    With mxfp8 the layer input is also written as FlashInfer's mxfp8_quantize(is_sf_swizzled_layout=True) would.
    post_only: [cold add] + mHC post only (the kernel stops after the streams; layer_input is None)."""
    T = x.shape[0]
    dev = x.device
    res_out = out if out is not None else torch.empty_like(residual)
    li = None if post_only else torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=dev)
    mxfp8 = mxfp8 and not post_only
    aux = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=dev) if capture_aux else residual.new_empty(0, HIDDEN)
    q = sf = None
    if mxfp8:  # persistent per-T buffers: the scales' padding rows were zeroed once, the kernel skips them
        q, sf = _scratch(dev).mx[T - 1]
    ch, cp, co, cr = cold if cold is not None else (None,) * 4
    f = p3fin or {}
    if p3fin is not None:
        ch = f["has"]
    prm = SeamParams(x=_p(x), residual=_p(residual), post=_p(post), comb=_p(comb), pre=_p(pre), norm_w=_p(norm_w),
                     cold_has=_p(ch), cold_pos=_p(cp), cold_out=_p(co), cold_rows=_p(cr), residual_out=_p(res_out),
                     aux_out=_p(aux) if capture_aux else None, layer_input=_p(li), fi_q=_p(q), fi_sf_swz=_p(sf),
                     shared_block_m=0, T=T,
                     p3_seg=ctypes.c_void_p(f["seg"]) if p3fin else None, p3_state=_p(f.get("state")),
                     p3_stats=_p(f.get("stats")), p3_stats_words=f["stats"].numel() if p3fin else 0,
                     p3_layer=f.get("layer", 0), p3_spins=f.get("spins", 0), p3_y=_p(x) if p3fin else None,
                     p3_mirror=ctypes.c_void_p(f["mirror"]) if f.get("mirror") else None,
                     sf_pad_ready=int(mxfp8), dbg=_p(dbg), **_combine_fields(f.get("combine")))
    _check(lib().w4_seam_launch(ctypes.byref(prm), CLUSTER, GSIZE, 1, _stream()), "w4 seam")
    return res_out, li, aux, (None if q is None else (q, sf))


def mix(rn, fn_img, scale, base, stream=None, dbg=None, pdl=False, out=None, prefetch=None):
    """Next-sublayer post / comb / pre from residual_new (DeepGEMM tf32 GEMM + TileLang stats, bit-exact).
    pdl: launch as the PDL dependent of the router just launched with trigger_late on the current stream (M3).
    out: preallocated (post [T, 4], comb [T, 16], pre [T, 4]).
    prefetch: an hc_fn image (mix_fn_image) the CTAs pull into L2 at their start (the next mix's), or None."""
    T = rn.shape[0]
    dev = rn.device
    s = _scratch(dev)
    if out is None:
        out = (torch.empty(T, HC, dtype=torch.float32, device=dev),
               torch.empty(T, HC * HC, dtype=torch.float32, device=dev),
               torch.empty(T, HC, dtype=torch.float32, device=dev))
    post, comb, pre = out
    prm = MixParams(rn=_p(rn), fn_img=_p(fn_img), scale=_p(scale), base=_p(base), mul=_p(s.mul), sqr=_p(s.sqr),
                    post_out=_p(post), comb_out=_p(comb), pre_out=_p(pre), counters=_p(s.mix_counters), T=T,
                    pdl_mode=(2 if MIX_EARLY else 1) if pdl else 0, l2_prefetch=_p(prefetch),
                    l2_prefetch_bytes=0 if prefetch is None else prefetch.numel() * prefetch.element_size(),
                    dbg=_p(dbg))
    _check(lib().w4_mix_launch(ctypes.byref(prm), int(pdl), _stream(stream)), "w4 mix")
    return post, comb, pre


# An FFN-side mix deferred to the MoE call (MIX_PDL): its inputs, outputs and the mHC side stream of the boundary.
_DEFERRED_MIX: list[dict] = []


def _flush_deferred_mix(join: bool = False) -> None:
    """The deferred FFN-side mix's consumer is not our router: launch it the side-stream way (left pending, or joined
    at once with join)."""
    if not _DEFERRED_MIX:
        return
    d = _DEFERRED_MIX.pop()
    st = d["stream"]
    st.wait_stream(torch.cuda.current_stream())
    mix(d["rn"], d["img"], d["scale"], d["base"], stream=st, out=d["out"])
    for t in (d["rn"], *d["out"]):
        t.record_stream(st)
    _set_pending(st, False)
    if join:
        join_pending()


_CUDA_DRV: list = []


def _set_capture_deps(dst: torch.cuda.Stream, src: torch.cuda.Stream) -> None:
    """In a stream capture: dst's capture dependency set := src's (cuStreamUpdateCaptureDependencies SET; a stream
    wait would merge, keeping dst's old tail). A no-op outside a capture."""
    if not torch.cuda.is_current_stream_capturing():
        return
    if not _CUDA_DRV:
        _CUDA_DRV.append(ctypes.CDLL("libcuda.so.1"))
    cu = _CUDA_DRV[0]
    status, cid, graph = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_void_p()
    deps, n = ctypes.POINTER(ctypes.c_void_p)(), ctypes.c_size_t()
    rc = cu.cuStreamGetCaptureInfo_v2(ctypes.c_void_p(src.cuda_stream), ctypes.byref(status), ctypes.byref(cid),
                                      ctypes.byref(graph), ctypes.byref(deps), ctypes.byref(n))
    if rc != 0:
        raise RuntimeError(f"w4: cuStreamGetCaptureInfo_v2 returned CUresult {rc}")
    if status.value != 1:  # CU_STREAM_CAPTURE_STATUS_ACTIVE
        return
    nodes = (ctypes.c_void_p * n.value)(*[deps[i] for i in range(n.value)])
    rc = cu.cuStreamUpdateCaptureDependencies(ctypes.c_void_p(dst.cuda_stream), nodes, n, 1)  # 1: SET
    if rc != 0:
        raise RuntimeError(f"w4: cuStreamUpdateCaptureDependencies returned CUresult {rc}")


def _launch_deferred_mix() -> None:
    """Right after the router (launched with trigger_late): the deferred FFN-side mix as its PDL dependent.

    Then the graph topology is put back to the side-stream one's shape (bench/seam_pdl.py: without this the FFN-side
    seam of every layer loses its early PDL launch, as in the ab-mixv3-1021 trace). The mHC stream's tail is still the
    attention-side mix, which vLLM joined before this layer's FFN boundary: its pending entry is dropped (the MoE
    call's join would add an edge from it into MegaMoE), and the mHC stream's capture dependencies are SET to the
    main stream's, so vLLM's end-of-layer wait on it adds no edge from that mix into the next layer either."""
    d = _DEFERRED_MIX.pop()
    mix(d["rn"], d["img"], d["scale"], d["base"], pdl=True, out=d["out"])
    if _PENDING and not _PENDING[-1][1] and _PENDING[-1][0].cuda_stream == d["stream"].cuda_stream:
        _PENDING.clear()
    _set_capture_deps(d["stream"], torch.cuda.current_stream())


def _seam_eligible(x, residual, post_mix, comb_res_mix, fn, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                   hc_post_mult_value, sinkhorn_repeat, pre_mix, norm_weight, norm_eps, stream, reduce_results) -> bool:
    if stream is None or reduce_results or not isinstance(x, torch.Tensor) or pre_mix is None or norm_weight is None:
        return False
    T = x.shape[0]
    return (0 < T <= SEAM_MAX_T and fn.data_ptr() in _IMAGES and rms_eps == 1e-20 and norm_eps == 1e-20
            and hc_pre_eps == 1e-6 and hc_sinkhorn_eps == 1e-6 and hc_post_mult_value == 2.0 and sinkhorn_repeat == 20
            and x.dtype == torch.bfloat16 and x.is_contiguous() and tuple(x.shape) == (T, HIDDEN)
            and residual.dtype == torch.bfloat16 and residual.is_contiguous() and tuple(residual.shape) == (T, HC, HIDDEN)
            and post_mix.is_contiguous() and post_mix.numel() == T * HC and comb_res_mix.is_contiguous()
            and comb_res_mix.numel() == T * HC * HC and pre_mix.is_contiguous() and pre_mix.numel() == T * HC
            and norm_weight.dtype == torch.bfloat16 and norm_weight.numel() == HIDDEN)


def mega(x, residual, post, comb, pre, norm_w, fn, scale, base, *, hc_norm_eps, hc_pre_eps, hc_post_scale,
         sinkhorn_eps, sinkhorn_iters, rms_eps, side=None, prefetch=None):
    """T <= 64: DeepGEMM Mega mHC (shifted, bf16 out) bit for bit: mega_seam on the current stream
    (residual_new + layer input), mega_mix (next post / comb / pre) on `side` (left pending) or inline.
    prefetch: a tensor (the next kernel's weights) the seam pulls into L2 at its start, or None."""
    T = x.shape[0]
    dev = x.device
    s = _scratch(dev)
    res_out = torch.empty_like(residual)
    y = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=dev)
    post_o = torch.empty(T, HC, dtype=torch.float32, device=dev)
    comb_o = torch.empty(T, HC * HC, dtype=torch.float32, device=dev)
    pre_o = torch.empty(T, HC, dtype=torch.float32, device=dev)
    prm = MegaParams(x=_p(x), residual=_p(residual), post=_p(post), comb=_p(comb), pre=_p(pre), norm_w=_p(norm_w),
                     fn=_p(fn), scale=_p(scale), base=_p(base), residual_out=_p(res_out), y_out=_p(y),
                     post_out=_p(post_o), comb_out=_p(comb_o), pre_out=_p(pre_o), gemm_part=_p(s.mega_gemm),
                     hc_part=_p(s.mega_hc), counters=_p(s.mega_counters), hc_norm_eps=hc_norm_eps,
                     hc_pre_eps=hc_pre_eps, hc_post_scale=hc_post_scale, sinkhorn_eps=sinkhorn_eps, rms_eps=rms_eps,
                     rms_scale=1.0, sinkhorn_iters=sinkhorn_iters, T=T, l2_prefetch=_p(prefetch),
                     l2_prefetch_bytes=0 if prefetch is None else prefetch.numel() * prefetch.element_size())
    if side is not None and MEGA_MIX_FIRST:  # the mix reads only the boundary's inputs: start it beside the seam
        side.wait_stream(torch.cuda.current_stream())
        _check(lib().w4_mega_mix_launch(ctypes.byref(prm), 0, _stream(side)), "w4 mega mix")
    _check(lib().w4_mega_seam_launch(ctypes.byref(prm), 1, _stream()), "w4 mega seam")
    if side is None:
        _check(lib().w4_mega_mix_launch(ctypes.byref(prm), 1, _stream()), "w4 mega mix")
    else:
        if not MEGA_MIX_FIRST:
            side.wait_stream(torch.cuda.current_stream())
            _check(lib().w4_mega_mix_launch(ctypes.byref(prm), 0, _stream(side)), "w4 mega mix")
        for t in (x, residual, post, comb, pre, post_o, comb_o, pre_o):
            t.record_stream(side)
        _set_pending(side, True)
    return res_out, post_o, comb_o, y, pre_o


def _entry_eligible(residual, fn, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                    pre_mix, x, norm_weight, norm_eps) -> bool:
    """The drafter's first attention side: no carried pre-mix (TileLang collapses stream 0), no projection input."""
    if not (DRAFT_ON and SEAM_ON and pre_mix is None and x is None and norm_weight is not None
            and isinstance(residual, torch.Tensor) and fn.data_ptr() in _DRAFT_ENTRY):
        return False
    T = residual.shape[0]
    return (0 < T <= MEGA_MAX_T and rms_eps == 1e-20 and norm_eps == 1e-20 and hc_pre_eps == 1e-6
            and hc_sinkhorn_eps == 1e-6 and hc_post_mult_value == 2.0 and sinkhorn_repeat == 20
            and residual.dtype == torch.bfloat16 and residual.is_contiguous()
            and tuple(residual.shape) == (T, HC, HIDDEN) and norm_weight.dtype == torch.bfloat16
            and norm_weight.numel() == HIDDEN and norm_weight.is_contiguous())


def draft_entry(residual, fn, hc_scale, hc_base, norm_weight):
    """mhc_pre_delayed_tilelang(residual, fn, ..., pre_mix=None) bit for bit: the layer input (RMSNorm of stream 0,
    TileLang's collapse with the one-hot pre) from the seam on the current stream, the next post / comb / pre from the
    mix on our side stream beside it (left pending: joined at the next boundary or MoE call), or inline in piecewise
    graphs. Returns (post [T, 4, 1], comb [T, 4, 4], layer input, pre)."""
    from vllm.forward_context import in_piecewise_cudagraph

    T = residual.shape[0]
    dev = residual.device
    x0, post0, comb0, pre0 = _entry_consts(dev)
    img = _DRAFT_ENTRY[fn.data_ptr()]
    rn = residual.view(T, HC * HIDDEN)
    side = None if in_piecewise_cudagraph() else _scratch(dev).mega_side
    if side is not None:  # the mix reads only the residual: start it beside the seam
        side.wait_stream(torch.cuda.current_stream())
        post, comb, pre = mix(rn, img, hc_scale, hc_base, stream=side)
        for t in (residual, post, comb, pre):
            t.record_stream(side)
        _set_pending(side, True)
    _, li, _, _ = seam(x0[:T], residual, post0[:T], comb0[:T], pre0[:T], norm_weight)
    if side is None:
        post, comb, pre = mix(rn, img, hc_scale, hc_base)
    return post.view(T, HC, 1), comb.view(T, HC, HC), li, pre


def _post_eligible(x, residual, post_mix, comb_res_mix) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    T = x.shape[0]
    return (0 < T <= SEAM_MAX_T and x.dtype == torch.bfloat16 and x.is_contiguous() and tuple(x.shape) == (T, HIDDEN)
            and residual.dtype == torch.bfloat16 and residual.is_contiguous()
            and tuple(residual.shape) == (T, HC, HIDDEN) and post_mix.dtype == torch.float32
            and post_mix.is_contiguous() and post_mix.numel() == T * HC and comb_res_mix.dtype == torch.float32
            and comb_res_mix.is_contiguous() and comb_res_mix.numel() == T * HC * HC)


def _make_post(orig):
    """model.mhc_post_tilelang (the post before an Engram injection, and the final post): with a deferred peer-v3
    finish of x pending (FUSE_FINISH, the MoE layer feeding a post-only boundary), the seam's post-only mode does
    finish + post in one launch, bit for bit; otherwise the finish (if any) runs first and the stock post follows."""
    def mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix, *args, **kwargs):
        _flush_deferred_mix(join=True)  # the mixes this post reads (a deferred FFN-side mix, our T > 16 mix)
        join_pending(ours_only=True)
        _PREQ.clear()
        fin = _PENDING_FINISH.pop() if _PENDING_FINISH else None
        if fin is None:
            return orig(x, residual, post_layer_mix, comb_res_mix, *args, **kwargs)
        if (args or kwargs or not _post_eligible(x, residual, post_layer_mix, comb_res_mix)
                or x.data_ptr() != fin["y"].data_ptr() or x.shape != fin["y"].shape):
            _finish_now(fin)
            return orig(x, residual, post_layer_mix, comb_res_mix, *args, **kwargs)
        m = sys.modules["fastpath_peer3"]
        p3 = fin["p3"]
        p3fin = dict(seg=p3.seg.dptr, state=p3.state, stats=p3.stats, layer=fin["layer"], spins=m.SPINS,
                     has=fin["has"], mirror=fin.get("mirror"), combine=fin.get("combine"))
        T = x.shape[0]
        post = post_layer_mix.reshape(T, HC)
        # (pre / norm_w only need to be valid addresses: the post-only kernel stops before the collapse uses them)
        res_new, _, _, _ = seam(x, residual, post, comb_res_mix.reshape(T, HC * HC), post, x, p3fin=p3fin,
                                post_only=True)
        return res_new

    mhc_post_tilelang.__w4_boundary__ = True
    return mhc_post_tilelang


def _make_pre_overlap(orig):
    """model.mhc_pre_delayed_overlap (an Engram layer's attention entry, on the injected streams, with the carried
    pre-mix; ENGRAM_PRE): bit for bit, the layer input from the seam with post = 0 and comb = I (the streams pass
    through unchanged; TileLang's collapse with the carried pre + RMSNorm) on the current stream, and the next post /
    comb / pre from the mix on the given mHC stream (left pending; vLLM joins that stream before the FFN boundary)."""
    def mhc_pre_delayed_overlap(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                                hc_post_mult_value, sinkhorn_repeat, pre_mix=None, x=None, norm_weight=None,
                                norm_eps=1e-6, *, stream, layer_input=None):
        T = residual.shape[0] if isinstance(residual, torch.Tensor) else 0
        if not (ENGRAM_PRE and SEAM_ON and x is None and layer_input is None and pre_mix is not None
                and stream is not None and 0 < T <= SEAM_MAX_T and fn.data_ptr() in _IMAGES
                and norm_weight is not None and norm_weight.dtype == torch.bfloat16
                and norm_weight.numel() == HIDDEN and rms_eps == 1e-20 and norm_eps == 1e-20 and hc_pre_eps == 1e-6
                and hc_sinkhorn_eps == 1e-6 and hc_post_mult_value == 2.0 and sinkhorn_repeat == 20
                and residual.dtype == torch.bfloat16 and residual.is_contiguous()
                and tuple(residual.shape) == (T, HC, HIDDEN) and pre_mix.is_contiguous()
                and pre_mix.numel() == T * HC):
            return orig(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                        sinkhorn_repeat, pre_mix=pre_mix, x=x, norm_weight=norm_weight, norm_eps=norm_eps,
                        stream=stream, layer_input=layer_input)
        _flush_deferred_mix(join=True)
        join_pending(ours_only=True)
        _PREQ.clear()
        dev = residual.device
        x0, post0, comb0, _ = _entry_consts(dev)
        rn = residual.view(T, HC * HIDDEN)
        stream.wait_stream(torch.cuda.current_stream())
        post, comb, pre = mix(rn, _IMAGES[fn.data_ptr()], hc_scale, hc_base, stream=stream)
        for t in (residual, post, comb, pre):
            t.record_stream(stream)
        _set_pending(stream, False)
        _, li, _, _ = seam(x0[:T], residual, post0[:T], comb0[:T], pre_mix.reshape(T, HC), norm_weight)
        return post.view(T, HC, 1), comb.view(T, HC, HC), li, pre

    mhc_pre_delayed_overlap.__w4_boundary__ = True
    return mhc_pre_delayed_overlap


def _make_pre_delayed(orig):
    def mhc_pre_delayed_tilelang(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                                 hc_post_mult_value, sinkhorn_repeat, pre_mix=None, x=None, norm_weight=None,
                                 norm_eps=1e-6):
        if _entry_eligible(residual, fn, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                           pre_mix, x, norm_weight, norm_eps):
            _flush_deferred_mix(join=True)
            join_pending(ours_only=True)
            _PREQ.clear()
            return draft_entry(residual, fn, hc_scale, hc_base, norm_weight)
        return orig(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                    sinkhorn_repeat, pre_mix=pre_mix, x=x, norm_weight=norm_weight, norm_eps=norm_eps)

    mhc_pre_delayed_tilelang.__w4_boundary__ = True
    return mhc_pre_delayed_tilelang


def head(x, residual, post, comb, pre, dbg=None):
    """DSparkDeepseekV4Model's tail, hc_collapse_triton(mhc_post_tilelang(x, residual, post, comb), pre), bit for
    bit in one launch: [T, H] bf16."""
    T = x.shape[0]
    out = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=x.device)
    prm = HeadParams(x=_p(x), residual=_p(residual), post=_p(post), comb=_p(comb), pre=_p(pre), out=_p(out), T=T,
                     dbg=_p(dbg))
    _check(lib().w4_head_launch(ctypes.byref(prm), 0, _stream()), "w4 head")
    return out


def _head_eligible(x, residual, post, comb, pre) -> bool:
    if not all(isinstance(t, torch.Tensor) for t in (x, residual, post, comb, pre)):
        return False
    T = x.shape[0]
    return (T > 0 and x.dtype == torch.bfloat16 and x.is_contiguous() and tuple(x.shape) == (T, HIDDEN)
            and residual.dtype == torch.bfloat16 and residual.is_contiguous()
            and tuple(residual.shape) == (T, HC, HIDDEN)
            and all(t.dtype == torch.float32 and t.is_contiguous() for t in (post, comb, pre))
            and post.numel() == T * HC and comb.numel() == T * HC * HC and pre.numel() == T * HC)


def _make_dspark_forward(module, orig):
    """DSparkDeepseekV4Model.forward with its tail (mhc_post_tilelang, then hc_collapse_triton) as one w4 head launch:
    for the duration of the call the module's two names resolve to a pair that defers the post to the collapse."""
    stock_post, stock_collapse = module.mhc_post_tilelang, module.hc_collapse_triton

    def forward(self, input_ids, positions, inputs_embeds=None):
        if not (DRAFT_ON and SEAM_ON):
            return orig(self, input_ids, positions, inputs_embeds)
        pending = []

        def post_deferred(x, residual, post_mix, res_mix):
            pending.append((x, residual, post_mix, res_mix))
            return x

        def collapse(h, pre_mix):
            x, residual, post_mix, res_mix = pending.pop()
            if h is x and _head_eligible(x, residual, post_mix, res_mix, pre_mix):
                join_pending(ours_only=True)
                return head(x, residual, post_mix, res_mix, pre_mix)
            return stock_collapse(stock_post(x, residual, post_mix, res_mix), pre_mix)

        module.mhc_post_tilelang, module.hc_collapse_triton = post_deferred, collapse
        try:
            return orig(self, input_ids, positions, inputs_embeds)
        finally:
            module.mhc_post_tilelang, module.hc_collapse_triton = stock_post, stock_collapse

    forward.__w4_boundary__ = True
    return forward


def _mega_eligible(orig, x, residual, post_mix, comb_res_mix, fn, hc_scale, hc_base, pre_mix, norm_weight,
                   capture_aux, stream, reduce_results) -> bool:
    if (not MEGA_ON or stream is not None or capture_aux or not isinstance(x, torch.Tensor) or pre_mix is None
            or norm_weight is None):
        return False
    T = x.shape[0]
    min_t = 0 if DRAFT_ON and fn.data_ptr() in _DRAFT_FNS else SEAM_MAX_T  # the drafter runs Mega mHC at every T
    if not min_t < T <= MEGA_MAX_T:
        return False
    if reduce_results:
        from vllm.distributed import get_tp_group

        if get_tp_group().world_size != 1:
            return False
    can = orig.__globals__.get("can_use_mega_mhc")
    return (callable(can) and can(x, residual, pre_mix, norm_weight, capture_aux)
            and x.dtype == torch.bfloat16 and x.is_contiguous() and tuple(x.shape) == (T, HIDDEN)
            and residual.dtype == torch.bfloat16 and residual.is_contiguous() and tuple(residual.shape) == (T, HC, HIDDEN)
            and post_mix.dtype == torch.float32 and post_mix.is_contiguous() and post_mix.numel() == T * HC
            and comb_res_mix.dtype == torch.float32 and comb_res_mix.is_contiguous()
            and comb_res_mix.numel() == T * HC * HC
            and pre_mix.dtype == torch.float32 and pre_mix.is_contiguous() and pre_mix.numel() == T * HC
            and norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous() and norm_weight.numel() == HIDDEN
            and fn.dtype == torch.float32 and fn.is_contiguous() and tuple(fn.shape) == (MIX, HC * HIDDEN)
            and hc_scale.dtype == torch.float32 and hc_scale.numel() == 3 and hc_scale.is_contiguous()
            and hc_base.dtype == torch.float32 and hc_base.numel() == MIX and hc_base.is_contiguous())


def _make_shifted_post_pre(orig):
    def mhc_shifted_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                             hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix=None, norm_weight=None,
                             norm_eps=1e-6, capture_aux=False, *, stream=None, reduce_results=False):
        _flush_deferred_mix(join=True)  # an FFN-side mix nobody launched yet feeds this boundary
        join_pending(ours_only=True)  # the T > 16 mix on our own side stream feeds this boundary
        _PREQ.clear()
        fin = _PENDING_FINISH.pop() if _PENDING_FINISH else None
        if fin is not None and not (
                isinstance(x, torch.Tensor) and x.data_ptr() == fin["y"].data_ptr() and x.shape == fin["y"].shape
                and _seam_eligible(x, residual, post_layer_mix, comb_res_mix, fn, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                                   hc_post_mult_value, sinkhorn_repeat, pre_mix, norm_weight, norm_eps, stream,
                                   reduce_results)):
            _finish_now(fin)
            fin = None
        if _mega_eligible(orig, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, pre_mix,
                          norm_weight, capture_aux, stream, reduce_results):
            from vllm.forward_context import in_piecewise_cudagraph

            T = x.shape[0]
            side = None if in_piecewise_cudagraph() else _scratch(x.device).mega_side
            res_new, post, comb, li, pre = mega(
                x, residual, post_layer_mix, comb_res_mix, pre_mix, norm_weight, fn, hc_scale, hc_base,
                hc_norm_eps=rms_eps, hc_pre_eps=hc_pre_eps, hc_post_scale=hc_post_mult_value,
                sinkhorn_eps=hc_sinkhorn_eps, sinkhorn_iters=sinkhorn_repeat, rms_eps=norm_eps, side=side,
                prefetch=_DRAFT_GATES.get(fn.data_ptr()) if DRAFT_ON and ROUTER_ON else None)
            return res_new, post.view(T, HC, 1), comb.view(T, HC, HC), li, pre, x.new_empty(0, x.shape[1])
        if not _seam_eligible(x, residual, post_layer_mix, comb_res_mix, fn, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                              hc_post_mult_value, sinkhorn_repeat, pre_mix, norm_weight, norm_eps, stream,
                              reduce_results):
            return orig(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix=pre_mix,
                        norm_weight=norm_weight, norm_eps=norm_eps, capture_aux=capture_aux, stream=stream,
                        reduce_results=reduce_results)
        T = x.shape[0]
        p3fin = None
        if fin is not None:
            m = sys.modules["fastpath_peer3"]
            p3 = fin["p3"]
            p3fin = dict(seg=p3.seg.dptr, state=p3.state, stats=p3.stats, layer=fin["layer"], spins=m.SPINS,
                         has=fin["has"], mirror=fin.get("mirror"), combine=fin.get("combine"))
        want_q = MXFP8_OUT and fn.data_ptr() in _ATTN_FNS
        res_new, li, aux, q = seam(x, residual, post_layer_mix, comb_res_mix, pre_mix, norm_weight,
                                   capture_aux=capture_aux, p3fin=p3fin, mxfp8=want_q)
        if q is not None:
            _PREQ.append(dict(x=li, q=q[0], sf=q[1]))
        img = _IMAGES[fn.data_ptr()]
        if MIX_PDL and ROUTER_ON and fn.data_ptr() not in _ATTN_FNS:  # FFN side: the MoE call launches the mix
            out = (torch.empty(T, HC, dtype=torch.float32, device=x.device),
                   torch.empty(T, HC * HC, dtype=torch.float32, device=x.device),
                   torch.empty(T, HC, dtype=torch.float32, device=x.device))
            _DEFERRED_MIX.append(dict(rn=res_new, img=img, scale=hc_scale, base=hc_base, out=out, stream=stream))
            post, comb, pre = out
            return res_new, post.view(T, HC, 1), comb.view(T, HC, HC), li, pre, aux
        main = torch.cuda.current_stream()
        stream.wait_stream(main)
        # the attention side's mix (beside the attention) pulls this layer's FFN-side image into L2 for the FFN mix
        post, comb, pre = mix(res_new, img, hc_scale, hc_base, stream=stream,
                              prefetch=_FFN_IMAGE_OF.get(fn.data_ptr()) if L2_PREFETCH else None)
        for t in (res_new, post, comb, pre):
            t.record_stream(stream)
        _set_pending(stream, False)
        return res_new, post.view(T, HC, 1), comb.view(T, HC, HC), li, pre, aux

    mhc_shifted_post_pre.__w4_boundary__ = True
    return mhc_shifted_post_pre


# ------------------------------------------------------------------------------------------------ router
def router(li, gate_w, bias, row_map, mega_idx, mega_w, *, bias_vl=None, input_ids=None, is_padding=None, layer=0,
           prep=None, peer=None, peer_pad=64, p3=None, image_mask=None, dbg=None, lean=None, trigger_late=False,
           mega_ctl=None):
    """Gate GEMM + top-6 + hot/cold split (+ MegaMoE staging, + a cold-row publish: peer v2 `peer`, or peer v3
    `p3` = (segment device pointer, peer3 state tensor)). T <= 16: one launch replicating ll_bf16_gemm + dsv4_topk;
    16 < T <= 64: two launches replicating DeepGEMM's bf16 Mega gate (image tokens from `image_mask`, the
    MegaGateRoutingMetadata mask). Returns (topk_ids int64, topk_w f32, hot_ids int64, has int8, pos int32, rows)."""
    if peer is not None and p3 is not None:
        raise ValueError("w4 router: one publish per layer call (peer v2 or v3, not both)")
    T = li.shape[0]
    dev = li.device
    s = _scratch(dev)
    topk_ids = torch.empty(T, TOPK, dtype=torch.int64, device=dev)
    topk_w = torch.empty(T, TOPK, dtype=torch.float32, device=dev)
    hot = torch.empty(T, TOPK, dtype=torch.int64, device=dev)
    has = torch.empty(T, dtype=torch.int8, device=dev)
    pos = torch.empty(T, dtype=torch.int32, device=dev)
    rows = torch.empty(1, dtype=torch.int32, device=dev)
    px, psf, pssf, sbm = prep if prep is not None else (None, None, None, 0)
    small = T <= ROUTER_MAX_T
    prm = RouterParams(li=_p(li), gate_w=_p(gate_w), bias=_p(bias), bias_vl=_p(bias_vl), input_ids=_p(input_ids),
                       row_map=_p(row_map), is_padding=_p(is_padding), routed_scale=1.5, T=T, layer=layer,
                       logits=None, cand_v=_p(s.box[0] if small else s.cand_v),
                       cand_w=_p(s.box[1] if small else s.cand_w), cand_i=_p(s.box[2] if small else s.cand_i),
                       sync=_p(s.router_sync), topk_ids=_p(topk_ids), topk_w=_p(topk_w),
                       hot_ids=_p(hot), mega_idx=_p(mega_idx), mega_w=_p(mega_w), cold_ids=None, has=_p(has),
                       pos=_p(pos), rows=_p(rows), prep_x=_p(px), prep_sf=_p(psf),
                       prep_sf_sm=psf.stride(0) if psf is not None else 0,
                       prep_sf_sk=psf.stride(1) if psf is not None else 0, prep_shared_sf=_p(pssf),
                       prep_ssf_sm=pssf.stride(0) if pssf is not None else 0,
                       prep_ssf_sk=pssf.stride(1) if pssf is not None else 0, shared_block_m=sbm,
                       cold_q=None, cold_sf=None, quant_cold=0,
                       peer_x=_p(peer.px) if peer else None, peer_xs=_p(peer.pxs) if peer else None,
                       peer_ids=_p(peer.pids) if peer else None, peer_w=_p(peer.pw) if peer else None,
                       peer_words=_p(peer.words) if peer else None, peer_header=_p(peer.header) if peer else None,
                       peer_seq=_p(peer.seq) if peer else None, peer_pad=peer_pad,
                       p3_seg=ctypes.c_void_p(p3[0]) if p3 else None, p3_state=_p(p3[1]) if p3 else None,
                       image_mask=_p(image_mask), mg_part=_p(s.mg_part), mg_count=_p(s.mg_count),
                       lean_fence=int(LEAN_FENCE if lean is None else lean), trigger_late=int(trigger_late),
                       mega_ctl=_p(mega_ctl), mega_ctl_val=1, dbg=_p(dbg))
    if T > ROUTER_MAX_T:
        _check(lib().w4_router_mega_launch(ctypes.byref(prm), 1, _stream()), "w4 router (mega gate)")
    else:
        _check(lib().w4_router_launch(ctypes.byref(prm), 1, _stream()), "w4 router")
    return topk_ids, topk_w, hot, has, pos, rows


def draft_router(li, gate_w, bias, topk_idx, topk_w, prep, *, bias_vl=None, image_mask=None, is_padding=None,
                 routed_scale=1.5, logits=None, dbg=None, pdl=False):
    """The DSpark drafter's MoE glue, 2 <= T <= 64, one launch: bf16_mega_gate (128 experts, top-3) and
    prepare_megamoe_inputs bit for bit, writing MegaMoE's staged rows: topk_idx / topk_w [T, 3] and
    prep = (x fp8 [T, H], x_sf int32 [T, H/128], shared_x_sf int32 or None, shared_block_m).
    Launched without PDL by default: resident early beside the seam, its 16-CTA cluster slows the side-stream mix
    (the FFN side's critical path) by ~2.5 us; the seam prefetches its gate into L2 instead (mega(prefetch=))."""
    T = li.shape[0]
    px, psf, pssf, sbm = prep
    prm = DraftRouterParams(li=_p(li), gate_w=_p(gate_w), bias=_p(bias), bias_vl=_p(bias_vl),
                            image_mask=_p(image_mask), is_padding=_p(is_padding), routed_scale=routed_scale, T=T,
                            topk_idx=_p(topk_idx), topk_w=_p(topk_w), prep_x=_p(px), prep_sf=_p(psf),
                            prep_sf_sm=psf.stride(0), prep_sf_sk=psf.stride(1), prep_shared_sf=_p(pssf),
                            prep_ssf_sm=pssf.stride(0) if pssf is not None else 0,
                            prep_ssf_sk=pssf.stride(1) if pssf is not None else 0, shared_block_m=sbm,
                            logits=_p(logits), dbg=_p(dbg))
    _check(lib().w4_draft_router_launch(ctypes.byref(prm), int(pdl), _stream()), "w4 draft router")


def _hook_globals(experts):
    """mega_peer_hook's module globals. peer3 wraps the hook's forward; the hook's finalize_weights stays."""
    cls = type(experts)
    for fn in (cls.forward, getattr(cls, "finalize_weights", None)):
        g = getattr(fn, "__globals__", {})
        if callable(g.get("_peer_tier2")) and callable(g.get("_fused2")):
            return g
    return None


def _peer3_installed(experts) -> bool:
    """W6's `peer3` component wraps DeepseekV4MegaMoEExperts.forward when it is installed."""
    return getattr(type(experts).forward, "__module__", "") == "fastpath_peer3"


_P3_CHECKED = False


def _peer3_handle(device):
    """peer3's process singleton, created outside graph capture (the eager profile run) and checked once against the
    segment offsets this library writes. None until it exists."""
    global _P3_CHECKED
    m = sys.modules.get("fastpath_peer3")
    if m is None:
        return None
    if m._state is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        m._peer3(device)
    h = m._state
    if not _P3_CHECKED:
        want = {"OFF_REC": lib().w4_abi(9), "OFF_XQ": lib().w4_abi(10), "OFF_XS": lib().w4_abi(11),
                "OFF_OUT": lib().w4_abi(14), "OFF_DONE": 128}
        got = {k: int(getattr(h.ext, k)) for k in want}
        if got != want or int(h.ext.MAX_TOKENS) < ROUTER_MAX_T:
            raise RuntimeError(f"peer3 segment layout {got} (max T {h.ext.MAX_TOKENS}) != w4 router's {want}")
        _P3_CHECKED = True
        _log(f"peer tier v3 publish from the router (segment offsets {got})")
    return h


def _moe_eligible(moe, hidden_states, input_ids, g, v3: bool) -> bool:
    exp = moe.experts
    st = getattr(exp, "_mp", None)
    T = hidden_states.shape[0]
    gate = moe.gate
    bias_vl = getattr(gate, "bias_vl", None)
    min_t = ROUTER_MIN_T if v3 or not _peer3_installed(exp) else 5
    return ((v3 or (g is not None and g.get("PEER2") and g.get("FUSED_SEND2"))) and st is not None
            and min_t <= T <= ROUTER_MAX_T and moe.use_mega_moe and not moe.use_fi_mega_moe
            and getattr(gate, "allow_ll_bf16_gemm", False) and T <= getattr(gate, "LL_BF16_MAX_TOKENS", 0)
            and gate.tid2eid is None and gate.e_score_correction_bias is not None
            and gate.e_score_correction_bias.dtype == torch.float32
            and moe.scoring_func == "sqrtsoftplus" and moe.renormalize and moe.n_activated_experts == TOPK
            and moe.routed_scaling_factor == 1.5 and tuple(gate.weight.shape) == (EXPERTS, HIDDEN)
            and gate.weight.dtype == torch.bfloat16 and exp.has_fused_shared_experts
            and exp.eplb_state.logical_to_physical_map is None and exp.capture_fn is None
            and hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous()
            and hidden_states.dim() == 2 and hidden_states.shape[1] == HIDDEN
            and (bias_vl is None or (input_ids is not None and input_ids.dtype == torch.int64
                                     and moe.image_sentinel_lo == IMAGE_SENTINEL_LO)))


def _symm_ok(symm) -> bool:
    """The router writes MegaMoE's symmetric-buffer rows the way _prepare_megamoe_inputs does: fp8 x, int32-packed
    ue8m0 x_sf / shared_l1_acts_sf (any strides), int64 topk_idx, fp32 topk_weights."""
    sf, ssf = symm.x_sf, symm.shared_l1_acts_sf
    return (symm.x.dtype == torch.float8_e4m3fn and symm.x.is_contiguous() and symm.x.shape[1] == HIDDEN
            and sf.dtype == torch.int32 and sf.dim() == 2 and sf.shape[1] == HIDDEN // 128
            and ssf is not None and ssf.dtype == torch.int32 and ssf.dim() == 2 and ssf.shape[1] == HIDDEN // 128
            and symm.topk_idx.dtype == torch.int64 and symm.topk_idx.is_contiguous()
            and symm.topk_weights.dtype == torch.float32 and symm.topk_weights.is_contiguous())


def _entry_consts(device: torch.device) -> tuple:
    c = _ENTRY_CONST.get(device.index)
    if c is None:
        x0 = torch.zeros(MEGA_MAX_T, HIDDEN, dtype=torch.bfloat16, device=device)
        post0 = torch.zeros(MEGA_MAX_T, HC, dtype=torch.float32, device=device)
        comb0 = torch.eye(HC, dtype=torch.float32, device=device).reshape(1, HC * HC).repeat(MEGA_MAX_T, 1).contiguous()
        pre0 = torch.zeros(MEGA_MAX_T, HC, dtype=torch.float32, device=device)
        pre0[:, 0] = 1.0
        c = _ENTRY_CONST[device.index] = (x0, post0, comb0, pre0)
    return c


def register_draft_layers(layers) -> int:
    """Mark the DSpark drafter's decoder layers: their stream-less mHC boundaries take w4 mega at every T <= 64, the
    first layer's attention side the w4 entry (seam collapse of stream 0 + side-stream mix), and their MoE calls the
    drafter router. Returns the number of layers."""
    layers = list(layers)
    if layers:
        fn = getattr(layers[0], "hc_attn_fn", None)
        if (isinstance(fn, torch.Tensor) and fn.dim() == 2 and tuple(fn.shape) == (MIX, HC * HIDDEN)
                and fn.dtype == torch.float32):
            _DRAFT_ENTRY[fn.data_ptr()] = mix_fn_image(fn.data)
            _entry_consts(fn.device)
    n = 0
    for layer in layers:
        for name in ("hc_attn_fn", "hc_ffn_fn"):
            fn = getattr(layer, name, None)
            if isinstance(fn, torch.Tensor):
                _DRAFT_FNS.add(fn.data_ptr())
                _DRAFT_REFS.append(fn)
        ffn = getattr(layer, "ffn", None)
        if ffn is not None:
            _DRAFT_MOES.add(id(ffn))
            _DRAFT_REFS.append(ffn)
            gate_w = getattr(getattr(ffn, "gate", None), "weight", None)
            fn = getattr(layer, "hc_ffn_fn", None)
            if isinstance(gate_w, torch.Tensor) and isinstance(fn, torch.Tensor) and gate_w.is_contiguous():
                _DRAFT_GATES[fn.data_ptr()] = gate_w
        n += 1
    return n


def _draft_eligible(moe, hidden_states, meta) -> bool:
    """A drafter MoE call that production routes with bf16_mega_gate (128 experts, top-3) and stages with
    prepare_megamoe_inputs into its own MegaMoE (no peer tier)."""
    exp, gate = moe.experts, moe.gate
    T = hidden_states.shape[0]
    bias = gate.e_score_correction_bias
    bias_vl = getattr(gate, "bias_vl", None)
    mask = None if meta is None else meta.image_token_mask
    if not (DRAFT_MIN_T <= T <= MEGA_MAX_T and meta is not None and moe.use_mega_moe and not moe.use_fi_mega_moe
            and getattr(exp, "_mp", None) is None and gate.tid2eid is None
            and tuple(gate.weight.shape) == (DRAFT_EXPERTS, HIDDEN) and gate.weight.dtype == torch.bfloat16
            and gate.weight.is_contiguous() and bias is not None and bias.dtype == torch.float32
            and bias.numel() == DRAFT_EXPERTS
            and (bias_vl is None or (bias_vl.dtype == torch.float32 and bias_vl.numel() == DRAFT_EXPERTS))
            and (mask is None) == (bias_vl is None)  # bf16_mega_gate takes the image bias and mask together
            and (mask is None or (mask.dtype == torch.bool and mask.dim() == 1 and mask.numel() >= T
                                  and mask.is_contiguous()))
            and moe.scoring_func == "sqrtsoftplus" and moe.renormalize and moe.n_activated_experts == DRAFT_TOPK
            and exp.has_fused_shared_experts and exp.eplb_state.logical_to_physical_map is None
            and exp.capture_fn is None and T <= exp.max_num_tokens
            and hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous()
            and hidden_states.dim() == 2 and hidden_states.shape[1] == HIDDEN):
        return False
    symm = exp.get_symm_buffer()
    return _symm_ok(symm) and symm.topk_idx.shape[1] == DRAFT_TOPK


def _draft_moe(moe, hidden_states, meta):
    """DeepseekV4MoE.forward of a drafter layer: w4 drafter router (gate + top-3 + MegaMoE staging, one launch)
    -> [join the pending mix] -> MegaMoE, as production's bf16_mega_gate -> DeepseekV4MegaMoEExperts.forward."""
    from vllm import envs
    from vllm.distributed import get_ep_group
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.utils.deep_gemm import _import_deep_gemm

    dg = _import_deep_gemm()
    exp, gate = moe.experts, moe.gate
    T = hidden_states.shape[0]
    symm = exp.get_symm_buffer()
    sbm = int(dg.get_block_m_for_mega_moe(get_ep_group().world_size, exp.num_experts,
                                          symm.num_max_tokens_per_rank, T, exp.top_k, "fp8xfp4"))
    is_padding = None
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        ip = get_forward_context().is_padding
        if ip is not None:
            is_padding = ip[:T].to(torch.bool).contiguous()
    mask = meta.image_token_mask
    bias_vl = getattr(gate, "bias_vl", None)
    draft_router(hidden_states, gate.weight, gate.e_score_correction_bias.data, symm.topk_idx[:T],
                 symm.topk_weights[:T], (symm.x, symm.x_sf, symm.shared_l1_acts_sf, sbm),
                 bias_vl=None if mask is None else bias_vl.data, image_mask=None if mask is None else mask[:T],
                 is_padding=is_padding, routed_scale=float(moe.routed_scaling_factor))
    join_pending()  # the FFN-side mix must not hold SMs when MegaMoE starts
    y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    dg.fp8_fp4_mega_moe(y, exp._transformed_l1_weights, exp._transformed_l2_weights, symm,
                        shared_l1_weights=exp._transformed_shared_l1_weights,
                        shared_l2_weights=exp._transformed_shared_l2_weights,
                        activation_clamp=float(moe.swiglu_limit) if moe.swiglu_limit is not None else None,
                        fast_math=True)
    return y.view(hidden_states.shape)


_FALLBACK_LOGGED: set = set()


def _log_routing(moe, reason: str) -> None:
    """Once per (reason): a T <= 16 MoE call that leaves the w4 router path, with its layer (boot-log diagnosis)."""
    if reason in _FALLBACK_LOGGED:
        return
    _FALLBACK_LOGGED.add(reason)
    st = getattr(getattr(moe, "experts", None), "_mp", None)
    _log(f"MoE layer {getattr(st, 'idx', '?')} falls back to the stock path at T <= {ROUTER_MAX_T}: {reason}")


def _make_moe_forward(orig):
    def forward(self, hidden_states, input_ids=None, mega_gate_metadata=None):
        flush_finish()  # never publish a call before the previous call's finish
        _PREQ.clear()
        if id(self) in _DRAFT_MOES:
            _flush_deferred_mix()
            if ROUTER_ON and DRAFT_ON and _draft_eligible(self, hidden_states, mega_gate_metadata):
                return _draft_moe(self, hidden_states, mega_gate_metadata)
            _join_for_moe(self)
            return orig(self, hidden_states, input_ids, mega_gate_metadata)
        g = _hook_globals(self.experts) if ROUTER_ON else None
        want_v3 = ROUTER_ON and PEER3_ON and _peer3_installed(self.experts)
        p3 = _peer3_handle(hidden_states.device) if want_v3 else None
        if want_v3 and (p3 is None or hidden_states.shape[0] > sys.modules["fastpath_peer3"].MAX_T):
            if hidden_states.shape[0] <= ROUTER_MAX_T:
                _log_routing(self, "peer3 installed but no peer3 state yet" if p3 is None else "T above peer3 MAX_T")
            _join_for_moe(self)  # peer3's own forward publishes this call (one path per width)
            return orig(self, hidden_states, input_ids, mega_gate_metadata)
        if not _moe_eligible(self, hidden_states, input_ids, g, p3 is not None):
            if ROUTER_MIN_T <= hidden_states.shape[0] <= ROUTER_MAX_T:
                fwd = type(self.experts).forward
                _log_routing(self, "not eligible (peer3 installed: %s, experts forward from %s, input_ids %s)" % (
                    want_v3, getattr(fwd, "__module__", "?"), "given" if input_ids is not None else "None"))
            _join_for_moe(self)
            return orig(self, hidden_states, input_ids, mega_gate_metadata)
        from vllm import envs
        from vllm.distributed import get_ep_group
        from vllm.forward_context import get_forward_context, is_forward_context_available
        from vllm.utils.deep_gemm import _import_deep_gemm

        dg = _import_deep_gemm()
        exp = self.experts
        T = hidden_states.shape[0]
        layer = exp._mp.idx
        if p3 is None:
            peer = g["_peer_tier2"](hidden_states.device)
            pf2 = g["_fused2"]()
            pad = sys.modules["peer_tier2"].SMALL_ROWS
        else:
            peer, pad = None, 64
        symm = exp.get_symm_buffer()
        if not _symm_ok(symm) or T > exp.max_num_tokens:
            _join_for_moe(self)
            return orig(self, hidden_states, input_ids, mega_gate_metadata)
        sbm = int(dg.get_block_m_for_mega_moe(get_ep_group().world_size, exp.num_experts,
                                              symm.num_max_tokens_per_rank, T, exp.top_k, "fp8xfp4"))
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            ip = get_forward_context().is_padding
            if ip is not None:
                is_padding = ip[:T].to(torch.bool).contiguous()
        bias_vl = getattr(self.gate, "bias_vl", None)
        fused = p3 is not None and FUSE_FINISH and layer in _FUSE_LAYERS
        if fused and T > COMBINE_MAX_T:
            _combine_t_skip(T)
            lay = None
        else:
            lay = _c11_for(symm) if fused else None  # the seam combines: the router sets v11's skip_combine

        _, _, _, has, pos, rows = router(
            hidden_states, self.gate.weight, self.gate.e_score_correction_bias.data, exp._mp.row_map,
            symm.topk_idx[:T], symm.topk_weights[:T], bias_vl=None if bias_vl is None else bias_vl.data,
            input_ids=input_ids if bias_vl is not None else None, is_padding=is_padding, layer=layer,
            prep=(symm.x, symm.x_sf, symm.shared_l1_acts_sf, sbm), peer=peer, peer_pad=pad,
            p3=None if p3 is None else (p3.seg.dptr, p3.state), trigger_late=bool(_DEFERRED_MIX),
            mega_ctl=sys.modules["fastpath_megamoe"].c11_skip(symm) if lay is not None else None)
        cst = (_copy_state(p3, hidden_states.device)
               if p3 is not None and FUSE_FINISH and layer in _FUSE_LAYERS else None)
        if cst is not None:
            _copy_ahead(cst, p3, T)  # beside MegaMoE: the RTX's rows into the device mirror
        if _DEFERRED_MIX:
            _launch_deferred_mix()  # PDL dependent of the router; it completes only after the router has
        if not MIX_OFFPATH:
            join_pending()  # a side-stream mix must not hold SMs when MegaMoE starts
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        _mega_moe_call(dg, y, exp._transformed_l1_weights, exp._transformed_l2_weights, symm,
                            shared_l1_weights=exp._transformed_shared_l1_weights,
                            shared_l2_weights=exp._transformed_shared_l2_weights,
                            activation_clamp=float(self.swiglu_limit) if self.swiglu_limit is not None else None,
                            fast_math=True)
        if p3 is None:
            pf2.finish2(peer, y, has, pos, rows)
        elif fused:
            _PENDING_FINISH.append(dict(y=y, has=has, layer=layer, p3=p3,
                                        mirror=None if cst is None else cst["mirror"].data_ptr(),
                                        combine=None if lay is None else dict(layout=lay, topk=symm.topk_idx[:T],
                                                                              symm=symm)))
        else:
            _finish_now(dict(y=y, has=has, layer=layer, p3=p3))
        return y

    forward.__w4_boundary__ = True
    return forward


# ------------------------------------------------------------------------------------------------ patches
@register("boundary", TARGET_MODEL)
def _patch_model(module) -> None:
    lib()  # fail at boot, not at capture, if the library or its ABI is wrong
    if SEAM_ON and not getattr(module.mhc_shifted_post_pre, "__w4_boundary__", False):
        module.mhc_shifted_post_pre = _make_shifted_post_pre(module.mhc_shifted_post_pre)
    if SEAM_ON and not getattr(module.mhc_pre_delayed_tilelang, "__w4_boundary__", False):
        module.mhc_pre_delayed_tilelang = _make_pre_delayed(module.mhc_pre_delayed_tilelang)
    if SEAM_ON and POST_FUSE and not getattr(module.mhc_post_tilelang, "__w4_boundary__", False):
        module.mhc_post_tilelang = _make_post(module.mhc_post_tilelang)
        _log(f"patched {TARGET_MODEL}.mhc_post_tilelang: post-only seam with the fused peer-v3 finish (POST_FUSE)")
    if SEAM_ON and ENGRAM_PRE and not getattr(module.mhc_pre_delayed_overlap, "__w4_boundary__", False):
        module.mhc_pre_delayed_overlap = _make_pre_overlap(module.mhc_pre_delayed_overlap)
        _log(f"patched {TARGET_MODEL}.mhc_pre_delayed_overlap: Engram attention entry on the seam + mix (ENGRAM_PRE)")
    model_cls = module.DeepseekV4Model
    orig_finalize = model_cls.finalize_mhc_broadcast_weights

    def finalize_mhc_broadcast_weights(self, _orig=orig_finalize):
        with bounded("DeepseekV4Model.finalize_mhc_broadcast_weights (mHC hc_fn images)", 300):
            _orig(self)
            if not SEAM_ON:
                return
            layers = [self.layers[i] for i in range(self.start_layer, self.end_layer)]
            nbytes = build_images(layers)
            for cur, nxt in zip(layers, layers[1:] + [None]):
                st = getattr(getattr(getattr(cur, "ffn", None), "experts", None), "_mp", None)
                if st is None:
                    continue
                if nxt is not None and getattr(nxt, "engram", None) is None:
                    _FUSE_LAYERS.add(int(st.idx))  # the next layer's mhc_shifted_post_pre
                elif POST_FUSE:
                    _FUSE_LAYERS.add(int(st.idx))  # the post-only boundary before an Engram injection / the final post
            if FUSE_FINISH:
                _log(f"peer-v3 finish fused into the next seam for MoE layers {sorted(_FUSE_LAYERS)}")
            _log(f"hc_fn tf32 fragment images for {len(layers)} layers: {nbytes} bytes ({nbytes / 2**20:.1f} MiB)")

    model_cls.finalize_mhc_broadcast_weights = finalize_mhc_broadcast_weights
    if COPY_AHEAD and not getattr(model_cls.forward, "__w4_boundary__", False):
        orig_forward = model_cls.forward

        def forward(self, *args, _orig=orig_forward, **kwargs):
            out = _orig(self, *args, **kwargs)
            join_copy()  # the copy-ahead chains of this forward (a capture must not end with them unjoined)
            return out

        forward.__w4_boundary__ = True
        model_cls.forward = forward
    _log(f"patched {TARGET_MODEL}.mhc_shifted_post_pre (seam {'on' if SEAM_ON else 'off'}, cluster {CLUSTER}, "
         f"T <= {SEAM_MAX_T}); images built in DeepseekV4Model.finalize_mhc_broadcast_weights")


@register("boundary", TARGET_BREAKABLE)
def _patch_breakable(module) -> None:
    """vLLM's breakable capture ends a stream capture at every eager break: the copy-ahead chains forked in the
    segment are released and joined first (a capture must not end with forked work unjoined)."""
    if not COPY_AHEAD:
        return
    cls = module.BreakableCUDAGraphCapture
    if getattr(cls._end_segment, "__w4_boundary__", False):
        return
    orig = cls._end_segment

    def _end_segment(self, _orig=orig):
        if getattr(self, "_capturing", False):
            release_join_copy()
        return _orig(self)

    _end_segment.__w4_boundary__ = True
    cls._end_segment = _end_segment
    _log(f"patched {TARGET_BREAKABLE}.BreakableCUDAGraphCapture._end_segment: copy-ahead chains joined at every break")


@register("boundary", TARGET_FI_LINEAR)
def _patch_fi_linear(module) -> None:
    if not MXFP8_OUT:
        return
    if not getattr(module.mxfp8_e4m3_quantize, "__w4_boundary__", False):
        module.mxfp8_e4m3_quantize = _make_fi_quantize(module.mxfp8_e4m3_quantize)
    _log(f"patched {TARGET_FI_LINEAR}.mxfp8_e4m3_quantize: the attention input comes pre-quantized from the seam")


@register("boundary", SPECULATOR_MODULE)
def _patch_speculator(module) -> None:
    """Register the DSpark drafter's layers once the speculator has loaded (and post-processed) the draft model."""
    cls = module.DSparkSpeculator
    if getattr(cls.load_draft_model, "__w4_boundary__", False):
        return
    orig_load = cls.load_draft_model

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = orig_load(self, target_model, target_attn_layer_names)
        if DRAFT_ON:
            n = register_draft_layers(getattr(getattr(model, "model", None), "layers", []))
            _log(f"DSpark drafter: {n} layers registered (w4 mega at T <= {MEGA_MAX_T}, drafter router at "
                 f"{DRAFT_MIN_T} <= T <= {MEGA_MAX_T}, router {'on' if ROUTER_ON else 'off'})")
        return model

    load_draft_model.__w4_boundary__ = True
    cls.load_draft_model = load_draft_model


@register("boundary", DSPARK_MODULE)
def _patch_dspark(module) -> None:
    """The drafter head (DSparkDeepseekV4Model.forward's tail) as one w4 launch."""
    cls = module.DSparkDeepseekV4Model
    if not SEAM_ON or getattr(cls.forward, "__w4_boundary__", False):
        return
    cls.forward = _make_dspark_forward(module, cls.forward)
    _log(f"patched {DSPARK_MODULE}.DSparkDeepseekV4Model.forward: the head (mhc_post + hc_collapse) in one launch")


@register("boundary", TARGET_V4)
def _patch_moe(module) -> None:
    cls = module.DeepseekV4MoE
    if not getattr(cls.forward, "__w4_boundary__", False):
        cls.forward = _make_moe_forward(cls.forward)
    mixs = (f"FFN-side mix as the router's PDL dependent (M3, capture deps SET, MegaMoE released "
            f"{'at the mix start' if MIX_EARLY else 'at the mix exit'})" if MIX_PDL else
            f"MIX_OFFPATH on: the FFN-side mix stays off MegaMoE's path"
            f"{f', MegaMoE at kNumSMs {MEGA_SMS}' if MEGA_SMS > 0 else ''}" if MIX_OFFPATH
            else "mix joined before MegaMoE")
    _log(f"patched {TARGET_V4}.DeepseekV4MoE.forward (router r1 candidate mailbox {'on' if ROUTER_ON else 'off'}, "
         f"{ROUTER_MIN_T} <= T <= {ROUTER_MAX_T}, {mixs}, "
         f"peer v3 {'on when peer3 is installed' if PEER3_ON else 'off'})")
