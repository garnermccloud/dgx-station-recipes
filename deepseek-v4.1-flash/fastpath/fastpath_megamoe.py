"""ds41 fastpath component "megamoe" (W6): a scheduling-only fork of DeepGEMM's fp8 x fp4 MegaMoE kernel.

What changes (two headers of the vendored DeepGEMM, nothing else; every tile's math and the combine order are stock, so
outputs are bit-identical to stock MegaMoE, checked over T = 1..128 by megamoe_fd_check.py):
  - fast dispatch (single rank, T <= 64): every SM counts the call's routes itself; no send-count atomics, grid syncs
    or NVLink barrier before the token pull;
  - fair first wave: a cluster's second task claim waits until every cluster has claimed one;
  - L1-first routed schedule (fast-dispatch calls): every cluster claims routed L1 tasks until none are left, then L2
    tasks in pool-block order, instead of alternating L2/L1 claims after a short warm-up (removes the last experts'
    L1 -> L2 dependency tail);
  - lane-parallel combine (single rank): every epilogue warp takes (token, 128-channel) items and loads all present
    slots at once instead of 6-12 warps walking the slots one TMA load at a time;
  - combine-in-seam (v11, single rank): a second control block (C11, fold control + 256 B) carries a per-call
    skip_combine word (written by its consumer before the launch: W4's seam) and the combine slot layout the kernel
    exports every call (c11_layout(symm)). With skip_combine = 1 (and the fold not enabled) the kernel stops after the L2
    epilogues, the epilogue grid sync and the cleanup: no combine, no y; the consumer sums the slots, or calls
    combine_into(y, symm, T) (libw6combine.so: the same lane-parallel combine as a standalone kernel). 0 = v5;
  - peer3 finish fold (v5, single rank, enabled per call): the peer tier v3 finish (wait for the RTX acknowledgement,
    y[t] = bf16(float(y[t]) + float(out[t])) for cold tokens, advance the call counter) runs inside the combine, so the
    call needs no finish kernel. A call enables the fold when its publisher writes has[t] into the fold control block
    (p3fold_has(symm)) instead of its own has buffer; calls without it use the v2 kernel. p3fold_setup() writes the
    block's static fields once (segment, call-counter state, wait statistics, spin bound, protocol offsets);
  - all-token L1 head and shared expert early (decode: single rank, fast dispatch, T <= 16): each routed expert's L1
    block reads the call's tokens directly (one block per expert, the original top-k slots and weights; absent routes
    never write a combine slot); the shared expert's L1 tasks have a static per-cluster owner and are published, with
    their weight loads, before the kernel's grid-dependency wait, so they overlap the previous kernel; the shared L2
    tasks fill the last routed wave instead of parking the first wave on its L1. Activations, dispatch, epilogues and
    peer reads still wait for the predecessor.
Measured on the GB300 (graph replay, quiet GPU, E=64, real-like routing): T=6 D=15/16/21 72.8/73.7/88.2 us stock ->
64.6/68.7/83.0 us; T=12 D=37 134.2 -> 129.1; T=16 D=43 152.7 -> 146.5.

How: when vllm.utils.deep_gemm is imported, its `_import_deep_gemm` is wrapped. On the first call, after the vendored
deep_gemm loads, this component assembles an include root (a copy of the vendored include tree with the two fork
headers) under /tmp/ds41-megamoe-fork and points DeepGEMM's JIT at it (`_C.init(root)`). All other DeepGEMM kernels
compile from byte-identical headers. The image and the fork headers are fixed by the release; compatibility is
asserted once in preflight (`vllm/vllm-openai:v0.31.0`). There is no per-component checksum or stock fallback: a
missing header or a failed JIT init raises at boot.

Enable: copy this file and the megamoe_fork/ directory next to components.py, add
`import fastpath_megamoe  # noqa: F401,E402  W6 MegaMoE scheduling fork` to components.py and boot with "megamoe" in
DS41_FASTPATH. This component prints the hook's one `ACTIVE megamoe` line after the JIT include root is live, then the root path.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile

from fastpath_hook import activate as _hook_activate, bounded, register

_DIR = os.path.dirname(os.path.abspath(__file__))
PATCH_DIR = os.path.join(_DIR, "megamoe_fork")
ROOT_BASE = os.path.join(tempfile.gettempdir(), "ds41-megamoe-fork")
FILES = (
    "include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh",
    "include/deep_gemm/scheduler/mega_moe.cuh",
)


def _root_name() -> str:
    """Content-addressed include-root name: a changed fork header gets a new root (and new JIT cache keys)."""
    h = hashlib.sha256()  # allow-sha256: content-addressed cache key, not an integrity check
    for rel in FILES:
        with open(os.path.join(PATCH_DIR, rel), "rb") as f:
            h.update(f.read())
    return "shared-early-" + h.hexdigest()[:16]


ROOT_NAME = _root_name()

# peer3 finish fold control block: byte offset in the symm buffer of MegaMoESignals::l2_empty_count[2^20 - 128]
# (128 bytes; below the v3 fold word's slot at 21009788), layout in tools/megamoe_p3fold_patch.py
P3FOLD_CTL_OFFSET = 21009280
P3FOLD_HAS = 64  # has[64] int8 within the block
# v11 combine-in-seam block (32 bytes at fold control + 256): u32 skip_combine, u32 magic, u64 slot base, u64 slot
# stride, u64 token stride (layout in megamoe_c11_patch.py)
C11_OFFSET = P3FOLD_CTL_OFFSET + 256
C11_MAGIC = 0x31314336
COMBINE_LIB = os.path.join(_DIR, "libw6combine.so")
_COMBINE = {"lib": None}
ACTIVE = False   # set once the fork's include root is active in this process


def _log(msg: str) -> None:
    sys.stderr.write(f"W6_MEGAMOE: {msg}\n")


def build_root(vendored_dir: str) -> str:
    """Assemble and return the fork's include root for this vendored DeepGEMM. Raises on any failure."""
    root = os.path.join(ROOT_BASE, ROOT_NAME)
    if os.path.exists(os.path.join(root, ".complete")):
        return root
    os.makedirs(ROOT_BASE, exist_ok=True)
    tmp = f"{root}.tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(os.path.join(vendored_dir, "include"), os.path.join(tmp, "include"))
    for rel in FILES:
        shutil.copyfile(os.path.join(PATCH_DIR, rel), os.path.join(tmp, rel))
    open(os.path.join(tmp, ".complete"), "w").close()
    try:
        os.rename(tmp, root)  # atomic; another process may have won the race
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if not os.path.exists(os.path.join(root, ".complete")):
            raise
    return root


def activate(dg) -> None:
    """Point an imported DeepGEMM module's JIT at the fork's include root. Raises if that cannot be done."""
    with bounded("W6 JIT include root + DeepGEMM _C.init", 300):
        root = build_root(os.path.dirname(os.path.abspath(dg.__file__)))
        dg._C.init(root)
    global ACTIVE
    ACTIVE = True
    _hook_activate("megamoe")
    _log(f"JIT include root {root} (pipeline variant shared-early)")


def p3fold_has(symm):
    """int8[64] view of the fold control block's has[] in a deep_gemm SymmBuffer (the publisher writes it)."""
    import torch

    raw = symm.buffer.view(torch.uint8)
    return raw[P3FOLD_CTL_OFFSET + P3FOLD_HAS: P3FOLD_CTL_OFFSET + P3FOLD_HAS + 64].view(torch.int8)


def p3fold_setup(symm, seg_dptr: int, state, stats, max_spins: int, off_done: int, off_rec: int, off_out: int,
                 hint: bool) -> None:
    """Write the fold control block's static fields (eager, before any CUDA graph capture) and reset has[] to -1.
    seg_dptr: device pointer of the peer3 segment; state: int32[2] call-counter tensor; stats: int64 GbStats tensor
    (empty = none); hint: every SM's idle dispatch warp polls the done word ahead of the combine (the patch's
    "done hint")."""
    import torch

    raw = symm.buffer.view(torch.uint8)
    assert raw.numel() >= P3FOLD_CTL_OFFSET + 128, "symm buffer smaller than the fold control block's offset"
    words = torch.tensor([seg_dptr, state.data_ptr(), stats.data_ptr() if stats.numel() else 0, max_spins,
                          off_done | (off_rec << 32), off_out | (int(hint) << 32), 0, 0], dtype=torch.int64)
    ctl = raw[P3FOLD_CTL_OFFSET: P3FOLD_CTL_OFFSET + 128]
    ctl[:64].copy_(words.view(torch.uint8))
    ctl[64:].fill_(0xFF)


def c11_skip(symm):
    """int32[1] view of the C11 block's skip_combine word (1: the next call leaves the combine to its consumer)."""
    import torch

    raw = symm.buffer.view(torch.uint8)
    assert raw.numel() >= C11_OFFSET + 32, "symm buffer smaller than the C11 block's offset"
    return raw[C11_OFFSET: C11_OFFSET + 4].view(torch.int32)


def c11_layout(symm):
    """(slot base, slot stride, token stride) the fork exported on its last call on this symm buffer, or None before
    the first call (one host read; call eagerly, outside graph capture)."""
    import torch

    raw = symm.buffer.view(torch.uint8)
    words = raw[C11_OFFSET: C11_OFFSET + 32].cpu()
    magic = int(words[4:8].view(torch.int32)[0])
    if magic != C11_MAGIC:
        return None
    base, slot_stride, token_stride = (int(v) for v in words[8:32].view(torch.int64).tolist())
    return base, slot_stride, token_stride


def combine_into(y, symm, num_tokens: int, layout=None, grid: int = 0, stream=None) -> None:
    """y[:num_tokens] = the fork's combine of the call that ran with skip_combine = 1 (bitwise the in-kernel combine,
    no peer3 fold). layout: c11_layout(symm), read once (fixed per symm buffer)."""
    import ctypes

    import torch

    if _COMBINE["lib"] is None:
        so = ctypes.CDLL(COMBINE_LIB)
        so.w6_combine_launch.restype = ctypes.c_int
        so.w6_combine_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_ulonglong,
                                         ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _COMBINE["lib"] = so
    base, slot_stride, token_stride = layout if layout is not None else c11_layout(symm)
    assert y.dtype == torch.bfloat16 and y.is_contiguous() and y.shape[-1] == 5120
    topk = symm.topk_idx
    assert topk.dtype == torch.int64 and topk.is_contiguous()
    s = (stream or torch.cuda.current_stream()).cuda_stream
    g = grid or max(1, min(152, (num_tokens * 40 + 7) // 8))
    err = _COMBINE["lib"].w6_combine_launch(y.data_ptr(), base, slot_stride, token_stride, topk.data_ptr(),
                                            int(num_tokens), 1, g, s)
    if err:
        raise RuntimeError(f"w6_combine_launch failed ({err})")


@register("megamoe", "vllm.utils.deep_gemm", defer_active=True)
def _patch(module) -> None:
    orig = module._import_deep_gemm
    state = {"done": False}

    def _import_deep_gemm():
        dg = orig()
        if dg is None:
            raise RuntimeError("DeepGEMM import returned None")
        if not state["done"]:
            state["done"] = True
            activate(dg)
        return dg

    module._import_deep_gemm = _import_deep_gemm
