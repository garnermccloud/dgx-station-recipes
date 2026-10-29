"""ds41 fastpath component `peer3`: the GB300 side of peer tier v3 for MoE layer calls with T <= 64 tokens.

Wraps DeepseekV4MegaMoEExperts.forward as installed by the MegaMoE hook (mega_peer_hook, MEGA_PEER=2). For a
target layer with T <= PEER3_MAX_T tokens:
    send    one kernel, grid T: route split (hot rows for MegaMoE), per-token MXFP8 quantize + publish of the cold
            routes to the v3 segment (system-scope release per token)
    MegaMoE the stock DeepGEMM call on the hot routes (shared expert fused), unchanged
    finish  one kernel, grid T, a programmatic dependent of MegaMoE: wait for the RTX v3 sidecar's acknowledgement,
            add the cold rows, advance
Larger batches (prefill), layers outside the rowmap and MEGA_COUNT runs fall through to the hook's own forward
(peer tier v2 through the b12x sidecar). Needs the v3 sidecar (peer_server3.py) on the same segment.

Env: VLLM_EXP_PEER3_SHM (default /dev/shm/vllm_peer_tier3), PEER3_MAX_T (default 64), PEER3_SPINS (wait bound,
default 20,000,000 polls, as the v2 path), PEER3_PREBUILT (directory of prebuilt peer3_gb300_sm103*.so) or
PEER3_BUILD_DIR (JIT build cache). PEER3_DIR (default: the peer3/ directory next to this file).
PEER3_DETERMINISTIC_PREFILL (default 0; a slow test mode): calls with T > PEER3_MAX_T also go through v3, in 64-token
chunks after MegaMoE (send + finish per chunk), instead of b12x, so greedy runs are bit-reproducible end to end.
PEER3_PDL (default 1): launch the finish as a programmatic dependent of MegaMoE (its wait and host reads overlap
the kernel boundary). PEER3_FOLD (default 1): when the MegaMoE fork v5 (fastpath_megamoe, peer3 finish fold) is
active, this component's calls arm the fold instead of launching the finish: the send writes has[] into the fork's
fold control block, and MegaMoE's combine waits for the RTX, adds the cold rows and advances the call counter itself
(bitwise the finish's numerics, the same one statistics wait per call). 0 = the finish kernel as before. PEER3_VERSION_WAIT_S (default 120): how long the first call waits for the sidecar's protocol version stamp
before it fails the process (a sidecar from another peer3 build must never serve this one).
"""
from __future__ import annotations

import os
import sys
import time

from fastpath_hook import activate as _hook_activate, register  # noqa: E402

PEER3_DIR = os.environ.get("PEER3_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "peer3"))
SHM = os.environ.get("VLLM_EXP_PEER3_SHM", "/dev/shm/vllm_peer_tier3")
MAX_T = int(os.environ.get("PEER3_MAX_T", "64"))
SPINS = int(os.environ.get("PEER3_SPINS", "20000000"))
VERSION_WAIT_S = float(os.environ.get("PEER3_VERSION_WAIT_S", "120"))
PDL = os.environ.get("PEER3_PDL", "1") != "0"  # finish as a programmatic dependent of MegaMoE
DET_PREFILL = os.environ.get("PEER3_DETERMINISTIC_PREFILL", "0") == "1"
FOLD = os.environ.get("PEER3_FOLD", "1") != "0"  # fold the finish into MegaMoE (fork v5) when available
FOLD_HINT = os.environ.get("PEER3_FOLD_HINT", "1") != "0"  # the fold's done hint (idle dispatch warps poll ahead)
TARGET = "vllm.models.deepseek_v4.nvidia.model"

_state = None


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH peer3: {msg}\n")


class _Peer3:
    """Process-wide GB300 state: extension, segment, call counter, wait statistics."""

    def __init__(self, device):
        import torch

        sys.path.insert(0, os.path.join(PEER3_DIR, "py"))
        import peer3_ext
        import peer3_seg

        self.ext = peer3_ext.load("gb300")
        self.ext.preload()
        self.seg = peer3_seg.Segment(SHM, self.ext.SEG_BYTES, device, create=True, zero_header=False)
        self._check_version()
        # Start the call counter past the last acknowledged tag, so a sidecar that outlived a previous vLLM never
        # mistakes our first call for one it already served.
        done = self.seg.u64(self.ext.OFF_DONE)
        counter = (done >> 12) & 15
        first = 1 if counter in (0, 15) else counter + 1
        self.state = torch.tensor([first, 0], dtype=torch.int32, device=device)
        self.stats = torch.zeros(self.ext.GB_STATS_WORDS, dtype=torch.int64, device=device)
        self.fold = {}  # symm buffer address -> fold has[] view (False: the fold is unavailable for that buffer)
        _log(f"segment {SHM} ({self.ext.SEG_BYTES} B), first counter {first}, T <= {MAX_T}, "
             f"finish {'PDL dependent of MegaMoE' if PDL else 'serial'}")

    def _check_version(self) -> None:
        """The sidecar stamps its protocol version into the segment; a mismatched pair must not run."""
        want = self.ext.PROTO_VERSION
        deadline = time.time() + VERSION_WAIT_S
        ver = self.seg.u64(self.ext.OFF_VERSION)
        while ver != want and time.time() < deadline:
            time.sleep(0.5)
            ver = self.seg.u64(self.ext.OFF_VERSION)
        if ver != want:
            raise RuntimeError(f"peer3: segment {SHM} carries protocol version {ver}, this GB300 build speaks {want}: "
                               f"start the sidecar from the same peer3 build (or disarm peer3)")


def _peer3(device) -> _Peer3:
    global _state
    if _state is None:
        _state = _Peer3(device)
    return _state


def fold_has(p3: _Peer3, module):
    """The MegaMoE fork's fold has[] (int8[64], device) for `module`'s symm buffer, or None when the fold is off or
    unavailable (PEER3_FOLD=0, fork v5 not active, during a CUDA graph capture before the first eager call).
    A publisher that writes has[t] for t < T here instead of its own buffer enables the call's fold: it then passes
    folded=True to mega_and_finish, which launches no finish."""
    if not FOLD:
        return None
    fm = sys.modules.get("fastpath_megamoe")
    if fm is None or not getattr(fm, "ACTIVE", False) or not hasattr(fm, "p3fold_has"):
        return None
    symm = module.get_symm_buffer()
    key = symm.buffer.data_ptr()
    view = p3.fold.get(key)
    if view is None:
        import torch

        if torch.cuda.is_current_stream_capturing():
            return None  # the static fields are written eagerly; this call keeps the finish
        ranks = getattr(symm, "num_ranks", 1)
        if symm.hidden != 5120 or ranks != 1:  # the fold is single-rank; out rows are peer3::kHidden wide
            p3.fold[key] = False
            _log(f"fold unavailable for symm buffer {key:#x} (hidden {symm.hidden}, ranks {ranks})")
            return None
        fm.p3fold_setup(symm, p3.seg.dptr, p3.state, p3.stats, SPINS, p3.ext.OFF_DONE, p3.ext.OFF_REC,
                        p3.ext.OFF_OUT, FOLD_HINT)
        view = fm.p3fold_has(symm)
        p3.fold[key] = view
        _hook_activate("peer3")
        _log(f"finish folded into MegaMoE (fork v5) for symm buffer {key:#x}")
    return None if view is False else view


def mega_and_finish(p3: _Peer3, stock_forward, module, hidden_states, topk_weights, hot_ids, has, layer: int,
                    activation_clamp, fast_math=True, folded: bool = False):
    """Everything after a call's publish: MegaMoE on the hot routes, then the cold finish (a PDL dependent of
    MegaMoE unless PEER3_PDL=0), or no finish when the publisher enabled the fold (folded=True: has is fold_has()).
    For any publisher (this component's send, W4's router): call it right after the publish, on the same stream."""
    y = stock_forward(module, hidden_states, topk_weights, hot_ids, activation_clamp=activation_clamp,
                      fast_math=fast_math)
    if not folded:
        p3.ext.finish(y, has, p3.seg.dptr, p3.state, p3.stats, SPINS, layer, PDL)
    return y


def _chunked(module, st, stock_forward, hidden_states, topk_weights, topk_ids, activation_clamp, fast_math):
    """PEER3_DETERMINISTIC_PREFILL: MegaMoE on the hot routes of all T tokens, then the cold rows through v3 in chunks
    of up to 64 tokens (one send + finish each, exactly the decode path's numerics)."""
    import torch

    p3 = _peer3(hidden_states.device)
    T = hidden_states.shape[0]
    ids = topk_ids.contiguous()
    rm = st.row_map[ids.clamp(min=0)]
    hot_ids = torch.where((ids >= 0) & (rm >= 0), rm.to(ids.dtype), torch.full_like(ids, -1))  # = split_routes
    y = stock_forward(module, hidden_states, topk_weights, hot_ids, activation_clamp=activation_clamp,
                      fast_math=fast_math)
    w = topk_weights.contiguous().float()
    x = hidden_states.contiguous()
    has = torch.empty(T, dtype=torch.int8, device=ids.device)
    scratch = torch.empty(MAX_T, ids.shape[1], dtype=ids.dtype, device=ids.device)
    for c0 in range(0, T, MAX_T):
        c1 = min(T, c0 + MAX_T)
        p3.ext.send(ids[c0:c1], w[c0:c1], st.row_map, x[c0:c1], scratch[:c1 - c0], has[c0:c1], p3.seg.dptr, p3.state,
                    st.idx, p3.stats)
        p3.ext.finish(y[c0:c1], has[c0:c1], p3.seg.dptr, p3.state, p3.stats, SPINS, st.idx, False)
    return y


@register("peer3", TARGET, defer_active=True)
def patch(module) -> None:
    """Route T <= MAX_T calls of the hook's MegaMoE experts through peer tier v3."""
    cls = module.DeepseekV4MegaMoEExperts
    hook_forward = cls.forward
    hook_globals = getattr(hook_forward, "__globals__", {})
    if "_orig" not in hook_globals or "forward" not in hook_globals["_orig"]:
        raise RuntimeError("peer3 needs the MegaMoE hook (mega_peer_hook) installed first")
    stock_forward = hook_globals["_orig"]["forward"]

    def forward(self, hidden_states, topk_weights, topk_ids, *, activation_clamp, fast_math=True):
        import torch

        st = getattr(self, "_mp", None)
        T = hidden_states.shape[0]
        if DET_PREFILL and st is not None and T > MAX_T:
            return _chunked(self, st, stock_forward, hidden_states, topk_weights, topk_ids, activation_clamp, fast_math)
        if st is None or T > MAX_T or T == 0:
            return hook_forward(self, hidden_states, topk_weights, topk_ids, activation_clamp=activation_clamp,
                                fast_math=fast_math)
        p3 = _peer3(hidden_states.device)
        ids = topk_ids.contiguous()
        hot_ids = torch.empty_like(ids)
        has = fold_has(p3, self)
        folded = has is not None
        if not folded:
            has = torch.empty(T, dtype=torch.int8, device=ids.device)
        p3.ext.send(ids, topk_weights.contiguous().float(), st.row_map, hidden_states.contiguous(), hot_ids, has,
                    p3.seg.dptr, p3.state, st.idx, p3.stats)
        return mega_and_finish(p3, stock_forward, self, hidden_states, topk_weights, hot_ids, has, st.idx,
                               activation_clamp, fast_math, folded=folded)

    cls.forward = forward
    _log(f"{TARGET}.DeepseekV4MegaMoEExperts.forward: T <= {MAX_T} through peer tier v3")
