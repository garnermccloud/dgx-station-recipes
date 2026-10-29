"""MegaMoE hot tier + cold tier for DeepSeek-V4.1-Flash on one GB300 (+ RTX PRO 6000).

Each of the 40 target MoE layers keeps its 295 hot experts (static rowmap) in HBM as a
DeepGEMM MegaMoE module (shared expert fused). The 89 cold experts never reach HBM: the
RTX sidecar owns them in VRAM and computes them with b12x's SM120 fused MoE. Per forward:

  T <= peer_fused2.FUSED_MAX:
      one send kernel packs the cold rows and publishes; MegaMoE runs the hot experts;
      one finish kernel waits for the sidecar and adds its rows.       (decode / verify)
  T > peer_fused2.FUSED_MAX:
      the pack and scatter-add run as separate kernels; MegaMoE runs between them.
                                                                        (prefill)

The DSpark drafter's MoE layers (128 experts, layers 40+) are untouched.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import re
import sys
import threading

import torch

E = 384
TOPK = 6
HIDDEN = 5120
ROWMAP_PATH = os.environ.get(
    "MEGA_ROWMAP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rowmap-mix-v1-h255.json"))
FI_TUNE_FILE = os.environ.get("MEGA_FI_TUNE_FILE", "")
TARGET = "vllm.models.deepseek_v4.nvidia.model"
_PREFIX = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.ffn\.experts$")

_LOG = lambda m: sys.stderr.write(f"MEGA_PEER {m}\n")

_rowmap = None
_peer2 = None
_pt2_mod = None
_pf2 = None
_orig = {}


def _load_rowmap() -> dict:
    global _rowmap
    if _rowmap is None:
        with open(ROWMAP_PATH) as f:
            raw = json.load(f)
        layers = raw["layers"] if "layers" in raw else raw
        _rowmap = {int(k): {"hot": [int(x) for x in v["hot"]], "cold": [int(x) for x in v["cold"]]}
                   for k, v in layers.items()}
        for idx, v in _rowmap.items():
            if set(v["hot"]) | set(v["cold"]) != set(range(E)) or set(v["hot"]) & set(v["cold"]):
                raise RuntimeError(f"MEGA_PEER rowmap layer {idx} is not a partition of 0..{E - 1}")
        _LOG(f"rowmap {ROWMAP_PATH}: {len(_rowmap)} layers")
    return _rowmap


class ColdState:
    """Per-layer split: row[e] = hot row, or -(cold row + 1)."""

    def __init__(self, idx: int, hot: list[int], cold: list[int]):
        self.idx = idx
        self.hot = hot
        self.cold = cold
        self.n_hot = len(hot)
        self.n_cold = len(cold)
        self.row = [0] * E
        for r, e in enumerate(hot):
            self.row[e] = r
        for c, e in enumerate(cold):
            self.row[e] = -(c + 1)
        self.row_map = None   # int32 [E] on device, built at finalize


def _target_layer(prefix: str, num_experts: int, num_local_experts: int):
    m = _PREFIX.search(prefix)
    if m is None or num_experts != E or num_local_experts != E:
        return None
    idx = int(m.group(1))
    return idx if idx in _load_rowmap() else None


# ---------------------------------------------------------------------------
# DeepseekV4MegaMoEExperts patches
# ---------------------------------------------------------------------------


def _init(self, vllm_config, *, num_experts, num_local_experts, experts_start_idx, **kw):
    idx = _target_layer(kw.get("prefix", ""), num_experts, num_local_experts)
    if idx is None:
        _orig["init"](self, vllm_config, num_experts=num_experts, num_local_experts=num_local_experts,
                      experts_start_idx=experts_start_idx, **kw)
        self._mp = None
        return
    split = _load_rowmap()[idx]
    n_hot = len(split["hot"])
    kw["num_logical_experts"] = E
    _orig["init"](self, vllm_config, num_experts=n_hot, num_local_experts=n_hot,
                  experts_start_idx=0, **kw)
    self._mp = ColdState(idx, split["hot"], split["cold"])
    if idx == 0:
        _LOG(f"layer 0 ({kw.get('prefix')}): {n_hot} hot experts in MegaMoE, "
             f"{E - n_hot} cold in the RTX sidecar")


def _weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id,
                   return_success=False):
    st = getattr(self, "_mp", None)
    if st is None:
        return _orig["weight_loader"](self, param, loaded_weight, weight_name, shard_id,
                                      expert_id, return_success)
    r = st.row[expert_id]
    if r < 0:
        return True if return_success else None  # cold experts live only on the 6000
    dst = param.data[r]
    if shard_id in ("w1", "w3"):
        if "w13_" not in weight_name:
            return False if return_success else None
        offset = 0 if shard_id == "w1" else self.intermediate_size
        dst = dst.narrow(0, offset, self.intermediate_size)
    elif shard_id == "w2":
        if "w2_" not in weight_name:
            return False if return_success else None
    else:
        raise ValueError(f"Unsupported expert shard id: {shard_id}")
    if dst.shape != loaded_weight.shape:
        raise ValueError(f"MEGA_PEER shape mismatch for {weight_name}: {tuple(dst.shape)} "
                         f"vs {tuple(loaded_weight.shape)}")
    dst.copy_(loaded_weight)
    return True if return_success else None


_weight_loader.supports_moe_loading = True


def _finalize_weights(self, shared_experts=None):
    st = getattr(self, "_mp", None)
    if st is None:
        _orig["finalize_weights"](self, shared_experts)
        return
    # This runs on vLLM's weight-loader thread, which has no current CUDA context (the driver logs
    # CUDA_ERROR_INVALID_CONTEXT for it).  Do not ask torch for the current device here and do not
    # allocate the row map directly on a CUDA device: build it on the CPU and copy it to the expert
    # weight's device, exactly as the old cold-tier finalize did.  The copy is asynchronous; the
    # first forward uses it only after the loader's stream has been synchronized by vLLM.
    dev = self.w13_weight.device if self.w13_weight is not None else torch.device("cuda")
    done = threading.Event()

    def _watchdog():
        if not done.wait(600):
            _LOG(f"FATAL finalize_weights layer {st.idx} did not return within 600 s "
                 f"(waiting for MegaMoE weight finalize; no boundary hook has run)")
            os._exit(1)

    threading.Thread(target=_watchdog, name=f"mega-finalize-{st.idx}", daemon=True).start()
    if st.idx in (0, 39):
        _LOG(f"finalize layer {st.idx}: start")
    try:
        _orig["finalize_weights"](self, shared_experts)
        if st.row_map is None:
            st.row_map = torch.tensor(st.row, dtype=torch.int32).to(dev)
        # finalize_weights drops the loader-side Parameters; return their blocks to the driver before the
        # next layer allocates its transformed weights.  The old cold-tier path did this per layer.
        torch.cuda.empty_cache()
        if st.idx in (0, 39):
            _LOG(f"finalize layer {st.idx}: done")
    finally:
        done.set()


def _quant(x: torch.Tensor):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_impl)

    return _mxfp8_e4m3_quantize_impl(x.contiguous(), is_sf_swizzled_layout=False)


def _peer_tier2(device):
    global _peer2, _pt2_mod
    if _peer2 is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import peer_tier2

        _pt2_mod = peer_tier2
        _peer2 = peer_tier2.PeerTier2(device)
        _LOG(f"ACTIVE peer tier v2: max_rows={_peer2.max_rows}, small-row routing <= {peer_tier2.SMALL_ROWS}")
    return _peer2


def _fused2():
    global _pf2
    if _pf2 is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import peer_fused2

        _pf2 = peer_fused2
    return _pf2


def _forward(self, hidden_states, topk_weights, topk_ids, *, activation_clamp, fast_math=True):
    st = getattr(self, "_mp", None)
    if st is None:
        return _orig["forward"](self, hidden_states, topk_weights, topk_ids,
                                activation_clamp=activation_clamp, fast_math=fast_math)
    if st.row_map is None:
        raise RuntimeError(f"MEGA_PEER layer {st.idx}: row_map not finalized")
    T = hidden_states.shape[0]
    peer2 = _peer_tier2(hidden_states.device)
    if T <= _fused2().FUSED_MAX:
        pf2 = _fused2()
        hot_ids, has, pos, rows = pf2.route_send2(
            peer2, topk_ids, topk_weights, st.row_map, hidden_states, st.idx, HIDDEN, TOPK,
            _pt2_mod.SMALL_ROWS)
        y = _orig["forward"](self, hidden_states, topk_weights, hot_ids,
                             activation_clamp=activation_clamp, fast_math=fast_math)
        pf2.finish2(peer2, y, has, pos, rows)
        return y

    valid = topk_ids >= 0
    rm = st.row_map[topk_ids.clamp_min(0).long()]
    hot_ids = torch.where(valid & (rm >= 0), rm, -1).to(topk_ids.dtype)
    cold_ids = torch.where(valid & (rm < 0), -rm - 1, -1).to(torch.int32)
    xq, xs = _quant(hidden_states)
    has, pos, rows = peer2.send(xq, xs, cold_ids, topk_weights, st.idx)
    y = _orig["forward"](self, hidden_states, topk_weights, hot_ids,
                         activation_clamp=activation_clamp, fast_math=fast_math)
    peer2.finish(y, has, pos, rows)
    return y


def _patch(module):
    cls = module.DeepseekV4MegaMoEExperts
    if getattr(cls, "_mega_peer", False):
        return
    for name, fn in (("init", _init), ("weight_loader", _weight_loader),
                     ("finalize_weights", _finalize_weights), ("forward", _forward)):
        attr = "__init__" if name == "init" else name
        _orig[name] = getattr(cls, attr)
        setattr(cls, attr, fn)
    cls._mega_peer = True
    _LOG(f"ACTIVE MegaMoE hot/cold hook on {TARGET}.DeepseekV4MegaMoEExperts (peer v2)")


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, target, patch):
        self.target, self.patch = target, patch

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig_exec=orig_exec, _patch_fn=self.patch):
            _orig_exec(module)
            _patch_fn(module)

        spec.loader.exec_module = exec_module
        return spec


def _patch_ll_gemm(module):
    spec = importlib.util.spec_from_file_location("ll_gemm", os.path.join(os.path.dirname(__file__), "ll_gemm.py"))
    ll = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ll)
    ll.patch(module)


def _check_pin(src) -> None:
    """Refuse to boot if the pinned FlashInfer tactics would be ignored."""
    from flashinfer.autotuner.autotuner import (
        _classify_metadata_mismatches,
        _collect_metadata,
        _format_meta_mismatches,
    )

    meta = json.loads(src.read_text()).get("_metadata")
    if not isinstance(meta, dict):
        raise RuntimeError(f"pinned FlashInfer tune file {src} has no _metadata")
    hard, soft = _classify_metadata_mismatches(meta, _collect_metadata())
    if hard or soft:
        raise RuntimeError(
            f"pinned FlashInfer tune file {src} is incompatible: "
            f"{_format_meta_mismatches({**hard, **soft})}")


def _patch_fi_tune(module):
    import hashlib
    import shutil
    from pathlib import Path

    src = Path(FI_TUNE_FILE)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    cache = Path(os.environ.get("VLLM_CACHE_ROOT", "/root/.cache/vllm")) / "ds41-fi-tune"
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"{digest[:16]}.json"

    def resolve_flashinfer_autotune_file(runner) -> Path:
        _check_pin(src)
        shutil.copyfile(src, dst)
        return dst

    module.resolve_flashinfer_autotune_file = resolve_flashinfer_autotune_file
    _LOG(f"ACTIVE FlashInfer autotune pin {src} -> {dst}")


def install():
    targets = [(TARGET, _patch),
               ("vllm.model_executor.kernels.linear.mxfp8.flashinfer", _patch_ll_gemm)]
    if FI_TUNE_FILE:
        # Must patch before kernel_warmup and flashinfer_sparse_mla_warmup import the name.
        targets.append(("vllm.model_executor.warmup.flashinfer_autotune_cache", _patch_fi_tune))
    for target, patch in targets:
        if target in sys.modules:
            patch(sys.modules[target])
        else:
            sys.meta_path.insert(0, _Finder(target, patch))
