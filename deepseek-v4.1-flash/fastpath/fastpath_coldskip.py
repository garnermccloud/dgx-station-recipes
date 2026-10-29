"""ds41 fastpath component "coldskip" (boot-time read filter, bitwise).

The GB300 keeps only the rowmap's hot experts in HBM. The cold experts are computed on the RTX PRO 6000 sidecar and
the hook's weight loader returns before copying them (`hook/mega_peer_hook.py`, `_weight_loader`:
`return True if return_success else None  # cold experts live only on the 6000`).

The safetensors iterator still reads every cold tensor from disk first: `should_skip_weight` (called before
`get_tensor`, weight_utils.py) only knows the vLLM expert-parallel filter, and EP size is 1 here. For the h255
rowmap that is 40 layers x 129 cold experts x 6 tensors = 30,960 tensors, 90.3 GiB of the 377.5 GiB the target
reads every boot (measured from the checkpoint headers, 2026-10-25).

This component patches `should_skip_weight` to skip exactly those names before they are read. It is a read filter:
the same tensors were already discarded after reading, so the loaded weights are identical. The cold set comes from
the rowmap the hook and the sidecar use (MEGA_ROWMAP, set by serve.sh). A missing or unreadable rowmap, or a target
load that skips a different number of tensors than the rowmap implies, raises at boot.

Armed by "coldskip" in DS41_FASTPATH. It prints its ACTIVE line once the target model load has skipped exactly the
rowmap's cold tensors.
"""
from __future__ import annotations

import json
import os
import re
import sys

from fastpath_hook import activate, register

_NAME = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")
TENSORS_PER_EXPERT = 6  # w1/w2/w3 weight + scale
_STATE: dict = {"cold": None, "layers": 0, "cold_n": 0, "skipped": 0, "loads": 0}


def _log(msg: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH coldskip: {msg}\n")


def _cold_sets() -> dict[int, set[int]]:
    """Cold expert ids per layer from MEGA_ROWMAP; raises when the rowmap is missing or malformed."""
    if _STATE["cold"] is not None:
        return _STATE["cold"]
    path = os.environ.get("MEGA_ROWMAP", "")
    if not path:
        raise RuntimeError("coldskip: MEGA_ROWMAP is unset; serve.sh sets it to the recipe rowmap")
    with open(path) as f:
        doc = json.load(f)
    layers = doc["layers"] if "layers" in doc else doc
    cold = {int(k): set(int(x) for x in v["cold"]) for k, v in layers.items()}
    if not cold or any(not v for v in cold.values()):
        raise RuntimeError(f"coldskip: rowmap {path} has no layers or an empty cold set")
    _STATE.update(cold=cold, layers=len(cold), cold_n=sum(len(v) for v in cold.values()) // len(cold))
    return cold


def _expected() -> int:
    return sum(len(v) for v in _cold_sets().values()) * TENSORS_PER_EXPERT


@register("coldskip", "vllm.model_executor.model_loader.weight_utils", defer_active=True)
def patch_weight_utils(module) -> None:
    """Filter cold-expert names before the safetensors iterator reads them."""
    orig = module.should_skip_weight
    if getattr(orig, "_ds41_coldskip", False):
        return
    cold = _cold_sets()

    def should_skip_weight(weight_name, local_expert_ids):
        m = _NAME.match(weight_name)
        if m is not None and int(m.group(2)) in cold.get(int(m.group(1)), ()):
            _STATE["skipped"] += 1
            return True
        return orig(weight_name, local_expert_ids)

    should_skip_weight._ds41_coldskip = True
    module.should_skip_weight = should_skip_weight


@register("coldskip", "vllm.model_executor.model_loader.default_loader", defer_active=True)
def patch_default_loader(module) -> None:
    """Check the skip count after each load_weights call (target, then drafter) and log it."""
    cls = module.DefaultModelLoader
    orig = cls.load_weights
    if getattr(orig, "_ds41_coldskip", False):
        return

    def load_weights(self, model, model_config):
        before = _STATE["skipped"]
        out = orig(self, model, model_config)
        got = _STATE["skipped"] - before
        _STATE["loads"] += 1
        want = _expected()
        if _STATE["loads"] == 1:
            # The target load iterates every shard: it must skip exactly the rowmap's cold tensors.
            if got != want:
                raise RuntimeError(f"coldskip: target load skipped {got} cold-expert tensors; the rowmap implies "
                                   f"{want} ({_STATE['layers']} layers x {_STATE['cold_n']} cold x "
                                   f"{TENSORS_PER_EXPERT})")
            activate("coldskip")
        _log(f"load {_STATE['loads']}: skipped {got} cold-expert tensors before reading "
             f"({_STATE['layers']} layers x {_STATE['cold_n']} cold experts x {TENSORS_PER_EXPERT})")
        return out

    load_weights._ds41_coldskip = True
    cls.load_weights = load_weights
