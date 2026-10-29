"""ds41 fastpath component "fastload" (boot-time, read filter, bitwise).

Evidence (2026-10-22): the checkpoint is 475.3 GiB in 48 safetensors shards, of which **189 GiB is the two
Engram tables** (`layers.1/14.engram.embed.weight`, 91.6 GiB each, CPU-offloaded) and 7.4 GiB is the DSpark
drafter's `mtp.*` weights. The default safetensors iterator yields every tensor in every shard; the model
then discards what it does not need:

  * the DSpark draft model (`dspark.py:367`) keeps only `mtp.*` and skips everything else **after** it has
    been read;
  * the target model (`model.py:995`) skips `vision.*`/`aligner.*`/`image_*` and never loads `mtp.*`.

The drafter's "Loading weights took 43.38 s" is therefore mostly reads that are thrown away. This component
patches `weight_utils.should_skip_weight`, which the safetensors iterator calls **before** `get_tensor`
(`weight_utils.py:1147`):

  * while the DSpark draft model's `load_weights` runs (context var), every non-`mtp.` tensor is skipped
    before it is read;
  * outside that window, `mtp.` is skipped (the target model never loads it).

Only tensors that the model already discarded are skipped, so the loaded weights are identical: this is a
read filter, not a math change. The expected effect is the drafter's 43.4 s -> ~2-3 s (7.4 GiB at the
measured ~4 GB/s), and ~2 s off the target (7.4 GiB of `mtp` it never loads). It does **not** touch the
target's 189 GiB Engram read (that needs a file-backed mmap).

Armed by "fastload" in DS41_FASTPATH. The image is fixed by preflight; a missing class or hook raises at boot.
No env, no install-time work beyond the component import.
"""
from __future__ import annotations

import contextvars

from fastpath_hook import register

_DRAFT = contextvars.ContextVar("ds41_fastload_draft", default=False)
_COUNT = {"draft_skipped": 0, "target_skipped": 0}


@register("fastload", "vllm.models.deepseek_v41.nvidia.dspark")
def patch_dspark(module) -> None:
    """Set the 'draft load in progress' flag around the DSpark draft model's load_weights."""
    cls = getattr(module, "DSparkDeepseekV4ForCausalLM", None)
    if cls is None:
        raise RuntimeError("fastload: DSparkDeepseekV4ForCausalLM not found")
    orig = cls.load_weights
    if getattr(orig, "_ds41_fastload", False):
        return

    def load_weights(self, weights):
        token = _DRAFT.set(True)
        try:
            return orig(self, weights)
        finally:
            _DRAFT.reset(token)
            from vllm.logger import init_logger

            init_logger(__name__).info(
                "fastload: draft load skipped %d non-mtp tensors before reading; target skipped %d mtp",
                _COUNT["draft_skipped"],
                _COUNT["target_skipped"],
            )

    load_weights._ds41_fastload = True
    cls.load_weights = load_weights


@register("fastload", "vllm.model_executor.model_loader.weight_utils")
def patch_weight_utils(module) -> None:
    """Filter names before the safetensors iterator reads them (weight_utils.py:1147)."""
    orig = module.should_skip_weight
    if getattr(orig, "_ds41_fastload", False):
        return

    def should_skip_weight(weight_name, local_expert_ids):
        if _DRAFT.get():
            if not weight_name.startswith("mtp."):
                _COUNT["draft_skipped"] += 1
                return True
            return orig(weight_name, local_expert_ids)
        if weight_name.startswith("mtp."):
            _COUNT["target_skipped"] += 1
            return True
        return orig(weight_name, local_expert_ids)

    should_skip_weight._ds41_fastload = True
    module.should_skip_weight = should_skip_weight
