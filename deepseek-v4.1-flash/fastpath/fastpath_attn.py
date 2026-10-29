"""ds41 fastpath component "attn" (W1): DeepSeek-V4.1 decode attention on W1's bitwise-parallel kernel.

Patches ``DeepseekV4MegaAttnAttention._forward_decode_mega`` (vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn):
for the installed layer classes and decode widths the FlashMLA mega op is replaced by W1's single persistent PDL kernel
(w1attn/w1_attn_ext.so), which writes bit-identical fp8 + packed ue8m0 output; everything else falls through to the
installed method untouched. See w1attn/integrate.py for the exact conditions. Armed layers prepare W1 (extension,
smem opt-in, 1.1 MiB scratch) in their constructor, so the decode graphs captured later already contain W1; after
capture_model the log carries "captured decode attention calls, W1/stock per width: ...".

Env (defaults): W1_LAYERS=sparse,swa (comma list of sparse, swa, draft), W1_MAX_SQ=16 (sparse layers),
W1_MAX_SQ_SWA=24 (window-only layers), W1_NUM_LAYERS=40,
The prebuilt w1_attn_ext.so next to w1attn/ops.py is the only runtime extension.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastpath_hook import register  # noqa: E402

TARGET_ATTN = "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn"
RUNNERS = ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner")


@register("attn", TARGET_ATTN)
def patch_attention(module) -> None:
    from w1attn import integrate

    integrate.install(module)


def _patch_runner(module) -> None:
    from w1attn import integrate

    integrate.install_capture_summary(module)


for _runner in RUNNERS:
    register("attn", _runner)(_patch_runner)
