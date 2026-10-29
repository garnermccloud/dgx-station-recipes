"""ds41 fastpath components (W3, DeepSeek-V4.1 pre-attention chain), all bitwise identical to the stock path:

  preattn        q/kv norm + MXFP8 q + RoPE/MXFP8 SWA insert as one PDL launch between fused_wqa_wkv and wq_b, and
                 the per-step memo of positions.to(int32) + the compressed key table (#61868 item 11)
  preattn_cand   layer-20 candidate selection on W3's exact select over each row's valid prefix
  preattn_index  candidate-consumer index layers (24/28/32/36): indexer branch on a side stream beside wq_b (#14),
                 and the consumers' expansion + schedule forked at layer 24's start; arms the memo too
  preattn_compress  compressor + indexer layers (2/8/14/20): the compressor's state save / pooling / norm on a W3
                 kernel (bitwise, ring rows saved in parallel), and the indexer queries, compressor + K store and the
                 main-cache insert on three side streams beside wq_b (#14)

  (index chain, W8) on those compressor + indexer layers the decode top-k writes every row itself, so the stock
                 top-k buffer pre-fill is skipped (noclear), and the top-k + the key-table translation run as one
                 launch (topkg: libw8topkg.so = vLLM cooperative_topk body + compute_global_topk_indices_and_lens),
                 the key table handed to attention through the step memo. W3_DISABLE=noclear,topkg turns them off.

Side streams (FULL graphs and eager forwards only) never leave one layer's pre-attention region: each forks after
the layer's start and joins before its attention, so none overlaps attention, the router or MegaMoE.
W3_SIDE_STREAMS: 1 (default) all, 0 none, or a comma list of index (indexer branch), schedule (layer-24 expansion +
schedule fork) and cstreams (compressor-layer streams) for bisecting in-server.

Patch conditions and fallbacks: w3preattn/integrate.py. CUDA extension: w3preattn/w3_ext.so, prebuilt for sm_103a
in the vLLM af7f9488 image (the prebuilt w3_ext*.so next to integrate.py). Each patched module logs one "DS41_FASTPATH preattn"
line; the first fused qkv call logs "W3 preattn: qkv fused path active".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastpath_hook import register  # noqa: E402

ATTN = "vllm.models.deepseek_v41.attention"
MEGA = "vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn"
INDEXER = "vllm.model_executor.layers.sparse_attn_indexer"
COMPRESSOR = "vllm.models.deepseek_v41.compressor"


def _arm(feature: str):
    def patch(module) -> None:
        from w3preattn import integrate

        getattr(integrate, f"patch_{feature}")(module)
        sys.stderr.write(f"DS41_FASTPATH preattn: patched {module.__name__} ({feature})\n")

    patch.__name__ = f"patch_{feature}"
    return patch


register("preattn", ATTN)(_arm("qkv"))
register("preattn", MEGA)(_arm("memo"))
register("preattn_cand", INDEXER)(_arm("cand"))
register("preattn_index", ATTN)(_arm("index"))
register("preattn_index", MEGA)(_arm("memo"))
register("preattn_compress", ATTN)(_arm("cstreams"))
register("preattn_compress", COMPRESSOR)(_arm("compress"))
