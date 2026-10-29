"""Registered fastpath components for the production DeepSeek-V4.1-Flash-1017 recipe.

Only components used by the default recipe are imported here.  The measurement-only and experimental
components from the development tree are deliberately not shipped; the kernel sources for the shipped set
live in ../kernels/.  Each import registers one or more names with fastpath_hook; serve.sh arms the names
listed in DS41_FASTPATH.
"""

import fastpath_tail_markov_fused
import fastpath_tail_swa_meta
import fastpath_tail_dflash_prep
import fastpath_tail_ctxkv
import fastpath_tail_replay_skip
import fastpath_tail_embed
import fastpath_tail_drafthead
import fastpath_attn
import fastpath_preattn
import fastpath_tail_draft_embed
import fastpath_boundary
import fastpath_w9_suffix
import fastpath_ds41_api
import fastpath_topp
import fastpath_w8oproj
import fastpath_tail_proj_side
import fastpath_lowb_aux
import fastpath_engram_side
import fastpath_fastload
import fastpath_coldskip
import fastpath_peer3
import fastpath_megamoe
import fastpath_quant_producer
import fastpath_a12_retire
import fastpath_a12_memory_guard

REGISTERED = (
    fastpath_tail_markov_fused,
    fastpath_tail_swa_meta,
    fastpath_tail_dflash_prep,
    fastpath_tail_ctxkv,
    fastpath_tail_replay_skip,
    fastpath_tail_embed,
    fastpath_tail_drafthead,
    fastpath_attn,
    fastpath_preattn,
    fastpath_tail_draft_embed,
    fastpath_boundary,
    fastpath_w9_suffix,
    fastpath_ds41_api,
    fastpath_topp,
    fastpath_w8oproj,
    fastpath_tail_proj_side,
    fastpath_lowb_aux,
    fastpath_engram_side,
    fastpath_fastload,
    fastpath_coldskip,
    fastpath_peer3,
    fastpath_megamoe,
    fastpath_quant_producer,
    fastpath_a12_retire,
    fastpath_a12_memory_guard,
)
