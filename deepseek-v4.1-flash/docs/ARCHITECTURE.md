# Architecture

This is the "why" document. You do not need it to run the server. It is here so that when something surprises
you, you can find out whether it is physics, a policy choice, or a bug.

## 1. The box, and why the weights are split

The checkpoint is DeepSeek-V4.1-Flash-1017: **MXFP4 routed experts, FP8 dense weights, FP8 Engram tables held in
host memory, and an NVFP4 compressed KV cache** (`nvfp4_ds_mla`: an MXFP8 sliding window plus the NVFP4
compressed store). A decode step has to read a lot of bytes. At one stream:

| what | per step |
|---|---|
| dense weights, shared expert, gate, mHC (40 layers) | 6.8 GB |
| routed hot experts (≈20 distinct × 18.8 MB per layer) | 14.8 GB |
| routed cold experts | 1.4 GB |
| LM head (bf16) | 1.3 GB |
| DSpark drafter (MTP layer, its experts, draft head) | ~1.4 GB |
| **total** | **~24.4 GB** |

The GB300's HBM sustains **6.84 TB/s** on contiguous bulk transfers (measured on this box, not a datasheet
number), so the floor for one step is **~3.6 ms**. The RTX PRO 6000 adds 1.63 TB/s of GDDR7 and the Grace CPU
adds 0.355 TB/s over NVLink-C2C. Neither is fast enough to hold the model, but both are fast enough to take a
slice of the bytes — and that is the whole design: **one GB300 that streams most of the model, one RTX that
streams the cold tail, and a host that holds what neither GPU can afford to keep.**

The step is not latency-bound by physics. At the shipped recipe the GB300's HBM is busy roughly half the step;
the other half is the latency-bound dense chain (attention, projections, the seam, the router) plus the fixed
cost of the MoE kernel. That gap is what the fast paths attack.

## 2. The peer tier

The MoE layers are the only place where the model is bigger than one GPU's HBM. Peer tier v3 splits each MoE
layer three ways:

* **GB300 — hot experts.** A static rowmap (`rowmap-mix-v1-h255.json`, built from measured route counts) keeps
  255 of each layer's 384 routed experts on the GB300, computed by DeepGEMM's MegaMoE kernel.
* **RTX PRO 6000 — cold experts.** The other 129 per layer live only in the RTX's memory. For each MoE call the
  GB300 publishes the tokens routed to cold experts; a sidecar computes them on the RTX while MegaMoE computes the
  hot ones, and the results are added in the same layer.
* **Host — the Engram tables.** The two ~94 GiB Engram tables stay in pinned host memory (`cpu_offload`).

The RTX is a second memory system working in parallel, not a slow fallback: its own bandwidth streams the cold
experts while the GB300 streams the hot ones. Moving more experts onto the GB300 therefore does not make a step
faster; a 285-hot rowmap measured 1-2 % slower than the shipped 255 (2026-10-28, same session).

The tier is armed by the `peer3` component for MoE calls with up to 64 tokens; above that it steps aside. The RTX
is not optional: the GB300 loads 198.6 GiB of the checkpoint and has 252 GiB, so the ~87 GiB of cold experts have
to live on the sidecar. `./ds41 install` refuses to continue without it.

## 3. The fast-path layer

Every kernel-level change in this recipe is a **component**: a Python module that patches a specific function in
vLLM or in the model, registers itself under a name, and is armed by listing that name in the container's
`DS41_FASTPATH`, which the launcher sets from the fixed list in `config/recipe.env`. The hook is small:

```python
from fastpath_hook import register

@register("my_component", "vllm.some.module")
def _patch(module):
    ...  # wrap or replace what you need
```

The list is ordered — order matters where one component wraps another (for example the drafter's input glue must
be applied *before* the boundary component, so that the boundary's fused head sits outside it). The release ships
exactly one list (`config/recipe.env`); there is no stock switch and no per-component switch.

Two rules every component in this repo follows:

* **Bit-identical where it claims to be, explicit where it is not.** A component that replaces a kernel either
  produces the same bits as the kernel it replaces (and says so in its source), or it changes numerics *by design*
  and is listed as such below. A numeric change is a decision with a quality check behind it, never an accident —
  and the stack as a whole is therefore **not** bit-identical to stock vLLM.
* **It can be turned off alone.** No component depends on another being on, except where the ordering above
  requires it.

### What each component does

| component | what it does | state | numerics |
|---|---|---|---|
| `peer3` | GB300 side of peer tier v3: hot/cold expert split for MoE calls ≤ 64 tokens | on by default | changes numerics |
| `megamoe` | scheduling-only fork of DeepGEMM's FP8×FP4 MegaMoE kernel (shared-early pipeline: the shared expert's weights load before the kernel's dependency wait) | on by default | bit-identical tile math |
| `fastload` | boot-time read filter: skip the drafter's non-mtp tensors before reading them | on by default | bit-identical |
| `coldskip` | boot-time read filter: skip the rowmap's cold-expert tensors (computed on the RTX) before reading them | on by default | bit-identical |
| `tail_markov_fused` | DSpark's whole sequential Markov loop in one launch | on by default | bit-identical |
| `tail_swa_meta` | one kernel per sliding-window metadata build for decode batches | on by default | bit-identical |
| `tail_dflash_prep` | the drafter's input preparation for every draft KV group in one launch | on by default | bit-identical |
| `tail_ctxkv` | the drafter's context-KV projection (3 draft layers) as one quantize + one GEMM | on by default | — |
| `tail_replay_skip` | skip the per-step replay upload when the device already holds the values | on by default | — |
| `tail_embed` | next step's input embeddings as one gather straight into the graph's input buffer | on by default | bit-identical |
| `tail_drafthead_mxfp8` | the drafter's draft-head GEMM in MXFP8 | on by default | changes numerics |
| `tail_draft_embed` | the drafter's 6-launch input glue (cast, gather, 3 mask compares, hc expand) as one kernel | on by default | bit-identical |
| `attn` | DeepSeek-V4.1 decode attention on a bitwise-parallel kernel | on by default | bit-identical |
| `preattn` | the pre-attention chain: QKV, candidate keys, indexer, compression (four sub-components) | on by default | bit-identical |
| `boundary` | one-launch mHC boundaries and a one-launch MoE router | on by default | bit-identical |
| `suffix` | copy/suffix drafting on top of DSpark, with a wide verify for batch-1 copy runs | on by default | policy (distribution-preserving) |
| `ds41_sampling`, `ds41_stop` | DeepSeek-recipe serving semantics in the API server | on by default | policy |
| `topp_fast` | exact top-p masking in one launch for the verify/sampling path | on by default | — |
| `w8oproj` | attention output projection `wo_a` + MXFP8 quantize as one kernel | on by default | changes numerics |
| `tail_proj_side` | DSpark's target-hidden projection on a side stream | on by default | — |
| `lowb_aux` | attention side projections on the CuTe DSL low-latency BF16 path for small row counts | on by default | changes numerics |
| `engram_side` | the Engram layers' KV GEMM on the prefetch stream, joined when attention needs it | on by default | bit-identical |
| `quant_producer` | at T = 6 / 8 (one request) the layer-end seam also writes the next attention input as MXFP8, so the wq_a GEMM skips its own quantize launch | on by default | bit-identical (stock rounding, scale and packing) |
| `a12_retire` | MegaMoE scale factors copied once at load into compressible device memory (byte equality checked), originals released before capture | on by default | bit-identical |
| `a12_memory_guard` | after capture, refuse to serve unless free memory covers the measured runtime peak + `DS41_GPU_RESERVE_MIB` + 150 MiB | on by default | — (startup check) |

`bit-identical` = the source asserts equality with the kernel it replaces. `changes numerics` = a deliberate numeric
change with a quality check behind it. `—` = no bit-equality claim either way; the end-to-end quality suite is the
check.

### Which parts are bit-identical

The source asserts bit-equality with the kernel it replaces for: `attn`, the four `preattn` components, `boundary`,
`tail_markov_fused`, `tail_swa_meta`, `tail_dflash_prep`, `tail_embed`, `tail_draft_embed`, `engram_side`,
`quant_producer`, `a12_retire` (a byte-identical copy), `fastload` and `coldskip` (read filters) and `megamoe`
(scheduling only, no arithmetic change).

These change numerics by design: `w8oproj` (o_proj in MXFP8; sparse and drafter layers use a deterministic 2-part
split-K), `lowb_aux` (a different K summation order, closer to
fp64 for the score), `tail_drafthead_mxfp8` (quantized draft head) and `peer3` (the sidecar path accumulates in
bf16 with unordered atomics — 0.9 % rms between replays at M ≥ 2). `suffix`, `ds41_sampling` and `ds41_stop` are
policy, not numerics.

Measured 2026-10-23 with speculation off and identical batch shapes, stock and fast-path outputs diverge from token
0/4/9 on the fixed prompt set. That is expected, and it is why quality is validated by GSM8K-200, the needle test
and the tool call rather than by bit-equality.

## 4. Dynamic draft length and the nested DSpark block

DSpark drafts several tokens ahead, then the target model verifies them in one pass. Two parts of the design
matter:

* **Nested block (5 rows).** The draft block is nested so that rows 0..4 of a 7-row block only ever see
  the first 5 block tokens. That makes the first 5 drafts of a 7-row block *identical* to the drafts a 5-row
  block would produce — which is what makes the dynamic length legal.
* **Dynamic length (5 or 7 tokens).** Per request, each step drafts either 5 or 7 tokens. The
  chooser measures the step cost of each width online and picks the better one; it explores the long width
  periodically so a bad estimate cannot lock it out. Because of the nested block, a 5-token step's drafts are the
  same drafts the 7-token block would have produced, so the verifier's pairs stay consistent and the sampled
  distribution is unchanged.

Why bother: a 7-token step costs about 1 ms more than a 5-token step, and it only pays when the extra two drafts
are likely to be accepted. On reasoning traffic the acceptance tail is thin, so the chooser spends most steps at
5 and still takes 7 when the content rewards it.

## 5. The MegaMoE fork

DeepGEMM's `sm100_fp8_fp4_mega_moe` kernel is the MoE workhorse: it pulls the dispatch, runs the two GEMMs with
SwiGLU and MXFP8 quantisation between them, and combines. The fork in this recipe changes **scheduling only** —
no arithmetic, no tile shapes, no accumulation order — and is bitwise identical to the upstream kernel.

What it changes: the order in which the shared expert's tasks and the routed tasks are published, so the shared
expert's first weight tile is issued before the grid dependency wait and the shared expert's L2 tasks are
published at the routed L1→L2 transition instead of in front of the routed wave. On this box that is worth
−1.8 µs per call at 6 tokens and −0.6 µs at 8 tokens — small, but it is a real per-layer saving and it was
gated bit-for-bit at E=24/40/255 experts over 1..128 tokens before it shipped.

A later revision of the same fork (two more scheduling changes, v12) was built, gated bitwise, armed in-server
and measured as **no gain**; it is not in the recipe. See `WHAT-DID-NOT-WORK.md`.

## 6. The switches

Everything below lives in `~/.ds41/config.env` and has a commented default. You should not need any of them.

| switch | default | what it does |
|---|---|---|
| `DS41_PORT` | 30003 | OpenAI endpoint port |
| `DS41_UTIL` | `DS41_UTIL_DEFAULT` in `config/recipe.env` | fraction of the GB300 the server may use |
| `DS41_GPU_RESERVE_MIB` | 0 | memory another process on the GB300 may still take after boot; the startup guard keeps it free |
| `DS41_MAXLEN` | 524288 | maximum context length |
| `DS41_KV` | `nvfp4_ds_mla` | KV-cache format |
| `DS41_SEQS` | 32 | maximum concurrent sequences |
| `DS41_OFFLOAD_GB` | 120 | host KV overflow in GB |
| `DS41_MODEL_DIR` | `~/.ds41/models/DeepSeek-V4.1-Flash-1017` | checkpoint path; install skips the download if it is complete |
| `DS41_LOG_LEVEL` | `INFO` | container log level |

`config/ds41.env.example` is the complete list, with a comment on every entry.

Three things are **fixed release constants**, not knobs, and live in `config/recipe.env`: the fast-path component
list (`DS41_FASTPATH_COMPONENTS`), the MegaMoE rowmap (`rowmap-mix-v1-h255.json`) and the pinned FlashInfer tune
file (`fiset-w6pf-post1-c7483c4a.json`). The same file holds the default utilisation (`DS41_UTIL_DEFAULT`), which
the operator config may override. The release ships exactly one recipe; there is no no-spec or A/B arm.

## 7. Where the time goes

Per decode step at 6 tokens (one stream, ~8.1 ms):

* MoE layers (40 of them, MegaMoE each): the largest single block.
* The dense chain: attention, the projections, the seam, the router, the LM head.
* The drafter: ~0.6 ms of graph work, of which the Markov loop and the draft-head GEMM dominate.

`THEORETICAL-MAX.md`-style accounting says the floor for this box is ~3.6 ms per step. The shipped recipe is at
~8 ms. The gap is the dense chain's latency and the MoE kernel's fixed cost, which is why the remaining wins in
this repo are all small and local.
