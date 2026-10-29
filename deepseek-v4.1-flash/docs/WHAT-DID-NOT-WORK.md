# What did not work

The measured negative results, so you can skip them. Every item here was built, gated and measured on this
box — none of it is a guess. Numbers are per call or per step as labelled.

**How a lever was judged.** First a gate against the kernel it replaces (bit-equality where the change claims it,
otherwise a documented numeric change with a quality check). Then one in-server A/B against a same-window baseline, judged per
draft width from the trace: the drafter graph's span and the time to a fixed landmark kernel, split into 5-token
and 7-token steps. Anything that could not beat the measured drift (span ±3.2 µs, landmark ±1.7 µs) was dropped,
even when it looked good in the aggregate tok/s — because aggregate tok/s moves with acceptance, and acceptance
moves with the content.

## 1. Side-stream weight prefetch (`wo_b` on its own stream)

Prefetching the next layer's `wo_b` weights on a side stream, with `evict_last` L2 hints, made `wo_b` itself
0.9 µs faster — and the layer 9.7 µs slower: router→MegaMoE end **+5.6 µs/layer** (+8.5 µs on window layers),
attention **+2.5 µs**, plain-layer period **147.8 → 157.5 µs**.

The cause is that the ~29 MB of prefetched lines sit in L2 while the next kernel (MegaMoE) needs to stream
*through* L2, and the side-stream fork/join costs 2.3–4.3 µs per layer on its own. Prefetching into a cache the
next kernel must evict is a tax, not a gift.

If you want to retry it: put the prefetch on the main stream, as a prologue, with `evict_normal`/`evict_first` —
never `evict_last` — and measure the *layer period*, not the kernel you helped.

## 2. L2 persistence / evict hints for expert weights

No gain in-server, rejected. MegaMoE runs **one accumulation chain per SM** and is consumer-bound at about
6.7 TB/s, so making expert bytes L2-resident does not speed it up: the kernel is not waiting on the cache. The
L2 is 2 × 57 MiB, far too small to hold a useful expert working set (14.8 GB of hot experts per step) anyway.

The order that would matter: give the consumer more chains first (four or more chains reaches 13 TB/s
SMEM-resident, 9.6 TB/s L2-fed in measurement), *then* worry about where the bytes live. L2 hints without that
are noise.

## 3. Launch-order edges on GB300 / CUDA 13 (`MIX_ORDER`)

The idea: add a launch-order edge from the mHC mix to MegaMoE so the MoE kernel is resident earlier. The output
was bit-identical in every case (six checks), but the diagnostic that the edge preserves launch completion
failed on both variants: `order lead ≥ none lead − 0.5 µs` and `nofork lead ≥ none lead − 0.5 µs` were both
false. In other words the edge did not buy the residency it was supposed to buy on this driver, and there was
nothing left to time. The PDL-attribute variant did pass its lead check (≤ −0.5 µs), but that win would have
been inside the noise.

Worth knowing: on GB300 + CUDA 13, an edge between two kernels that are already back-to-back buys less than you
expect, because the launch latency is not what you are waiting on.

## 4. MegaMoE fork v12

Two more scheduling changes on top of the shipped fork: issue the shared expert's first weight tile *before* the
grid-dependency wait, and publish the shared expert's L2 tasks at the routed L1→L2 transition instead of in
front of the routed wave. Both are pure scheduling; the fork stayed bitwise identical (gated at 24/40/255
experts over 1..128 tokens), and it was armed in-server.

Measured: **no gain.** Acceptance-corrected step time was +0.3 % to +2.2 % (i.e. if anything slightly slower),
and the largest step-time sample (n ≈ 16 k) was +0.3 %. The +3.7 % single-stream tok/s that showed up in the arm
was entirely the dynamic-draft mix: 3.81 versus 3.61 tokens per step. One mechanism-level warning came out of
it: the predecessor kernel's grid-dependency wait rose from p90 0 µs to 4–5 µs, consistent with the early TMA
traffic contending with the kernel it was supposed to overlap.

The shipped fork is the earlier revision; v12 is not in the recipe.

## 5. Moving the mHC mix off the main path (`MIX_OFFPATH`)

The mix kernel is small but sits on the critical chain. Moving it off the main path requires **reserving SMs for
it** (`MEGA_SMS=144` of 152): without the reservation the MoE kernel takes every SM and the mix starves, which is
worse than leaving it on the chain. With the reservation the reservation's own cost is larger than the overlap
saves. And the launch-order arm that motivated the whole exercise failed its diagnostic (see 3).

## 6. Rebalancing expert bytes toward the RTX

Giving the RTX PRO 6000 a larger share of expert bytes: C4 992 → **974**, C8 1,369 → **1,343** (worse at
multi-stream), while C1 greedy 549 → **580** and coding 725 → **756** (better at batch 1). Not a clean win, so it
was not adopted.

The RTX is busy only about 22 % of the step at the shipped share, so the headroom is real — but the transfer
cost per byte eats it at multi-stream, where the GB300 is already saturated. A rebalance only pays if it comes
with cheaper transfers (bigger batching per transfer, or a topology where the sidecar's bytes arrive without
competing with the main stream).

## 7. Fused rejection sampler

One persistent kernel for the whole rejection chain (stats → cumulative log-prob → residual mass → rejection →
resample → insert), replacing five small launches and their drain gaps. It passed its bitwise gate, and it was
armed — but the A/B could not be judged in that window because the trace capture came back empty, so it was never
adopted. The design estimate was −14..−20 µs per step at one stream.

If you pick this up: the risk is the reduction layout. The stock phases run with different `num_warps`; one fused
kernel has one, so the `tl.sum`/`tl.max` order over the 16-block vectors has to be checked equal, or the
per-block combines written out in the stock tree order.

## 8. Five-block drafter (`W6_DYNK_BLOCK5`)

On a 5-token step, draft a captured 5-block graph instead of the 7-block one. It was verified bitwise on the
device — **10,752 comparisons, 0 draft mismatches, 0 q-row mismatches** — and then armed and judged per draft
width: the 5-token drafter span moved **−6.6 µs** (the bar was ≥ 15 µs to justify the second graph), and the
7-token span moved −8.4 µs, which is drift (that graph is untouched by the change). Not adopted.

Why it disappointed: the dynamic-length early exit already removes most of what the 5-block was meant to remove
(the Markov loop and the tail of the block), so what is left is ~7 µs of MegaMoE and attention work, not 24.

## 9. Adaptive verification (confidence-based draft length)

vLLM's adaptive verification for DSpark: **no gain at one stream, −20 % on prose at 16 streams** — acceptance
collapses to about 1.01 at 16 streams because the confidence estimate is per-request and the batch shares the
step. Rejected. The dynamic length in this recipe is a different mechanism: it measures the *cost* of each width
per request and never lowers the draft count below what the verifier can use.

## 10. Two smaller ones

* **First-task prefetch inside MegaMoE** (claim the first task earlier so its loads start sooner): +1–2 µs, i.e.
  a small cost, because the claim was never the critical path. The shipped fork instead moves the *shared
  expert's* tile, which is statically known and needs nothing from the predecessor.
* **Contiguous FP4 expert tiles**: +1 % with scales, already at the FP4 ceiling for the stream. Not worth a
  layout change; the dense FP8 weights are where a contiguous relayout pays (~10 % of dense streaming time).
