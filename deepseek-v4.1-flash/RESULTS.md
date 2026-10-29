# Results

Every number in `README.md` comes from here, with the run it came from and the protocol it used. If you only
read one section, read "How to reproduce" and "What these numbers do not include".

**Release measurement: 2026-10-28.** Official `vllm/vllm-openai:v0.31.0`, release defaults (`DS41_UTIL` 0.95,
rowmap h255, nothing else on the GB300), a fresh clone, one `ds41 bench --full` session plus `ds41 verify`
(6 PASS / 0 FAIL) and the boot gate (all 29 components ACTIVE, no fallback lines, memory guard PASS). The
2026-10-23 tables further down were measured at `DS41_UTIL=0.84` beside another GPU process, before the
2026-10-28 fast paths; they are kept as history.

## Release run, 2026-10-28 (`ds41 bench --full`)

### Single stream

| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, thinking off) | 733.6 (703.3-751.0) | 138 (136-142) | 5.43 (5.17-5.54) | 135.4 (135.2-135.8) | 10 |
| coding prompt (2,048 out, greedy, thinking off) | 795.5 (791.5-796.5) | 138 (137-140) | 6.31 (6.29-6.32) | 126.1 (125.4-126.1) | 5 |
| GSM8K decode, T=1.0, thinking on (60 s window) | 500.5 | — | 3.826 | — | window |
| GSM8K decode, greedy, thinking off (60 s window) | 588.6 | — | 4.7 | — | window |

### Many streams (GSM8K, T=1.0, thinking on)

| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |
|---|---|---|---|---|
| 4 | 983 | 245.8 | 3.704 | 60 |
| 8 | 1,373 | 171.6 | 3.051 | 60 |
| 16 | 1,936 | 121.0 | 3.065 | 60 |
| 32 | 2,524 | 78.9 | 3.051 | 60 |

### Prefill (uncached)

| prompt tokens | TTFT ms | prefill tok/s |
|---|---|---|
| 14239 (14238-14255) | 388 (364-1290) | 36772 (21633-39303) |
| 56832 (56813-56850) | 1429 (1419-1430) | 39759 (39724-40053) |
| 113688 (113674-113706) | 2966 (2931-2984) | 38324 (38094-38804) |

### Capacity and quality

GPU KV cache 11,372,409 tokens. Sixteen distinct ~110K-token contexts decoding 2,500 forced tokens each, plus a
~434K-token needle request: 0 preemptions, peak KV use 12 %, every answer correct. GSM8K-200 (thinking on,
temperature 0.3) 196/200 = 98.0 %; needles 3/3 at 173,517 tokens; tool call PASS.

### How the defaults were chosen

All with nothing else on the GB300 and `DS41_UTIL` 0.95.

| configuration | GPU KV tokens | fixed-prompt steps/s | GSM8K 1 stream | 8 streams | 32 streams |
|---|---|---|---|---|---|
| h255, without quant_producer/A12 (production tree) | 11.37M | 131.6 | 485 | 1,351 | — |
| h285, without quant_producer/A12 (production tree) | 3.35M | 130.25 | 477 | 1,318 | — |
| rebal-rows (h255 with 37 hot/cold swaps), release tree | 11.37M | 135.8 | 502 | 1,319 | 2,120 |
| **h255, release tree (shipped)** | 11.37M | 135.4 | 501 | 1,373 | 2,524 |

The first two rows ran on the production tree before quant_producer/A12 supported other splits (40 s windows);
compare them only with each other. Keeping more experts on the GB300 (h285) is slower: the RTX computes its
experts in parallel, so moving them to the GB300 adds bytes to the busier GPU and takes KV space. Swapping
popular experts onto the RTX (rebal-rows) is within noise at one stream and loses 4 % at 8 streams and 16 %
at 32, because the RTX's work grows with every token routed to it.

## How to reproduce

```bash
./ds41 bench --full          # the 2026-10-28 suite above, one session, ~40 minutes; JSON plus a markdown table
./ds41 bench                 # the quick check: 1, 4, 8 concurrent streams, coding
./ds41 verify                # six checks: four preflight, a tool call, needles at ~174K tokens
./ds41 verify --full         # adds GSM8K-200, decode and coding checks
```

`ds41 bench` (the quick check) runs the same protocol as the 2026-10-23 runs below:

* **Decode**: a fixed reasoning prompt, temperature 1.0 / top_p 0.95, a 60-second measurement window after
  warm-up, reporting aggregate tok/s, per-stream tok/s, and **tokens per verify step** (the number of tokens
  each speculative round actually emitted). Step time is derived as `tokens_per_step / per_stream_tok_s`, so a
  change in acceptance cannot masquerade as a change in kernel speed.
* **Greedy**: the same prompt at temperature 0, thinking off, 45-second window.
* **Coding**: one batch-1 prompt, 2,048 output tokens, three runs, reporting output tok/s and TTFT.
* **Quality**: GSM8K-200, needles at 173 k tokens, one tool call, and the boot gate.

## The 2026-10-23 table (`DS41_UTIL=0.84`, beside another GPU process)

Reference box: 1 × GB300 (252 GiB), 1 × RTX PRO 6000 (96 GB), 494 GiB host RAM, driver 595.58.03, the shipped
recipe at `DS41_UTIL=0.84` with another GPU process co-resident. Reasoning traffic unless stated. Two passes,
same boot, same clone.

| streams | aggregate tok/s | per stream | tokens/step | derived step time |
|---|---|---|---|---|
| 1 | 475 / 467 | 475 / 467 | 3.81 / 3.76 | 8.02 / 8.05 ms |
| 4 | 970 / 971 | 243 / 243 | 3.63 / 3.65 | 14.97 / 15.04 ms |
| 8 | 1,379 / 1,366 | 172 / 171 | 3.06 / 3.05 | 17.75 / 17.86 ms |
| 16 | 1,941 / 1,933 | 121 / 121 | 3.03 / 3.05 | 24.98 / 25.25 ms |

Across the two passes: **~470** single-stream, **~970** at 4, **~1,370** at 8, **~1,940** at 16 streams.

## Batch-1 coding prompt (2,048 output tokens, greedy, thinking off) (2026-10-23)

| run | output tok/s | TTFT | runs |
|---|---|---|---|
| verify | 750.4 | 163 ms | [750] |
| bench | 737.9 | 146 ms | [745, 730] |

Three runs on the same boot: **750 / 745 / 730 tok/s**, TTFT **146–163 ms**. The spread within one measurement is
the honest repeatability of a single 2,048-token generation, not a different configuration.

## Quality and the boot gate (2026-10-23)

| check | result |
|---|---|
| `ds41 verify` | **9 PASS / 0 FAIL** |
| GSM8K-200 (thinking on, reasoning_effort 50, T 0.3, 16 at a time) | **196/200 = 98.0 %**, 0 runaways, 0 errors, median 160 completion tokens, 44 s wall |
| Needles, 173,517-token prompt, 10 % / 50 % / 90 % depth | **3/3 PASS** (19 s / 9 s / 9 s) |
| Tool call | **PASS** — correct function, correct arguments |
| Boot gate | **PASS** — 40 `ACTIVE` lines across processes (all 25 components), 0 `DS41_FASTPATH armed`, 0 `falling back`, 0 `stock:`, 0 `No tuned config covers`, 0 `Ignoring cached configs`, 0 `will be REJECTED`, 0 `FATAL`, 0 `Traceback` |

The boot gate is the check that the release is actually running the recipe: every component must print its
`ACTIVE` line and no code path may print a fallback line.

## What the shipped changes are worth, individually (2026-10-23)

Development measurements on the recipe's own levers (not part of the headline table, which is end to end). They
are small on purpose: the recipe is a stack of ~1 % wins, and each one had to pass its check (a quality run, or
bit-equality against the kernel it replaces) before it was allowed in.

| lever | what it saves | how it was measured |
|---|---|---|
| Drafter input glue (6 launches → 1) | **−12 to −13 µs per step**, every step, both draft widths | drafter graph span and seam landmark over ~150 real captures: K5 −12.0 µs span / −13.0 µs landmark, K7 −12.8 µs landmark |
| Fast load (skip the second weight load) | start-up only: second "Loading weights took" **< 10 s** instead of minutes | boot log comparison |
| Dynamic draft length (5 vs 7) | 0.35 % of tokens per unit cost, and it stops the "always 5" lock-in | simulated streams + in-server A/B |
| Nested DSpark block (5 rows) | keeps drafts 1..5 identical to a full 7-block, which is what makes the 5-block path legal | device check: 10,752 comparisons, 0 draft mismatches |
| MegaMoE fork (v11) | per-call kernel time, measured at −1.8 µs (T=6) / −0.6 µs (T=8) per call | per-width trace attribution |

## What these numbers do not include

* **No prefix cache in the headline rows.** The decode windows start cold, so a real chat workload with a warm
  prefix cache is faster than the table, not slower.
* **Reasoning traffic at temperature 1.0.** The greedy numbers are a different mode; do not compare a greedy
  number with a sampled one.
* **A co-resident GPU process and `DS41_UTIL=0.84`.** The final run shared the GB300 with another process and
  left 16 % of it free. The dedicated-machine default is 0.92: that changes KV-cache size, not decode speed, and
  was not re-measured here.
* **First boot is slow.** With cold caches and a fresh home, boot to healthy was **835 s**. Later boots reuse the
  caches in `~/.ds41/state` and are faster; the warm time was not re-measured for this tree.
* **Context length matters.** All decode numbers are at short context (< 4 k). At 100 k+ tokens the KV cache
  reads dominate and per-stream throughput drops; that is expected and not measured here.

## History: the nightly-image era

The numbers below were measured on the previous nightly image
(`vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b`, vLLM 0.30.1rc1.dev245+gaf7f9488c) before the
release moved to v0.31.0. They are kept for comparison only; they are **not** the published numbers.

| streams | aggregate tok/s | tokens/step | run |
|---|---|---|---|
| 1 | 467 / 464 / 460 | 3.78 / 3.70 / 3.66 | `prod-numbers-1021`, `prod-no-tenant-1021` passes 1, 2 |
| 4 | 992 / 982 / 956 | 3.73 / 3.71 / 3.62 | same |
| 8 | 1,369 / 1,368 / 1,371 | 3.06 / 3.07 / 3.06 | same |
| 16, reasoning | 1,970 | 3.05 | `prod-c16-1023` |
| 16, greedy (thinking off) | 2,260 | 3.39 | `prod-c16-1023` |

Coding (2,048 tokens, greedy, thinking off): 724.7 / 742.1 / 715.6 tok/s across the three runs. GSM8K-200
196/200, median 161 completion tokens. Needles 3/3 at 173,517 tokens (5 s / 5 s / 7 s).

## Provenance

| | |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017: MXFP4 routed experts, FP8 dense weights, FP8 Engram tables in host memory, NVFP4 compressed KV cache (`nvfp4_ds_mla`: MXFP8 sliding window + NVFP4 compressed) |
| Image | `vllm/vllm-openai:v0.31.0` |
| Tree | commit `b4edcad`, fresh clone |
| Driver / CUDA | 595.58.03 / CUDA 13.0 (in-image runtime) |
| GPUs | GB300 252 GiB usable; RTX PRO 6000 96 GB |
| Host RAM | 494 GiB; ~317 GiB in use while serving (FP8 Engram tables pinned + 120 GB host KV overflow) |
| Weights | 476 GB on disk, official DeepSeek release |
| Utilisation | 0.84, beside another GPU process; the dedicated-machine default is 0.92 (KV cache only, not speed; untested here) |
| KV cache | ~927 k tokens at `DS41_UTIL=0.84` (1.77× a 524,288-token request) |
| Boot | 835 s to healthy with cold caches (fresh home) |
| Date | 2026-10-23 |
| Logs | `release-v031-final-1023` (`F-verify.log`, `F-bench.log`, `F-gate.txt`, `F-boot-time.txt`) |
