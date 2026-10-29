---
language:
  - en
  - zh
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: vllm
pipeline_tag: text-generation
tags:
  - deepseek
  - mixture-of-experts
  - fp8
  - nvfp4
  - blackwell
  - gb300
  - dgx-station
  - speculative-decoding
  - single-node
  - vllm
---

# DeepSeek-V4.1-Flash-1017 — DGX Station GB300 serving recipe

This repository is a **serving recipe**, not a model. It does not contain weights. It tells you how to run the
official [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) checkpoint on one
NVIDIA DGX Station GB300 plus one RTX PRO 6000, and it gets you this:

| | |
|---|---|
| One stream, code | **~730 tok/s** (fixed coding prompt, greedy; ~800 on the longer coding prompt) |
| One stream, reasoning | **~500 tok/s** (temperature 1.0, thinking on) |
| 4 / 8 / 16 / 32 streams | **983 / 1,373 / 1,936 / 2,524 tok/s** aggregate |
| Prefill | ~38–40K tok/s; first token in 3.0 s at 114K tokens |
| Context | up to 512K tokens per request; 11.4M-token GPU KV cache |
| Quality | GSM8K-200 **98.0 %**, needles **3/3** at 173,517 tokens, tool calls pass |

Measured on the release: `vllm/vllm-openai:v0.31.0`, release defaults (`DS41_UTIL` 0.95, nothing else on the
GB300), a fresh clone, one `ds41 bench --full` session (2026-10-28). The protocol and the raw numbers are in
the repo's `RESULTS.md`.

## How it works, in one paragraph

The checkpoint is DeepSeek-V4.1-Flash-1017 (MXFP4 routed experts, FP8 dense weights, FP8 Engram tables in host
memory, NVFP4 compressed KV cache). It is ~24.4 GB of weights to read per decode step, and the GB300's HBM
sustains 6.84 TB/s, so the floor for a step is ~3.6 ms. The recipe splits the MoE layers across three tiers — hot experts on the GB300, cold
experts on the RTX PRO 6000 sidecar, the rest in host memory — and then removes fixed costs with a set of
kernel-level fast paths: a bit-identical DeepGEMM MegaMoE scheduling fork, a fused DSpark Markov loop, a one-launch
drafter input glue, fused mHC boundaries and router, and dynamic draft length (5 or 7 tokens per step) on top of a
nested DSpark block. Most fast paths are bit-identical to the kernel they replace; a few (`w8oproj`/o_proj in
MXFP8, `lowb_aux`, the quantized draft head, the peer tier's bf16 sidecar path) change numerics by design, so the
stack as a whole is not bit-identical to stock vLLM. Quality is validated by GSM8K-200, needles and the tool-call
check, not by bit-equality. Each component has its own switch and can be turned off alone.

## Run it

```bash
git clone https://github.com/garnermccloud/dgx-station-recipes.git && cd dgx-station-recipes/deepseek-v4.1-flash
./ds41 install      # preflight, pulls the official vllm/vllm-openai:v0.31.0 image, weights
./ds41 up           # first boot compiles kernels (~14 min with cold caches); later boots are faster
```

The release runs the stock `vllm/vllm-openai:v0.31.0` image and mounts the recipe into it; there is no image build.

Then:

```bash
curl -s http://localhost:30003/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/.ds41/api-key)" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

`./ds41 status` gives you one screen; `./ds41 verify` gives PASS/FAIL per check and compares your speed with the
numbers above.

**Hardware you need:** a GB300 (252 GiB usable) **and** an RTX PRO 6000 (96 GB) — the RTX is required, the model
does not fit in the GB300 alone — **the Station's ~496 GB LPDDR5X** (about 320 GB of it is in use while serving:
the FP8 Engram tables pinned in host memory plus a 120 GB host KV overflow), **700 GB of NVMe**, driver 595.58.03
or newer, and Docker with the NVIDIA Container Toolkit. The image is `linux/arm64`: this runs on the GB300's Grace
CPU, not on x86. No CUDA toolkit, no Python and no compilers on the host.

## What is in the repo

* `README.md` — quick start, requirements, everyday commands, performance and quality with methodology.
* `docs/ARCHITECTURE.md` — the peer tier, every fast-path component and its switch, dynamic draft length, the
  MegaMoE fork.
* `docs/LIMITATIONS.md` — read this before you build on it: identical greedy requests can diverge past ~512 tokens
  of context (upstream top-k tie ordering), thinking mode is always sampled, first start compiles kernels.
* `docs/WHAT-DID-NOT-WORK.md` — the measured negative results. Genuinely useful if you are about to try one.
* `RESULTS.md` — every measurement with its protocol and its run-to-run variance.
* `ACKNOWLEDGEMENTS.md` — whose work this stands on.

## Scope and honest caveats

* **This box, this checkpoint.** The kernels, the peer tier and the tuning target GB300 + RTX PRO 6000 and this
  checkpoint's module structure. It will not run on Hopper, and an RTX 5090 is not a substitute for the PRO 6000.
* **Determinism.** Past ~512 tokens of context, two identical greedy requests can diverge — a known
  characteristic of the current vLLM top-k, which places the selected indices with an atomic counter, so their
  order (and occasionally the tie set) varies between runs. Below that, every layer is bit-identical. The details
  and our mitigation are in `docs/LIMITATIONS.md`.
* **Thinking mode is sampled, never greedy**, by design; the client's temperature is honoured only with thinking
  off.
* **No weights here.** Get them from the official release:
  [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). The weights
  are MIT, like this recipe's own code.

## Credits

The model, DeepGEMM and DSpark are DeepSeek's. The engine is vLLM's. The low-latency GEMM paths are FlashInfer's,
and the sidecar's kernels started life in b12x (now part of FlashInfer). The lane itself grew out of
`original-el8/dgx-station-gb300-research` and `local-inference-lab`'s work, and the design owes a lot to NVIDIA's
megakernel write-ups, SGLang's kernel series, Cohere's megakernel lessons and the public vLLM issue threads. Full
credits in `ACKNOWLEDGEMENTS.md`; third-party licence texts in `NOTICE`.

## Citation

If this is useful to you, cite the model and link this recipe:

```bibtex
@misc{deepseek_v41_flash_dgx_station,
  title  = {DeepSeek-V4.1-Flash-1017 on a DGX Station GB300: serving recipe},
  author = {the ds41 contributors},
  year   = {2026},
  note   = {https://github.com/garnermccloud/dgx-station-recipes},
}
```
