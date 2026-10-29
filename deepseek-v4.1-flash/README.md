# DeepSeek-V4.1-Flash on a DGX Station GB300 (with an RTX PRO 6000 sidecar)

This repo turns one NVIDIA DGX Station GB300 plus one RTX PRO 6000 into a fast, private
DeepSeek-V4.1-Flash server: an OpenAI-compatible endpoint that does **~730 tok/s single-stream on code**
(~500 on sampled reasoning), **~1,370 tok/s at 8 concurrent streams** and **~2,500 at 32**, with an
11.4-million-token KV cache and the model's weights split across the two GPUs by a peer tier
(hot experts on the GB300, cold experts on the RTX) and a set of kernel-level fast paths. Most replace a kernel
with a bit-identical one; a few change numerics by design (the output projection runs in MXFP8, for example), so
the stack as a whole is **not** bit-identical to stock vLLM — quality is validated by the checks below, not by
bit-equality.

You do not need to be a kernel engineer to run it. One command checks your machine, one starts it.

## Quick start

```bash
git clone https://github.com/garnermccloud/dgx-station-recipes.git && cd dgx-station-recipes/deepseek-v4.1-flash
./ds41 install          # checks the box, pulls vllm/vllm-openai:v0.31.0, downloads the weights (~40 min on a good link)
./ds41 up               # first boot compiles kernels (~15 min with cold caches); later boots take ~9 min
```

The release uses the stock `vllm/vllm-openai:v0.31.0` image; it does not build one. At launch the recipe mounts,
read-only from this checkout: the 1017 frontend overlay, the vLLM #58132 decoder overlay, `sitecustomize.py`,
`fastpath/`, the peer hook with its rowmap and tuning files, and b12x. `ds41 install` pulls the image.

Then ask it something:

```bash
curl -s http://localhost:30003/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/.ds41/api-key)" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Explain what a KV cache is, in two sentences."}],"max_tokens":128}'
```

`./ds41 status` gives you one screen: healthy or not, the endpoint URL, a copy-paste curl, GPU memory and uptime.

That is the whole quick start. Everything below is optional reading, in the order you are likely to want it.

---

## What you get

| | |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017: MXFP4 routed experts, FP8 dense weights, FP8 Engram tables in host memory, NVFP4 compressed KV cache (`nvfp4_ds_mla`: MXFP8 sliding window + NVFP4 compressed) |
| Endpoint | OpenAI-compatible `http://localhost:30003/v1` (`/chat/completions`, `/completions`, `/models`) |
| Speed, 1 stream | ~730 tok/s on code (fixed prompt, greedy; ~800 on the longer coding prompt), ~500 tok/s on reasoning at temperature 1.0 |
| Speed, many streams | 983 / 1,373 / 1,936 / 2,524 tok/s aggregate at 4 / 8 / 16 / 32 streams |
| Prefill | ~38–40K tok/s; first token in 0.4 s at 14K tokens, 1.4 s at 57K, 3.0 s at 114K |
| Context | up to 512K tokens per request; an 11.4M-token GPU KV cache |
| Quality | GSM8K-200 98.0 %, needles 3/3 at 174K tokens, tool calls pass |
| Drafting | DSpark speculative decoding, draft length chosen per request from 5 or 7 tokens |

## Requirements

**Hardware**

* **GB300** with 252 GiB of usable HBM3e, 256 GB nominal (the dense weights, the hot experts, attention and the
  KV cache live here).
* **RTX PRO 6000 Blackwell** with 96 GB GDDR7 — **required.** It holds the cold experts (~87 GiB of them); the
  GB300 loads 198.6 GiB and has 252 GiB, so the model does not fit without the sidecar.
* **Host RAM: the Station's ~496 GB LPDDR5X.** While serving, about **320 GB** of it is in use: the FP8 Engram
  tables pinned in host memory plus a 120 GB host KV overflow (`DS41_OFFLOAD_GB`). Measured 316.7 GiB on a live
  server. This is the requirement people underestimate.
* **Disk: 700 GB free NVMe.** The container image is ~34 GB and the weights are 476 GB, plus scratch for the
  kernel cache and the CUDA graphs.
* One machine, no networking between machines. The two GPUs talk over PCIe inside the box.

**Software**

* NVIDIA driver **595.58.03** or newer (CUDA 13.0 runtime ships in the image; you do not install CUDA yourself).
* Docker with the **NVIDIA Container Toolkit** (`docker run --gpus all nvidia-smi` must work).
* Linux **aarch64** (Grace) — what a DGX Station GB300 runs. Nothing else: no Python, no CUDA toolkit, no
  compilers on the host.

`./ds41 install` checks every one of these before it downloads anything, and tells you the exact fix if one is
missing. Example of what it prints when something is wrong:

```
GB300 ....................... PASS
RTX ......................... FAIL
      found only the GB300 / Fix: install the RTX PRO 6000 (96 GB); the model does not fit in the GB300 alone.
```

## Everyday commands

| command | what it does |
|---|---|
| `./ds41 up` | start the server (zero config; auto-detects both GPUs) |
| `./ds41 down` | stop/remove both containers and return `~/.ds41/state` to you (always safe to run) |
| `./ds41 status` | one screen: health, endpoint, sample curl, GPU memory, uptime |
| `./ds41 verify` | preflight (image, weights, GPUs, boot gate), a tool call and needles, one PASS/FAIL line each; `--full` adds GSM8K, decode and coding |
| `./ds41 bench` | quick speed check; `--full` runs the whole published suite in one session (~40 min) and writes JSON plus a markdown table |
| `./ds41 logs` | follow the server log (`--boot` for the last start, `--kernel` for the JIT) |
| `./ds41 update` | refresh the recipe (git pull) and pull the official image; keeps your config |
| `./ds41 install` | safe to re-run at any time; it resumes where it stopped |
| `./ds41 uninstall [--weights]` | asks first; stops containers, fixes state ownership, removes `~/.ds41` (keeps weights that live inside it); weights are deleted only with `--weights` and typing the full path |

`./ds41 verify` is the one to run after any change. It runs six checks: four preflight checks, a tool call and
needles at ~174K tokens. `./ds41 verify --full` adds GSM8K-200, decode and coding checks.

```
preflight image PASS (vllm/vllm-openai:v0.31.0; DeepGEMM base hashes; overlay targets)
preflight weights PASS (48 shards)
preflight GPUs PASS (GB300 + RTX PRO 6000)
preflight boot PASS (all components ACTIVE; W6 MegaMoE; peer3 finish fold; FlashInfer configs loaded; memory guard PASS; no fallback lines)
tool call PASS [('get_weather', '{"city": "Denver"}')]
needle depth=10% prompt_tokens=173517 time=9s PASS reply='violet-anchor-5817'
needle depth=50% prompt_tokens=173517 time=7s PASS reply='violet-anchor-5817'
needle depth=90% prompt_tokens=173517 time=4s PASS reply='violet-anchor-5817'
needle 3/3
== 6 PASS, 0 FAIL
```

## Performance, and how it was measured

All numbers below come from one `./ds41 bench --full` session on the reference box: one GB300 plus one RTX PRO 6000
(driver 595.58.03), the official `vllm/vllm-openai:v0.31.0` image, release defaults (`DS41_UTIL` 0.95, rowmap h255,
nothing else on the GB300), a fresh clone, 2026-10-28. Medians, with the interquartile range in brackets where a
shape was run more than once. "Tokens per step" is how many tokens each speculative-decoding round actually
committed; it moves with the content, so it sits next to every speed.

**Single stream**

| shape | output tok/s | time to first token | tokens/step | steps/s |
|---|---|---|---|---|
| fixed coding prompt (~2.6K in, 2,048 out, greedy, thinking off; 10 runs) | 734 (703–751) | 138 ms | 5.43 | 135.4 |
| coding prompt (~4.4K in, 2,048 out, greedy, thinking off; 5 runs) | 796 (792–797) | 138 ms | 6.31 | 126.1 |
| GSM8K, temperature 1.0, thinking on (60 s) | 501 | — | 3.83 | — |
| GSM8K, greedy, thinking off (60 s) | 589 | — | 4.70 | — |

**Many streams** (GSM8K, temperature 1.0, thinking on, 60 s each)

| streams | aggregate tok/s | per stream | tokens/step |
|---|---|---|---|
| 4 | 983 | 246 | 3.70 |
| 8 | 1,373 | 172 | 3.05 |
| 16 | 1,936 | 121 | 3.07 |
| 32 | 2,524 | 79 | 3.05 |

**Prefill** (uncached prompts)

| prompt | time to first token | prefill tok/s |
|---|---|---|
| ~14K tokens | 0.39 s | ~37K |
| ~57K tokens | 1.43 s | ~40K |
| ~114K tokens | 2.97 s | ~38K |

**Long context and many agents.** The GPU KV cache holds **11.4 million tokens** (up to 512K per request).
Sixteen distinct ~110K-token contexts decoding 2,500 tokens each, plus a ~434K-token needle request, used 12 %
of it with no preemptions, and every answer was correct.

The fixed prompt is the coding task from Fraser Price's BRRR bench, repeated ten times, so single-stream numbers
can be read against that bench's (see `NOTICE`).

**What moves the numbers**, in order: how many drafted tokens the content lets each step accept (about 3 on
sampled reasoning at many streams, 6+ on code), how many streams are in flight, and whether the prompt hits
the prefix cache. Expect a few percent of run-to-run spread on single-stream numbers; acceptance varies more
than step time.

`RESULTS.md` has the raw numbers and the configuration sweep behind the defaults, and `./ds41 bench --full`
reproduces all of this on your box in about 40 minutes.

## Quality

Measured on the shipped recipe, same box:

* **GSM8K-200** (thinking on, temperature 0.3, 200 problems): **196/200 = 98.0 %**.
* **Needles at a 173,517-token prompt**: 3/3 at 10 %, 50 % and 90 % depth.
* **Tool call**: correct function and arguments.
* **`ds41 verify`: 6 PASS / 0 FAIL**, including a boot gate that asserts all 29 components print `ACTIVE` and
  that no fallback line appears in the startup log.
* **Not bit-identical to stock, by design.** Most fast paths are bit-identical to the kernel they replace;
  `w8oproj` (o_proj in MXFP8), `lowb_aux` (a different K summation order) and the quantized draft head change
  numerics deliberately. Measured 2026-10-23 with speculation off and identical batch shapes, the outputs diverge
  from token 0/4/9. Quality is therefore validated by the checks above, not by bit-equality; `docs/ARCHITECTURE.md`
  lists which is which.

The one quality caveat is real and documented: past ~512 tokens of context, identical greedy requests can
diverge from each other (upstream top-k tie ordering, not this recipe). See `docs/LIMITATIONS.md` before you
build anything that depends on bit-exact reproducibility. That is a separate issue from the stock-vs-fast-path
difference above: even with a fixed configuration, the recipe is not bit-identical to stock.

## Configuration

`./ds41 up` needs no configuration. If you already have the weights, set `DS41_MODEL_DIR` in
`~/.ds41/config.env` before `./ds41 install` and it skips the download. Otherwise, when you want to change
something, `~/.ds41/config.env` is one file with every operator knob commented and its default set to the shipped
recipe:

```bash
# ~/.ds41/config.env  — uncomment only what you want to change
# DS41_PORT=30003                  # OpenAI endpoint port
# DS41_MODEL_DIR=/path/to/DeepSeek-V4.1-Flash-1017   # weights you already have; install then skips the download
# DS41_UTIL=0.95                   # fraction of the GB300 the server may use (default: DS41_UTIL_DEFAULT, config/recipe.env)
# DS41_GPU_RESERVE_MIB=0           # memory another process on the GB300 may still take after boot
# DS41_MAXLEN=524288               # maximum context length
# DS41_OFFLOAD_GB=120              # host KV overflow
```

The full list, with what each switch does and when to touch it, is in `docs/ARCHITECTURE.md`. You should not
need any of it to run the server. The fast-path component list, the MegaMoE rowmap and the pinned FlashInfer tune
file are **fixed release constants** in `config/recipe.env`, not knobs — the release ships exactly one recipe.

## If something goes wrong

| symptom | what to do |
|---|---|
| `install` stops at a preflight check | it prints the exact fix; nothing has been downloaded yet |
| the first `up` takes longer than later ones | the kernels compile once and are cached; with cold caches the first boot to healthy is ~15 min (~9 min after that), and later boots reuse the caches in `~/.ds41/state` |
| `status` says unhealthy | `./ds41 logs --boot`, then `./ds41 verify` |
| out of memory on start | something else is using the GB300; `./ds41 status` shows who, `nvidia-smi` shows what |
| want to rule the fast paths out | not a runtime option: the release ships one fixed recipe (`config/recipe.env`). Run `./ds41 verify` first. |

Nothing is left half-started: `./ds41 down` always cleans up, `./ds41 uninstall` removes the runtime and returns
the state tree to you, and `./ds41 install` can be re-run at any point.

## Going deeper

* `docs/ARCHITECTURE.md` — how the peer tier, the fast-path components and the MegaMoE fork work, and every switch.
* `docs/LIMITATIONS.md` — determinism past 512 tokens, forced sampling in thinking mode, first-boot JIT, and the
  other things this recipe does not fix.
* `docs/WHAT-DID-NOT-WORK.md` — the measured negative results. Read this before you spend a weekend on an idea.
* `docs/BUILD-FROM-SOURCE.md` — why this release does not build an image; kernel rebuild notes only.
* `RESULTS.md` — every measurement, with its protocol.
* `ACKNOWLEDGEMENTS.md` — the projects and people this stands on.
* `hf/MODEL_CARD.md` — the Hugging Face card; the weights are the official DeepSeek release, not re-hosted here.

## Licence

The recipe, scripts and documentation in this repo are released under the MIT licence (see `LICENSE`).
The model weights are DeepSeek's, released under the MIT licence — this repo downloads them from the
official release and does not redistribute them. Third-party components keep their own licences; see
`NOTICE`.
