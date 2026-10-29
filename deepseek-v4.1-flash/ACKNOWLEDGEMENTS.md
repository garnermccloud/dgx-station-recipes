# Acknowledgements

This recipe is a thin layer of scheduling and glue on top of a lot of other people's work. Almost none of the
hard parts are ours, and it is worth saying exactly whose they are.

## Where this started

**original-el8 / `dgx-station-gb300-research`.** The recipe itself comes from here: splitting the experts so the
hot ones stay on the GB300 and the cold ones live on an RTX PRO 6000 sidecar, and the "M3" lane this serving stack
descends from, are original-el8's work, published openly. Their notes on how the memory actually behaves on that
box are the reason any of this runs at all. If you are starting from nothing, start there; it will save you a
month. The published tree is the credit: the original-el8 lane is the source.

The **original-el8 lane** wrote the MegaMoE peer hook and much of the peer-tier work that this repo's `megamoe` and `peer3`
components build on — the part that makes a second GPU look like extra HBM instead of a network hop.

**local-inference-lab.** Their vLLM fork and their notes on serving large MoE models on a single box shaped a lot
of the decisions here — particularly what "good" looks like at one stream, and which parts of the stack are worth
touching at all.

## The model and the kernels underneath

**DeepSeek** — for the model, and for three things that make this recipe possible at all: **DeepGEMM**, whose
`sm100_fp8_fp4_mega_moe` kernel is the MoE workhorse here (we forked its *scheduling* and nothing else), **DSpark**,
the drafting scheme that gives us 3–4.8 tokens per verify step, and the MXFP4/FP8 weight formats that let 476 GB
of weights stream through one GPU at a useful rate. Their kernel code is unusually readable for how fast it is.

**vLLM** — for the engine this is all bolted onto: the V1 scheduler, the CUDA-graph capture path, the
OpenAI-compatible server, and the extension points that make a component layer like this possible without forking
the world. Every one of our fast-path components patches a function that a vLLM maintainer wrote and kept clean
enough to patch. Their issue tracker (#61868, #61869 and the surrounding threads) is also where half of our
roadmap came from.

**FlashInfer** — for the low-latency MXFP8 GEMM path used by the attention side projections, and for the CuTe DSL
kernels that make small-row GEMMs stop being embarrassing. **b12x** deserves a nod as its own project: the sidecar's
inference kernels came from there, and b12x has since moved into FlashInfer (`flashinfer/experimental/b12x`) —
which is the right outcome for good code.

## The public work that informed the design

* **NVIDIA's megakernel work and SemiAnalysis' write-up of it** — the clearest public statement that a decoder
  layer's problem is launch and dependency latency, not FLOPs. It reframed our whole to-do list.
* **SGLang** and **BBuf's kernel series** (`sglang` PR #44626 and the DSV4 kernel directory) — the best public
  reference for what a fused decode step looks like on Blackwell, and a steady source of "oh, they already did
  that".
* **Cohere's megakernel lessons** — the most useful write-up on what *doesn't* survive contact with a real
  scheduler, and the reason we tried putting weights before the dependency wait.
* **TensorRT-LLM's router GEMM note** — a small thing (8 rows → 3 µs) that saved us a day.
* **sgl-amd-mono's persistent-kernel design** — a design reference we read and did not copy, and the reason we
  know where the ceiling is on a persistent approach.
* **RonanLabs** — for two things. The GSM8K evaluation harness we used to check that the fast paths do not cost
  accuracy; it is the reason the 98.0 % number is comparable with anyone else's. And
  `RonanLabsAI/dgx-station-frontier` (the `recipes/dsv41-flash-sidecar` recipe), a public implementation of the
  same DGX Station lane, which was the most useful thing to compare notes against.
* **Fraser Price's BRRR** — a public DeepSeek-V4.1-Flash speed recipe and bench. Its coding task is our fixed
  single-stream prompt (Apache-2.0), so our single-stream numbers can be read against that bench's.
* The **vLLM**, **SGLang** and **FlashInfer** maintainers and issue-thread regulars who answer questions about
  kernels they did not have to answer questions about.

## The boring infrastructure that makes it work

Docker and the NVIDIA Container Toolkit, the CUDA driver team (595.58.03 has been boring in the best way), the
PyTorch and Triton compiler folks — every kernel in here is either Triton or compiled by nvcc through their
toolchain — and Hugging Face, for hosting the weights and for making a model card a useful thing rather than a
formality.

## And

The person who owns the box, paid for it, and decided that the results were worth publishing instead of keeping in
a drawer. This repo exists because of that decision.

---

Third-party components keep their own licences; the texts and notices are collected in
[`NOTICE`](NOTICE). If you think you should be on this list and are not, open an
issue — it is an omission, not a judgement.
