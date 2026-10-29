# fastpath

The production fastpath tree: Python patch components, prebuilt CUDA kernels, the peer-tier sidecar, and
the MegaMoE fork headers.

`components.py` imports only the components used by the default recipe. `sitecustomize.py` is baked over
the image's `sitecustomize.py` (or mounted when a checkout is used); it installs the MegaMoE hot/cold hook and
then arms the comma-separated `DS41_FASTPATH` list through `fastpath_hook.install()`.

## What is here

| path | what |
|---|---|
| `fastpath_*.py` | one component per kernel; each registers a patch with `fastpath_hook` |
| `libw4boundary.so` | W4 mHC boundaries, router, MegaMoE seam (`kernels/w4-boundary`) |
| `libw6mk.so` | W6 one-launch Markov loop (`kernels/w6-moe/markov`) |
| `libw6tp.so` | W6 exact top-p (`kernels/w6-moe/topp`) |
| `libw6combine.so` | W6 v11 lane-parallel combine (`kernels/w6-moe/combine`) |
| `libw8oproj.so` | W8 wo_a + MXFP8 quantize, split-K for sparse/drafter layers (`kernels/w8-persist`) |
| `lib_seam_quant_cs16.so` | the W4 seam variant `quant_producer` launches at T 6/8 (`kernels/w4-boundary/seam-quant`) |
| `native_scales.py`, `a12_memory_policy.py`, `producer_*.py` | helpers of `a12_retire`, `a12_memory_guard`, `quant_producer` |
| `w1attn/` | W1 decode attention runtime + `w1_attn_ext.so` |
| `w3preattn/` | W3 pre-attention runtime + `w3_ext.so` + `libw8topkg.so` |
| `peer3/` | peer tier v3: GB300/RTX kernels, prebuilt `.so`, sidecar, protocol tests |
| `megamoe_fork/` | the two modified DeepGEMM headers (v11 + shared-early pipeline) |
| `hook/` | upstream MegaMoE hook, `peer_tier2.py`, rowmaps, FlashInfer tune files, licences |

## Build

The prebuilt `.so` files are the production binaries for the release image `vllm/vllm-openai:v0.31.0`
(CUDA 13.0, sm_103a); they link only libtorch 2.13.0+cu130/libcudart 13 (no vLLM `_C`/FlashInfer), and all eleven load
in that image (no GPU; `ctypes`), so no rebuild is needed. To rebuild, run `../kernels/build_all.sh`; it writes
the `.so` files back here. Rebuilds are functionally equivalent but not byte-identical because nvcc embeds
source paths; see `../kernels/README.md` and `../kernels/BUILD-NOTES.md`.

## Credit

The fastpath components build on vLLM, DeepGEMM, FlashInfer, b12x (Local Inference Lab), and the
DGX Station GB300 research lane (`original-el8/dgx-station-gb300-research`) whose upstream hook and peer
tier this tree extends. The MegaMoE fork headers are DeepGEMM (MIT); the b12x package is Apache-2.0; the
upstream hook is Apache-2.0. See `THIRD_PARTY.md` and the licence files next to each third-party tree.
