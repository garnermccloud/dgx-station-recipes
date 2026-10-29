# kernels

Sources and reproducible builds for the prebuilt kernels in `../fastpath/`.  Python components and the MegaMoE fork
headers live only in `../fastpath/`; this directory keeps the C/CUDA build sources and scripts.
The binaries are built for
the release image (`vllm/vllm-openai:v0.31.0`, CUDA 13.0, `sm_103a`; peer3 also `sm_120a`); the binaries were built against the nightly with the same torch 2.13.0+cu130/CUDA 13.0 and load unchanged in v0.31.0.
No GPU is needed to compile.

## Build everything

```bash
./build_all.sh                 # writes ../fastpath/*.so
IMAGE=... ./build_all.sh       # a different pinned image
DRY_RUN=1 ./build_all.sh       # resolve every canonical source, no docker/nvcc
```

Each subdirectory also builds on its own.  The build runs inside the pinned image, so the host only needs
Docker; `nvcc` and the CUDA headers come from the image.

| kernel | source | output |
|---|---|---|
| W4 boundary | `w4-boundary/src/csrc/*.cu` | `fastpath/libw4boundary.so` |
| W4 seam, MXFP8 producer (CS16) | `w4-boundary/seam-quant/seam_quant_cs16.cu` | `fastpath/lib_seam_quant_cs16.so` |
| W6 Markov | `w6-moe/markov/w6mk.cu` | `fastpath/libw6mk.so` |
| W6 top-p | `w6-moe/topp/w6tp.cu` | `fastpath/libw6tp.so` |
| W6 combine (v11) | `w6-moe/combine/w6combine.cu` | `fastpath/libw6combine.so` |
| W8 wo_a | `w8-persist/oproj/w8oproj_lib.cu` | `fastpath/libw8oproj.so` |
| W8 top-k/key-table | `w8-persist/topkg/w8topk_global.cu` | `fastpath/w3preattn/libw8topkg.so` |
| W1 attention | `w1-attn/csrc/w1_attn.cu` | `fastpath/w1attn/w1_attn_ext.so` |
| W3 pre-attention | `w3-preattn/csrc/w3_ext.cu` | `fastpath/w3preattn/w3_ext.so` |
| peer3 | `../fastpath/peer3/csrc/*.cu` | `fastpath/peer3/prebuilt/peer3_*.so` |

The MegaMoE fork headers (`../fastpath/megamoe_fork/`) are DeepGEMM source, not a compiled kernel; they
are assembled at runtime by `fastpath_megamoe.py`.

## Credit

The kernels are original work for this recipe, built on DeepGEMM (MIT, DeepSeek), FlashInfer
(Apache-2.0), vLLM (Apache-2.0), and b12x (Apache-2.0, Local Inference Lab).  The peer protocol and the
upstream hook come from the DGX Station GB300 research lane (`original-el8/dgx-station-gb300-research`,
Apache-2.0).  See `../fastpath/THIRD_PARTY.md` and the licence files next to the sources.
