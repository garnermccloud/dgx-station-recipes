# Third-party notices

This tree contains or links against:

- **vLLM** — Apache-2.0. The image, the `sitecustomize` mechanism, and the modules the components patch.
  The overlay files under `../docker/overlay-1017` and `../docker/overlay-58132` carry their own SPDX
  headers and are vLLM code (Apache-2.0).
- **DeepGEMM** — MIT, Copyright (c) 2025 DeepSeek. `megamoe_fork/` is a fork of two DeepGEMM headers;
  `LICENSE.DeepGEMM` is included.
- **FlashInfer** — Apache-2.0. Used by vLLM and by several components (GEMM, top-k, attention helpers).
- **PyTorch** — BSD-3-Clause, Meta. The tensor/runtime layer under every component and the sidecar.
- **Triton** — BSD-3-Clause, OpenAI. The sidecar's dequant/route kernels and several fastpath helpers.
- **b12x** — Apache-2.0, Local Inference Lab. The RTX sidecar imports `b12x.moe.fused_moe` and
  `b12x.preparation`; `../docker/b12x/LICENSE` is included.
- **DGX Station GB300 research lane** — Apache-2.0 (`original-el8/dgx-station-gb300-research`). The
  upstream MegaMoE hook, peer tier v2, rowmap tools, and the peer protocol originate there; `hook/LICENSE`
  and `hook/SOURCE` are included.
- **RonanLabs GSM8K harness** — Apache-2.0, Copyright 2026 RonanLabs. The first 200 GSM8K rows and the
  scoring script in `../verify/data` come from that harness; `../verify/data/LICENSE.ronanlabs` is included.
  The underlying GSM8K questions are from OpenAI's grade-school-math (MIT).

The prebuilt `.so` files are built from the sources under `../kernels/`; the DeepGEMM fork headers are
shipped as source because DeepGEMM compiles them at runtime.
