# serve

The engine behind the `ds41` CLI.  It starts the RTX sidecar, then the GB300 vLLM server, and verifies
readiness.  The user-facing commands are `ds41 up`, `ds41 down`, `ds41 status`, `ds41 logs`,
`ds41 verify`; `serve.sh` is the same lifecycle without the friendly wrapper.

```bash
./serve.sh start      # RTX sidecar (required) + vLLM
./serve.sh status     # health, endpoint, sample curl, GPU memory, uptime
./serve.sh doctor     # preflight in plain English
./serve.sh logs       # follow the vLLM log
./serve.sh logs sidecar
./serve.sh stop       # stop ds41 + sidecar; return state ownership (never touches anything else)
./serve.sh down       # stop + remove both containers; return state ownership
```

## Configuration

`~/.ds41/config.env` (optional).  `../config/ds41.env.example` documents the operator knobs; the fixed
recipe constants (fastpath component list, MegaMoE rowmap, pinned FlashInfer tune file) live in
`../config/recipe.env` and are not user knobs.  Environment variables win over the file.  The only required
values after `ds41 install` are the model path and the API key file, and the installer writes them.

- **GPUs** are auto-detected by name: the first `NVIDIA GB300` and the first `RTX PRO 6000`.  Set
  `DS41_GPU_GB300` / `DS41_GPU_RTX` to a UUID to pin one.  The RTX A2000 (if present) is ignored.
- **Utilisation** defaults to `DS41_UTIL_DEFAULT` in `../config/recipe.env` (a GB300 that runs only ds41).  Lower
  `DS41_UTIL` if another process shares the GB300, and set `DS41_GPU_RESERVE_MIB` to what that process may still
  allocate after ds41 is up: the startup memory guard keeps that much free after capture.
- **API key**: generated once at `~/.ds41/api-key` (mode 600) if missing; the CLI prints it once.
  The server receives it through a private `--env-file`, never on the command line.
- **Both GPUs are required.**  The GB300 loads 198.6 GiB and has 252 GiB; the ~87 GiB of cold experts live on
  the RTX PRO 6000.  If it is missing, `serve.sh` refuses to start: `found only the GB300 / Fix: install the
  RTX PRO 6000 (96 GB); the model does not fit in the GB300 alone.`
- **Port**: `30003` by default; OpenAI-compatible at `/v1`.

## What it runs

- RTX sidecar: `peer_server3.py` on the RTX PRO 6000, b12x cold experts in VRAM, peer tier v3/v2 over
  `/dev/shm`; `--ipc host --network none`, `PEER3_PREFETCH_K=4`, `PEER3_WAIT_NS=30s`.
- Both containers execute `/usr/bin/numactl` from the image.  The compatibility/preflight check requires it;
  there is no no-numactl fallback, and the host binary is never mounted.
- GB300: the official `vllm/vllm-openai:v0.31.0` image with the 1017 frontend and the 58132 decoder overlay
  mounted read-only from this checkout, `deep_gemm_mega_moe`,
  `nvfp4_ds_mla` KV, 524,288 context, DSpark dynamic K (5/7), fastload, coldskip, peer3, MegaMoE (shared-early fork), quant_producer, a12_retire
  with its memory guard, and the
  production fastpath list.
- No co-tenant, no site-specific container names beyond `ds41` and `ds41-sidecar`.

The image is `vllm/vllm-openai:v0.31.0` by default; `ds41 install` pulls it.  There is no image build.
At launch `serve.sh` mounts, read-only:

- `docker/overlay-1017/vllm/` and `docker/overlay-58132/vllm/` onto the image's `vllm` package;
- `fastpath/sitecustomize.py` to `/usr/lib/python3.12/sitecustomize.py`;
- `fastpath/` to `/fastpath`, `fastpath/hook/` to `/w` (rowmaps + FlashInfer tune files);
- `docker/b12x/` into the RTX sidecar.

`serve.sh` refuses to start if the image tag or the image's vendored DeepGEMM base hashes do not match the one
compatibility assertion, the official image lacks the v0.31.0 overlay targets or numactl, either overlay/fastpath/
hook/b12x tree is missing or empty, the model is incomplete, the RTX PRO 6000 is missing, or the API key is missing.
`ds41 verify` additionally requires one `DS41_FASTPATH ACTIVE <component>` line per component, the
`W6_MEGAMOE ACTIVE:` line, the peer3 finish-fold line, and a complete FlashInfer tune-file load with no
`No tuned config covers` fallback.
