# docker

The release does **not** build an image.  It runs the stock
`vllm/vllm-openai:v0.31.0` image and mounts the recipe from this checkout at launch, read-only:

- `overlay-1017/vllm/` -> the matching files under the image's `vllm` package;
- `overlay-58132/vllm/` -> the same (required: `serve/serve.sh` refuses to start without it);
- `../fastpath/sitecustomize.py` -> `/usr/lib/python3.12/sitecustomize.py`;
- `../fastpath/` -> `/fastpath`;
- `../fastpath/hook/` -> `/w` (rowmaps and FlashInfer tune files);
- `b12x/` -> `/opt/ds41/b12x` in the RTX sidecar.

`ds41 install` pulls the official image.  There is no `Dockerfile`, no `build.sh`, and no
`DS41_IMAGE_PULL` switch.

Licences: vLLM and the overlays Apache-2.0; DeepGEMM MIT; b12x Apache-2.0; the model MIT.  The base
image's PyTorch and Triton are BSD-3-Clause.  See `../fastpath/THIRD_PARTY.md` and `../NOTICE`.
