# DGX Station recipes

Serving recipes for an NVIDIA **DGX Station GB300**, each one a tested, one-command way to run a model on it and
get a private, OpenAI-compatible endpoint. Every recipe carries its own CLI, docs, measured results and licence
notices, so you can copy a recipe's directory on its own.

| recipe | model | extra hardware | single stream | many streams | context |
|---|---|---|---|---|---|
| [`deepseek-v4.1-flash/`](deepseek-v4.1-flash/) | DeepSeek-V4.1-Flash-1017 | one RTX PRO 6000 (96 GB) | ~730 tok/s on code, ~500 on reasoning | 1,373 tok/s at 8 streams, 2,524 at 32 | 512K per request, 11.4M-token KV cache |

Numbers are medians from each recipe's own benchmark on our reference box; the recipe's `README.md` and
`RESULTS.md` say exactly how they were measured, and its bench command reproduces them on yours.

## Quick start

```bash
git clone https://github.com/garnermccloud/dgx-station-recipes.git
cd dgx-station-recipes/deepseek-v4.1-flash
./ds41 install
./ds41 up
```

Then read that recipe's `README.md`.

## Licence

MIT (see `LICENSE`), copyright 2026 Garner McCloud. Each recipe's `NOTICE` lists the third-party code and data it
includes and their licences.
