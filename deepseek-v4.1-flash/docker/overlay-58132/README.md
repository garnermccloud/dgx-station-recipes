# vllm#58132 overlay (decoder SWA bounded replay)

[vllm-project/vllm#58132](https://github.com/vllm-project/vllm/pull/58132) ("Decoder-side SWA bounded replay for
DeepSeek-V4.1") was open and maintainer-approved but unmerged when we ran it on 2026-09-28; it is still unmerged in
vLLM 0.31.0. It lets layers 21-39 of DeepSeek-V4.1-Flash, which hold only sliding-window KV and take long-range
context from layer 20's compressed KV, run on each request's last 128 tokens in eager prefill steps.

`vllm/` is the patched tree: the four vLLM 0.31.0 originals (`vllm/config/cache.py`,
`vllm/models/deepseek_v41/nvidia/{model,model_state,vl_model}.py`) plus the new
`vllm/models/deepseek_v41/decoder_replay_layers.py`, with `pr58132-runtime.diff` applied. The four base files are
byte-identical between the nightly and v0.31.0, so the v0.31.0 originals + the diff are byte-identical to the
production tree in `docker/overlay-58132/vllm`.

`pr58132-runtime.diff` is the PR's diff restricted to its five runtime files (tests and CI config dropped), as
fetched with `gh pr diff 58132` at head `394e46fd`. It is vLLM code, Apache-2.0 (see
[`../../../NOTICE`](../../../NOTICE)). It applies cleanly to the files in image
`vllm/vllm-openai:v0.31.0` (and to the nightly, whose four base files are identical).

The release's `serve.sh` auto-mounts `docker/overlay-58132/vllm/…` over the image's package
(`REPLAY_OVERLAY=0` boots the stock files). To rebuild the tree:

```bash
IMG=vllm/vllm-openai:v0.31.0
C=$(docker create "$IMG"); P=/usr/local/lib/python3.12/dist-packages
mkdir -p vllm/config vllm/models/deepseek_v41/nvidia
for f in vllm/config/cache.py vllm/models/deepseek_v41/nvidia/{model,model_state,vl_model}.py; do
  docker cp "$C:$P/$f" "$f"
done
docker rm "$C"
patch -p1 < ../pr58132-runtime.diff
```

The boot log confirms it with `Decoder SWA bounded replay: in eager prefill steps, layers 21-39 run on each
request's last 128 tokens only.` Results: [`../DETAILS.md`](../DETAILS.md#decoder-swa-bounded-replay-vllm58132).
