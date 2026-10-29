# scripts

One-time setup helpers.  The `ds41 install` CLI owns the friendly preflight/progress flow; these scripts
are the idempotent steps it calls (and can be run directly from a checkout).  b12x is vendored under
`docker/b12x/`; there is no clone step.  The FlashInfer tune file is shipped and fixed by
`config/recipe.env`; there is no tuning pipeline in the release.

| script | what |
|---|---|
| `setup_model.sh` | download `deepseek-ai/DeepSeek-V4.1-Flash-1017` at the pinned revision with resume; uses host `hf` if present, otherwise the release image (no host Python); verifies 48 shards |
| `setup_assets.sh` | verify/place the rowmap and FlashInfer tune file |
| `setup_engram_store.sh` | verify the two Engram tables and the host RAM they are pinned into (not `/dev/shm`) |
| `setup.sh` | model + assets + Engram checks, re-runnable |

The official checkpoint is `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, MIT.  No weights are re-uploaded by this repo.

Engram: the tables (~189 GiB) are read from the checkpoint and pinned in host RAM at boot; 480+ GiB host RAM
is recommended (the Station reference is ~496 GiB).  There is no separate SSD copy to make.  The peer tier
uses `/dev/shm`: on the reference Station it is 248 GiB total and ~120 GiB free while production runs; 16+ GiB
free is recommended.  `/dev/shm` is not where the Engram tables live.
