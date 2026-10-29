#!/usr/bin/env bash
# Download the official DeepSeek-V4.1-Flash-1017 checkpoint (MIT) with resume + progress.
#   setup_model.sh [--check]
# Env: DS41_MODEL_DIR (default ~/.ds41/models/DeepSeek-V4.1-Flash-1017), HF_TOKEN (optional), DS41_MODEL_REV.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/serve/compat.sh"
MODEL=${DS41_MODEL_DIR:-$HOME/.ds41/models/DeepSeek-V4.1-Flash-1017}
REV=${DS41_MODEL_REV:-1bae0e1088993b9b82de08f59de5be5a34e42170}
REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
IMAGE=${DS41_IMAGE:-$DS41_SUPPORTED_IMAGE}
log() { echo "[model] $*"; }
die() { echo "[model] ERROR: $*" >&2; exit 1; }
check() {
  [ -f "$MODEL/config.json" ] || return 1
  [ -f "$MODEL/model.safetensors.index.json" ] || return 1
  [ "$(find "$MODEL" -maxdepth 1 -name 'model-*.safetensors' | wc -l)" -ge 48 ] || return 1
}
if [ "${1:-}" = --check ]; then
  check && { log "model present at $MODEL"; exit 0; } || { log "model incomplete at $MODEL"; exit 1; }
fi
if check; then log "model already present at $MODEL"; exit 0; fi
mkdir -p "$MODEL"
command -v df >/dev/null && log "free space: $(df -P "$MODEL" | awk 'NR==2 {print int($4/1024/1024)}') GiB (need ~500 GiB)"
if command -v hf >/dev/null 2>&1; then
  log "downloading $REPO @ ${REV:0:8} with hf (resumes automatically)"
  hf download "$REPO" --revision "$REV" --local-dir "$MODEL"
elif command -v huggingface-cli >/dev/null 2>&1; then
  log "downloading $REPO @ ${REV:0:8} with huggingface-cli --resume-download"
  huggingface-cli download "$REPO" --revision "$REV" --local-dir "$MODEL" --resume-download
elif docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "downloading $REPO @ ${REV:0:8} inside $IMAGE (no host Python needed)"
  docker run --rm --network host --user "$(id -u):$(id -g)" \
    -v "$MODEL:/model" \
    -e "REPO=$REPO" -e "REV=$REV" -e "HF_TOKEN=${HF_TOKEN:-}" -e HF_HOME=/tmp/hf \
    --entrypoint python3 "$IMAGE" -c \
    'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["REPO"], revision=os.environ["REV"], local_dir="/model", max_workers=8)'
else
  die "no Hugging Face CLI and no image $IMAGE; run: ds41 install (pulls the official image) or install huggingface_hub[cli]"
fi
check || die "download finished but the checkpoint is incomplete; re-run this script to resume"
[ -f "$MODEL/LICENSE" ] || log "WARNING: LICENSE missing; the official checkpoint is MIT"
log "model ready at $MODEL"
du -sh "$MODEL"
