#!/usr/bin/env bash
# Check the Engram side of the checkpoint and the host memory it needs.
# The two Engram tables (layers 1 and 14, ~189 GiB) are read from the checkpoint and pinned in host RAM at
# boot (--engram-config '{"cpu_offload": true}').  They are not copied to a second store; keep the
# checkpoint on fast NVMe and give the host enough RAM.  This script verifies both.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/serve/compat.sh"
MODEL=${DS41_MODEL_DIR:-$HOME/.ds41/models/DeepSeek-V4.1-Flash-1017}
CACHE=${DS41_CACHE_DIR:-$HOME/.ds41/state}
IMAGE=${DS41_IMAGE:-$DS41_SUPPORTED_IMAGE}
log() { echo "[engram] $*"; }
die() { echo "[engram] ERROR: $*" >&2; exit 1; }
[ -f "$MODEL/model.safetensors.index.json" ] || die "model index missing at $MODEL; run setup_model.sh"
if command -v python3 >/dev/null 2>&1; then
  python3 - "$MODEL" <<'PY'
import json, sys
model = sys.argv[1]
idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
names = [n for n in idx if ".engram.embed.weight" in n]
if len(names) < 2:
    raise SystemExit(f"expected 2 Engram tables, found {len(names)}: {names}")
print("Engram tables:")
for n in sorted(names):
    print(f"  {n} -> {idx[n]}")
PY
else
  command -v docker >/dev/null 2>&1 || die "no host python3 and docker is not installed"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || die "no host python3 and image $IMAGE is missing; run: ds41 install"
  docker run --rm -i --network none -v "$MODEL:/model:ro" --entrypoint python3 "$IMAGE" - /model <<'PY'
import json, sys
model = sys.argv[1]
idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
names = [n for n in idx if ".engram.embed.weight" in n]
if len(names) < 2:
    raise SystemExit(f"expected 2 Engram tables, found {len(names)}: {names}")
print("Engram tables:")
for n in sorted(names):
    print(f"  {n} -> {idx[n]}")
PY
fi
mkdir -p "$CACHE/engram"
mem=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)
shm=$(df -P /dev/shm | awk 'NR==2 {print int($4/1024/1024)}')
shm_total=$(df -P /dev/shm | awk 'NR==2 {print int($2/1024/1024)}')
log "host RAM: ${mem} GiB (480+ GiB recommended; the Station reference is ~496 GiB)"
log "/dev/shm: ${shm} GiB free of ${shm_total} GiB (peer tier; the Engram tables are pinned in host RAM, not /dev/shm)"
[ "$mem" -ge 480 ] || log "WARNING: less than 480 GiB host RAM; the reference Station has ~496 GiB and boot may fail while pinning the Engram tables"
[ "$shm" -ge 16 ] || log "WARNING: /dev/shm has less than 16 GiB free; the peer tier may fail"
log "Engram checkpoint is on disk; no second copy is made. Cache marker: $CACHE/engram"
date -Is > "$CACHE/engram/.checked"
