#!/usr/bin/env bash
# Preflight checks shared by `ds41 verify` and `ds41 bench`: compatibility, image, weights, GPUs,
# and the boot-log active-component gate.
#   preflight.sh image|weights|gpus|boot
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
. "$ROOT/serve/compat.sh"
: "${DS41_HOME:=$HOME/.ds41}"
CONFIG=${DS41_CONFIG:-$DS41_HOME/config.env}
if [ -f "$CONFIG" ]; then . "$CONFIG"; fi
. "$ROOT/config/recipe.env"
: "${DS41_IMAGE:=$DS41_SUPPORTED_IMAGE}"
: "${DS41_MODEL_DIR:=$DS41_HOME/models/DeepSeek-V4.1-Flash-1017}"

fail() { echo "preflight ${1} FAIL: ${2}"; exit 1; }

case "${1:-}" in
  image)
    command -v docker >/dev/null 2>&1 || fail image "Docker is not installed"
    docker image inspect "$DS41_IMAGE" >/dev/null 2>&1 || fail image "image $DS41_IMAGE is missing; run: ds41 install"
    ds41_assert_compat || fail image "image $DS41_IMAGE is not the v0.31.0 base (see above)"
    ds41_image_layout_ok || fail image "image $DS41_IMAGE is not the v0.31.0 layout (overlay targets missing)"
    echo "preflight image PASS ($DS41_IMAGE; DeepGEMM base hashes; overlay targets)"
    ;;
  weights)
    [ -d "$DS41_MODEL_DIR" ] || fail weights "model directory missing at $DS41_MODEL_DIR; run: ds41 install"
    [ -f "$DS41_MODEL_DIR/config.json" ] || fail weights "config.json missing at $DS41_MODEL_DIR; run: ds41 install"
    [ -f "$DS41_MODEL_DIR/model.safetensors.index.json" ] || fail weights "index missing at $DS41_MODEL_DIR; run: ds41 install"
    n=$(find "$DS41_MODEL_DIR" -maxdepth 1 -name 'model-*.safetensors' | wc -l)
    [ "$n" -ge 48 ] || fail weights "found $n/48 weight shards at $DS41_MODEL_DIR; re-run: ds41 install"
    echo "preflight weights PASS ($n shards at $DS41_MODEL_DIR)"
    ;;
  gpus)
    command -v nvidia-smi >/dev/null 2>&1 || fail gpus "nvidia-smi is not installed"
    names=$(nvidia-smi --query-gpu=name --format=csv,noheader) || fail gpus "nvidia-smi query failed"
    echo "$names" | grep -qi GB300 || fail gpus "no NVIDIA GB300 visible; this recipe needs one"
    echo "$names" | grep -qi 'RTX PRO 6000' || fail gpus "found only the GB300 / Fix: install the RTX PRO 6000 (96 GB); the model does not fit in the GB300 alone."
    echo "preflight GPUs PASS (GB300 + RTX PRO 6000)"
    ;;
  boot)
    command -v docker >/dev/null 2>&1 || fail boot "Docker is not installed"
    logf=$(mktemp)
    trap 'rm -f "$logf"' EXIT
    docker logs ds41 >"$logf" 2>&1 || fail boot "no boot log from container ds41; run: ds41 status"
    count() { awk -v p="$1" 'index($0, p) { c++ } END { print c + 0 }' "$logf"; }
    missing=""
    for c in ${DS41_FASTPATH_COMPONENTS//,/ }; do
      # components load in both the API-server and engine processes, so each logs ACTIVE once per process
      [ "$(grep -cxF "DS41_FASTPATH ACTIVE $c" "$logf")" -ge 1 ] || missing="$missing $c"
    done
    [ -z "$missing" ] || fail boot "components that never logged ACTIVE:$missing"
    [ "$(count 'W6_MEGAMOE: JIT include root')" -ge 1 ] || fail boot "W6 MegaMoE include root was not activated"
    [ "$(count 'DS41_FASTPATH peer3: finish folded into MegaMoE')" -ge 1 ] || fail boot "peer3 finish fold did not engage"
    [ "$(grep -cE 'Loaded [0-9]+ configs' "$logf")" -ge 1 ] || fail boot "FlashInfer tune pin was not loaded"
    [ "$(grep -F 'A12_INSTALL_MEMORY' "$logf" | grep -cF '"status": "PASS"')" -ge 1 ] \
      || fail boot "the a12 memory guard did not pass after capture"
    for bad in 'No tuned config covers' 'will be REJECTED' 'Ignoring cached configs' 'W6_MEGAMOE stock' 'FATAL' \
               'falls back to the stock path' 'COMBINE off (' 'quant_producer: INVALID' '"status": "REFUSED"' \
               ': stock DSpark'; do
      [ "$(count "$bad")" -eq 0 ] || fail boot "boot log contains '$bad': $(grep -m1 -F "$bad" "$logf" | cut -c1-300)"
    done
    echo "preflight boot PASS (all components ACTIVE; W6 MegaMoE; peer3 finish fold; FlashInfer configs loaded; memory guard PASS; no fallback lines)"
    ;;
  *) echo "usage: preflight.sh image|weights|gpus|boot" >&2; exit 2;;
esac
