#!/usr/bin/env bash
# w1_attn_ext.so: W1 decode attention.  JIT-built with the image's torch + nvcc 13.4, sm_103a.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common.sh"
mkdir -p "$OUT/w1attn"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
need "$FASTPATH/w1attn" "$HERE/csrc" "$HERE/include"
cp -a "$FASTPATH/w1attn" "$HERE/csrc" "$HERE/include" "$TMP/"
mkdir -p "$TMP/build"
run_docker -v "$TMP:/w" -w /w -e TORCH_CUDA_ARCH_LIST=10.3a \
  --entrypoint python3 "$IMAGE" -c "import sys; sys.path.insert(0,'/w'); import w1attn.ops as o; print(o.build().__file__)"
[ "$DRY_RUN" = 1 ] && exit 0
cp "$TMP/build/w1_attn/w1_attn_ext.so" "$OUT/w1attn/w1_attn_ext.so"
sha256sum "$OUT/w1attn/w1_attn_ext.so"
