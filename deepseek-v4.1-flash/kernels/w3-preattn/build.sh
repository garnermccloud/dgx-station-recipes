#!/usr/bin/env bash
# w3_ext.so: W3 fused pre-attention chain.  JIT-built with the image's torch + nvcc 13, sm_103a.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common.sh"
mkdir -p "$OUT/w3preattn"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
need "$HERE/src" "$HERE/csrc" "$HERE/include"
cp -a "$HERE/src" "$HERE/csrc" "$HERE/include" "$TMP/"
mkdir -p "$TMP/build"
run_docker -v "$TMP:/w" -w /w -e TORCH_CUDA_ARCH_LIST=10.3a --entrypoint python3 "$IMAGE" \
  -c "import sys; sys.path.insert(0,'/w/src'); from w3.fused import ext; print(ext().__file__)"
[ "$DRY_RUN" = 1 ] && exit 0
cp "$TMP/build/w3_ext/w3_ext.so" "$OUT/w3preattn/w3_ext.so"
sha256sum "$OUT/w3preattn/w3_ext.so"
