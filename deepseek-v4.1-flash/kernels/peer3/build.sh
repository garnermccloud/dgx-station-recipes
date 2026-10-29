#!/usr/bin/env bash
# peer3_gb300_sm103.so and peer3_rtx_sm120.so: the peer tier v3 device halves.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common.sh"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
need "$FASTPATH/peer3/csrc" "$FASTPATH/peer3/py"
cp -a "$FASTPATH/peer3/csrc" "$FASTPATH/peer3/py" "$TMP/"
mkdir -p "$TMP/build" "$OUT/peer3/prebuilt"
for arch in 103 120; do
  side=gb300; [ "$arch" = 120 ] && side=rtx
  archlist=10.3; [ "$arch" = 120 ] && archlist=12.0a
  run_docker -v "$TMP:/w" -w /w -e "PEER3_ARCH=$arch" -e PEER3_BUILD_DIR=/w/build \
    -e "TORCH_CUDA_ARCH_LIST=$archlist" --entrypoint python3 "$IMAGE" \
    -c "import sys; sys.path.insert(0,'/w/py'); import peer3_ext; print(peer3_ext.load('$side').__file__)"
  if [ "$DRY_RUN" = 1 ]; then continue; fi
  cp "$TMP/build/peer3_${side}_sm${arch}/peer3_${side}_sm${arch}.so" "$OUT/peer3/prebuilt/peer3_${side}_sm${arch}.so"
done
[ "$DRY_RUN" = 1 ] && exit 0
sha256sum "$OUT"/peer3/prebuilt/*.so
