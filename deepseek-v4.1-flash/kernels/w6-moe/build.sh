#!/usr/bin/env bash
# libw6mk.so (Markov loop), libw6tp.so (top-p), libw6combine.so (v11 combine).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common.sh"
mkdir -p "$OUT"
need "$HERE/markov/w6mk.cu" "$HERE/topp/w6tp.cu" "$HERE/combine/w6combine.cu"
run_docker -v "$HERE:/w" -w /w --entrypoint bash "$IMAGE" -c \
  "set -e; \
   nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -Xcompiler -fPIC -shared \
     markov/w6mk.cu -o /w/libw6mk.so; \
   nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -Xcompiler -fPIC -shared \
     topp/w6tp.cu -o /w/libw6tp.so; \
   nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -Xcompiler -fPIC -shared \
     combine/w6combine.cu -o /w/libw6combine.so"
[ "$DRY_RUN" = 1 ] && exit 0
for f in libw6mk.so libw6tp.so libw6combine.so; do cp "$HERE/$f" "$OUT/$f"; rm -f "$HERE/$f"; done
sha256sum "$OUT"/libw6mk.so "$OUT"/libw6tp.so "$OUT"/libw6combine.so
