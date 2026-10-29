#!/usr/bin/env bash
# libw8oproj.so (wo_a + MXFP8 quantize) and libw8topkg.so (indexer top-k + key table).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common.sh"
mkdir -p "$OUT/w3preattn"
need "$HERE/oproj/w8oproj_lib.cu" "$HERE/topkg/w8topk_global.cu" "$HERE/include" "$HERE/../w4-boundary/include"
run_docker -v "$HERE:/w" -v "$HERE/../w4-boundary/include:/w4/include:ro" -w /w --entrypoint bash "$IMAGE" -c \
  "set -e; \
   nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a --expt-relaxed-constexpr -shared -Xcompiler -fPIC \
     -DW8_SPIN_TIMEOUT_NS=10000000000ull -lineinfo -I include -I oproj -I /w4/include \
     oproj/w8oproj_lib.cu -o /w/libw8oproj.so; \
   nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a --expt-relaxed-constexpr -shared -Xcompiler -fPIC \
     -DW8TG_SORT=0 -I topkg topkg/w8topk_global.cu -o /w/libw8topkg.so"
[ "$DRY_RUN" = 1 ] && exit 0
cp "$HERE/libw8oproj.so" "$OUT/libw8oproj.so"
cp "$HERE/libw8topkg.so" "$OUT/w3preattn/libw8topkg.so"
rm -f "$HERE/libw8oproj.so" "$HERE/libw8topkg.so"
sha256sum "$OUT/libw8oproj.so" "$OUT/w3preattn/libw8topkg.so"
