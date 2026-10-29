#!/usr/bin/env bash
# libw4boundary.so: W4 mHC boundaries, router, MegaMoE seam.  Plain C ABI, no torch.
# lib_seam_quant_cs16.so: the seam variant quant_producer launches at T 6/8 (cluster 16, gsize 1); it also writes
# the next attention input as MXFP8.  seam-quant/include/w4/seam.cuh replaces include/w4/seam.cuh for this build only
# (its 4-lane reductions use the active mask: inactive lanes skip the last chunk).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/../common.sh"
mkdir -p "$OUT"
need "$HERE/include" "$HERE/src/csrc/boundary.cu" "$HERE/src/csrc/mix.cu" "$HERE/src/csrc/router.cu" \
  "$HERE/src/csrc/router_mega.cu" "$HERE/src/csrc/mega.cu" "$HERE/src/csrc/draft_router.cu" "$HERE/src/csrc/head.cu" \
  "$HERE/seam-quant/seam_quant_cs16.cu" "$HERE/seam-quant/include/w4/seam.cuh"
run_docker -v "$HERE:/w" -w /w --entrypoint bash "$IMAGE" -c \
  "set -e; \
   nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -shared -Xcompiler -fPIC -lineinfo -I include \
   src/csrc/boundary.cu src/csrc/mix.cu src/csrc/router.cu src/csrc/router_mega.cu src/csrc/mega.cu \
   src/csrc/draft_router.cu src/csrc/head.cu -o /w/libw4boundary.so; \
   nvcc -O3 -std=c++17 -gencode=arch=compute_103a,code=sm_103a -shared -Xcompiler -fPIC -lineinfo -Xptxas=-v \
   -I seam-quant/include -I include -I include/w4 seam-quant/seam_quant_cs16.cu -o /w/lib_seam_quant_cs16.so"
[ "$DRY_RUN" = 1 ] && exit 0
cp "$HERE/libw4boundary.so" "$OUT/libw4boundary.so"
cp "$HERE/lib_seam_quant_cs16.so" "$OUT/lib_seam_quant_cs16.so"
rm -f "$HERE/libw4boundary.so" "$HERE/lib_seam_quant_cs16.so"
sha256sum "$OUT/libw4boundary.so" "$OUT/lib_seam_quant_cs16.so"
