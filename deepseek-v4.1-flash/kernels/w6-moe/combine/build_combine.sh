#!/usr/bin/env bash
# Build libw6combine.so inside the production image (run.sh none "bash v11-wip/build_combine.sh").
set -euo pipefail
cd "$(dirname "$0")"
nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -cudart shared \
  -gencode arch=compute_100a,code=sm_100a -gencode arch=compute_103a,code=sm_103a \
  w6combine.cu -o libw6combine.so
cuobjdump --list-elf libw6combine.so
