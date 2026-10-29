#!/usr/bin/env bash
# Rebuild every prebuilt kernel used by the default recipe.  Requires Docker + the pinned vLLM image.
# Writes the .so files into ../fastpath/ (or $OUT).  No GPU is needed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export FASTPATH=${FASTPATH:-$HERE/../fastpath}
export DRY_RUN=${DRY_RUN:-0}
for d in w4-boundary w6-moe w8-persist w1-attn w3-preattn peer3; do
  echo "=== $d"
  "$HERE/$d/build.sh"
done
if [ "$DRY_RUN" = 1 ]; then
  echo "=== dry run: every canonical source resolved"
  exit 0
fi
echo "=== done; sha256:"
find "${OUT:-$FASTPATH}" -maxdepth 3 -name '*.so' -print0 | sort -z | xargs -0 sha256sum
