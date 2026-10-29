#!/usr/bin/env bash
# One-time setup for a source checkout: model + rowmap/fi-tune + Engram checks.
# The `ds41 install` CLI calls the same steps with preflight and progress; this script is the low-level
# engine and is safe to re-run.
#   setup.sh [--no-model] [--check]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NO_MODEL=0; CHECK=0
for a in "$@"; do case "$a" in --no-model) NO_MODEL=1;; --check) CHECK=1;; *) echo "bad arg $a" >&2; exit 2;; esac; done
if [ "$CHECK" = 1 ]; then
  rc=0
  "$HERE/setup_model.sh" --check || rc=1
  "$HERE/setup_assets.sh" --check || rc=1
  "$HERE/setup_engram_store.sh" || rc=1
  exit "$rc"
fi
[ "$NO_MODEL" = 1 ] || "$HERE/setup_model.sh"
"$HERE/setup_assets.sh"
"$HERE/setup_engram_store.sh"
echo "[setup] done. Next: ds41 up"
