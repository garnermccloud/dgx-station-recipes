#!/usr/bin/env bash
# Place / verify the rowmap and FlashInfer tune file used by the default recipe.
# The repo carries them under fastpath/hook/; this script checks them and, with --install-to DIR, copies the
# hook tree to a runtime location.  The release mounts the repo hook at /w; no image build is involved.
#   setup_assets.sh [--check] [--install-to DIR]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/config/recipe.env"
HOOK=${DS41_HOOK_DIR:-$ROOT/fastpath/hook}
ROWMAP=$DS41_ROWMAP
FI=$DS41_FI_TUNE_FILE
log() { echo "[assets] $*"; }
die() { echo "[assets] ERROR: $*" >&2; exit 1; }
[ -f "$HOOK/$ROWMAP" ] || die "rowmap missing: $HOOK/$ROWMAP"
[ -f "$HOOK/fi-tune/$FI" ] || die "FlashInfer tune file missing: $HOOK/fi-tune/$FI"
log "rowmap $ROWMAP ($(stat -c '%s' "$HOOK/$ROWMAP") bytes)"
log "fi-tune $FI ($(stat -c '%s' "$HOOK/fi-tune/$FI") bytes)"
if [ "${1:-}" = --install-to ] || [ "${2:-}" = --install-to ]; then
  DEST=${3:-${2:-}}
  [ -n "$DEST" ] || die "--install-to needs a directory"
  mkdir -p "$DEST"
  cp -a "$HOOK/." "$DEST/"
  log "copied hook assets to $DEST"
fi
