#!/usr/bin/env bash
# Run a verify/ Python check on the host when Python 3 is present, otherwise inside the release image.
# This keeps the host requirement to Docker + the NVIDIA driver (no host Python).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
. "$ROOT/serve/compat.sh"
: "${DS41_HOME:=$HOME/.ds41}"
CONFIG=${DS41_CONFIG:-$DS41_HOME/config.env}
if [ -f "$CONFIG" ]; then . "$CONFIG"; fi
: "${DS41_IMAGE:=$DS41_SUPPORTED_IMAGE}"
: "${DS41_API_KEY_FILE:=$DS41_HOME/api-key}"
if command -v python3 >/dev/null 2>&1; then
  exec python3 "$@"
fi
[ -s "$DS41_API_KEY_FILE" ] || { echo "no API key at $DS41_API_KEY_FILE; run ds41 install" >&2; exit 2; }
# DS41_PY_MOUNT: a directory the script writes results into (set by `ds41 bench --full`), mounted read-write.
extra=()
if [ -n "${DS41_PY_MOUNT:-}" ]; then extra=(-v "$DS41_PY_MOUNT:$DS41_PY_MOUNT"); fi
exec docker run --rm --network host --user "$(id -u):$(id -g)" \
  -v "$HERE:$HERE:ro" -v "$DS41_API_KEY_FILE:$DS41_API_KEY_FILE:ro" "${extra[@]}" \
  --entrypoint python3 "$DS41_IMAGE" "$@"
