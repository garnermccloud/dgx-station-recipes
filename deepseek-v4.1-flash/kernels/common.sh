# Shared build settings.  Source from a per-kernel build.sh; do not execute directly.
KROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FASTPATH=${FASTPATH:-$KROOT/fastpath}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.31.0}
OUT=${OUT:-$FASTPATH}
DRY_RUN=${DRY_RUN:-0}
DOCKER_RUN=(docker run --rm --network none --user "$(id -u):$(id -g)")

need() {
  local p
  for p in "$@"; do
    [ -e "$p" ] || { echo "missing build source: $p" >&2; exit 1; }
  done
}

run_docker() {
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY docker'; printf ' %q' "$@"; echo
    return 0
  fi
  "${DOCKER_RUN[@]}" "$@"
}
