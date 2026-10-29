#!/usr/bin/env bash
# One-command scrub for the release tree.
#
#   tools/scrub/scrub.sh
#   tools/scrub/scrub.sh --external --report SCRUB-REPORT.md
#
# First runs pyflakes 3.2.0 inside the supported vLLM image over every tracked
# Python file except the two vendored upstream paths. Then runs the secret and
# antipattern scanner. Any lint finding fails the scrub. It never prints a
# secret value; reports are redacted. Exit 0 clean, 1 findings, 2 usage error.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRUB_ROOT:-}"
if [ -z "$ROOT" ]; then
  ROOT="$(cd "$HERE/../.." && pwd)"   # the recipe directory, also inside a multi-recipe repository
fi

# The scanner's own self-test and help do not need Docker or the image.
for arg in "$@"; do
  case "$arg" in
    --self-test|--help|-h)
      exec python3 "$HERE/scrub.py" --root "$ROOT" "$@"
      ;;
  esac
done

IMAGE="${DS41_IMAGE:-vllm/vllm-openai:v0.31.0}"
FILES="$(git -C "$ROOT" ls-files '*.py' \
  | grep -v -e '^docker/b12x/' \
  | grep -v -e '^docker/overlay-1017/vllm/tokenizers/deepseek_v41_encoding.py$')"
if [ -z "$FILES" ]; then
  echo "scrub: lint: no tracked Python files under $ROOT" >&2
  exit 2
fi

LINT_OUT="$(mktemp)"
trap 'rm -f "$LINT_OUT"' EXIT

echo "scrub: lint: pyflakes 3.2.0 in $IMAGE" >&2
if ! docker run --rm \
    -v "$ROOT:/src:ro" -w /src \
    --entrypoint sh "$IMAGE" \
    -c 'pip install -q pyflakes==3.2.0 && pyflakes "$@"' \
    sh $FILES >"$LINT_OUT" 2>&1; then
  echo "scrub: lint: FAIL" >&2
  cat "$LINT_OUT" >&2
  exit 1
fi
echo "scrub: lint: PASS" >&2

exec python3 "$HERE/scrub.py" --root "$ROOT" "$@"
