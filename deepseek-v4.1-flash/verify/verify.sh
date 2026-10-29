#!/usr/bin/env bash
# One-screen PASS/FAIL verification against the published numbers.
#   verify.sh [--quick|--full] [--skip-gsm8k] [--skip-decode] [--skip-coding]
# Env: DS41_CONFIG, DS41_PORT, DS41_API_KEY_FILE, DS41_SERVED, DS41_MIN_PCT (default 80),
#      DS41_MIN_ACC (default 95), DS41_BASE_URL.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
: "${DS41_HOME:=$HOME/.ds41}"
CONFIG=${DS41_CONFIG:-$DS41_HOME/config.env}
if [ -f "$CONFIG" ]; then . "$CONFIG"; fi
: "${DS41_PORT:=30003}"
: "${DS41_API_KEY_FILE:=$DS41_HOME/api-key}"
: "${DS41_SERVED:=deepseek-v4.1-flash}"
: "${DS41_BASE_URL:=http://127.0.0.1:$DS41_PORT}"
: "${DS41_MIN_PCT:=80}"
: "${DS41_MIN_ACC:=95}"
WINDOW=60; WARMUP=20; RUNS=1; SKIP_GSM=0; SKIP_DECODE=0; SKIP_CODING=0
for a in "$@"; do case "$a" in
  --quick) WINDOW=30; WARMUP=10;;
  --full) WINDOW=60; WARMUP=20; RUNS=3;;
  --skip-gsm8k) SKIP_GSM=1;;
  --skip-decode) SKIP_DECODE=1;;
  --skip-coding) SKIP_CODING=1;;
  *) echo "bad arg $a" >&2; exit 2;;
esac; done
[ -s "$DS41_API_KEY_FILE" ] || { echo "verify: no API key at $DS41_API_KEY_FILE (run ds41 install)" >&2; exit 2; }
KEY="$DS41_API_KEY_FILE"
common=(--base-url "$DS41_BASE_URL" --key-file "$KEY" --model "$DS41_SERVED")
pass=0; fail=0
run() { local name=$1; shift; if "$@"; then pass=$((pass+1)); else fail=$((fail+1)); fi; }
echo "== ds41 verify: $DS41_BASE_URL ($DS41_SERVED)"
run "preflight image" "$HERE/preflight.sh" image
run "preflight weights" "$HERE/preflight.sh" weights
run "preflight GPUs" "$HERE/preflight.sh" gpus
run "preflight boot" "$HERE/preflight.sh" boot
if [ "$fail" != 0 ]; then
  echo "== $pass PASS, $fail FAIL (preflight failed; fix the FAIL lines above)"
  exit 1
fi
curl -fsS -m 10 -H "Authorization: Bearer $(cat "$KEY")" "$DS41_BASE_URL/v1/models" | grep -q "$DS41_SERVED" \
  || { echo "server FAIL: $DS41_BASE_URL/v1 is not serving $DS41_SERVED; run: ds41 status"; exit 1; }
run "tool call" "$HERE/py.sh" "$HERE/tool.py" "${common[@]}"
run "needles 200K" "$HERE/py.sh" "$HERE/needle.py" "${common[@]}" --tokens 200000
[ "$SKIP_GSM" = 1 ] || run "GSM8K-200" "$HERE/py.sh" "$HERE/gsm8k.py" "${common[@]}" --min-acc "$DS41_MIN_ACC"
[ "$SKIP_DECODE" = 1 ] || run "decode C1/C4/C8/C16" "$HERE/py.sh" "$HERE/decode.py" "${common[@]}" \
  --conc 1,4,8,16 --window "$WINDOW" --warmup "$WARMUP" --min-pct "$DS41_MIN_PCT"
[ "$SKIP_CODING" = 1 ] || run "coding prompt" "$HERE/py.sh" "$HERE/coding.py" "${common[@]}" --runs "$RUNS" --min-pct "$DS41_MIN_PCT"
echo "== $pass PASS, $fail FAIL"
[ "$fail" = 0 ]
