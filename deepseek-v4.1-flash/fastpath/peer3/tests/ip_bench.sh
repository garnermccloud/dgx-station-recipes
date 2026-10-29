#!/usr/bin/env bash
# v2 + v3 interplay against the FULL peer_server3.py (b12x for v2 + the persistent kernel for v3): 8 cold experts of
# layer 0, b12x buckets 1..128 (PEER3_TEST_BUCKETS), private segments. WARM=1: only start the sidecar (compile cache).
set -euo pipefail
GB300=${GB300:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " "/GB300/{print $1; exit}")}
RTX=${RTX:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " "/RTX PRO 6000/{print $1; exit}")}
[ -n "$GB300" ] && [ -n "$RTX" ] || { echo "set GB300 and RTX to GPU UUIDs (nvidia-smi --query-gpu=uuid,name --format=csv,noheader)" >&2; exit 1; }
cd "$(dirname "$0")/../.."
: "${DS41_ROOT:=$PWD/..}"
IMAGE=vllm/vllm-openai:v0.31.0
. peer3/tests/bench_locks.sh
mkdir -p /dev/shm/w4rtx  # before docker, which would create the bind-mount source as root
S3=/dev/shm/w4rtx/peer3_ip S2=/dev/shm/w4rtx/peer2_ip
mkdir -p logs/ip-logs home/b12x-cache
python3 - <<'PY'
import json
cold = list(range(7, 7 + 13 * 8, 13))
json.dump({"layers": {"0": {"hot": [e for e in range(384) if e not in cold], "cold": cold}}}, open("logs/rowmap-ip.json", "w"))
PY
common=(--rm --network none --user "$(id -u):$(id -g)" --ipc private --ulimit memlock=-1 -e HOME=/w/home -e PYTHONUNBUFFERED=1
        -v "$PWD":/w -v /dev/shm/w4rtx:/dev/shm/w4rtx -v "${DS41_MODEL_DIR:-$HOME/.ds41/models/DeepSeek-V4.1-Flash-1017}":/model:ro -w /w
        -e VLLM_EXP_PEER3_SHM=$S3 -e VLLM_EXP_PEER2_SHM=$S2 -e VLLM_EXP_PEER2_SMALL=64 -e PEER3_DIR=/w/peer3 -e PEER3_PREBUILT
        -v "${DS41_ROOT:-/opt/ds41}/fastpath/hook":/opt/ds41/hook:ro
        -v "${DS41_ROOT:-/opt/ds41}/fastpath/hook":/hook:ro --entrypoint bash "$IMAGE" -c)
if [ "${WARM:-0}" = 1 ]; then take_bench_locks rtx; else take_bench_locks slot; fi  # slot: taken before the sidecar starts
rm -f "$S3" "$S2"
srvcmd="timeout -s KILL 150 python3 peer3/sidecar/peer_server3.py --rowmap logs/rowmap-ip.json --model /model --layers 1"
docker run --name "w4rtx-ip-rtx-$$" --device "nvidia.com/gpu=$RTX" -e CUDA_MODULE_LOADING=EAGER \
  -e PEER3_TEST_BUCKETS=1,2,4,8,16,32,64,128 -e B12X_AUTOTUNE=0 -e B12X_COMPILE_CACHE_DIR=/w/home/b12x-cache \
  -e TRITON_CACHE_DIR=/w/home/.triton -v "${DS41_ROOT:-/opt/ds41}/docker/b12x":/opt/ds41/b12x:ro -e PYTHONPATH=/opt/ds41/b12x \
  -v "$PWD/logs/ip-logs":/var/lib/ds41/sidecar "${common[@]}" "$srvcmd" > logs/ip-server.log 2>&1 &
srv=$!
if [ "${WARM:-0}" = 1 ]; then wait $srv || true; tail -5 logs/ip-server.log; exit 0; fi
for i in $(seq 1 110); do grep -q "^serving" logs/ip-server.log 2>/dev/null && break; sleep 0.5; done
grep -q "^serving" logs/ip-server.log || { echo "sidecar not serving:"; tail -5 logs/ip-server.log; wait $srv || true; exit 1; }
docker run --name "w4rtx-ip-gb-$$" --device "nvidia.com/gpu=$GB300" "${common[@]}" \
  "timeout -s KILL 120 python3 peer3/tests/test_interplay.py" || echo "client failed: $?"
wait $srv || true
grep -E "served|v3|peer3" logs/ip-server.log | tail -5
rm -f "$S3" "$S2"
