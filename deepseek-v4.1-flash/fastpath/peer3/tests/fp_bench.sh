#!/usr/bin/env bash
# End-to-end test of the deliverable pair: fastpath_peer3 (GB300, stand-in experts class) against peer_server3.py
# in PEER3_ONLY mode (16 real cold experts of layer 0 in b12x's prepared layout). Own segment, all three locks.
# SERVER_GPU=gb300 runs the sidecar on the GB300 as well (two processes, time-sliced: correctness only), for when the
# production sidecar leaves too little RTX memory.
set -euo pipefail
GB300=${GB300:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " "/GB300/{print $1; exit}")}
RTX=${RTX:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " "/RTX PRO 6000/{print $1; exit}")}
[ -n "$GB300" ] && [ -n "$RTX" ] || { echo "set GB300 and RTX to GPU UUIDs (nvidia-smi --query-gpu=uuid,name --format=csv,noheader)" >&2; exit 1; }
cd "$(dirname "$0")/../.."
: "${DS41_ROOT:=$PWD/..}"
IMAGE=vllm/vllm-openai:v0.31.0
. peer3/tests/bench_locks.sh
mkdir -p /dev/shm/w4rtx  # before docker, which would create the bind-mount source as root
SEG=/dev/shm/w4rtx/peer3_fp
mkdir -p logs
python3 - <<'PY'
import json
cold = list(range(7, 7 + 13 * 16, 13))
json.dump({"layers": {"0": {"hot": [e for e in range(384) if e not in cold], "cold": cold}}}, open("logs/rowmap-fp.json", "w"))
PY
common=(--rm --network none --user "$(id -u):$(id -g)" --ipc private --ulimit memlock=-1 -e HOME=/w/home -e PYTHONUNBUFFERED=1
        -v "$PWD":/w -v /dev/shm/w4rtx:/dev/shm/w4rtx -v "${DS41_MODEL_DIR:-$HOME/.ds41/models/DeepSeek-V4.1-Flash-1017}":/model:ro -w /w
        -e VLLM_EXP_PEER3_SHM=$SEG -e PEER3_DIR=/w/peer3 -e PEER3_PREBUILT=${PEER3_PREBUILT:-} -e PEER3_DETERMINISTIC_PREFILL -e PEER3_PREFETCH_K
        -e PEER3_PDL -e PEER3_STATS_FILE=/w/logs/fp-peer3_stats.json -e PEER3_STATS_DUMP_S=1 --entrypoint bash "$IMAGE" -c)
take_bench_locks timing
rm -f "$SEG"
if [ "${SERVER_GPU:-rtx}" = gb300 ]; then sdev=(--device "nvidia.com/gpu=$GB300"); else
  sdev=(--device "nvidia.com/gpu=$RTX" -e CUDA_MODULE_LOADING=EAGER); fi
docker run --name "w6-fp-srv-$$" "${sdev[@]}" -e PEER3_ONLY=1 -e PEER3_IDLE_S=1 \
  -v "${DS41_ROOT:-/opt/ds41}/fastpath/hook":/opt/ds41/hook:ro -v "${DS41_ROOT:-/opt/ds41}/docker/b12x":/opt/ds41/b12x:ro \
  -e PYTHONPATH=/opt/ds41/b12x "${common[@]}" \
  "timeout -s KILL 60 python3 peer3/sidecar/peer_server3.py --rowmap logs/rowmap-fp.json --model /model --layers 1" &
srv=$!
for i in $(seq 1 120); do [ -e "$SEG" ] && break; sleep 0.5; done
sleep 4
docker run --name "w6-fp-gb-$$" --device "nvidia.com/gpu=$GB300" "${common[@]}" \
  "timeout -s KILL 55 python3 peer3/tests/test_fastpath_peer3.py" || echo "client failed: $?"
wait $srv || echo "server exit: $?"
[ -f logs/fp-peer3_stats.json ] && ./run.sh none "python3 peer3/py/peer3_stats.py show logs/fp-peer3_stats.json" | head -12
rm -f logs/fp-peer3_stats.json
rm -f "$SEG"
