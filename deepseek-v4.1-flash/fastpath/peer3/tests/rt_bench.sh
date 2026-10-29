#!/usr/bin/env bash
# Cross-GPU round trip of peer tier v3: RTX server (persistent kernel, real experts) + GB300 client (graph replays).
# Refuses to run while production moves cold-tier traffic (the shared RTX would be time-sliced with the sidecar).
#   peer3/tests/rt_bench.sh [client args...]      env: SERVER_ARGS (extra rt_server.py args)
set -euo pipefail
GB300=${GB300:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " "/GB300/{print $1; exit}")}
RTX=${RTX:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " "/RTX PRO 6000/{print $1; exit}")}
[ -n "$GB300" ] && [ -n "$RTX" ] || { echo "set GB300 and RTX to GPU UUIDs (nvidia-smi --query-gpu=uuid,name --format=csv,noheader)" >&2; exit 1; }
cd "$(dirname "$0")/../.."
IMAGE=vllm/vllm-openai:v0.31.0
. peer3/tests/bench_locks.sh
common=(--rm --network none --user "$(id -u):$(id -g)" --ipc private --ulimit memlock=-1 -e HOME=/w/home
        -e PYTHONUNBUFFERED=1 -v "$PWD":/w -v /dev/shm/w4rtx:/dev/shm/w4rtx -v "${DS41_MODEL_DIR:-$HOME/.ds41/models/DeepSeek-V4.1-Flash-1017}":/model:ro
        -w /w --entrypoint bash "$IMAGE" -c)
take_bench_locks timing
prod_before=$(_prod_words)
mkdir -p /dev/shm/w4rtx
SEG=/dev/shm/w4rtx/peer3_rt
rm -f "$SEG" /dev/shm/w4rtx/rt_ready  # the server creates and sizes it
docker run --name "w4rtx-rt-rtx-$$" --device "nvidia.com/gpu=$RTX" -e PEER3_VARIANT -e PEER3_DEFINES -e PEER3_FC2_GATE -e PEER3_PREBUILT -e PEER3_PREFETCH_K -e PEER3_HINT_K -e PEER3_HINT_MIN_LAYER -e PEER3_HINT_STAGGER_NS -e PEER3_BENCH_DISCARD "${common[@]}" \
  "timeout -s KILL 60 python3 peer3/tests/rt_server.py --seg $SEG --idle 25 ${SERVER_ARGS:-}" &
srv=$!
docker run --name "w4rtx-rt-gb-$$" --device "nvidia.com/gpu=$GB300" -e PEER3_VARIANT -e PEER3_DEFINES -e PEER3_PREBUILT "${common[@]}" \
  "timeout -s KILL 55 python3 peer3/tests/${CLIENT:-rt_client.py} --seg $SEG $*" || echo "client failed: $?"
wait $srv || echo "server exit: $?"
[ "$(_prod_words)" = "$prod_before" ] || echo "CONTAMINATED: production served cold-tier calls during this run (timings invalid)"
rm -f "$SEG" /dev/shm/w4rtx/rt_ready
