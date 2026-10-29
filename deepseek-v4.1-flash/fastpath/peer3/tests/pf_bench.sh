#!/usr/bin/env bash
# Finish placement with MegaMoE in the graph: GB300 pf_client.py (serial / serial-pdl / serial-sms / parallel) against
# the RTX server (persistent kernel, real experts). The client warms DeepGEMM's JIT first; the server starts after
# that, so its run stays short. Holds the RTX lock and both GB300 slots; refuses to run under production traffic.
#   peer3/tests/pf_bench.sh [client args...]
set -euo pipefail
GB300=${GB300:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " "/GB300/{print $1; exit}")}
RTX=${RTX:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F", " "/RTX PRO 6000/{print $1; exit}")}
[ -n "$GB300" ] && [ -n "$RTX" ] || { echo "set GB300 and RTX to GPU UUIDs (nvidia-smi --query-gpu=uuid,name --format=csv,noheader)" >&2; exit 1; }
cd "$(dirname "$0")/../.."
IMAGE=vllm/vllm-openai:v0.31.0
. peer3/tests/bench_locks.sh
mkdir -p /dev/shm/w4rtx  # before docker, which would create the bind-mount source as root
common=(--rm --network none --user "$(id -u):$(id -g)" --ipc private --ulimit memlock=-1 -e HOME=/w/home
        -e PYTHONUNBUFFERED=1 -v "$PWD":/w -v /dev/shm/w4rtx:/dev/shm/w4rtx -v "${DS41_MODEL_DIR:-$HOME/.ds41/models/DeepSeek-V4.1-Flash-1017}":/model:ro
        -w /w --entrypoint bash "$IMAGE" -c)
take_bench_locks timing
SEG=/dev/shm/w4rtx/peer3_pf
rm -f "$SEG" /dev/shm/w4rtx/rt_ready /dev/shm/w4rtx/pf_warm
docker run --name "w6-pf-gb-$$" --device "nvidia.com/gpu=$GB300" "${common[@]}" \
  "timeout -s KILL 400 python3 peer3/tests/pf_client.py --seg $SEG $*" &
cli=$!
for i in $(seq 1 700); do [ -e /dev/shm/w4rtx/pf_warm ] && break; kill -0 $cli 2>/dev/null || break; sleep 0.5; done
if [ -e /dev/shm/w4rtx/pf_warm ]; then
  docker run --name "w6-pf-rtx-$$" --device "nvidia.com/gpu=$RTX" "${common[@]}" \
    "timeout -s KILL 60 python3 peer3/tests/rt_server.py --seg $SEG --idle 25 ${SERVER_ARGS:-}" || echo "server exit: $?"
fi
wait $cli || echo "client exit: $?"
rm -f "$SEG" /dev/shm/w4rtx/rt_ready /dev/shm/w4rtx/pf_warm
