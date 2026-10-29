#!/usr/bin/env bash
# ds41 serve: lifecycle for DeepSeek-V4.1-Flash-1017 on a DGX Station GB300 + RTX PRO 6000.
# The user-facing entry point is the `ds41` CLI; this script is its engine.
#   serve.sh start|stop|down|status|logs|doctor
# Configuration: ~/.ds41/config.env (optional; every value documented there), environment wins.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
. "$ROOT/serve/compat.sh"
: "${DS41_HOME:=$HOME/.ds41}"
CONFIG=${DS41_CONFIG:-$DS41_HOME/config.env}
if [ -f "$CONFIG" ]; then . "$CONFIG"; fi
# Recipe constants are fixed after the operator config so a stale config cannot change the pin.
. "$ROOT/config/recipe.env"
: "${DS41_MODEL_DIR:=$DS41_HOME/models/DeepSeek-V4.1-Flash-1017}"
: "${DS41_CACHE_DIR:=$DS41_HOME/state}"
: "${DS41_FASTPATH_DIR:=$ROOT/fastpath}"
: "${DS41_HOOK_DIR:=$DS41_FASTPATH_DIR/hook}"
: "${DS41_B12X_DIR:=$ROOT/docker/b12x}"
: "${DS41_OVERLAY_1017_DIR:=$ROOT/docker/overlay-1017/vllm}"
: "${DS41_OVERLAY_58132_DIR:=$ROOT/docker/overlay-58132/vllm}"
: "${DS41_SITECUSTOMIZE:=$ROOT/fastpath/sitecustomize.py}"
: "${DS41_PORT:=30003}"
: "${DS41_API_KEY_FILE:=$DS41_HOME/api-key}"
: "${DS41_SERVED:=deepseek-v4.1-flash}"
: "${DS41_ALIASES:=deepseek DeepSeek-V4.1-Flash dsv41}"
: "${DS41_EFFORT:=high}"
: "${DS41_GPU_GB300:=auto}"
: "${DS41_GPU_RTX:=auto}"
: "${DS41_GPU_MODE:=cdi}"
: "${DS41_UTIL:=$DS41_UTIL_DEFAULT}"
: "${DS41_GPU_RESERVE_MIB:=0}"
: "${DS41_MAXLEN:=524288}"
: "${DS41_KV:=nvfp4_ds_mla}"
: "${DS41_SEQS:=32}"
: "${DS41_OFFLOAD_GB:=120}"
: "${DS41_OFFLOAD_THRESHOLD:=2}"
: "${DS41_IMAGE:=$DS41_SUPPORTED_IMAGE}"
: "${DS41_LOG_LEVEL:=INFO}"
GB300_UUID=""; RTX_UUID=""; GB300_NAME=""; RTX_NAME=""
OVERLAY_MOUNTS=()

log()  { echo "[ds41] $*"; }
die()  {
  if [ "${STARTING:-0}" = 1 ] && [ "${STARTUP_LOGS_SAVED:-0}" = 0 ]; then
    save_startup_logs
  fi
  echo "[ds41] ERROR: $*" >&2
  exit 1
}
need() { command -v "$1" >/dev/null 2>&1 || die "missing $1"; }
running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" = true ]; }  # allow-2>/dev/null: absent container is "not running"; allow-||-true: same
ready() {
  # Health comes from the server, not from Docker health or the key file.  The key is optional for
  # status on a fresh box; when it exists, also require the authenticated model list to answer.
  curl -fsS -m 5 "http://127.0.0.1:$DS41_PORT/health" >/dev/null 2>&1 || return 1
  if [ -s "$DS41_API_KEY_FILE" ]; then
    curl -fsS -m 5 -H "Authorization: Bearer $(cat "$DS41_API_KEY_FILE")" \
      "http://127.0.0.1:$DS41_PORT/v1/models" >/dev/null 2>&1 || return 1
  fi
}

STARTING=0
STARTUP_LOGS_SAVED=0
build_overlay_mounts() {
  OVERLAY_MOUNTS=()
  local dir f rel
  for dir in "$DS41_OVERLAY_1017_DIR" "$DS41_OVERLAY_58132_DIR"; do
    [ -d "$dir" ] || die "missing overlay tree at $dir"
    while IFS= read -r -d '' f; do
      rel=${f#"$dir"/}
      OVERLAY_MOUNTS+=(-v "$f:$DS41_PKG/vllm/$rel:ro")
    done < <(find "$dir" -type f -print0 | sort -z)
  done
}

check_runtime_files() {
  case "$DS41_GPU_RESERVE_MIB" in ''|*[!0-9]*) die "DS41_GPU_RESERVE_MIB must be a whole number of MiB (0 on a dedicated GB300)";; esac
  [ -f "$DS41_SITECUSTOMIZE" ] || die "missing $DS41_SITECUSTOMIZE"
  [ -d "$DS41_FASTPATH_DIR" ] || die "missing fastpath tree at $DS41_FASTPATH_DIR"
  [ -d "$DS41_HOOK_DIR" ] || die "missing hook tree at $DS41_HOOK_DIR"
  [ -d "$DS41_B12X_DIR" ] || die "missing b12x tree at $DS41_B12X_DIR"
  [ -d "$DS41_FASTPATH_DIR/peer3" ] || die "missing peer3 tree at $DS41_FASTPATH_DIR/peer3"
  [ -d "$DS41_OVERLAY_1017_DIR" ] || die "missing 1017 overlay at $DS41_OVERLAY_1017_DIR"
  [ -f "$DS41_HOOK_DIR/$DS41_ROWMAP" ] || die "missing rowmap $DS41_HOOK_DIR/$DS41_ROWMAP"
  [ -f "$DS41_HOOK_DIR/fi-tune/$DS41_FI_TUNE_FILE" ] || die "missing tune file $DS41_HOOK_DIR/fi-tune/$DS41_FI_TUNE_FILE"
  [ -d "$DS41_OVERLAY_58132_DIR" ] || die "missing 58132 decoder overlay at $DS41_OVERLAY_58132_DIR"
  [ -n "$(find "$DS41_OVERLAY_58132_DIR" -type f -print -quit)" ] || die "58132 decoder overlay is empty at $DS41_OVERLAY_58132_DIR"
}

ensure_key() {
  mkdir -p "$(dirname "$DS41_API_KEY_FILE")"
  chmod 700 "$(dirname "$DS41_API_KEY_FILE")"
  if [ ! -s "$DS41_API_KEY_FILE" ]; then
    command -v openssl >/dev/null 2>&1 || die "openssl is required to generate the API key"
    umask 077
    openssl rand -hex 24 > "$DS41_API_KEY_FILE"
    log "generated API key at $DS41_API_KEY_FILE (keep it private)"
    echo "API key (save it): $(cat "$DS41_API_KEY_FILE")"
  fi
  chmod 600 "$DS41_API_KEY_FILE"
}

detect_gpus() {
  need nvidia-smi
  local line uuid name
  while IFS= read -r line; do
    uuid=${line%%,*}; name=${line#*, }
    uuid=$(echo "$uuid" | xargs); name=$(echo "$name" | xargs)
    case "$name" in
      *GB300*)                 [ -z "$GB300_UUID" ] && { GB300_UUID=$uuid; GB300_NAME=$name; } ;;
      *"RTX PRO 6000"*)        [ -z "$RTX_UUID" ]   && { RTX_UUID=$uuid;   RTX_NAME=$name; } ;;
    esac
  done < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader)
  if [ "$DS41_GPU_GB300" != auto ]; then
    GB300_UUID=$DS41_GPU_GB300
    GB300_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader --id="$GB300_UUID") || die "configured DS41_GPU_GB300=$GB300_UUID is not visible to nvidia-smi"
  fi
  if [ "$DS41_GPU_RTX" != auto ]; then
    RTX_UUID=$DS41_GPU_RTX
    RTX_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader --id="$RTX_UUID") || die "configured DS41_GPU_RTX=$RTX_UUID is not visible to nvidia-smi"
  fi
  [ -n "$GB300_UUID" ] || die "no NVIDIA GB300 found; this recipe needs a GB300 (nvidia-smi sees: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -))"
  [ -n "$RTX_UUID" ] || die "found only the GB300 / Fix: install the RTX PRO 6000 (96 GB); the model does not fit in the GB300 alone."
}

device_args() {
  if [ "$DS41_GPU_MODE" = cdi ]; then printf '%s\n' --device "nvidia.com/gpu=$1"; else printf '%s\n' --gpus "device=$1"; fi
}

check_model() {
  [ -d "$DS41_MODEL_DIR" ] || die "model not found at $DS41_MODEL_DIR; run: ds41 install"
  [ -f "$DS41_MODEL_DIR/config.json" ] || die "incomplete model at $DS41_MODEL_DIR (no config.json); run: ds41 install"
  [ -f "$DS41_MODEL_DIR/model.safetensors.index.json" ] || die "incomplete model at $DS41_MODEL_DIR (no index); run: ds41 install"
  [ -f "$DS41_MODEL_DIR/LICENSE" ] || log "WARNING: model LICENSE missing at $DS41_MODEL_DIR"
  local n; n=$(find "$DS41_MODEL_DIR" -maxdepth 1 -name 'model-*.safetensors' | wc -l)
  [ "$n" -ge 48 ] || die "model has $n weight shards; expected 48. Re-run: ds41 install (it resumes)"
}

check_image() {
  need docker
  docker image inspect "$DS41_IMAGE" >/dev/null 2>&1 \
    || die "image $DS41_IMAGE not found; run: ds41 install (it pulls $DS41_SUPPORTED_IMAGE)"
  ds41_assert_compat || die "compatibility check failed (see above)"
  ds41_image_layout_ok \
    || die "image $DS41_IMAGE is not the v0.31.0 layout (overlay targets missing)"
  ds41_image_numactl_ok \
    || die "image $DS41_IMAGE lacks numactl; this recipe requires the image's /usr/bin/numactl"
}

prepare_dirs() {
  mkdir -p "$DS41_CACHE_DIR"/{vllm,prof,flashinfer,sidecar-logs,b12x}
  mkdir -p "$DS41_CACHE_DIR"/jit/{tilelang,triton,nv,dj}
}

cleanup_shm() {
  docker run --rm -v /dev/shm:/dev/shm --entrypoint /bin/rm "$DS41_IMAGE" \
    -f /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3 >/dev/null
}

fix_state_ownership() {
  # Containers run as root and leave root-owned cache files.  Return the state tree to the
  # invoking user through a throwaway container so `rm -rf ~/.ds41` works without sudo.
  [ -d "$DS41_CACHE_DIR" ] || return 0
  command -v docker >/dev/null 2>&1 || return 0
  if ! docker image inspect "$DS41_IMAGE" >/dev/null 2>&1; then
    log "WARNING: image $DS41_IMAGE is missing; cannot return $DS41_CACHE_DIR ownership"
    return 0
  fi
  if docker run --rm --network none -v "$DS41_CACHE_DIR:/s" \
       --entrypoint /usr/bin/chown "$DS41_IMAGE" -R "$(id -u):$(id -g)" /s >/dev/null 2>&1; then
    log "returned $DS41_CACHE_DIR to $(id -u):$(id -g)"
  else
    log "WARNING: could not return $DS41_CACHE_DIR ownership"
  fi
}

prune_startup_logs() {
  local name=$1
  local -a old
  mapfile -t old < <(find "$DS41_CACHE_DIR/logs" -maxdepth 1 -type f -name "*-$name.log" \
    -printf '%T@ %p\n' | sort -rn | tail -n +11 | cut -d' ' -f2-)
  if [ "${#old[@]}" -gt 0 ]; then
    rm -f -- "${old[@]}"
  fi
}

print_startup_root_cause() {
  local log=$1 line
  if line=$(grep -nE -m1 '(EngineCore.*(ERROR|Exception)|Traceback|FATAL|Error:)' "$log" | cut -d: -f1); then
    printf '[ds41] root cause (line %s, 40 lines after):\n' "$line" >&2
    sed -n "${line},$((line + 40))p" "$log" >&2
  else
    printf '[ds41] root cause: no EngineCore/Traceback/FATAL/Error line found\n' >&2
  fi
  printf '[ds41] last 20 lines:\n' >&2
  tail -n 20 "$log" >&2
}

save_startup_logs() {
  [ "${STARTUP_LOGS_SAVED:-0}" = 1 ] && return 0
  STARTUP_LOGS_SAVED=1
  local dir="$DS41_CACHE_DIR/logs" stamp ds41_log sidecar_log
  if ! mkdir -p "$dir"; then
    printf '[ds41] could not create startup log directory %s\n' "$dir" >&2
    return 0
  fi
  if ! stamp=$(date -u +%Y%m%dT%H%M%SZ); then
    stamp=unknown
  fi
  ds41_log="$dir/$stamp-ds41.log"
  sidecar_log="$dir/$stamp-sidecar.log"
  if ! docker logs ds41 > "$ds41_log" 2>&1; then
    printf '[ds41] could not read full logs for ds41; see %s\n' "$ds41_log" >> "$ds41_log"
  fi
  if ! docker logs ds41-sidecar > "$sidecar_log" 2>&1; then
    printf '[ds41] could not read full logs for ds41-sidecar; see %s\n' "$sidecar_log" >> "$sidecar_log"
  fi
  prune_startup_logs ds41
  prune_startup_logs sidecar
  printf '[ds41] startup logs: %s %s\n' "$ds41_log" "$sidecar_log" >&2
  print_startup_root_cause "$ds41_log"
}

container_status() {
  docker inspect -f '{{.State.Status}} {{.State.ExitCode}} {{.RestartCount}}' "$1"
}

require_alive() {
  local name=$1 phase=$2 state status code restarts
  if ! state=$(container_status "$name"); then
    die "$name disappeared during $phase"
  fi
  read -r status code restarts <<< "$state"
  case "$status" in
    running|created) ;;
    exited|dead|restarting)
      die "$name $status during $phase (exit code ${code:-unknown}, restarts ${restarts:-unknown})"
      ;;
    *)
      die "$name unexpected state '${status:-unknown}' during $phase"
      ;;
  esac
  if [ "${restarts:-0}" -ne 0 ]; then
    die "$name restarted $restarts time(s) during $phase"
  fi
}

wait_sidecar() {
  local t=0 summary
  while :; do
    if docker logs ds41-sidecar 2>&1 | grep -q '^serving'; then break; fi
    require_alive ds41-sidecar "RTX sidecar startup"
    t=$((t + 5))
    if [ "$t" -gt 3600 ]; then
      die "RTX sidecar did not report 'serving' within 60 min"
    fi
    sleep 5
  done
  if summary=$(docker logs ds41-sidecar 2>&1 | grep -E '^prepared|^serving' | tail -2); then
    printf '%s\n' "$summary"
  fi
  docker update --restart unless-stopped ds41-sidecar >/dev/null
}

remove_container() {
  if docker inspect "$1" >/dev/null 2>&1; then  # allow-2>/dev/null: idempotent cleanup; absent container is normal
    docker rm -f "$1" >/dev/null
  fi
}

start_sidecar() {
  if running ds41-sidecar; then log "RTX sidecar already running"; return 0; fi
  remove_container ds41-sidecar
  cleanup_shm
  local dev; mapfile -t dev < <(device_args "$RTX_UUID")
  local mounts=(-v "$DS41_MODEL_DIR:/model:ro"
                -v "$DS41_CACHE_DIR/sidecar-logs:/var/lib/ds41/sidecar"
                -v "$DS41_CACHE_DIR/b12x:/var/lib/ds41/b12x-cache"
                -v "$DS41_HOOK_DIR:/opt/ds41/hook:ro"
                -v "$DS41_FASTPATH_DIR/peer3:/opt/ds41/peer3:ro"
                -v "$DS41_B12X_DIR:/opt/ds41/b12x:ro")
  local envs=(-e PYTHONPATH=/opt/ds41/b12x -e PYTHONUNBUFFERED=1
              -e B12X_COMPILE_CACHE_DIR=/var/lib/ds41/b12x-cache/compile
              -e TRITON_CACHE_DIR=/var/lib/ds41/b12x-cache/triton
              -e PEER3_DIR=/opt/ds41/peer3 -e PEER3_PREBUILT=/opt/ds41/peer3/prebuilt
              -e CUDA_MODULE_LOADING=EAGER
              -e PEER3_PREFETCH_K=4)  # read by the prebuilt RTX kernel (peer3_rtx.cu): v3.2 next-layer L2 prefetch
  log "starting RTX sidecar on $RTX_NAME ($RTX_UUID)"
  docker run -d --name ds41-sidecar --restart no "${dev[@]}" --ipc host --network none \
    --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 \
    "${mounts[@]}" "${envs[@]}" --entrypoint /usr/bin/numactl "$DS41_IMAGE" --membind=0 \
    python3 /opt/ds41/peer3/sidecar/peer_server3.py \
    --rowmap "/opt/ds41/hook/$DS41_ROWMAP" --model /model >/dev/null
  wait_sidecar
}

start_vllm() {
  remove_container ds41
  prepare_dirs
  local fastpath=$DS41_FASTPATH_COMPONENTS
  local dev; mapfile -t dev < <(device_args "$GB300_UUID")
  build_overlay_mounts
  local mounts=(-v "$DS41_MODEL_DIR:/model:ro" -v "$DS41_CACHE_DIR/vllm:/root/.cache/vllm"
                -v "$DS41_CACHE_DIR/prof:/prof" -v "$DS41_CACHE_DIR/flashinfer:/root/.cache/flashinfer"
                -v "$DS41_CACHE_DIR/jit/tilelang:/root/.tilelang" -v "$DS41_CACHE_DIR/jit/triton:/root/.triton"
                -v "$DS41_CACHE_DIR/jit/nv:/root/.nv" -v "$DS41_CACHE_DIR/jit/dj:/root/.dj"
                -v "$DS41_HOOK_DIR:/w:ro"
                -v "$DS41_SITECUSTOMIZE:$DS41_SITE_TARGET:ro"
                -v "$DS41_FASTPATH_DIR:/fastpath:ro"
                "${OVERLAY_MOUNTS[@]}")
  # Every variable below is read by vLLM or by the mounted hook/fastpath; the fastpath's tuning values are code
  # constants in its components, not container env.
  local envs=(-e HF_HUB_OFFLINE=1 -e "VLLM_LOGGING_LEVEL=$DS41_LOG_LEVEL" -e CUDA_LOG_FILE=stderr
              -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_round_threshold_mb:512
              -e "MEGA_ROWMAP=/w/$DS41_ROWMAP"
              -e VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=mxfp8_gemm
              -e "MEGA_FI_TUNE_FILE=/w/fi-tune/$DS41_FI_TUNE_FILE"
              -e "DS41_FASTPATH=$fastpath"
              -e "DS41_GPU_RESERVE_MIB=$DS41_GPU_RESERVE_MIB"
              -e PEER3_PREBUILT=/fastpath/peer3/prebuilt -e VLLM_EXP_PEER3_SHM=/dev/shm/vllm_peer_tier3)
  local envf; envf=$(mktemp); trap "rm -f '$envf'" RETURN
  chmod 600 "$envf"; echo "VLLM_API_KEY=$(cat "$DS41_API_KEY_FILE")" > "$envf"
  local spec=(--speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,8,3],[9,32,3]],"draft_sample_method":"probabilistic","rejection_sample_method":"block"}')
  local offload=()
  if [ "$DS41_OFFLOAD_GB" != 0 ]; then offload=(--kv-offloading-size "$DS41_OFFLOAD_GB" --kv-offloading-backend native --kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"store_threshold\":$DS41_OFFLOAD_THRESHOLD}}"); fi
  local aliases; read -r -a aliases <<< "$DS41_ALIASES"
  log "starting GB300 vLLM on $GB300_NAME ($GB300_UUID); util=$DS41_UTIL, reserve=${DS41_GPU_RESERVE_MIB} MiB, rowmap=$DS41_ROWMAP, maxlen=$DS41_MAXLEN"
  docker run -d --name ds41 --restart no "${dev[@]}" --ipc host --network host \
    --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --security-opt label=disable \
    --env-file "$envf" "${mounts[@]}" "${envs[@]}" --entrypoint /usr/bin/numactl "$DS41_IMAGE" --membind=0 \
    vllm serve --model /model --served-model-name "$DS41_SERVED" "${aliases[@]}" --trust-remote-code \
    --tensor-parallel-size 1 --moe-backend deep_gemm_mega_moe --enable-expert-parallel \
    --engram-config '{"cpu_offload": true}' --max-model-len "$DS41_MAXLEN" --max-num-seqs "$DS41_SEQS" \
    --max-num-batched-tokens 8192 --gpu-memory-utilization "$DS41_UTIL" --kv-cache-dtype "$DS41_KV" \
    --attention-config '{"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": true}' \
    --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
    --long-prefill-token-threshold 6144 --enable-prefix-caching "${spec[@]}" \
    --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 20 24 28 32 36 40 48 56 64 72 80 96 128 144 160 192 224 256 \
    --override-generation-config '{"temperature": 1.0, "top_p": 0.95}' \
    --default-chat-template-kwargs "{\"reasoning_effort\":\"$DS41_EFFORT\"}" "${offload[@]}" \
    --host 0.0.0.0 --port "$DS41_PORT" >/dev/null
  local t=0 summary
  while :; do
    if ready; then break; fi
    require_alive ds41 "vLLM startup"
    t=$((t + 5))
    if [ "$t" -gt 3600 ]; then
      die "vLLM did not become ready within 60 min"
    fi
    sleep 5
  done
  docker update --restart unless-stopped ds41 >/dev/null
  if summary=$(docker logs ds41 2>&1 | grep -E "GPU KV cache size|Decoder SWA bounded replay|A12_INSTALL_MEMORY|MEGA" | cut -c1-240 | tail -5); then
    printf '%s\n' "$summary"
  fi
  log "ready: http://127.0.0.1:$DS41_PORT/v1  (model $DS41_SERVED)"
  log "sample: curl -s http://127.0.0.1:$DS41_PORT/v1/models -H \"Authorization: Bearer \$(cat $DS41_API_KEY_FILE)\""
}

cmd_start() {
  detect_gpus
  check_model
  check_image
  check_runtime_files
  ensure_key
  prepare_dirs
  if running ds41; then
    if ready; then log "already up"; cmd_status; return 0; fi
    log "ds41 is running but not ready; restarting"
    cmd_stop
  fi
  # If either container fails during startup, do not leave the sidecar (or a half-started ds41)
  # behind.  The README promises that `ds41 up` is not left half-started.
  STARTUP_LOGS_SAVED=0
  STARTING=1
  trap 'rc=$?; if [ "${STARTING:-0}" = 1 ]; then save_startup_logs; remove_container ds41; remove_container ds41-sidecar; fix_state_ownership; fi; exit $rc' EXIT
  start_sidecar
  start_vllm
  STARTING=0
  trap - EXIT
}

cmd_stop() {
  if running ds41; then
    docker update --restart no ds41 >/dev/null
    docker stop -t 60 ds41 >/dev/null
  fi
  if running ds41-sidecar; then
    docker update --restart no ds41-sidecar >/dev/null
    docker stop -t 30 ds41-sidecar >/dev/null
  fi
  fix_state_ownership
  log "stopped"
}

cmd_down() {
  cmd_stop
  remove_container ds41
  remove_container ds41-sidecar
  log "removed ds41 and ds41-sidecar"
}

cmd_status() {
  local health=unhealthy
  if running ds41 && ready; then health=healthy; fi
  echo "ds41: $health"
  if running ds41; then
    echo "model: $DS41_SERVED ($(docker inspect -f '{{.State.StartedAt}}' ds41 2>/dev/null))"  # allow-2>/dev/null: status probe; running() already guarded
    echo "endpoint: http://127.0.0.1:$DS41_PORT/v1"
    echo "curl: curl -s http://127.0.0.1:$DS41_PORT/v1/models -H \"Authorization: Bearer \$(cat $DS41_API_KEY_FILE)\""
    [ -s "$DS41_API_KEY_FILE" ] || echo "api key: not created yet (run ds41 install)"
    echo "sidecar: $(running ds41-sidecar && echo running || echo 'not running')"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | grep -E 'GB300|RTX PRO 6000' | sed 's/^/gpu: /' || true  # allow-2>/dev/null: status probe; allow-||-true: optional GPU display
    docker stats --no-stream --format 'cpu/mem: {{.CPUPerc}} {{.MemUsage}}' ds41 2>/dev/null || true  # allow-2>/dev/null: status probe; allow-||-true: optional stats
  else
    echo "model: not loaded"
    echo "endpoint: http://127.0.0.1:$DS41_PORT/v1 (not serving)"
    echo "start: ds41 up"
  fi
}

cmd_logs() {
  local svc=${1:-ds41}
  case "$svc" in
    --boot) docker logs --tail 200 ds41; return;;
    --kernel) docker logs ds41 2>&1 | grep -iE 'jit|kernel|compile|nvcc|flashinfer' | tail -200; return;;
  esac
  [ "$svc" = sidecar ] && svc=ds41-sidecar
  docker logs --tail 200 -f "$svc"
}

cmd_doctor() {
  local ok=0
  p() { printf '%-28s %s\n' "$1" "$2"; }
  if command -v nvidia-smi >/dev/null 2>&1; then
    p "nvidia-smi" "PASS ($(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1))"
  else p "nvidia-smi" "FAIL: install the NVIDIA driver"; ok=1; fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n; n=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c GB300 || true)  # allow-||-true: grep -c exits 1 on zero matches; zero is handled below
    [ "$n" -ge 1 ] && p "GB300" "PASS" || { p "GB300" "FAIL: no GB300 visible"; ok=1; }
    n=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c 'RTX PRO 6000' || true)  # allow-||-true: grep -c exits 1 on zero matches; zero is handled below
    if [ "$n" -ge 1 ]; then
      p "RTX PRO 6000" "PASS"
    else
      p "RTX PRO 6000" "FAIL: found only the GB300 / Fix: install the RTX PRO 6000 (96 GB); the model does not fit in the GB300 alone."
      ok=1
    fi
  fi
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then p "docker" "PASS"; else p "docker" "FAIL: install Docker + NVIDIA Container Toolkit"; ok=1; fi
  if docker image inspect "$DS41_IMAGE" >/dev/null 2>&1; then
    if ds41_assert_compat; then p "compat" "PASS ($DS41_SUPPORTED_IMAGE; DeepGEMM base hashes)"; else p "compat" "FAIL: see above"; ok=1; fi
    if ds41_image_layout_ok; then p "image" "PASS ($DS41_IMAGE; overlay targets)"; else p "image" "FAIL: $DS41_IMAGE lacks v0.31.0 overlay targets"; ok=1; fi
  else
    p "compat" "WARN: image not pulled yet"
    p "image" "FAIL: run ds41 install (pulls $DS41_SUPPORTED_IMAGE)"; ok=1
  fi
  if [ -d "$DS41_MODEL_DIR" ] && [ -f "$DS41_MODEL_DIR/config.json" ]; then p "model" "PASS"; else p "model" "FAIL: run ds41 install"; ok=1; fi
  if [ -s "$DS41_API_KEY_FILE" ] && [ "$(stat -c '%a' "$DS41_API_KEY_FILE")" = 600 ]; then p "api key" "PASS (mode 600)"; else p "api key" "FAIL: run ds41 install"; ok=1; fi
  if docker image inspect "$DS41_IMAGE" >/dev/null 2>&1; then
    if ds41_image_numactl_ok; then p "numactl" "PASS (in image)"; else p "numactl" "FAIL: image lacks numactl"; ok=1; fi
  else
    p "numactl" "WARN: image not pulled yet"
  fi
  local mem; mem=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)
  if [ "$mem" -ge 480 ]; then p "host RAM" "PASS (${mem} GiB; 480+ recommended)"; else p "host RAM" "WARN: ${mem} GiB; 480+ GiB recommended"; fi
  local shm; shm=$(df -P /dev/shm | awk 'NR==2 {print int($4/1024/1024)}')
  if [ "$shm" -ge 16 ]; then p "/dev/shm" "PASS (${shm} GiB free; peer tier)"; else p "/dev/shm" "WARN: ${shm} GiB free; peer tier needs at least 16 GiB"; fi
  local disk=""; disk=$(df -P "$(dirname "$DS41_MODEL_DIR")" 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}' || true); if [ -n "$disk" ] && [ "$disk" -ge 600 ]; then p "model disk" "PASS (${disk} GiB free)"; else p "model disk" "WARN: ${disk:-unknown} GiB free; need ~600 GiB"; fi  # allow-2>/dev/null: filesystem probe; allow-||-true: unknown disk is a WARN
  [ "$ok" = 0 ] && echo "doctor: PASS" || echo "doctor: FAIL (fix the FAIL lines above)"
  return "$ok"
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  down) cmd_down ;;
  status) cmd_status ;;
  logs) shift; cmd_logs "${1:-ds41}" ;;
  doctor) cmd_doctor ;;
  *) echo "usage: $0 start|stop|down|status|logs [ds41|sidecar|--boot|--kernel]|doctor"; exit 2 ;;
esac
