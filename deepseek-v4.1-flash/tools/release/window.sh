#!/usr/bin/env bash
# Release validation window: a fresh clone of a recipe branch, booted with the release defaults (or the util /
# rowmap given here), gated, verified and benchmarked; optionally a sweep over util:rowmap candidates first.
#
#   tools/release/window.sh --model-dir DIR [--repo PATH|URL] [--branch NAME] [--util U] [--rowmap FILE|NAME]
#                           [--reserve-mib N] [--sweep U:R,U:R,...] [--sweep-skip STAGES] [--skip-final]
#                           [--out DIR] [--cache-dir DIR] [--port N]
#
#   --model-dir   an existing complete DeepSeek-V4.1-Flash-1017 checkpoint; checked read-only, never downloaded
#   --repo        the recipe repository to clone (default: the checkout this script lives in)
#   --branch      the branch to clone (default: that checkout's current branch)
#   --util        --gpu-memory-utilization for the final run (default: DS41_UTIL_DEFAULT in the clone's recipe.env)
#   --rowmap      rowmap for the final run: a file (copied into the clone's fastpath/hook/) or a name already there
#                 (default: DS41_ROWMAP in the clone's recipe.env)
#   --reserve-mib DS41_GPU_RESERVE_MIB for every boot (default 0: nothing else runs on the GB300)
#   --sweep       util:rowmap candidates, each booted and measured with a short bench before the final run;
#                 an empty util or rowmap takes the recipe default (e.g. 0.90:,0.92:/path/rowmap-h265.json)
#   --sweep-skip  bench --full stages skipped for sweep candidates (default greedy_c1,coding,prefill,quality)
#   --skip-final  only the sweep
#   --out         results directory (default ./release-window-<time>)
#   --cache-dir   persistent kernel/JIT cache shared by every boot of this window and later windows
#                 (default <out>/state; a cold cache makes the first boot much longer)
#
# The final run: ds41 install --no-model, ds41 up, the boot gate (every release component ACTIVE, no fallback
# line, memory guard PASS), ds41 verify (preflight, tool call, needles; GSM8K-200 runs in the bench), and
# ds41 bench --full (one session: single stream, 4-32 streams, prefill, capacity, quality). Everything runs
# under a private DS41_HOME inside --out; the checkpoint is mounted read-only.
#
# Safety: the window refuses to start if containers named ds41 or ds41-sidecar already exist (the recipe uses
# those names and would replace them), if the port is in use, or if the GB300 or the RTX PRO 6000 already has
# more than 2 GiB in use. It removes only the ds41/ds41-sidecar containers it created, on every exit.
# It ends with one PASS/FAIL summary (SUMMARY.md and summary.json in --out).
set -euo pipefail

SELF_DIR=$(cd "$(dirname "$0")" && pwd)
RECIPE_DIR=$(cd "$SELF_DIR/../.." && pwd)
REPO=$(git -C "$RECIPE_DIR" rev-parse --show-toplevel)
RECIPE=$(realpath --relative-to="$REPO" "$RECIPE_DIR")   # the recipe's directory inside the repository
BRANCH=""
MODEL_DIR=${DS41_MODEL_DIR:-}
UTIL=""
ROWMAP=""
RESERVE=0
SWEEP=""
SWEEP_SKIP=greedy_c1,coding,prefill,quality
SKIP_FINAL=0
OUT=""
CACHE=""
PORT=30003
IDLE_MIB=2048

die() { echo "[window] FAIL: $*" >&2; exit 1; }
log() { echo "[window] $(date -u +%H:%M:%S) $*"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO=$2; shift;;
    --branch) BRANCH=$2; shift;;
    --model-dir) MODEL_DIR=$2; shift;;
    --util) UTIL=$2; shift;;
    --rowmap) ROWMAP=$2; shift;;
    --reserve-mib) RESERVE=$2; shift;;
    --sweep) SWEEP=$2; shift;;
    --sweep-skip) SWEEP_SKIP=$2; shift;;
    --skip-final) SKIP_FINAL=1;;
    --out) OUT=$2; shift;;
    --cache-dir) CACHE=$2; shift;;
    --port) PORT=$2; shift;;
    -h|--help) sed -n '2,32p' "$0"; exit 0;;
    *) die "unknown option $1 (--help)";;
  esac
  shift
done
[ -n "$MODEL_DIR" ] || die "--model-dir is required (an existing complete checkpoint)"
# Nothing from the caller's environment may change the recipe: every DS41_* setting comes from this script.
while IFS= read -r v; do unset "$v"; done < <(env | sed -n 's/^\(DS41_[A-Z0-9_]*\)=.*/\1/p')
[ -z "$BRANCH" ] && BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
OUT=$(mkdir -p "${OUT:-./release-window-$(date -u +%Y%m%dT%H%M%SZ)}" && cd "${OUT:-.}" && pwd)
[ -n "$OUT" ] || die "cannot create the output directory"
CACHE=${CACHE:-$OUT/state}
mkdir -p "$CACHE"
exec > >(tee -a "$OUT/window.log") 2>&1

CLONE_TOP=$OUT/repo
CLONE=$CLONE_TOP/$RECIPE
WHOME=$OUT/home
CREATED=0
RESULTS=()   # one JSON object per configuration

# ---------------------------------------------------------------------------------------------- safety checks
container_exists() { docker container inspect "$1" >/dev/null 2>&1; }  # allow-2>/dev/null: absence is the answer
gpu_used_mib() {
  nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' -v n="$1" 'index($1, n) { print $2; exit }'
}
preflight_window() {
  for c in ds41 ds41-sidecar; do
    container_exists "$c" && die "a container named $c already exists; stop and remove it first (the recipe uses this name)"
  done
  if command -v ss >/dev/null && [ -n "$(ss -ltnH "sport = :$PORT")" ]; then die "port $PORT is in use"; fi
  local g r
  g=$(gpu_used_mib GB300); r=$(gpu_used_mib "RTX PRO 6000")
  [ -n "$g" ] || die "no GB300 visible to nvidia-smi"
  [ -n "$r" ] || die "no RTX PRO 6000 visible to nvidia-smi"
  [ "$g" -le "$IDLE_MIB" ] || die "the GB300 has ${g} MiB in use; the window needs it idle (<= $IDLE_MIB MiB)"
  [ "$r" -le "$IDLE_MIB" ] || die "the RTX PRO 6000 has ${r} MiB in use; the window needs it idle (<= $IDLE_MIB MiB)"
  log "GPUs idle: GB300 ${g} MiB, RTX PRO 6000 ${r} MiB in use"
}

# ---------------------------------------------------------------------------------------------- clone + config
ds41() { DS41_HOME="$WHOME" DS41_MODEL_DIR="$MODEL_DIR" DS41_CACHE_DIR="$CACHE" DS41_PORT="$PORT" \
  DS41_GPU_RESERVE_MIB="$RESERVE" "$CLONE/ds41" "$@"; }

prepare_clone() {
  rm -rf "$CLONE_TOP"
  git clone -q --branch "$BRANCH" "$REPO" "$CLONE_TOP"
  log "clone $(git -C "$CLONE" rev-parse HEAD) ($BRANCH from $REPO)"
  DS41_MODEL_DIR="$MODEL_DIR" "$CLONE/scripts/setup_model.sh" --check \
    || die "the checkpoint at $MODEL_DIR is incomplete; the window never downloads"
  for f in $(git -C "$CLONE" ls-files '*.sh') ds41; do bash -n "$CLONE/$f" || die "bash -n $f"; done
  mkdir -p "$WHOME"
  ds41 install --no-model
}

# Set the util default and rowmap in the clone's recipe.env (the release's one place for them).
configure() {
  local util=$1 rowmap=$2 recipe=$CLONE/config/recipe.env
  git -C "$CLONE" checkout -q -- config/recipe.env
  if [ -n "$rowmap" ]; then
    if [ -f "$rowmap" ]; then
      cp "$rowmap" "$CLONE/fastpath/hook/"
      rowmap=$(basename "$rowmap")
    fi
    [ -f "$CLONE/fastpath/hook/$rowmap" ] || die "rowmap $rowmap is neither a file nor in fastpath/hook/"
    sed -i "s|^DS41_ROWMAP=.*|DS41_ROWMAP='$rowmap'|" "$recipe"
  fi
  if [ -n "$util" ]; then
    case "$util" in 0.[0-9]*) ;; *) die "util $util is not a fraction";; esac
    sed -i "s|^DS41_UTIL_DEFAULT=.*|DS41_UTIL_DEFAULT='$util'|" "$recipe"
  fi
  # shellcheck disable=SC1090
  ( . "$recipe"; echo "util=$DS41_UTIL_DEFAULT rowmap=$DS41_ROWMAP" )
}

# ---------------------------------------------------------------------------------------------- one boot
teardown() {
  if [ "$CREATED" = 1 ]; then
    log "removing the window's ds41 / ds41-sidecar containers"
    ds41 down || log "ds41 down reported an error; check 'docker ps -a'"
    CREATED=0
  fi
}
trap 'rc=$?; teardown; exit $rc' EXIT

# ds41 up is bounded by serve.sh itself (60 min per container, fail-fast when a container dies).
boot() {
  local dir=$1
  CREATED=1
  local t0=$SECONDS
  if ds41 up > "$dir/up.log" 2>&1; then
    echo $((SECONDS - t0)) > "$dir/boot_s"
    docker logs ds41 > "$dir/ds41.log" 2>&1
    docker logs ds41-sidecar > "$dir/sidecar.log" 2>&1
    return 0
  fi
  echo $((SECONDS - t0)) > "$dir/boot_s"
  docker logs ds41 > "$dir/ds41.log" 2>&1 || echo "no ds41 container log" > "$dir/ds41.log"
  tail -40 "$dir/up.log"
  return 1
}

# The boot gate: the repo's own preflight boot check plus the evidence line of each production component.
gate() {
  local dir=$1 logf=$1/ds41.log ok=0
  DS41_HOME="$WHOME" "$CLONE/verify/preflight.sh" boot > "$dir/gate.txt" 2>&1 || ok=1
  {
    echo "--- evidence"
    grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$logf" || { echo "MISSING GPU KV cache size"; ok=1; }
    grep -m1 -oE 'W6_MEGAMOE: JIT include root .*shared-early.*' "$logf" || { echo "MISSING shared-early MegaMoE"; ok=1; }
    grep -m1 -oE 'coldskip: load 1: skipped [0-9]+ cold-expert tensors[^)]*\)' "$logf" || { echo "MISSING coldskip count"; ok=1; }
    grep -m2 -oE 'w8oproj: split policy [a-z]+: split=[0-9]+' "$logf" || { echo "MISSING w8oproj split policy"; ok=1; }
    grep -m1 -oE '"fused_t6_keys": \[[^]]+\]' "$logf" || { echo "MISSING quant_producer T6 capture"; ok=1; }
    grep -m1 -oE '"fused_t8_keys": \[[^]]+\]' "$logf" || { echo "MISSING quant_producer T8 capture"; ok=1; }
    grep -F 'DS41_A12_RETIRE' "$logf" | grep -m1 -oF '"event": "POST_LOAD"' || { echo "MISSING a12 POST_LOAD"; ok=1; }
    grep -F 'A12_INSTALL_MEMORY' "$logf" | grep -m1 -oE '"headroom_bytes": -?[0-9]+' || { echo "MISSING guard headroom"; ok=1; }
  } >> "$dir/gate.txt"
  cat "$dir/gate.txt"
  return $ok
}

field() { grep -m1 -oE "$2" "$1" | grep -oE -- '-?[0-9][0-9,.]*' | tail -1 | tr -d , ; }

# run_config NAME UTIL ROWMAP BENCH_ARGS...: boot, gate, verify (final only), bench; append a result object.
run_config() {
  local name=$1 util=$2 rowmap=$3; shift 3
  local dir=$OUT/$name boot_ok=false gate_ok=false verify_ok=null bench_ok=false
  mkdir -p "$dir"
  configure "$util" "$rowmap" > "$dir/config.txt"
  log "=== $name: $(cat "$dir/config.txt")"
  if boot "$dir"; then
    boot_ok=true
    log "$name: ready after $(cat "$dir/boot_s") s"
    if gate "$dir"; then gate_ok=true; fi
    if [ "$name" = final ]; then
      verify_ok=false
      if DS41_HOME="$WHOME" DS41_MODEL_DIR="$MODEL_DIR" DS41_PORT="$PORT" "$CLONE/verify/verify.sh" \
           --skip-gsm8k --skip-decode --skip-coding > "$dir/verify.txt" 2>&1; then verify_ok=true; fi
      cat "$dir/verify.txt"
    fi
    if ds41 bench --full --out "$dir/bench" "$@" > "$dir/bench.txt" 2>&1; then bench_ok=true; fi
    tail -60 "$dir/bench.txt"
  else
    log "$name: boot FAILED (see $dir/up.log, $dir/ds41.log)"
  fi
  local bench_json; bench_json=$(ls "$dir"/bench/bench-*.json 2>/dev/null | tail -1 || true)  # allow-2>/dev/null: no bench JSON after a failed boot; allow-||-true: same
  local kv headroom
  kv=$(field "$dir/ds41.log" 'GPU KV cache size: [0-9,]+ tokens' || true)  # allow-||-true: absent after a failed boot; reported as null
  headroom=$(grep -F 'A12_INSTALL_MEMORY' "$dir/ds41.log" | field /dev/stdin '"headroom_bytes": -?[0-9]+' || true)  # allow-||-true: same
  teardown
  RESULTS+=("$(printf '{"name":"%s","config":"%s","boot":%s,"boot_s":%s,"gate":%s,"verify":%s,"bench":%s,"kv_cache_tokens":%s,"guard_headroom_bytes":%s,"bench_json":"%s"}' \
    "$name" "$(cat "$dir/config.txt")" "$boot_ok" "$(cat "$dir/boot_s" 2>/dev/null || echo null)" "$gate_ok" "$verify_ok" "$bench_ok" \
    "${kv:-null}" "${headroom:-null}" "$bench_json")")  # allow-2>/dev/null: boot_s is absent only if boot never ran
}

summary() {
  local overall=PASS r
  {
    echo "# Release window $(date -u +%Y-%m-%dT%H:%MZ)"
    echo
    echo "Recipe: $REPO @ $BRANCH ($(git -C "$CLONE" rev-parse HEAD)); checkpoint $MODEL_DIR (read-only); reserve $RESERVE MiB."
    echo
    echo "| run | config | boot | boot s | gate | verify | bench | KV tokens | guard headroom GiB | bench JSON |"
    echo "|---|---|---|---|---|---|---|---|---|---|"
    for r in "${RESULTS[@]}"; do
      python3 -c 'import json,sys; d=json.loads(sys.argv[1]); h=d["guard_headroom_bytes"]
print("| {name} | {config} | {boot} | {boot_s} | {gate} | {verify} | {bench} | {kv_cache_tokens} | {h} | {bench_json} |".format(
    h="n/a" if h is None else round(h/2**30, 2), **d))' "$r"
    done
    echo
    for r in "${RESULTS[@]}"; do
      local j; j=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["bench_json"])' "$r")
      [ -n "$j" ] && [ -f "${j%.json}.md" ] && { echo "## $(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["name"])' "$r")"; echo; sed 1d "${j%.json}.md"; echo; }
    done
  } > "$OUT/SUMMARY.md"
  printf '[%s]\n' "$(IFS=,; echo "${RESULTS[*]}")" > "$OUT/summary.json"
  for r in "${RESULTS[@]}"; do
    python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d["boot"] and d["gate"] and d["bench"] and d["verify"] is not False else 1)' "$r" \
      || overall=FAIL
  done
  [ "${#RESULTS[@]}" -gt 0 ] || overall=FAIL
  cat "$OUT/SUMMARY.md"
  echo
  echo "WINDOW $overall  ($OUT/SUMMARY.md, $OUT/summary.json)"
  [ "$overall" = PASS ]
}

# ---------------------------------------------------------------------------------------------- main
command -v python3 >/dev/null || die "the window script needs host python3 (the recipe itself does not)"
preflight_window
prepare_clone
i=0
IFS=, read -r -a CANDS <<< "$SWEEP"
for cand in "${CANDS[@]}"; do
  [ -n "$cand" ] || continue
  i=$((i + 1))
  run_config "sweep$i" "${cand%%:*}" "${cand#*:}" --skip "$SWEEP_SKIP" --reps 5 --conc 8,32
done
if [ "$SKIP_FINAL" = 0 ]; then
  run_config final "$UTIL" "$ROWMAP"
fi
summary
