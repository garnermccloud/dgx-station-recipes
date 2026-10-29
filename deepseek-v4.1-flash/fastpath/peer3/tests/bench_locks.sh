# Sourced by the peer3 benches: the shared GPU bench locks. The order is fixed for everyone: /tmp/gb300-gate.lock (held
# only while taking the GB300 locks, coordinator rule), gb300-bench (GB300 slot A), then gb300-bench2 (slot B), then rtx-bench. Each lock after the first is waited for at most 120 s; on a timeout all
# held locks are released and the sequence is retried after a random 5-25 s back-off, so a holder that takes the locks
# out of order cannot deadlock with us. LOCKS_HELD=1: the caller already holds all three (fds 9, 7, 8 inherited).
#   take_bench_locks timing   both GB300 slots, then the RTX (timing runs)
#   take_bench_locks slot     one GB300 slot (A if free, else wait for B), then the RTX (correctness runs)
#   take_bench_locks rtx      the RTX only
_bench_locks_try() {
  case $1 in
    timing)
      flock 6 || return 1
      flock 9 || { flock -u 6; return 1; }
      flock -w 120 7 || { flock -u 9; flock -u 6; return 1; }
      flock -u 6
      flock -w 120 8 || { flock -u 7; flock -u 9; return 1; } ;;
    slot)
      local held
      flock 6 || return 1
      while :; do
        if flock -n 9; then held=9; break; fi
        if flock -n 7; then held=7; break; fi
        sleep 1
      done
      flock -u 6
      flock -w 120 8 || { flock -u "$held"; return 1; } ;;
    rtx)
      flock 8 || return 1 ;;
    *)
      echo "take_bench_locks: unknown mode $1" >&2; exit 2 ;;
  esac
}
take_bench_locks() {
  [ "${LOCKS_HELD:-0}" = 1 ] && return 0
  exec 6>/tmp/gb300-gate.lock 9>/tmp/gb300-bench.lock 7>/tmp/gb300-bench2.lock 8>/tmp/rtx-bench.lock
  until _bench_locks_try "$1"; do
    echo "bench locks busy ($1): backing off" >&2
    sleep $((RANDOM % 20 + 5))
  done
}
