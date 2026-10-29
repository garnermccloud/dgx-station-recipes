"""Peer tier v3 call statistics: the RTX kernel's per-call records (the segment's stats ring) folded into per-layer
and per-batch histograms. Pure numpy over the mapped segment: no CUDA, safe in a sidecar thread or on a read-only
mapping of the live segment.

  python3 peer3_stats.py live [SEGMENT] [SECONDS]    fold a window of the live segment (read-only) and print it
  python3 peer3_stats.py show [JSON]                 print a dump written by the sidecar

Record (peer3_proto.cuh, kOffStatRing): u64 seq, u32 tag, u16 rows, u16 pairs, u16 distinct cold experts,
u16 distinct hot experts (0xFFFF = unknown), u32 RTX busy ns (0 for calls without cold work), u32 the GB300's
exposed wait (ns) for its previous call and u32 that call's tag (both 0 when the publisher does not forward it).
"""
from __future__ import annotations

import json
import mmap
import os
import sys
import threading
import time

import numpy as np

# Mirrors of peer3_proto.cuh (the sidecar asserts them against the extension).
OFF_STAT_RING = 1007616
STAT_RING = 32768
REC = np.dtype([("seq", "<u8"), ("tag", "<u4"), ("rows", "<u2"), ("pairs", "<u2"), ("n_cold", "<u2"),
                ("n_hot", "<u2"), ("busy_ns", "<u4"), ("prev_wait_ns", "<u4"), ("prev_tag", "<u4")])
assert REC.itemsize == 32

LAYERS, TMAX = 64, 64
COLD_BINS, HOT_BINS = 65, 257          # distinct experts per call (cold <= 64 tokens' picks, hot <= 256)
BUSY_US_BIN, BUSY_BINS = 2, 256        # RTX busy time, 2 us bins up to 512 us (last bin = overflow)
WAIT_US_BIN, WAIT_BINS = 1, 256        # GB300 exposed wait, 1 us bins up to 256 us (last bin = overflow)
LAG = 8                                # records this close to the newest are left for the next pass


def _wait_fields(hist: np.ndarray) -> dict:
    """GB300 exposed wait (finish polling the acknowledgement), from the forwarded per-call values."""
    if hist.sum() == 0:
        return {"gb_wait_us_mean": None, "gb_wait_us_p50": None, "gb_wait_us_p90": None, "gb_wait_over_5us": None}
    return {"gb_wait_us_mean": round(_mean(hist, WAIT_US_BIN) + WAIT_US_BIN / 2, 1),
            "gb_wait_us_p50": _pct(hist, 0.5, WAIT_US_BIN), "gb_wait_us_p90": _pct(hist, 0.9, WAIT_US_BIN),
            "gb_wait_over_5us": round(float(hist[5 // WAIT_US_BIN:].sum()) / float(hist.sum()), 3)}


def _pct(hist: np.ndarray, q: float, scale: float = 1.0) -> float | None:
    n = int(hist.sum())
    if n == 0:
        return None
    return float(np.searchsorted(np.cumsum(hist), q * n, side="left")) * scale


def _mean(hist: np.ndarray, scale: float = 1.0) -> float | None:
    n = int(hist.sum())
    if n == 0:
        return None
    return float((hist * np.arange(hist.size)).sum()) / n * scale


class StatsFold:
    """Cumulative histograms over the records consumed so far."""

    def __init__(self) -> None:
        self.calls = np.zeros((LAYERS, TMAX + 1), np.int64)
        self.work = np.zeros((LAYERS, TMAX + 1), np.int64)
        self.rows = np.zeros((LAYERS, TMAX + 1), np.int64)
        self.pairs = np.zeros((LAYERS, TMAX + 1), np.int64)
        self.cold = np.zeros((LAYERS, TMAX + 1, COLD_BINS), np.int64)
        self.hot = np.zeros((LAYERS, HOT_BINS), np.int64)
        self.hot_unknown = np.zeros(LAYERS, np.int64)
        self.busy = np.zeros((LAYERS, TMAX + 1, BUSY_BINS), np.int64)
        self.wait = np.zeros((LAYERS, TMAX + 1, WAIT_BINS), np.int64)  # by the waited call's layer and T
        self.next = 1
        self.lost = 0
        self.t0 = time.time()

    def fold(self, recs: np.ndarray) -> None:
        tag = recs["tag"].astype(np.int64)
        layer, T = (tag >> 6) & 63, (tag & 63) + 1
        np.add.at(self.calls, (layer, T), 1)
        has = recs["rows"] > 0
        np.add.at(self.work, (layer[has], T[has]), 1)
        np.add.at(self.rows, (layer, T), recs["rows"].astype(np.int64))
        np.add.at(self.pairs, (layer, T), recs["pairs"].astype(np.int64))
        np.add.at(self.cold, (layer, T, np.minimum(recs["n_cold"], COLD_BINS - 1)), 1)
        known = recs["n_hot"] != 0xFFFF
        np.add.at(self.hot, (layer[known], np.minimum(recs["n_hot"][known], HOT_BINS - 1)), 1)
        np.add.at(self.hot_unknown, layer[~known], 1)
        b = np.minimum(recs["busy_ns"][has] // (BUSY_US_BIN * 1000), BUSY_BINS - 1).astype(np.int64)
        np.add.at(self.busy, (layer[has], T[has], b), 1)
        fw = recs["prev_tag"] != 0
        ptag = recs["prev_tag"][fw].astype(np.int64)
        wb = np.minimum(recs["prev_wait_ns"][fw] // (WAIT_US_BIN * 1000), WAIT_BINS - 1).astype(np.int64)
        np.add.at(self.wait, ((ptag >> 6) & 63, (ptag & 63) + 1, wb), 1)

    def consume(self, ring: np.ndarray, idle: bool) -> int:
        """Fold the ring's records from self.next up to the newest (minus LAG unless idle). Returns the count."""
        seqs = ring["seq"].copy()
        newest = int(seqs.max())
        if newest < self.next:
            if newest + STAT_RING < self.next:  # the writer restarted (new sidecar): start over
                self.next = 1
            return 0
        limit = newest if idle else newest - LAG
        if limit < self.next:
            return 0
        if limit - self.next + 1 > STAT_RING:  # fell behind by a whole ring
            skip = limit - STAT_RING + 1 - self.next
            self.lost += skip
            self.next += skip
        sel = (seqs >= self.next) & (seqs <= limit)
        recs = ring[sel].copy()
        self.lost += (limit - self.next + 1) - recs.size
        self.next = limit + 1
        if recs.size:
            self.fold(recs)
        return int(recs.size)

    def summary(self) -> dict:
        calls = int(self.calls.sum())
        out = {"window_s": round(time.time() - self.t0, 1), "calls": calls, "calls_with_cold": int(self.work.sum()),
               "lost_records": int(self.lost), "layers": {}, "by_T": {}}
        for l in np.nonzero(self.calls.sum(1))[0]:
            c = int(self.calls[l].sum())
            cold = self.cold[l].sum(0)
            busy = self.busy[l].sum(0)
            out["layers"][str(int(l))] = {
                "calls": c, "cold_frac": round(int(self.work[l].sum()) / c, 3),
                "rows": round(int(self.rows[l].sum()) / c, 2), "pairs": round(int(self.pairs[l].sum()) / c, 2),
                "cold_exp_mean": round(_mean(cold), 2), "cold_exp_p90": _pct(cold, 0.9),
                "hot_exp_mean": None if self.hot[l].sum() == 0 else round(_mean(self.hot[l]), 1),
                "hot_unknown": int(self.hot_unknown[l]),
                "busy_us_mean": None if busy.sum() == 0 else round(_mean(busy, BUSY_US_BIN) + BUSY_US_BIN / 2, 1),
                "busy_us_p90": _pct(busy, 0.9, BUSY_US_BIN),
                **_wait_fields(self.wait[l].sum(0))}
        for T in np.nonzero(self.calls.sum(0))[0]:
            c = int(self.calls[:, T].sum())
            cold = self.cold[:, T].sum(0)
            busy = self.busy[:, T].sum(0)
            out["by_T"][str(int(T))] = {
                "calls": c, "cold_frac": round(int(self.work[:, T].sum()) / c, 3),
                "pairs": round(int(self.pairs[:, T].sum()) / c, 2),
                "cold_exp_mean": round(_mean(cold), 2), "cold_exp_p50": _pct(cold, 0.5),
                "cold_exp_p90": _pct(cold, 0.9),
                "cold_exp_hist": [int(v) for v in np.trim_zeros(cold, "b")],
                "busy_us_mean": None if busy.sum() == 0 else round(_mean(busy, BUSY_US_BIN) + BUSY_US_BIN / 2, 1),
                "busy_us_p50": _pct(busy, 0.5, BUSY_US_BIN), "busy_us_p90": _pct(busy, 0.9, BUSY_US_BIN),
                **_wait_fields(self.wait[:, T].sum(0))}
        return out


def ring_view(buf) -> np.ndarray:
    return np.frombuffer(buf, dtype=REC, count=STAT_RING, offset=OFF_STAT_RING)


class StatsThread(threading.Thread):
    """Sidecar daemon: folds the ring every second, dumps JSON every `dump_s` (reset: touch `<file>.reset`)."""

    def __init__(self, buf, path: str, dump_s: float = float(os.environ.get("PEER3_STATS_DUMP_S", "10"))) -> None:
        super().__init__(name="peer3-stats", daemon=True)
        self.ring, self.path, self.dump_s = ring_view(buf), path, dump_s
        self.fold = StatsFold()

    def run(self) -> None:
        last_dump, last_newest, warned = time.time(), -1, False
        while True:
            time.sleep(1.0)
            newest = int(self.ring["seq"].max())
            self.fold.consume(self.ring, idle=newest == last_newest)
            last_newest = newest
            if time.time() - last_dump < self.dump_s:
                continue
            last_dump = time.time()
            try:
                if os.path.exists(self.path + ".reset"):
                    os.remove(self.path + ".reset")
                    nxt = self.fold.next
                    self.fold = StatsFold()
                    self.fold.next = nxt
                with open(self.path + ".tmp", "w") as f:
                    json.dump(self.fold.summary(), f)
                os.replace(self.path + ".tmp", self.path)
            except OSError as e:
                if not warned:
                    print(f"peer3: stats dump to {self.path} failed ({e}); folding continues", flush=True)
                    warned = True


def _print(s: dict) -> None:
    print(f"window {s['window_s']} s: {s['calls']} v3 layer calls, {s['calls_with_cold']} with cold rows, "
          f"lost records {s['lost_records']}; GB300 wait = the finish's polling after MegaMoE (exposed)")
    nan = float("nan")
    print("   T    calls  cold%  pairs  coldexp mean/p50/p90  RTX busy us mean/p50/p90  GB300 wait us mean/p90 >5us")
    for T, v in sorted(s["by_T"].items(), key=lambda kv: int(kv[0])):
        print(f"  {T:>2} {v['calls']:8d} {100 * v['cold_frac']:5.1f}% {v['pairs']:6.2f}   "
              f"{v['cold_exp_mean']:5.2f} {v['cold_exp_p50']:4.0f} {v['cold_exp_p90']:4.0f}"
              f"           {v['busy_us_mean'] or nan:6.1f} {v['busy_us_p50'] or nan:5.0f} {v['busy_us_p90'] or nan:5.0f}"
              f"      {v['gb_wait_us_mean'] or nan:6.1f} {v['gb_wait_us_p90'] or nan:5.0f} "
              f"{100 * (v['gb_wait_over_5us'] or 0):4.0f}%")
    print(" layer  calls  cold%  coldexp mean/p90  hotexp mean  busy mean/p90  GB300 wait mean/p90")
    for l, v in sorted(s["layers"].items(), key=lambda kv: int(kv[0])):
        print(f"  {l:>4} {v['calls']:6d} {100 * v['cold_frac']:5.1f}%  {v['cold_exp_mean']:5.2f} "
              f"{v['cold_exp_p90']:4.0f}      {v['hot_exp_mean'] or nan:6.1f}   "
              f"{v['busy_us_mean'] or nan:6.1f} {v['busy_us_p90'] or nan:5.0f}   "
              f"{v['gb_wait_us_mean'] or nan:6.1f} {v['gb_wait_us_p90'] or nan:5.0f}")


def main() -> None:
    if sys.argv[1] == "show":
        _print(json.load(open(sys.argv[2] if len(sys.argv) > 2 else
                              os.path.join(os.environ.get("DS41_CACHE_DIR", os.path.expanduser("~/.ds41/state")), "sidecar-logs/peer3_stats.json"))))
        return
    path = sys.argv[2] if len(sys.argv) > 2 else "/dev/shm/vllm_peer_tier3"
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = mmap.mmap(fd, OFF_STAT_RING + STAT_RING * REC.itemsize, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
    ring = ring_view(buf)
    fold = StatsFold()
    fold.next = int(ring["seq"].max()) + 1  # only calls inside the window
    end = time.time() + secs
    while time.time() < end:
        time.sleep(min(1.0, max(0.0, end - time.time())))
        fold.consume(ring, idle=False)
    time.sleep(0.2)
    fold.consume(ring, idle=True)
    _print(fold.summary())


if __name__ == "__main__":
    main()
