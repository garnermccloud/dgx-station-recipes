"""GB300 side of the real-routing exposure replay against the RTX server (rt_server.py; via rt_bench.sh with
CLIENT=rt_replay.py). For each S in the replay file (results/replay_calls.pt, from w4-boundary tests/make_replay.py:
real captured calls, distinct cold experts renumbered onto the server's resident ones), CUDA graphs of chunks of
calls, per call:  send -> a GPU sleep of W = a + b * D_hot (MegaMoE's window) -> finish -> a GPU sleep of the dense
chain (the RTX idles). The finish's wait is then the RTX time not hidden by the window: the exposure. Reports per S
the mean / p90 exposure per call (from GbStats and per-chunk deltas) and x40 per step; the server's own stats ring
(RTX busy, GB300 wait) is printed by rt_server at exit.
  python3 peer3/tests/rt_replay.py --seg SEG --window 26.5,2.94 [--dense-us 40]"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, HERE)
import peer3_ext  # noqa: E402
import peer3_seg  # noqa: E402
from rt_server import cold_experts  # noqa: E402

E_TOTAL = 384
CHUNK = 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--calls", default="/w/results/replay_calls.pt")
    ap.add_argument("--window", default="26.5,2.94", help="a,b: MegaMoE window W = a + b * D_hot (us)")
    ap.add_argument("--dense-us", type=float, default=40.0)
    ap.add_argument("--clock-mhz", type=float, default=2070.0, help="GPU clock for the sleep kernels")
    ap.add_argument("--reps", type=int, default=3, help="replays of each chunk graph")
    ap.add_argument("--ready", default="/dev/shm/w4rtx/rt_ready")
    a = ap.parse_args()
    wa, wb = (float(v) for v in a.window.split(","))
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    g = peer3_ext.load("gb300")
    g.preload()
    cold = cold_experts(a.experts)
    row_map = torch.empty(E_TOTAL, dtype=torch.int32)
    hot = 0
    for e in range(E_TOTAL):
        if e in cold:
            row_map[e] = -(cold.index(e) + 1)
        else:
            row_map[e] = hot
            hot += 1
    row_map_d = row_map.to(dev)
    hot_pool = [e for e in range(E_TOTAL) if e not in cold]
    replay = torch.load(a.calls)
    gen = torch.Generator().manual_seed(0)
    state = torch.tensor([1, 0], dtype=torch.int32, device=dev)
    stats = torch.zeros(g.GB_STATS_WORDS, dtype=torch.int64, device=dev)
    deadline = time.time() + 60
    while not os.path.exists(a.ready):
        if time.time() > deadline:
            raise SystemExit("rtx server never became ready")
        time.sleep(0.05)
    seg = peer3_seg.Segment(a.seg, g.SEG_BYTES, dev, create=False, zero_header=False)
    stream = torch.cuda.Stream()
    cyc = lambda us: max(1, int(us * a.clock_mhz))  # noqa: E731
    print(f"real-routing replay, window W = {wa} + {wb} D_hot us, dense chain {a.dense_us} us, {a.experts} resident "
          f"cold experts (call-local renumbering)", flush=True)
    for S in sorted(replay):
        calls = replay[S]
        per_call = []
        for c0 in range(0, len(calls), CHUNK):
            chunk = calls[c0:c0 + CHUNK]
            keep = []
            for c in chunk:
                ids = torch.empty(S, 6, dtype=torch.int64)
                for t, row in enumerate(c["slots"]):
                    used = {cold[j] for j in row if j >= 0}
                    others = [e for e in (hot_pool[int(i)] for i in torch.randperm(len(hot_pool), generator=gen)[:8])
                              if e not in used]
                    ids[t] = torch.tensor([cold[j] if j >= 0 else others.pop() for j in row])
                w = torch.rand(S, 6, generator=gen) * 0.4 + 0.05
                x = (torch.randn(S, 5120, generator=gen) * 0.8).to(torch.bfloat16)
                keep.append(dict(ids=ids.to(dev), w=w.to(dev), x=x.to(dev), hot=torch.empty(S, 6, dtype=torch.int64,
                                 device=dev), has=torch.empty(S, dtype=torch.int8, device=dev),
                                 y=torch.zeros(S, 5120, dtype=torch.bfloat16, device=dev), W=wa + wb * c["d_hot"]))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for k in keep:
                    g.send(k["ids"], k["w"], row_map_d, k["x"], k["hot"], k["has"], seg.dptr, state, a.layer, stats)
                    torch.cuda._sleep(cyc(k["W"]))
                    g.finish(k["y"], k["has"], seg.dptr, state, stats, 50_000_000, a.layer)
                    torch.cuda._sleep(cyc(a.dense_us))
            for _ in range(a.reps):
                before = stats.cpu().clone()
                graph.replay()
                stream.synchronize()
                after = stats.cpu()
                n = int(after[0] - before[0])
                per_call.append((int(after[2] - before[2]) / max(1, n) / 1e3, n))
            del graph, keep
        mean = sum(e * n for e, n in per_call) / sum(n for _, n in per_call)
        chunk_means = sorted(e for e, _ in per_call)
        print(f"S={S:2d}: {len(calls)} calls x {a.reps}: mean exposed wait {mean:5.2f} us/call ({mean * 40:6.1f} "
              f"us/step over 40 layers); chunk means p10 {chunk_means[len(chunk_means) // 10]:.2f} p90 "
              f"{chunk_means[9 * len(chunk_means) // 10]:.2f}", flush=True)
    st = stats.cpu()
    print(f"finish waits {int(st[0])}, timeouts {int(st[3])}, mean wait {int(st[2]) / max(1, int(st[0])) / 1e3:.2f} us",
          flush=True)
    seg.set_u64(g.OFF_CTRL, 1)


if __name__ == "__main__":
    main()
