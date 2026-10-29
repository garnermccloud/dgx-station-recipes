"""GB300: duration of the peer3 send kernel and of production's route_send2, inside CUDA graphs.

A graph of N back-to-back launches is replayed; per-launch time = graph time / N (launch gaps included, as in a
serving graph). Routes with cold picks on every token (worst case: every row quantized and published).
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, HERE)
sys.path.insert(0, "/hook")
import peer3_ext  # noqa: E402
import peer3_seg  # noqa: E402
import peer_fused2  # noqa: E402
from test_send_gb300 import FakePeer2  # noqa: E402

E, K, H, N = 384, 6, 5120, 20


def capture(fn, warm=True):
    """A graph of N back-to-back launches of fn (torch's graph capture synchronizes the device first)."""
    s = torch.cuda.Stream()
    if warm:
        with torch.cuda.stream(s):
            fn()
        s.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(N):
            fn()
    return g, s


def replay_time(g, s, reps=200) -> float:
    ts = []
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        with torch.cuda.stream(s):
            e0.record(s)
            g.replay()
            e1.record(s)
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / N)
    return statistics.median(ts)


def time_graph(fn, reps=200) -> float:
    return replay_time(*capture(fn), reps)


def main() -> None:
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    g = peer3_ext.load("gb300")
    rowmap = json.load(open("/hook/rowmap-mix-v1-h255.json"))["layers"]["10"]
    rm = torch.empty(E, dtype=torch.int32)
    for r, e in enumerate(rowmap["hot"]):
        rm[e] = r
    for c, e in enumerate(rowmap["cold"]):
        rm[e] = -(c + 1)
    row_map = rm.to(dev)
    seg_path = "/dev/shm/w4rtx/peer3_sendbench"
    if os.path.exists(seg_path):
        os.remove(seg_path)
    seg = peer3_seg.Segment(seg_path, g.SEG_BYTES, dev, create=True, zero_header=True)
    state = torch.tensor([1, 0], dtype=torch.int32, device=dev)
    peer = FakePeer2(dev)
    # host-mapped buffers for route_send2 too (production writes to pinned host memory)
    host = torch.zeros(64 * H * 2 + 64 * 1024, dtype=torch.uint8).pin_memory()
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    hv = get_accelerator_view_from_cpu_tensor(host)
    peer.px = hv[: 64 * H]
    peer.pxs = hv[64 * H: 64 * H + 64 * (H // 32)]
    cold = rowmap["cold"]
    gen = torch.Generator().manual_seed(0)
    g.preload()
    cases = []
    for T in (1, 6, 12, 24, 32, 64):
        ids = torch.stack([torch.cat([torch.tensor([cold[t % len(cold)]]), torch.tensor(rowmap["hot"])[torch.randperm(255, generator=gen)[:K - 1]]])
                           for t in range(T)]).to(dev)
        w = torch.rand(T, K, device=dev)
        x = torch.randn(T, H, device=dev).to(torch.bfloat16)
        hot3 = torch.empty_like(ids)
        has3 = torch.empty(T, dtype=torch.int8, device=dev)
        y = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
        print(f"T={T}: timing send", flush=True)
        t3 = time_graph(lambda: g.send(ids, w, row_map, x, hot3, has3, seg.dptr, state, 10))
        print(f"T={T}: timing route_send2", flush=True)
        t2 = time_graph(lambda: peer_fused2.route_send2(peer, ids, w, row_map, None, x, 10, H, K, 64))
        cases.append((T, ids, w, x, hot3, has3, y, t3, t2))
    # A spinning kernel blocks any module load (lazy or JIT) and any device-wide synchronize (torch's graph capture
    # does one), so every graph is captured first and the responder starts just before the replays.
    stats = torch.zeros(g.GB_STATS_WORDS, dtype=torch.int64, device=dev)
    pairs = []
    for T, ids, w, x, hot3, has3, y, t3, t2 in cases:
        fn = (lambda ids=ids, w=w, x=x, hot3=hot3, has3=has3, y=y:
              (g.send(ids, w, row_map, x, hot3, has3, seg.dptr, state, 10),
               g.finish(y, has3, seg.dptr, state, stats, 50_000_000, 10)))
        pairs.append(capture(fn, warm=False))
    torch.cuda.synchronize()
    print("captured; starting the responder", flush=True)
    g.autoack_start(seg.dptr)
    for (T, ids, w, x, hot3, has3, y, t3, t2), (gp, sp) in zip(cases, pairs):
        tp = replay_time(gp, sp)
        print(f"T={T:2d} (every token cold): peer3 send {t3:5.2f} us per launch, route_send2 {t2:5.2f} us; "
              f"peer3 send + finish with an instant local acknowledgement {tp:5.2f} us", flush=True)
    seg.set_u64(g.OFF_CTRL, 1)
    g.autoack_wait()

if __name__ == "__main__":
    main()
