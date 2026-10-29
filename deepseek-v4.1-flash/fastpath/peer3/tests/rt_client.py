"""GB300 side of the round-trip benchmark: CUDA-graph-replayed [send, finish] calls against the RTX server.

  python3 peer3/tests/rt_client.py --seg /dev/shm/w4rtx/peer3_rt --experts 16 [--shapes T:rows:pairs:distinct,...]
For each shape it captures R graphs whose cold picks rotate through all of the server's experts (so consecutive
calls stream different experts and the working set exceeds the RTX L2), times graph replays of [send, finish] with
CUDA events (GPU time from the send's start to the finish's end: publish, RTX compute, acknowledgement, cold-row
add), checks the first call of every shape against the fp64 recipe reference, then asks the server to exit.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, HERE)
import peer3_ext  # noqa: E402
import peer3_seg  # noqa: E402
import ref  # noqa: E402
from rt_server import cold_experts  # noqa: E402

E_TOTAL = 384
DEFAULT_SHAPES = ("6:0:0:0,6:1:1:1,6:1:2:2,6:2:2:1,6:2:2:2,6:2:4:4,6:3:3:3,6:3:4:4,6:4:4:4,6:4:6:6,6:6:6:6,"
                  "6:6:8:8,12:5:7:7,24:13:18:16,32:25:35:16")


def shape_ids(T: int, rows: int, pairs: int, subset: list[int], hot_pool: list[int], gen) -> torch.Tensor:
    """T x 6 routes: `pairs` cold picks over the first `rows` tokens, experts taken cyclically from `subset`."""
    picks = [[] for _ in range(T)]
    k = 0
    for i in range(pairs):
        t = i % rows
        while subset[k % len(subset)] in picks[t]:
            k += 1
        picks[t].append(subset[k % len(subset)])
        k += 1
    ids = torch.empty(T, 6, dtype=torch.int64)
    for t in range(T):
        mine = picks[t][:6]
        rest = [hot_pool[int(i)] for i in torch.randperm(len(hot_pool), generator=gen)[: 6 - len(mine)]]
        row = mine + rest
        perm = torch.randperm(6, generator=gen)
        ids[t] = torch.tensor([row[int(i)] for i in perm])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--rotate", type=int, default=8, help="graphs per shape, cycling through the experts")
    ap.add_argument("--gap-us", type=float, default=300.0)
    ap.add_argument("--ready", default="/dev/shm/w4rtx/rt_ready")
    ap.add_argument("--shapes", default=DEFAULT_SHAPES, help="T:rows:pairs:distinct,...")
    ap.add_argument("--check", type=int, default=1, help="check the first call of every shape against fp64")
    ap.add_argument("--layers", default="", help="alternate the calls over these layer slots (default: --layer)")
    ap.add_argument("--zipf", type=float, default=0.0,
                    help="> 0: each graph's cold experts are drawn with weight 1/(rank+1)^zipf (popular experts recur)")
    ap.add_argument("--hint-lead-us", type=float, default=0.0,
                    help="> 0: write the expert_pred hint word this long before each call's send (0: no hints)")
    ap.add_argument("--hint-k", type=int, default=3, help="hinted cold experts per call")
    ap.add_argument("--hint-miss", type=int, default=0, help="of which this many are not in the call (mispredicted)")
    ap.add_argument("--save", default="", help="save every call's output rows (eager + last replay) to this .pt")
    ap.add_argument("--ref", default="", help="compare every call's output rows byte for byte with a --save file")
    a = ap.parse_args()
    outs, ref_outs = {}, (torch.load(a.ref) if a.ref else None)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    g = peer3_ext.load("gb300")
    g.preload()
    cold = cold_experts(a.experts)
    Wc = ref.load_experts("/model", a.layer, cold) if a.check else None
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
    shapes = [tuple(int(v) for v in sh.split(":")) for sh in a.shapes.split(",")]
    lay_list = [int(x) for x in a.layers.split(",")] if a.layers else [a.layer]
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
    print(f"{a.experts} cold experts of layer {a.layer} on the RTX; GPU time of graph-replayed [send, finish], "
          f"{a.reps} calls per shape over {a.rotate} rotating expert subsets, {a.gap_us:.0f} us host gap", flush=True)
    print(f"{'T':>3} {'rows':>4} {'pairs':>5} {'experts':>7}  {'median':>7} {'p10':>7} {'p90':>7} {'max':>7}  check",
          flush=True)
    for T, rows, pairs, distinct in shapes:
        graphs, keep = [], []
        check = "-"
        key = f"{T}:{rows}:{pairs}:{distinct}"
        outs[key] = []
        for r in range(a.rotate):
            if distinct and a.zipf > 0:
                wts = torch.tensor([1.0 / (i + 1) ** a.zipf for i in range(a.experts)])
                subset = [cold[int(i)] for i in torch.multinomial(wts, distinct, replacement=False, generator=gen)]
            elif distinct:
                off = (r * distinct) % a.experts
                subset = [cold[(off + i) % a.experts] for i in range(distinct)]
            else:
                subset = []
            layer = lay_list[r % len(lay_list)]
            ids = shape_ids(T, rows, pairs, subset, hot_pool, gen)
            w = torch.rand(T, 6, generator=gen) * 0.4 + 0.05
            x = (torch.randn(T, 5120, generator=gen) * 0.8).to(torch.bfloat16)
            ids_d, w_d, x_d = ids.to(dev), w.to(dev), x.to(dev)
            hot_d = torch.empty_like(ids_d)
            has_d = torch.empty(T, dtype=torch.int8, device=dev)
            y = torch.zeros(T, 5120, dtype=torch.bfloat16, device=dev)
            with torch.cuda.stream(stream):
                g.send(ids_d, w_d, row_map_d, x_d, hot_d, has_d, seg.dptr, state, layer, stats)
                g.finish(y, has_d, seg.dptr, state, stats, 50_000_000, layer)
            stream.synchronize()
            outs[key].append(y.cpu().clone())
            if r == 0 and a.check and rows:
                rm = row_map[ids]
                cold_ids = torch.where(rm < 0, -rm - 1, -1)
                xq = seg.host[g.OFF_XQ:g.OFF_XQ + 64 * 5120].view(64, 5120)[:T].clone()
                xs = seg.host[g.OFF_XS:g.OFF_XS + 64 * 160].view(64, 160)[:T].clone()
                refo = ref.cold_rows(Wc, xq, xs, cold_ids, w)
                got = y.cpu().double()
                has = (cold_ids >= 0).any(dim=1)
                diff = (got[has] - refo[has]).abs()
                rel = float(diff.max() / refo[has].abs().max().clamp_min(1e-30))
                check = f"{int((diff == 0).sum())}/{diff.numel()} bit-equal, max|d| {float(diff.max()):.2e} ({rel:.1e} rel)"
            gs, gf = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(gs, stream=stream):
                g.send(ids_d, w_d, row_map_d, x_d, hot_d, has_d, seg.dptr, state, layer, stats)
            with torch.cuda.graph(gf, stream=stream):
                g.finish(y, has_d, seg.dptr, state, stats, 50_000_000, layer)
            hint_rows = [cold.index(e) for e in subset][: max(0, a.hint_k - a.hint_miss)]
            others = [i for i in range(a.experts) if cold[i] not in subset]
            hint_rows += others[(r * 5) % max(1, len(others)):][: a.hint_k - len(hint_rows)]
            graphs.append((gs, gf, layer, hint_rows))
            keep.append((ids_d, w_d, x_d, hot_d, has_d, y))
        times, t_send, t_fin = [], [], []
        ev0, ev1, ev2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        for i in range(a.reps + 20):
            gs, gf, layer_i, hint_rows = graphs[i % len(graphs)]
            if a.hint_lead_us > 0:  # the GB300's predictor at A1: one 64-bit word, then the dense chain's time
                hrow = sum((hint_rows[j] if j < len(hint_rows) else 0xFF) << (8 * j) for j in range(4))
                seg.set_u64(1024, hrow | (layer_i << 32) | (len(hint_rows) << 38) | ((i & 0xFFFF) << 48))
                t_hint = time.perf_counter() + a.hint_lead_us * 1e-6
                while time.perf_counter() < t_hint:
                    pass
            with torch.cuda.stream(stream):
                ev0.record(stream)
                gs.replay()
                ev1.record(stream)
                gf.replay()
                ev2.record(stream)
            ev2.synchronize()
            if i >= 20:
                times.append(ev0.elapsed_time(ev2) * 1e3)
                t_send.append(ev0.elapsed_time(ev1) * 1e3)
                t_fin.append(ev1.elapsed_time(ev2) * 1e3)
            t_end = time.perf_counter() + a.gap_us * 1e-6
            while time.perf_counter() < t_end:
                pass
        outs[key] += [k[-1].cpu().clone() for k in keep]
        vs = ""
        if ref_outs is not None:
            got, want = torch.stack(outs[key]).view(torch.uint8), torch.stack(ref_outs[key]).view(torch.uint8)
            vs = f"  vs ref: {int((got != want).sum())}/{got.numel()} bytes differ"
        times.sort()
        q = lambda f: times[min(len(times) - 1, int(f * len(times)))]
        print(f"{T:3d} {rows:4d} {pairs:5d} {distinct:7d}  {statistics.median(times):7.1f} {q(0.1):7.1f} {q(0.9):7.1f} "
              f"{times[-1]:7.1f}  send {statistics.median(t_send):5.1f} wait+add {statistics.median(t_fin):6.1f}  {check}{vs}",
              flush=True)
        del graphs, keep
    if a.save:
        torch.save(outs, a.save)
    st = stats.cpu()
    print(f"finish waits {int(st[0])}, timeouts {int(st[3])}, mean wait {int(st[2]) / max(1, int(st[0])) / 1e3:.1f} us",
          flush=True)
    seg.set_u64(g.OFF_CTRL, 1)


if __name__ == "__main__":
    main()
