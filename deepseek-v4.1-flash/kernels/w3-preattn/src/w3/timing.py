"""CUDA-graph replay timing with an L2 flush before every replay, plus a per-kernel profile of one replay."""
from __future__ import annotations

import statistics
from collections.abc import Callable

import torch

FLUSH_BYTES = 128 << 20  # > the GB300 115.5 MiB L2; read-only (process budget 1.2 GiB incl. ~850 MiB context)


class Flusher:
    def __init__(self, device: torch.device):
        self.buf = torch.ones(FLUSH_BYTES // 4, dtype=torch.int32, device=device)
        self.sink = torch.empty((), dtype=torch.int32, device=device)

    def __call__(self) -> None:
        torch.amax(self.buf, dim=0, out=self.sink)


def capture(fn: Callable[[], object], warmup: int = 3) -> tuple[torch.cuda.CUDAGraph, object]:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def time_graph(g: torch.cuda.CUDAGraph, flusher: Flusher | None, iters: int = 200) -> dict[str, float]:
    """Median / p10 / p90 microseconds of one replay (L2 flushed before each when ``flusher`` is given)."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(10):
        if flusher:
            flusher()
        g.replay()
    torch.cuda.synchronize()
    for i in range(iters):
        if flusher:
            flusher()
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()
    us = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))
    return {"median": statistics.median(us), "p10": us[iters // 10], "p90": us[iters * 9 // 10]}


def profile_graph(g: torch.cuda.CUDAGraph, flusher: Flusher | None, reps: int = 21) -> list[tuple[str, float, float]]:
    """Kernels of one replay: (name, median start offset us, median duration us) over ``reps`` replays."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            if flusher:
                flusher()
            torch.cuda._sleep(200_000)  # ~100 us spin: guarantees a gap between replays
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    evs = [e for e in evs if "amax" not in e.name and "reduce_kernel" not in e.name.lower() and "spin" not in e.name.lower()]
    evs.sort(key=lambda e: e.time_range.start)
    # split into replays by large gaps
    groups: list[list] = [[]]
    for e in evs:
        if groups[-1] and e.time_range.start - groups[-1][-1].time_range.end > 50:
            groups.append([])
        groups[-1].append(e)
    n = len(groups[-1])
    groups = [gr for gr in groups if len(gr) == n]
    rows = []
    for i in range(n):
        starts = sorted(gr[i].time_range.start - gr[0].time_range.start for gr in groups)
        durs = sorted(gr[i].time_range.end - gr[i].time_range.start for gr in groups)
        rows.append((groups[-1][i].name, starts[len(starts) // 2], durs[len(durs) // 2]))
    return rows


def span_us(rows: list[tuple[str, float, float]]) -> float:
    return max(s + d for _, s, d in rows) if rows else 0.0


def fmt_profile(rows: list[tuple[str, float, float]]) -> str:
    out = []
    for name, start, dur in rows:
        out.append(f"  +{start:6.1f} {dur:6.1f}  {name[:110]}")
    if rows:
        span = max(s + d for _, s, d in rows)
        out.append(f"  span {span:.1f} us, {len(rows)} kernels")
    return "\n".join(out)
