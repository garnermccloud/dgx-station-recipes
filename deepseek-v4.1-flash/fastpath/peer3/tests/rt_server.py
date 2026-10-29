"""RTX side of the round-trip benchmark: real cold experts (b12x prepared layout) served by the persistent kernel.

  python3 peer3/tests/rt_server.py --seg /dev/shm/w4rtx/peer3_rt --layer 10 --experts 4 [--idle 30] [--flush-mb 96]
Allocates < 200 MiB: the experts (18.8 MB each), the kernel's work buffers and an L2 flush buffer swept after each
call (so every call streams its weights from DRAM, as in serving). Exits when the control word is set or after
--idle seconds without a call, then prints the kernel's own per-call statistics.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "b12x-spec"))
import peer3_ext  # noqa: E402
import peer3_seg  # noqa: E402
import peer3_stats  # noqa: E402
import ref  # noqa: E402
import spec  # noqa: E402


def cold_experts(n: int) -> list[int]:
    return list(range(7, 7 + 13 * n, 13))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--also-layers", default="", help="more layer slots (comma list) serving the same experts")
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--idle", type=float, default=30.0)
    ap.add_argument("--flush-mb", type=int, default=0, help="L2 sweep after each call (0: rely on rotation)")
    ap.add_argument("--grid", type=int, default=0)
    ap.add_argument("--target", type=int, default=0, help="FC1/FC2 work items per call (0 = grid size)")
    ap.add_argument("--ready", default="/dev/shm/w4rtx/rt_ready")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    r = peer3_ext.load("rtx")
    Wc = ref.load_experts("/model", a.layer, cold_experts(a.experts))
    P = spec.logical_to_prepared(Wc["w13"], Wc["s13"], Wc["w2"], Wc["s2"], hidden_size=5120, intermediate_size=2304)
    W = {"w13": P.w13_rp.view(torch.uint8).reshape(-1).to(dev), "s13": P.w13_sfb.view(torch.uint8).reshape(-1).to(dev),
         "w2": P.w2_rp.view(torch.uint8).reshape(-1).to(dev), "s2": P.w2_sfb.view(torch.uint8).reshape(-1).to(dev)}
    layers = torch.zeros(r.MAX_LAYERS * r.LAYER_BYTES, dtype=torch.uint8)
    rec = struct.pack("<QQQQii", W["w13"].data_ptr(), W["s13"].data_ptr(), W["w2"].data_ptr(), W["s2"].data_ptr(),
                      a.experts, 1)
    for lay in [a.layer] + [int(x) for x in a.also_layers.split(",") if x]:
        layers[lay * r.LAYER_BYTES:(lay + 1) * r.LAYER_BYTES] = torch.frombuffer(bytearray(rec), dtype=torch.uint8)
    layers = layers.to(dev)
    work = torch.zeros(r.WORK_BYTES, dtype=torch.uint8, device=dev)
    xstage = torch.zeros(64 * 5120, dtype=torch.bfloat16, device=dev)
    part1 = torch.zeros(r.PART1_FLOATS, dtype=torch.float32, device=dev)
    part2 = torch.zeros(r.PART2_FLOATS, dtype=torch.float32, device=dev)
    inter = torch.zeros(r.MAX_PAIRS * 2304, dtype=torch.uint8, device=dev)
    inter_sf = torch.zeros(r.MAX_PAIRS * 72, dtype=torch.uint8, device=dev)
    flush = torch.ones(a.flush_mb << 20, dtype=torch.uint8, device=dev) if a.flush_mb else None
    torch.cuda.synchronize()
    print(f"rtx server: {a.experts} experts of layer {a.layer}, device memory "
          f"{torch.cuda.memory_allocated() / 2**20:.0f} MiB", flush=True)
    seg = peer3_seg.Segment(a.seg, r.SEG_BYTES, dev, create=True, zero_header=True)
    free, total = torch.cuda.mem_get_info()
    print(f"rtx server: device free {free / 2**20:.0f} MiB of {total / 2**20:.0f} before launch", flush=True)
    t0 = time.time()
    grid = r.launch(seg.dptr, layers, work, xstage, part1, part2, inter, inter_sf, int(a.idle * 1e9), int(2e9),
                    a.target, a.grid, flush)
    print(f"rtx server: persistent kernel on {grid} CTAs", flush=True)
    with open(a.ready, "w") as f:
        f.write("ready\n")
    r.wait_exit()
    fold = peer3_stats.StatsFold()  # the kernel's per-call records, as the sidecar's stats thread folds them
    fold.consume(peer3_stats.ring_view(seg.buf), idle=True)
    peer3_stats._print(fold.summary())
    wb = work.cpu().numpy().tobytes()
    calls, with_work, pairs, experts, busy, _, t_fc1, _ = struct.unpack_from("<8Q", wb, r.WORK_OFF_STATS)
    n = max(1, with_work)
    if not hasattr(r, "WORK_OFF_EV_MIN_SUM"):  # an older prebuilt (no event export)
        return
    nev = getattr(r, "NUM_EV_SLOTS", 16)
    ev = struct.unpack_from(f"<{nev}Q", wb, r.WORK_OFF_EV_SUM)
    evm = struct.unpack_from(f"<{nev}Q", wb, r.WORK_OFF_EV_MIN_SUM)
    names = ["x staged", "FC1 math", "FC1 epilogue", "FC2 got FC1", "FC2 math", "done", "first tile", "awake",
             "fin start", "fin fenced", "plan out (CTA 0)", "tiles issued", "x wait done", "x slice in smem",
             "plan built", "x wait entered", "CTA 0 published", "x host loads in", "FC2 tiles issued"]
    order = [16, 7, 14, 11, 18, 17, 0, 15, 12, 13, 6, 1, 2, 3, 4, 8, 9, 5] if nev > 16 else [7, 14, 11, 0, 15, 12, 13, 6, 1, 2, 3, 4, 8, 9, 5]
    print("rtx server: mean event times after the plan release, earliest / latest CTA (us):", flush=True)
    for i in order:
        print(f"    {names[i]:18s} {evm[i] / n / 1e3:6.1f} / {ev[i] / n / 1e3:6.1f}", flush=True)
    print(f"rtx server: exited after {time.time() - t0:.1f} s; {calls} calls ({with_work} with work), "
          f"{pairs} pairs, {experts} expert-calls; per call with work, from the plan's release (CTA 0): "
          f"own FC1 done {t_fc1 / n / 1e3:.1f} us, all CTAs done {busy / n / 1e3:.1f} us", flush=True)


if __name__ == "__main__":
    main()
