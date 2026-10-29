"""Launch the persistent kernel, publish one no-cold call, then dump its work-state words and the segment words."""
from __future__ import annotations

import os
import struct
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
import peer3_ext  # noqa: E402
import peer3_seg  # noqa: E402


def dump(tag, work, seg, g, r):
    w = work.cpu().numpy().tobytes()
    names = ["epoch", "tag", "exit_flag", "tok_arrive", "plan_epoch", "bar_count", "bar_gen", "done_count", "n_exp",
             "n_pairs", "ks", "last_tag"]
    vals = struct.unpack_from("<12I", w, 0)
    print(tag, dict(zip(names, vals)), "seg done", seg.u64(g.OFF_DONE), "word0", hex(seg.u64(g.OFF_REC)),
          "ctrl", seg.u64(g.OFF_CTRL), flush=True)


def main() -> None:
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    g = peer3_ext.load("gb300")
    r = peer3_ext.load("rtx")
    path = "/dev/shm/w4rtx/peer3_dbg"
    if os.path.exists(path):
        os.remove(path)
    seg = peer3_seg.Segment(path, g.SEG_BYTES, dev, create=True, zero_header=True)
    layers = torch.zeros(r.MAX_LAYERS * r.LAYER_BYTES, dtype=torch.uint8, device=dev)
    work = torch.zeros(r.WORK_BYTES, dtype=torch.uint8, device=dev)
    xstage = torch.zeros(64 * 5120, dtype=torch.bfloat16, device=dev)
    part1 = torch.zeros(r.PART1_FLOATS, dtype=torch.float32, device=dev)
    part2 = torch.zeros(r.PART2_FLOATS, dtype=torch.float32, device=dev)
    inter = torch.zeros(r.MAX_PAIRS * 2304, dtype=torch.uint8, device=dev)
    inter_sf = torch.zeros(r.MAX_PAIRS * 72, dtype=torch.uint8, device=dev)
    row_map = torch.arange(384, dtype=torch.int32, device=dev)
    state = torch.tensor([1, 0], dtype=torch.int32, device=dev)
    ids = torch.randint(0, 384, (1, 6), device=dev)
    w = torch.rand(1, 6, device=dev)
    x = torch.randn(1, 5120, device=dev).to(torch.bfloat16)
    hot = torch.empty_like(ids)
    has = torch.empty(1, dtype=torch.int8, device=dev)
    g.preload()
    torch.cuda.synchronize()
    side, peek = torch.cuda.Stream(), torch.cuda.Stream()
    if True:
        grid = r.launch(seg.dptr, layers, work, xstage, part1, part2, inter, inter_sf, int(os.environ.get("IDLE", "20000000000")), int(2e9), 0,
                        int(os.environ.get("GRID", "8")))
    print("grid", grid, flush=True)
    time.sleep(0.5)
    with torch.cuda.stream(peek):
        dump("before", work, seg, g, r)
    with torch.cuda.stream(side):
        g.send(ids, w, row_map, x, hot, has, seg.dptr, state, 3)
        print("send launched", flush=True)
        side.synchronize()
    print("sent", flush=True)
    for i in range(5):
        time.sleep(0.3)
        with torch.cuda.stream(peek):
            dump(f"t+{i}", work, seg, g, r)
    seg.set_u64(g.OFF_CTRL, 1)
    t0 = time.time()
    while r.running() and time.time() - t0 < 10:
        time.sleep(0.1)
    print("serve done:", not r.running(), flush=True)
    with torch.cuda.stream(peek):
        dump("after", work, seg, g, r)


if __name__ == "__main__":
    main()
