"""Launch the persistent kernel alone (no experts) at several grid sizes and report memory and errors."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
import peer3_ext  # noqa: E402
import peer3_seg  # noqa: E402


def main() -> None:
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    r = peer3_ext.load("rtx")
    from cuda.bindings import runtime as rt

    if os.environ.get("STACK"):
        print("set stack", rt.cudaDeviceSetLimit(rt.cudaLimit.cudaLimitStackSize, int(os.environ["STACK"])), flush=True)
    print("stack limit", rt.cudaDeviceGetLimit(rt.cudaLimit.cudaLimitStackSize), flush=True)
    path = "/dev/shm/w4rtx/peer3_dbgl"
    if os.path.exists(path):
        os.remove(path)
    seg = peer3_seg.Segment(path, r.SEG_BYTES, dev, create=True, zero_header=True)
    layers = torch.zeros(r.MAX_LAYERS * r.LAYER_BYTES, dtype=torch.uint8, device=dev)
    work = torch.zeros(r.WORK_BYTES, dtype=torch.uint8, device=dev)
    xstage = torch.zeros(64 * 5120, dtype=torch.bfloat16, device=dev)
    part1 = torch.zeros(r.PART1_FLOATS, dtype=torch.float32, device=dev)
    part2 = torch.zeros(r.PART2_FLOATS, dtype=torch.float32, device=dev)
    inter = torch.zeros(r.MAX_PAIRS * 2304, dtype=torch.uint8, device=dev)
    inter_sf = torch.zeros(r.MAX_PAIRS * 72, dtype=torch.uint8, device=dev)
    torch.cuda.synchronize()
    print(f"free before module load {torch.cuda.mem_get_info()[0] / 2**20:.0f} MiB", flush=True)
    print("kernel attrs (local, regs, smem, maxthreads):", r.preload(), flush=True)
    print(f"free after module load {torch.cuda.mem_get_info()[0] / 2**20:.0f} MiB", flush=True)
    for grid in [int(g) for g in os.environ.get("GRIDS", "1,188").split(",")]:
        print(f"grid {grid}: free before {torch.cuda.mem_get_info()[0] / 2**20:.0f} MiB", flush=True)
        try:
            g = r.launch(seg.dptr, layers, work, xstage, part1, part2, inter, inter_sf, int(0.3e9), int(1e9), 0, grid)
            print(f"  launched {g}; free {torch.cuda.mem_get_info()[0] / 2**20:.0f} MiB", flush=True)
            r.wait_exit()
            print("  exited", flush=True)
        except Exception as exc:
            print("  failed:", exc, flush=True)


if __name__ == "__main__":
    main()
