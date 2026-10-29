# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garner McCloud
"""Summarize a w8post timeline dump (fastpath_w8.py W8_TIMELINE=1, trigger file): per (layer, width) the kernel span
(first entry .. last exit over the 152 CTAs), the effective SM clock, and per-stage end times relative to the earliest
entry (median over snapshots of the latest CTA to stamp each slot).
Slots (post_attn_block.cuh, per CTA, last task wins): 10 entry, 0 compute start (tag read), 1 woa task start,
2 woa done, 14 woared start (MoE start with the in-grid MoE), 7 woared done, 3 wob task start, 5 wob done,
9 SeamPost start, 11 SeamPost done, 12 SeamNorm done, 8 router done, 13 MixGemm done, 6 compute end, 15 exit.
Usage: python3 tools/tl_summary.py DUMP.pt [--layers 3,20]   (dump: {'calls': {(layer, S): {'tl', 'clk'}}})
"""
from __future__ import annotations

import statistics
import sys

import torch

ORDER = [(0, "compute"), (1, "woa start"), (2, "woa done"), (14, "woared start"), (7, "woared done"),
         (3, "wob start"), (5, "wob done"), (9, "SeamPost start"), (11, "SeamPost done"), (12, "SeamNorm done"),
         (8, "router done"), (13, "MixGemm done"), (6, "compute end"), (15, "exit")]


def main() -> None:
    d = torch.load(sys.argv[1])
    layers = None
    if "--layers" in sys.argv:
        layers = {int(v) for v in sys.argv[sys.argv.index("--layers") + 1].split(",")}
    for (li, S), b in sorted(d["calls"].items()):
        if layers is not None and li not in layers:
            continue
        if True:
            spans, mhz, stages = [], [], {s: [] for s, _ in ORDER}
            for k in range(b["tl"].shape[0]):
                t = b["tl"][k].double()
                c = b["clk"][k].double()
                ent, ext = t[:, 10], t[:, 15]
                if (ent <= 0).any() or (ext <= 0).any():
                    continue
                t0 = ent.min()
                if (ent.max() - t0) > 20000:  # entries spread > 20 us: the snapshot mixed two calls
                    continue
                spans.append((ext.max() - t0).item() / 1e3)
                dt = (ext - ent).clamp_min(1)
                mhz.append(((c[:, 1] - c[:, 0]) / dt * 1e3).median().item())
                for s, _ in ORDER:
                    v = t[:, s]
                    v = v[v >= t0]
                    if v.numel():
                        stages[s].append((v.max() - t0).item() / 1e3)
            if not spans:
                continue
            line = " ".join(f"{name} {statistics.median(stages[s]):.1f}" for s, name in ORDER if stages[s])
            print(f"layer {li:2d} S={S:2d} n={len(spans):2d}: span {statistics.median(spans):6.2f} us "
                  f"(min {min(spans):.2f} max {max(spans):.2f}), SM clock {statistics.median(mhz):.0f} MHz | {line}")


if __name__ == "__main__":
    main()
