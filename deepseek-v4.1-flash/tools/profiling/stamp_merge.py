# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garner McCloud
"""Unprofiled in-kernel stamps of one decode layer's kernels, merged on the GPU's %globaltimer (the counterpart of
tools/profiling/boundary_report.py without CUPTI). Every component that records per-CTA %globaltimer stamps into a
per-(layer, T) buffer captured into the FULL graphs and dumped on a trigger file plugs in here as a loader:

  W1 attention   W1_STAMPS=sparse,swa,draft     trigger /prof/w1_stamps.dump    -> w1_stamps_<unix>.pt
                 per CTA 8 globaltimer slots of its last task (words 64 + 8 cta + k)
  W8 woa         W8_TIMELINE=1                  trigger /prof/w8_timeline.dump  -> w8oproj_timeline_<unix>.pt
                 per CTA [entry, weights issued, accumulator ready, done]
  (add here: W4 seam / router / mix, W3 stage, W6 MegaMoE phase trace, as their owners expose the same buffers)

The two dumps' snapshot loops are independent, so records are paired per layer by time: a woa record belongs to the
attention record of the same layer whose window [start, end + 50 us] contains the woa's first CTA entry.
Per layer kind (sparse / swa / draft from W1's class) and over all pairs (medians, p90):
  attn dur                       first CTA stamp -> last CTA stamp
  woa lead                       attention start -> woa's first CTA entry (PDL residency during attention)
  woa issued                     attention end -> median CTA "weights issued" (negative: prologue done before)
  release -> first accumulator   attention end -> first CTA "accumulator ready"
  release -> last accumulator    attention end -> last CTA "accumulator ready"
  attn end -> woa done           attention end -> last CTA "done" (the boundary + op cost, unprofiled)
Usage (needs torch: run in the image, CPU only):
  python3 tools/profiling/stamp_merge.py --w1 /prof/w1_stamps_X.pt --w8 /prof/w8oproj_timeline_Y.pt
"""
from __future__ import annotations

import argparse
import re
import statistics

import torch


def med(v):
    return statistics.median(v) if v else float("nan")


def p90(v):
    return sorted(v)[int(0.9 * (len(v) - 1))] if v else float("nan")


def layer_of(prefix) -> int | None:
    m = re.search(r"layers\.(\d+)\.", str(prefix))
    return int(m.group(1)) if m else (int(prefix) if isinstance(prefix, int) else None)


def load_w1(path: str) -> dict:
    """{layer: [(class, T, start_ns, end_ns)]} over every snapshot."""
    d = torch.load(path, map_location="cpu")
    out: dict = {}
    for (prefix, T), v in d["calls"].items():
        L = layer_of(prefix)
        if L is None:
            continue
        st = v["stamps"]  # [snaps, words]
        cta = st[:, 64:64 + 152 * 8].reshape(st.shape[0], 152, 8)
        for s in range(cta.shape[0]):
            x = cta[s]
            nz = x[x > 0]
            if nz.numel() == 0:
                continue
            out.setdefault(L, []).append((v["class"], int(T), int(nz.min()), int(nz.max())))
    return out


def load_w8(path: str) -> dict:
    """{layer: [(T, entry_min, issued_med, acc_first, acc_last, done_max)]} over every snapshot."""
    d = torch.load(path, map_location="cpu")
    out: dict = {}
    for (L, T), tl in d["calls"].items():  # [snaps, 128 CTAs, 4]
        if not torch.is_tensor(tl):  # the older w8post component's dump ({"tl", "clk"}): not this format
            raise SystemExit(f"{path}: not a w8oproj timeline (W8_TIMELINE of fastpath_w8oproj.py)")
        for s in range(tl.shape[0]):
            x = tl[s]
            used = x[:, 0] > 0
            if not bool(used.any()):
                continue
            x = x[used]
            out.setdefault(int(L), []).append((int(T), int(x[:, 0].min()), int(x[:, 1].median()), int(x[:, 2].min()),
                                               int(x[:, 2].max()), int(x[:, 3].max())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", required=True)
    ap.add_argument("--w8", required=True)
    a = ap.parse_args()
    w1, w8 = load_w1(a.w1), load_w8(a.w8)
    print(f"W1 attention records: {sum(len(v) for v in w1.values())} over {len(w1)} layers; "
          f"W8 woa records: {sum(len(v) for v in w8.values())} over {len(w8)} layers")
    rows: dict = {}
    for L, wo in w8.items():
        for T, e0, iss, a0, a1, dn in wo:
            cand = [r for r in w1.get(L, []) if r[1] == T and r[2] <= e0 <= r[3] + 50_000]
            if not cand:
                continue
            cls_, _, s, e = min(cand, key=lambda r: e0 - r[2])
            r = rows.setdefault(cls_, {k: [] for k in ("attn dur", "woa lead", "woa issued", "release -> first acc",
                                                        "release -> last acc", "attn end -> woa done")})
            r["attn dur"].append((e - s) / 1e3)
            r["woa lead"].append((e0 - s) / 1e3)
            r["woa issued"].append((iss - e) / 1e3)
            r["release -> first acc"].append((a0 - e) / 1e3)
            r["release -> last acc"].append((a1 - e) / 1e3)
            r["attn end -> woa done"].append((dn - e) / 1e3)
    if not rows:
        print("no (attention, woa) pairs: the dumps must come from the same boot, triggered within a few seconds")
        return
    for cls_, r in sorted(rows.items()):
        n = len(r["attn dur"])
        print(f"== {cls_}: {n} paired layer records (us)")
        for k, v in r.items():
            print(f"   {k:24s} median {med(v):7.2f}   p90 {p90(v):7.2f}")


if __name__ == "__main__":
    main()
