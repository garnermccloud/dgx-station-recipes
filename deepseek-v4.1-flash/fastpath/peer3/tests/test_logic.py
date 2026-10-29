"""Single-GPU end-to-end check of peer tier v3: GB300-side send/finish kernels and the persistent RTX kernel in one
process (the persistent kernel on part of the SMs), real checkpoint experts, against the fp64 reference recipe.

  python3 peer3/tests/test_logic.py [--layer 10] [--experts 4] [--calls 40] [--grid 20] [--fmt ckpt]
Exercises: T = 1..8 (and 64), calls with no cold pick, repeated tags, determinism (same call twice -> same bits),
and the kernel's per-call statistics records (rows, pairs, distinct cold / hot experts) against the host's count.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))
import peer3_ext  # noqa: E402
import peer3_seg  # noqa: E402
import peer3_stats  # noqa: E402
import ref  # noqa: E402

E_TOTAL = 384


def pack_layer(w: dict[str, torch.Tensor], n_cold: int, fmt: int) -> bytes:
    import struct

    return struct.pack("<QQQQii", w["w13"].data_ptr(), w["s13"].data_ptr(), w["w2"].data_ptr(), w["s2"].data_ptr(),
                       n_cold, fmt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fmt", choices=("rp",), default="rp", help="weight layout handed to the RTX kernel")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    g = peer3_ext.load("gb300")
    r = peer3_ext.load("rtx")
    print(f"built: seg {g.SEG_BYTES} B, work {r.WORK_BYTES} B, smem {r.SMEM_BYTES} B", flush=True)

    cold_global = list(range(7, 7 + 13 * a.experts, 13))  # global expert ids that are cold in this test
    Wcpu = ref.load_experts("/model", a.layer, cold_global)
    if a.fmt == "rp":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "b12x-spec"))
        import spec

        P = spec.logical_to_prepared(Wcpu["w13"], Wcpu["s13"], Wcpu["w2"], Wcpu["s2"], hidden_size=5120,
                                     intermediate_size=2304)
        W = {"w13": P.w13_rp.view(torch.uint8).reshape(-1).to(dev), "s13": P.w13_sfb.view(torch.uint8).reshape(-1).to(dev),
             "w2": P.w2_rp.view(torch.uint8).reshape(-1).to(dev), "s2": P.w2_sfb.view(torch.uint8).reshape(-1).to(dev)}
        n_cold = Wcpu["w13"].shape[0]
    else:
        W = {k: v.to(dev) for k, v in Wcpu.items()}
        n_cold = Wcpu["w13"].shape[0]
    row_map = torch.empty(E_TOTAL, dtype=torch.int32)
    hot = 0
    for e in range(E_TOTAL):
        if e in cold_global:
            row_map[e] = -(cold_global.index(e) + 1)
        else:
            row_map[e] = hot
            hot += 1
    row_map = row_map.to(dev)

    seg_path = "/dev/shm/w4rtx/peer3_logic"
    if os.path.exists(seg_path):
        os.remove(seg_path)
    seg = peer3_seg.Segment(seg_path, g.SEG_BYTES, dev, create=True, zero_header=True)
    state = torch.tensor([1, 0], dtype=torch.int32, device=dev)
    stats = torch.zeros(g.GB_STATS_WORDS, dtype=torch.int64, device=dev)
    layers = torch.zeros(r.MAX_LAYERS * r.LAYER_BYTES, dtype=torch.uint8)
    layers[a.layer * r.LAYER_BYTES:(a.layer + 1) * r.LAYER_BYTES] = torch.frombuffer(bytearray(pack_layer(W, n_cold, 1 if a.fmt == "rp" else 0)),
                                                                                       dtype=torch.uint8)
    layers = layers.to(dev)
    work = torch.zeros(r.WORK_BYTES, dtype=torch.uint8, device=dev)
    xstage = torch.zeros(64 * 5120, dtype=torch.bfloat16, device=dev)
    part1 = torch.zeros(r.PART1_FLOATS, dtype=torch.float32, device=dev)
    part2 = torch.zeros(r.PART2_FLOATS, dtype=torch.float32, device=dev)
    inter = torch.zeros(r.MAX_PAIRS * 2304, dtype=torch.uint8, device=dev)
    inter_sf = torch.zeros(r.MAX_PAIRS * 72, dtype=torch.uint8, device=dev)
    print(f"device memory allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB", flush=True)

    # Everything the loop touches on the device is allocated and every kernel loaded before the persistent kernel
    # starts: a lazy module load or cudaMalloc behind a spinning kernel stalls the device (single-GPU test only).
    MAXT = 64
    ids_d = torch.zeros(MAXT, 6, dtype=torch.int64, device=dev)
    w_d = torch.zeros(MAXT, 6, dtype=torch.float32, device=dev)
    x_d = torch.zeros(MAXT, 5120, dtype=torch.bfloat16, device=dev)
    hot_d = torch.zeros_like(ids_d)
    has_d = torch.zeros(MAXT, dtype=torch.int8, device=dev)
    y_d = torch.zeros(MAXT, 5120, dtype=torch.bfloat16, device=dev)
    y2_d = torch.zeros_like(y_d)
    ids_h = torch.zeros(MAXT, 6, dtype=torch.int64).pin_memory()
    w_h = torch.zeros(MAXT, 6, dtype=torch.float32).pin_memory()
    x_h = torch.zeros(MAXT, 5120, dtype=torch.bfloat16).pin_memory()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for t in (ids_d, w_d, x_d):
            t.copy_(t.clone().cpu().pin_memory(), non_blocking=True)
        y_d.zero_(); y2_d.zero_()
        _ = torch.equal(y_d, y2_d)
        _ = y_d[:1].cpu(); _ = hot_d[:1].cpu(); _ = has_d[:1].cpu()
    side.synchronize()
    g.preload()
    torch.cuda.synchronize()
    grid = r.launch(seg.dptr, layers, work, xstage, part1, part2, inter, inter_sf, int(120e9), int(3e9), 0, a.grid)
    print(f"persistent kernel launched on {grid} CTAs", flush=True)

    gen = torch.Generator().manual_seed(a.seed)
    worst_rel = 0.0
    worst_abs = 0.0
    n_exact = n_elems = 0
    sizes = [1, 2, 3, 4, 5, 6, 8, 1, 6, 6, 64] + [int(torch.randint(1, 9, (1,), generator=gen)) for _ in range(a.calls)]
    pool_hot = [e for e in range(E_TOTAL) if e not in cold_global]
    expected = []  # per published call: (tag, rows, pairs, distinct cold, distinct hot)
    cold_sets = []  # per call with work: its distinct cold-local experts
    for call, T in enumerate(sizes):
        ids = torch.empty(T, 6, dtype=torch.int64)
        for t in range(T):
            n_cold = int(torch.randint(0, 3, (1,), generator=gen)) if call % 5 else 0
            chosen = [cold_global[int(i)] for i in torch.randperm(len(cold_global), generator=gen)[:n_cold]]
            rest = [pool_hot[int(i)] for i in torch.randperm(len(pool_hot), generator=gen)[: 6 - len(chosen)]]
            row = chosen + rest
            perm = torch.randperm(6, generator=gen)
            ids[t] = torch.tensor([row[int(i)] for i in perm])
        w = torch.rand(T, 6, generator=gen, dtype=torch.float32) * 0.4 + 0.05
        x = (torch.randn(T, 5120, generator=gen) * 0.8).to(torch.bfloat16)
        ids_h[:T].copy_(ids); w_h[:T].copy_(w); x_h[:T].copy_(x)
        outs = []
        for y in (y_d, y2_d):  # the same call twice: determinism
            with torch.cuda.stream(side):
                ids_d[:T].copy_(ids_h[:T], non_blocking=True)
                w_d[:T].copy_(w_h[:T], non_blocking=True)
                x_d[:T].copy_(x_h[:T], non_blocking=True)
                y[:T].zero_()
                tag = (int(state[0].item()) << 12) | (a.layer << 6) | (T - 1)
                t0 = time.perf_counter()
                g.send(ids_d[:T], w_d[:T], row_map, x_d[:T], hot_d[:T], has_d[:T], seg.dptr, state, a.layer, stats)
                g.finish(y[:T], has_d[:T], seg.dptr, state, stats, 200_000_000, a.layer)
                side.synchronize()
                dt = time.perf_counter() - t0
                outs.append(y[:T].cpu())
                rmc = row_map.cpu()[ids]
                expected.append((tag, int((rmc < 0).any(dim=1).sum()), int((rmc < 0).sum()),
                                 len(set(ids[rmc < 0].tolist())), len(set(ids[rmc >= 0].tolist()))))
                if (rmc < 0).any():
                    cold_sets.append(set((-rmc[rmc < 0] - 1).tolist()))
        # Reference on the exact MXFP8 rows the send published.
        xq = seg.host[g.OFF_XQ:g.OFF_XQ + 64 * 5120].view(64, 5120)[:T].clone()
        xs = seg.host[g.OFF_XS:g.OFF_XS + 64 * 160].view(64, 160)[:T].clone()
        rm = row_map.cpu()[ids]
        cold_ids = torch.where(rm < 0, -rm - 1, -1)
        exp_hot = torch.where(rm >= 0, rm, -1)
        with torch.cuda.stream(side):
            hot_c, has_c = hot_d[:T].cpu(), has_d[:T].cpu()
        assert torch.equal(hot_c, exp_hot), "hot ids differ"
        has_ref = (cold_ids >= 0).any(dim=1)
        assert torch.equal(has_c.bool(), has_ref), "has differs"
        refo = ref.cold_rows(Wcpu, xq, xs, cold_ids, w)
        got = outs[0].double()
        mask = has_ref
        if mask.any():
            d = (got[mask] - refo[mask]).abs()
            scale = refo[mask].abs().max().clamp_min(1e-30)
            worst_abs = max(worst_abs, float(d.max()))
            worst_rel = max(worst_rel, float(d.max() / scale))
            n_exact += int((d == 0).sum())
            n_elems += d.numel()
        assert torch.all(got[~mask] == 0), "token without cold picks got a nonzero row"
        assert torch.equal(outs[0], outs[1]), "not deterministic"
        print(f"call {call:3d} T={T:2d} cold pairs={int((cold_ids >= 0).sum()):3d} host {dt * 1e6:8.1f} us "
              f"max|d|={float((got - refo).abs().max()):.3e}", flush=True)
    seg.set_u64(g.OFF_CTRL, 1)
    r.wait_exit()
    ring = peer3_stats.ring_view(seg.buf)
    recs = ring[ring["seq"] > 0]
    recs = recs[recs["seq"].argsort()]
    assert recs.size == len(expected) and list(recs["seq"]) == list(range(1, recs.size + 1)), "stats ring: missing records"
    for rec, (tag, rows, pairs, n_cold, n_hot) in zip(recs, expected):
        got = (int(rec["tag"]), int(rec["rows"]), int(rec["pairs"]), int(rec["n_cold"]), int(rec["n_hot"]))
        assert got == (tag, rows, pairs, n_cold, n_hot), f"stats record {got} != expected {(tag, rows, pairs, n_cold, n_hot)}"
        assert (rec["busy_ns"] > 0) == (rows > 0), "stats record: busy time without work (or none with work)"
    prev_tags = [0] + [e[0] for e in expected[:-1]]  # each send forwards the GB300's wait of the call before it
    assert list(recs["prev_tag"]) == prev_tags, "stats records: forwarded GB300 waits carry the wrong tags"
    if os.environ.get("PEER3_PREFETCH_K", "0") != "0":  # the kernel's learned popularity and top list
        wb = work.cpu().numpy().tobytes()
        n_cold = len(cold_global)
        pop_dev = np.frombuffer(wb, dtype=np.uint32, count=n_cold, offset=r.WORK_OFF_POP + a.layer * 129 * 4)
        pop_host = np.zeros(n_cold, dtype=np.uint32)
        for cs in cold_sets:
            for c in cs:
                pop_host[c] += 1
        assert np.array_equal(pop_dev, pop_host), f"popularity {pop_dev} != {pop_host}"
        top_n = int(np.frombuffer(wb, dtype=np.uint32, count=1, offset=r.WORK_OFF_TOP_N + a.layer * 4)[0])
        top = list(np.frombuffer(wb, dtype=np.uint16, count=top_n, offset=r.WORK_OFF_TOP + a.layer * r.PREFETCH_MAX * 2))
        want = [int(e) for e in sorted(np.nonzero(pop_host)[0], key=lambda e: (-int(pop_host[e]), int(e)))][:r.PREFETCH_MAX]
        assert top == want, f"top list {top} != {want}"
        print(f"learned popularity matches; top {top}", flush=True)
    st = stats.cpu()
    print(f"OK: {len(sizes)} calls; vs fp64 recipe: max abs {worst_abs:.3e}, max abs / max|ref| {worst_rel:.3e}, "
          f"{n_exact}/{n_elems} elements bit-equal; GB300-side waits {int(st[0])}, timeouts {int(st[3])}; "
          f"{recs.size} stats records match")


if __name__ == "__main__":
    main()
