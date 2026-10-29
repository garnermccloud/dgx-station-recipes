"""GB300 side of the finish-placement benchmark: one CUDA graph per call = [send, prepare + MegaMoE, finish] against
the RTX server (rt_server.py):

  serial      finish after MegaMoE (v3.0)                                  MegaMoE on DeepGEMM's default SMs
  serial-pdl  the same finish launched as a programmatic dependent          (PEER3_PDL=1, v3.1's default)
  serial-sms  serial, MegaMoE on --sms SMs

(A parallel placement -- a 1-CTA kernel on a side branch waiting and copying the rows during a 136-SM MegaMoE, then a
device-local add -- measured 0-4 us slower than serial on 2026-10-20 and was removed: the finish's cost is its
launch, not its host reads.)

MegaMoE is the production DeepGEMM call (fp8 x fp4, fused fp8 shared expert, clamp 10, fast math) with synthetic
weights for --hot local experts (W4's megamoe_bw.py setup); each call's hot routes cover `d_hot` of them, rotating.
Cold routes go to the server's real experts. Reported: graph GPU time per call (one graph launch included, the same
in every mode), and whether every mode's output is bit-identical to `serial`'s.

  python3 peer3/tests/pf_client.py --seg SEG [--hot 16] [--sms 136] [--shapes T:d_hot:cold_rows:cold_distinct,...]
"""
from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, HERE)
import peer3_ext  # noqa: E402
import peer3_seg  # noqa: E402
from rt_server import cold_experts  # noqa: E402

H, I, K, E_TOTAL = 5120, 2304, 6, 384
FLUSH_BYTES = 120 << 20  # > GB300 L2: each call sees the weights cold, as in the server


def trim_context() -> torch.device:
    torch.empty(1, device="cuda")
    cuda = ctypes.CDLL("libcuda.so.1")
    for limit, value in ((0, 256), (1, 4096), (2, 1 << 20)):  # stack, printf FIFO, malloc heap
        cuda.cuCtxSetLimit(limit, ctypes.c_size_t(value))
    return torch.device("cuda", 0)


class Mega:
    """Production MegaMoE call on synthetic weights (shapes, dtypes and layouts of DSV4.1-Flash)."""

    def __init__(self, dg, E: int, dev) -> None:
        from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs

        self.dg, self.E, self.prep = dg, E, prepare_megamoe_inputs
        gen = torch.Generator(device="cuda").manual_seed(0)

        def sf(n, mn, k, val=2.0 ** -7):
            s = torch.full((n, mn, k // 32), val, dtype=torch.float32, device=dev)
            return dg.transform_sf_into_required_layout(s, mn, k, (1, 32), n)

        w13 = torch.randint(-128, 127, (E, 2 * I, H // 2), dtype=torch.int8, device=dev, generator=gen)
        w2 = torch.randint(-128, 127, (E, H, I // 2), dtype=torch.int8, device=dev, generator=gen)
        self.l1, self.l2 = dg.transform_weights_for_mega_moe((w13, sf(E, 2 * I, H)), (w2, sf(E, H, I)))
        del w13, w2
        gu = (torch.randn(2 * I, H, device=dev) * 0.02).to(torch.float8_e4m3fn)
        dn = (torch.randn(H, I, device=dev) * 0.02).to(torch.float8_e4m3fn)
        self.sl1, self.sl2 = dg.transform_weights_for_mega_moe((gu, sf(1, 2 * I, H, 1.0).squeeze(0)),
                                                               (dn, sf(1, H, I, 1.0).squeeze(0)))
        del gu, dn
        torch.cuda.empty_cache()
        self.symm = dg.get_symm_buffer_for_mega_moe(dist.group.WORLD, E, 64, K, H, I, num_shared_experts=1)

    def __call__(self, x, w, hot_ids, y, sms: int) -> None:
        T = x.shape[0]
        s = self.symm
        sbm = int(self.dg.get_block_m_for_mega_moe(1, self.E, s.num_max_tokens_per_rank, T, K, "fp8xfp4"))
        self.prep(x, w, hot_ids, s.x[:T], s.x_sf[:T], s.topk_idx[:T], s.topk_weights[:T],
                  shared_x_sf=s.shared_l1_acts_sf, shared_block_m=sbm)
        prev = self.dg.get_num_sms()
        if sms:
            self.dg.set_num_sms(sms)
        try:
            self.dg.fp8_fp4_mega_moe(y, self.l1, self.l2, s, shared_l1_weights=self.sl1, shared_l2_weights=self.sl2,
                                     activation_clamp=10.0, fast_math=True)
        finally:
            self.dg.set_num_sms(prev)


def make_ids(T, d_hot, cold_rows, cold_distinct, hot_rows_pool, cold_pool, rot, gen) -> torch.Tensor:
    """T x 6 global ids: tokens < cold_rows carry one cold pick (cold_distinct experts, cyclic); every other route
    is hot, spread over d_hot distinct hot experts (a rotating window of the pool)."""
    hot = [hot_rows_pool[(rot * d_hot + i) % len(hot_rows_pool)] for i in range(d_hot)]
    cold = [cold_pool[(rot * max(1, cold_distinct) + i) % len(cold_pool)] for i in range(max(1, cold_distinct))]
    ids = torch.empty(T, K, dtype=torch.int64)
    h = 0
    for t in range(T):
        row = []
        if t < cold_rows:
            row.append(cold[t % len(cold)])
        while len(row) < K:
            e = hot[h % len(hot)]
            h += 1
            if e not in row:
                row.append(e)
        perm = torch.randperm(K, generator=gen)
        ids[t] = torch.tensor([row[int(i)] for i in perm])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--experts", type=int, default=16, help="the server's cold experts")
    ap.add_argument("--hot", type=int, default=16, help="MegaMoE local experts (synthetic)")
    ap.add_argument("--sms", type=int, default=136)
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--rotate", type=int, default=4)
    ap.add_argument("--gap-us", type=float, default=300.0)
    ap.add_argument("--ready", default="/dev/shm/w4rtx/rt_ready")
    ap.add_argument("--warm", default="/dev/shm/w4rtx/pf_warm")
    ap.add_argument("--modes", default="serial,serial-pdl,serial-sms")
    ap.add_argument("--shapes", default="6:8:0:0,6:8:2:1,6:8:4:2,6:8:6:4,6:16:0:0,6:16:3:2,6:16:6:4,6:16:6:6")
    a = ap.parse_args()
    dev = trim_context()
    dist.init_process_group("gloo", init_method=f"file:///dev/shm/w4rtx/pg_{os.getpid()}", rank=0, world_size=1)
    from vllm.utils.deep_gemm import _import_deep_gemm

    dg = _import_deep_gemm()
    dg.set_pdl(True)
    g = peer3_ext.load("gb300")
    g.preload()
    mega = Mega(dg, a.hot, dev)
    print(f"MegaMoE: {a.hot} synthetic local experts, default {dg.get_num_sms()} SMs; device memory "
          f"{torch.cuda.memory_allocated() / 2**20:.0f} MiB", flush=True)
    cold = cold_experts(a.experts)
    hot_globals = [e for e in range(E_TOTAL) if e not in cold]
    row_map = torch.empty(E_TOTAL, dtype=torch.int32)
    for c, e in enumerate(cold):
        row_map[e] = -(c + 1)
    for r, e in enumerate(hot_globals):
        row_map[e] = r % a.hot  # MegaMoE has a.hot local experts; only the first a.hot hot globals are routed
    row_map_d = row_map.to(dev)
    hot_pool = hot_globals[:a.hot]
    state = torch.tensor([1, 0], dtype=torch.int32, device=dev)
    stats = torch.zeros(g.GB_STATS_WORDS, dtype=torch.int64, device=dev)
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.int32, device=dev)
    stream = torch.cuda.Stream(dev)
    shapes = [tuple(int(v) for v in sh.split(":")) for sh in a.shapes.split(",")]
    modes = a.modes.split(",")
    gen = torch.Generator().manual_seed(0)

    def call(mode, ids_d, w_d, x_d, hot_d, has_d, y):
        g.send(ids_d, w_d, row_map_d, x_d, hot_d, has_d, seg.dptr, state, a.layer, stats)
        mega(x_d, w_d, hot_d, y, a.sms if mode == "serial-sms" else 0)
        g.finish(y, has_d, seg.dptr, state, stats, 50_000_000, a.layer, mode == "serial-pdl")

    # Inputs first, then DeepGEMM's JIT for both SM counts (minutes at worst, longer than the server's idle limit),
    # then -- with the server up -- one eager call per graph (reference outputs) and the captures. Nothing compiles
    # while the timing runs.
    cases = []
    for T, d_hot, cold_rows, cold_distinct in shapes:
        sets = []
        for r in range(a.rotate):
            ids = make_ids(T, d_hot, cold_rows, cold_distinct, hot_pool, cold, r, gen)
            w = torch.rand(T, K, generator=gen) * 0.4 + 0.05
            x = (torch.randn(T, H, generator=gen) * 0.8).to(torch.bfloat16)
            sets.append((ids.to(dev), w.to(dev), x.to(dev), torch.empty(T, K, dtype=torch.int64, device=dev),
                         torch.empty(T, dtype=torch.int8, device=dev)))
        cases.append(((T, d_hot, cold_rows, cold_distinct), sets))
    ids_d, w_d, x_d, hot_d, _ = cases[0][1][0]
    hot_d.copy_(torch.where(row_map_d[ids_d] >= 0, row_map_d[ids_d], -1).long())
    for sms in (0, a.sms):
        mega(x_d, w_d, hot_d, torch.zeros(ids_d.shape[0], H, dtype=torch.bfloat16, device=dev), sms)
    torch.cuda.synchronize()
    with open(a.warm, "w") as f:  # the bench script starts the RTX server only now (its runs stay short)
        f.write("warm\n")
    deadline = time.time() + 60
    while not os.path.exists(a.ready):
        if time.time() > deadline:
            raise SystemExit("rtx server never became ready")
        time.sleep(0.05)
    seg = peer3_seg.Segment(a.seg, g.SEG_BYTES, dev, create=False, zero_header=False)
    graphs = {}
    outs = {}
    with torch.cuda.stream(stream):
        for (shape, sets) in cases:
            for mode in modes:
                for r, (ids_d, w_d, x_d, hot_d, has_d) in enumerate(sets):
                    y = torch.zeros(shape[0], H, dtype=torch.bfloat16, device=dev)
                    call(mode, ids_d, w_d, x_d, hot_d, has_d, y)  # eager: JIT + a reference output
                    stream.synchronize()
                    if r == 0:
                        outs[(shape, mode)] = y.clone()
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr, stream=stream):
                        call(mode, ids_d, w_d, x_d, hot_d, has_d, y)
                    graphs[(shape, mode, r)] = (gr, y)
    print(f"captured {len(graphs)} graphs; device memory {torch.cuda.memory_allocated() / 2**20:.0f} MiB", flush=True)
    print(f"{'T':>3} {'d_hot':>5} {'rows':>4} {'d_cold':>6}  " + "  ".join(f"{m:>18s}" for m in modes)
          + "   (median / p90 us per call)", flush=True)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for shape, sets in cases:
        cells = []
        for mode in modes:
            times = []
            for i in range(a.reps + 10):
                gr, y = graphs[(shape, mode, i % len(sets))]
                with torch.cuda.stream(stream):
                    flush.add_(1)
                    ev0.record(stream)
                    gr.replay()
                    ev1.record(stream)
                ev1.synchronize()
                if i >= 10:
                    times.append(ev0.elapsed_time(ev1) * 1e3)
                t_end = time.perf_counter() + a.gap_us * 1e-6
                while time.perf_counter() < t_end:
                    pass
            times.sort()
            cells.append(f"{statistics.median(times):7.1f} / {times[int(0.9 * len(times))]:6.1f}")
        same = all(torch.equal(outs[(shape, m)], outs[(shape, "serial")]) for m in modes if (shape, m) in outs)
        print(f"{shape[0]:3d} {shape[1]:5d} {shape[2]:4d} {shape[3]:6d}  " + "  ".join(f"{c:>18s}" for c in cells)
              + f"   outputs {'bit-identical' if same else 'DIFFER'} across modes", flush=True)
    st = stats.cpu()
    print(f"waits {int(st[0])}, timeouts {int(st[3])}", flush=True)
    seg.set_u64(g.OFF_CTRL, 1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
