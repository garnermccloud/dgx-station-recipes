"""RTX PRO 6000 sidecar for peer tiers v2 + v3: cold experts in VRAM, prepared once for b12x (w4a8_mx).

v2 (any T, the GB300 hook's peer_tier2 protocol on /dev/shm/vllm_peer_tier2): b12x graph replays per (layer, row
bucket).
v3 (T <= 64, fastpath `peer3` on /dev/shm/vllm_peer_tier3): one persistent kernel (peer3/csrc) serves every call from
device-side polling, reading the same b12x-prepared weights in place.

The persistent kernel owns every SM while it runs, so the host loop only polls host memory then. A v2 request
(prefill) stops it (control word), runs b12x, and the kernel is relaunched as soon as a v3 call is pending. It also
parks itself after PEER3_IDLE_S (default 2) seconds without a v3 call, so an idle sidecar does not spin the RTX. The v3 module is preloaded before the first launch:
CUDA's lazy loading would otherwise stall any first launch behind the spinning kernel.

  CUDA_VISIBLE_DEVICES=<6000> python peer_server3.py --rowmap rowmap.json --model /model
Env: PEER3_DIR (peer3/ checkout, default ../ next to this file), PEER3_PREBUILT or PEER3_BUILD_DIR,
PEER3_IDLE_S (seconds before the persistent kernel parks; 0 = never), PEER3_STATS_DIR (sidecar stats
directory; `peer3/py/peer3_stats.py show` prints the JSON). Test-only: PEER3_ONLY, PEER3_TEST_BUCKETS.
"""
import argparse
import gc
import json
import os
import struct
import sys
import time
from dataclasses import replace

import numpy as np
import torch
import triton
import triton.language as tl

HOOK_DIR = os.environ.get("PEER3_HOOK_DIR", "/opt/ds41/hook")
B12X_DIR = os.environ.get("PEER3_B12X_DIR", "/opt/ds41/b12x")
if not os.path.isdir(HOOK_DIR):
    raise RuntimeError(f"sidecar hook dir missing: {HOOK_DIR}")
sys.path.insert(0, HOOK_DIR)
import peer_tier2 as pt  # noqa: E402

import b12x.moe.fused_moe as fm  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

PEER3_DIR = os.environ.get("PEER3_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(PEER3_DIR, "py"))
import peer3_ext  # noqa: E402
import peer3_seg  # noqa: E402
import peer3_stats  # noqa: E402

H, I, K, LIMIT = pt.HIDDEN, 2304, pt.TOPK, 10.0
BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
if os.environ.get("PEER3_TEST_BUCKETS"):  # tests only: fewer b12x buckets (and a smaller largest batch)
    BUCKETS = [int(b) for b in os.environ["PEER3_TEST_BUCKETS"].split(",")]
PEER3_SHM = "/dev/shm/vllm_peer_tier3"
STATS_DIR = os.environ.get("PEER3_STATS_DIR", "/var/lib/ds41/sidecar")
if not os.path.isdir(STATS_DIR):
    raise RuntimeError(f"sidecar stats dir missing: {STATS_DIR}")
STATS_FILE3 = os.path.join(STATS_DIR, "peer3_stats.json")
# The persistent kernel parks itself after PEER3_IDLE_S without a v3 call (0 = never) and is relaunched when the next
# call is pending (about 0.1 ms once): an idle kernel would otherwise spin the whole RTX (~180 W) and time-slice any
# other process on it.
IDLE_NS = int(float(os.environ.get("PEER3_IDLE_S", "2")) * 1e9)
WAIT_NS = 30_000_000_000  # in-call wait bound before the peer3 kernel traps


@triton.jit
def _dequant(X, XS, A, H: tl.constexpr, BLOCK: tl.constexpr):
    """MXFP8 row (E4M3 + E8M0 per 32) -> BF16."""
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + t * H + cols).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    e = tl.load(XS + t * (H // 32) + cols // 32).to(tl.int32) - 127
    tl.store(A + t * H + cols, (x * tl.exp2(e.to(tl.float32))).to(tl.bfloat16))


def tensor_reader(model):
    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    headers = {}

    def tensor(name):
        file = os.path.join(model, index[name])
        if file not in headers:
            with open(file, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                headers[file] = (8 + size, json.loads(f.read(size)))
        base, header = headers[file]
        meta = header[name]
        start, end = meta["data_offsets"]
        with open(file, "rb") as f:
            f.seek(base + start)
            data = f.read(end - start)
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(meta["shape"])

    return tensor


class Peer3Server:
    """The v3 persistent kernel: buffers, layer table, launch / stop."""

    def __init__(self, dev, keep: list[dict], n_cold: list[int]):
        self.r = peer3_ext.load("rtx")
        print(f"peer3: kernel module loaded {self.r.preload()}", flush=True)
        self.seg = peer3_seg.Segment(PEER3_SHM, self.r.SEG_BYTES, dev, create=True, zero_header=True)
        self.seg.set_u64(self.r.OFF_DONE, 0)
        assert (peer3_stats.OFF_STAT_RING, peer3_stats.STAT_RING) == (self.r.OFF_STAT_RING, self.r.STAT_RING)
        self.seg.host[self.r.OFF_STAT_RING:self.r.SEG_BYTES].zero_()
        # The GB300 side refuses a segment whose version differs from its own build (both halves ship together).
        self.seg.set_u64(self.r.OFF_VERSION, self.r.PROTO_VERSION)
        # d1: this RTX kernel sets the host ready word after each done tag (W4 COPY_AHEAD); advertise it
        self.seg.set_u64(self.r.OFF_READY, 0)
        self.seg.set_u64(self.r.OFF_CAP, self.r.CAP_READY)
        self.stats = peer3_stats.StatsThread(self.seg.buf, STATS_FILE3)
        table = torch.zeros(self.r.MAX_LAYERS * self.r.LAYER_BYTES, dtype=torch.uint8)
        for layer, W in enumerate(keep):
            rec = struct.pack("<QQQQii", W["w13"].data_ptr(), W["s13"].data_ptr(), W["w2"].data_ptr(),
                              W["s2"].data_ptr(), n_cold[layer], 1)
            table[layer * self.r.LAYER_BYTES:(layer + 1) * self.r.LAYER_BYTES] = torch.frombuffer(
                bytearray(rec), dtype=torch.uint8)
        self.layers = table.to(dev)
        self.work = torch.zeros(self.r.WORK_BYTES, dtype=torch.uint8, device=dev)
        self.xstage = torch.zeros(64 * H, dtype=torch.bfloat16, device=dev)
        self.part1 = torch.zeros(self.r.PART1_FLOATS, dtype=torch.float32, device=dev)
        self.part2 = torch.zeros(self.r.PART2_FLOATS, dtype=torch.float32, device=dev)
        self.inter = torch.zeros(self.r.MAX_PAIRS * I, dtype=torch.uint8, device=dev)
        self.inter_sf = torch.zeros(self.r.MAX_PAIRS * (I // 32), dtype=torch.uint8, device=dev)
        torch.cuda.synchronize()
        self.live = False
        self.launches = 0
        self.parks = 0

    def start(self):
        self.seg.set_u64(self.r.OFF_CTRL, 0)
        self.seg.set_u64(self.r.OFF_RTX_STATE, 1)
        self.r.launch(self.seg.dptr, self.layers, self.work, self.xstage, self.part1, self.part2, self.inter,
                      self.inter_sf, IDLE_NS, WAIT_NS, 0, 0)
        self.live = True
        self.launches += 1

    def parked(self) -> bool:
        """The kernel exited on its own (idle limit): reap it."""
        if self.live and self.seg.u64(self.r.OFF_RTX_STATE) == 0:
            self.r.wait_exit()  # raises if it trapped
            self.live = False
            self.parks += 1
            return True
        return False

    def stop(self):
        if not self.live:
            return
        self.seg.set_u64(self.r.OFF_CTRL, 1)
        self.r.wait_exit()  # raises if the kernel trapped
        self.seg.set_u64(self.r.OFF_CTRL, 0)
        self.live = False

    def pending(self) -> bool:
        """A v3 call published but not yet acknowledged (slot 0's tag differs from done)."""
        w0 = self.seg.u64(4096)  # slot 0's word0 (peer3::kOffRec)
        return (w0 >> 48) != 0 and (w0 >> 48) != self.seg.u64(self.r.OFF_DONE)


def serve_v3_only(a, dev):
    """Test mode (PEER3_ONLY=1): no b12x and no v2 protocol. The cold experts are put into b12x's prepared layout
    with the CPU spec (b12x-spec/spec.py, verified bit for bit against prepare_weights), then the v3 kernel serves
    until its control word asks it to exit."""
    sys.path.insert(0, os.path.join(PEER3_DIR, "..", "b12x-spec"))
    import spec

    tensor = tensor_reader(a.model)
    rowmap = json.load(open(a.rowmap))["layers"]
    keep, n_cold = [], []
    for layer in range(a.layers):
        cold = list(rowmap[str(layer)]["cold"])
        parts = {k: [] for k in ("w13", "s13", "w2", "s2")}
        for e in cold:
            prefix = f"layers.{layer}.ffn.experts.{e}"
            parts["w13"].append(torch.cat([tensor(f"{prefix}.w1.weight"), tensor(f"{prefix}.w3.weight")]))
            parts["s13"].append(torch.cat([tensor(f"{prefix}.w1.scale"), tensor(f"{prefix}.w3.scale")]))
            parts["w2"].append(tensor(f"{prefix}.w2.weight"))
            parts["s2"].append(tensor(f"{prefix}.w2.scale"))
        Wl = {k: torch.stack(v) for k, v in parts.items()}
        P = spec.logical_to_prepared(Wl["w13"], Wl["s13"], Wl["w2"], Wl["s2"], hidden_size=H, intermediate_size=I)
        keep.append({"w13": P.w13_rp.view(torch.uint8).to(dev), "s13": P.w13_sfb.view(torch.uint8).to(dev),
                     "w2": P.w2_rp.view(torch.uint8).to(dev), "s2": P.w2_sfb.view(torch.uint8).to(dev)})
        n_cold.append(len(cold))
        print(f"layer {layer}: {len(cold)} cold experts (spec layout)", flush=True)
    v3 = Peer3Server(dev, keep, n_cold)
    v3.start()
    v3.stats.start()
    print("serving (v3 only)", flush=True)
    while True:  # until the control word asks to exit; relaunch after idle parks, as the full server does
        v3.r.wait_exit()
        v3.live = False
        if v3.seg.u64(v3.r.OFF_CTRL) != 0:
            break
        v3.parks += 1
        while not v3.pending() and v3.seg.u64(v3.r.OFF_CTRL) == 0:
            time.sleep(20e-6)
        if v3.seg.u64(v3.r.OFF_CTRL) != 0:
            break
        v3.start()
    print(f"v3 kernel exited; launches {v3.launches}, idle parks {v3.parks}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rowmap", required=True)
    p.add_argument("--model", default="/model")
    p.add_argument("--layers", type=int, default=40)
    a = p.parse_args()

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    if os.environ.get("PEER3_ONLY") == "1":
        serve_v3_only(a, dev)
        return
    if not os.path.isdir(B12X_DIR):
        raise RuntimeError(f"sidecar b12x dir missing: {B12X_DIR}")
    sys.path.insert(0, B12X_DIR)
    from benchmarks.moe_preparation import request_for_capacity

    tensor = tensor_reader(a.model)
    rowmap = json.load(open(a.rowmap))["layers"]
    rows_max = max(BUCKETS)
    x_buf = torch.empty(rows_max, H, dtype=torch.uint8, device=dev)
    xs_buf = torch.empty(rows_max, H // 32, dtype=torch.uint8, device=dev)
    a_buf = torch.zeros(rows_max, H, dtype=torch.bfloat16, device=dev)
    ids_buf = torch.full((rows_max, K), -1, dtype=torch.int32, device=dev)
    w_buf = torch.zeros(rows_max, K, dtype=torch.float32, device=dev)
    out_buf = torch.zeros(rows_max, H, dtype=torch.bfloat16, device=dev)
    g = torch.Generator(device=dev).manual_seed(0)
    a_buf.normal_(0.0, 0.3, generator=g)
    live = torch.rand(rows_max, K, device=dev, generator=g) < (1.4 / K)
    live[:, 0] = True
    n_cold_guess = len(rowmap["0"]["cold"])
    ids_buf.copy_(torch.where(live, torch.randint(0, n_cold_guess, (rows_max, K), device=dev, generator=g,
                                                  dtype=torch.int32), -1))
    w_buf.copy_(torch.where(live, torch.rand(rows_max, K, device=dev, generator=g) * 0.3, 0.0))

    plan = None
    experts, decls, keep, n_cold = {}, {}, [], []
    shared_scratch, shared_src = {}, {}
    start = time.time()
    for layer in range(a.layers):
        cold = list(rowmap[str(layer)]["cold"])
        parts = {k: [] for k in ("w13", "s13", "w2", "s2")}
        for e in cold:
            prefix = f"layers.{layer}.ffn.experts.{e}"
            parts["w13"].append(torch.cat([tensor(f"{prefix}.w1.weight"), tensor(f"{prefix}.w3.weight")]))
            parts["s13"].append(torch.cat([tensor(f"{prefix}.w1.scale"), tensor(f"{prefix}.w3.scale")]))
            parts["w2"].append(tensor(f"{prefix}.w2.weight"))
            parts["s2"].append(tensor(f"{prefix}.w2.scale"))
        W = {k: torch.stack(v).to(dev) for k, v in parts.items()}
        del parts
        ones = torch.ones(len(cold), dtype=torch.float32, device=dev)
        one = torch.ones((), dtype=torch.float32, device=dev)
        if plan is None:
            plan = fm.plan_weights(
                source=fm.PackedSource(format=fm.PackedSourceFormat("fp4_e8m0_k32"),
                                       w13_layout=fm.W13Layout("w31")),
                activation=fm.ActivationSpec(mode=fm.ActivationMode.A8, nonlinearity="silu",
                                             io_dtype=torch.bfloat16, swiglu_limit=LIMIT,
                                             swiglu_alpha=None, swiglu_beta=None),
                geometry=fm.MoEGeometry(num_experts=len(cold), hidden_size=H, intermediate_size=I))
        ex = fm.prepare_weights(plan=plan, weights=fm.PackedWeights(
            w13=W["w13"], w2=W["w2"],
            w13_block_scales=W["s13"].clamp_(max=247).view(torch.float8_e8m0fnu),
            w2_block_scales=W["s2"].clamp_(max=247).view(torch.float8_e8m0fnu),
            w13_global_scales=ones, w2_global_scales=ones, input_scale=one, intermediate_scale=one,
            immutable_input_scales=True))
        decl = fm.plan_execution(experts=ex, capacity=fm.ExecutionCapacity(
            max_tokens=rows_max, top_k=K, warmup_token_counts=tuple(BUCKETS), route_num_experts=0))

        def call(st, m, ex=ex):
            specs = st.scratch.scratch_specs()
            cur = shared_scratch.get(m)
            if cur is None or len(cur) != len(specs) or any(
                    t.numel() < int(sp.shape[0]) or t.dtype != sp.dtype for t, sp in zip(cur, specs)):
                shared_scratch[m] = tuple(
                    torch.empty(max(int(sp.shape[0]), cur[i].numel() if cur and i < len(cur) else 0),
                                dtype=sp.dtype, device=sp.device)
                    for i, sp in enumerate(specs))
            if m not in shared_src:
                shared_src[m] = a_buf[:m].clone()
            binding = st.bind(scratch=shared_scratch[m], a=a_buf[:m], experts=ex, topk_ids=ids_buf[:m],
                              topk_weights=w_buf[:m], output=out_buf[:m], input_scales_static=True)
            return PreparedCall(run=binding.run, output=out_buf[:m], owners=(binding,))

        calls = {m: (lambda st, m=m: call(st, m)) for m in decl.token_counts}

        def race(st, m):
            trial = call(st, m)
            return replace(trial, produce=lambda: a_buf[:m].copy_(shared_src[m]))

        session = PreparationSession(device=dev)
        session.prepare((request_for_capacity(
            decl, name=f"peer.cold.l{layer}", calls=calls,
            benchmark_calls={m: (lambda st, m=m: race(st, m)) for m in calls}),))
        del session, calls  # `race` stays: the benchmark lambdas above close over it
        gc.collect()
        torch.cuda.empty_cache()
        experts[layer], decls[layer] = ex, decl
        keep.append(W)  # prepare_weights rewrote these in place: b12x's prepared layout, read by v3 as well
        n_cold.append(len(cold))
        print(f"layer {layer}: {len(cold)} cold experts prepared, "
              f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated", flush=True)
    print(f"prepared {a.layers} layers in {time.time() - start:.0f}s; "
          f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated", flush=True)

    from b12x.preparation.types import require_prepared
    scratch = {}
    for m in BUCKETS:
        variant = decls[0].variants[m]
        state = require_prepared(variant, variant.component_id)
        specs = state.scratch.scratch_specs()
        assert all(t.numel() >= int(s.shape[0]) and t.dtype == s.dtype for t, s in zip(shared_scratch[m], specs))
        scratch[m] = shared_scratch[m]

    buf = pt.open_shared(create=True)
    host, _ = pt.register(buf, dev)
    words = np.frombuffer(buf, dtype=np.int64, count=8)
    header = np.frombuffer(buf, dtype=np.int64, count=pt.HEADER, offset=pt.SLOT)
    x_host = host[pt.X_OFF:pt.XS_OFF]
    xs_host = host[pt.XS_OFF:pt.IDS_OFF]
    ids_host = host[pt.IDS_OFF:pt.W_OFF].view(torch.int32)
    w_host = host[pt.W_OFF:pt.OUT_OFF].view(torch.float32)
    out_host = host[pt.OUT_OFF:pt.OUT_OFF + rows_max * H * 2].view(torch.bfloat16)
    stream = torch.cuda.Stream(dev)
    torch.cuda.set_stream(stream)
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    ids_buf[:, 0] = 0
    w_buf[:, 0] = 0.1
    for layer in range(a.layers):
        for m in BUCKETS:
            binding = fm.bind(decls[layer].variants[m], scratch=scratch[m], a=a_buf[:m], experts=experts[layer],
                              topk_ids=ids_buf[:m], topk_weights=w_buf[:m], output=out_buf[:m],
                              input_scales_static=True)

            def work(m=m, binding=binding):
                small = m <= pt.SMALL_ROWS
                if small:
                    x_buf[:m].view(-1).copy_(x_host[: m * H], non_blocking=True)
                    xs_buf[:m].view(-1).copy_(xs_host[: m * (H // 32)], non_blocking=True)
                    ids_buf[:m].view(-1).copy_(ids_host[: m * K], non_blocking=True)
                    w_buf[:m].view(-1).copy_(w_host[: m * K], non_blocking=True)
                _dequant[(m, H // 1024)](x_buf, xs_buf, a_buf, H=H, BLOCK=1024)
                fm.run(binding=binding)
                if small:
                    out_host[: m * H].copy_(out_buf[:m].view(-1), non_blocking=True)

            work()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                work()
            graphs[(layer, m)] = graph
    ids_buf.fill_(-1)
    torch.cuda.synchronize()
    print(f"captured {len(graphs)} graphs; {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated", flush=True)

    v3 = Peer3Server(dev, keep, n_cold)
    print(f"peer3: segment {PEER3_SHM}; {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated", flush=True)
    v3.start()
    v3.stats.start()
    print(f"serving; v3 call statistics -> {STATS_FILE3}", flush=True)
    last, served, busy, rows_total = int(words[0]), 0, 0.0, 0
    STATS_FILE = os.path.join(STATS_DIR, "peer_stats.json")
    stats, last_dump = {}, 0.0
    bucket_of = [0] + [next(m for m in BUCKETS if m >= r) for r in range(1, rows_max + 1)]
    while True:
        seq = int(words[0])
        if seq == last:
            v3.parked()
            if not v3.live and v3.pending():
                v3.start()  # a v3 call is waiting: (re)launch the persistent kernel
            continue
        if int(header[0]) != seq:
            continue
        if v3.live:
            v3.stop()
        layer, rows = int(header[1]), int(header[2])
        if rows > 0:
            t0 = time.perf_counter()
            m = bucket_of[rows]
            if m <= pt.SMALL_ROWS:
                graphs[(layer, m)].replay()
            else:
                x_buf[:rows].view(-1).copy_(x_host[: rows * H], non_blocking=True)
                xs_buf[:rows].view(-1).copy_(xs_host[: rows * (H // 32)], non_blocking=True)
                ids_buf[:rows].view(-1).copy_(ids_host[: rows * K], non_blocking=True)
                w_buf[:rows].view(-1).copy_(w_host[: rows * K], non_blocking=True)
                if rows < m:
                    ids_buf[rows:m].fill_(-1)
                graphs[(layer, m)].replay()
                out_host[: rows * H].copy_(out_buf[:rows].view(-1), non_blocking=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            busy += dt
            served += 1
            rows_total += rows
            st = stats.setdefault(m, [0, 0.0, 0])
            st[0] += 1
            st[1] += dt
            st[2] += rows
            if served % 256 == 0:
                now = time.time()
                if now - last_dump > 2:
                    last_dump = now
                    with open(STATS_FILE + ".tmp", "w") as f:
                        json.dump({str(k): {"calls": v[0], "mean_us": round(v[1] / v[0] * 1e6, 1),
                                            "mean_rows": round(v[2] / v[0], 1)} for k, v in sorted(stats.items())}, f)
                    os.replace(STATS_FILE + ".tmp", STATS_FILE)
            if served % 2000 == 0:
                print(f"served {served} v2 cold layer calls, mean {busy / served * 1e6:.1f} us, "
                      f"mean rows {rows_total / served:.1f}; v3 kernel launches {v3.launches}, idle parks {v3.parks}",
                      flush=True)
        words[1] = seq
        last = seq


if __name__ == "__main__":
    main()
