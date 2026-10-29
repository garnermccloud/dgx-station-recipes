"""GB300: v2 (prefill-sized, b12x) and v3 (decode-sized, persistent kernel) calls interleaved against one full
peer_server3.py sidecar, as production will drive it.

v3 calls go through the fastpath_peer3 wrapper (stand-in experts class, as test_fastpath_peer3.py); v2 calls go
through the hook's own peer_tier2.PeerTier2 send/finish on a private v2 segment. Checks every v3 output against
the fp64 recipe and every v2 output against it loosely (b12x numerics: cosine), and times the first v3 call after
a v2 burst (the sidecar must stop and relaunch its persistent kernel around v2 work).
"""
from __future__ import annotations

import os
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/hook")
import ref  # noqa: E402
from rt_server import cold_experts  # noqa: E402

E_TOTAL, H, N_COLD = 384, 5120, 8


def main() -> None:
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    registered = []
    fh = types.ModuleType("fastpath_hook")
    fh.register = lambda comp, target: (lambda fn: registered.append(fn) or fn)
    sys.modules["fastpath_hook"] = fh
    sys.path.insert(0, "/w/fastpath")
    import fastpath_peer3
    import peer_tier2
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_impl

    def stock_forward(self, hidden_states, topk_weights, topk_ids, *, activation_clamp, fast_math=True):
        return torch.zeros(hidden_states.shape[0], H, dtype=torch.bfloat16, device=hidden_states.device)

    hook_ns = {"_orig": {"forward": stock_forward}}
    exec("def _forward(self, *a, **k):\n    raise AssertionError('v2 through the hook is not used here')\n", hook_ns)
    mod = types.ModuleType("vllm.models.deepseek_v4.nvidia.model")
    mod.DeepseekV4MegaMoEExperts = type("DeepseekV4MegaMoEExperts", (), {"forward": hook_ns["_forward"]})
    registered[0](mod)
    cls = mod.DeepseekV4MegaMoEExperts
    cold = cold_experts(N_COLD)
    row_map = torch.empty(E_TOTAL, dtype=torch.int32)
    hot = 0
    for e in range(E_TOTAL):
        if e in cold:
            row_map[e] = -(cold.index(e) + 1)
        else:
            row_map[e] = hot
            hot += 1
    self = cls()
    self._mp = types.SimpleNamespace(row_map=row_map.to(dev), idx=0, counts=None)
    Wc = ref.load_experts("/model", 0, cold)
    hot_pool = [e for e in range(E_TOTAL) if e not in cold]
    peer2 = peer_tier2.PeerTier2(dev)
    gen = torch.Generator().manual_seed(2)

    def make(T, cold_frac):
        ids = torch.empty(T, 6, dtype=torch.int64)
        for t in range(T):
            nc = 1 if torch.rand(1, generator=gen).item() < cold_frac else 0
            mine = [cold[int(i)] for i in torch.randperm(N_COLD, generator=gen)[:nc]]
            rest = [hot_pool[int(i)] for i in torch.randperm(len(hot_pool), generator=gen)[: 6 - nc]]
            row = mine + rest
            ids[t] = torch.tensor([row[int(i)] for i in torch.randperm(6, generator=gen)])
        w = torch.rand(T, 6, generator=gen) * 0.4 + 0.05
        x = (torch.randn(T, H, generator=gen) * 0.8).to(torch.bfloat16)
        return ids, w, x

    v3_after_v2 = []
    for rnd in range(6):
        # v2: a prefill-sized call (T = 100)
        ids, w, x = make(100, 0.4)
        ids_d, w_d, x_d = ids.to(dev), w.to(dev), x.to(dev)
        rm = row_map[ids]
        cold_ids = torch.where(rm < 0, -rm - 1, -1).to(torch.int32)
        xq, xs = _mxfp8_e4m3_quantize_impl(x_d.contiguous(), is_sf_swizzled_layout=False)
        y2 = torch.zeros(100, H, dtype=torch.bfloat16, device=dev)
        has, pos, rows = peer2.send(xq, xs, cold_ids.to(dev), w_d, 0)
        peer2.finish(y2, has, pos, rows)
        torch.cuda.synchronize()
        refo = ref.cold_rows(Wc, xq.view(torch.uint8).cpu(), xs.view(torch.uint8).cpu(), cold_ids, w)
        live = (cold_ids >= 0).any(dim=1)
        cos = torch.nn.functional.cosine_similarity(y2.cpu().double()[live].flatten(), refo[live].flatten(), dim=0)
        # v3 right after it: the sidecar has to hand the SMs back to the persistent kernel first.
        ids, w, x = make(6, 0.5)
        ids_d, w_d, x_d = ids.to(dev), w.to(dev), x.to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y3 = cls.forward(self, x_d, w_d, ids_d, activation_clamp=10.0)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        v3_after_v2.append(dt)
        rm = row_map[ids]
        cold3 = torch.where(rm < 0, -rm - 1, -1)
        p3 = fastpath_peer3._state
        xq3 = p3.seg.host[p3.ext.OFF_XQ:p3.ext.OFF_XQ + 64 * H].view(64, H)[:6].clone()
        xs3 = p3.seg.host[p3.ext.OFF_XS:p3.ext.OFF_XS + 64 * 160].view(64, 160)[:6].clone()
        ref3 = ref.cold_rows(Wc, xq3, xs3, cold3, w)
        d3 = (y3.cpu().double() - ref3).abs()
        # a few more v3 calls back to back (kernel already live)
        t1 = time.perf_counter()
        for _ in range(20):
            cls.forward(self, x_d, w_d, ids_d, activation_clamp=10.0)
        torch.cuda.synchronize()
        steady = (time.perf_counter() - t1) / 20
        print(f"round {rnd}: v2 T=100 rows {int(rows)} cos vs fp64 {float(cos):.5f}; first v3 after v2 "
              f"{dt * 1e6:7.1f} us host, steady v3 {steady * 1e6:6.1f} us host, v3 max|d| vs fp64 {float(d3.max()):.2e}",
              flush=True)
        assert float(cos) > 0.995, "v2 output off"
        assert float(d3.max()) < 2e-2, "v3 output off"
    st = fastpath_peer3._state.stats.cpu()
    print(f"OK: v2/v3 interleaved; v3 waits {int(st[0])}, timeouts {int(st[3])}; v2 timeouts {int(peer2.words[2])}")
    fastpath_peer3._state.seg.set_u64(fastpath_peer3._state.ext.OFF_CTRL, 1)


if __name__ == "__main__":
    main()
