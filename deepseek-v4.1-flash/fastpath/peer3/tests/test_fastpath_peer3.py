"""GB300: the `peer3` fastpath component end to end against the real v3 sidecar (peer_server3.py, PEER3_ONLY mode).

A stand-in for vLLM's DeepseekV4MegaMoEExperts (class + the hook's forward, whose module globals carry `_orig`)
is patched by fastpath_peer3.patch exactly as fastpath_hook would. The stock MegaMoE forward is replaced by a stub
that returns zeros (and records the hot ids it was given), so y == the cold rows the sidecar computed, which are
checked against the fp64 recipe reference. Eager calls and CUDA-graph replays, T = 1..64, int64 ids.
"""
from __future__ import annotations

import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ref  # noqa: E402
from rt_server import cold_experts  # noqa: E402

E_TOTAL, H = 384, 5120


def main() -> None:
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    # fastpath_hook stand-in: record the registration, patch by hand.
    registered = []
    fh = types.ModuleType("fastpath_hook")
    fh.register = lambda comp, target: (lambda fn: registered.append((comp, target, fn)) or fn)
    sys.modules["fastpath_hook"] = fh
    sys.path.insert(0, "/w/fastpath")
    import fastpath_peer3

    (comp, target, patch_fn), = registered
    assert comp == "peer3" and target == "vllm.models.deepseek_v4.nvidia.model"

    seen_hot = {}

    def stock_forward(self, hidden_states, topk_weights, topk_ids, *, activation_clamp, fast_math=True):
        seen_hot["ids"] = topk_ids
        return torch.zeros(hidden_states.shape[0], H, dtype=torch.bfloat16, device=hidden_states.device)

    hook_ns = {"_orig": {"forward": stock_forward}, "torch": torch}
    exec("def _forward(self, hidden_states, topk_weights, topk_ids, *, activation_clamp, fast_math=True):\n"
         "    raise AssertionError('the v2 path must not run for T <= 64')\n", hook_ns)
    mod = types.ModuleType(target)
    mod.DeepseekV4MegaMoEExperts = type("DeepseekV4MegaMoEExperts", (), {"forward": hook_ns["_forward"]})
    patch_fn(mod)
    cls = mod.DeepseekV4MegaMoEExperts

    cold = cold_experts(16)
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
    gen = torch.Generator().manual_seed(1)
    worst = 0.0
    n_eq = n_el = 0
    det = os.environ.get("PEER3_DETERMINISTIC_PREFILL") == "1"
    for T in (1, 2, 6, 6, 12, 24, 32, 64) + ((100, 130) if det else ()):
        ids = torch.empty(T, 6, dtype=torch.int64)
        for t in range(T):
            nc = int(torch.randint(0, 3, (1,), generator=gen))
            mine = [cold[int(i)] for i in torch.randperm(16, generator=gen)[:nc]]
            rest = [hot_pool[int(i)] for i in torch.randperm(len(hot_pool), generator=gen)[: 6 - nc]]
            row = mine + rest
            ids[t] = torch.tensor([row[int(i)] for i in torch.randperm(6, generator=gen)])
        w = torch.rand(T, 6, generator=gen) * 0.4 + 0.05
        x = (torch.randn(T, H, generator=gen) * 0.8).to(torch.bfloat16)
        ids_d, w_d, x_d = ids.to(dev), w.to(dev), x.to(dev)
        y = cls.forward(self, x_d, w_d, ids_d, activation_clamp=10.0)
        torch.cuda.synchronize()
        rm = row_map[ids]
        cold_ids = torch.where(rm < 0, -rm - 1, -1)
        assert torch.equal(seen_hot["ids"].cpu(), torch.where(rm >= 0, rm, -1)), "hot ids handed to MegaMoE differ"
        p3 = fastpath_peer3._state
        c0 = (T - 1) // 64 * 64  # T > 64 (chunked prefill): the segment holds the last chunk's rows only
        n = T - c0
        xq = p3.seg.host[p3.ext.OFF_XQ:p3.ext.OFF_XQ + 64 * H].view(64, H)[:n].clone()
        xs = p3.seg.host[p3.ext.OFF_XS:p3.ext.OFF_XS + 64 * 160].view(64, 160)[:n].clone()
        refo = ref.cold_rows(Wc, xq, xs, cold_ids[c0:], w[c0:])
        d = (y[c0:].cpu().double() - refo).abs()
        worst = max(worst, float(d.max()))
        n_eq += int((d == 0).sum())
        n_el += d.numel()
        # The same call captured in a CUDA graph and replayed.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            yg = cls.forward(self, x_d, w_d, ids_d, activation_clamp=10.0)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        assert torch.equal(yg, y), f"graph replay differs from the eager call (T={T})"
        print(f"T={T:2d}: cold rows max|d| vs fp64 {float(d.max()):.2e}; graph replay identical", flush=True)
    st = fastpath_peer3._state.stats.cpu()
    print(f"OK: fastpath peer3 end to end; {n_eq}/{n_el} elements bit-equal to the fp64 recipe (max |d| {worst:.2e}); "
          f"waits {int(st[0])}, timeouts {int(st[3])}")
    fastpath_peer3._state.seg.set_u64(fastpath_peer3._state.ext.OFF_CTRL, 1)  # stop the test sidecar


if __name__ == "__main__":
    main()
