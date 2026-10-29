"""GB300: peer3 send vs the production peer_fused2.route_send2 (MEGA_FUSED_SEND=2) on the same inputs.

Checks, bit for bit: the hot ids handed to MegaMoE, the per-token "has cold" flags, the MXFP8 bytes and scales of
every cold token's row, and the cold ids / route weights of every cold slot. Real rowmap (rowmap-mix-v1-h255),
random routes and activations at T = 1..64, int64 and int32 top-k ids.
"""
from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, "/hook")
import peer3_ext  # noqa: E402
import peer3_seg  # noqa: E402
import peer_fused2  # noqa: E402  (production)

E, K, H = 384, 6, 5120


class FakePeer2:
    """Device buffers in place of PeerTier2's host-mapped segment (route_send2 only needs the pointers)."""

    def __init__(self, dev):
        n = 64
        self.px = torch.zeros(n * H, dtype=torch.uint8, device=dev)
        self.pxs = torch.zeros(n * (H // 32), dtype=torch.uint8, device=dev)
        self.pids = torch.zeros(n * K, dtype=torch.int32, device=dev)
        self.pw = torch.zeros(n * K, dtype=torch.float32, device=dev)
        self.words = torch.zeros(64, dtype=torch.int64, device=dev)
        self.header = torch.zeros(64, dtype=torch.int64, device=dev)
        self.seq = torch.zeros(1, dtype=torch.int64, device=dev)


def main() -> None:
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    g = peer3_ext.load("gb300")
    rowmap = json.load(open("/hook/rowmap-mix-v1-h255.json"))["layers"]
    seg_path = "/dev/shm/w4rtx/peer3_sendtest"
    if os.path.exists(seg_path):
        os.remove(seg_path)
    seg = peer3_seg.Segment(seg_path, g.SEG_BYTES, dev, create=True, zero_header=True)
    state = torch.tensor([1, 0], dtype=torch.int32, device=dev)
    peer = FakePeer2(dev)
    gen = torch.Generator().manual_seed(0)
    n_cases = 0
    for layer in (0, 10, 39):
        rm = torch.empty(E, dtype=torch.int32)
        for r, e in enumerate(rowmap[str(layer)]["hot"]):
            rm[e] = r
        for c, e in enumerate(rowmap[str(layer)]["cold"]):
            rm[e] = -(c + 1)
        row_map = rm.to(dev)
        for T in (1, 2, 3, 6, 8, 13, 24, 32, 48, 64):
            for dtype in (torch.int64, torch.int32):
                ids = torch.stack([torch.randperm(E, generator=gen)[:K] for _ in range(T)]).to(dtype)
                if T > 2:
                    ids[1, 3] = -1  # a masked route
                w = torch.rand(T, K, generator=gen)
                x = (torch.randn(T, H, generator=gen) * torch.logspace(-3, 1, T)[:, None]).to(torch.bfloat16)
                x[0, :64] = 0  # an all-zero MXFP8 block
                ids_d, w_d, x_d = ids.to(dev), w.to(dev), x.to(dev)
                hot2, has2, pos2, rows2 = peer_fused2.route_send2(peer, ids_d, w_d, row_map, None, x_d, layer, H, K, 64)
                hot3 = torch.empty_like(ids_d)
                has3 = torch.empty(T, dtype=torch.int8, device=dev)
                g.send(ids_d, w_d.contiguous(), row_map, x_d, hot3, has3, seg.dptr, state, layer)
                torch.cuda.synchronize()
                assert torch.equal(hot2, hot3), "hot ids differ"
                assert torch.equal(has2, has3), "has differs"
                state[0] = (state[0] % 15) + 1  # no finish here: advance the call counter by hand
                pos = pos2.cpu()
                xq3 = seg.host[g.OFF_XQ:g.OFF_XQ + 64 * H].view(64, H)
                xs3 = seg.host[g.OFF_XS:g.OFF_XS + 64 * (H // 32)].view(64, H // 32)
                px = peer.px.view(64, H).cpu()
                pxs = peer.pxs.view(64, H // 32).cpu()
                pids = peer.pids.view(64, K).cpu()
                pw = peer.pw.view(64, K).cpu()
                for t in range(T):
                    if not bool(has2[t]):
                        continue
                    r = int(pos[t])
                    assert torch.equal(px[r], xq3[t]), f"MXFP8 bytes differ (layer {layer} T {T} token {t})"
                    assert torch.equal(pxs[r], xs3[t]), f"MXFP8 scales differ (layer {layer} T {T} token {t})"
                    rec = seg.host[g.OFF_REC + t * 128:g.OFF_REC + t * 128 + 32]
                    word0 = int(rec[:8].view(torch.int64))
                    w3 = rec[8:32].view(torch.float32)
                    for k in range(K):
                        cid = (word0 >> (8 * k)) & 0xFF
                        exp = int(pids[r, k])
                        assert (cid == 0xFF and exp == -1) or cid == exp, "cold ids differ"
                        if exp >= 0:
                            assert float(w3[k]) == float(pw[r, k]), "weights differ"
                n_cases += 1
    print(f"OK: {n_cases} cases (layers 0/10/39, T = 1..64, int64 and int32 ids): hot ids, flags, MXFP8 rows, "
          f"cold ids and route weights bit-identical to peer_fused2.route_send2")


if __name__ == "__main__":
    main()
