"""Reference math for the cold-expert path (fp64 with the exact quantization steps), plus checkpoint loading.

Recipe (DeepGEMM MegaMoE's per-expert numerics, which follow the checkpoint's inference/model.py):
  x      MXFP8 row from the GB300 send (e4m3 * 2^(s - 127) per 32), exact
  g, u   W1 x, W3 x (fp64 here; fp32 accumulation in the kernels)
  act    silu(bf16(min(g, 10))) * bf16(clamp(u, -10, 10)) * route_weight
  a      MXFP8 of act per 32 channels, UE8M0 scale with MegaMoE's rule (floor 2^-22, carry above mantissa 1.75)
  o      bf16(W2 a) per (token, slot)
  out    bf16(sum of the token's cold slots' o in top-k order)
"""
from __future__ import annotations

import json
import os
import struct

import torch

H, I, K = 5120, 2304, 6
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                          dtype=torch.float64)


def tensor_reader(model: str):
    """Read raw checkpoint tensors (uint8 views) by name, like peer_server2.tensor_reader."""
    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    headers = {}

    def tensor(name: str) -> torch.Tensor:
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


def load_experts(model: str, layer: int, experts: list[int]) -> dict[str, torch.Tensor]:
    """Stack experts exactly as peer_server2 does: w13 = [w1; w3], scales clamped to 247 (uint8 views)."""
    tensor = tensor_reader(model)
    parts = {k: [] for k in ("w13", "s13", "w2", "s2")}
    for e in experts:
        prefix = f"layers.{layer}.ffn.experts.{e}"
        parts["w13"].append(torch.cat([tensor(f"{prefix}.w1.weight"), tensor(f"{prefix}.w3.weight")]))
        parts["s13"].append(torch.cat([tensor(f"{prefix}.w1.scale"), tensor(f"{prefix}.w3.scale")]))
        parts["w2"].append(tensor(f"{prefix}.w2.weight"))
        parts["s2"].append(tensor(f"{prefix}.w2.scale"))
    out = {k: torch.stack(v).contiguous() for k, v in parts.items()}
    out["s13"].clamp_(max=247)
    out["s2"].clamp_(max=247)
    return out


def dequant_fp4(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """[rows, k/2] packed FP4 (low nibble = even k) with [rows, k/32] E8M0 -> fp64 [rows, k]."""
    lo = (w & 0xF).long()
    hi = (w >> 4).long()
    vals = torch.stack([FP4_VALUES[lo], FP4_VALUES[hi]], dim=-1).reshape(w.shape[0], -1)
    scale = torch.pow(2.0, s.double() - 127.0).repeat_interleave(32, dim=1)
    return vals * scale


def dequant_mxfp8(xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """[n, H] e4m3 bytes + [n, H/32] e8m0 -> fp64."""
    vals = xq.view(torch.float8_e4m3fn).double()
    return vals * torch.pow(2.0, xs.double() - 127.0).repeat_interleave(32, dim=1)


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).double()


def ue8m0_exp_megamoe(amax: torch.Tensor) -> torch.Tensor:
    """deep_gemm get_ue8m0_sf_exp<e4m3>(fp32 amax)."""
    bits = amax.float().view(torch.int32).long() & 0xFFFFFFFF
    rounded = (bits + 0x7FFFFF - 0x600000) >> 23
    return torch.clamp(rounded, min=113) - 8


def quant_mxfp8_megamoe(act: torch.Tensor) -> torch.Tensor:
    """MXFP8 (per 32) of fp32-representable values with MegaMoE's scale rule; returns the dequantized fp64 values."""
    a32 = act.float().view(-1, 32)
    sf = ue8m0_exp_megamoe(a32.abs().amax(dim=1))
    inv = torch.pow(2.0, 127.0 - sf.double()).float()
    q = (a32 * inv[:, None]).to(torch.float8_e4m3fn)
    return (q.double() * torch.pow(2.0, sf.double() - 127.0)[:, None]).view(act.shape)


def expert_outputs(W: dict[str, torch.Tensor], e: int, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """o = bf16(W2 MXFP8(act)) for the rows x [n, H] (fp64) routed to cold-local expert e with route weights [n]."""
    w13 = dequant_fp4(W["w13"][e], W["s13"][e])
    gate, up = w13[:I], w13[I:]
    g = x @ gate.T
    u = x @ up.T
    gb = bf16(torch.clamp(bf16(g), max=10.0))
    ub = bf16(torch.clamp(bf16(u), min=-10.0, max=10.0))
    act = (gb * torch.sigmoid(gb)) * ub * weights[:, None].double()
    a = quant_mxfp8_megamoe(act.float().double())
    w2 = dequant_fp4(W["w2"][e], W["s2"][e])
    return bf16(a @ w2.T)


def cold_rows(W: dict[str, torch.Tensor], xq: torch.Tensor, xs: torch.Tensor, cold_ids: torch.Tensor,
              weights: torch.Tensor) -> torch.Tensor:
    """bf16-valued fp64 [T, H]: per token, the fp32 sum of its cold slots' bf16 outputs in top-k order."""
    T = cold_ids.shape[0]
    x = bf16(dequant_mxfp8(xq, xs))
    out = torch.zeros(T, H, dtype=torch.float32)
    per = {}
    for t in range(T):
        for k in range(K):
            e = int(cold_ids[t, k])
            if e >= 0:
                per.setdefault(e, []).append((t, k))
    slot_out = {}
    for e, pairs in per.items():
        rows = torch.stack([x[t] for t, _ in pairs])
        wts = torch.tensor([float(weights[t, k]) for t, k in pairs], dtype=torch.float64)
        o = expert_outputs(W, e, rows, wts)
        for i, tk in enumerate(pairs):
            slot_out[tk] = o[i]
    for t in range(T):
        for k in range(K):
            if (t, k) in slot_out:
                out[t] = out[t] + slot_out[(t, k)].float()
    return out.to(torch.bfloat16).double()
