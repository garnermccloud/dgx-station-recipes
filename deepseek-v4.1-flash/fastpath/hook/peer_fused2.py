"""Peer tier v2 decode path: two launches per MoE layer instead of four.

The previous send path needed its activations pre-quantized (a separate MXFP8 quantize launch) and ran in one CTA,
so it packed up to 64 rows x 5 KB serially; PeerTier2.finish was two launches (_wait, then _scatter_add).

route_send2 (grid = T CTAs): every CTA recomputes the route split for all T tokens (T*K <= 384 ids, cheap) to get
has/pos/rows, then CTA t writes its own hot ids and, if token t has a cold route, quantizes its bf16 row to MXFP8
in registers and packs it at pos[t]. CTA 0 also writes has/pos/rows and the pad masking. Each CTA
fences its payload at system scope and bumps a device arrival counter; the last to arrive publishes the header and
the sequence (release, sys) and resets the counter, so the kernel is graph-replay safe.

finish2 (grid = T CTAs): CTA t exits at once unless token t has a cold route (CTA 0 always waits so a missing
peer is detected); otherwise it spins until the peer completed this sequence and adds the peer's row pos[t] into
y[t]. A lost peer traps; it never drops the cold rows.

The MXFP8 quantizer matches vLLM's _mxfp8_e4m3_quantize_torch recipe: per 32 values, e8m0 = clamp(ceil(log2(amax /
448)) + 127, 0, 254) with all-zero blocks at 0, values scaled by the exact power of two and rounded to e4m3 (RNE).
test_peer_fused2.py checks it bit for bit against the quantizer the live path uses (FlashInfer cute-dsl).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

FUSED_MAX = 64


@triton.jit
def _fence_sys(dep):
    return tl.inline_asm_elementwise("fence.acq_rel.sys; mov.u32 $0, 0;", "=r,r", [dep], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit(do_not_specialize=["LAYER", "T"])
def _route_send2(IDS, WTS, ROWMAP, HOT, HAS, POS, ROWS, X,
                 PX, PXS, PIDS, PW, WORDS, HEADER_PTR, SEQ, ARRIVE, LAYER, T,
                 H: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr,
                 BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr):
    me = tl.program_id(0)
    t = tl.arange(0, BLOCK_T)
    k = tl.arange(0, BLOCK_K)
    live = t < T
    m = live[:, None] & (k[None, :] < K)
    ids = tl.load(IDS + t[:, None] * K + k[None, :], mask=m, other=-1).to(tl.int32)
    valid = ids >= 0
    safe = tl.where(valid, ids, 0)
    rm = tl.load(ROWMAP + safe, mask=m, other=0)
    cold = tl.where(valid & (rm < 0), -rm - 1, -1)
    has = tl.where(live, tl.max((cold >= 0).to(tl.int32), axis=1), 0)
    pos = tl.cumsum(has, axis=0) - 1
    rows = tl.sum(has, axis=0)

    if me == 0:
        tl.store(HAS + t, has.to(tl.int8), mask=live)
        tl.store(POS + t, pos, mask=live)
        tl.store(ROWS, rows)
        if PAD > 0:
            # The peer copies whole buckets of up to PAD rows inside its graphs: rows past `rows` carry masked routes.
            r = tl.arange(0, PAD)
            padm = (r[:, None] >= rows) & (k[None, :] < K)
            tl.store(PIDS + r[:, None] * K + k[None, :], tl.full((PAD, BLOCK_K), -1, tl.int32), mask=padm)

    # This CTA's token.
    km = k < K
    my_ids = tl.load(IDS + me * K + k, mask=km, other=-1).to(tl.int32)
    my_valid = my_ids >= 0
    my_rm = tl.load(ROWMAP + tl.where(my_valid, my_ids, 0), mask=km, other=0)
    my_hot = tl.where(my_valid & (my_rm >= 0), my_rm, -1)
    my_cold = tl.where(my_valid & (my_rm < 0), -my_rm - 1, -1)
    tl.store(HOT + me * K + k, my_hot.to(HOT.dtype.element_ty), mask=km)
    my_has = tl.max((my_cold >= 0).to(tl.int32), axis=0)
    my_pos = tl.sum(tl.where(t < me, has, 0), axis=0)  # == pos[me] when my_has
    if my_has > 0:
        dst = my_pos.to(tl.int64)
        tl.store(PIDS + dst * K + k, my_cold, mask=km)
        tl.store(PW + dst * K + k, tl.load(WTS + me * K + k, mask=km, other=0.0).to(tl.float32), mask=km)
        nb: tl.constexpr = H // 32
        e = tl.arange(0, 32)
        for c0 in tl.static_range(0, nb, 32):  # 32 scale blocks (1,024 values) per step
            b = c0 + tl.arange(0, 32)
            bm = b < nb
            x = tl.load(X + me.to(tl.int64) * H + b[:, None] * 32 + e[None, :], mask=bm[:, None], other=0.0)
            x = x.to(tl.float32)
            amax = tl.max(tl.abs(x), axis=1)
            q = tl.math.div_rn(amax, 448.0)
            bits = q.to(tl.uint32, bitcast=True)
            ex = ((bits >> 23) & 0xFF).to(tl.int32)
            mant = (bits & 0x7FFFFF).to(tl.int32)
            # clamp(ceil(log2(q)) + 127, 0, 254): exact from the fp32 bits; subnormal q only reaches 1 above 2^-127
            sb = tl.where(ex == 0, (mant > 0x400000).to(tl.int32), ex + (mant != 0).to(tl.int32))
            sb = tl.minimum(sb, 253)
            inv = ((254 - sb) << 23).to(tl.uint32).to(tl.float32, bitcast=True)  # 2^(127 - sb), exact
            xq = (x * inv[:, None]).to(tl.float8e4nv)
            tl.store(PX + dst * H + b[:, None] * 32 + e[None, :], xq.to(tl.uint8, bitcast=True), mask=bm[:, None])
            tl.store(PXS + dst * nb + b, sb.to(tl.uint8), mask=bm)

    # Every thread's payload stores are ordered before this CTA's arrival.
    _fence_sys(rows)
    tl.debug_barrier()
    arrived = tl.atomic_add(ARRIVE, 1, sem="acq_rel", scope="gpu")
    if arrived == T - 1:
        # Every CTA fenced its payload at system scope before arriving; this CTA acquired all arrivals, so its
        # system-scope release below orders the whole payload for the peer.
        seq = tl.load(SEQ) + 1
        tl.store(SEQ, seq)
        tl.store(HEADER_PTR + 1, LAYER.to(tl.int64))
        tl.store(HEADER_PTR + 2, rows.to(tl.int64))
        tl.store(HEADER_PTR + 3, T.to(tl.int64))
        tl.store(HEADER_PTR + 0, seq)
        tl.atomic_xchg(WORDS, seq, sem="release", scope="sys")
        tl.store(ARRIVE, 0)


@triton.jit
def _trap(dep):
    return tl.inline_asm_elementwise("trap; mov.u32 $0, 0;", "=r,r", [dep], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def _finish2(Y, OUT, HAS, POS, ROWS, WORDS, SEQ, H: tl.constexpr, TIMEOUT: tl.constexpr):
    me = tl.program_id(0)
    if tl.load(ROWS) == 0:
        return
    mine = tl.load(HAS + me) != 0
    if (me != 0) & (mine == 0):
        return
    seq = tl.load(SEQ)
    spins = 0
    done = tl.atomic_add(WORDS + 1, 0, sem="acquire", scope="sys")
    while (done < seq) & (spins < TIMEOUT):
        done = tl.atomic_add(WORDS + 1, 0, sem="acquire", scope="sys")
        spins += 1
    if done < seq:
        if me == 0:
            tl.atomic_add(WORDS + 2, 1, sem="relaxed", scope="sys")
        _trap(seq)
    if mine:
        r = tl.load(POS + me).to(tl.int64)
        row = me.to(tl.int64) * H
        for c0 in tl.static_range(0, H, 1024):
            cols = c0 + tl.arange(0, 1024)
            cm = cols < H
            y = tl.load(Y + row + cols, mask=cm).to(tl.float32)
            o = tl.load(OUT + r * H + cols, mask=cm).to(tl.float32)
            tl.store(Y + row + cols, (y + o).to(Y.dtype.element_ty), mask=cm)


_arrive: dict = {}


def route_send2(peer, topk_ids, topk_weights, row_map, x, layer: int, hidden: int, topk: int, pad: int):
    """Split routes, quantize + pack cold rows and publish in one launch; returns (hot_ids, has, pos, rows)."""
    tokens, k = topk_ids.shape
    if tokens > FUSED_MAX or k != topk or (pad and pad & (pad - 1)) or hidden % 32 or x.dtype != torch.bfloat16:
        raise ValueError(f"route_send2 handles up to {FUSED_MAX} bf16 tokens of top-{topk} routes, power-of-two pad")
    dev = topk_ids.device
    arrive = _arrive.get(dev)
    if arrive is None:
        arrive = _arrive[dev] = torch.zeros(1, dtype=torch.int32, device=dev)
    ids = topk_ids.contiguous()
    hot = torch.empty_like(ids)
    has = torch.empty(tokens, dtype=torch.int8, device=dev)
    pos = torch.empty(tokens, dtype=torch.int32, device=dev)
    rows = torch.empty(1, dtype=torch.int32, device=dev)
    _route_send2[(tokens,)](
        ids, topk_weights.contiguous(), row_map, hot, has, pos, rows,
        x.contiguous(), peer.px, peer.pxs, peer.pids, peer.pw, peer.words, peer.header, peer.seq, arrive,
        layer, tokens, H=hidden, K=topk, PAD=pad, BLOCK_T=max(16, triton.next_power_of_2(tokens)), BLOCK_K=8,
        num_warps=4)
    return hot, has, pos, rows


def finish2(peer, y, has, pos, rows) -> None:
    """Wait for the peer and add its rows into y (one launch)."""
    _finish2[(y.shape[0],)](y, peer.pout, has, pos, rows, peer.words, peer.seq, H=y.shape[-1],
                            TIMEOUT=peer.timeout, num_warps=4)
