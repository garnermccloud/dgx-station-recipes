# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garner McCloud
"""Per-boundary timing of the decode layer from a torch-profiler trace (CUPTI kernel spans of the replayed
FULL graphs): rank the dense chain's kernel boundaries by the time each adds. CPU-only; pure python
(gzip + json).

A layer instance = the kernels between two consecutive target MegaMoE calls (E=255) within one step. Every kernel is
labelled by name (LABELS; unknown kernels keep a short name). Per layer kind (plain / compressor+indexer / window,
from which attention and indexer kernels it contains) and label, over all instances:
  start, end      relative to the previous MoE's end (us; medians and p90)
  dur             CUPTI duration (with PDL this includes time spent before griddepcontrol.wait)
Along the critical chain (CHAIN: the order of the main dependency chain, labels missing in a kind are skipped):
  adds            this milestone's end minus the previous milestone's end: the time the boundary + op add to the chain
  lead            previous milestone's end minus this kernel's start: how early it was resident (PDL / early trigger)
  tail            this end minus max(previous end, this start): the work left once both are true
The chain total is the next MoE's start. Boundaries are ranked by median `adds`.

Usage: python3 tools/boundary_report.py TRACE.json.gz [TRACE2 ...] [--kind plain|indexer|window|all] [--steps N]
       [--by-t] [--csv out.csv]   (--by-t: separate verify widths, e.g. dynk's 6 vs 8 rows, from woa_kernel<S>)   (several traces: one table per trace, plus a delta table vs the first)
"""
from __future__ import annotations

import argparse
import re
import collections
import gzip
import json
import statistics

TARGET_MOE = "mega_moe_impl<9600u, 5120u, 2304u, 255u"
BY_T = False
LABELS = (  # (substring of the kernel name, label); first match wins
    ("mega_moe_impl<9600u, 5120u, 2304u, 128u", "draft_moe"),
    (TARGET_MOE, "moe"),
    ("seam_kernel", "seam"),
    ("mega_seam_kernel", "mega_seam"),
    ("mega_mix_kernel", "mega_mix"),
    ("mix_kernel", "mix"),
    ("router_kernel", "router"),
    ("_fused_engram_post_wkv", "engram_post"),
    ("mxfp8_quantize", "fi_quant"),
    ("quantizationkernelsmxfp8", "fi_quant"),
    ("QkvStage", "w3_qkv"),
    ("CandKeysStage", "w3_cand_keys"),
    ("CandSplitStage", "w3_cand_split"),
    ("CandFinalStage", "w3_cand_final"),
    ("RemapStage", "w3_remap"),
    ("topk_global_cs", "w8tg_topk"),
    ("topk_select_bf16", "deepselect_topk"),
    ("sparse_mqa_logits", "idx_logits"),
    ("paged_mqa_logits", "idx_paged_logits"),
    ("_fused_save_compress_norm", "compress_norm"),
    ("_rope_quant_insert", "rope_insert"),
    ("_k_store_fp32", "k_store"),
    ("attn_par_kernel", "attn"),
    ("attn_ss_kernel", "attn"),
    ("attn_cx_kernel", "attn"),
    ("woa_kernel", "woa"),
    ("low_latency_blockscaled_gemm", "fi_ll_gemm"),
    ("dense_blockscaled_gemm", "fi_dense_gemm"),
    ("linearcute_dsl_ll", "ll_bf16"),
    ("_prepare_megamoe_inputs", "prep_moe"),
)
# the main dependency chain of a target layer, in order (fi_ll_gemm occurs twice before attention: wq_a+wkv, wq_b;
# once after: wo_b; the occurrence index is appended by the parser)
CHAIN = ("seam", "fi_quant", "fi_ll_gemm", "w3_qkv", "fi_ll_gemm#1", "idx_paged_logits", "w8tg_topk",
         "deepselect_topk", "w3_remap", "attn", "woa", "fi_ll_gemm#2", "seam#1", "router", "moe")
# side work that runs beside the chain (reported, not chained): mix (W4 mHC mix on another stream), engram, indexer
# side streams, ...


def label(name: str) -> str:
    for sub, lab in LABELS:
        if sub in name:
            return lab
    return name.split("(")[0].split("<")[0].replace("void ", "").strip()[:28]


def med(v):
    return statistics.median(v) if v else float("nan")


def p90(v):
    return sorted(v)[int(0.9 * (len(v) - 1))] if v else float("nan")


def layers(path: str, max_steps: int | None):
    """Yield (kind, [(label#k, start, end, dur)], next_moe_start) per target layer instance, times relative to the
    previous MoE's end."""
    d = json.load(gzip.open(path)) if path.endswith(".gz") else json.load(open(path))
    ev = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
    ev.sort(key=lambda e: e["ts"])
    moe = [i for i, e in enumerate(ev) if TARGET_MOE in e["name"]]
    n_layers = 0
    for a, b in zip(moe, moe[1:]):
        ea, eb = ev[a], ev[b]
        t0 = ea["ts"] + ea["dur"]
        if eb["ts"] - t0 > 400:  # a step boundary (LM head, sampling, drafter) lies between them
            continue
        seq, seen = [], collections.Counter()
        for e in ev[a + 1:b]:
            lab = label(e["name"])
            k = seen[lab]
            seen[lab] += 1
            seq.append((f"{lab}#{k}" if k else lab, e["ts"] - t0, e["ts"] + e["dur"] - t0, e["dur"]))
        seq.append(("moe", eb["ts"] - t0, eb["ts"] + eb["dur"] - t0, eb["dur"]))  # this layer's MegaMoE
        labs = {s[0].split("#")[0] for s in seq}
        kind = "indexer" if ("w8tg_topk" in labs or "deepselect_topk" in labs or "idx_paged_logits" in labs) else (
            "plain" if any(e[0] == "attn" for e in seq) and any("attn_par" in ev[i]["name"] for i in range(a + 1, b))
            else "window")
        if BY_T:  # the verify width from the W8 woa kernel's template (woa_kernel<S, ...>); '?' if absent
            m = next((re.search(r"woa_kernel<(\d+),", ev[i]["name"]) for i in range(a + 1, b)
                      if "woa_kernel<" in ev[i]["name"]), None)
            kind = f"{kind} T{m.group(1) if m else '?'}"
        yield kind, seq, eb["ts"] - t0
        n_layers += 1
        if max_steps and n_layers >= 40 * max_steps:
            return


def report(path: str, kinds: list[str], max_steps: int | None) -> dict:
    by_kind = collections.defaultdict(list)
    for kind, seq, nxt in layers(path, max_steps):
        by_kind[kind].append((seq, nxt))
    out = {}
    print(f"\n#### {path}: layer instances " + ", ".join(f"{k} {len(v)}" for k, v in sorted(by_kind.items())))
    for kind in kinds:
        inst = by_kind.get(kind, [])
        if not inst:
            continue
        st, en, du = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
        for seq, _ in inst:
            for lab, s, e, dd in seq:
                st[lab].append(s)
                en[lab].append(e)
                du[lab].append(dd)
        totals = [n for _, n in inst]
        print(f"\n== {kind}: {len(inst)} instances; chain total (MoE end -> next MoE start) median {med(totals):.1f} "
              f"p90 {p90(totals):.1f} us")
        print(f"   {'label':18s} {'n':>6s} {'start':>7s} {'end':>7s} {'end p90':>8s} {'dur':>6s}")
        common = [lab for lab in st if len(st[lab]) >= 0.4 * len(inst)]
        for lab in sorted(common, key=lambda x: med(en[x])):
            print(f"   {lab:18s} {len(st[lab]):6d} {med(st[lab]):7.1f} {med(en[lab]):7.1f} {p90(en[lab]):8.1f} "
                  f"{med(du[lab]):6.1f}")
        # chain boundaries: per instance, the CHAIN milestones it contains, in order; each milestone's `adds` is its end
        # minus the previous present milestone's end (aggregated per (previous, milestone) pair)
        agg = collections.defaultdict(lambda: ([], [], []))
        order = []
        for seq, _ in inst:
            m = {lab: (s_, e_) for lab, s_, e_, _ in seq}
            prev, pe = "moe(prev)", 0.0
            for c in CHAIN:
                if c not in m:
                    continue
                s_, e_ = m[c]
                key = (prev, c)
                if key not in agg:
                    order.append(key)
                a_, l_, t_ = agg[key]
                a_.append(e_ - pe)
                l_.append(pe - s_)
                t_.append(e_ - max(pe, s_))
                prev, pe = c, e_
        rows = [(c, p, med(agg[(p, c)][0]), p90(agg[(p, c)][0]), med(agg[(p, c)][1]), med(agg[(p, c)][2]),
                 len(agg[(p, c)][0])) for p, c in order if len(agg[(p, c)][0]) >= 0.2 * len(inst)]
        lay = [seq[-1][2] for seq, _ in inst]  # this layer's MoE end = the layer's period
        print(f"   layer period (previous MoE end -> this MoE end): median {med(lay):.1f} p90 {p90(lay):.1f} us")
        print(f"   {'boundary (previous -> milestone)':40s} {'n':>6s} {'adds':>6s} {'p90':>6s} {'lead':>6s} {'tail':>6s}")
        for c, p, a_, a9, l_, t_, n in rows:
            print(f"   {p + ' -> ' + c:40s} {n:6d} {a_:6.1f} {a9:6.1f} {l_:6.1f} {t_:6.1f}")
        print("   ranked by median adds: " + ", ".join(f"{p}->{c} {a_:.1f}" for c, p, a_, *_ in
                                                     sorted(rows, key=lambda r: -r[2])[:6]))
        out[kind] = {"total": med(totals), "rows": rows}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--kind", default="all")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--csv", default="")
    ap.add_argument("--by-t", action="store_true", help="split layer kinds by verify width (woa_kernel<S>)")
    a = ap.parse_args()
    global BY_T
    BY_T = a.by_t
    base_kinds = ["plain", "indexer", "window"] if a.kind == "all" else [a.kind]
    kinds = [f"{k} T{t}" for k in base_kinds for t in (1, 2, 4, 5, 6, 7, 8)] if a.by_t else base_kinds
    res = [report(t, kinds, a.steps or None) for t in a.traces]
    if len(res) > 1:
        print("\n#### delta vs the first trace (median adds, us)")
        for kind in kinds:
            if kind not in res[0]:
                continue
            base = {(r[1], r[0]): r[2] for r in res[0][kind]["rows"]}
            for t, r in zip(a.traces[1:], res[1:]):
                if kind not in r:
                    continue
                print(f"   {kind} {t}: total {r[kind]['total'] - res[0][kind]['total']:+.1f}; " + ", ".join(
                    f"{p}->{c} {v - base[(p, c)]:+.1f}" for c, p, v, *_ in r[kind]["rows"] if (p, c) in base))
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("trace,kind,boundary,adds_med,adds_p90,lead_med,tail_med\n")
            for t, r in zip(a.traces, res):
                for kind, v in r.items():
                    for c, p, a_, a9, l_, t_, _n in v["rows"]:
                        f.write(f"{t},{kind},{p}->{c},{a_:.2f},{a9:.2f},{l_:.2f},{t_:.2f}\n")


if __name__ == "__main__":
    main()
