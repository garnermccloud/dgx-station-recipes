# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garner McCloud
"""Batch-1 T=0 decode decomposition on the live server: tokens/step and ms/step from vLLM's spec-decode counters.

    step_decomp.py <coding_bench_source_file> <key_file> [--runs 3]
"""
import argparse
import json
import re
import secrets
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:30003"
FROM_SCRATCH = ("Write a complete, production-quality Python module implementing a thread-safe LRU cache with per-entry "
                "TTL, size-based eviction, hit/miss statistics, a decorator for memoizing functions, and full type "
                "hints and docstrings. Then write a complete pytest suite for it. Output only the code.")


def counters() -> dict[str, float]:
    """Spec-decode counters summed over label sets."""
    text = urllib.request.urlopen(f"{URL}/metrics", timeout=30).read().decode()
    out: dict[str, float] = {}
    for name in ("num_drafts_total", "num_accepted_tokens_total", "num_draft_tokens_total"):
        vals = re.findall(rf"^vllm:spec_decode_{name}\{{[^}}]*\}} ([0-9.e+]+)$", text, re.M)
        out[name] = sum(float(v) for v in vals)
    return out


def run(key: str, prompt: str) -> tuple[float, int, float]:
    """One streamed T=0 request, thinking off, exactly 2048 tokens; returns (ttft_s, completion_tokens, decode_s)."""
    body = {"model": "DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048, "min_tokens": 2048, "ignore_eos": True, "temperature": 0.0, "top_p": 1.0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter(); first = None; usage = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            if first is None and (chunk.get("choices") or [{}])[0].get("delta", {}).get("content"):
                first = time.perf_counter()
    end = time.perf_counter()
    return first - t0, int(usage.get("completion_tokens", 0)), end - first


def main() -> None:
    """Run each prompt N times and print tokens/step, ms/step, tok/s."""
    ap = argparse.ArgumentParser()
    ap.add_argument("source"); ap.add_argument("key_file"); ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    key = Path(a.key_file).read_text().strip()
    src = Path(a.source).read_text()
    prompts = {
        "tests-for-module (our coding prompt)": lambda n: (
            f"Run {n}.\n\nYou are reviewing the following Python module.\n\n```python\n{src}\n```\n\n"
            "Write a complete pytest test suite for this module: cover every public function, edge cases and "
            "error paths. Output only the code."),
        "from-scratch codegen": lambda n: f"Run {n}.\n\n{FROM_SCRATCH}",
    }
    for name, make in prompts.items():
        for _ in range(a.runs):
            c0 = counters()
            ttft, toks, dec = run(key, make(secrets.token_hex(8)))
            c1 = counters()
            steps = c1["num_drafts_total"] - c0["num_drafts_total"]
            acc = c1["num_accepted_tokens_total"] - c0["num_accepted_tokens_total"]
            tps = (toks - 1) / dec
            print(f"{name:38s} tok/s {tps:6.1f}  tokens/step {1 + acc / steps:5.2f}  ms/step {1000 * dec / steps:5.2f}  "
                  f"steps {steps:.0f}  TTFT {1000 * ttft:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
