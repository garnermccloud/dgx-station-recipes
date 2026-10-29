#!/usr/bin/env python3
"""Batch-1 coding prompt: ~2,700-token module, exactly 2,048 output tokens, greedy thinking off."""
import argparse, json, secrets, statistics, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLISHED = json.load(open(HERE / "published.json"))["coding_greedy_off_tok_s"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30003")
    p.add_argument("--key-file", default="~/.ds41/api-key")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--min-pct", type=float, default=80)
    a = p.parse_args()
    key = open(a.key_file.replace("~", str(Path.home()))).read().strip()
    source = (HERE / "data/coding_source.py").read_text()
    def once():
        nonce = secrets.token_hex(8)
        prompt = (f"Run {nonce}.\n\nYou are reviewing the following Python module.\n\n```python\n{source}\n```\n\n"
                  "Write a complete pytest test suite for this module: cover every public function, edge cases and "
                  "error paths. Output only the code.")
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048,
                "min_tokens": 2048, "ignore_eos": True, "temperature": 0.0, "top_p": 1.0, "stream": True,
                "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}}
        req = urllib.request.Request(a.base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t0 = time.perf_counter(); first = None; usage = {}
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]": continue
                ch = json.loads(line[6:])
                if ch.get("usage"): usage = ch["usage"]
                delta = (ch.get("choices") or [{}])[0].get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")):
                    first = time.perf_counter()
        end = time.perf_counter()
        if first is None: raise RuntimeError("no tokens")
        n = usage.get("completion_tokens", 0)
        return first - t0, (n - 1) / (end - first), usage.get("prompt_tokens", 0), n
    once()  # warm
    runs = [once() for _ in range(a.runs)]
    speed = statistics.median(r[1] for r in runs)
    pct = 100 * speed / PUBLISHED
    print(f"coding greedy thinking off: {speed:.1f} tok/s (published {PUBLISHED}, {pct:.0f}%) "
          f"TTFT {1000*statistics.median(r[0] for r in runs):.0f} ms prompt {runs[0][2]} tok output {runs[0][3]} tok "
          f"runs {[round(r[1]) for r in runs]} -> {'PASS' if pct >= a.min_pct else 'FAIL'}")
    raise SystemExit(0 if pct >= a.min_pct else 1)

if __name__ == "__main__":
    main()
