#!/usr/bin/env python3
"""Decode throughput at C1/C4/C8/C16 from the server's own counters."""
import argparse, json, re, secrets, threading, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLISHED = {"C1": 467, "C4": 992, "C8": 1369, "C16": 1970}

def counters(base):
    text = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    out = {}
    for m in re.finditer(r"^(vllm:[a-z_]+?)(?:_total)?(?:\{[^}]*\})? ([0-9.e+]+)$", text, re.M):
        out[m[1]] = out.get(m[1], 0.0) + float(m[2])
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30003")
    p.add_argument("--key-file", default="~/.config/ds41/api.key")
    p.add_argument("--model", default="DeepSeek-V4.1-Flash")
    p.add_argument("--conc", default="1,4,8,16")
    p.add_argument("--window", type=float, default=60)
    p.add_argument("--warmup", type=float, default=20)
    p.add_argument("--min-pct", type=float, default=80)
    a = p.parse_args()
    key = open(a.key_file.replace("~", str(Path.home()))).read().strip()
    base = a.base_url.rstrip("/")
    questions = [json.loads(l)["question"] for l in open(HERE / "data/gsm8k-200.jsonl")]
    def post(body):
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        return urllib.request.urlopen(req, timeout=1800)
    def worker(stop):
        i = secrets.randbelow(len(questions))
        while not stop.is_set():
            body = {"model": a.model, "temperature": 1.0, "top_p": 0.95, "max_tokens": 4096, "stream": True,
                    "messages": [{"role": "user", "content": questions[i % len(questions)]}],
                    "chat_template_kwargs": {"thinking": True}}
            try:
                with post(body) as r:
                    for _ in r:
                        if stop.is_set(): break
            except OSError:
                time.sleep(1)
            i += 1
    ok = True
    for c in [int(x) for x in a.conc.split(",")]:
        stop = threading.Event()
        threads = [threading.Thread(target=worker, args=(stop,), daemon=True) for _ in range(c)]
        for t in threads: t.start()
        time.sleep(a.warmup)
        x, t0 = counters(base), time.time()
        time.sleep(a.window)
        y, t1 = counters(base), time.time()
        stop.set()
        d = {k: y.get(k, 0) - x.get(k, 0) for k in y}
        agg = d.get("vllm:generation_tokens", 0) / (t1 - t0)
        drafts = d.get("vllm:spec_decode_num_drafts", 0)
        accept = 1 + d.get("vllm:spec_decode_num_accepted_tokens", 0) / drafts if drafts else float("nan")
        ref = PUBLISHED.get(f"C{c}")
        pct = 100 * agg / ref if ref else float("nan")
        good = ref is None or pct >= a.min_pct
        ok &= good
        print(f"C{c}: {agg:,.0f} tok/s aggregate, {agg/c:,.0f} per user, {accept:.2f} tokens/step; "
              f"published {ref}, {pct:.0f}% -> {'PASS' if good else 'FAIL'}", flush=True)
        for t in threads: t.join(timeout=60)
    raise SystemExit(0 if ok else 1)

if __name__ == "__main__":
    main()
