#!/usr/bin/env python3
"""The full metrics suite behind `ds41 bench --full`: one server session, one JSON, one markdown table.

Stages (each can be skipped with --skip STAGE,...):
  fixed     single stream, the fixed prompt (data/fixed_prompt.txt, ~2.6K tokens), greedy, thinking off,
            2,048 output tokens; --reps runs after one warm-up: output tok/s, TTFT, accepted tokens per step, steps/s
  gsm8k_c1  single stream, GSM8K questions, T=1.0 / top_p 0.95, thinking on (the verify decode shape), timed window
  greedy_c1 single stream, GSM8K questions, greedy, thinking off, timed window
  coding    single stream, the coding prompt (data/coding_source.py, verify/coding.py's shape), --coding-reps runs
  streams   C4 / C8 / C16 / C32, the verify decode shape: aggregate and per-stream tok/s, tokens per step
  prefill   uncached prompts of ~16K / 64K / 128K tokens (distinct filler + cache_salt): TTFT and prefill tok/s
  capacity  kvfill.py: 16 distinct ~110K-token contexts decoding 2,500 forced tokens at once + one needle request
  quality   GSM8K-200, the 200K needles and the tool call (the verify scripts)
Timed windows read the server's own counters (/metrics) after a warm-up. Medians come with the interquartile range.
The table states our numbers and how they were measured; it makes no comparison with other systems.
"""
import argparse
import json
import re
import secrets
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("fixed", "gsm8k_c1", "greedy_c1", "coding", "streams", "prefill", "capacity", "quality")
WORDS = ("river mountain harbor lantern orchard meadow granite copper willow thunder quiet ancient northern valley "
         "signal archive compass ember glacier prairie canyon beacon harvest silver distant market bridge village "
         "forest meridian cedar falcon tide marble season journal engine cable garden").split()


class Client:
    def __init__(self, base: str, key_file: str, model: str):
        self.base = base.rstrip("/")
        self.key = Path(key_file).expanduser().read_text().strip()
        self.model = model

    def request(self, body: dict, timeout: float = 1800):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        return urllib.request.urlopen(req, timeout=timeout)

    def counters(self) -> dict:
        text = urllib.request.urlopen(self.base + "/metrics", timeout=10).read().decode()
        out: dict = {}
        for m in re.finditer(r"^(vllm:[a-z_]+?)(?:_total)?(?:\{[^}]*\})? ([0-9.e+]+)$", text, re.M):
            out[m[1]] = out.get(m[1], 0.0) + float(m[2])
        return out

    def stream(self, body: dict) -> dict:
        """One streamed chat: TTFT, first/last token times, usage, and the server's spec-decode counter deltas."""
        body = dict(body, model=self.model, stream=True,
                    stream_options={"include_usage": True, "continuous_usage_stats": True})
        before = self.counters()
        t0 = time.perf_counter()
        first = last = None
        count = 0
        usage: dict = {}
        with self.request(body) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                ev = json.loads(line[5:])
                usage = ev.get("usage") or usage
                n = usage.get("completion_tokens", 0) if usage else 0
                if n > count:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    count = n
        after = self.counters()
        if first is None:
            raise RuntimeError("the server streamed no tokens")
        d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
        drafts = d.get("vllm:spec_decode_num_drafts", 0.0)
        accepted = d.get("vllm:spec_decode_num_accepted_tokens", 0.0)
        span = last - first
        return dict(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=count,
                    ttft_ms=round((first - t0) * 1000, 1),
                    out_tok_s=round((count - 1) / span, 1) if count > 1 and span > 0 else None,
                    tokens_per_step=round(1 + accepted / drafts, 3) if drafts else None,
                    steps_s=round(drafts / span, 1) if drafts and span > 0 else None)


def spread(values: list) -> dict:
    """Median and interquartile range of a list of numbers (None entries dropped)."""
    v = sorted(x for x in values if x is not None)
    if not v:
        return dict(n=0)
    if len(v) == 1:
        return dict(n=1, median=v[0], q1=v[0], q3=v[0])
    q = statistics.quantiles(v, n=4, method="inclusive")
    return dict(n=len(v), median=round(statistics.median(v), 3), q1=round(q[0], 3), q3=round(q[2], 3))


def summarize(runs: list) -> dict:
    return {k: spread([r[k] for r in runs]) for k in ("out_tok_s", "ttft_ms", "tokens_per_step", "steps_s",
                                                      "completion_tokens", "prompt_tokens")}


def fixed_prompt(c: Client, a) -> dict:
    text = (HERE / "data/fixed_prompt.txt").read_text()
    def body():
        return {"temperature": 0, "max_tokens": 2048, "chat_template_kwargs": {"thinking": False},
                "messages": [{"role": "user", "content": f"[cache-bust nonce: {secrets.token_hex(16)}]\n{text}"}]}
    c.stream(body())  # warm-up
    runs = [c.stream(body()) for _ in range(a.reps)]
    return dict(shape="fixed prompt, greedy, thinking off, max_tokens 2048, nonce line first", runs=runs,
                summary=summarize(runs))


def coding(c: Client, a) -> dict:
    source = (HERE / "data/coding_source.py").read_text()
    def body():
        prompt = (f"Run {secrets.token_hex(8)}.\n\nYou are reviewing the following Python module.\n\n```python\n"
                  f"{source}\n```\n\nWrite a complete pytest test suite for this module: cover every public function, "
                  "edge cases and error paths. Output only the code.")
        return {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048, "min_tokens": 2048, "ignore_eos": True,
                "chat_template_kwargs": {"thinking": False}, "messages": [{"role": "user", "content": prompt}]}
    c.stream(body())  # warm-up
    runs = [c.stream(body()) for _ in range(a.coding_reps)]
    return dict(shape="verify/coding.py: coding module prompt, greedy, thinking off, exactly 2,048 output tokens",
                runs=runs, summary=summarize(runs))


def window(c: Client, conc: int, a, *, greedy: bool) -> dict:
    """conc closed-loop streams over GSM8K questions; tokens counted by the server between two counter reads."""
    questions = [json.loads(line)["question"] for line in open(HERE / "data/gsm8k-200.jsonl")]
    errors = []

    def worker(stop, seed):
        i = seed
        while not stop.is_set():
            body = {"model": c.model, "max_tokens": 4096, "stream": True,
                    "messages": [{"role": "user", "content": questions[i % len(questions)]}]}
            if greedy:
                body.update(temperature=0.0, chat_template_kwargs={"thinking": False})
            else:
                body.update(temperature=1.0, top_p=0.95, chat_template_kwargs={"thinking": True})
            try:
                with c.request(body) as r:
                    for _ in r:
                        if stop.is_set():
                            break
            except OSError as exc:
                errors.append(repr(exc)[:200])
                time.sleep(1)
            i += 1

    stop = threading.Event()
    threads = [threading.Thread(target=worker, args=(stop, secrets.randbelow(200)), daemon=True)
               for _ in range(conc)]
    for t in threads:
        t.start()
    time.sleep(a.warmup)
    x, t0 = c.counters(), time.time()
    time.sleep(a.window)
    y, t1 = c.counters(), time.time()
    stop.set()
    for t in threads:
        t.join(timeout=120)
    d = {k: y.get(k, 0.0) - x.get(k, 0.0) for k in y}
    agg = d.get("vllm:generation_tokens", 0.0) / (t1 - t0)
    drafts = d.get("vllm:spec_decode_num_drafts", 0.0)
    accepted = d.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    return dict(conc=conc, greedy=greedy, window_s=round(t1 - t0, 1), warmup_s=a.warmup,
                aggregate_tok_s=round(agg, 1), per_stream_tok_s=round(agg / conc, 1),
                tokens_per_step=round(1 + accepted / drafts, 3) if drafts else None,
                preemptions=d.get("vllm:num_preemptions", 0.0), request_errors=len(errors))


def filler(tokens: int, seed: int) -> str:
    import random
    rng = random.Random(seed)
    words: list = []
    while len(words) < tokens / 1.35:
        k = rng.randint(8, 16)
        s = [rng.choice(WORDS) for _ in range(k)]
        s[0] = s[0].capitalize()
        s[-1] += "."
        words += s
    return " ".join(words)


def prefill(c: Client, a) -> dict:
    rows = []
    for target in (16384, 65536, 131072):
        runs = []
        for rep in range(a.prefill_reps):
            prompt = filler(target, secrets.randbits(31)) + "\n\nSummarize the text above in one sentence."
            r = c.stream({"temperature": 0, "max_tokens": 1, "cache_salt": secrets.token_hex(16),
                          "chat_template_kwargs": {"thinking": False},
                          "messages": [{"role": "user", "content": prompt}]})
            r["prefill_tok_s"] = round(r["prompt_tokens"] / (r["ttft_ms"] / 1000), 1)
            runs.append(r)
        rows.append(dict(target_tokens=target, runs=runs,
                         prompt_tokens=spread([r["prompt_tokens"] for r in runs]),
                         ttft_ms=spread([r["ttft_ms"] for r in runs]),
                         prefill_tok_s=spread([r["prefill_tok_s"] for r in runs])))
    return dict(shape="distinct random-word filler + per-request cache_salt (no prefix-cache hit), max_tokens 1",
                rows=rows)


def run_script(args: list) -> tuple:
    p = subprocess.run([sys.executable] + args, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def capacity(a, common: list) -> dict:
    out_dir = Path(a.out).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rc, text = run_script([str(HERE / "kvfill.py"), str(out_dir)] + common +
                          ["--n", str(a.capacity_n), "--tokens", str(a.capacity_tokens),
                           "--max-tokens", str(a.capacity_max_tokens), "--needle-seed", str(secrets.randbits(31))])
    record = json.loads((out_dir / "kvfill.json").read_text())
    record.update(rc=rc, wall_s=round(time.time() - t0, 1), n=a.capacity_n, tokens=a.capacity_tokens,
                  max_tokens=a.capacity_max_tokens, all_running_at_once=record.get("running_max") == a.capacity_n + 1)
    return record


def quality(a, common: list) -> dict:
    out = {}
    rc, text = run_script([str(HERE / "tool.py")] + common)
    out["tool_call"] = dict(passed=rc == 0, output=text.splitlines()[-1] if text else "")
    rc, text = run_script([str(HERE / "needle.py")] + common + ["--tokens", "200000"])
    m = re.search(r"needle (\d+)/(\d+)", text)
    out["needles_200k"] = dict(passed=rc == 0, found=int(m[1]) if m else None, total=int(m[2]) if m else None,
                               lines=[l for l in text.splitlines() if l.startswith("needle depth")])
    rc, text = run_script([str(HERE / "gsm8k.py")] + common + ["--min-acc", str(a.min_acc),
                                                              "--out", str(Path(a.out).with_suffix(".gsm8k.jsonl"))])
    last = text.splitlines()[-1] if text else "{}"
    try:
        g = json.loads(last)
    except json.JSONDecodeError:
        g = dict(raw=last[:300])
    out["gsm8k_200"] = dict(g, passed=rc == 0)
    return out


def read_env(path: str) -> dict:
    out: dict = {}
    for line in Path(path).read_text().splitlines():
        k, _, v = line.partition("=")
        out[k] = v.split(",") if k in ("components", "components_active") else v
    if out.get("kv_cache_tokens"):
        out["kv_cache_tokens"] = int(out["kv_cache_tokens"])
    out["tree_dirty"] = out.get("tree_dirty") == "true"
    return out


def fmt(s: dict, key: str = "median", nd: int = 1) -> str:
    if not s or s.get("n", 0) == 0:
        return "n/a"
    if s["n"] == 1:
        return f"{s['median']:.{nd}f}"
    return f"{s[key]:.{nd}f} ({s['q1']:.{nd}f}-{s['q3']:.{nd}f})"


def markdown(rec: dict) -> str:
    e = rec.get("environment", {})
    lines = ["# ds41 bench --full", "",
             f"Image `{e.get('image', '?')}`, driver {e.get('driver', '?')}, GPUs {e.get('gpus', '?')}, "
             f"util {e.get('util', '?')}, rowmap `{e.get('rowmap', '?')}`, recipe commit `{e.get('commit', '?')}`, "
             f"GPU KV cache {e.get('kv_cache_tokens', '?')} tokens. One server session; "
             f"started {rec.get('started')}. Medians with the interquartile range in brackets.", "",
             "## Single stream", "",
             "| shape | output tok/s | TTFT ms | accepted tokens/step | steps/s | runs |", "|---|---|---|---|---|---|"]
    for key, label in (("fixed", "fixed prompt (~2.6K in, 2,048 out, greedy, thinking off)"),
                       ("coding", "coding prompt (2,048 out, greedy, thinking off)")):
        s = rec.get(key)
        if s:
            m = s["summary"]
            lines.append(f"| {label} | {fmt(m['out_tok_s'])} | {fmt(m['ttft_ms'], nd=0)} | "
                         f"{fmt(m['tokens_per_step'], nd=2)} | {fmt(m['steps_s'])} | {m['out_tok_s'].get('n', 0)} |")
    for key, label in (("gsm8k_c1", "GSM8K decode, T=1.0, thinking on"), ("greedy_c1", "GSM8K decode, greedy, thinking off")):
        s = rec.get(key)
        if s:
            lines.append(f"| {label} ({s['window_s']:.0f} s window) | {s['aggregate_tok_s']:.1f} | — | "
                         f"{s['tokens_per_step'] or 'n/a'} | — | window |")
    if rec.get("streams"):
        lines += ["", "## Many streams (GSM8K, T=1.0, thinking on)", "",
                  "| streams | aggregate tok/s | per stream tok/s | tokens/step | window s |", "|---|---|---|---|---|"]
        for s in rec["streams"]:
            lines.append(f"| {s['conc']} | {s['aggregate_tok_s']:,.0f} | {s['per_stream_tok_s']:,.1f} | "
                         f"{s['tokens_per_step'] or 'n/a'} | {s['window_s']:.0f} |")
    if rec.get("prefill"):
        lines += ["", "## Prefill (uncached)", "", "| prompt tokens | TTFT ms | prefill tok/s |", "|---|---|---|"]
        for r in rec["prefill"]["rows"]:
            lines.append(f"| {fmt(r['prompt_tokens'], nd=0)} | {fmt(r['ttft_ms'], nd=0)} | {fmt(r['prefill_tok_s'], nd=0)} |")
    k = rec.get("capacity")
    if k:
        lines += ["", "## Long-context capacity", "",
                  f"GPU KV cache: {e.get('kv_cache_tokens', '?')} tokens. {k['n']} distinct ~{k['tokens']:,}-token "
                  f"contexts decoding {k['max_tokens']:,} forced tokens each, plus one needle request: "
                  f"{'all ran at once' if k.get('all_running_at_once') else 'not all ran at once'} "
                  f"(max running {k.get('running_max')}, max waiting {k.get('waiting_max')}), "
                  f"{k.get('preemptions', 0):.0f} preemptions, peak KV use "
                  f"{100 * (k.get('kv_usage_max') or 0):.0f} %, wall {k.get('wall_s')} s, "
                  f"{'all requests correct' if k.get('all_ok') else 'FAILED requests'}."]
    q = rec.get("quality")
    if q:
        g = q["gsm8k_200"]
        n = q["needles_200k"]
        lines += ["", "## Quality (same session)", "", "| check | result |", "|---|---|",
                  f"| GSM8K-200 (thinking on, T=0.3) | {g.get('correct', '?')}/{g.get('n', '?')} "
                  f"({g.get('acc', '?')} %) |",
                  f"| needles at 200K tokens | {n.get('found')}/{n.get('total')} |",
                  f"| tool call | {'PASS' if q['tool_call']['passed'] else 'FAIL'} |"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:30003")
    p.add_argument("--key-file", default="~/.ds41/api-key")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--out", required=True, help="JSON path; the markdown table goes next to it (.md)")
    p.add_argument("--env-file", help="environment record (key=value lines) written by `ds41 bench --full`")
    p.add_argument("--skip", default="", help="comma-separated stages to skip: " + ",".join(STAGES))
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--coding-reps", type=int, default=5)
    p.add_argument("--prefill-reps", type=int, default=3)
    p.add_argument("--window", type=float, default=60)
    p.add_argument("--warmup", type=float, default=20)
    p.add_argument("--conc", default="4,8,16,32")
    p.add_argument("--capacity-n", type=int, default=16)
    p.add_argument("--capacity-tokens", type=int, default=110000)
    p.add_argument("--capacity-max-tokens", type=int, default=2500)
    p.add_argument("--min-acc", type=float, default=95.0)
    a = p.parse_args()
    skip = {s for s in a.skip.split(",") if s}
    unknown = skip - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}; known: {', '.join(STAGES)}")
    c = Client(a.base_url, a.key_file, a.model)
    common = ["--base-url", a.base_url, "--key-file", a.key_file, "--model", a.model]
    rec: dict = dict(schema="ds41-bench-full-v1", started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     environment=read_env(a.env_file) if a.env_file else {},
                     settings={k: v for k, v in vars(a).items() if k not in ("key_file",)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def save():
        Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
        Path(a.out).with_suffix(".md").write_text(markdown(rec))

    steps = [("fixed", lambda: fixed_prompt(c, a)),
             ("gsm8k_c1", lambda: window(c, 1, a, greedy=False)),
             ("greedy_c1", lambda: window(c, 1, a, greedy=True)),
             ("coding", lambda: coding(c, a)),
             ("streams", lambda: [window(c, int(x), a, greedy=False) for x in a.conc.split(",")]),
             ("prefill", lambda: prefill(c, a)),
             ("capacity", lambda: capacity(a, common)),
             ("quality", lambda: quality(a, common))]
    failed = []
    for name, fn in steps:
        if name in skip:
            continue
        t0 = time.time()
        print(f"[bench] {name} ...", flush=True)
        try:
            rec[name] = fn()
        except Exception as exc:  # recorded in the JSON and fails the run
            rec[name] = dict(error=f"{type(exc).__name__}: {exc}"[:500])
            failed.append(name)
        print(f"[bench] {name} done in {time.time() - t0:.0f} s", flush=True)
        save()
    q = rec.get("quality") or {}
    gates = dict(
        tool_call=(q.get("tool_call") or {}).get("passed"),
        needles=(q.get("needles_200k") or {}).get("passed"),
        gsm8k=(q.get("gsm8k_200") or {}).get("passed"),
        capacity=(rec.get("capacity") or {}).get("all_ok"))
    rec["gates"] = gates
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rec["status"] = "PASS" if not failed and all(v is not False for v in gates.values()) else "FAIL"
    rec["failed_stages"] = failed
    save()
    print(markdown(rec))
    print(f"[bench] {rec['status']}: {a.out} (+ .md)" + (f"; failed stages {failed}" if failed else ""))
    raise SystemExit(0 if rec["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
