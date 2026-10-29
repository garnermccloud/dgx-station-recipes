#!/usr/bin/env python3
"""Fill the KV pool: N concurrent long distinct prompts plus one needle request, sampling KV usage and preemptions.

  kvfill.py OUT_DIR [--n 16] [--tokens 110000] [--needle-tokens 500000] [--max-tokens 2500] [--needle-seed 1234]
            [--base-url URL] [--key-file FILE] [--model NAME]

Every long prompt is distinct filler (no shared prefix), thinking off, temperature 0, exactly --max-tokens output tokens
(ignore_eos), so the contexts stay resident together. The needle prompt puts a passphrase at 50% depth and must return
it. Writes OUT_DIR/kvfill.json and prints one summary line per request and a final summary. The API key is never
printed.
"""
import argparse
import json
import os
import random
import re
import threading
import time
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:30003'
KEY = ''
MODEL = 'deepseek-v4.1-flash'
WORDS = ('river mountain harbor lantern orchard meadow granite copper willow thunder quiet ancient northern valley '
         'signal archive compass ember glacier prairie canyon beacon harvest silver distant market bridge village '
         'forest meridian cedar falcon tide marble season journal engine cable garden').split()
PASSPHRASE = 'violet-anchor-5817'


def filler(tokens: int, seed: int) -> list[str]:
    """Random sentences of about `tokens` tokens (1.35 tokens per word), deterministic per seed."""
    rng = random.Random(seed)
    words: list[str] = []
    while len(words) < tokens / 1.35:
        k = rng.randint(8, 16)
        sentence = [rng.choice(WORDS) for _ in range(k)]
        sentence[0] = sentence[0].capitalize()
        sentence[-1] += '.'
        words += sentence
    return words


def chat(prompt: str, max_tokens: int, ignore_eos: bool = False) -> dict:
    """One non-streamed chat completion; returns the parsed response."""
    body = {'model': MODEL, 'temperature': 0, 'max_tokens': max_tokens, 'chat_template_kwargs': {'thinking': False},
            'ignore_eos': ignore_eos, 'messages': [{'role': 'user', 'content': prompt}]}
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {KEY}'})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def metrics() -> dict[str, float]:
    """KV usage, running/waiting requests and the preemption counter from the server's /metrics."""
    text = urllib.request.urlopen(BASE + '/metrics', timeout=10).read().decode()
    out = {}
    for name in ('vllm:kv_cache_usage_perc', 'vllm:num_requests_running', 'vllm:num_requests_waiting',
                 'vllm:num_preemptions_total'):
        vals = [float(m[1]) for m in re.finditer(rf'^{re.escape(name)}(?:\{{[^}}]*\}})? ([0-9.e+-]+)$', text, re.M)]
        out[name] = sum(vals)
    return out


def run(args: argparse.Namespace) -> dict:
    """Launch the fill and the needle together, sample metrics until both finish, and return the record."""
    results: list[dict] = []
    lock = threading.Lock()

    def long_request(i: int) -> None:
        t0 = time.time()
        try:
            out = chat(' '.join(filler(args.tokens, 1000 + i)) + '\n\nContinue the text above.', args.max_tokens, True)
            rec = {'id': f'fill{i}', 'ok': True, 'prompt_tokens': out['usage']['prompt_tokens'],
                   'completion_tokens': out['usage']['completion_tokens'],
                   'finish': out['choices'][0]['finish_reason'], 's': round(time.time() - t0, 1)}
        except Exception as exc:  # every failure is recorded and fails the run
            rec = {'id': f'fill{i}', 'ok': False, 'error': f'{type(exc).__name__}: {exc}'[:200], 's': round(time.time() - t0, 1)}
        with lock:
            results.append(rec)
        print(json.dumps(rec), flush=True)

    def needle_request() -> None:
        words = filler(args.needle_tokens, args.needle_seed)
        cut = len(words) // 2
        words = words[:cut] + [f'Remember this carefully: the secret passphrase is {PASSPHRASE}.'] + words[cut:]
        prompt = ' '.join(words) + ('\n\nQuestion: What is the secret passphrase mentioned somewhere in the text above? '
                                    'Reply with the passphrase only.')
        t0 = time.time()
        try:
            out = chat(prompt, 40)
            reply = out['choices'][0]['message'].get('content') or ''
            rec = {'id': 'needle', 'ok': PASSPHRASE in reply, 'prompt_tokens': out['usage']['prompt_tokens'],
                   'reply': reply.strip()[:60], 's': round(time.time() - t0, 1)}
        except Exception as exc:  # recorded; fails the run
            rec = {'id': 'needle', 'ok': False, 'error': f'{type(exc).__name__}: {exc}'[:200], 's': round(time.time() - t0, 1)}
        with lock:
            results.append(rec)
        print(json.dumps(rec), flush=True)

    before = metrics()
    threads = [threading.Thread(target=needle_request)] + [threading.Thread(target=long_request, args=(i,))
                                                          for i in range(args.n)]
    for t in threads:
        t.start()
    samples = []
    misses = 0
    while any(t.is_alive() for t in threads):
        try:
            samples.append(metrics())
        except OSError:  # counted and reported: a busy server may miss a 1 s sample
            misses += 1
        time.sleep(1)
    after = metrics()
    return {
        'results': sorted(results, key=lambda r: r['id']),
        'all_ok': all(r['ok'] for r in results) and len(results) == args.n + 1,
        'kv_usage_max': max((s['vllm:kv_cache_usage_perc'] for s in samples), default=None),
        'running_max': max((s['vllm:num_requests_running'] for s in samples), default=None),
        'waiting_max': max((s['vllm:num_requests_waiting'] for s in samples), default=None),
        'preemptions': after['vllm:num_preemptions_total'] - before['vllm:num_preemptions_total'],
        'prompt_tokens_total': sum(r.get('prompt_tokens', 0) for r in results),
        'metric_samples': len(samples), 'metric_sample_misses': misses,
    }


def main() -> None:
    global BASE, KEY, MODEL
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--tokens', type=int, default=110000)
    p.add_argument('--needle-tokens', type=int, default=500000)
    p.add_argument('--max-tokens', type=int, default=2500)
    p.add_argument('--needle-seed', type=int, default=1234, help='filler seed; change it to avoid the prefix cache')
    p.add_argument('--base-url', default=BASE)
    p.add_argument('--key-file', default='~/.ds41/api-key')
    p.add_argument('--model', default=MODEL)
    args = p.parse_args()
    BASE, MODEL = args.base_url.rstrip('/'), args.model
    KEY = Path(os.path.expanduser(args.key_file)).read_text().strip()
    record = run(args)
    Path(args.out, 'kvfill.json').write_text(json.dumps(record, indent=1))
    summary = {k: v for k, v in record.items() if k != 'results'}
    print('KVFILL', json.dumps(summary), flush=True)
    raise SystemExit(0 if record['all_ok'] else 1)


if __name__ == '__main__':
    main()
