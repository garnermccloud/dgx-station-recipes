#!/usr/bin/env python3
"""GSM8K-200: thinking on, reasoning_effort 50, T=0.3, 16 concurrent, score the final number."""
import argparse, json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NUM = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
HERE = Path(__file__).resolve().parent

def norm(s):
    s = s.replace(',', '').rstrip('.')
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s

def pred(text):
    text = text or ''
    m = re.search(r'####\s*(.*)', text)
    seg = m.group(1) if m else (re.findall(r'\\boxed\{([^}]*)\}', text) or [text])[-1]
    nums = NUM.findall(seg.replace('$', ''))
    return norm(nums[0] if m and nums else (nums[-1] if nums else ''))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30003")
    p.add_argument("--key-file", default="~/.ds41/api-key")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--conc", type=int, default=16)
    p.add_argument("--min-acc", type=float, default=95.0)
    p.add_argument("--out", default="/tmp/ds41-gsm8k-200.jsonl")
    a = p.parse_args()
    key = open(a.key_file.replace("~", str(Path.home()))).read().strip()
    rows = [json.loads(l) for l in open(HERE / "data/gsm8k-200.jsonl")][:200]
    def ask(i):
        r = rows[i]
        gold = norm(r['answer'].split('####')[-1].strip())
        body = {'model': a.model, 'temperature': 0.3, 'max_tokens': 24000,
                'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 50},
                'messages': [{'role': 'user', 'content': r['question'] +
                              "\n\nSolve the problem. End your reply with a final line of the form '#### <number>'."}]}
        t = time.time()
        try:
            req = urllib.request.Request(a.base_url.rstrip('/') + '/v1/chat/completions',
                                         data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
            d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
            ch = d['choices'][0]
            content = ch['message'].get('content') or ''
            p = pred(content)
            return dict(i=i, gold=gold, pred=p, ok=p == gold, finish=ch.get('finish_reason'),
                        ctoks=d['usage']['completion_tokens'], sec=round(time.time() - t, 1), tail=content[-200:])
        except Exception as e:
            return dict(i=i, gold=gold, pred=None, ok=False, finish='error', error=str(e)[:300], sec=round(time.time() - t, 1))
    t0 = time.time()
    with ThreadPoolExecutor(a.conc) as ex, open(a.out, 'w') as f:
        res = []
        for r in ex.map(ask, range(len(rows))):
            res.append(r); f.write(json.dumps(r) + '\n'); f.flush()
    ok = sum(r['ok'] for r in res)
    runaways = sum(r['finish'] == 'length' for r in res)
    errors = sum(r['finish'] == 'error' for r in res)
    acc = 100 * ok / len(res)
    print(json.dumps(dict(n=len(res), correct=ok, acc=round(acc, 2), runaways=runaways, errors=errors,
                          wall_s=round(time.time() - t0), median_ctoks=sorted(r.get('ctoks', 0) for r in res)[len(res) // 2])))
    raise SystemExit(0 if acc >= a.min_acc else 1)

if __name__ == "__main__":
    main()
