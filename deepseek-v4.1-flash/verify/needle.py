#!/usr/bin/env python3
"""200K-token needle retrieval at 10/50/90% depth; thinking off, temperature 0."""
import argparse, json, random, time, urllib.request

WORDS = ("river mountain harbor lantern orchard meadow granite copper willow thunder quiet ancient northern valley "
         "signal archive compass ember glacier prairie canyon beacon harvest silver distant market bridge village "
         "forest meridian cedar falcon tide marble season journal engine cable garden").split()
PASSPHRASE = "violet-anchor-5817"

def filler(n_words, seed):
    rng = random.Random(seed)
    sentences, count = [], 0
    while count < n_words:
        k = rng.randint(8, 16)
        sentences.append(" ".join(rng.choice(WORDS) for _ in range(k)).capitalize() + ".")
        count += k
    return " ".join(sentences)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30003")
    p.add_argument("--key-file", default="~/.ds41/api-key")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    p.add_argument("--tokens", type=int, default=200000)
    p.add_argument("--depths", default="0.1,0.5,0.9")
    a = p.parse_args()
    key = open(a.key_file.replace("~", str(__import__("pathlib").Path.home()))).read().strip()
    n_words = int(a.tokens / 1.35)
    passed = 0
    depths = [float(d) for d in a.depths.split(",")]
    for depth in depths:
        words = filler(n_words, seed=1234).split(" ")
        cut = int(len(words) * depth)
        needle = f"Remember this carefully: the secret passphrase is {PASSPHRASE}."
        haystack = " ".join(words[:cut] + [needle] + words[cut:])
        prompt = (f"{haystack}\n\nQuestion: What is the secret passphrase mentioned somewhere in the text above? "
                  "Reply with the passphrase only.")
        body = {"model": a.model, "temperature": 0, "max_tokens": 40,
                "chat_template_kwargs": {"thinking": False},
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(a.base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.load(r)
        reply = out["choices"][0]["message"].get("content") or ""
        n = out["usage"]["prompt_tokens"]
        ok = PASSPHRASE in reply
        passed += ok
        print(f"needle depth={depth:.0%} prompt_tokens={n} time={time.time()-t0:.0f}s {'PASS' if ok else 'FAIL'} "
              f"reply={reply.strip()[:60]!r}", flush=True)
    print(f"needle {passed}/{len(depths)}")
    raise SystemExit(0 if passed == len(depths) else 1)

if __name__ == "__main__":
    main()
