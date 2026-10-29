#!/usr/bin/env python3
"""One chat with a tool definition must come back as a parsed tool call."""
import argparse, json, urllib.request

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30003")
    p.add_argument("--key-file", default="~/.ds41/api-key")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    a = p.parse_args()
    key = open(a.key_file.replace("~", str(__import__("pathlib").Path.home()))).read().strip()
    weather = {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    body = {"model": a.model, "max_tokens": 2048, "tools": [weather], "tool_choice": "auto",
            "messages": [{"role": "user", "content": "What's the weather in Denver right now? Use the tool."}]}
    req = urllib.request.Request(a.base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    out = json.load(urllib.request.urlopen(req, timeout=600))
    calls = out["choices"][0]["message"].get("tool_calls") or []
    ok = any(c["function"]["name"] == "get_weather" and "denver" in c["function"]["arguments"].lower() for c in calls)
    print("tool call", "PASS" if ok else "FAIL", [(c["function"]["name"], c["function"]["arguments"]) for c in calls])
    raise SystemExit(0 if ok else 1)

if __name__ == "__main__":
    main()
