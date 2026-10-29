"""Exactness and timing of the overlay's frontend path as the chat endpoint drives it.

1. Render: the overlay (memoized find_last_user_index) renders byte-identical prompts to the unpatched 1017 reference.
2. Tokenize: wrapper(prompt, truncation=..., max_length=..., add_special_tokens=False) returns the same input_ids and
   attention_mask as the plain tokenizer, for full and truncated lengths and adversarial strings.
3. Timing of the server's sequence: apply_chat_template(tokenize=False) then the tokenizer call, cold vs next turn.

    tokcache_check2.py <tokenizer_dir> <source_dir> <reference_encoding_dir>
"""
import copy
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import vllm.tokenizers.deepseek_v41 as dsv41
from transformers import AutoTokenizer

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]


def conversation(sources: list[str], rounds: int) -> list[dict]:
    """A coding-agent conversation with code-sized tool results and a mid-conversation system message."""
    msgs = [{"role": "system", "content": "You are a coding agent. " + sources[0][:20000]},
            {"role": "user", "content": "Find and fix the failing tests."}]
    for i in range(rounds):
        msgs.append({"role": "assistant", "content": "", "reasoning_content": f"Step {i}: module {i}. 中文 123",
                     "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"cmd": f"cat m{i}.py"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": sources[i % len(sources)][:4500]})
        if i == rounds // 2:
            msgs.append({"role": "system", "content": "Reminder: keep edits minimal."})
    return msgs


def main() -> None:
    """Assert exactness, then time the server path."""
    tok_dir, src_dir, ref_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    spec = importlib.util.spec_from_file_location("ref_encoding", ref_dir / "encoding.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    base = AutoTokenizer.from_pretrained(tok_dir)
    wrapper = dsv41.DeepseekV41Tokenizer.from_pretrained(tok_dir)
    sources = [p.read_text(errors="ignore") for p in sorted(src_dir.rglob("*.py")) if p.stat().st_size > 5000][:200]
    checks = 0
    for rounds in (0, 1, 3, 20, 150):
        for tools, thinking in ((TOOLS, True), (None, True), (None, False)):
            msgs = conversation(sources, rounds)
            got = wrapper.apply_chat_template(copy.deepcopy(msgs), tools=tools, tokenize=False, reasoning_effort="max",
                                              thinking=thinking)
            conv = dsv41._normalize_messages(copy.deepcopy(msgs))
            if tools:
                conv[0]["tools"] = tools
            want = ref.encode_messages(conv, thinking_mode="thinking" if thinking else "chat", reasoning_effort="max")
            assert got == want, f"render differs at rounds={rounds} tools={bool(tools)} thinking={thinking}"
            checks += 1
            ids_full = base(got, add_special_tokens=False)
            for max_length in (None, 1, 777, len(ids_full["input_ids"]) - 1, len(ids_full["input_ids"]), len(ids_full["input_ids"]) + 1, 524289):
                kw = {"add_special_tokens": False, "truncation": max_length is not None, "max_length": max_length}
                a, b = wrapper(got, **kw), base(got, **kw)
                assert a["input_ids"] == b["input_ids"] and a["attention_mask"] == b["attention_mask"], f"call differs {kw}"
                checks += 1
    rng = random.Random(1)
    added = list(base.get_added_vocab())
    pieces = [" ", "\n", "123", "中文", "<", "｜", "<｜Use", "</think", "x", "🙂", "\r\n"]
    for _ in range(2000):
        s = "".join(rng.choice(pieces + added) for _ in range(rng.randint(1, 30)))
        kw = {"add_special_tokens": False, "truncation": True, "max_length": rng.randint(1, 40)}
        assert wrapper(s, **kw)["input_ids"] == base(s, **kw)["input_ids"], f"call differs on {s!r}"
        checks += 1
    print(f"exact on {checks} render/tokenize checks")

    msgs = conversation(sources, 150)
    nxt = msgs + [{"role": "assistant", "content": "", "reasoning_content": "Next.", "tool_calls": [{"id": "c999", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
                  {"role": "tool", "tool_call_id": "c999", "content": sources[9][:4500]}]
    fresh = dsv41.DeepseekV41Tokenizer.from_pretrained(tok_dir)
    kw = {"add_special_tokens": False, "truncation": True, "max_length": 524289}
    for name, m in (("cold", msgs), ("next turn", nxt)):
        t0 = time.perf_counter()
        prompt = fresh.apply_chat_template(m, tools=TOOLS, tokenize=False, reasoning_effort="max")
        t1 = time.perf_counter()
        n = len(fresh(prompt, **kw)["input_ids"])
        t2 = time.perf_counter()
        print(f"{name:9s}: render {1000 * (t1 - t0):6.1f} ms + tokenize {1000 * (t2 - t1):6.1f} ms ({n:,} tokens, {len(m)} messages)")
    t = time.perf_counter(); base(prompt, **kw); print(f"plain tokenizer call on the same prompt: {1000 * (time.perf_counter() - t):.1f} ms")


if __name__ == "__main__":
    main()
