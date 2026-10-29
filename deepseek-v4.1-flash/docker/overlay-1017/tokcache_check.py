"""Prove the overlay's segment-cached tokenization is identical to whole-prompt tokenization, and time it.

Equality on: rendered agent conversations of several sizes, and adversarial strings that put added tokens next to
whitespace, digits, CJK, punctuation, partial special tokens and literal special-token text. Timing: a cold
245K-token prompt, then the same conversation with one more turn (the agent's next request).

    tokcache_check.py <tokenizer_dir> <source_dir>
"""
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
    """A coding-agent conversation with code-sized tool results."""
    msgs = [{"role": "system", "content": "You are a coding agent. " + sources[0][:20000]},
            {"role": "user", "content": "Find and fix the failing tests."}]
    for i in range(rounds):
        msgs.append({"role": "assistant", "content": "", "reasoning_content": f"Step {i}: look at module {i}. 中文 测试 123 ",
                     "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"cmd": f"cat m{i}.py"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": sources[i % len(sources)][:4500]})
    return msgs


def adversarial(added: list[str], rng: random.Random) -> list[str]:
    """Strings mixing added tokens with tricky neighbors."""
    pieces = [" ", "  ", "\n", "\n\n", "\t", "123", "4567", "中文", "テスト", "!", "<", "｜", "<｜Use", "User｜>", "</think", "x", "é", "🙂", "\r\n", ".", "``"]
    out = []
    for _ in range(3000):
        s = "".join(rng.choice(pieces + added) for _ in range(rng.randint(1, 40)))
        out.append(s)
    return out


def main() -> None:
    """Assert equality everywhere, then report timings."""
    tok_dir, src_dir = sys.argv[1], Path(sys.argv[2])
    base = AutoTokenizer.from_pretrained(tok_dir)
    wrapper = dsv41.DeepseekV41Tokenizer.from_pretrained(tok_dir)
    enc = dsv41._SegmentEncoder(base, base.get_added_vocab())
    added = list(base.get_added_vocab())
    sources = [p.read_text(errors="ignore") for p in sorted(src_dir.rglob("*.py")) if p.stat().st_size > 5000][:200]
    checked = 0
    for rounds in (0, 1, 5, 40, 150):
        msgs = conversation(sources, rounds)
        prompt = wrapper.apply_chat_template(msgs, tools=TOOLS, tokenize=False, reasoning_effort="max")
        assert enc.encode(prompt) == base.encode(prompt, add_special_tokens=False), f"mismatch at rounds={rounds}"
        assert wrapper.apply_chat_template(msgs, tools=TOOLS, reasoning_effort="max") == base.encode(prompt, add_special_tokens=False)
        checked += 1
    for s in adversarial(added, random.Random(0)):
        assert enc.encode(s) == base.encode(s, add_special_tokens=False), f"mismatch on {s!r}"
        checked += 1
    print(f"identical ids on {checked} prompts/strings")

    msgs = conversation(sources, 150)
    prompt = wrapper.apply_chat_template(msgs, tools=TOOLS, tokenize=False, reasoning_effort="max")
    fresh = dsv41._SegmentEncoder(base, base.get_added_vocab())
    t = time.perf_counter(); full = base.encode(prompt, add_special_tokens=False); t_full = time.perf_counter() - t
    t = time.perf_counter(); fresh.encode(prompt); t_cold = time.perf_counter() - t
    nxt = msgs + [{"role": "assistant", "content": "", "reasoning_content": "Next step.", "tool_calls": [{"id": "c999", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
                  {"role": "tool", "tool_call_id": "c999", "content": sources[7][:4500]}]
    prompt2 = wrapper.apply_chat_template(nxt, tools=TOOLS, tokenize=False, reasoning_effort="max")
    t = time.perf_counter(); fresh.encode(prompt2); t_warm = time.perf_counter() - t
    print(f"{len(full):,}-token prompt: whole-prompt tokenize {1000 * t_full:.0f} ms | segmented cold {1000 * t_cold:.0f} ms | "
          f"next turn (warm cache) {1000 * t_warm:.1f} ms")


if __name__ == "__main__":
    main()
