"""Fix #4 check: special-token spellings in client text encode as plain text, exactly as deepseek-recipe 0.2.0 does.

Seeded conversations put added-token spellings (sampled from the whole added vocabulary), grounding tokens, literal
U+E000 escape characters and near-miss fragments into every client text field: system, user, tool results, assistant
content, reasoning, tool-call arguments and tool descriptions. For each one, the overlay wrapper's three tokenizing
entry points must give deepseek-recipe's ids (recipe Chat Completions conversion + DeepseekV41Encoding.encode):
apply_chat_template(tokenize=True), the chat endpoint's sequence (tokenize=False, then the wrapper call with
truncation/max_length), and wrapper.encode on the rendered text. The unescaped rendered text must equal the 1017
reference encoder's rendering of the unescaped messages.

    r2_special_check.py <tokenizer_dir> <recipe_tokenizer.json> <reference_encoding_dir> [--n 1500]
"""

import argparse
import copy
import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import deepseek_recipe as dsr
import vllm.tokenizers.deepseek_v41 as dsv41
from transformers import AutoTokenizer

GROUNDING = ["<｜ref｜>", "<｜/ref｜>", "<｜box｜>", "<｜/box｜>", "<｜point｜>", "<｜/point｜>"]
PLAIN = ["fix the bug", " ", "\n", "\n\n", "中文", "🙂", "x = 1", "<", "｜", "<｜Use", "User｜>", "</think", "think>",
         "<|EOT|>", "", "", "<<", ">>", "\"", "\\", "</tool_result>", "<tool_result>"]
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]


def text(rng: random.Random, added: list[str], extra: list[str] = ()) -> str:
    return "".join(rng.choice(PLAIN + [rng.choice(added)] * 3 + list(extra)) for _ in range(rng.randint(1, 25)))


def conversation(rng: random.Random, added: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tools = copy.deepcopy(TOOLS)
    tools[0]["function"]["description"] = "Run a shell command " + text(rng, added)
    msgs: list[dict[str, Any]] = [{"role": "system", "content": text(rng, added)}]
    for turn in range(rng.randint(1, 3)):
        msgs.append({"role": "user", "content": text(rng, added)})
        for step in range(rng.randint(0, 2)):
            cid = f"c{turn}_{step}"
            msgs.append({"role": "assistant", "content": text(rng, added, GROUNDING) if rng.random() < 0.5 else "",
                         "reasoning_content": text(rng, added, GROUNDING),
                         "tool_calls": [{"id": cid, "type": "function", "function": {
                             "name": "bash", "arguments": json.dumps({"cmd": text(rng, added)}, ensure_ascii=rng.random() < 0.3)}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": text(rng, added)})
        msgs.append({"role": "assistant", "content": text(rng, added, GROUNDING), "reasoning_content": text(rng, added)})
    msgs.append({"role": "user", "content": text(rng, added)})
    return msgs, tools


def recipe_ids(msgs: list[dict[str, Any]], tools: list[dict[str, Any]], effort: str, enc: Any) -> list[int]:
    body = {"model": "DeepSeek-V4.1-Flash", "messages": msgs, "tools": tools, "reasoning_effort": effort}
    conv = dsr.ChatCompletionRequest(json.dumps(body)).convert(dsr.ConversionOptions()).conversation
    return enc.encode(conv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tok_dir")
    ap.add_argument("recipe_tokenizer")
    ap.add_argument("ref_dir", type=Path)
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("ref_encoding", a.ref_dir / "encoding.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    base = AutoTokenizer.from_pretrained(a.tok_dir)
    wrapper = dsv41.DeepseekV41Tokenizer.from_pretrained(a.tok_dir)
    enc = dsr.DeepseekV41Encoding().with_tokenizer(dsr.Tokenizer.from_file(a.recipe_tokenizer))
    added = [t for t in base.get_added_vocab() if t not in ("<｜deepseek_image｜>",)]
    rng = random.Random(4)
    checks = differs_from_plain_tokenizer = 0
    for k in range(a.n):
        msgs, tools = conversation(rng, added)
        effort = rng.choice(["low", "high", "max"])
        want = recipe_ids(copy.deepcopy(msgs), tools, effort, enc)
        ids = wrapper.apply_chat_template(copy.deepcopy(msgs), tools=tools, reasoning_effort=effort)
        if ids != want:
            i = next((j for j in range(min(len(ids), len(want))) if ids[j] != want[j]), min(len(ids), len(want)))
            raise SystemExit(f"case {k}: apply_chat_template(tokenize=True) != recipe at token {i}:\n"
                             f"  overlay {ids[max(0, i - 4):i + 6]} {base.convert_ids_to_tokens(ids[max(0, i - 4):i + 6])}\n"
                             f"  recipe  {want[max(0, i - 4):i + 6]} {base.convert_ids_to_tokens(want[max(0, i - 4):i + 6])}\n"
                             f"  overlay text {base.decode(ids[max(0, i - 15):i + 10])!r}\n"
                             f"  recipe text  {base.decode(want[max(0, i - 15):i + 10])!r}")
        prompt = wrapper.apply_chat_template(copy.deepcopy(msgs), tools=tools, tokenize=False, reasoning_effort=effort)
        call = wrapper(prompt, add_special_tokens=False, truncation=True, max_length=524289)["input_ids"]
        assert call == want, f"case {k}: chat-endpoint tokenizer call != recipe"
        assert wrapper.encode(prompt, add_special_tokens=False) == want, f"case {k}: encode != recipe"
        cut = wrapper(prompt, add_special_tokens=False, truncation=True, max_length=len(want) // 2)["input_ids"]
        assert cut == want[: len(want) // 2], f"case {k}: truncation"
        conv = dsv41._normalize_messages(copy.deepcopy(msgs))
        conv[0]["tools"] = dsv41._prompt_tools(tools)
        assert dsv41._unescape(prompt) == ref.encode_messages(conv, thinking_mode="thinking", reasoning_effort=effort), \
            f"case {k}: unescaped render != 1017 reference"
        differs_from_plain_tokenizer += ids != base.encode(dsv41._unescape(prompt), add_special_tokens=False)
        checks += 5
    print(f"fix #4: {a.n} conversations with special-token spellings in every client field, {checks} checks: "
          f"overlay ids == deepseek-recipe 0.2.0 ids on all; {differs_from_plain_tokenizer} of {a.n} differ from "
          f"tokenizing the whole rendered prompt (the old behaviour)")


if __name__ == "__main__":
    main()
