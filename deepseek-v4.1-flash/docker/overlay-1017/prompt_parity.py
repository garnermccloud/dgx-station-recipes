"""Prompt parity for the DeepSeek-V4.1-Flash-1017 lane: render each test request three ways and diff them.

  V  what the live vLLM server renders (POST /tokenize; Anthropic requests go through vLLM's own
     AnthropicServingMessages conversion first, exactly as /v1/messages does)
  R  DeepSeek's production stack: deepseek-recipe request conversion + its V4.1 renderer (released before 1017)
  E  the target: deepseek-recipe request conversion, then the 1017 checkpoint's encoding/encoding.py

    prompt_parity.py <ref_encoding_dir> <tokenizer.json> <key_file> [--url http://127.0.0.1:30003]
"""
import argparse
import copy
import importlib.util
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import deepseek_recipe as dsr
from tokenizers import Tokenizer

MERGE_INLINE_SYSTEM = True  # what the unpatched server detects; the 1017 overlay never merges
PNG_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}, "timeout": {"type": "integer"}},
                         "required": ["cmd"]}}},
         {"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
ANTHROPIC_TOOLS = [{"name": t["function"]["name"], "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"]} for t in TOOLS]


@dataclass(frozen=True)
class Case:
    """One request: its API ('chat' or 'messages') and body."""
    name: str
    api: str
    body: dict[str, Any]


def call(i: int, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """An OpenAI assistant tool call."""
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def agent_loop() -> list[dict[str, Any]]:
    """Codex-style loop: task, two tool steps with reasoning, ending on a tool result."""
    msgs = [{"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Find the failing test and fix it."},
            {"role": "assistant", "content": "", "reasoning_content": "Run the tests first.", "tool_calls": [call(1, "bash", {"cmd": "pytest -q"})]},
            {"role": "tool", "tool_call_id": "call_1", "content": "1 failed: test_parse"},
            {"role": "assistant", "content": "", "reasoning_content": "Read the test.", "tool_calls": [call(2, "read_file", {"path": "test_parse.py"})]},
            {"role": "tool", "tool_call_id": "call_2", "content": "def test_parse(): assert parse('1') == 1"}]
    return msgs


def chat_cases() -> list[Case]:
    """Chat Completions requests covering what our clients send."""
    loop = agent_loop()
    two_turns = agent_loop() + [
        {"role": "assistant", "content": "Fixed the off-by-one.", "reasoning_content": "Patch applied."},
        {"role": "user", "content": "Now run the whole suite."}]
    parallel = [{"role": "user", "content": "Check both files."},
                {"role": "assistant", "content": "", "reasoning_content": "Read both.",
                 "tool_calls": [call(1, "read_file", {"path": "a.py"}), call(2, "read_file", {"path": "b.py"})]},
                {"role": "tool", "tool_call_id": "call_2", "content": "B"},
                {"role": "tool", "tool_call_id": "call_1", "content": "A"}]
    return [
        Case("simple system+user", "chat", {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 17*19?"}]}),
        Case("user only", "chat", {"messages": [{"role": "user", "content": "Hello"}]}),
        Case("agent loop, tools", "chat", {"messages": loop, "tools": TOOLS}),
        Case("agent loop, no tools declared", "chat", {"messages": agent_loop()}),
        Case("two turns, tools", "chat", {"messages": two_turns, "tools": TOOLS}),
        Case("two turns, no tools declared", "chat", {"messages": two_turns}),
        Case("developer role + tools", "chat", {"messages": [{"role": "developer", "content": "Follow AGENTS.md."}, {"role": "user", "content": "List files."}], "tools": TOOLS}),
        Case("system + developer + user", "chat", {"messages": [{"role": "system", "content": "Base."}, {"role": "developer", "content": "Dev rules."}, {"role": "user", "content": "Go."}]}),
        Case("content as text parts", "chat", {"messages": [{"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]}]}),
        Case("tool result as text parts", "chat", {"messages": loop[:3] + [{"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}]}], "tools": TOOLS}),
        Case("mid-conversation system", "chat", {"messages": [{"role": "system", "content": "Base."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, {"role": "system", "content": "Now answer in French."}, {"role": "user", "content": "How are you?"}]}),
        Case("thinking off", "chat", {"messages": [{"role": "user", "content": "Hi"}], "chat_template_kwargs": {"thinking": False}}),
        Case("effort low", "chat", {"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "low"}),
        Case("effort max", "chat", {"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "max"}),
        Case("effort xhigh", "chat", {"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "xhigh"}),
        Case("parallel calls, results out of order", "chat", {"messages": parallel, "tools": TOOLS}),
        Case("assistant text + tool call", "chat", {"messages": [{"role": "user", "content": "Run it."}, {"role": "assistant", "content": "Running now.", "reasoning_content": "ok", "tool_calls": [call(1, "bash", {"cmd": "make", "timeout": 60})]}, {"role": "tool", "tool_call_id": "call_1", "content": "done"}], "tools": TOOLS}),
        Case("tool_choice required", "chat", {"messages": [{"role": "user", "content": "List files."}], "tools": TOOLS, "tool_choice": "required"}),
        Case("tool result with image", "chat", {"messages": loop[:3] + [{"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "saved"}, {"type": "image_url", "image_url": {"url": PNG_URL}}]}], "tools": TOOLS}),
        Case("user text + image", "chat", {"messages": [{"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": PNG_URL}}]}]}),
        Case("json_object response format", "chat", {"messages": [{"role": "user", "content": "Give JSON."}], "response_format": {"type": "json_object"}}),
    ]


def messages_cases() -> list[Case]:
    """Anthropic Messages requests shaped like Claude Code's."""
    base = {"model": "DeepSeek-V4.1-Flash", "max_tokens": 64000, "tools": ANTHROPIC_TOOLS,
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=1.0; hash=abc"},
                       {"type": "text", "text": "You are Claude Code."}, {"type": "text", "text": "Tool rules."}]}
    turn = [{"role": "user", "content": [{"type": "text", "text": "<system-reminder>\nContext.\n</system-reminder>"}, {"type": "text", "text": "Fix the test."}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "Run tests.", "signature": "sig"},
                                              {"type": "text", "text": "Running the tests."},
                                              {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "pytest -q"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "1 failed"}]}]
    return [
        Case("claude code turn, thinking on", "messages", {**base, "messages": turn, "thinking": {"type": "enabled", "budget_tokens": 8000}}),
        Case("claude code turn, no thinking field", "messages", {**base, "messages": turn}),
        Case("claude code, inline system message", "messages", {**base, "messages": turn + [{"role": "assistant", "content": [{"type": "text", "text": "One failure."}]}, {"role": "system", "content": "Todo list changed."}, {"role": "user", "content": "Continue."}]}),
        Case("claude code, tool_result blocks + error", "messages", {**base, "messages": turn[:2] + [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True, "content": [{"type": "text", "text": "exit 1"}, {"type": "text", "text": "stderr: boom"}]}]}]}),
        Case("claude code, tool_result image + tool_reference", "messages", {**base, "messages": turn[:2] + [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "shot saved"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_URL.split(",", 1)[1]}}, {"type": "tool_reference", "tool_name": "read_file"}]}]}]}),
        Case("claude code, two thinking blocks", "messages", {**base, "messages": [turn[0], {"role": "assistant", "content": [{"type": "thinking", "thinking": "First.", "signature": "s1"}, {"type": "thinking", "thinking": "Second.", "signature": "s2"}, {"type": "text", "text": "Done."}]}, {"role": "user", "content": "Next."}]}),
        Case("claude code, thinking disabled", "messages", {**base, "messages": turn[:1], "thinking": {"type": "disabled"}}),
    ]


def load_ref_encoding(ref_dir: Path) -> Any:
    """Import the 1017 checkpoint's encoding.py under a private name."""
    spec = importlib.util.spec_from_file_location("ref1017_encoding", ref_dir / "encoding.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {ref_dir / 'encoding.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dsr_convert(case: Case) -> Any:
    """deepseek-recipe's request conversion (the DeepSeek API's protocol handling)."""
    body = {"model": "DeepSeek-V4.1-Flash", **copy.deepcopy(case.body)}
    for m in body.get("messages", []):
        if case.api == "chat" and m.get("role") == "developer":
            m["role"] = "user"
    kwargs = body.pop("chat_template_kwargs", None) or {}
    if kwargs.get("thinking") is False:
        body["thinking"] = {"type": "disabled"}
    req = dsr.ChatCompletionRequest(body) if case.api == "chat" else dsr.MessagesRequest(body)
    return req.convert(dsr.ConversionOptions()).conversation


def conversation_to_openai(conv: Any) -> tuple[list[dict[str, Any]], str, Any]:
    """deepseek-recipe Conversation -> OpenAI-style messages for encoding.py, plus thinking mode and effort."""
    out: list[dict[str, Any]] = []
    for m in conv.messages:
        msg: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.reasoning_content is not None:
            msg["reasoning_content"] = m.reasoning_content
        if m.tool_calls:
            msg["tool_calls"] = [{"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}} for t in m.tool_calls]
        if m.tool_call_id is not None:
            msg["tool_call_id"] = m.tool_call_id
        out.append(msg)
    if conv.tools:
        tools = [{"type": "function", "function": {k: v for k, v in (("name", t.name), ("description", t.description), ("parameters", t.parameters)) if v is not None}} for t in conv.tools]
        if out and out[0]["role"] == "system":
            out[0]["tools"] = tools
        else:
            out.insert(0, {"role": "system", "content": "", "tools": tools})
    if conv.response_format == "json_object" and out:
        out[0]["response_format"] = {"type": "json_object"}
    return out, ("thinking" if conv.thinking_mode else "chat"), conv.reasoning_effort


def vllm_render_offline(body: dict[str, Any], wrapper: Any) -> str:
    """Render through vLLM's DeepSeek V4.1 tokenizer wrapper directly, with the server's default effort."""
    kwargs = dict(body.get("chat_template_kwargs") or {})
    if body.get("reasoning_effort") is not None:
        kwargs["reasoning_effort"] = body["reasoning_effort"]
    kwargs.setdefault("reasoning_effort", "high")
    try:
        return wrapper.apply_chat_template(body["messages"], tools=body.get("tools"), tokenize=False, add_generation_prompt=True, **kwargs)
    except Exception as exc:
        return f"<render error {type(exc).__name__}: {exc}>"


def vllm_chat_body(case: Case) -> dict[str, Any]:
    """The chat-completions body vLLM renders for this case (Anthropic converted the way /v1/messages does)."""
    if case.api == "chat":
        return copy.deepcopy(case.body)
    from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
    req = AnthropicServingMessages.to_chat_completion_request(AnthropicMessagesRequest.model_validate(case.body), merge_inline_system=MERGE_INLINE_SYSTEM)
    body = json.loads(req.model_dump_json(exclude_none=True))
    keep = {k: body[k] for k in ("messages", "tools", "chat_template_kwargs", "reasoning_effort") if k in body}
    return keep


def vllm_render(url: str, key: str, body: dict[str, Any], tok: Tokenizer) -> str:
    """Render through the live server's /tokenize, then decode the ids back to text."""
    payload = {"model": "DeepSeek-V4.1-Flash", "add_generation_prompt": True, **body}
    effort = payload.pop("reasoning_effort", None)
    if effort is not None:
        payload.setdefault("chat_template_kwargs", {})["reasoning_effort"] = effort
    payload.pop("tool_choice", None)
    payload.pop("response_format", None)
    req = urllib.request.Request(f"{url}/tokenize", json.dumps(payload).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        ids = json.loads(urllib.request.urlopen(req, timeout=60).read())["tokens"]
    except urllib.error.HTTPError as e:
        return f"<HTTP {e.code}: {e.read()[:300]!r}>"
    return tok.decode(ids, skip_special_tokens=False)


def first_diff(a: str, b: str, label_a: str, label_b: str) -> str:
    """The first point where two prompts diverge, with context."""
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    return (f"      at char {i} ({len(a)} vs {len(b)} chars)\n"
            f"      {label_a}: {a[max(0, i - 90):i + 150]!r}\n      {label_b}: {b[max(0, i - 90):i + 150]!r}")


def main() -> None:
    """Render every case three ways and report mismatches against the 1017 target."""
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir", type=Path)
    ap.add_argument("tokenizer", type=Path)
    ap.add_argument("key_file", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:30003")
    ap.add_argument("--server-effort", default="high", help="the server's default reasoning_effort")
    ap.add_argument("--full-diff", action="store_true", help="print every differing segment, split at special tokens")
    ap.add_argument("--only", default="", help="run only cases whose name contains this text")
    ap.add_argument("--offline", default="", help="tokenizer dir: render V through the (overlaid) wrapper instead of the server")
    a = ap.parse_args()
    ref = load_ref_encoding(a.ref_dir)
    tok = Tokenizer.from_file(str(a.tokenizer))
    key = a.key_file.read_text().strip()
    global MERGE_INLINE_SYSTEM
    wrapper = None
    if a.offline:
        from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
        wrapper = DeepseekV41Tokenizer.from_pretrained(a.offline)
        MERGE_INLINE_SYSTEM = False
    for case in [c for c in chat_cases() + messages_cases() if a.only in c.name]:
        try:
            conv = dsr_convert(case)
            r = dsr.DeepseekV41Encoding().render_conversation(conv).prompt
            msgs, mode, effort = conversation_to_openai(conv)
            e = ref.encode_messages(msgs, thinking_mode=mode, reasoning_effort=effort if effort is not None else a.server_effort)
        except Exception as exc:  # report and keep going: a conversion error is itself a finding
            print(f"[{case.api:8s}] {case.name:40s} reference error: {type(exc).__name__}: {exc}")
            continue
        body = vllm_chat_body(case)
        v = vllm_render_offline(body, wrapper) if wrapper else vllm_render(a.url, key, body, tok)
        status = "V=E" if v == e else "V!=E"
        status += "  R=E" if r == e else "  R!=E"
        print(f"[{case.api:8s}] {case.name:40s} {status}")
        if v != e:
            print(first_diff(v, e, "vLLM", "1017"))
            if a.full_diff:
                import difflib, re
                split = lambda t: re.split(r"(?=<｜)", t)
                for line in difflib.unified_diff(split(v), split(e), "vLLM", "1017", n=0, lineterm=""):
                    print("        " + repr(line)[:400])
        if r != e:
            print(first_diff(r, e, "recipe", "1017"))


if __name__ == "__main__":
    main()
