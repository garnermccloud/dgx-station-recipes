# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import array
import collections
import copy
import functools
import json
import re
import threading
from typing import Any

from transformers import BatchEncoding, TokenizersBackend

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from . import deepseek_v41_encoding as _encoding
from .deepseek_v41_encoding import (
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
)

_find_last_user_index = _encoding.find_last_user_index
_last_user_memo = threading.local()


def _memo_find_last_user_index(messages):
    """The reference's find_last_user_index, memoized per message list for one render.

    render_message calls it once per rendered message, rescanning the whole list each time, so a render
    is quadratic in message count. Within one render the list never changes, so caching the answer by
    list identity is exact; encode_messages resets the memo for every render.
    """
    memo = getattr(_last_user_memo, "value", None)
    if memo is not None and memo[0] is messages:
        return memo[1]
    index = _find_last_user_index(messages)
    _last_user_memo.value = (messages, index)
    return index


_encoding.find_last_user_index = _memo_find_last_user_index
_encode_messages_uncached = encode_messages


def encode_messages(*args, **kwargs):  # noqa: F811
    """encode_messages with the find_last_user_index memo reset around each render."""
    _last_user_memo.value = None
    try:
        return _encode_messages_uncached(*args, **kwargs)
    finally:
        _last_user_memo.value = None
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike


_TEXT_PARTS = ("text", "input_text", "output_text")
_IMAGE_PARTS = ("image_url", "input_image", "image_pil")
_MM_URL_PREFIX = "vllm-mm-image:"

# deepseek-recipe 0.2.0 ("explicit special tokens"): only the template's own control tokens encode as special tokens.
# A special-token spelling inside client text (user, system, tool, assistant content, reasoning, tool arguments and
# schemas) encodes as plain text, except the grounding tokens in assistant content and reasoning. Client text is
# escaped before rendering by inserting _ESC after the first character of every added-token spelling (a literal _ESC
# is doubled first), so the rendered prompt splits only at template tokens; _SegmentEncoder unescapes such segments
# and encodes them without the added vocabulary, exactly as the recipe's PromptTokenizer does.
_ESC = "\ue000"
_UNESCAPE = re.compile("\ue000(\ue000?)")
_GROUNDING_TOKENS = ("<｜ref｜>", "<｜/ref｜>", "<｜box｜>", "<｜/box｜>",
                     "<｜point｜>", "<｜/point｜>", "<｜polygon｜>", "<｜/polygon｜>")
# deepseek-recipe maps these request efforts the same way (minimal/low -> 50, medium/high -> 75).
_EFFORT_ALIASES = {"minimal": "low", "medium": "high"}


def _unescape(text: str) -> str:
    """Undo _SpecialSpellingEscaper.escape: a doubled _ESC is one literal _ESC, a lone _ESC is dropped."""
    return _UNESCAPE.sub(lambda m: m.group(1), text)


class _SpecialSpellingEscaper:
    """Escape added-token spellings in client text so they render, and later encode, as plain text."""

    def __init__(self, added_vocab: dict[str, int]) -> None:
        def pattern(tokens):
            return re.compile("|".join(map(re.escape, sorted(tokens, key=len, reverse=True))))

        self._all = pattern(added_vocab)
        self._assistant = pattern(t for t in added_vocab if t not in _GROUNDING_TOKENS)

    def escape(self, text: str, assistant: bool = False) -> str:
        if not isinstance(text, str):
            return text
        pattern = self._assistant if assistant else self._all
        if _ESC not in text and pattern.search(text) is None:
            return text
        text = text.replace(_ESC, _ESC + _ESC)
        while True:
            escaped = pattern.sub(lambda m: m.group(0)[0] + _ESC + m.group(0)[1:], text)
            if escaped == text:
                return text
            text = escaped

    def escape_json(self, value: Any) -> Any:
        """Escape every string, key or value, inside a JSON-like value."""
        if isinstance(value, str):
            return self.escape(value)
        if isinstance(value, list):
            return [self.escape_json(v) for v in value]
        if isinstance(value, dict):
            return {self.escape(k): self.escape_json(v) for k, v in value.items()}
        return value

    def escape_message(self, message: dict[str, Any]) -> None:
        """Escape the client text of one normalized message in place."""
        assistant = message.get("role") == "assistant"
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = self.escape(content, assistant)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    block["text"] = self.escape(block.get("text", ""), assistant)
        if isinstance(message.get("reasoning_content"), str):
            message["reasoning_content"] = self.escape(message["reasoning_content"], assistant)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("arguments") is not None:
                function["arguments"] = self.escape_arguments(function["arguments"])
            if isinstance(function.get("name"), str):
                function["name"] = self.escape(function["name"])

    def escape_arguments(self, arguments: Any) -> Any:
        """Escape tool-call arguments where the encoder will see their text: it decodes a JSON string (twice for a
        double-encoded one) into an object and renders its values, else renders the original string as is. JSON
        escapes such as \\u003c would otherwise decode into a raw special-token spelling after escaping."""
        if not isinstance(arguments, str):
            return self.escape_json(arguments)
        decoded: Any = arguments
        for _ in range(2):
            if not isinstance(decoded, str):
                break
            try:
                decoded = json.loads(decoded)
            except ValueError:
                break
        if isinstance(decoded, dict):
            return json.dumps(self.escape_json(decoded), ensure_ascii=False)
        return self.escape(arguments)


def _prompt_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tool list as deepseek-recipe renders it: name, description (absent -> ""), parameters (absent -> {}),
    nothing else; deferred tools (defer_loading) are left out of the tool prompt until a tool_reference loads them."""
    result = []
    for tool in tools:
        function = tool.get("function", tool)
        if tool.get("defer_loading") or function.get("defer_loading"):
            continue
        parameters = function.get("parameters")
        result.append({"type": "function", "function": {
            "name": function["name"],
            "description": function.get("description") or "",
            "parameters": {} if parameters is None else parameters,
        }})
    return result


def _response_format_schema(response_format: Any) -> Any:
    """The schema deepseek-recipe writes into the prompt for a response format, or None."""
    if response_format is None:
        return None
    if hasattr(response_format, "model_dump"):
        response_format = response_format.model_dump(by_alias=True, exclude_none=True)
    kind = response_format.get("type")
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        json_schema = response_format.get("json_schema") or {}
        return json_schema.get("schema", json_schema.get("json_schema"))
    return None


def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
    escaper: "_SpecialSpellingEscaper | None" = None,
) -> list[dict[str, Any]]:
    """Request messages -> the 1017 reference encoder's OpenAI-style input.

    ``developer`` becomes ``user`` (DeepSeek's own request converter maps it the
    same way). List content stays a list of text / image_url blocks so the
    encoder joins, places and validates them itself; each image is tagged with
    its index in vLLM's multimodal order, which apply_chat_template checks.
    """
    result = [dict(message) for message in copy.deepcopy(messages)]
    image_index = 0
    for message in result:
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {role}")
        if role == "developer":
            message["role"] = "user"
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        content = message.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                part_type = block.get("type")
                if part_type in _TEXT_PARTS:
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif part_type in _IMAGE_PARTS:
                    url = f"{_MM_URL_PREFIX}{image_index}"
                    blocks.append({"type": "image_url", "image_url": {"url": url}})
                    image_index += 1
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            message["content"] = blocks
        if escaper is not None:
            escaper.escape_message(message)
    return result


def _check_image_order(images: list[dict[str, Any]]) -> None:
    """Fail when the encoder's prompt order of images differs from vLLM's.

    The encoder sorts tool results into tool-call order; vLLM binds image data
    in request order. Results that arrive out of call order and carry images
    would bind the wrong image to each placeholder, so they are refused.
    """
    order = [int(image["url"].removeprefix(_MM_URL_PREFIX)) for image in images]
    if order != sorted(order):
        raise ValueError(
            "Tool results that carry images must be sent in the order of their "
            "tool calls."
        )


class _SegmentEncoder:
    """Tokenize a rendered prompt one added-token-delimited segment at a time, caching segments.

    The tokenizer has no normalizer and no added token strips whitespace, so it extracts added tokens
    (leftmost-longest) before pre-tokenizing, and encoding the text between them independently gives the
    same ids as encoding the whole prompt. Agent conversations grow append-only, so a turn re-tokenizes
    only its new segments instead of the whole conversation.
    """

    def __init__(self, tokenizer: HfTokenizer, added_vocab: dict[str, int], max_cached_tokens: int = 32_000_000) -> None:
        self._encode = functools.partial(tokenizer.encode, add_special_tokens=False)
        self._backend = tokenizer.backend_tokenizer
        self._plain_tokenizer = None
        self._added = dict(added_vocab)
        alternatives = sorted(self._added, key=len, reverse=True)
        self._split = re.compile("(" + "|".join(map(re.escape, alternatives)) + ")")
        self._cache: collections.OrderedDict[str, array.array] = collections.OrderedDict()
        self._cached_tokens = 0
        self._max_cached_tokens = max_cached_tokens
        self._lock = threading.Lock()

    def encode(self, prompt: str) -> list[int]:
        """Token ids of prompt, identical to tokenizer.encode(prompt, add_special_tokens=False)."""
        ids: list[int] = []
        for index, part in enumerate(self._split.split(prompt)):
            if index % 2:
                ids.append(self._added[part])
            elif part:
                ids.extend(self._segment(part))
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        """Encode text without the added vocabulary (deepseek-recipe PromptTokenizer's plain regions)."""
        if self._plain_tokenizer is None:
            import tokenizers

            data = json.loads(self._backend.to_str())
            data["added_tokens"] = []
            self._plain_tokenizer = tokenizers.Tokenizer.from_str(json.dumps(data))
        return self._plain_tokenizer.encode(text, add_special_tokens=False).ids

    def _segment(self, text: str) -> array.array:
        """Ids of one added-token-free segment, from the LRU cache when seen before. A segment carrying escaped
        special-token spellings from client text is unescaped and encoded as plain text."""
        with self._lock:
            hit = self._cache.get(text)
            if hit is not None:
                self._cache.move_to_end(text)
                return hit
        if _ESC in text:
            ids = array.array("i", self._encode_plain(_unescape(text)))
        else:
            ids = array.array("i", self._encode(text))
        with self._lock:
            if text not in self._cache:
                self._cache[text] = ids
                self._cached_tokens += len(ids)
                while self._cached_tokens > self._max_cached_tokens and self._cache:
                    _, evicted = self._cache.popitem(last=False)
                    self._cached_tokens -= len(evicted)
        return ids


def get_deepseek_v41_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """Wrap an HF tokenizer with the V4.1 prompt encoder."""
    wrapped = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()
    segments = _SegmentEncoder(tokenizer, added_vocab)
    escaper = _SpecialSpellingEscaper(added_vocab)

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list[ChatCompletionMessageParam],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> str | list[int]:
            # The generic renderer's conversation has already flattened text
            # parts with '\n'. V4.1 encodes the original messages with '\n\n'.
            conversation = _normalize_messages(messages, escaper)
            tools = escaper.escape_json(_prompt_tools(tools or []))
            response_schema = escaper.escape_json(_response_format_schema(kwargs.get("ds_response_format")))
            if tools or response_schema is not None:
                # deepseek-recipe appends both to the first message when it is a system message, else to a new
                # empty system message inserted first.
                system = conversation[0] if conversation and conversation[0]["role"] == "system" else None
                if system is None:
                    system = {"role": "system", "content": ""}
                    conversation.insert(0, system)
                if tools:
                    system["tools"] = tools
                if response_schema is not None:
                    system["response_format"] = response_schema
            if kwargs.get("continue_final_message") and conversation and conversation[-1]["role"] == "assistant":
                conversation[-1]["wo_eos"] = True

            thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
            if "thinking" not in kwargs and "enable_thinking" not in kwargs:
                thinking = True
            effort = _EFFORT_ALIASES.get(kwargs.get("reasoning_effort"), kwargs.get("reasoning_effort"))
            if effort == "none":
                thinking = False
                effort = None
            if effort is None:
                effort = "high"
            if not (
                (type(effort) is int and 1 <= effort <= 100)
                or (isinstance(effort, str) and effort in REASONING_EFFORT_MAPPINGS)
            ):
                raise ValueError(
                    "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, "
                    "or an integer within [1, 100] in chat_template_kwargs"
                )

            prompt, media = encode_messages(
                conversation,
                thinking_mode="thinking" if thinking else "chat",
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=effort,
                return_multi_modal_data=True,
            )
            _check_image_order(media["images"])
            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {
                    key: kwargs[key]
                    for key in ("truncation", "max_length")
                    if key in kwargs
                }
                ids = segments.encode(prompt)
                max_length = tokenizer_kwargs.get("max_length")
                if tokenizer_kwargs.get("truncation") and max_length is not None and len(ids) > max_length:
                    ids = ids[:max_length] if self.truncation_side == "right" else ids[-max_length:]
                return ids
            return prompt

        def encode(self, text, *args, **kwargs):
            # A rendered prompt carrying escaped client text must be encoded segment-wise.
            if isinstance(text, str) and _ESC in text and not args and not kwargs.get("add_special_tokens", False):
                ids = segments.encode(text)
                max_length = kwargs.get("max_length")
                if kwargs.get("truncation") and max_length is not None and len(ids) > max_length:
                    ids = ids[:max_length] if self.truncation_side == "right" else ids[-max_length:]
                return ids
            return super().encode(text, *args, **kwargs)

        def __call__(self, text=None, *args, **kwargs):
            # The chat endpoint renders text (tokenize=False) and tokenizes it here with
            # truncation/max_length and no special tokens: serve that from the segment cache.
            if (
                isinstance(text, str)
                and not args
                and kwargs.get("add_special_tokens") is False
                and set(kwargs) <= {"add_special_tokens", "truncation", "max_length"}
            ):
                ids = segments.encode(text)
                max_length = kwargs.get("max_length")
                if kwargs.get("truncation") and max_length is not None and len(ids) > max_length:
                    ids = ids[:max_length] if self.truncation_side == "right" else ids[-max_length:]
                return BatchEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)})
            return super().__call__(text, *args, **kwargs)

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v41_tokenizer, (tokenizer,)

    _DeepseekV41Tokenizer.__name__ = f"DSV41{tokenizer.__class__.__name__}"
    wrapped.__class__ = _DeepseekV41Tokenizer
    return wrapped


class DeepseekV41Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v41_tokenizer(tokenizer))
