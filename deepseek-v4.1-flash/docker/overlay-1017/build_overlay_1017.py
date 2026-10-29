"""Build the DeepSeek-V4.1-Flash-1017 frontend overlay for vLLM v0.31.0.

Four files, each derived from its original by exact, asserted replacements:
  tokenizers/deepseek_v41_encoding.py  the 1017 checkpoint's encoding/encoding.py, plus the "xhigh" alias
  tokenizers/deepseek_v41.py           feeds that encoder OpenAI-style content blocks instead of pre-flattened text,
                                       and tokenizes prompts per cached special-token-delimited segment
  entrypoints/anthropic/serving.py     /v1/messages conversion aligned with DeepSeek's own (deepseek-recipe 0.1.1)
  renderers/deepseek_v4.py              passes the request response_format to the tokenizer wrapper as ds_response_format (r2)

    build_overlay_1017.py <orig_dir> <out_dir>
      orig_dir: deepseek_v41.py, serving.py (from the image) and encoding_1017.py (from the checkpoint)
"""
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, what: str) -> str:
    """Replace exactly one occurrence of old, failing loudly when the source drifted."""
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{what}: expected 1 match, found {count}")
    return text.replace(old, new)


def build_encoding(src: str) -> str:
    """The 1017 reference encoder with DeepSeek's production "xhigh" alias added."""
    header = ("# vLLM overlay: deepseek-ai/DeepSeek-V4.1-Flash-1017 encoding/encoding.py (HF revision 1bae0e10), verbatim\n"
              "# except REASONING_EFFORT_MAPPINGS gains \"xhigh\" -> 75, the value DeepSeek's own request converter\n"
              "# (deepseek-recipe, encoding/src/v4/dsv41.rs) and this image's encoder both use.\n")
    src = replace_once(src, '''    "high": 75,
    "max": 100,''', '''    "high": 75,
    "xhigh": 75,
    "max": 100,''', "effort mapping")
    return header + src


NORMALIZE_OLD = '''def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    result = [dict(message) for message in copy.deepcopy(messages)]
    for message in result:
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {role}")
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    parts.append(block.get("text", ""))
                elif part_type in ("image_url", "input_image", "image_pil"):
                    parts.append(IMAGE_PLACEHOLDER)
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            message["content"] = "\\n\\n".join(parts)
    return result
'''

NORMALIZE_NEW = '''_TEXT_PARTS = ("text", "input_text", "output_text")
_IMAGE_PARTS = ("image_url", "input_image", "image_pil")
_MM_URL_PREFIX = "vllm-mm-image:"


def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
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
'''

ENCODE_OLD = '''            prompt = encode_messages(
                conversation,
                thinking_mode="thinking" if thinking else "chat",
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=effort,
            )
'''

ENCODE_NEW = '''            prompt, media = encode_messages(
                conversation,
                thinking_mode="thinking" if thinking else "chat",
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=effort,
                return_multi_modal_data=True,
            )
            _check_image_order(media["images"])
'''


SEGMENT_ENCODER = '''

class _SegmentEncoder:
    """Tokenize a rendered prompt one added-token-delimited segment at a time, caching segments.

    The tokenizer has no normalizer and no added token strips whitespace, so it extracts added tokens
    (leftmost-longest) before pre-tokenizing, and encoding the text between them independently gives the
    same ids as encoding the whole prompt. Agent conversations grow append-only, so a turn re-tokenizes
    only its new segments instead of the whole conversation.
    """

    def __init__(self, tokenizer: HfTokenizer, added_vocab: dict[str, int], max_cached_tokens: int = 32_000_000) -> None:
        self._encode = functools.partial(tokenizer.encode, add_special_tokens=False)
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

    def _segment(self, text: str) -> array.array:
        """Ids of one added-token-free segment, from the LRU cache when seen before."""
        with self._lock:
            hit = self._cache.get(text)
            if hit is not None:
                self._cache.move_to_end(text)
                return hit
        ids = array.array("i", self._encode(text))
        with self._lock:
            if text not in self._cache:
                self._cache[text] = ids
                self._cached_tokens += len(ids)
                while self._cached_tokens > self._max_cached_tokens and self._cache:
                    _, evicted = self._cache.popitem(last=False)
                    self._cached_tokens -= len(evicted)
        return ids


def get_deepseek_v41_tokenizer('''


def build_tokenizer(src: str) -> str:
    """vLLM's V4.1 tokenizer wrapper: reference encoder fed content blocks, prompts tokenized per cached segment."""
    src = replace_once(src, '''from .deepseek_v41_encoding import (
    IMAGE_PLACEHOLDER,
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
)''', '''from .deepseek_v41_encoding import (
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
)''', "encoder import")
    src = replace_once(src, "import copy\n", "import array\nimport collections\nimport copy\nimport functools\nimport re\nimport threading\n", "imports")
    src = replace_once(src, NORMALIZE_OLD, NORMALIZE_NEW, "normalizer")
    src = replace_once(src, ENCODE_OLD, ENCODE_NEW, "encode call")
    src = replace_once(src, "\n\ndef get_deepseek_v41_tokenizer(", SEGMENT_ENCODER, "segment encoder class")
    src = replace_once(src, '''    added_vocab = tokenizer.get_added_vocab()
''', '''    added_vocab = tokenizer.get_added_vocab()
    segments = _SegmentEncoder(tokenizer, added_vocab)
''', "segment encoder instance")
    src = replace_once(src, '''                return self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)
''', '''                if not tokenizer_kwargs:
                    return segments.encode(prompt)
                return self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)
''', "tokenize call")
    src = replace_once(src, '''        def num_special_tokens_to_add(self) -> int:
''', '''        def __call__(self, text=None, *args, **kwargs):
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
''', "cached __call__")
    src = replace_once(src, "from transformers import TokenizersBackend\n", "from transformers import BatchEncoding, TokenizersBackend\n", "BatchEncoding import")
    src = replace_once(src, '''from .deepseek_v41_encoding import (
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
)''', '''from . import deepseek_v41_encoding as _encoding
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
        _last_user_memo.value = None''', "find_last_user_index memo")
    return src


TOOL_RESULT_OLD_START = '''    @classmethod
    def _convert_user_tool_result(
        cls, block, openai_messages: list[dict[str, Any]]
    ) -> None:'''
TOOL_RESULT_OLD_END = '''    @classmethod
    def _build_base_request('''

TOOL_RESULT_NEW = '''    @classmethod
    def _convert_user_tool_result(
        cls, block, openai_messages: list[dict[str, Any]]
    ) -> None:
        """Convert a user tool_result into one tool message, as DeepSeek does.

        Blocks keep their order inside the result: text, images (rendered inside
        the <tool_result>), tool references as their tool name, documents as
        DeepSeek's "[Unsupported Document]" placeholder. The encoder joins them
        with blank lines.
        """
        parts: list[dict[str, Any]] = []
        if isinstance(block.content, str):
            parts.append({"type": "text", "text": block.content})
        elif isinstance(block.content, list):
            for item in block.content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    parts.append({"type": "text", "text": item.get("text", "")})
                elif item_type == "image":
                    url = cls._convert_image_source_to_url(item.get("source", {}))
                    if url:
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                elif item_type == "tool_reference":
                    ref_name = item.get("tool_name") or item.get("name")
                    if ref_name:
                        parts.append({"type": "text", "text": ref_name})
                elif item_type == "document":
                    parts.append({"type": "text", "text": "[Unsupported Document]"})

        content: str | list[dict[str, Any]] = parts
        if not parts:
            content = ""
        elif len(parts) == 1 and parts[0]["type"] == "text":
            content = parts[0]["text"]
        openai_messages.append(
            {
                "role": "tool",
                "tool_call_id": block.tool_use_id or "",
                "content": content,  # type: ignore[dict-item]
            }
        )

'''


def build_anthropic(src: str) -> str:
    """vLLM's /v1/messages conversion, aligned with deepseek-recipe's for DeepSeek V4.1."""
    src = replace_once(src, '''    @classmethod
    def _should_merge_inline_system(cls, online_renderer: OnlineRenderer) -> bool:
        """Probe the chat templates the renderer applies, not the CLI override.

        Both the tool and non-tool templates are checked since they may differ.
        """
''', '''    @classmethod
    def _should_merge_inline_system(cls, online_renderer: OnlineRenderer) -> bool:
        """Never merge: inline system messages become system reminders in place.

        DeepSeek's own conversion (deepseek-recipe) renders each inline system
        message as a user turn wrapped in <system-reminder> tags where it
        appears, which also keeps the prompt prefix stable for caching.
        """
        return False
''', "merge probe")
    src = replace_once(src, '''            openai_messages.append({"role": "system", "content": "".join(system_parts)})''',
                       '''            openai_messages.append({"role": "system", "content": "\\n\\n".join(system_parts)})''',
                       "system join")
    src = replace_once(src, '''                text = cls._extract_system_text(msg)
                if text:
                    openai_messages.append({"role": "system", "content": text})
                continue''', '''                text = cls._extract_system_text(msg)
                if text:
                    openai_messages.append(
                        {
                            "role": "user",
                            "content": f"<system-reminder>\\n{text}\\n</system-reminder>",
                        }
                    )
                continue''', "inline system")
    src = replace_once(src, '''            openai_msg["reasoning"] = "".join(reasoning_parts)''',
                       '''            openai_msg["reasoning"] = "\\n\\n".join(reasoning_parts)''', "thinking join")
    src = replace_once(src, '''                if block.text.startswith("x-anthropic-billing-header"):
                    continue
                parts.append(block.text)
        return "".join(parts) if parts else None''', '''                if block.text.startswith("x-anthropic-billing-header"):
                    continue
                parts.append(block.text)
        return "\\n\\n".join(parts) if parts else None''', "inline system join")
    start = src.index(TOOL_RESULT_OLD_START)
    end = src.index(TOOL_RESULT_OLD_END, start)
    return src[:start] + TOOL_RESULT_NEW + src[end:]


def main() -> None:
    """Write the three overlay files under out_dir/vllm/."""
    orig, out = Path(sys.argv[1]), Path(sys.argv[2])
    targets = {
        "tokenizers/deepseek_v41_encoding.py": build_encoding((orig / "encoding_1017.py").read_text()),
        "tokenizers/deepseek_v41.py": build_tokenizer((orig / "deepseek_v41.py").read_text()),
        "entrypoints/anthropic/serving.py": build_anthropic((orig / "serving.py").read_text()),
    }
    for rel, text in targets.items():
        path = out / "vllm" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
