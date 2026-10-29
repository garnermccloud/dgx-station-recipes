"""ds41 fastpath components "ds41_stop" and "ds41_sampling": deepseek-recipe 0.2.0 serving semantics in the API server.

ds41_stop (vllm.v1.engine.detokenizer): stop strings match only in the answer, never in reasoning. deepseek-recipe's
output parser matches stop sequences in answer content and JSON output only; vLLM matches them anywhere in the output
text, so a stop string the model writes while reasoning ends the request early. For a request whose prompt ends with
<think> (thinking mode), each detokenizer update is split at the first </think>: stop strings are not checked before
it, and afterwards a match may not start before the end of </think>. Requests without stop strings are untouched.

ds41_sampling (vllm.entrypoints.openai.chat_completion.protocol): thinking-mode requests sample at temperature 1.0 and
top_p = max(requested or the server default, 0.95), whatever the client sent, and non-thinking requests at top_p 1.0
(deepseek-recipe 0.2.0 ModelVersion::resolve_top_p for V4.1; the recipe's fallback for an unspecified top_p in thinking
mode is 1.0, ours is the server default, 0.95, by operator decision). /v1/chat/completions
and /v1/messages both build their SamplingParams through ChatCompletionRequest.to_sampling_params. Thinking mode is
decided exactly as the V4.1 tokenizer wrapper and parser decide it.

Env: DS41_THINK_START_ID (default 128821, <think>), DS41_THINK_END_ID (default 128822, </think>),
DS41_NONTHINK_TOP_P (default 1.0).
No torch or vllm import happens at module import time.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastpath_hook import register  # noqa: E402

THINK_START_ID = int(os.environ.get("DS41_THINK_START_ID", "128821"))
THINK_END_ID = int(os.environ.get("DS41_THINK_END_ID", "128822"))
THINKING_TOP_P_FLOOR = 0.95
_REASONING = "reasoning"
# Earliest output-text offset a stop string may start at, set only while an answer-phase update runs. The output
# processor calls update() synchronously, so a module-level slot is enough.
_stop_floor: list[int | None] = [None]


def _windowed_check(check_stop_strings, output_text: str, new_char_count: int, stop: list[str],
                    include_in_output: bool):
    """check_stop_strings restricted to text at or after _stop_floor, offsets mapped back to output_text."""
    floor = _stop_floor[0]
    if floor is None or not stop:
        return check_stop_strings(output_text, new_char_count, stop, include_in_output)
    start = max(floor, len(output_text) - new_char_count - max(len(s) for s in stop) + 1, 0)
    window = output_text[start:]
    hit = check_stop_strings(window, min(new_char_count, len(window)), stop, include_in_output)
    if hit is None:
        return None
    stop_string, offset = hit
    return stop_string, offset if offset == -1 else offset + start


@register("ds41_stop", "vllm.v1.engine.detokenizer")
def patch_detokenizer(module) -> None:
    base = module.BaseIncrementalDetokenizer
    original_init, original_update = base.__init__, base.update
    original_check = module.check_stop_strings

    def __init__(self, request) -> None:
        original_init(self, request)
        prompt = getattr(request, "prompt_token_ids", None) or []
        self._ds41_stop_state = _REASONING if self.stop and prompt and prompt[-1] == THINK_START_ID else None

    def _update_without_stop(self, token_ids: list[int], stop_terminated: bool) -> None:
        saved, self.stop = self.stop, []
        try:
            original_update(self, token_ids, stop_terminated)
        finally:
            self.stop = saved

    def update(self, new_token_ids: list[int], stop_terminated: bool):
        state = getattr(self, "_ds41_stop_state", None)
        if state is None or not new_token_ids:
            return original_update(self, new_token_ids, stop_terminated)
        if state == _REASONING:
            if THINK_END_ID not in new_token_ids:
                _update_without_stop(self, new_token_ids, stop_terminated)
                return None
            cut = new_token_ids.index(THINK_END_ID) + 1
            head, new_token_ids = new_token_ids[:cut], new_token_ids[cut:]
            _update_without_stop(self, head, stop_terminated and not new_token_ids)
            self._ds41_stop_state = len(self.output_text)
            if not new_token_ids:
                return None
        _stop_floor[0] = self._ds41_stop_state
        try:
            return original_update(self, new_token_ids, stop_terminated)
        finally:
            _stop_floor[0] = None

    def check_stop_strings(output_text, new_char_count, stop, include_in_output):
        return _windowed_check(original_check, output_text, new_char_count, stop, include_in_output)

    base.__init__ = __init__
    base.update = update
    module.check_stop_strings = check_stop_strings


def is_thinking(request) -> bool:
    """Thinking mode of a ChatCompletionRequest, decided as build_chat_params, the V4.1 tokenizer wrapper and the
    DeepSeek V4 parser decide it: chat_template_kwargs thinking/enable_thinking (reasoning_effort sets
    enable_thinking when the client did not), default on, and reasoning_effort "none" turns it off."""
    kwargs = dict(request.chat_template_kwargs or {})
    effort = request.reasoning_effort
    if effort is not None and "enable_thinking" not in kwargs:
        kwargs["enable_thinking"] = effort != "none"
    thinking = True
    if "thinking" in kwargs or "enable_thinking" in kwargs:
        thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
    if (effort if effort is not None else kwargs.get("reasoning_effort")) == "none":
        thinking = False
    return thinking


@register("ds41_sampling", "vllm.entrypoints.openai.chat_completion.protocol")
def patch_sampling(module) -> None:
    request_class = module.ChatCompletionRequest
    original = request_class.to_sampling_params
    nonthinking_top_p = float(os.environ.get("DS41_NONTHINK_TOP_P", "1.0"))

    def to_sampling_params(self, max_tokens, default_sampling_params):
        if is_thinking(self):
            requested = self.top_p if self.top_p is not None else default_sampling_params.get("top_p", 1.0)
            self.temperature = 1.0
            self.top_p = max(float(requested), THINKING_TOP_P_FLOOR)
        else:
            self.top_p = nonthinking_top_p
        return original(self, max_tokens, default_sampling_params)

    request_class.to_sampling_params = to_sampling_params
