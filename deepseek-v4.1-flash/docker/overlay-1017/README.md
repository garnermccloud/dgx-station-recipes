# overlay-1017: DeepSeek-V4.1-Flash-1017 prompt format for vLLM v0.31.0

A temporary bridge. The release's `serve.sh` mounts these four files over the image's copies; the four v0.31.0
base files are byte-identical to the nightly's, so the overlay applies as-is. A file missing from the overlay
directory leaves the image's copy in place. Delete this directory once the serving image ships the 1017 encoder
and the fixes below.

Revision r2 (2026-10-20) aligns the frontend with deepseek-recipe 0.2.0 (released 2026-10-20), DeepSeek's maintained
request-conversion and prompt-encoding toolkit. Its diff against the 1017 revision is `overlay-r2.diff`.

| File | Source | Change |
|---|---|---|
| `vllm/tokenizers/deepseek_v41_encoding.py` | the checkpoint's `encoding/encoding.py` (HF revision 1bae0e10) | verbatim, plus `"xhigh": 75` (deepseek-recipe maps it the same way). Unchanged in r2 |
| `vllm/tokenizers/deepseek_v41.py` | the image's file | hands the encoder OpenAI-style text/image blocks; `developer` becomes `user`; refuses image-carrying tool results sent out of tool-call order; tokenizes rendered prompts one special-token-delimited segment at a time through an LRU cache; memoizes the reference's `find_last_user_index` per render. **r2:** special-token spellings in client text encode as plain text (below); `reasoning_effort` `minimal`/`medium` map to `low`/`high`; the tool list is rendered as deepseek-recipe does (`name`, `description` or `""`, `parameters` or `{}`, nothing else; `defer_loading` tools left out); tools and a `json_object`/`json_schema` response format go into the first message only when it is a system message, else into a new empty system message inserted first; `continue_final_message` continues the last assistant message without an end-of-sentence token |
| `vllm/entrypoints/anthropic/serving.py` | the image's file | system and thinking blocks joined with blank lines; one tool message per tool_result. **r2:** inline `system` messages stay mid-conversation system messages (`<｜System｜>`, deepseek-recipe 0.2.0 for V4.1; 0.1 used `<system-reminder>` user turns); a `tool_reference` in a tool result loads the referenced tool with a `### Updated Tool Schemas` system message after that user message; document placeholders name the media type; `thinking.budget_tokens` is accepted and ignored (vLLM mapped it to `thinking_token_budget`, which force-ends reasoning); `tool_choice: any` becomes `auto` |
| `vllm/renderers/deepseek_v4.py` | the image's file | **r2, new:** passes the request's `response_format` to the tokenizer wrapper as `ds_response_format`, so the schema is written into the prompt as deepseek-recipe does |

## Special-token spellings in client text (r2)

deepseek-recipe 0.2.0 ("explicit special tokens"): only the control tokens the template inserts are special tokens. A
special-token spelling in user, system or tool text, in assistant content or reasoning, in tool-call arguments or in
tool schemas encodes as plain text (the grounding tokens `<｜ref｜>`, `<｜box｜>`, ... stay special in assistant content
and reasoning). Before r2 a tool result holding `</think>`, `<｜User｜>` or `<｜DSML｜ invoke` (our own sources do) put
those control tokens into the prompt.

The wrapper escapes each added-token spelling in client text (inserts U+E000 after its first character; a literal U+E000
is doubled), so the rendered prompt splits only at template tokens. `_SegmentEncoder` unescapes such a segment and
encodes it without the added vocabulary, as the recipe's `PromptTokenizer` does. Text without spellings takes the old
path unchanged. Tool-call arguments are escaped after JSON decoding (as the encoder decodes them), so a JSON escape such
as `<｜User｜>` cannot reintroduce a control token.

## Known differences from deepseek-recipe 0.2.0, left as they are

- The prompt layout follows the 1017 checkpoint's `encoding.py`, which differs from the recipe's renderer in seven
  places dsh traffic does not reach: without a tool prompt, current-turn reasoning is kept (1017 README documents this);
  `latest_reminder` placement; no generation header after a final assistant message; double-encoded tool arguments are
  decoded; a mid-conversation tool addition repeats the full tools block; a reasoning-only prefix continuation closes
  `</think>`; no `<｜DSML｜ calls>` prefill for a required tool choice (vLLM forces the same text with a structural tag).
- `tool_choice` `required`/named is accepted in thinking mode (the recipe rejects it); a named choice lists every tool.
- `tool_choice: none` still renders the tool prompt (serve flag `--exclude-tools-when-tool-choice-none` would drop it).
- Sampling parameters pass through as clients send them (the recipe forces `temperature` 1.0 and `top_p` >= 0.95 in
  thinking mode, and `top_p` 1.0 outside it); stop strings also match inside reasoning; no `max_tokens` default.
- A user-level Anthropic `document` block is a 400 (vLLM's protocol model); invalid JSON tool arguments become `{}`.

Kept on purpose: the Anthropic billing-header block (`x-anthropic-billing-header`, a per-request hash) is stripped,
because keeping it would miss the prefix cache on every request.

## Rebuild and verify

    build_overlay_1017.py <orig_dir> <out_dir>   # the 1017 revision from the image files and the checkpoint encoder
    patch -d <out_dir> -p1 < overlay-r2.diff     # r2 (after copying the image's renderers/deepseek_v4.py into <out_dir>/vllm)

Verified 2026-10-20 in CPU containers of the production image (r2):
- DeepSeek's `encoding/test_encoding.py`: 55/55.
- `tokcache_check.py`: identical ids on 3,005 prompts/strings; next turn of a 192K-token conversation 24 ms.
- `tokcache_check2.py`: exact on 2,120 render/tokenize checks; next-turn render 12.3 ms (8.4 ms before r2) + tokenize 23.7 ms.
- `special_text_check.py`: 1,500 conversations with special-token spellings in every client field, 7,500 checks;
  the overlay's ids equal deepseek-recipe 0.2.0's on all three tokenizing paths and under truncation, and the unescaped
  rendering equals the 1017 reference encoder's.
- `prompt_parity.py --offline`: unchanged from the 1017 revision except the Anthropic inline-system case, now V=E.
- 200 dsh-shaped Chat Completions requests (2.07M prompt tokens, real source files as tool results): the 150 without
  special-token spellings tokenize identically to the 1017 revision and to deepseek-recipe 0.2.0; the 50 drawn from files with spellings
  now equal deepseek-recipe 0.2.0 (images compared with each image's token run collapsed to its placeholder).
