# verify

`./verify.sh` runs **nine** checks: four preflight checks and five quality/speed checks.

```bash
./verify.sh            # ~7 min: preflight + tool, needles, GSM8K-200, decode, coding
./verify.sh --quick    # ~4 min: shorter decode windows, one coding run
./verify.sh --full     # 3 coding runs
```

Each check prints PASS/FAIL; the script exits non-zero if any fails.  The preflight is one compatibility
assertion (image tag + vendored DeepGEMM base hashes), image present (with the 58132 decoder overlay), weights
complete (48 shards), both GPUs visible, and the boot-log active gate.  The gate requires exactly one
`DS41_FASTPATH ACTIVE <component>` line for every component, one `W6_MEGAMOE ACTIVE:`, one
`DS41_FASTPATH peer3: finish folded into MegaMoE`, and one `Loaded N configs` line with **no**
`No tuned config covers` fallback.  A missing RTX PRO 6000 is a preflight FAIL: `found only the GB300 / Fix:
install the RTX PRO 6000 (96 GB); the model does not fit in the GB300 alone.`  The quality/speed checks
use the same tolerances as the production lane: **80%** of the published decode/coding speed (the default
util is 0.95 on a dedicated machine, DS41_UTIL_DEFAULT in config/recipe.env; the references used 0.84 beside a co-tenant) and **95%** on GSM8K.

| check | reference | pass rule |
|---|---|---|
| preflight image | `vllm/vllm-openai:v0.31.0` with v0.31.0 overlay targets | image tag, DeepGEMM base hashes and targets present |
| preflight boot | one ACTIVE line per component; W6 MegaMoE; peer3 finish fold; no tune fallback | all present |
| preflight weights | 48 shards | all 48 present |
| preflight GPUs | GB300 + RTX PRO 6000 | both visible |
| tool call | PASS | parsed `get_weather` with Denver |
| needles 200K | 3/3 at 10/50/90% | all three |
| GSM8K-200 | 196/200 (98%) | >= 95% |
| C1/C4/C8/C16 | 467 / 992 / 1369 / 1970 tok/s | >= 80% each |
| coding prompt | 724.7 tok/s | >= 80% |

## `ds41 bench --full`: the metrics suite

`ds41 bench --full` runs `bench_full.py` against the running server in one session and writes
`~/.ds41/state/bench/bench-<time>.json` (every number, plus the environment read from the running container: image,
driver, GPUs, util, rowmap, recipe commit, KV-cache tokens from the boot log) and a markdown table next to it.

| stage | shape |
|---|---|
| `fixed` | single stream, `data/fixed_prompt.txt` (~2.6K tokens), greedy, thinking off, 2,048 output tokens, 10 runs after a warm-up: output tok/s, TTFT, accepted tokens per step, steps/s (median + IQR) |
| `gsm8k_c1`, `greedy_c1` | single stream over GSM8K questions for a 60 s window after 20 s warm-up: T=1.0 thinking on, and greedy thinking off |
| `coding` | `coding.py`'s shape, 5 runs |
| `streams` | 4 / 8 / 16 / 32 streams, the `decode.py` shape: aggregate and per-stream tok/s, tokens per step |
| `prefill` | uncached ~16K / 64K / 128K-token prompts (distinct filler + `cache_salt`): TTFT and prefill tok/s |
| `capacity` | `kvfill.py`: 16 distinct ~110K-token contexts decoding 2,500 forced tokens at once plus a 500K-token needle: all running at once, preemptions, wall time |
| `quality` | GSM8K-200, the 200K needles and the tool call |

It takes about 40 minutes. `--skip STAGE,...`, `--reps`, `--conc`, `--window` and `--out DIR` adjust it
(`ds41 bench --full --help`). Tokens are counted by the server (`/metrics`), not by the client.

The references are in `published.json`; C1/C4/C8 are the 2026-10-22 production run (`prod-numbers-1021`),
C16 is the 2026-10-23 production reasoning run (`prod-c16-1023`, same temperature 1.0 / top_p 0.95 /
thinking-on sampling as `decode.py`), and the coding prompt is the 2,048-token greedy thinking-off shape.
The quality checks are the same ones the production lane used: 200K-token retrieval, GSM8K-200, a tool
call, and the coding prompt.

`DS41_IMAGE` and `DS41_MODEL_DIR` select the image and checkpoint.  The Python checks use host Python when
it is present and otherwise run inside the release image (`py.sh`), so Python is not a host requirement.

`data/gsm8k-200.jsonl` is the first 200 rows of RonanLabs' GSM8K harness (Apache-2.0; the underlying
GSM8K is MIT, OpenAI).  `data/coding_source.py` is the 1017 overlay builder used as the coding prompt.
