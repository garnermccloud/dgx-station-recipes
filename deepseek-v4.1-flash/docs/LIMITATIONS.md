# Limitations

Things this recipe does not fix, so you do not spend a day rediscovering them. Each one says what you will see,
why, and what to do about it.

## 1. Identical greedy requests can diverge past ~512 tokens of context

**What you see.** Two runs of the same prompt at temperature 0 produce the same tokens for a while, then one
token differs, and from there the outputs are unrelated. It never happens below ~512 tokens of context.

**Why.** This is a known characteristic of the current vLLM top-k used by the sparse indexer. For rows longer
than `top_k`, the selected indices are placed with an atomic counter (`indices[atomicAdd(&counter, 1)] = i`), so
the emitted *order* varies between runs, and boundary ties are filled in arrival order, so occasionally the *set*
varies too. Sparse MLA accumulates over that list in order, so the result differs in the last bits, and a greedy
decode flips at the next near-tie. Short rows (≤ `top_k`) take a deterministic iota path, which is why short tests
look fine and this only shows up at long context.

**How we know.** Every one of 16 late divergences across six configurations (stock, stock without MegaMoE,
stock without speculation, production, production without speculation, production without the lean fence)
started at a context between 521 and 743 tokens — never below 512. A bitwise dump of every layer slot showed all
203 slots identical on the step before the divergence, and the next step's layer-0 input differing because a
different token had been sampled. A dedicated probe found the top-k order differing in 19 of 19 calls, with the
selected set differing on tie-heavy rows.

**Our mitigation.** For evaluation work, keep the context under 512 tokens — below that, two runs of the same
configuration are bit-identical and the numbers are reproducible. Speculation amplifies the divergence but is not
its cause. The indexer also has a determinism build (ascending rows, lowest-index ties,
`W8TG_SORT=1`) which removes the ordering variation; it is not the default because it changes the indexer kernel
for every user, and it is not part of the shipped kernels (see `docs/BUILD-FROM-SOURCE.md`).

**What is *not* affected.** Below 512 tokens, two runs of the *same* configuration are bit-identical (203/203 slots
on every decode step of a 384-token run). That is run-to-run stability, not stock-vs-fast-path equality — those
differ by design, see below.

**Stock vs fast path is not bit-identical.** Measured 2026-10-23 with speculation off and identical batch shapes,
the fixed prompt set diverges from token 0/4/9. Most components are bit-identical to the kernel they replace, but
`w8oproj` (o_proj in MXFP8), `lowb_aux` (a different K summation order), the quantized draft head and the peer
tier's bf16 sidecar path change numerics deliberately. Quality is validated by GSM8K-200, the needle test and the
tool call; do not expect stock-vs-fast-path equality.

## 2. Thinking mode is always sampled, never greedy

**What you see.** You send `temperature: 0` with thinking on and get a different answer each time.

**Why.** This is a deliberate policy in the serving layer (`ds41_sampling`): every thinking-mode request is
sampled at temperature 1.0 with top_p ≥ 0.95, whatever the client asks for. The DeepSeek recipe's reasoning
quality assumes sampling; greedy decoding of a long reasoning trace tends to loop.

**What to do.** If you want greedy, turn thinking off. `ds41 bench` does exactly that for its greedy row, which
is why the greedy numbers are higher (4.6–4.8 tokens per step versus 3.0–3.8). If you want the client's
temperature honoured in thinking mode, that is a policy change you make in `ds41_sampling` — it is not a bug.

## 3. First start is slow: the kernels compile

**What you see.** The first `ds41 up` takes longer than later ones, and the log shows JIT compiles. Even a warm
start is not instant: measured on a live server, **5 min 11 s** from container start to serving, of which 172 s is
loading the weights and 94 s is engine init and graph capture.

**Why.** DeepGEMM and the fast-path kernels are compiled for your exact GPU on first use. The result is cached
under `~/.ds41/state/` (and inside the container's cache directory) for the kernel source, so later starts reuse
it.

**What to do.** Nothing — it is a one-time cost. If you update the driver, expect one more slow start. If a start
seems stuck, `ds41 logs --kernel` shows the compile that is running.

## 4. Context, KV cache and concurrency are finite

* The KV cache holds **11,372,409 tokens** at the default `DS41_UTIL=0.95` on the reference box (21.7× a single
  524,288-token request). Lower the util to share the GB300 and the cache shrinks by ~380K tokens per GiB; past
  its size, many long requests at once queue for space.
* Decode throughput falls with context length: the KV cache reads start to dominate. The published numbers are
  at short context (< 4 k tokens).
* The peer tier is armed for MoE calls with **up to 64 tokens**. Above that the tier steps aside and that layer
  runs on the GB300 alone, which is correct but slower. (This is about the token count, not the GPUs: the RTX PRO
  6000 itself is required, see §7.)
* One box, two GPUs. There is no multi-node path in this recipe.

## 5. Utilisation trades KV cache, not speed

`DS41_UTIL` (default: `DS41_UTIL_DEFAULT` in `config/recipe.env`) is the fraction of the GB300 the server may use. Raising it buys KV cache, not
throughput: decode here is bandwidth-bound, so the published tok/s numbers do not move with it. Lower it if you
share the GPU with something else — a desktop session, a second job — and expect a proportionally smaller KV
cache. `RESULTS.md` states the utilisation each number was measured at.

## 6. The 16-stream number is a band, not a point

Two passes gave 1,970 and 2,260 tok/s aggregate at 16 streams. That is real run-to-run variation from queueing
and prefix-cache state, not a measurement mistake. Quote it as a range. Single-stream, 4-stream and 8-stream are
stable to about ±1 %.

## 7. What is not measured here

* Long-context throughput (100 k+) — expected to be lower, not measured.
* Multi-turn chat with a warm prefix cache — expected to be higher, not measured.
* Other GPUs or other CPUs. The recipe targets GB300 + RTX PRO 6000 on **aarch64 (Grace)**; the kernel
  selection, the peer tier and the tuning are for that pair, and the image is `linux/arm64`. It will not run on
  Hopper, an RTX 5090 is not a substitute for the PRO 6000, and there is no x86 build in this release.
* Other model checkpoints. The fast paths are written against this checkpoint's module structure and shapes.
