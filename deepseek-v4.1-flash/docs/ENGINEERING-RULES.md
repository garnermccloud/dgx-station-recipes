# Engineering rules

Short rules for this release tree. They exist because a silent downgrade once
looked like a performance regression for a full A/B window.

## 1. Never degrade silently

Apply the fast path or raise with one line saying why. Do not return `None`,
`continue`, or `pass` into a stock path. A warning is not a substitute for a
decision.

**Real example:** an SPDX header was added to the vendored DeepGEMM fork
headers. That changed their bytes, `fastpath/fastpath_megamoe.py:106` rejected
the shipped header checksum, logged `stock:`, and MegaMoE quietly turned off.
The fix is to restore the validated bytes and put the licence in
`LICENSE.DeepGEMM` beside them — never edit a vendored or checksummed file for
a header.

## 2. One compatibility assertion, up front

Do not checksum our own files or gate components on version strings that git
already records. Assert the supported image once in preflight
(`vllm/vllm-openai:v0.31.0`), then trust the tree.

**Real example:** the pinned FlashInfer tune file recorded
`flashinfer_version: 0.7.0`; under v0.31.0's `0.7.0.post1` the autotuner
rejected it and silently ran a different live tune
(`fastpath-vllm.log:546`). The compatibility check belongs in preflight, not
in each component's metadata.

## 3. No experiment scaffolding in production

If production always sets an env knob to the same value, make that value the
behaviour and delete the knob and the other branch. Keep a knob only when an
owner would legitimately change it; then document it in
`config/ds41.env.example`. No A/B arms, no registered-but-unarmed components.

**Real example:** `W8O_REQUIRE` in `fastpath/fastpath_w8oproj.py:34` is an A/B
boot switch. If production always runs one arm, remove the switch and the
fallback branch.

## 4. Logs reflect real state

Do not print `armed`, `active`, or `enabled` before the component is actually
engaged. Each component prints exactly one `ACTIVE` line after it engages.
Stock/off paths log once, with the reason.

**Real example:** `fastpath/hook/mega_peer_hook.py:565` warns that a rejected
pin means default tactics, but the process still continues. Replace the
warning-and-continue with apply-or-raise, and print the one ACTIVE line only
after the pin is loaded.

## 5. Do not mask failures in scripts

`|| true`, `2>/dev/null` on a step whose failure matters, and ignored return
codes hide exactly the failures this release cannot afford. Deliberate
best-effort cleanup must carry an explicit reason comment
(`# allow-||-true: ...`).

**Real example:** `serve/serve.sh:399-400` pipes `nvidia-smi` and
`docker stats` through `2>/dev/null ... || true`; a broken status command looks
like a healthy empty status. Either let it fail or state why it is best-effort.

## 6. One copy, one owner

Do not ship duplicate source trees, generated outputs beside their generator
inputs, or unused fixtures. If a file is generated, generate it in the build;
do not commit it.

**Real example:** `fastpath/peer3/` duplicates `kernels/peer3/` (same
`peer3_gb300.cu`, `peer3_ext.py`, layouts and tests). Pick the source tree and
delete or generate the other.

## 7. No unexplained site constants

Hard-coded host paths, usernames, GPU offsets, and fixture paths need one
explicit assertion that validates them, or they belong in config. Do not let a
stale default silently point at an old artifact.

**Real example:** `fastpath/peer3/tests/bench_send_gb300.py:62` hard-codes
`/hook/rowmap-mix-v1-h255.json`. Assert the fixture exists and matches the
intended rowmap, or take it from config.

## 8. Every wait is bounded; a dead container fails fast

Every boot wait has a deadline and an exit-on-death check. A restart policy is
not a retry strategy: if the engine dies, the launcher must stop waiting and
print the container's last log lines. Never sleep until a 60-minute timeout
while the engine is in a crash loop.

**Real example:** the final validation boot crashed at drafter load with
`NameError: name 'os' is not defined`; Docker's restart policy restarted the
engine in a loop and `ds41 up` waited the full 60 minutes. The launcher should
detect the first non-running/restarting state and fail with the tail of
`F-up.log`.

## 9. Every shipped module is linted and imported in CI-like checks

Run pyflakes over every shipped `.py` file and import every lazily loaded
module in the supported image. Undefined names are bugs; unused imports are
dead code. No `# noqa` ignores except an explicit registration mechanism that
actually consumes the registered names.

**Real example:** `fastpath/w6_markov.py:14` used `os.path` after commit
`5a353bc` removed `import os`. The CPU `components` import did not reach the
lazy module, so the first real boot failed. `pyflakes==3.2.0` catches it; the
new scrub lint stage runs it over all 89 shipped Python files.

## Review checklist

- Does every failure either apply the feature or raise one line?
- Is compatibility asserted once, in preflight?
- Is every env knob one an owner would change?
- Does each component print one truthful ACTIVE line?
- Are `|| true` / `2>/dev/null` justified in place?
- Is there exactly one copy of each source and no committed build output?
- Is every site constant asserted or configurable?
- Does every boot wait have a deadline and a fail-fast check for a dead container?
- Does every shipped Python module pass pyflakes and get imported in the image?
