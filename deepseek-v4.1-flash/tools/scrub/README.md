# Scrub

One command before every commit:

```bash
tools/scrub/scrub.sh --external --report SCRUB-REPORT.md
```

It first runs **pyflakes 3.2.0** inside `vllm/vllm-openai:v0.31.0` over every
tracked `.py` file except `docker/b12x/` and the checkpoint encoder. Any
pyflakes finding fails the scrub: undefined names are bugs, unused imports are
dead code. There are no ignores except the explicit registration mechanism in
`fastpath/components.py`.

It then scans the whole staging tree (text and printable strings in binaries) for:

- API keys, provider tokens, JWTs, private keys, cloud credentials;
- `.env`/key/credential filenames;
- private IPv4 ranges and internal hostnames;
- usernames in `/home/...`, `/Users/...` and Windows paths;
- workstation-style hostnames (`ws` + digits);
- hard-coded GPU UUIDs (the installer should detect the GPUs);
- internal registry hostnames;
- silent-degradation antipatterns: `except Exception: pass`, bare
  `except:`, `falling back` / `-> stock` log calls, self-integrity
  `sha256sum -c` / comparison checks, and `|| true` or `2>/dev/null` in
  `serve/` or `ds41`. Plain build digests and content-addressed cache keys
  are not self-integrity checks.

Site-specific service names are intentionally not hard-coded in this public
tool. Pass your own private terms when you run it:

```bash
tools/scrub/scrub.sh --deny-term your-service --deny-file ~/.ds41-private-terms
```

The term is used as a case-insensitive literal and is never written to the
report.

The tool prints only the category, path and line. It never prints the value.
It skips `.git`, caches, build output and its own directory, and it reduces
binaries to `strings` so build-time paths in the prebuilt kernels are visible.
`--external` also runs `gitleaks` and `trufflehog` if they are already
installed; it counts their findings and does not copy their raw output.

Exit status is 0 when clean, 1 when it finds something, 2 for a bad invocation.
`--exit-zero` is available for a report-only run. `--json` emits redacted
findings for a CI job. `--self-test` checks the scanner itself.

## Antipattern allowlist

The launcher masking checks are intentionally strict. A deliberate masking
step must say why on the same line or the line immediately above:

```bash
chmod 700 "$dir" 2>/dev/null || true  # allow-||-true: best-effort cleanup
```

Use `allow-2>/dev/null: <reason>` for a deliberate stderr suppression and
`allow-sha256: <reason>` for a generated-output filename hash. There is no
allowlist for silent exceptions or fallback logs: fix the code.

Only two vendored upstream paths are excluded from the antipattern checks,
each with a reason in `scrub.py`:

- `docker/b12x/` — vendored b12x upstream subset; do not edit for style.
- `docker/overlay-1017/vllm/tokenizers/deepseek_v41_encoding.py` — vendored
  checkpoint encoder; upstream file, do not edit.

Everything else, including `fastpath/megamoe_fork/` and overlay-derived files,
is scanned. Do not use the allowlist for our own code.

## Fixing a finding

- **Secret**: remove it from the tree, rotate it, and replace it with an
  environment variable or a generated file. Do not commit a scrubbed value.
- **Internal IP/hostname**: use a placeholder such as `your-host` or a public
  URL. For a GPU UUID, detect it at runtime.
- **Username path**: use `$HOME`, a relative path, or `/workspace`.
- **Site-specific service**: describe the optional capability generically, or
  move the detail into a private operator note that is not in the repo.
- **Binary build path**: rebuild with a neutral prefix, or strip the path.

The report is deliberately redacted. If you need the exact line, open the
file; the line number in the report points to it.
