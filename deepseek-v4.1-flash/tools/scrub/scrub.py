#!/usr/bin/env python3
"""Scan a release tree for secrets, site-specific detail, and silent-degradation
antipatterns.

The scanner is deliberately boring: no network, no third-party Python
dependencies, and it never prints a secret value. It checks text files and the
printable strings in binaries (the prebuilt kernels can carry build paths).
`--external` also runs gitleaks/trufflehog when they are already available.

The antipattern checks are intentionally narrow and line-oriented:
  * silent `except` / bare `except`
  * "falling back" / "-> stock" log strings
  * `|| true` and `2>/dev/null` in `serve/` or `ds41`, unless the same or
    previous line carries `allow-||-true: <reason>` / `allow-2>/dev/null: <reason>`
  * self-integrity sha256 checks (`sha256sum`, `hashlib.sha256`), unless the
    line carries `allow-sha256: <reason>` for generated-output naming

Only two vendored upstream paths are excluded: `docker/b12x/` (b12x subset)
and the checkpoint encoder
`docker/overlay-1017/vllm/tokenizers/deepseek_v41_encoding.py`. Each has a
reason in `VENDORED_UPSTREAM`; everything else is scanned.

Exit status: 0 clean, 1 findings, 2 usage/tool error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", ".venv", "venv", "dist", "build", ".cache",
    "tools/scrub",  # the scanner contains the patterns; it is not release content
}
SKIP_SUFFIXES = {
    ".pyc", ".pyo", ".o", ".a", ".lib", ".dll", ".dylib", ".png", ".jpg",
    ".jpeg", ".webp", ".gif", ".pdf", ".zip", ".gz", ".bz2", ".xz", ".7z",
    ".tar", ".whl", ".mp4", ".mov",
}
BINARY_SUFFIXES = {".so", ".bin", ".pt", ".pth", ".safetensors", ".onnx"}
MAX_BYTES = 64 * 1024 * 1024

ALLOWED_USERS = {
    "root", "vllm", "user", "app", "workspace", "ubuntu", "nvidia", "ds41",
    "runner", "codespace", "node", "docker", "model", "serve",
}
PLACEHOLDERS = {
    "", "...", "xxx", "xxxx", "changeme", "change-me", "your-key",
    "your_api_key", "your-api-key", "redacted", "placeholder", "dummy",
    "example", "none", "null", "false", "true",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    category: str
    path: str
    line: str
    message: str


# (severity, category, compiled regex, message). Order is stable for reports.
RULES = [
    ("high", "secret.private_key",
     re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
     "private key block"),
    ("high", "secret.provider_token",
     re.compile(r"\b(?:sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|"
                r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
                r"AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|"
                r"AIza[0-9A-Za-z_-]{35})\b"),
     "provider/API token"),
    ("high", "secret.jwt",
     re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
     "JSON Web Token"),
    ("high", "secret.aws_secret",
     re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}"),
     "AWS secret access key"),
    ("high", "secret.gcp_service_account",
     re.compile(r'"type"\s*:\s*"service_account"'),
     "GCP service-account JSON"),
    ("medium", "secret.assignment",
     re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|bearer|authorization)"
                r"\b\s*[:=]\s*['\"]([^'\"]{16,})['\"]"),
     "quoted credential-looking assignment"),
    ("medium", "secret.key_path",
     re.compile(r"(?i)(?:~|/|\\)(?:\.glm_api_key|\.netrc|\.pypirc|id_rsa|id_ed25519|"
                r"id_ecdsa|credentials(?:\.json)?|kubeconfig|\.env)\b"),
     "credential/private-key path"),
    ("high", "network.private_ipv4",
     re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
                r"192\.168\.\d{1,3}\.\d{1,3}|"
                r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
                r"100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})\b"),
     "private/internal IPv4 address"),
    ("high", "network.internal_host",
     re.compile(r"(?i)\b(?:[A-Za-z0-9-]+\.)+(?:ts\.net|lan|internal|home\.arpa)\b"),
     "internal hostname"),
    ("medium", "network.mdns_host",
     re.compile(r"(?i)(?:\b[a-z0-9][a-z0-9-]*-[a-z0-9-]*\.local\b|"
                r"https?://[^/\s]+\.local\b)"),
     "mDNS/internal .local hostname (review)"),
    ("high", "site.hostname",
     re.compile(r"(?i)\bws\d{2,4}\b"),
     "workstation-style hostname"),
    ("medium", "path.internal_repo",
     re.compile(r"(?i)\bds41-(?:prod|kernels|cache|results)\b"),
     "internal repository/path name"),
    ("high", "hardware.gpu_uuid",
     re.compile(r"\bGPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
     "hard-coded GPU UUID"),
    ("medium", "network.internal_registry",
     re.compile(r"(?i)\b(?:harbor|nexus|registry)\.(?:[a-z0-9-]+\.)+"
                r"(?:local|lan|internal)\b"),
     "internal registry hostname"),
]

USER_PATH = re.compile(r"(?i)(?<![\w$])(/home/|/Users/|C:\\\\Users\\\\)([A-Za-z0-9._-]+)")
KEY_FILE_RX = re.compile(r"(?i)(?:^|/)(?:\.env(?:\.[^/]*)?|\.glm_api_key|\.netrc|\.pypirc|"
                         r"id_rsa|id_ed25519|id_ecdsa|credentials(?:\.json)?|kubeconfig|"
                         r"auth\.json|key\.json|service-account[^/]*\.json|"
                         r"[^/]+\.(?:pem|key|p12|pfx))$")


def sanitize_path(path: str) -> str:
    """Redact usernames/hosts from a path before it reaches a report."""
    s = path.replace("\\", "/")
    s = USER_PATH.sub(lambda m: m.group(1) + "<user>", s)
    s = re.sub(r"(?i)\bws\d{2,4}\b", "<host>", s)
    s = re.sub(r"(?i)\bds41-(?:prod|kernels|cache|results)\b", "<internal>", s)
    s = re.sub(r"\bGPU-[0-9a-fA-F-]{36}\b", "<gpu-uuid>", s)
    return s


def is_placeholder(value: str) -> bool:
    v = value.strip().strip("'\"").lower()
    if v in PLACEHOLDERS:
        return True
    return (v.startswith("$") or v.startswith("<") or v.startswith("{")
            or v.startswith("your") or v.startswith("example"))


ANTIPATTERN_SUFFIXES = {
    ".py", ".sh", ".bash", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc",
}
ANTIPATTERN_NAMES = {"ds41"}
# Only these vendored upstream paths are allowlisted. Each entry has a reason.
VENDORED_UPSTREAM = {
    "docker/b12x/": "vendored b12x upstream subset; do not edit for style",
    "docker/overlay-1017/vllm/tokenizers/deepseek_v41_encoding.py":
        "vendored checkpoint encoder; upstream file, do not edit",
}
EXCEPT_LINE = re.compile(r"^(\s*)except\b([^:]*):\s*(.*)$")
SILENT_BODY = re.compile(r"^(?:pass|continue|return\s+None)\b")
FALLBACK_LOG = re.compile(
    r"(?i)(?:falling\s+back|fallback\s+to|->\s*stock\b|\bstock:)"
)
LOG_CALL = re.compile(
    r"(?i)(?:_log|logger|logging|print|stderr|stdout|warn|error|info|debug|write)\s*\("
)
RETURN_STOCK = re.compile(
    r"(?i)(?:return\s+None.*(?:stock|fallback)|(?:stock|fallback).*return\s+None)"
)
SELF_SHA = re.compile(
    r"(?i)(?:sha256sum\s+(?:-c|--check)|shasum\s+-a\s*256\s+-c|"
    r"hashlib\.sha256\([^\n]*\)\s*(?:==|!=)|"
    r"(?:==|!=)\s*[^\n]*hashlib\.sha256)"
)


def _is_antipattern_code(rel: str) -> bool:
    p = Path(rel)
    return p.name in ANTIPATTERN_NAMES or p.suffix.lower() in ANTIPATTERN_SUFFIXES


def _allowed_masking(lines: list[str], idx: int, marker: str) -> bool:
    """Allow a masking construct only with an explicit reason comment."""
    for j in (idx, idx - 1):
        if 0 <= j < len(lines) and marker in lines[j]:
            return True
    return False


def scan_antipatterns(rel: str, text: str, findings: list[Finding]) -> None:
    """Fail on silent-degradation patterns that can be detected mechanically."""
    if any(rel == p or rel.startswith(p) for p in VENDORED_UPSTREAM):
        return
    is_code = _is_antipattern_code(rel)
    is_launcher = rel == "ds41" or rel.startswith("serve/")
    if not is_code and not is_launcher:
        return
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        lineno = str(idx + 1)
        if is_launcher:
            if "|| true" in line and not _allowed_masking(
                    lines, idx, "allow-||-true:"):
                findings.append(Finding(
                    "high", "antipattern.masked_failure", rel, lineno,
                    "|| true outside an explicit allow-||-true reason"))
            if "2>/dev/null" in line and not _allowed_masking(
                    lines, idx, "allow-2>/dev/null:"):
                findings.append(Finding(
                    "medium", "antipattern.masked_stderr", rel, lineno,
                    "2>/dev/null outside an explicit allow-2>/dev/null reason"))
        if not is_code:
            continue
        m = EXCEPT_LINE.match(line)
        if m:
            exc_type = m.group(2).strip()
            body = m.group(3).strip()
            if not exc_type:
                findings.append(Finding(
                    "high", "antipattern.bare_except", rel, lineno,
                    "bare except"))
            silent = bool(body and SILENT_BODY.match(body))
            if not silent and (not body or body.startswith("#")):
                j = idx + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and SILENT_BODY.match(lines[j].strip()):
                    silent = True
            if silent:
                findings.append(Finding(
                    "high", "antipattern.silent_except", rel, lineno,
                    "except swallows the error"))
        if FALLBACK_LOG.search(line) and LOG_CALL.search(line):
            findings.append(Finding(
                "high", "antipattern.fallback_log", rel, lineno,
                "silent fallback / stock log string"))
        if RETURN_STOCK.search(line):
            findings.append(Finding(
                "high", "antipattern.return_none_stock", rel, lineno,
                "return None to stock/fallback"))
        if SELF_SHA.search(line) and not _allowed_masking(
                lines, idx, "allow-sha256:"):
            findings.append(Finding(
                "high", "antipattern.self_sha256", rel, lineno,
                "self-integrity sha256 check"))


def read_text(path: Path) -> tuple[str, bool]:
    """Return (text, binary). Binaries are reduced to printable strings."""
    data = path.read_bytes()
    if b"\x00" in data[:8192] or path.suffix.lower() in BINARY_SUFFIXES:
        if shutil.which("strings"):
            p = subprocess.run(["strings", "-a", str(path)], capture_output=True,
                               text=True, errors="replace", timeout=60)
            return p.stdout, True
        return data.decode("utf-8", "replace"), True
    return data.decode("utf-8", "replace"), False


def scan_text(rel: str, text: str, findings: list[Finding]) -> None:
    lines = text.splitlines()
    for lineno, line in enumerate(lines, 1):
        for severity, category, rx, message in RULES:
            m = rx.search(line)
            if not m:
                continue
            if category == "secret.assignment":
                value = m.group(1) if m.lastindex else ""
                if is_placeholder(value):
                    continue
            findings.append(Finding(severity, category, rel, str(lineno), message))
        # Username in a path is a separate rule; report once per line.
        for m in USER_PATH.finditer(line):
            user = m.group(2)
            if user.lower() in ALLOWED_USERS:
                continue
            findings.append(Finding("medium", "path.username", rel, str(lineno),
                                    "username in path"))
            break


def scan_file(path: Path, root: Path, findings: list[Finding], stats: dict) -> None:
    rel = path.relative_to(root).as_posix()
    if path.name.lower() in {".env.example", ".env.sample", ".env.template"}:
        stats["skipped"] += 1
        return
    # Credential filenames are findings even if the file is empty.
    if KEY_FILE_RX.search("/" + rel):
        findings.append(Finding("medium", "secret.key_file", rel, "-",
                                "credential/private-key filename"))
    try:
        if path.stat().st_size > MAX_BYTES:
            stats["skipped"] += 1
            return
        text, binary = read_text(path)
    except Exception as exc:  # unreadable file is a review item, not a crash
        findings.append(Finding("low", "file.unreadable", rel, "-", str(exc)[:120]))
        stats["skipped"] += 1
        return
    stats["binary" if binary else "text"] += 1
    scan_text(rel, text, findings)
    if not binary:
        scan_antipatterns(rel, text, findings)


def walk(root: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    skipped_dirs: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        keep = []
        for d in dirnames:
            rel = (rel_dir + "/" + d).lstrip("./")
            if d in SKIP_DIRS or rel in SKIP_DIRS or rel.startswith("tools/scrub"):
                skipped_dirs.append(rel)
                continue
            keep.append(d)
        dirnames[:] = keep
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in SKIP_SUFFIXES:
                continue
            files.append(p)
    return files, skipped_dirs


def _docker_image(name: str) -> bool:
    if not shutil.which("docker"):
        return False
    p = subprocess.run(["docker", "image", "inspect", name], capture_output=True,
                       text=True, timeout=30)
    return p.returncode == 0


def run_external(root: Path) -> list[str]:
    """Run gitleaks/trufflehog if installed, or via an already-pulled image.

    Raw scanner output can contain the secret; it is captured and only counts
    are returned.
    """
    notes: list[str] = []
    gl = shutil.which("gitleaks")
    gl_img = "zricethezav/gitleaks:latest"
    if gl:
        cmd = [gl, "detect", "--source", str(root), "--no-git",
               "--redact", "--exit-code", "0"]
    elif _docker_image(gl_img):
        cmd = ["docker", "run", "--rm", "--network", "none",
               "-v", f"{root}:/repo:ro", gl_img, "detect", "--source", "/repo",
               "--no-git", "--redact", "--exit-code", "0"]
    else:
        cmd = None
    if cmd:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            text = (p.stdout or "") + (p.stderr or "")
            n = len(re.findall(r"(?im)^\s*Finding:\s", text))
            if n == 0 and re.search(r"(?i)no leaks found", text):
                n = 0
            notes.append(f"gitleaks: {n} finding(s) (redacted; raw output not stored)")
        except Exception as exc:
            notes.append(f"gitleaks: could not run ({exc})")
    else:
        notes.append("gitleaks: not installed; skipped")

    th = shutil.which("trufflehog")
    th_img = "trufflesecurity/trufflehog:latest"
    if th:
        cmd = [th, "filesystem", str(root), "--no-update", "--no-verification", "--json"]
    elif _docker_image(th_img):
        cmd = ["docker", "run", "--rm", "--network", "none",
               "-v", f"{root}:/repo:ro", th_img, "filesystem", "/repo",
               "--no-update", "--no-verification", "--json"]
    else:
        cmd = None
    if cmd:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            n = sum(1 for line in (p.stdout or "").splitlines()
                    if '"DetectorName"' in line or '"DetectorType"' in line)
            notes.append(f"trufflehog: {n} finding line(s) (raw output not stored)")
        except Exception as exc:
            notes.append(f"trufflehog: could not run ({exc})")
    else:
        notes.append("trufflehog: not installed; skipped")
    return notes

def report_markdown(root: Path, findings: list[Finding], stats: dict,
                    external: list[str]) -> str:
    by_sev = {}
    for f in findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
    out = [
        "# Scrub report",
        "",
        f"Root: `{sanitize_path(str(root))}`",
        "",
        f"Scanned **{stats['text']} text**, **{stats['binary']} binary** files; "
        f"skipped **{stats['skipped']}**.",
        "",
    ]
    if external:
        out += ["External scanners:", ""]
        out += [f"- {sanitize_path(x)}" for x in external]
        out += [""]
    if not findings:
        out += ["**No findings.** The tree is clean for the patterns this tool knows.", ""]
        return "\n".join(out)
    out += ["| Severity | Category | Path | Line | What |", "|---|---|---|---|---|"]
    for f in sorted(findings, key=lambda x: (x.severity, x.category, x.path, x.line)):
        out.append(f"| {f.severity} | {f.category} | `{sanitize_path(f.path)}` | "
                   f"{f.line} | {f.message} |")
    out += ["", "Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(by_sev.items())), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="tree to scan (default: .)")
    ap.add_argument("--report", help="write a redacted markdown report")
    ap.add_argument("--json", action="store_true", help="emit redacted JSON findings")
    ap.add_argument("--external", action="store_true",
                    help="also run gitleaks/trufflehog if installed")
    ap.add_argument("--exit-zero", action="store_true", help="always exit 0")
    ap.add_argument("--self-test", action="store_true", help="run the scanner self-test")
    ap.add_argument("--deny-term", action="append", default=[],
                    help="extra site-specific term to flag (repeatable; not written to the report)")
    ap.add_argument("--deny-file", help="file with one extra site-specific term per line")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    deny_terms = list(args.deny_term)
    if args.deny_file:
        for line in Path(args.deny_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                deny_terms.append(line)
    for term in deny_terms:
        if len(term) >= 3:
            RULES.append(("high", "site.deny_term",
                          re.compile(re.escape(term), re.I),
                          "operator denylist term"))

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"scrub: not a directory: {root}", file=sys.stderr)
        return 2
    files, skipped_dirs = walk(root)
    findings: list[Finding] = []
    stats = {"text": 0, "binary": 0, "skipped": 0}
    for p in files:
        scan_file(p, root, findings, stats)
    external = run_external(root) if args.external else []

    if args.json:
        print(json.dumps({
            "root": sanitize_path(str(root)),
            "findings": [asdict(f) for f in findings],
            "stats": stats,
            "external": [sanitize_path(x) for x in external],
            "skipped_dirs": [sanitize_path(x) for x in skipped_dirs],
        }, indent=2, sort_keys=True))
    else:
        print(f"DS41 scrub: root={sanitize_path(str(root))}")
        print(f"scanned {stats['text']} text, {stats['binary']} binary; "
              f"skipped {stats['skipped']} files")
        for x in external:
            print(f"external: {sanitize_path(x)}")
        for f in sorted(findings, key=lambda x: (x.severity, x.category, x.path, x.line)):
            print(f"[{f.severity}] {f.category} {sanitize_path(f.path)}:{f.line} — {f.message}")
        if findings:
            print(f"SUMMARY: {len(findings)} finding(s) in "
                  f"{len({f.path for f in findings})} file(s)")
        else:
            print("SUMMARY: clean")

    if args.report:
        Path(args.report).write_text(report_markdown(root, findings, stats, external))
        print(f"report: {sanitize_path(args.report)}")

    if args.exit_zero:
        return 0
    return 1 if findings else 0


def self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ds41-scrub-") as td:
        root = Path(td)
        (root / "bad.txt").write_text(
            "API_KEY = \"sk-" + "a" * 24 + "\"\n"
            "host=" + ".".join(["192", "168", "1", "9"]) + "\n"
            "path=/home/someone/project\n"
            "gpu=GPU-" + "-".join(["11111111", "2222", "3333", "4444", "555555555555"]) + "\n"
        )
        (root / ".env").write_text("TOKEN=abc\n")
        (root / "bad.py").write_text(
            "try:\n"
            "    run()\n"
            "except Exception:\n"
            "    pass\n"
            "try:\n"
            "    run()\n"
            "except:\n"
            "    pass\n"
            "_log('falling back to stock')\n"
        )
        (root / "serve").mkdir()
        (root / "serve" / "bad.sh").write_text(
            "cmd || true\n"
            "cmd 2>/dev/null\n"
            "sha256sum -c file\n"
        )
        findings: list[Finding] = []
        stats = {"text": 0, "binary": 0, "skipped": 0}
        for p in walk(root)[0]:
            scan_file(p, root, findings, stats)
        cats = {f.category for f in findings}
        want = {"secret.provider_token", "network.private_ipv4", "path.username",
                "hardware.gpu_uuid", "secret.key_file",
                "antipattern.silent_except", "antipattern.bare_except",
                "antipattern.fallback_log", "antipattern.masked_failure",
                "antipattern.masked_stderr", "antipattern.self_sha256"}
        missing = want - cats
        if missing:
            print("SELFTEST FAIL: missing " + ", ".join(sorted(missing)), file=sys.stderr)
            return 1
        print("SELFTEST PASS: scanner catches tokens, private IPs, usernames, "
              "GPU UUIDs, credential files, silent fallbacks and self-checksums")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
