"""
pii_scan.py — keep personal data and secrets out of a PUBLIC repository  (28 Sep 2026)
=====================================================================================
The repo is public on purpose (development is easier when an assistant can read it).
That is only safe if nothing personal or secret is ever committed. This scanner is the
guard: it runs as a test, and can run as a git pre-commit hook.

  * generic rules catch: real email addresses, real home-directory usernames, machine
    hostnames with a user prefix, private/LAN IPs, credential-shaped literals, private keys,
    tracker (Notion) links / workspace IDs;
  * an OPTIONAL local, UNTRACKED denylist (.pii_denylist, one string per line, '#' comments)
    holds the personal specifics (surname, town, employer, ...). It is gitignored, so the
    personal terms themselves never enter the public repo -- but they are still enforced.

    python src/pii_scan.py              # scan every tracked file (exit 1 on findings)
    python src/pii_scan.py --staged     # scan only what is staged (use in a pre-commit hook)

HISTORY IS NOT SCANNED HERE. Only the current tree (or the staged changes) is. A one-off
history scan was done on 28 Sep 2026 (no API keys; one personal email in a debug script).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_PREFIXES = ("tests/fixtures/",)
SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".gz", ".zip", ".pdf", ".lock")
# The scanner's own rule text and its test inputs are deliberately leak-SHAPED, so these two
# exact paths are exempt. The set is pinned by a test so it cannot quietly grow.
SELF_EXEMPT = {"src/pii_scan.py", "tests/test_pii_scan.py"}

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EMAIL_ALLOWED = re.compile(
    r"^(your@email\.com|git@github\.com|[^@]+@(example\.(com|org|net)|users\.noreply\.github\.com))$", re.I)

HOME_PATH = re.compile(r"/home/(?!user\b|yourname\b|username\b|youruser\b)[a-z][a-z0-9_-]{1,31}")
USER_HOSTNAME = re.compile(r"\b[a-z][a-z0-9]{1,20}-NUC[0-9A-Z]{4,}(?:-[A-Z])?\b")
PRIVATE_IP = re.compile(
    r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b")
SECRET_QUOTED = re.compile(
    r"""(?i)\b(api[_-]?key|secret[_-]?key|secret|token|passw(?:or)?d)\b\s*[=:]\s*["'](?![$<{])[A-Za-z0-9/_+=.-]{16,}["']""")
SECRET_ENV_LITERAL = re.compile(
    r"^\s*(?:export\s+)?[A-Z][A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD)[A-Z0-9_]*=(?![$<{\"'])(?!your)[A-Za-z0-9/_+.-]{8,}", re.M)
NOTION_LINK = re.compile(r"notion\.(?:so|com)/[A-Za-z0-9/_-]*[0-9a-f]{20,}|collection://[0-9a-f-]{20,}", re.I)
TOKEN_SHAPES = re.compile(
    r"\bsk-[A-Za-z0-9_-]{20,}|\bAKIA[0-9A-Z]{16}\b|\bghp_[A-Za-z0-9]{30,}|\b(?:PK|AK)[0-9A-Z]{18}\b"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----")


def _mask(s: str) -> str:
    s = s.strip()
    return (s[:3] + "…" + s[-2:]) if len(s) > 8 else "…"


def scan_text(text: str, path: str = "<text>", denylist: tuple[str, ...] = ()) -> list[dict]:
    """Return findings as dicts {path, line, rule, snippet} (snippet is masked)."""
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        for m in EMAIL.finditer(line):
            if not EMAIL_ALLOWED.match(m.group(0)):
                out.append({"path": path, "line": i, "rule": "email", "snippet": _mask(m.group(0))})
        for rule, rx in (("home_path", HOME_PATH), ("user_hostname", USER_HOSTNAME),
                         ("private_ip", PRIVATE_IP), ("secret_literal", SECRET_QUOTED),
                         ("token_shape", TOKEN_SHAPES), ("notion_link", NOTION_LINK)):
            for m in rx.finditer(line):
                out.append({"path": path, "line": i, "rule": rule, "snippet": _mask(m.group(0))})
        low = line.lower()
        for term in denylist:
            if term and term.lower() in low:
                out.append({"path": path, "line": i, "rule": "denylist", "snippet": "(local term)"})
    for m in SECRET_ENV_LITERAL.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        out.append({"path": path, "line": line_no, "rule": "secret_env_literal", "snippet": _mask(m.group(0))})
    return out


def load_denylist(root: Path = ROOT) -> tuple[str, ...]:
    f = root / ".pii_denylist"
    if not f.exists():
        return ()
    return tuple(ln.strip() for ln in f.read_text().splitlines() if ln.strip() and not ln.startswith("#"))


def _git(args: list[str], root: Path) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def tracked_files(root: Path = ROOT, staged: bool = False) -> list[str]:
    names = (_git(["diff", "--cached", "--name-only", "--diff-filter=ACM"], root) if staged
             else _git(["ls-files"], root)).split("\n")
    return [n for n in names if n and not n.startswith(SKIP_PREFIXES) and not n.endswith(SKIP_SUFFIXES)
            and n not in SELF_EXEMPT]


def scan_repo(root: Path = ROOT, staged: bool = False) -> list[dict]:
    deny = load_denylist(root)
    findings = []
    for name in tracked_files(root, staged):
        try:
            text = (_git(["show", f":{name}"], root) if staged else (root / name).read_text())
        except (UnicodeDecodeError, FileNotFoundError, subprocess.CalledProcessError):
            continue                                        # binary / deleted: nothing to scan
        findings += scan_text(text, name, deny)
    return findings


if __name__ == "__main__":
    staged = "--staged" in sys.argv
    found = scan_repo(staged=staged)
    for f in found:
        print(f"{f['path']}:{f['line']}  [{f['rule']}]  {f['snippet']}")
    print(f"\n{len(found)} finding(s)" + ("" if found else " -- clean"))
    sys.exit(1 if found else 0)
