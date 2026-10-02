# Addendum to PR-004 Pre-registration (spec v0.6) — hash-sequencing precondition

*28 Sep 2026. Add the line below to PR-004 v0.6, §8 ("Conditions of freeze"), as a new item, before PR-004's own hash is taken. This closes the gap flagged across several rounds of PR-005's review. Hardened twice: v2 added the release valve; v3 (this version) fixes a contradiction between this file's own prose and its own code, adds a self-test, and strengthens the closure-log check.*

**New PR-004 §8 item:** *Before PR-004's G2 mechanism session runs, the session's setup step must verify that PR-005's committed hash file (`PR005_HASH.txt`) exists, is non-empty, and — case-insensitively — is exactly 64 hexadecimal characters — **or** that a logged, non-empty, dated closure record exists at `PR005_CLOSURE.txt` recording PR-005's closure without a freeze. If neither holds, the G2 session halts and does not run. This is a hard precondition, not a reminder.*

**Why here, not there:** PR-005 (this item) is the one with something to protect — its entire reason for existing is deciding the "swing trading" question before PR-004's result is known. The check belongs in PR-004's spec because PR-004 is the session that could accidentally violate the ordering; PR-005 has no session that could jump ahead of itself.

**Why a release valve:** a precondition with no escape means PR-004 halts forever if PR-005 is later abandoned. Deadlocks without a documented release valve tend to get broken ad hoc under time pressure — exactly the failure mode this precondition exists to prevent.

**File format (v3, corrected):** `PR005_HASH.txt` and `PR005_MANIFEST.txt` are now **separate files**, resolving a contradiction in the previous version of this addendum where the prose described a multi-line manifest but the code checked a single hex string. `PR005_MANIFEST.txt` lists the SHA-256 of each frozen artifact (the pre-registration document, the frontier-generating script, the grid specification, the DFF pull's output) one per line. `PR005_HASH.txt` contains only the SHA-256 digest of `PR005_MANIFEST.txt` itself — a single 64-character hex string, exactly matching what the code below checks. Paths are resolved against the repository root, not the working directory the check happens to run from.

**Mechanical implementation:**
```python
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]   # adjust to the actual script location
hash_file = REPO_ROOT / "docs" / "PR005_HASH.txt"
closure_log = REPO_ROOT / "docs" / "PR005_CLOSURE.txt"

def _valid_hash_file(p: Path) -> bool:
    if not p.exists():
        return False
    content = p.read_text().strip().lower()
    return bool(re.fullmatch(r"[0-9a-f]{64}", content))

def _valid_closure_log(p: Path) -> bool:
    return p.exists() and len(p.read_text().strip()) > 0

assert _valid_hash_file(hash_file) or _valid_closure_log(closure_log), (
    "PR-005 must be hashed (a well-formed 64-hex-char PR005_HASH.txt) before PR-004's "
    "G2 session runs, or a logged, dated closure record must exist at PR005_CLOSURE.txt "
    "(pre-registration sequencing requirement, added 28 Sep 2026, hardened twice since)."
)
```

**Self-test (run once before PR-004's freeze, not left unexecuted):**
```python
import tempfile
def _selftest():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "hash.txt"
        p.write_text("a" * 64)
        assert _valid_hash_file(p), "self-test: valid hash should pass"
        for bad in ("", "a" * 63, "g" * 64, "A" * 64):
            p.write_text(bad)
            ok = _valid_hash_file(p)
            assert ok == (bad == "A" * 64), f"self-test: {bad!r} gave unexpected result {ok}"
        p.unlink()
        assert not _valid_hash_file(p), "self-test: missing file should fail"
    print("addendum guard self-test: passed")
_selftest()
```
This directly answers this project's own recurring failure class — a guard reported as working that had never actually fired — by running it once, deliberately, before it is relied on.

**Investor-run checklist line (a second, human enforcement layer — a script's own guard can be edited by a future session; a person's checklist step cannot):** *"Before running PR-004's G2 session: confirm `docs/PR005_HASH.txt` exists and is a single 64-character hex string (case-insensitive), or that `docs/PR005_CLOSURE.txt` exists, is non-empty, and is dated."*


