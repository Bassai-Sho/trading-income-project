"""
tests/test_fetch_stubs.py -- fetch_url "stub" results must NOT count as successfully read pages.

Found 21 Sep 2026 from a live Chainlit run: a quarantined domain returned
"[Skipped: domain quarantined ...]" in 0.0s and was shown as a successful (checkmark) page and counted
toward the "3 sources" target, so synthesis got fewer real articles plus a stub string.

Run from anywhere:   python tests/test_fetch_stubs.py     (needs chainlit, openai, httpx)
"""
import os
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
os.chdir(tempfile.mkdtemp())            # chainlit writes .chainlit/ into the cwd on import

import chainlit_app as app              # noqa: E402

STUBS = [
    "[Skipped: domain quarantined due to persistent 403/paywall -- trying next source]",
    "[No article body extracted from https://example.com/x]",
    "HTTP fetch failed: 403 Client Error: HTTP Forbidden for url: https://example.com/x",
    "Search failed: timed out",
    "",
]
REAL = [
    "SPY closed at $761.69, up 0.4%, as analysts at several banks raised year-end targets ...",
    "Stocks ended higher on Friday ...",
]


def main() -> None:
    for s in STUBS:
        assert app._tool_failed(s), f"must be treated as a failure/stub: {s[:60]!r}"
    for s in REAL:
        assert not app._tool_failed(s), f"real content must not be flagged: {s[:60]!r}"

    # Drift guard: the exact stub wording lives in tool_runner.py. If someone rewords it, this fails
    # and the prefixes in chainlit_app._TOOL_FAIL_PREFIXES must be updated to match.
    runner = (SRC / "tool_runner.py").read_text(encoding="utf-8")
    for needle in ("[Skipped: domain quarantined", "[No article body extracted"):
        assert needle in runner, f"tool_runner.py no longer contains {needle!r} -- update _TOOL_FAIL_PREFIXES"
    print("PASS: quarantine / empty-extraction / HTTP-failure stubs are failures; real page text is not")


if __name__ == "__main__":
    main()
