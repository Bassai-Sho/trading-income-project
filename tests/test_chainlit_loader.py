"""
tests/test_chainlit_loader.py -- call-time imports of sibling modules must work after Chainlit has loaded the app.

Found 21 Sep 2026: chainlit.config.load_module() inserts src/ at sys.path[0] only while it executes chainlit_app.py and
pops it afterwards. The app's old guard `if _src not in sys.path` never fired (Chainlit had just added it), so at request
time `import domain_telemetry` (the quarantine lookup) and `from tool_runner import dispatch_tool` (the run_tool direct
fallback) raised ModuleNotFoundError. The live log said: "quarantine lookup FAILED: ModuleNotFoundError: No module named
'domain_telemetry'". Plain imports in tests hide this, so this test loads the app with Chainlit's own loader.

Run from anywhere:   python tests/test_chainlit_loader.py     (needs chainlit, openai, httpx)
"""
import os
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
os.chdir(tempfile.mkdtemp())            # chainlit + the telemetry DB write into the cwd

from chainlit.config import load_module      # noqa: E402  (the loader `chainlit run` uses)

target = str(SRC / "chainlit_app.py")
load_module(target)
app = sys.modules[target]


def main() -> None:
    assert str(SRC) in sys.path, "src/ must still be on sys.path after Chainlit finished loading the app"

    import importlib
    for name in ("domain_telemetry", "tool_runner"):
        importlib.import_module(name)                     # a call-time import, as run_tool / _is_quarantined do

    import domain_telemetry as dt
    url = "https://stocktwits.com/news-articles/x"
    assert not app._is_quarantined(url)
    dt.record_domain_result(url, False, "HTTP_403"); dt.record_domain_result(url, False, "HTTP_403")
    assert app._is_quarantined(url), "the DB quarantine lookup must work inside the Chainlit process"
    print("PASS: after Chainlit's loader, call-time imports work and the DB quarantine lookup sees a real quarantine")


if __name__ == "__main__":
    main()
