"""
tests/test_with_timeout.py -- P2-100: _with_timeout must return at the deadline, not when the worker finishes.

The old helper ran the call in `with ThreadPoolExecutor(max_workers=1) as ex:`; the executor's __exit__ waits for the
worker, so `_with_timeout(sleep 4, timeout=1)` raised after 4.0 s and TOOL_TIMEOUT could not protect web_search /
fetch_url from a hung call. Also covers the pre-flight concurrency added with it (quote / headlines run alongside the
search instead of before it).

Run from anywhere:   python tests/test_with_timeout.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.chdir(tempfile.mkdtemp())                       # tool_runner initialises a telemetry DB in the cwd
os.environ.pop("BRAVE_SEARCH_API_KEY", None)

import tool_runner as tr                           # noqa: E402


def elapsed(fn):
    t0 = time.time()
    try:
        return fn(), time.time() - t0
    except Exception as e:                         # noqa: BLE001
        return e, time.time() - t0


def main() -> None:
    # --- behaviour that must not change ---------------------------------------------------------------------
    assert tr._with_timeout(lambda: 42, timeout=2) == 42
    assert tr._with_timeout(lambda a, b=0: a + b, args=(1,), kwargs={"b": 2}, timeout=2) == 3
    assert tr._with_timeout(lambda: None, timeout=2) is None, "a None result is a result"
    err, _ = elapsed(lambda: tr._with_timeout(lambda: (_ for _ in ()).throw(ValueError("boom")), timeout=2))
    assert isinstance(err, ValueError) and str(err) == "boom", f"the tool's own exception must propagate unchanged: {err!r}"

    # --- the fix: the caller gets control back at the deadline ---------------------------------------------
    err, dt = elapsed(lambda: tr._with_timeout(lambda: time.sleep(4), timeout=1))
    assert isinstance(err, TimeoutError), err
    assert dt < 1.6, f"asked for 1 s, took {dt:.1f} s (the old helper took ~4 s)"

    # --- and it reaches the tools: web_search returns 'timed out' on time, not when ddgs finally returns -----
    tr._YFINANCE = False
    tr._ddg_search = lambda query, n=10: time.sleep(3) or "late results"
    tr.TOOL_TIMEOUT = 0.5
    out, dt = elapsed(lambda: tr.tool_web_search("anything"))
    assert isinstance(out, str) and out.startswith("Search timed out"), out
    assert dt < 1.3, f"web_search took {dt:.1f} s with a 0.5 s TOOL_TIMEOUT"
    tr.TOOL_TIMEOUT = 45

    # --- pre-flights run concurrently with the search ------------------------------------------------------
    tr._YFINANCE = True
    tr._ddg_search = lambda query, n=10: time.sleep(1.5) or "DDG RESULTS"
    tr._yfinance_quote_line = lambda query, now=None: time.sleep(1.2) or "QUOTE LINE"
    tr._yfinance_headlines = lambda query, max_items=3: ""
    out, dt = elapsed(lambda: tr.tool_web_search("SPY price today"))
    assert out == "QUOTE LINE\n\n---\n\nDDG RESULTS", out
    assert dt < 2.2, f"quote (1.2 s) + search (1.5 s) took {dt:.1f} s -- they must overlap, not add up"

    # a hung pre-flight costs at most what is left of QUOTE_TIMEOUT after the search, and the search still returns
    tr._yfinance_quote_line = lambda query, now=None: time.sleep(5) or "never"
    tr.QUOTE_TIMEOUT = 0.4
    tr._ddg_search = lambda query, n=10: "DDG RESULTS"
    out, dt = elapsed(lambda: tr.tool_web_search("SPY price today"))
    assert out == "DDG RESULTS" and dt < 1.2, (out, dt)
    print("PASS: _with_timeout returns at the deadline (1 s on a 4 s call), keeps results and exceptions, and the "
          "pre-flights overlap with the search")


if __name__ == "__main__":
    main()
