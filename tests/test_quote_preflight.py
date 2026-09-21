"""
tests/test_quote_preflight.py -- P2-097: a labelled yfinance price line goes FIRST in web_search results.

Why: "What is SPY trading at today?" used to depend on some news article happening to quote a price (one live run got
$766.65 from one article and $766.69 from another; earlier runs found none). yfinance is stubbed here, so the test
needs no network.

Run from anywhere:   python tests/test_quote_preflight.py
"""
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.chdir(tempfile.mkdtemp())                       # tool_runner initialises a telemetry DB in the cwd
os.environ.pop("BRAVE_SEARCH_API_KEY", None)

import tool_runner as tr                           # noqa: E402

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:                                  # pragma: no cover
    from datetime import timedelta, timezone
    ET = timezone(timedelta(hours=-4))

MON_1004 = datetime(2026, 9, 21, 10, 4, tzinfo=ET)     # Monday, inside market hours
MON_1730 = datetime(2026, 9, 21, 17, 30, tzinfo=ET)    # Monday, after the close
SAT_1100 = datetime(2026, 9, 26, 11, 0, tzinfo=ET)     # Saturday


class FakeTicker:
    table = {"SPY": (766.65, 761.69), "QQQ": (600.0, 590.0), "AAPL": (float("nan"), 100.0), "HANG": None}
    def __init__(self, sym): self.sym = sym
    @property
    def fast_info(self):
        row = self.table[self.sym]                  # KeyError for unknown symbols
        if row is None:
            time.sleep(5)                           # a hung network call
        return {"last_price": row[0], "previous_close": row[1]}
    news = []


def install_fake(ticker_cls=FakeTicker):
    tr.yf = SimpleNamespace(Ticker=ticker_cls)
    tr._YFINANCE = True


def main() -> None:
    install_fake()

    # --- ticker detection: case-aware, no "PRICE"/"TODAY" false positives -------------------------------
    d = tr._detect_tickers
    assert d("SPY price today") == ["SPY"], d("SPY price today")
    assert d("what is the price of SPY today") == ["SPY"]
    assert d("$aapl quote") == ["AAPL"]
    assert d("spy price now") == ["SPY"], "unambiguous lower-case ETF"
    assert d("SPY vs QQQ price") == ["SPY", "QQQ"]
    assert d("SPY PRICE TODAY") == ["SPY"], "intent words typed in capitals are not tickers"
    assert d("what is the stock market doing today") == []
    assert d("ORB VWAP RSI strategy") == [], "the project's own jargon is not a ticker"

    # --- the quote line -------------------------------------------------------------------------------------
    q = tr._yfinance_quote_line("SPY price today", now=MON_1004)
    assert q.startswith("[Yahoo Finance quote] SPY: $766.65 (+0.65% vs previous close $761.69)"), q
    assert "retrieved 2026-09-21 10:04 ET" in q and "US market hours - quote may be delayed" in q, q
    assert q.endswith("URL: https://finance.yahoo.com/quote/SPY/"), q
    assert "outside US market hours" in tr._yfinance_quote_line("SPY price", now=MON_1730)
    assert "weekend" in tr._yfinance_quote_line("SPY price", now=SAT_1100)
    assert tr._yfinance_quote_line("what is SPY trading at", now=MON_1004).startswith("[Yahoo Finance quote] SPY:")
    assert tr._yfinance_quote_line("SPY paper trading strategy notes", now=MON_1004) == "", "'trading' alone is not price intent"
    assert tr._yfinance_quote_line("SPY closed above VWAP last week", now=MON_1004) == "", "'last'/'closed' alone are not price intent"
    assert tr._yfinance_quote_line("SPY momentum research paper", now=MON_1004) == "", "no price intent -> no quote"
    assert tr._yfinance_quote_line("XXXX price", now=MON_1004) == "", "unknown symbol fails quiet"
    assert tr._yfinance_quote_line("AAPL price", now=MON_1004) == "", "NaN price is rejected"
    two = tr._yfinance_quote_line("SPY vs QQQ price", now=MON_1004).splitlines()
    assert len(two) == 2 and two[1].startswith("[Yahoo Finance quote] QQQ: $600.00 (+1.69%"), two

    # --- bounded, fail-quiet ---------------------------------------------------------------------------------
    t0 = time.time()
    assert tr._call_with_deadline(lambda: time.sleep(3) or "late", 0.3) == ""
    assert time.time() - t0 < 1.5, "must return at the deadline, not wait for the worker"
    assert tr._call_with_deadline(lambda: 1 / 0, 1) == ""

    # --- web_search integration: quote FIRST, then headlines, then results; on the Brave path too -------------
    tr._ddg_search = lambda query, n=10: "DDG RESULTS"
    tr._yfinance_headlines = lambda query, max_items=3: "HEADLINES"
    out = tr.tool_web_search("SPY price today")
    assert out.startswith("[Yahoo Finance quote] SPY:"), out[:80]
    assert out.index("[Yahoo Finance quote]") < out.index("HEADLINES") < out.index("DDG RESULTS"), out
    os.environ["BRAVE_SEARCH_API_KEY"] = "test"
    tr._brave_search = lambda query, n, key: "BRAVE RESULTS"
    out = tr.tool_web_search("SPY price today")
    assert out.startswith("[Yahoo Finance quote] SPY:") and out.endswith("BRAVE RESULTS"), "prefix must not be dropped on the Brave path"
    os.environ.pop("BRAVE_SEARCH_API_KEY")
    assert tr.tool_web_search("SPY momentum research paper") == "HEADLINES\n\n---\n\nDDG RESULTS", "no quote without price intent"

    # --- a hung or broken yfinance never blocks or breaks the search ------------------------------------------
    tr.QUOTE_TIMEOUT = 0.3
    t0 = time.time()
    out = tr.tool_web_search("HANG price today")
    assert time.time() - t0 < 2.0 and "DDG RESULTS" in out, (time.time() - t0, out)

    class Boom:
        def __init__(self, sym): raise RuntimeError("network down")
    install_fake(Boom)
    assert "DDG RESULTS" in tr.tool_web_search("SPY price today")
    tr._YFINANCE = False
    assert tr._yfinance_quote_line("SPY price today") == "", "no yfinance installed -> no quote, no error"
    print("PASS: quote line is labelled and first (Brave and DDG paths); no-intent, unknown, NaN, hung and broken cases fail quiet")


if __name__ == "__main__":
    main()
