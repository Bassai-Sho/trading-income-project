"""Tests for the Alpaca download path of market_data_store.py (no network, no keys).

Covers: SIP feed requested by default; missing keys / 403 stop the download
instead of being logged per chunk; transient chunk failures are reported;
prev-day levels carry across chunk boundaries; reading a range that spans a
DST change works.
"""
import os, sys, types
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import market_data_store as mds
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.common.exceptions import APIError

CALLS = []

def _fake_bars(req):
    """1-min regular-session bars for each weekday in [start, end], Alpaca-shaped."""
    CALLS.append(req)
    days = pd.bdate_range(req.start.date(), req.end.date())
    frames = []
    for i, d in enumerate(days):
        idx = pd.date_range(f"{d.date()} 09:30", f"{d.date()} 15:59", freq="1min",
                            tz="America/New_York").tz_convert("UTC")
        base = 400 + i
        n = len(idx)
        frames.append(pd.DataFrame({
            "open": base, "high": [base + 1 + (k % 3) * 0.1 for k in range(n)],
            "low": base - 1, "close": [base + 0.5 + (k % 5) * 0.01 for k in range(n)],
            "volume": 1000, "trade_count": 10, "vwap": base,
        }, index=pd.MultiIndex.from_product([[req.symbol_or_symbols], idx],
                                             names=["symbol", "timestamp"])))
    return types.SimpleNamespace(df=pd.concat(frames) if frames else pd.DataFrame())

@pytest.fixture
def store(tmp_path, monkeypatch):
    CALLS.clear()
    monkeypatch.setenv("ALPACA_API_KEY", "k"); monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setattr(StockHistoricalDataClient, "get_stock_bars",
                        lambda self, req: _fake_bars(req))
    monkeypatch.setattr(mds.MarketDataStore, "_fetch_vix_daily", lambda *a, **k: {})
    monkeypatch.setattr(mds.time, "sleep", lambda s: None)
    return mds.MarketDataStore(str(tmp_path / "m.db"))

def test_sip_feed_by_default(store):
    store.download_and_store("SPY", date(2024, 1, 2), date(2024, 1, 5), vix_daily={})
    assert CALLS and all(c.feed.value == "sip" for c in CALLS)

def test_missing_secret_stops_download(store, monkeypatch):
    monkeypatch.delenv("ALPACA_SECRET_KEY")
    with pytest.raises(mds.AlpacaFatalError):
        store.download_and_store("SPY", date(2024, 1, 1), date(2024, 12, 31), vix_daily={})

def test_403_stops_after_first_chunk(store, monkeypatch):
    def deny(self, req):
        CALLS.append(req)
        e = APIError('{"message":"forbidden"}'); e.__dict__["_http_error"] = None
        raise e
    monkeypatch.setattr(StockHistoricalDataClient, "get_stock_bars", deny)
    monkeypatch.setattr(APIError, "status_code", property(lambda self: 403))
    with pytest.raises(mds.AlpacaFatalError):
        store.download_and_store("SPY", date(2024, 1, 1), date(2024, 12, 31), vix_daily={})
    assert len(CALLS) == 1          # old code tried all 4 chunks and "finished"

def test_transient_failure_reported(store, monkeypatch):
    real = _fake_bars
    def flaky(self, req):
        if req.start.month == 2:
            raise ConnectionError("reset by peer")
        return real(req)
    monkeypatch.setattr(StockHistoricalDataClient, "get_stock_bars", flaky)
    r = store.download_and_store("SPY", date(2024, 1, 1), date(2024, 3, 31),
                                 vix_daily={}, chunk_months=1)
    assert [c["start"] for c in r["failed_chunks"]] == ["2024-02-01"]
    assert r["total_sessions"] > 0

def test_prev_close_carries_across_chunks(store):
    store.download_and_store("SPY", date(2024, 1, 1), date(2024, 2, 29),
                             vix_daily={}, chunk_months=1)
    ctx = store.get_session_context("SPY", "2024-02-01")   # first day of chunk 2
    assert ctx["prev_close"] is not None and ctx["pdh"] is not None
    assert ctx["gap_pct"] is not None

def test_range_across_dst_reads_back(store):
    store.download_and_store("SPY", date(2024, 3, 1), date(2024, 3, 29),
                             vix_daily={}, chunk_months=1)          # DST starts 10 Mar
    df = store.get_bars_range("SPY", date(2024, 3, 1), date(2024, 3, 29))
    assert len(df) > 0
    assert str(df.index.tz) == "America/New_York"
    assert df.index.min().strftime("%H:%M") == "09:30"
    assert set(df.index.strftime("%H:%M").map(lambda t: t >= "09:30")) == {True}
