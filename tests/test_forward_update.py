"""Forward-data fixes (28 Sep 2026): last_complete_session, the SIP end-time cap in
_fetch_from_alpaca, update()'s complete-sessions-only window, and the runner logging a
failing store update. Every test calls the REAL function. Expected dates come from an
independent check against the exchange-calendar library, not from the code under test."""
import logging
import sqlite3
import subprocess
import sys
import types
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import market_data_store as mds
import runner
from alpaca.data.historical import StockHistoricalDataClient


def utc(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


# ── last_complete_session ─────────────────────────────────────────────────────

@pytest.mark.parametrize("now, expected", [
    (utc(2026, 9, 28, 20, 30), date(2026, 9, 28)),   # Mon 16:30 ET: after close+16 -> today
    (utc(2026, 9, 28, 20, 10), date(2026, 9, 25)),   # Mon 16:10 ET: before close+16 -> Friday
    (utc(2026, 9, 28, 9, 0),   date(2026, 9, 25)),   # Mon pre-open -> Friday
    (utc(2026, 9, 28, 17, 0),  date(2026, 9, 25)),   # Mon MID-SESSION -> Friday, never today
    (utc(2026, 9, 26, 15, 0),  date(2026, 9, 25)),   # Saturday -> Friday
    (utc(2025, 11, 28, 18, 10), date(2025, 11, 26)), # half-day, 13:10 ET: not yet +16 (prev session)
    (utc(2025, 11, 28, 18, 20), date(2025, 11, 28)), # half-day, 13:20 ET: closed at 13:00 -> complete
    (utc(2026, 1, 1, 20, 0),   date(2025, 12, 31)),  # New Year's Day holiday
    (utc(2026, 11, 2, 21, 10), date(2026, 10, 30)),  # first Monday after DST ends: 16:10 ET
    (utc(2026, 11, 2, 21, 20), date(2026, 11, 2)),   # ... and 16:20 ET (a fixed UTC-4 offset gets these wrong)
])
def test_last_complete_session(now, expected):
    assert mds.last_complete_session(now) == expected


def test_last_complete_session_boundary_is_inclusive():
    """Exactly close + 16 minutes (16:16 ET = 20:16 UTC in summer) counts as complete;
    one minute earlier does not."""
    assert mds.last_complete_session(utc(2026, 9, 28, 20, 16)) == date(2026, 9, 28)
    assert mds.last_complete_session(utc(2026, 9, 28, 20, 15)) == date(2026, 9, 25)


def test_last_complete_session_accepts_a_naive_datetime_as_utc():
    assert mds.last_complete_session(datetime(2026, 9, 28, 20, 30)) == date(2026, 9, 28)


# ── the SIP end-time cap in _fetch_from_alpaca ────────────────────────────────

REQS = []


@pytest.fixture
def store(tmp_path, monkeypatch):
    REQS.clear()

    def fake(self, req):
        REQS.append(req)
        return types.SimpleNamespace(df=pd.DataFrame())

    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setattr(StockHistoricalDataClient, "get_stock_bars", fake)
    monkeypatch.setattr(mds, "_utcnow", lambda: utc(2026, 9, 28, 10, 39))      # the failing morning
    return mds.MarketDataStore(str(tmp_path / "live.db"), sealed=False)


def test_sip_request_never_ends_inside_the_last_16_minutes(store):
    """The real bug: end = end-of-today (23:59:59 UTC) is later than 'now', so a plan that only
    permits SIP data older than ~15 minutes answers HTTP 403."""
    store._fetch_from_alpaca("SPY", date(2026, 7, 1), date(2026, 9, 28), feed="sip")
    assert REQS[0].end == datetime(2026, 9, 28, 10, 23)        # now (10:39) minus 16 minutes
    assert REQS[0].end < datetime(2026, 9, 28, 23, 59, 59)


def test_historical_sip_requests_are_left_alone(store):
    store._fetch_from_alpaca("SPY", date(2024, 1, 2), date(2024, 12, 31), feed="sip")
    assert REQS[0].end == datetime.combine(date(2024, 12, 31), datetime.max.time())


def test_other_feeds_are_not_capped(store):
    store._fetch_from_alpaca("SPY", date(2026, 7, 1), date(2026, 9, 28), feed="iex")
    assert REQS[0].end == datetime.combine(date(2026, 9, 28), datetime.max.time())


def test_a_window_that_is_empty_after_capping_makes_no_request(store, monkeypatch):
    """start after the capped end: asking for today's date at 00:05 UTC, so the capped end
    (23:49 UTC yesterday) is before the start (00:00 today)."""
    monkeypatch.setattr(mds, "_utcnow", lambda: utc(2026, 9, 28, 0, 5))
    out = store._fetch_from_alpaca("SPY", date(2026, 9, 28), date(2026, 9, 28), feed="sip")
    assert REQS == [] and out.empty and list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


# ── update(): complete sessions only ──────────────────────────────────────────

def _live_store_with_last_bar(tmp_path, monkeypatch, last_date):
    s = mds.MarketDataStore(str(tmp_path / "live.db"), sealed=False)
    with sqlite3.connect(s.db_path) as c:
        c.execute("INSERT INTO market_bars (ticker, ts, ts_date, ts_time, open, high, low, close, volume, "
                  "vwap, bar_interval, quality_flag) VALUES ('SPY',?,?,?,?,?,?,?,?,?,?,?)",
                  (f"{last_date}T09:30:00-04:00", last_date, "09:30", 1, 2, 1, 1, 10, 1, "1m", 0))
    calls = {"vix": [], "dl": []}
    monkeypatch.setattr(mds.MarketDataStore, "_fetch_vix_daily",
                        lambda self, a, b: calls["vix"].append((a, b)) or {})
    monkeypatch.setattr(mds.MarketDataStore, "download_and_store",
                        lambda self, t, a, b, vix=None, **k: calls["dl"].append((t, a, b)) or {"ok": True})
    return s, calls


def test_update_stops_at_the_last_complete_session_not_today(tmp_path, monkeypatch):
    s, calls = _live_store_with_last_bar(tmp_path, monkeypatch, "2026-06-30")
    s.update("SPY", now=utc(2026, 9, 28, 9, 0))                     # Monday pre-open
    assert calls["dl"] == [("SPY", date(2026, 7, 1), date(2026, 9, 25))]
    assert calls["vix"] == [(date(2026, 6, 30), date(2026, 9, 25))]


def test_update_mid_session_never_stores_a_half_finished_day(tmp_path, monkeypatch):
    s, calls = _live_store_with_last_bar(tmp_path, monkeypatch, "2026-09-25")
    r = s.update("SPY", now=utc(2026, 9, 28, 17, 0))                 # Monday 13:00 ET, market open
    assert r == {"message": "Already up to date", "latest": "2026-09-25"} and calls["dl"] == []


def test_update_after_the_close_includes_todays_complete_session(tmp_path, monkeypatch):
    s, calls = _live_store_with_last_bar(tmp_path, monkeypatch, "2026-09-25")
    s.update("SPY", now=utc(2026, 9, 28, 20, 30))                    # the runner's 16:30 ET job
    assert calls["dl"] == [("SPY", date(2026, 9, 26), date(2026, 9, 28))]


def test_update_with_no_stored_data_still_reports_it(tmp_path):
    s = mds.MarketDataStore(str(tmp_path / "live.db"), sealed=False)
    assert "error" in s.update("SPY", now=utc(2026, 9, 28, 20, 30))


# ── runner: a failing store update must leave a trace ─────────────────────────

class _Done:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_runner_logs_a_failing_store_update_with_its_error(monkeypatch, caplog):
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: _Done(
        1, "", "Traceback ...\nAlpacaFatalError: subscription does not permit querying recent SIP data"))
    with caplog.at_level(logging.INFO):
        runner.launch_store_update()
    assert "store_update FAILED (exit 1)" in caplog.text
    assert "does not permit querying recent SIP data" in caplog.text


def test_runner_is_quiet_about_failure_when_the_update_succeeds(monkeypatch, caplog):
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: _Done(0, "Store update: {'ok': 1}", ""))
    with caplog.at_level(logging.INFO):
        runner.launch_store_update()
    assert "FAILED" not in caplog.text and "Store update: {'ok': 1}" in caplog.text


def test_runner_logs_a_timeout_and_uses_the_longer_limit(monkeypatch, caplog):
    seen = {}

    def boom(*a, **k):
        seen["timeout"] = k.get("timeout")
        raise subprocess.TimeoutExpired("cmd", k.get("timeout"))

    monkeypatch.setattr(runner.subprocess, "run", boom)
    with caplog.at_level(logging.INFO):
        runner.launch_store_update()
    assert seen["timeout"] == runner.STORE_UPDATE_TIMEOUT_S == 600
    assert "Store update failed" in caplog.text
