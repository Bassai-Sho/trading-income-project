"""Quality checks, bookkeeping and adjustment handling in market_data_store.py.
No network, no keys: Alpaca's client is stubbed (see test_market_data_store_alpaca)."""
import sqlite3, sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import market_data_store as mds
from test_market_data_store_alpaca import store, CALLS, _fake_bars   # noqa: F401 (fixture)


def _day(d="2024-03-12", n=390, seed=1, price=500.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{d} 09:30", periods=n, freq="1min", tz="America/New_York")
    c = price + rng.normal(0, 0.05, n).cumsum()
    o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.normal(0, 0.03, n)) + 0.02
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + w,
                         "Low": np.minimum(o, c) - w, "Close": c,
                         "Volume": 1000.0}, index=idx)


def _types(issues):
    return {i["issue_type"] for i in issues}


# ── Checks ──────────────────────────────────────────────────────────────────

def test_isolated_spike_is_critical(store):
    df = _day(); df.iloc[200, df.columns.get_loc("High")] += 2.0     # bad print, reverts
    iss = store._validate_day(df, date(2024, 3, 12))
    assert "phantom_spike" in _types(iss) and store._is_quality_ok(iss) == 0
    assert "12:50" in next(i["detail"] for i in iss if i["issue_type"] == "phantom_spike")

def test_wide_opening_bar_is_not_a_spike(store):
    """The old mean+5 sigma test flagged this on most real sessions."""
    df = _day(); df.iloc[0, df.columns.get_loc("High")] += 3.0
    df.iloc[0, df.columns.get_loc("Low")] -= 3.0
    iss = store._validate_day(df, date(2024, 3, 12))
    assert "phantom_spike" not in _types(iss) and store._is_quality_ok(iss) == 1

def test_genuine_fast_move_is_not_a_spike(store):
    df = _day()
    k = np.arange(390); jump = np.where(k >= 150, 3.0 * np.minimum(k - 150, 10) / 10, 0)
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col] + jump                                   # sustained $3 move
    assert "phantom_spike" not in _types(store._validate_day(df, date(2024, 3, 12)))

def test_early_close_is_info_but_short_full_day_is_critical(store):
    early = _day(n=210)                                            # 09:30-12:59
    iss = store._validate_day(early, date(2024, 11, 29))
    assert "early_close" in _types(iss) and store._is_quality_ok(iss) == 1
    gappy = _day().iloc[::2]                                       # 195 bars to 15:58
    iss = store._validate_day(gappy, date(2024, 3, 12))
    assert "bar_count_low" in _types(iss) and store._is_quality_ok(iss) == 0

def test_orb_anomaly_and_few_stale_bars_are_info_only(store):
    df = _day()
    for i in range(300, 320):                                      # 20 stale bars
        df.iloc[i, :4] = df.iloc[i]["Close"]
    iss = store._validate_day(df, date(2024, 3, 12))
    assert "stale_bars" in _types(iss) and store._is_quality_ok(iss) == 1


# ── Bookkeeping ─────────────────────────────────────────────────────────────

def _q(store, sql, *a):
    with sqlite3.connect(store.db_path) as c:
        return c.execute(sql, a).fetchall()

def test_rerun_adds_nothing_and_counts_are_real(store):
    r1 = store.download_and_store("SPY", date(2024, 1, 2), date(2024, 1, 31), vix_daily={})
    rows1 = _q(store, "SELECT COUNT(*) FROM data_quality_log")[0][0]
    r2 = store.download_and_store("SPY", date(2024, 1, 2), date(2024, 1, 31), vix_daily={})
    assert r1["total_bars"] == 390 * r1["total_sessions"] > 0
    assert r2["total_bars"] == 0                                   # nothing new inserted
    assert _q(store, "SELECT COUNT(*) FROM data_quality_log")[0][0] == rows1
    meta = store.status()["meta"]
    assert meta["total_bars"] == _q(store, "SELECT COUNT(*) FROM market_bars")[0][0]
    assert meta["total_sessions"] == r1["total_sessions"]

def test_existing_duplicate_quality_rows_are_removed_on_open(tmp_path):
    db = str(tmp_path / "old.db")
    with sqlite3.connect(db) as c:                                 # pre-patch table, no UNIQUE
        c.execute("CREATE TABLE data_quality_log (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "ticker TEXT, session_date TEXT, issue_type TEXT, detail TEXT, logged_at TEXT)")
        c.executemany("INSERT INTO data_quality_log (ticker,session_date,issue_type) VALUES (?,?,?)",
                      [("SPY", "2019-01-02", "phantom_hl")] * 2 + [("SPY", "2019-01-03", "stale_bars")])
    mds.MarketDataStore(db)
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM data_quality_log").fetchone()[0] == 2


# ── Adjustment ──────────────────────────────────────────────────────────────

def test_raw_adjustment_requested_by_default(store):
    store.download_and_store("SPY", date(2024, 1, 2), date(2024, 1, 5), vix_daily={})
    assert CALLS and all(c.adjustment.value == "raw" for c in CALLS)
    assert store.stored_adjustment("SPY") == "raw"

def test_refuses_to_mix_adjustments_until_wiped(store):
    store.download_and_store("NVDA", date(2024, 1, 2), date(2024, 1, 5),
                             vix_daily={}, adjustment="all")
    with pytest.raises(mds.AlpacaFatalError, match="--wipe"):
        store.download_and_store("NVDA", date(2024, 1, 8), date(2024, 1, 12), vix_daily={})
    store.wipe("NVDA")
    assert store.stored_adjustment("NVDA") is None
    store.download_and_store("NVDA", date(2024, 1, 8), date(2024, 1, 12), vix_daily={})
    assert store.stored_adjustment("NVDA") == "raw"

def test_legacy_bars_without_setting_count_as_all(store):
    store.download_and_store("QQQ", date(2024, 1, 2), date(2024, 1, 5), vix_daily={})
    _q(store, "DELETE FROM store_settings")                        # simulate pre-patch DB
    assert store.stored_adjustment("QQQ") == "all"

def test_split_night_nulls_prior_day_levels_but_keeps_session(store, monkeypatch):
    real = _fake_bars
    def split(req):
        out = real(req)
        idx = out.df.index.get_level_values(1)
        after = idx >= pd.Timestamp("2024-01-04", tz="UTC")
        for col in ("open", "high", "low", "close"):
            out.df[col] = out.df[col].astype(float)
            out.df.loc[after, col] = out.df.loc[after, col] / 4.0  # 4:1 split overnight
        return out
    from alpaca.data.historical import StockHistoricalDataClient
    monkeypatch.setattr(StockHistoricalDataClient, "get_stock_bars", lambda self, req: split(req))
    store.download_and_store("TSLA", date(2024, 1, 2), date(2024, 1, 5), vix_daily={})
    ctx = store.get_session_context("TSLA", "2024-01-04")
    assert ctx["prev_close"] is None and ctx["gap_pct"] is None and ctx["quality_ok"] == 1
    assert ("overnight_discontinuity",) in _q(
        store, "SELECT issue_type FROM data_quality_log WHERE session_date='2024-01-04'")
    assert ("overnight_discontinuity",) not in _q(         # the day before is untouched
        store, "SELECT issue_type FROM data_quality_log WHERE session_date='2024-01-03'")


# ── Revalidate / data_corrector guard ──────────────────────────────────────

def test_revalidate_clears_legacy_flags(store):
    store.download_and_store("SPY", date(2024, 1, 2), date(2024, 1, 12), vix_daily={})
    _q(store, "UPDATE session_context SET quality_ok=0")
    _q(store, "INSERT INTO data_quality_log (ticker,session_date,issue_type,detail) "
              "VALUES ('SPY','2024-01-03','phantom_hl','old test')")
    r = store.revalidate("SPY")
    assert r["quality_ok"] == r["sessions"] > 0
    assert _q(store, "SELECT COUNT(*) FROM data_quality_log WHERE issue_type='phantom_hl'")[0][0] == 0

def test_data_corrector_refuses_raw_bars(store):
    import data_corrector
    store.download_and_store("SPY", date(2024, 1, 2), date(2024, 1, 5), vix_daily={})
    with pytest.raises(RuntimeError, match="corrupt"):
        data_corrector.correct_store(store.db_path, "SPY", date(2024, 1, 2), date(2024, 1, 5))


# ── Exchange calendar: early closes and non-sessions ───────────────────────

def test_half_day_trimmed_at_real_close(store):
    """Alpaca returns after-hours prints after a 13:00 close; they are dropped."""
    store.download_and_store("SPY", date(2024, 11, 27), date(2024, 12, 2), vix_daily={})
    n, last = _q(store, "SELECT COUNT(*), MAX(ts_time) FROM market_bars "
                        "WHERE ticker='SPY' AND ts_date='2024-11-29'")[0]
    assert (n, last) == (210, "12:59")
    ctx = store.get_session_context("SPY", "2024-11-29")
    assert ctx["quality_ok"] == 1 and ctx["n_bars"] == 210
    assert ("early_close",) in _q(store, "SELECT issue_type FROM data_quality_log "
                                         "WHERE session_date='2024-11-29'")
    nxt = store.get_session_context("SPY", "2024-12-02")      # prev close = 12:59 bar
    assert nxt["prev_close"] == _q(store, "SELECT close FROM market_bars WHERE ticker='SPY' "
                                          "AND ts_date='2024-11-29' AND ts_time='12:59'")[0][0]

def test_non_session_days_are_not_stored(store):
    store.download_and_store("SPY", date(2024, 11, 26), date(2024, 11, 29), vix_daily={})
    assert _q(store, "SELECT COUNT(*) FROM market_bars WHERE ts_date='2024-11-28'")[0][0] == 0

def test_revalidate_trims_already_stored_post_close_bars(store, monkeypatch):
    # Simulate the old store: no trimming at download time.
    monkeypatch.setattr(mds, "session_close", lambda d: pd.Timestamp("16:00").time())
    store.download_and_store("SPY", date(2024, 11, 27), date(2024, 12, 2), vix_daily={})
    assert _q(store, "SELECT COUNT(*) FROM market_bars WHERE ts_date='2024-11-29'")[0][0] == 390
    monkeypatch.undo()
    r = store.revalidate("SPY")
    assert r["bars_trimmed"] == 180
    assert _q(store, "SELECT MAX(ts_time) FROM market_bars WHERE ts_date='2024-11-29'")[0][0] == "12:59"
    ctx = store.get_session_context("SPY", "2024-12-02")
    assert ctx["prev_close"] == _q(store, "SELECT close FROM market_bars WHERE ticker='SPY' "
                                          "AND ts_date='2024-11-29' AND ts_time='12:59'")[0][0]
    assert store.get_session_context("SPY", "2024-11-29")["quality_ok"] == 1
