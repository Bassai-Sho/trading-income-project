"""P2-125 universe stage 3: 1-min sessions for selected stock-days only,
stored through MarketDataStore (contiguous=False)."""
import sqlite3, sys, types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import universe_minutes as um
from market_data_store import MarketDataStore
from universe_minutes import MinuteLoader
from universe_select import UniverseSelector
from universe_store import UniverseStore

DAYS = ["2019-03-12", "2019-09-17", "2019-11-29"]          # scattered; the last is a 13:00 close


class _Client:
    def __init__(self, invalid=(), absent=()):
        self.calls, self.invalid, self.absent = [], set(invalid), set(absent)

    def get_stock_bars(self, req):
        self.calls.append(req)
        bad = [s for s in req.symbol_or_symbols if s in self.invalid]
        if bad:
            from alpaca.common.exceptions import APIError
            raise APIError('{"message":"invalid symbol: %s"}' % bad[0])
        t0 = pd.Timestamp(req.start)
        t0 = (t0.tz_localize("UTC") if t0.tzinfo is None else t0).tz_convert("America/New_York")
        idx = pd.date_range(t0, periods=390, freq="1min")        # vendor sends a full day
        frames = []
        for i, s in enumerate(req.symbol_or_symbols):
            if s in self.absent:
                continue
            px = 50.0 + 10 * i + np.arange(390) * 0.01
            frames.append(pd.DataFrame({"open": px, "high": px + 0.05, "low": px - 0.05, "close": px,
                                        "volume": 1000.0},
                                       index=pd.MultiIndex.from_arrays(
                                           [[s] * 390, idx.tz_convert("UTC")], names=["symbol", "timestamp"])))
        return types.SimpleNamespace(df=pd.concat(frames) if frames else pd.DataFrame())


@pytest.fixture
def loader(tmp_path, monkeypatch):
    monkeypatch.setattr(um._time, "sleep", lambda s: None)
    u = UniverseStore(str(tmp_path / "u.db"))
    UniverseSelector(u)                                          # creates selection tables
    with sqlite3.connect(u.db_path) as c:
        c.executemany("INSERT INTO universe_assets VALUES (?,?,?,?,1,'us_equity','')",
                      [("AAA", "A Corp", "NYSE", "active"), ("AET_DELISTED", "Aetna", "NYSE", "inactive"),
                       ("GONE", "Gone Inc", "NYSE", "inactive")])
        c.executemany("INSERT INTO universe_selection VALUES (?,?,5,?,1,1)",
                      [(d, s, r) for d in DAYS for r, s in enumerate(["AAA", "AET_DELISTED", "GONE"], 1)])
    return MinuteLoader(u, MarketDataStore(str(tmp_path / "m.db")))


def test_stores_selected_stock_days_under_asset_symbols(loader):
    cl = _Client(absent={"GONE"})
    r = loader.download(cl, date(2019, 1, 1), date(2019, 12, 31))
    assert r["days"] == 3 and r["missing_stock_days"] == 3 and not r["failed_days"]
    assert len(cl.calls) == 3 and set(cl.calls[0].symbol_or_symbols) == {"AAA", "AET", "GONE"}
    with sqlite3.connect(loader.m.db_path) as c:
        n = dict(c.execute("SELECT ticker, COUNT(DISTINCT ts_date) FROM market_bars GROUP BY 1"))
        ctx = c.execute("SELECT prev_close, gap_pct, pdh FROM session_context").fetchall()
        half = c.execute("SELECT COUNT(*), MAX(ts_time) FROM market_bars WHERE ts_date='2019-11-29' "
                         "AND ticker='AAA'").fetchone()
    assert n == {"AAA": 3, "AET_DELISTED": 3}
    assert all(row == (None, None, None) for row in ctx)       # scattered days: no prior-day levels
    assert half == (210, "12:59")                              # early close trimmed by the calendar


def test_resume_per_day_and_invalid_symbol_dropped(loader):
    loader.download(_Client(invalid={"GONE"}), date(2019, 1, 1), date(2019, 6, 30))
    cl = _Client()
    r = loader.download(cl, date(2019, 1, 1), date(2019, 12, 31))
    assert r["days"] == 2 and len(cl.calls) == 2                # March already done
    assert loader.status()["days_done"] == 3


def test_contiguous_default_still_carries_prior_levels(tmp_path):
    """The default MarketDataStore behaviour is unchanged."""
    m = MarketDataStore(str(tmp_path / "c.db"))
    idx = pd.date_range("2019-03-11 09:30", periods=390, freq="1min", tz="America/New_York")
    idx = idx.append(pd.date_range("2019-03-12 09:30", periods=390, freq="1min", tz="America/New_York"))
    px = 100 + np.arange(780) * 0.001
    df = pd.DataFrame({"Open": px, "High": px + 0.05, "Low": px - 0.05, "Close": px, "Volume": 1000.0}, index=idx)
    m._store_bars("SPY", df, {})
    assert m.get_session_context("SPY", "2019-03-12")["prev_close"] is not None
