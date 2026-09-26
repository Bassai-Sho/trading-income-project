"""PR-002 runner end to end on a small synthetic universe (no network)."""
import sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr002_runner as pr
from evaluation.costs import cost_r
from market_data_store import MarketDataStore
from synthetic_bars import noisy_days
from universe_select import UniverseSelector
from universe_store import UniverseStore


def test_data_error_rule():
    idx = pd.date_range("2019-03-12 09:30", periods=3, freq="1min", tz="America/New_York")
    ok = pd.DataFrame({"Open": [10, 10, 10], "High": [10.1, 10.2, 10.1], "Low": [9.9, 9.8, 9.9],
                       "Close": [10, 10, 10], "Volume": 1.0}, index=idx)
    assert pr.data_error(ok, 10.2, 9.8, 0.01) is None
    assert pr.data_error(ok, 10.15, 9.8, 0.01) == "outside_daily_range"      # a print above the day's high
    assert pr.data_error(ok.assign(Low=[9.9, -1, 9.9]), 10.2, -2, 0.01) == "invalid_price"
    assert pr.data_error(ok.iloc[0:0], 1, 1, 0.01) == "no_bars"


@pytest.fixture
def dbs(tmp_path, monkeypatch):
    u = UniverseStore(str(tmp_path / "u.db"))
    UniverseSelector(u)
    m = MarketDataStore(str(tmp_path / "m.db"))
    rows_sel, rows_el, rows_d = [], [], []
    for i, sym in enumerate(("AAA", "BBB", "CCC")):
        df = noisy_days("2019-01-02", "2019-03-29", seed=40 + i, px=[40, 90, 150][i])
        from market_data_store import session_close
        for d, g in df.groupby(df.index.date):
            if session_close(d) is None:                               # real selections are sessions only
                continue
            d = str(d)
            m._store_bars(sym, g, {}, contiguous=False)
            rows_sel.append((d, sym, 3.0, i + 1, 1e5, 2e4))
            rows_el.append((d, sym, float(g["Open"].iloc[0]), 2e6, float((g["High"] - g["Low"]).mean() * 12)))
            hi, lo = float(g["High"].max()), float(g["Low"].min())
            if sym == "BBB" and d == "2019-02-05":
                hi -= 1.0                                              # daily bar disagrees: data error
            rows_d.append((sym, d, 0, hi, lo, 0, 0, 0, 0, 0, 0, 0))
    with sqlite3.connect(u.db_path) as c:
        c.executemany("INSERT INTO universe_selection VALUES (?,?,?,?,?,?)", rows_sel)
        c.executemany("INSERT INTO universe_eligibility VALUES (?,?,?,?,?)", rows_el)
        c.executemany("INSERT INTO universe_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows_d)
    monkeypatch.setitem(pr.SPEC, "is_window", ("2019-01-02", "2019-03-29"))
    monkeypatch.setitem(pr.SPEC, "null", {**pr.SPEC["null"], "draws": 60, "n_candidates": 3})
    return u.db_path, m.db_path, tmp_path


def test_stage1_end_to_end(dbs):
    udb, mdb, tmp = dbs
    r = pr.run_stage1(udb, mdb, None, out_dir=tmp / "out")
    d = r["diagnostics"]
    assert d["excluded"] == {"outside_daily_range": 1}
    assert d["trades"] > 30 and 0 < d["long_share"] < 1
    assert r["H2"]["expectancy"] < r["H1"]["expectancy"]                   # costs bite
    assert r["h2_bps_used"] == 5.0 and r["spread"]["validated_on_spy"] is False
    assert set(r["gates"]) == {"G1_null", "G2_in_sample", "G3_cost_stress"}
    assert r["null"] is not None and 0 < r["null"]["gate_p"] <= 1
    assert Path(r["path"]).exists()


def test_breakeven_really_zeroes_expectancy(dbs):
    udb, mdb, tmp = dbs
    r = pr.run_stage1(udb, mdb, None, skip_null=True, out_dir=tmp / "out")
    t = pr.trade_frame(pr.run_trades(pr.load_stock_days(udb, *pr.SPEC["is_window"], 20),
                                     MarketDataStore(mdb), 0.10, 0.01)[0])
    assert pr.net(t, r["breakeven_bps_per_side"]).mean() == pytest.approx(0, abs=1e-9)
    assert r["gates"]["G1_null"] is None and r["null"] is None


@pytest.mark.parametrize("spy, sel, expect", [(1.0, 8.0, 8.0),    # validated and wider: use the estimate
                                              (1.0, 3.0, 5.0),    # validated but narrower: keep 5
                                              (3.5, 8.0, 5.0)])   # estimator fails on SPY: keep 5
def test_pre_committed_spread_rule(dbs, monkeypatch, spy, sel, expect):
    udb, mdb, tmp = dbs
    monkeypatch.setattr(pr, "spread_check", lambda *a: {
        "spy_median_half_bps": spy, "validated_on_spy": spy <= 2.0,
        "selected_median_half_bps": sel, "sample_stock_days": 1})
    r = pr.run_stage1(udb, mdb, None, skip_null=True, out_dir=tmp / "out")
    assert r["h2_bps_used"] == expect


def test_trade_frame_r_signs_and_flags():
    from datetime import datetime
    from boxes.orb_stocks_in_play import TradeRecord
    ts = datetime(2019, 3, 12, 10, 0)
    long_win = TradeRecord("2019-03-12", "A", "long", 0, 0, 0, 0, 2.0, 101.0, ts, 101.0, 100.8,
                           datetime(2019, 3, 12, 16, 0), 101.6, "close")
    short_win = TradeRecord("2019-03-12", "B", "short", 0, 0, 0, 0, 2.0, 99.0, ts, 98.9, 99.1,
                            datetime(2019, 3, 12, 16, 0), 98.5, "close")
    short_stopped_same_minute = TradeRecord("2019-03-12", "C", "short", 0, 0, 0, 0, 2.0, 99.0, ts,
                                            99.0, 99.2, ts, 99.2, "stop")
    t = pr.trade_frame([long_win, short_win, short_stopped_same_minute])
    assert list(t["gross_r"].round(9)) == [3.0, 2.0, -1.0]
    assert list(t["entry_minute_stop"]) == [False, False, True]
    assert list(t["gapped_entry"]) == [False, True, False]


def test_optimistic_diagnostic_is_labelled_and_never_gates(dbs):
    udb, mdb, tmp = dbs
    cons = pr.run_stage1(udb, mdb, None, skip_null=True, out_dir=tmp / "out")
    opt = pr.run_stage1(udb, mdb, None, out_dir=tmp / "out", optimistic_ties=True)
    assert opt["mode"] == "optimistic_ties" and opt["null"] is None
    assert opt["verdict"].startswith("DIAGNOSTIC ONLY") and "stage1_optimistic_" in opt["path"]
    assert opt["diagnostics"]["entry_minute_stop_share"] == 0.0
    assert cons["diagnostics"]["entry_minute_stop_share"] > 0.0
