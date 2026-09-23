"""historical_sim._to_5min: 1-minute store/Alpaca bars must reach the canonical
5-minute backtest as correct 5-minute bars; 5-minute data passes unchanged."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import historical_sim as hs
from trading_engine import _backtest_orb_full_gate, CONFIG


def _one_min(days=("2024-03-08", "2024-03-11"), seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for d in days:                       # 08 Mar is EST, 11 Mar is EDT
        idx = pd.date_range(f"{d} 09:30", f"{d} 15:59", freq="1min", tz="America/New_York")
        c = 500 + rng.normal(0, 0.1, len(idx)).cumsum()
        o = np.r_[c[0], c[:-1]]
        frames.append(pd.DataFrame({
            "Open": o, "High": np.maximum(o, c) + 0.05, "Low": np.minimum(o, c) - 0.05,
            "Close": c, "Volume": rng.integers(1_000, 5_000, len(idx))}, index=idx))
    return pd.concat(frames)


def test_aggregation_is_correct():
    m1 = _one_min()
    m5 = hs._to_5min(m1)
    assert len(m5) == 2 * 78                                   # 390/5 per session
    first = m1.loc["2024-03-11 09:30":"2024-03-11 09:34"]
    bar = m5.loc[pd.Timestamp("2024-03-11 09:30", tz="America/New_York")]
    assert bar.Open == first.Open.iloc[0] and bar.Close == first.Close.iloc[-1]
    assert bar.High == first.High.max() and bar.Low == first.Low.min()
    assert bar.Volume == first.Volume.sum()


def test_no_cross_session_bins_and_start_labels():
    m5 = hs._to_5min(_one_min())
    for _, day in m5.groupby(m5.index.date):
        assert day.index[0].strftime("%H:%M") == "09:30"
        assert day.index[-1].strftime("%H:%M") == "15:55"
    assert (m5.index.minute % 5 == 0).all()


def test_five_minute_input_unchanged():
    m5 = hs._to_5min(_one_min())
    pd.testing.assert_frame_equal(hs._to_5min(m5), m5)


def _trend_days(days, k=0.06):
    """Flat 15-min ORB, then a steady trend (alternating up/down days)."""
    frames = []
    for i, d in enumerate(days):
        idx = pd.date_range(f"{d} 09:30", f"{d} 15:59", freq="1min", tz="America/New_York")
        t = np.arange(len(idx)); sign = 1 if i % 2 == 0 else -1
        c = 500 + np.where(t < 15, 0.3 * np.sin(t), sign * k * (t - 15) + 0.4 * np.sin(t / 3.0))
        o = np.r_[c[0], c[:-1]]
        frames.append(pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.03,
                                    "Low": np.minimum(o, c) - 0.03, "Close": c,
                                    "Volume": 2000.0}, index=idx))
    return pd.concat(frames)


def test_one_minute_bars_would_silently_suppress_signals():
    """Why _to_5min exists: the VWAP-slope gate's threshold is per BAR, so on
    raw 1-minute bars clear trend days produce no trades at all, while the
    same data at 5-minute resolution trades normally."""
    days = [str(d.date()) for d in pd.bdate_range("2024-03-04", "2024-03-08")]
    m1 = _trend_days(days)
    cfg = {**CONFIG, "ticker": "SPY"}
    on_5m = _backtest_orb_full_gate(hs._to_5min(m1), cfg).get("trades", [])
    on_1m = _backtest_orb_full_gate(m1, cfg).get("trades", [])
    assert len(on_5m) == 5                          # one trade per trend day
    assert len(on_1m) != len(on_5m)                 # raw 1m is NOT equivalent
    assert all(t["entry_ts"].minute % 5 == 0 for t in on_5m)
