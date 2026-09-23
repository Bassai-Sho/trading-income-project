"""pr001_fill_diagnostic: 'booked' reproduces the backtest's R exactly, and the
re-priced models move stop exits in the direction the fill bias implies."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr001_instrument_test as pr
import pr001_fill_diagnostic as fd
from historical_sim import _to_5min
from test_pr001_runner import store, _days  # noqa: F401  (fixture)
import numpy as np
import pandas as pd


def _reversal_days(start, end):
    """Breakout, a short run, then a hard reversal through the stop (with gaps)."""
    frames = []
    for i, d in enumerate(pd.bdate_range(start, end)):
        idx = pd.date_range(f"{d.date()} 09:30", f"{d.date()} 15:59", freq="1min", tz="America/New_York")
        t = np.arange(len(idx)); sign = 1 if i % 2 == 0 else -1
        path = np.where(t < 15, 0.3 * np.sin(t),
               np.where(t < 45, sign * 0.06 * (t - 15), sign * (1.8 - 0.12 * (t - 45))))
        c = 400 + path
        o = np.r_[c[0], c[:-1]] + np.where(t % 7 == 0, -sign * 0.15, 0)   # occasional gaps
        frames.append(pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.03,
                                    "Low": np.minimum(o, c) - 0.03, "Close": c,
                                    "Volume": 2000.0}, index=idx))
    return pd.concat(frames)


def test_booked_matches_backtest_and_close_fill_is_worse_on_stops(store):
    store._store_bars("SPY", _reversal_days("2019-05-01", "2019-05-31"), {})
    df, _ = pr.load_bars(store, "SPY", "2019-03-01", "2019-05-31")
    trades = pr.run_trades(df, "SPY")
    m = fd.reprice(trades, _to_5min(df))
    assert m["booked"] == pytest.approx([t["actual_r"] for t in trades], abs=1e-9)
    stops = [i for i, t in enumerate(trades) if t["exit_reason"] == "trailing_stop"]
    tgts  = [i for i, t in enumerate(trades) if t["exit_reason"] == "target_hit"]
    assert stops and tgts
    # a close beyond the stop is never better than the stop price
    assert all(m["close_fill"][i] <= m["booked"][i] + 1e-9 for i in stops)
    # a close beyond the target is never worse than the target price
    assert all(m["close_fill"][i] >= m["booked"][i] - 1e-9 for i in tgts)


def test_diagnostic_refuses_oos_window(store, monkeypatch):
    monkeypatch.setitem(pr.SPEC, "is_window", ("2023-01-01", "2024-12-31"))
    with pytest.raises(AssertionError, match="in-sample only"):
        fd.run(store)
